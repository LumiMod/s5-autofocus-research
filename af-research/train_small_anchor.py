#!/usr/bin/env python3
"""Bounded K4 small-anchor experiment, preserving the original S5 graph.

An explicit preflight is required. Starts from original weights, changes only
the five proposal scales, and trains only the original three 1x1 heads. Test
images are never loaded. The fixed final epoch is the primary result.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
from PIL import Image
import torch

from decode_parameters import recover
from extract import MODELS, require, write_json
from packing_investigation import pack, unpack
from pretraining_data import evaluate_detection, sha256
from proposal_hypothesis import anchors
from train_recognition import (CLASSES, HEADS, MODEL, Detector, assign_joint_targets,
    detection_loss, detections, ground_truth, load_image, source_hashes, state_digest)
from verify_external_baselines import independent_metrics

BASE=Path(__file__).resolve().parent
EPOCHS=6
ACCUMULATE=4
SEED=20261127
LR=1e-4
HEAD_PREFIXES=tuple('convs.'+i+'.' for i in HEADS.values())


def anchor_digest(value):
    return hashlib.sha256(np.asarray(value,dtype='<f4').tobytes()).hexdigest()


def backbone_digest(model):
    return state_digest({k:v for k,v in model.state_dict().items() if not k.startswith(HEAD_PREFIXES)})


def owner_digest(assigned):
    h=hashlib.sha256()
    for name,dtype in [('matched_gt','<i8'),('gt_classes','<i8'),('gt_boxes','<f4')]:
        h.update(np.asarray(assigned[name],dtype=dtype).tobytes())
    return h.hexdigest()


def new_anchors(data,scales):
    params,_=recover(data)
    detector=Detector(data,'gemm_k4')
    node=next(n for n in detector.model.layers if n['operator_id']==24)
    detector.close()
    ref=anchors(node['anchor_base_size_candidate'],params['proposal_blocks'][0]['ratio_candidates'],
                scales,16,24,node['feature_stride_candidate']).astype(np.float32)
    ref[:,2:]+=1
    require(ref.shape==(9600,4),'Original 25 anchors per location must be retained')
    return ref


def measure(dataset,frames,document):
    split=frames[0]['split']
    independent={}
    for cls in CLASSES:
        predictions={r['id']:[d for d in r['detections'] if d['class']==cls] for r in document['frames']}
        independent[cls]=independent_metrics({f['id']:f for f in frames},predictions,cls)
    legacy=evaluate_detection({'classes':list(CLASSES),'frames':frames},document,split,score_threshold=.5)
    for cls in CLASSES:
        require(abs(legacy['classes'][cls]['ap_at_iou']-independent[cls]['ap_at_iou'])<1e-12,'AP implementations disagree')
    face=independent['person_face']['size_strata']
    small={key:sum(face[b][key] for b in ('below16','16_to_below32'))
           for key in ('ground_truth','matched_at_score','matched_all_saved_scores')}
    small['recall_at_score']=small['matched_at_score']/small['ground_truth']
    return {'classes':independent,'small_faces':small,
            'mean_ap':sum(v['ap_at_iou'] for v in independent.values())/2,
            'legacy_crosscheck':legacy}


def prediction_document(rows,detector,plan_hash,manifest_hash):
    return {'dataset_manifest_sha256':manifest_hash,'plan_sha256':plan_hash,
            'model_state_sha256':state_digest(detector.model.state_dict()),
            'anchor_sha256':anchor_digest(detector.anchors),'frames':rows}


def evaluate(detector,dataset,frames,root,plan_hash,manifest_hash,output):
    detector.model.eval()
    rows=[]
    with torch.inference_mode():
        for frame in frames:
            logits,delta=detector(load_image(root,frame))
            rows.append({'id':frame['id'],'detections':detections(logits,delta,detector.anchors,frame)})
    doc=prediction_document(rows,detector,plan_hash,manifest_hash)
    metrics=measure(dataset,frames,doc)
    output.mkdir(parents=True,exist_ok=True)
    write_json(output/'predictions.json',doc)
    write_json(output/'metrics.json',metrics)
    return metrics,doc


def controls(detector,dataset,frames,root,plan_hash,manifest_hash,real,output):
    output.mkdir(parents=True,exist_ok=True)
    rows={'real':real['frames'],'same_mean':[],'fixed_gray117':[],'shuffled':[]}
    detector.model.eval()
    with torch.inference_mode():
        logits,delta=detector(torch.zeros(1,1,256,384))
        for frame in frames:
            rows['fixed_gray117'].append({'id':frame['id'],'detections':detections(logits,delta,detector.anchors,frame)})
        for frame in frames:
            value=load_image(root,frame)
            logits,delta=detector(torch.full_like(value,float(value.mean())))
            rows['same_mean'].append({'id':frame['id'],'detections':detections(logits,delta,detector.anchors,frame)})
    for i,frame in enumerate(frames):
        donor=frames[(i+1)%len(frames)]
        items=[]
        for item in rows['real'][(i+1)%len(frames)]['detections']:
            scales=[frame['width']/donor['width'],frame['height']/donor['height']]*2
            items.append({**item,'bbox_xyxy':[v*s for v,s in zip(item['bbox_xyxy'],scales)]})
        rows['shuffled'].append({'id':frame['id'],'detections':items})
    result={}
    for mode,items in rows.items():
        doc=prediction_document(items,detector,plan_hash,manifest_hash)
        write_json(output/(mode+'.json'),doc)
        result[mode]=measure(dataset,frames,doc)
    write_json(output/'metrics.json',result)
    return result


def save_checkpoint(path,detector,optimizer,epoch,step,plan_hash,rng):
    torch.save({'model_state':detector.model.state_dict(),'optimizer_state':optimizer.state_dict(),
                'completed_epoch':epoch,'global_step':step,'plan_sha256':plan_hash,
                'anchor_sha256':anchor_digest(detector.anchors),'numpy_rng_state':rng.bit_generator.state,
                'torch_rng_state':torch.get_rng_state()},path)


def quantized_proxy(detector,data):
    """Explicit host proxy with original tensor scales, never a firmware image."""
    params,tensors=recover(data)
    report=[]
    with torch.no_grad():
        for record in params['convolutions']:
            if str(record['index']) not in HEADS.values():continue
            module=detector.model.convs[str(record['index'])]
            item={'name':record['name'],'weight_scale':record['weight_scale'],'bias_scale':record['bias_scale']}
            for kind,scale,bits in [('weight',record['weight_scale'],16),('bias',record['bias_scale'],32)]:
                tensor=getattr(module,kind)
                value=tensor.detach().numpy().astype(np.float64)
                rounded=np.rint(value/scale)
                low,high=-(2**(bits-1)),2**(bits-1)-1
                clipped=(rounded<low)|(rounded>high)
                quant=np.clip(rounded,low,high).astype(np.int16 if bits==16 else np.int32)
                if kind=='weight':
                    packed=pack(quant,record,'gemm_k4')
                    require(np.array_equal(unpack(packed,record,'gemm_k4'),quant),'Quantized head packing mismatch')
                    raw=tensors[record['tensor_key']+'_weight_i16_packed']
                    original_integer=unpack(raw,record,'gemm_k4')
                    require(np.array_equal(pack(original_integer,record,'gemm_k4'),raw),'Original head bytes roundtrip differs')
                else:
                    original_integer=tensors[record['tensor_key']+'_bias_i32_pairs'][:len(quant),0]
                reconstructed=(quant.astype(np.float64)*scale).astype(np.float32)
                tensor.copy_(torch.from_numpy(reconstructed))
                item[kind]={'elements':value.size,'clipped_elements':int(clipped.sum()),
                            'max_abs_error':float(np.max(np.abs(reconstructed-value))),
                            'rms_error':float(np.sqrt(np.mean((reconstructed-value)**2))),
                            'integer_sha256':hashlib.sha256(quant.tobytes()).hexdigest(),
                            'integer_elements_different_from_original':int(np.count_nonzero(quant!=original_integer)),
                            'max_integer_lsb_difference_from_original':int(np.max(np.abs(quant.astype(np.int64)-original_integer.astype(np.int64))))}
            report.append(item)
    return {'scope':'Host head-weight/bias round-even+clip proxy using original scales; activations, accumulation and DSP execution are not emulated.',
            'rounding':'nearest_ties_to_even','float_division_dtype':'float64',
            'weight_integer_dtype':'int16','bias_integer_dtype':'int32','dequantization_dtype':'float32',
            'heads':report,'firmware_written':False,'camera_deployment_verified':False}


def acceptance(final,reference,control_metrics):
    original=reference['classes'];current=final['classes'];small=final['small_faces']
    checks={
        'small_face_recall_at_least_half':small['recall_at_score']>=.5,
        'small_face_matches_at_least_17_of34':small['matched_at_score']>=17 and small['ground_truth']==34,
        'face_ap_floor_and_improvement':current['person_face']['ap_at_iou']>=max(.5,original['person_face']['ap_at_iou']+.05),
        'person_ap_drop_at_most_0_05':current['person']['ap_at_iou']>=original['person']['ap_at_iou']-.05,
        'mean_ap_gain_at_least_0_02':final['mean_ap']>=reference['mean_ap']+.02}
    if control_metrics is not None:
        for cls in CLASSES:
            checks[cls+'_beats_every_control_by_0_15']=all(
                current[cls]['ap_at_iou']>=control_metrics[m]['classes'][cls]['ap_at_iou']+.15
                for m in ('same_mean','fixed_gray117','shuffled'))
    return {'checks':checks,'passed':all(checks.values()),'camera_readiness':False}


def run(args):
    require(not args.output.exists(),'Use a new run directory')
    preflight=json.loads(args.preflight.read_text())
    require(preflight['ready_for_bounded_training'] is True,'Preflight has not passed')
    for row in preflight['evidence']:
        require(sha256(BASE/row['path'])==row['sha256'],'Preflight evidence changed')
    data=args.firmware.read_bytes()
    require(sha256(args.firmware)==MODELS[MODEL],'Wrong original model')
    dataset=json.loads(args.dataset.read_text());root=args.dataset.parent
    manifest_hash=sha256(args.dataset)
    require(manifest_hash==preflight['manifest_sha256'],'Preflight dataset differs')
    train=sorted((f for f in dataset['frames'] if f['split']=='train'),key=lambda f:f['id'])
    validation=sorted((f for f in dataset['frames'] if f['split']=='validation'),key=lambda f:f['id'])
    require(len(train)==320 and len(validation)==80,'Unexpected fixed split sizes')
    for frame in train+validation:
        require(sha256(root/frame['path'])==frame['sha256'],'Image identity changed')
    torch.set_num_threads(2);torch.manual_seed(SEED);torch.use_deterministic_algorithms(True)
    rng=np.random.default_rng(SEED)
    detector=Detector(data,'gemm_k4')
    initial=copy.deepcopy(detector.model.state_dict())
    original_anchor=detector.anchors.copy()
    small_anchor=new_anchors(data,preflight['scales'])
    require(anchor_digest(small_anchor)==preflight['small_anchor_sha256'],'Preflight anchor differs')
    parameter_names=[n for n,p in detector.model.named_parameters() if p.requires_grad]
    require(len(parameter_names)==6 and all(n.startswith(HEAD_PREFIXES) for n in parameter_names),'Unexpected trainable graph')
    optimizer=torch.optim.AdamW([p for p in detector.model.parameters() if p.requires_grad],lr=LR,weight_decay=.0001)
    source_names=['train_small_anchor.py','verify_external_baselines.py']
    sources={**source_hashes(),**{n:sha256(BASE/n) for n in source_names}}
    plan={'schema_version':1,'model':MODEL,'model_sha256':MODELS[MODEL],'manifest_sha256':manifest_hash,
          'preflight_sha256':sha256(args.preflight),'scales':preflight['scales'],
          'small_anchor_scales':preflight['scales'],'layout':'gemm_k4',
          'preprocessing':{'geometry':'stretch','width':384,'height':256,'gray':'pillow_l','mean':117.,'scale':1.,
                           'resize':'Pillow bilinear','coordinates':'stored image pixels, no EXIF rotation'},
          'epochs':EPOCHS,'accumulate':ACCUMULATE,'lr':LR,'seed':SEED,'optimizer':'AdamW','weight_decay':.0001,
          'train_ids':[f['id'] for f in train],'validation_ids':[f['id'] for f in validation],
          'initial_state_sha256':state_digest(initial),'initial_backbone_sha256':backbone_digest(detector.model),
          'original_anchor_sha256':anchor_digest(original_anchor),'small_anchor_sha256':anchor_digest(small_anchor),
          'optimizer_parameter_names':parameter_names,'source_hashes':sources,
          'parameter_shapes':{n:list(p.shape) for n,p in detector.model.named_parameters()},
          'trainable_parameters':sum(p.numel() for p in detector.model.parameters() if p.requires_grad),
          'primary_result':'fixed epoch6; never choose a better epoch as the primary claim',
          'auxiliary_best':'validation meanAP50, earliest on ties, includes new-anchor epoch0',
          'acceptance':{'small_recall':.5,'small_matches':17,'face_ap_floor':.5,'face_ap_gain':.05,
                        'max_person_ap_drop':.05,'mean_ap_gain':.02,'per_class_control_margin':.15},
          'quantized_proxy':'Fixed original scales, round-even+clip head weights/bias only; report clipping and independently evaluate. No firmware export.',
          'test_inference':False,'camera_numerical_parity':False,
          'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__}}
    args.output.mkdir(parents=True)
    np.save(args.output/'anchors-small.npy',small_anchor,allow_pickle=False)
    np.save(args.output/'anchors-original.npy',original_anchor,allow_pickle=False)
    write_json(args.output/'plan.json',plan);plan_hash=sha256(args.output/'plan.json')
    snap=args.output/'source-snapshot';snap.mkdir()
    for name in sources:(snap/name).write_bytes((BASE/name).read_bytes())
    started=time.monotonic()
    original_metrics,_=evaluate(detector,dataset,validation,root,plan_hash,manifest_hash,args.output/'baseline_original')
    detector.anchors=small_anchor
    small_metrics,small_pred=evaluate(detector,dataset,validation,root,plan_hash,manifest_hash,args.output/'baseline_small')
    save_checkpoint(args.output/'baseline_small/checkpoint.pt',detector,optimizer,0,0,plan_hash,rng)
    epoch0_controls=controls(detector,dataset,validation,root,plan_hash,manifest_hash,small_pred,args.output/'controls/epoch0')
    # Verify original quantized head integers can reproduce the unchanged FP32 heads.
    q0=quantized_proxy(detector,data)
    q0max=max(float((detector.model.state_dict()[n]-value).abs().max()) for n,value in initial.items())
    require(q0max<1e-6,'Original head dequantization regression failed')
    require(all(h['weight']['integer_elements_different_from_original']==0 for h in q0['heads']),
            'Original integer head weights not recovered exactly')
    detector.model.load_state_dict(initial)
    write_json(args.output/'epoch0-quantization-check.json',{**q0,'max_state_difference':q0max})
    print(json.dumps({'phase':'baselines','original':original_metrics['mean_ap'],'small':small_metrics['mean_ap'],
                      'small_face_matches':small_metrics['small_faces']['matched_at_score']}),flush=True)
    best_epoch=0;best_ap=small_metrics['mean_ap'];step=0;epoch_metrics=[]
    with (args.output/'training.jsonl').open('w') as images_log,(args.output/'steps.jsonl').open('w') as steps_log:
        for epoch in range(1,EPOCHS+1):
            detector.model.train();optimizer.zero_grad(set_to_none=True)
            order=rng.permutation(len(train));loss_values=[]
            for offset,index in enumerate(order):
                frame=train[int(index)];gt=ground_truth(frame)
                assigned=assign_joint_targets(detector.anchors,gt)
                rng_before=copy.deepcopy(rng.bit_generator.state)
                logits,delta=detector(load_image(root,frame))
                loss,info=detection_loss(logits,delta,detector.anchors,gt,rng)
                require(bool(torch.isfinite(loss)),'Nonfinite loss')
                window=min(ACCUMULATE,len(order)-(offset//ACCUMULATE)*ACCUMULATE)
                (loss/window).backward()
                if (offset+1)%ACCUMULATE==0 or offset+1==len(order):
                    named=[(n,p) for n,p in detector.model.named_parameters() if p.requires_grad]
                    require(all(p.grad is not None and bool(p.grad.isfinite().all()) for n,p in named),'Missing/nonfinite gradient')
                    norm=torch.nn.utils.clip_grad_norm_([p for n,p in named],10,error_if_nonfinite=True)
                    old={n:p.detach().clone() for n,p in named}
                    optimizer.step();step+=1
                    changes={n:float((p.detach()-old[n]).norm()) for n,p in named}
                    require(any(v>0 for v in changes.values()),'No optimizer-driven parameter change')
                    bsha=backbone_digest(detector.model)
                    require(bsha==plan['initial_backbone_sha256'],'Frozen backbone changed')
                    steps_log.write(json.dumps({'epoch':epoch,'global_step':step,'grad_norm':float(norm),
                        'head_delta_l2':changes,'backbone_sha256':bsha,'parameter_names_with_gradient':[n for n,p in named]})+'\n');steps_log.flush()
                    optimizer.zero_grad(set_to_none=True)
                images_log.write(json.dumps({'epoch':epoch,'frame_id':frame['id'],'image_index':offset,
                    'optimizer_steps':step,'target_owner_sha256':owner_digest(assigned),'rng_before_loss':rng_before,**info})+'\n')
                loss_values.append(info['loss'])
            images_log.flush()
            folder=args.output/f'epoch-{epoch:02d}'
            metrics,pred=evaluate(detector,dataset,validation,root,plan_hash,manifest_hash,folder)
            save_checkpoint(folder/'checkpoint.pt',detector,optimizer,epoch,step,plan_hash,rng)
            row={'epoch':epoch,'global_step':step,'mean_training_loss':float(np.mean(loss_values)),
                 'metrics':metrics,'model_state_sha256':state_digest(detector.model.state_dict()),
                 'backbone_sha256':backbone_digest(detector.model)}
            epoch_metrics.append(row)
            if metrics['mean_ap']>best_ap:best_epoch=epoch;best_ap=metrics['mean_ap']
            print(json.dumps({'phase':'epoch','epoch':epoch,'step':step,'loss':row['mean_training_loss'],
                'person_ap':metrics['classes']['person']['ap_at_iou'],'face_ap':metrics['classes']['person_face']['ap_at_iou'],
                'small_face_matches':metrics['small_faces']['matched_at_score']}),flush=True)
    final_metrics=metrics
    final_control=controls(detector,dataset,validation,root,plan_hash,manifest_hash,pred,args.output/'controls/epoch6')
    quantization=quantized_proxy(detector,data)
    qmetrics,qpred=evaluate(detector,dataset,validation,root,plan_hash,manifest_hash,args.output/'final_quantized')
    qcontrols=controls(detector,dataset,validation,root,plan_hash,manifest_hash,qpred,args.output/'controls/final_quantized')
    torch.save({'model_state':detector.model.state_dict(),'completed_epoch':EPOCHS,'global_step':step,
                'plan_sha256':plan_hash,'anchor_sha256':anchor_digest(small_anchor),'host_quantized_proxy':True},args.output/'final_quantized/checkpoint.pt')
    write_json(args.output/'final_quantized/quantization.json',quantization)
    require(sha256(args.dataset)==manifest_hash and all(sha256(BASE/n)==v for n,v in sources.items()),'Inputs or source changed during run')
    summary={'status':'complete','plan_sha256':plan_hash,'epochs':epoch_metrics,'optimizer_steps':step,
             'baseline_original':original_metrics,'baseline_small':small_metrics,
             'primary_fixed_epoch6':{'metrics':final_metrics,'acceptance':acceptance(final_metrics,original_metrics,final_control)},
             'auxiliary_best_epoch':best_epoch,'auxiliary_best_mean_ap':best_ap,
             'final_quantized':{'metrics':qmetrics,'acceptance':acceptance(qmetrics,original_metrics,qcontrols),'quantization':quantization},
             'test_inference':False,'camera_deployment_verified':False,'elapsed_seconds':time.monotonic()-started,
             'backbone_unchanged':backbone_digest(detector.model)==plan['initial_backbone_sha256']}
    write_json(args.output/'summary.json',summary)
    detector.close()
    print(json.dumps({'phase':'complete','optimizer_steps':step,'float_acceptance':summary['primary_fixed_epoch6']['acceptance'],
                      'quantized_acceptance':summary['final_quantized']['acceptance'],'elapsed_seconds':summary['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--firmware',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=BASE/'private-data/recognition-v2/manifest.json')
    p.add_argument('--preflight',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
