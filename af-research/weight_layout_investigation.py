#!/usr/bin/env python3
"""Bounded weight-layout diagnosis from pinned bytes and TRAIN-only controls.

No optimizer, checkpoint fitting, validation selection, test pixels or test
predictions. Structural observations and host hypotheses are kept distinct.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch

from decode_parameters import recover
from extract import MODELS, require, write_json
from packing_investigation import dimensions
from pretraining_data import evaluate_detection, sha256
from research_paths import add_firmware_arguments, resolve_source
from train_recognition import CLASSES, MODEL, Detector, detections, load_image

LEGACY = {'o4_i4_hw': 'Ob_Ib_O_I_H_W', 'hw_o4_i4': 'Ob_Ib_H_W_O_I',
          'o4_hw_i4': 'Ob_Ib_O_H_W_I'}
# Freeze bounded candidates before image measurements. The row-interleaved
# hypotheses are motivated by conv1 row structure; plain OIHW and I/O reversal
# test ambiguities that 1x1 head padding alone cannot settle for 3x3 kernels.
CANDIDATES = [
    {'name': 'legacy_'+layout, 'conv1_order': 'O_H_W', 'spatial_order': order}
    for layout, order in LEGACY.items()] + [
    {'name': 'conv1_HOW_'+layout, 'conv1_order': 'H_O_W', 'spatial_order': order}
    for layout, order in LEGACY.items()] + [
    {'name': 'row_HOWI', 'conv1_order': 'H_O_W', 'spatial_order': 'Ob_Ib_H_O_W_I'},
    {'name': 'row_HOIW', 'conv1_order': 'H_O_W', 'spatial_order': 'Ob_Ib_H_O_I_W'},
    {'name': 'direct_spatial_OIHW', 'conv1_order': 'H_O_W', 'spatial_order': 'Ob_O_Ib_I_H_W'},
    {'name': 'spatial_inner_IO', 'conv1_order': 'H_O_W', 'spatial_order': 'Ob_Ib_I_O_H_W'},
    {'name': 'conv1_WOH', 'conv1_order': 'W_O_H', 'spatial_order': LEGACY['o4_i4_hw']},
    {'name': 'conv1_HWO', 'conv1_order': 'H_W_O', 'spatial_order': LEGACY['o4_i4_hw']},
]
FOLLOWUP_CANDIDATES = [
    {'name':'transpose_all_legacy','conv1_order':'O_H_W','spatial_order':LEGACY['o4_i4_hw'],
     'transpose_spatial':True},
    {'name':'transpose_all_HOW','conv1_order':'H_O_W','spatial_order':LEGACY['o4_i4_hw'],
     'transpose_spatial':True},
    {'name':'backbone_legacy_rpn_hw_o4_i4','conv1_order':'O_H_W','spatial_order':LEGACY['o4_i4_hw'],
     'rpn_spatial_order':LEGACY['hw_o4_i4']},
    {'name':'backbone_legacy_rpn_o4_hw_i4','conv1_order':'O_H_W','spatial_order':LEGACY['o4_i4_hw'],
     'rpn_spatial_order':LEGACY['o4_hw_i4']},
]


def unpack_probe_weight(raw, record, spatial_order='Ob_Ib_O_I_H_W', conv1_order='O_H_W'):
    """Return candidate int16 OIHW; 1x1 and upsample use the existing network."""
    logical, output, inputs, kh, kw = dimensions(record)
    if record['name'] == 'conv1':
        require(inputs == 1 and output % 4 == 0, 'Unexpected conv1 channels')
        order = ['Ob']+conv1_order.split('_')
        sizes = {'Ob': output//4, 'O': 4, 'H': kh, 'W': kw}
        require(set(order) == set(sizes), 'Invalid conv1 candidate order')
        array = raw.reshape([sizes[k] for k in order])
        return array.transpose(*[order.index(k) for k in ('Ob','O','H','W')]).reshape(output, 1, kh, kw)[:logical].copy()
    require(kh == kw == 3, 'Only 3x3 probe kernels use spatial_order')
    order = spatial_order.split('_')
    sizes = {'Ob': output//4, 'Ib': inputs//4, 'O': 4, 'I': 4, 'H': kh, 'W': kw}
    require(len(order) == len(sizes) and set(order) == set(sizes), 'Invalid spatial candidate order')
    array = raw.reshape([sizes[k] for k in order])
    array = array.transpose(*[order.index(k) for k in ('Ob','O','Ib','I','H','W')])
    return array.reshape(output, inputs, kh, kw)[:logical].copy()


def neighbor_energy(weights):
    x = weights.astype(np.float64)
    energy = np.mean(x*x)
    return float((np.mean(np.diff(x, axis=-1)**2)+np.mean(np.diff(x, axis=-2)**2))/(2*max(energy, 1e-30)))


def structural_model(data):
    manifest, tensors = recover(data)
    result = {'source_sha256': hashlib.sha256(data).hexdigest(), 'conv1': {}, 'heads_1x1': [], 'spatial_3x3': []}
    first = manifest['convolutions'][0]
    raw = tensors[first['tensor_key']+'_weight_i16_packed']
    result['conv1'].update(file_offset=first['weight']['data_offset'], bytes=first['weight']['bytes'],
                          stored_shape=first['weight']['stored_shape'], raw_first_64_i16=raw.reshape(-1)[:64].tolist())
    result['conv1']['orders'] = []
    for axes in itertools.permutations(('O','H','W')):
        order = '_'.join(axes)
        weights = unpack_probe_weight(raw, first, conv1_order=order)
        result['conv1']['orders'].append({'order': order, 'neighbor_difference_energy_ratio': neighbor_energy(weights),
                                         'first_four_kernels_i16': weights[:4,0].tolist()})
    for record in manifest['convolutions']:
        raw = tensors[record['tensor_key']+'_weight_i16_packed']
        if record['name'].startswith('rpn_cls_score'):
            logical, output, inputs, _, _ = dimensions(record)
            sizes = {'Ob':output//4, 'Ib':inputs//4, 'O':4, 'I':4}
            candidates = []
            for axes in itertools.permutations(sizes):
                w = raw.reshape([sizes[k] for k in axes]).transpose(*[axes.index(k) for k in ('Ob','O','Ib','I')]).reshape(output,inputs)
                nonzero = int(np.count_nonzero(w[logical:]))
                a,b = w[:25].astype(float),w[25:50].astype(float)
                cosine = (a*b).sum(1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-30)
                candidates.append({'order':'_'.join(axes), 'padding_nonzero_elements':nonzero,
                                   'bg_fg_cosine_max':float(cosine.max()), 'bg_fg_cosine_mean':float(cosine.mean()),
                                   'passes_padding_and_all_pair_cosines_below_minus_0_98':not nonzero and bool(np.all(cosine < -.98))})
            result['heads_1x1'].append({'name':record['name'], 'file_offset':record['weight']['data_offset'],
                                      'orders':candidates,
                                      'input_lane_permutation_limit':'Any consistent permutation of input columns preserves padding and row-pair cosines; these invariants cannot identify it.'})
            tiled = raw.reshape(output//4, inputs//4, 4, 4)
            lane_checks = {}
            for kind in ('output','input'):
                passing = []
                for permutation in itertools.permutations(range(4)):
                    changed = tiled[:,:,permutation,:] if kind == 'output' else tiled[:,:,:,permutation]
                    w = changed.transpose(0,2,1,3).reshape(output,inputs)
                    a,b = w[:25].astype(float),w[25:50].astype(float)
                    cosine = (a*b).sum(1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-30)
                    if not np.count_nonzero(w[logical:]) and np.all(cosine < -.98):
                        passing.append(list(permutation))
                lane_checks[kind+'_lane_permutations_passing'] = passing
            result['heads_1x1'][-1]['lane_checks'] = lane_checks
        if record['name'] == 'upsample':
            blocks = raw.reshape(256,4,4,4)
            phase = blocks[0,:,0,:].reshape(2,2,2,2)
            kernel = np.empty((4,4),raw.dtype)
            for py,px,ty,tx in itertools.product(range(2), repeat=4):
                kernel[py+2*ty,px+2*tx] = phase[py,px,ty,tx]
            expected = np.zeros_like(blocks)
            expected[:128,:,0,:] = phase.reshape(4,4)
            expected[:128,:,1,:] = phase.reshape(4,4)
            values = np.unique(raw)
            a,b,c = values[1:]
            histogram_kernel = np.array([[a,b,b,a],[b,c,c,b],[b,c,c,b],[a,b,b,a]])
            result['upsample'] = {'file_offset':record['weight']['data_offset'],
                                  'stored_shape':record['weight']['stored_shape'],
                                  'candidate_axes':'[128 live pairs + 128 zero blocks, 2x2 phase, 2 live + 2 zero lanes, 2x2 taps]',
                                  'all_raw_bytes_explained_by_phase_table_and_padding':bool(np.array_equal(blocks,expected)),
                                  'recovered_phase_table_i16':phase.tolist(), 'phase_reconstructed_4x4_kernel_i16':kernel.tolist(),
                                  'matches_existing_histogram_candidate_exactly':bool(np.array_equal(kernel,histogram_kernel)),
                                  'limitation':'Exact coefficient/padding pattern supports a bilinear polyphase table, not original channel routing or phase address execution.'}
    # Keep O-block/I-block axes fixed here; these statistics cannot identify
    # output/input permutations with unchanged per-kernel coefficient vectors.
    for axes in itertools.permutations(('O','I','H','W')):
        order = 'Ob_Ib_'+'_'.join(axes)
        rows = []
        for record in manifest['convolutions']:
            if record['kernel_hw_candidate'] != [3,3]: continue
            weights = unpack_probe_weight(tensors[record['tensor_key']+'_weight_i16_packed'],record,order)
            rows.append({'name':record['name'], 'neighbor_difference_energy_ratio':neighbor_energy(weights)})
        result['spatial_3x3'].append({'order':order, 'mean_neighbor_difference_energy_ratio':float(np.mean([r['neighbor_difference_energy_ratio'] for r in rows])), 'layers':rows})
    return result


def build_probe_detector(data, specification):
    """Create a fresh restored FP32 hypothesis; no optimization or trained state."""
    detector = Detector(data, 'o4_i4_hw')
    detector.layout = specification['name']
    manifest,tensors = recover(data)
    with torch.no_grad():
        for record in manifest['convolutions']:
            if record['name'] == 'conv1' or record['kernel_hw_candidate'] == [3,3]:
                raw = tensors[record['tensor_key']+'_weight_i16_packed']
                order = (specification.get('rpn_spatial_order',specification['spatial_order'])
                         if record['name'].startswith('rpn_') else specification['spatial_order'])
                weights = unpack_probe_weight(raw,record,order,specification['conv1_order'])
                if specification.get('transpose_spatial',False):
                    weights = weights.transpose(0,1,3,2).copy()
                detector.model.convs[str(record['index'])].weight.copy_(torch.from_numpy(weights.astype(np.float32)*record['weight_scale']))
    detector.model.requires_grad_(False)
    detector.model.eval()
    return detector


def runtime_probe(data, dataset, root, manifest_hash, specifications, folder):
    frames = sorted((f for f in dataset['frames'] if f['split']=='train'),key=lambda f:f['id'])[:32]
    require(len(frames)==32, 'Exactly 32 train frames are required')
    require(dataset['classes']==list(CLASSES), 'Unexpected class list')
    selected = {'classes':list(CLASSES),'frames':frames}
    plan = {'dataset_manifest_sha256':manifest_hash, 'sample_selection':'train only, ascending id, first32',
            'frames':[{'id':f['id'],'sha256':f['sha256']} for f in frames],
            'preprocessing':'existing stored RGB->L/bilinear384x256/subtract117; held fixed to isolate packing',
            'candidates':specifications, 'controls':['constant gray117', 'cyclic next-image predictions against same annotations'],
            'source_model_sha256':hashlib.sha256(data).hexdigest(), 'script_sha256':sha256(Path(__file__)),
            'no_training_no_validation_no_test':True}
    write_json(folder/'frozen-probe-plan.json',plan)
    images = [load_image(root,frame) for frame in frames]
    reports=[]
    for specification in specifications:
        begin=time.monotonic()
        detector=build_probe_detector(data,specification)
        nodes={str(n['index']):n for n in detector.model.layers if n['operator_id']==1}
        captured={}
        handles=[]
        for key,module in detector.model.convs.items():
            def capture(_module,_inputs,value,key=key):
                captured[key] = (value.relu() if nodes[key]['relu_candidate'] else value).detach()
            handles.append(module.register_forward_hook(capture))
        with torch.inference_mode():
            gray_logits,gray_delta=detector(torch.zeros(1,1,256,384))
            gray_logits={k:v.clone() for k,v in gray_logits.items()}
            gray_delta=gray_delta.clone()
            gray_features={k:v.clone() for k,v in captured.items()}
            layer_rows={k:[] for k in captured}
            real_rows=[];gray_rows=[];normalized=[];scores={c:[] for c in CLASSES}
            for frame,tensor in zip(frames,images):
                logits,delta=detector(tensor)
                rows=detections(logits,delta,detector.anchors,frame)
                real_rows.append({'id':frame['id'],'detections':rows})
                normalized.append([{**r,'bbox_xyxy':(np.asarray(r['bbox_xyxy'])/np.array([frame['width'],frame['height']]*2)).tolist()} for r in rows])
                gray_rows.append({'id':frame['id'],'detections':detections(gray_logits,gray_delta,detector.anchors,frame)})
                for cls in CLASSES: scores[cls].append(logits[cls].softmax(1)[:,1].numpy().copy())
                for key,value in captured.items():
                    gray=gray_features[key]
                    real_rms=float(value.square().mean().sqrt());gray_rms=float(gray.square().mean().sqrt())
                    signal=float((value-gray).square().mean().sqrt())
                    layer_rows[key].append({'real_rms':real_rms,'gray_rms':gray_rms,'signal_rms':signal,
                                            'signal_relative_to_max_rms':signal/max(real_rms,gray_rms,1e-12),
                                            'real_zero_fraction':float((value==0).float().mean()),
                                            'gray_zero_fraction':float((gray==0).float().mean())})
            wrong_rows=[]
            for i,frame in enumerate(frames):
                wrong_rows.append({'id':frame['id'],'detections':[{**r,'bbox_xyxy':(np.asarray(r['bbox_xyxy'])*np.array([frame['width'],frame['height']]*2)).tolist()} for r in normalized[(i+1)%len(frames)]]})
        metrics={}
        for label,rows in [('real',real_rows),('constant_gray117',gray_rows),('wrong_image',wrong_rows)]:
            metrics[label]=evaluate_detection(selected,{'dataset_manifest_sha256':manifest_hash,'frames':rows},'train')
        traces=[]
        for key,rows in layer_rows.items():
            traces.append({'layer':int(key),'name':nodes[key]['name'],
                           **{stat:float(np.mean([r[stat] for r in rows])) for stat in rows[0]}})
        report={'candidate':specification,'metrics':metrics,'layer_signal':traces,
                'foreground_score_across_image_std_mean':{c:float(np.stack(v).std(0).mean()) for c,v in scores.items()},
                'elapsed_seconds':time.monotonic()-begin,
                'limitations':['Train32 diagnostic only; no validation selection or test evidence.',
                               'AP depends on unchanged unverified preprocessing/anchor decoder.',
                               'Response to real images does not prove original DSP numerical parity.']}
        write_json(folder/(specification['name']+'.json'),report)
        reports.append(report)
        for handle in handles:handle.remove()
        detector.close()
        print(json.dumps({'candidate':specification['name'], 'mean_ap':{k:v['mean_ap'] for k,v in metrics.items()},
                          'elapsed_seconds':report['elapsed_seconds']}),flush=True)
    write_json(folder/'runtime-summary.json',{'plan':plan,'candidates':reports})
    return reports


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    add_firmware_arguments(parser)
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--static-only',action='store_true')
    parser.add_argument('--followup',action='store_true',help='Only four predefined spatial-transpose/RPN-group checks')
    args=parser.parse_args()
    require(not (args.output/'structural-evidence.json').exists(),'Refusing to overwrite investigation')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'weight_layout_investigation-source.py').write_bytes(Path(__file__).read_bytes())
    source=resolve_source(args)
    originals={stem:(source/(stem+'.bin')).read_bytes() for stem in MODELS}
    require(all(hashlib.sha256(b).hexdigest()==MODELS[s] for s,b in originals.items()),'Source digest mismatch')
    structure={'models':{s:structural_model(b) for s,b in originals.items()},
               'script_sha256':sha256(Path(__file__)),
               'limitations':['Spatial smoothness is a heuristic, not proof of trained kernel order.',
                              'All byte permutations are equally invertible; roundtrip is not a selection criterion.',
                              'Runtime XM4 kernel pointers exist but no XM4 execution/decoder evidence is available.']}
    write_json(args.output/'structural-evidence.json',structure)
    if not args.static_only:
        require(args.dataset is not None,'--dataset is required for train controls')
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        runtime_probe(originals[MODEL],json.loads(args.dataset.read_text()),args.dataset.parent,
                      sha256(args.dataset),FOLLOWUP_CANDIDATES if args.followup else CANDIDATES,args.output)


if __name__=='__main__':main()
