#!/usr/bin/env python3
"""Finish reporting after a floating-boundary error, without optimizer updates.

Preserves the frozen training source/checkpoints and the original failed wrong-
image control. Only <=1e-9 pixel boundary overshoots are clamped. Quantization is
the already declared head-only host proxy, evaluated from fixed epoch6.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from extract import require, write_json
from pretraining_data import sha256
from train_small_anchor import (BASE, Detector, acceptance, backbone_digest, controls,
    evaluate, measure, quantized_proxy, state_digest)


def fix_controls(folder,dataset,frames):
    by_id={f['id']:f for f in frames}
    changes=[];metrics={}
    for mode in ('real','same_mean','fixed_gray117','shuffled'):
        path=folder/(mode+'.json')
        document=json.loads(path.read_text())
        require(len(document['frames'])==80 and {r['id'] for r in document['frames']}==set(by_id),'Incomplete control')
        original=copy.deepcopy(document)
        for row in document['frames']:
            frame=by_id[row['id']]
            for index,item in enumerate(row['detections']):
                before=item['bbox_xyxy'];after=[]
                for j,value in enumerate(before):
                    cap=frame['width'] if j%2==0 else frame['height']
                    require(-1e-9<=value<=cap+1e-9,'Material coordinate error, not roundoff')
                    after.append(max(0.,min(float(cap),value)))
                require(after[0]<after[2] and after[1]<after[3],'Empty box cannot be fixed as roundoff')
                if before!=after:
                    require(mode=='shuffled','Unexpected affected control')
                    changes.append({'mode':mode,'id':row['id'],'prediction_index':index,'before':before,'after':after})
                    item['bbox_xyxy']=after
        if document!=original:
            backup=folder/(mode+'-before-float-boundary-fix.json')
            require(not backup.exists(),'Already fixed; preserve prior evidence')
            backup.write_bytes(path.read_bytes())
            write_json(path,document)
        metrics[mode]=measure(dataset,frames,document)
    write_json(folder/'metrics.json',metrics)
    return metrics,changes


def run(args):
    started=time.monotonic();folder=args.run
    require(not (folder/'summary.json').exists(),'Do not overwrite a completed run')
    plan=json.loads((folder/'plan.json').read_text());ph=sha256(folder/'plan.json')
    data=args.firmware.read_bytes();require(sha256(args.firmware)==plan['model_sha256'],'Wrong firmware')
    dataset=json.loads(args.dataset.read_text());require(sha256(args.dataset)==plan['manifest_sha256'],'Wrong data')
    frames=sorted((f for f in dataset['frames'] if f['split']=='validation'),key=lambda f:f['id'])
    rows=[json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines()]
    steps=[json.loads(line) for line in (folder/'steps.jsonl').read_text().splitlines()]
    require(len(rows)==1920 and len(steps)==480 and steps[-1]['global_step']==480,'Training did not finish')
    protected=[folder/'plan.json',folder/'training.jsonl',folder/'steps.jsonl',*folder.glob('epoch-*/*'),*folder.glob('baseline_*/*')]
    before={str(p.relative_to(folder)):sha256(p) for p in protected if p.is_file()}
    final_metrics=json.loads((folder/'epoch-06/metrics.json').read_text())
    reference=json.loads((folder/'baseline_original/metrics.json').read_text())
    baseline=json.loads((folder/'baseline_small/metrics.json').read_text())
    control_metrics,fixes=fix_controls(folder/'controls/epoch6',dataset,frames)
    require(bool(fixes),'Expected the known failed boundary control')
    checkpoint=torch.load(folder/'epoch-06/checkpoint.pt',map_location='cpu',weights_only=False)
    require(checkpoint['completed_epoch']==6 and checkpoint['global_step']==480 and checkpoint['plan_sha256']==ph,'Wrong fixed checkpoint')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(plan['seed'])
    detector=Detector(data,'gemm_k4');detector.anchors=np.load(folder/'anchors-small.npy',allow_pickle=False)
    detector.model.load_state_dict(checkpoint['model_state']);detector.model.requires_grad_(False)
    qreport=quantized_proxy(detector,data)
    qmetrics,qpred=evaluate(detector,dataset,frames,args.dataset.parent,ph,plan['manifest_sha256'],folder/'final_quantized')
    try:
        controls(detector,dataset,frames,args.dataset.parent,ph,plan['manifest_sha256'],qpred,folder/'controls/final_quantized')
    except ValueError as error:
        require(str(error)=='Box outside frame or empty','Unexpected control failure: '+str(error))
    qcontrols,qfixes=fix_controls(folder/'controls/final_quantized',dataset,frames)
    torch.save({'model_state':detector.model.state_dict(),'completed_epoch':6,'global_step':480,
        'plan_sha256':ph,'anchor_sha256':plan['small_anchor_sha256'],'host_quantized_proxy':True},folder/'final_quantized/checkpoint.pt')
    write_json(folder/'final_quantized/quantization.json',qreport)
    epochs=[]
    best_epoch=0;best_ap=baseline['mean_ap']
    for epoch in range(1,7):
        m=json.loads((folder/f'epoch-{epoch:02d}/metrics.json').read_text())
        c=torch.load(folder/f'epoch-{epoch:02d}/checkpoint.pt',map_location='cpu',weights_only=False)
        epochs.append({'epoch':epoch,'global_step':epoch*80,
            'mean_training_loss':float(np.mean([r['loss'] for r in rows if r['epoch']==epoch])),
            'metrics':m,'model_state_sha256':state_digest(c['model_state']),
            'backbone_sha256':steps[epoch*80-1]['backbone_sha256']})
        if m['mean_ap']>best_ap:best_epoch=epoch;best_ap=m['mean_ap']
    require(all(sha256(folder/p)==digest for p,digest in before.items()),'Protected training artifact changed')
    record={'reason':'Reporting failed after all480 updates because two shuffled-control y2 endpoints exceeded768 by1.14e-13.',
            'finalizer_sha256':sha256(Path(__file__)),'frozen_plan_sha256':ph,'model_updates':0,
            'fixed_epoch6_control_changes':fixes,'quantized_control_changes':qfixes,
            'protected_artifacts_before_after':before,'protected_artifacts_unchanged':True,
            'elapsed_seconds':time.monotonic()-started}
    write_json(folder/'finalization.json',record)
    write_json(folder/'summary.json',{'status':'complete','plan_sha256':ph,'epochs':epochs,'optimizer_steps':480,
        'baseline_original':reference,'baseline_small':baseline,
        'primary_fixed_epoch6':{'metrics':final_metrics,'acceptance':acceptance(final_metrics,reference,control_metrics)},
        'auxiliary_best_epoch':best_epoch,'auxiliary_best_mean_ap':best_ap,
        'final_quantized':{'metrics':qmetrics,'acceptance':acceptance(qmetrics,reference,qcontrols),'quantization':qreport},
        'test_inference':False,'camera_deployment_verified':False,
        'backbone_unchanged':backbone_digest(detector.model)==plan['initial_backbone_sha256'],
        'reporting_recovery':record,'training_elapsed_seconds':None})
    detector.close()
    print(json.dumps({'status':'complete','extra_optimizer_steps':0,'boundary_fixes':len(fixes)+len(qfixes),
                      'float_acceptance':acceptance(final_metrics,reference,control_metrics),
                      'quantized_acceptance':acceptance(qmetrics,reference,qcontrols)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--firmware',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=BASE/'private-data/recognition-v2/manifest.json')
    run(p.parse_args())
