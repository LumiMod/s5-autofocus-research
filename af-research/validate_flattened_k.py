#!/usr/bin/env python3
"""Fixed K4 validation of two already measured train-only input policies.

Neither positive AP nor a better policy grants training or camera readiness.
Small-face behavior, original DSP arithmetic and camera input remain separate.
"""
import argparse
import gzip
import json
from pathlib import Path

import torch

from decode_parameters import recover
from extract import MODELS, require, write_json
from flattened_k_diagnosis import BASE, build_detector
from preprocessing_investigation import identity, run_variant
from pretraining_data import sha256
from train_recognition import MODEL, source_hashes
from verify_external_baselines import independent_metrics


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    require(not args.output.exists(), 'Preserve previous runs')
    args.output.mkdir(parents=True)
    data = args.firmware.read_bytes()
    require(sha256(args.firmware)==MODELS[MODEL], 'Wrong original source')
    dataset = json.loads(args.dataset.read_text())
    frames = sorted((f for f in dataset['frames'] if f['split']=='validation'), key=lambda f:f['id'])
    require(len(frames)==80, 'Expected fixed existing validation80')
    policies = {name: dict(color='pillow_l',mean=117.,scale=1.,geometry=geometry)
                for name,geometry in [('k4_stretch','stretch'),('k4_letterbox','letterbox')]}
    sources = {**source_hashes(), **{n:sha256(BASE/n) for n in (
        'validate_flattened_k.py','flattened_k_diagnosis.py',
        'preprocessing_investigation.py','verify_external_baselines.py')}}
    snapshot = args.output/'source-snapshot'; snapshot.mkdir()
    for name in sources:
        (snapshot/name).write_bytes((BASE/name).read_bytes())
    plan = {'status':'Frozen before validation inference', 'policies':policies,
            'model':MODEL, 'model_sha256':MODELS[MODEL], 'manifest_sha256':sha256(args.dataset),
            'frames':[{'id':f['id'],'sha256':f['sha256']} for f in frames], 'sources':sources,
            'comparison':'Same existing validation80 and fixed external PicoDet/BlazeFace reports; no fitting or validation-driven parameter selection',
            'rationale':'K4 from original-byte constraints and train32 controls; letterbox is an explicitly separate host geometry policy, not a recovered camera routine',
            'small_face_definition':'sqrt(GT area after stretch to384x256)<32 for every model/policy',
            'thresholds':{'score':.5,'iou':.5}, 'training_gate':'Remains closed if small faces are not recovered, irrespective of overall AP',
            'optimizer_steps':0,'test_inference':False,'camera_numerical_parity':False}
    write_json(args.output/'frozen-plan.json',plan)
    detector = build_detector(data)
    before = identity(detector.model.state_dict())
    params,_=recover(data)
    results={}
    try:
        for name,cfg in policies.items():
            result=run_variant(detector,params,cfg,name,frames,args.dataset.parent,plan['manifest_sha256'],args.output)
            pred=json.loads(gzip.decompress((args.output/name/'predictions.json.gz').read_bytes()))
            verified={}
            for mode,document in pred.items():
                require(document['dataset_manifest_sha256']==plan['manifest_sha256'],'Prediction source differs')
                verified[mode]={}
                for cls in dataset['classes']:
                    filtered={r['id']:[item for item in r['detections'] if item['class']==cls] for r in document['frames']}
                    independent=independent_metrics({f['id']:f for f in frames},filtered,cls)
                    error=abs(independent['ap_at_iou']-result['metrics'][mode]['classes'][cls]['ap_at_iou'])
                    require(error<1e-12,'Independent AP mismatch')
                    verified[mode][cls]={**independent,'ap_recomputation_error':error}
            results[name]={'metrics':result['metrics'],'independent':verified,
                           'prediction_sha256':sha256(args.output/name/'predictions.json.gz')}
            print(name,json.dumps({m:{c:v['ap_at_iou'] for c,v in cs.items()} for m,cs in verified.items()}),flush=True)
    finally:
        detector.close()
    require(identity(detector.model.state_dict())==before,'Weights changed')
    require(all(sha256(BASE/n)==v for n,v in sources.items()),'Sources changed')
    require(sha256(args.dataset)==plan['manifest_sha256'],'Manifest changed')
    write_json(args.output/'summary.json',{'status':'complete','results':results,'state_sha256':before,
        'state_unchanged':True,'frozen_plan_sha256':sha256(args.output/'frozen-plan.json'),
        'optimizer_steps':0,'test_inference':False,'camera_numerical_parity':False,
        'ready_to_resume_recognition_training':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--firmware',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=BASE/'private-data/recognition-v2/manifest.json')
    p.add_argument('--output',type=Path,default=BASE/'artifacts/recognition-diagnosis-v3/k4-validation80')
    run(p.parse_args())
