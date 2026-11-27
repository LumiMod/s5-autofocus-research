#!/usr/bin/env python3
"""Repeat previously declared input/decoder contrasts with the new K4 layout.

Only first32 training IDs (preprocessing) / first8 (decoder). No fitting,
validation selection, or new test exposure. All candidates/results retained.
"""
import argparse
import itertools
import json
from pathlib import Path
import time

import numpy as np
import torch

from anchor_decode_investigation import ordered_anchors, predictions_from_heads, self_check
from decode_parameters import recover
from extract import MODELS, require, write_json
from flattened_k_diagnosis import BASE, build_detector
from preprocessing_investigation import candidates, compressed_json, identity, run_variant
from pretraining_data import evaluate_detection, sha256
from train_recognition import CLASSES, MODEL, load_image, source_hashes


def run(args):
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    data = args.firmware.read_bytes()
    require(sha256(args.firmware) == MODELS[MODEL], 'Wrong original model')
    dataset = json.loads(args.dataset.read_text())
    count = 32 if args.mode == 'preprocessing' else 8
    frames = sorted((f for f in dataset['frames'] if f['split']=='train'), key=lambda f: f['id'])[:count]
    require(len(frames)==count and not args.output.exists(), 'Need fixed samples and a new destination')
    args.output.mkdir(parents=True)
    params, _ = recover(data)
    detector = build_detector(data)
    before = identity(detector.model.state_dict())
    sources = {**source_hashes(), **{name: sha256(BASE/name) for name in (
        'flattened_k_diagnosis.py', 'flattened_k_followups.py',
        'preprocessing_investigation.py', 'anchor_decode_investigation.py')}}
    snap = args.output/'source-snapshot'; snap.mkdir()
    for name in sources:
        (snap/name).write_bytes((BASE/name).read_bytes())
    if args.mode == 'preprocessing':
        specs = candidates(params['convolutions'][0]['input_scale'])
    else:
        specs = {f'decoder-{i:02d}': dict(anchor_order=a, bbox_grouping=b,
                foreground_half=f, semantic_head_mapping=m)
                 for i, (a,b,f,m) in enumerate(itertools.product(
                    ('ratio_major','scale_major'), ('anchor_major','coordinate_major'),
                    (1,0), ('score1_person','score2_person')))}
    plan = {'mode': args.mode, 'reason': 'K4 improved image-specific signal; earlier contrasts used incorrect or unsupported spatial layouts',
            'specifications': specs, 'sample_selection': f'first{count} lexical train IDs',
            'frames': [{'id': f['id'], 'sha256': f['sha256']} for f in frames],
            'model_sha256': MODELS[MODEL], 'model_state_sha256': before,
            'manifest_sha256': sha256(args.dataset), 'sources': sources,
            'selection': 'Diagnostic contrasts only; retain baseline unless independent structural and behavior evidence supports a change',
            'optimizer_steps': 0, 'validation_inference': False, 'test_inference': False}
    write_json(args.output/'frozen-plan.json', plan)
    started = time.monotonic()
    results = {}
    try:
        if args.mode == 'preprocessing':
            for name, cfg in specs.items():
                results[name] = run_variant(detector, params, cfg, name, frames,
                    args.dataset.parent, plan['manifest_sha256'], args.output)
                print(name, json.dumps({m:r['mean_ap'] for m,r in results[name]['metrics'].items()}), flush=True)
        else:
            self_check()
            node = next(n for n in detector.model.layers if n['operator_id']==24)
            heads = []
            with torch.inference_mode():
                for frame in frames:
                    require(sha256(args.dataset.parent/frame['path'])==frame['sha256'], 'Image hash changed')
                    detector(load_image(args.dataset.parent, frame))
                    heads.append({k: detector.raw[key].cpu().numpy().copy()
                                  for k,key in [('score1','person'),('score2','person_face'),('bbox','bbox')]})
                detector(torch.zeros(1,1,256,384))
                gray = {k: detector.raw[key].cpu().numpy().copy()
                        for k,key in [('score1','person'),('score2','person_face'),('bbox','bbox')]}
            np.savez_compressed(args.output/'raw-heads.npz', **{
                f'image_{i}_{k}': value for i,raw in enumerate(heads) for k,value in raw.items()},
                **{'gray_'+k:v for k,v in gray.items()})
            for name, spec in specs.items():
                ref = ordered_anchors(node, params['proposal_blocks'][0], spec['anchor_order'])
                predictions = {}
                for mode in ('real','gray117','wrong_image'):
                    predictions[mode] = {'dataset_manifest_sha256': plan['manifest_sha256'], 'frames': []}
                    for i, frame in enumerate(frames):
                        raw = heads[i] if mode=='real' else gray if mode=='gray117' else heads[(i+1)%len(frames)]
                        predictions[mode]['frames'].append({'id':frame['id'], 'detections':predictions_from_heads(raw,ref,frame,spec,node)})
                results[name] = {'specification':spec, 'metrics':{m:evaluate_detection(
                    {'classes':list(CLASSES),'frames':frames}, p, 'train', score_threshold=0)
                    for m,p in predictions.items()}}
                compressed_json(args.output/(name+'-predictions.json.gz'),predictions)
                print(name, json.dumps({m:r['mean_ap'] for m,r in results[name]['metrics'].items()}), flush=True)
    finally:
        detector.close()
    require(identity(detector.model.state_dict())==before, 'Model weights changed')
    require(all(sha256(BASE/n)==v for n,v in sources.items()), 'Source changed during inference')
    require(sha256(args.dataset)==plan['manifest_sha256'], 'Manifest changed')
    write_json(args.output/'summary.json', {'status':'complete','results':results,
        'state_unchanged':True, 'elapsed_seconds':time.monotonic()-started,
        'frozen_plan_sha256':sha256(args.output/'frozen-plan.json'),
        'optimizer_steps':0,'validation_inference':False,'test_inference':False,
        'ready_to_resume_recognition_training':False})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('preprocessing','decoder'),required=True)
    p.add_argument('--firmware',type=Path,required=True)
    p.add_argument('--dataset',type=Path,default=BASE/'private-data/recognition-v2/manifest.json')
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())
