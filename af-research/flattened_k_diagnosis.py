#!/usr/bin/env python3
"""Frozen, train-only test of a byte-supported flattened K4 weight layout.

Does not modify the historical network implementation, fit weights, or unlock
training. Transposed convolution and the existing decoder remain hypotheses.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from decode_parameters import recover
from extract import MODELS, require, write_json
from packing_investigation import dimensions, unpack
from preprocessing_investigation import identity, run_variant
from pretraining_data import sha256
from train_recognition import Detector, MODEL, source_hashes

BASE = Path(__file__).resolve().parent
CONFIG = dict(color='pillow_l', mean=117., scale=1., geometry='stretch')


def unpack_flattened_k4(raw, record):
    logical, physical, inputs, kh, kw = dimensions(record)
    k = inputs * kh * kw
    require(physical % 4 == 0 and k % 4 == 0, 'K4 dimensions do not align')
    result = np.asarray(raw).reshape(physical//4, k//4, 4, 4).transpose(0, 2, 1, 3)
    result = result.reshape(physical, inputs, kh, kw)
    require(not np.count_nonzero(result[logical:]), 'Nonzero output padding')
    return result[:logical].copy()


def pack_flattened_k4(weights, record):
    logical, physical, inputs, kh, kw = dimensions(record)
    require(weights.shape == (logical, inputs, kh, kw), 'Unexpected logical shape')
    padded = np.zeros((physical, inputs, kh, kw), dtype=weights.dtype)
    padded[:logical] = weights
    return padded.reshape(physical//4, 4, inputs*kh*kw//4, 4).transpose(0, 2, 1, 3).reshape(record['weight']['stored_shape']).copy()


def build_detector(data, profile='flattened_k4'):
    require(profile in ('baseline_o4_i4_hw', 'flattened_k4'), 'Unknown fixed profile')
    detector = Detector(data, 'o4_i4_hw')
    if profile == 'flattened_k4':
        params, tensors = recover(data)
        with torch.no_grad():
            for record in params['convolutions']:
                if record['name'] == 'upsample':
                    continue
                raw = tensors[record['tensor_key']+'_weight_i16_packed']
                weights = unpack_flattened_k4(raw, record)
                require(np.array_equal(pack_flattened_k4(weights, record), raw), 'K4 roundtrip differs')
                if record['kernel_hw_candidate'] == [1, 1]:
                    require(np.array_equal(weights, unpack(raw, record, 'o4_i4_hw')), 'K4 changes constrained 1x1 layout')
                detector.model.convs[str(record['index'])].weight.copy_(torch.from_numpy(weights.astype(np.float32)*record['weight_scale']))
    detector.layout = profile
    detector.model.requires_grad_(False)
    detector.model.eval()
    return detector


def run(args):
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    data = args.firmware.read_bytes()
    require(hashlib.sha256(data).hexdigest() == MODELS[MODEL], 'Wrong original model')
    dataset = json.loads(args.dataset.read_text())
    frames = sorted((f for f in dataset['frames'] if f['split'] == 'train'), key=lambda f: f['id'])[:32]
    require(len(frames) == 32, 'Need the fixed first32 training frames')
    require(not args.output.exists(), 'Use a new output directory; preserve past evidence')
    args.output.mkdir(parents=True)
    params, _ = recover(data)
    sources = {**source_hashes(), **{name: sha256(BASE/name) for name in (
        'flattened_k_diagnosis.py', 'preprocessing_investigation.py')}}
    snapshot = args.output/'source-snapshot'
    snapshot.mkdir()
    for name in sources:
        (snapshot/name).write_bytes((BASE/name).read_bytes())
    profiles = ('baseline_o4_i4_hw', 'flattened_k4')
    plan = {
        'status': 'Frozen before inference', 'model': MODEL, 'model_sha256': MODELS[MODEL],
        'manifest_sha256': sha256(args.dataset), 'source_sha256': sources,
        'selection': 'first32 lexical train IDs, identical to previous packing probes',
        'frames': [{'id': f['id'], 'sha256': f['sha256']} for f in frames],
        'profiles': list(profiles), 'preprocessing': CONFIG,
        'layout': '[O/4, (I*H*W)/4, O4, K4] -> OIHW, all ordinary convolutions',
        'basis': 'Whole zero input channels align with silent upstream output channels in both pinned model artifacts; conv1 reduces to HOW and 1x1 to the previously constrained layout.',
        'controls': ['real', 'same_mean', 'fixed_gray117', 'shuffled'],
        'screen': 'Each class real AP50 >= 0.5 and >= 0.15 above each control warrants subsequent independent validation; it does not unlock training.',
        'unchanged_hypotheses': ['bias scaling/order', 'maxpool/activations', 'upsample', 'camera preprocessing', 'anchors and float bbox decoder'],
        'optimizer_steps': 0, 'validation_inference': False, 'test_inference': False}
    write_json(args.output/'frozen-plan.json', plan)
    started = time.monotonic()
    results = {}
    for profile in profiles:
        detector = build_detector(data, profile)
        before = identity(detector.model.state_dict())
        try:
            result = run_variant(detector, params, CONFIG, profile, frames,
                                 args.dataset.parent, plan['manifest_sha256'], args.output)
            result['state_sha256'] = before
            result['state_unchanged'] = before == identity(detector.model.state_dict())
            require(result['state_unchanged'], 'Model changed during inference')
            results[profile] = result
            print(profile, json.dumps({mode: metric['mean_ap'] for mode, metric in result['metrics'].items()}), flush=True)
        finally:
            detector.close()
    require(all(sha256(BASE/name) == value for name, value in sources.items()), 'Source changed during probe')
    require(sha256(args.dataset) == plan['manifest_sha256'], 'Manifest changed')
    summary = {'status': 'complete', 'frozen_plan_sha256': sha256(args.output/'frozen-plan.json'),
               'results': results, 'elapsed_seconds': time.monotonic()-started,
               'optimizer_steps': 0, 'validation_inference': False, 'test_inference': False,
               'verified_dsp_numerical_parity': False, 'ready_to_resume_recognition_training': False}
    write_json(args.output/'summary.json', summary)
    print(json.dumps({'output': str(args.output), 'elapsed_seconds': summary['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--firmware', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=BASE/'private-data/recognition-v2/manifest.json')
    parser.add_argument('--output', type=Path, default=BASE/'artifacts/recognition-diagnosis-v3/flattened-k4-train32')
    parser.add_argument('--threads', type=int, default=2)
    run(parser.parse_args())
