#!/usr/bin/env python3
"""Frozen train32 comparison of one fixed anchor-transfer initialization.

Original anchors and direct changed anchors are reference conditions. Exactly
one transfer strategy is declared before inference; results cannot alter it.
No optimizer, validation/test image access, checkpoint fitting or firmware write.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from anchor_head_transfer import apply_transfer, transfer_plan
from decode_parameters import recover
from extract import MODELS, require, write_json
from preprocessing_investigation import run_variant
from pretraining_data import sha256
from proposal_hypothesis import anchors
from research_paths import add_firmware_arguments, resolve_source
from train_recognition import Detector, MODEL, source_hashes, state_digest
from verify_external_baselines import independent_metrics

BASE = Path(__file__).resolve().parent
MANIFEST_SHA = 'af5459ddebd98947b154580cd176e7846abcceffe5539ac6f0001f6a485a3fc8'
CONFIG = dict(color='pillow_l', mean=117., scale=1., geometry='stretch')
PROFILES = ('original_anchors', 'direct_small_anchors', 'nearest_log_affine_transfer')
SCALES = [1, 2, 4, 8, 16]


def run(args):
    require(not args.output.exists(), 'Preserve previous evidence; choose a new output directory')
    require(sha256(args.dataset) == MANIFEST_SHA, 'Unexpected frozen dataset')
    data = (resolve_source(args)/(MODEL+'.bin')).read_bytes()
    require(hashlib.sha256(data).hexdigest() == MODELS[MODEL], 'Unexpected original model bytes')
    dataset = json.loads(args.dataset.read_text())
    frames = sorted((f for f in dataset['frames'] if f['split'] == 'train'), key=lambda f: f['id'])[:32]
    del dataset
    require(len(frames) == 32 and all(f['split'] == 'train' for f in frames), 'Expected fixed train32')
    params, _ = recover(data)
    newanchors = anchors(16, params['proposal_blocks'][0]['ratio_candidates'], SCALES, 16, 24, [16, 16]).astype(np.float32)
    newanchors[:, 2:] += 1
    plan_mapping = transfer_plan(data, newanchors)
    files = ('anchor_head_transfer.py', Path(__file__).name, 'preprocessing_investigation.py',
             'verify_external_baselines.py', 'research_paths.py')
    sources = {**source_hashes(), **{name: sha256(BASE/name) for name in files}}
    args.output.mkdir(parents=True)
    snapshot = args.output/'source-snapshot'; snapshot.mkdir()
    for name in sources:
        (snapshot/name).write_bytes((BASE/name).read_bytes())
    plan = dict(status='frozen_before_photo_inference', manifest_sha256=MANIFEST_SHA,
        model_sha256=MODELS[MODEL], source_hashes=sources,
        split='train', frames=[dict(id=f['id'], sha256=f['sha256']) for f in frames],
        profiles=list(PROFILES), controls=['real', 'same_mean', 'fixed_gray117', 'shuffled'],
        preprocessing=CONFIG, scales=SCALES, mapping=plan_mapping,
        selection_policy='One fixed nearest-log-scale plus exact bbox affine transfer; no search or selection from results.',
        comparison='Original anchors and direct-small anchors are same-source reference conditions, not alternate transfer candidates.',
        acceptance_policy='Diagnostic only; cannot itself authorize training, deployment, or original-camera numerical equivalence.',
        optimizer_steps=0, validation_inference=False, test_inference=False, firmware_writes=0)
    write_json(args.output/'frozen-plan.json', plan)
    torch.set_num_threads(args.threads); torch.manual_seed(20261127); torch.use_deterministic_algorithms(True)
    started = time.monotonic(); results = {}
    for name in PROFILES:
        d = Detector(data, 'gemm_k4')
        provenance = None
        if name == 'direct_small_anchors':
            d.anchors = newanchors.copy()
        elif name == 'nearest_log_affine_transfer':
            provenance = apply_transfer(d, data, newanchors)
        d.model.requires_grad_(False); d.model.eval()
        before = state_digest(d.model.state_dict())
        try:
            result = run_variant(d, params, CONFIG, name, frames, args.dataset.parent, MANIFEST_SHA, args.output)
            require(before == state_digest(d.model.state_dict()), 'State changed during inference')
            with gzip.open(args.output/name/'predictions.json.gz', 'rt') as stream:
                predictions = json.load(stream)
            metrics = {}
            for mode, document in predictions.items():
                metrics[mode] = {}
                for cls in ('person', 'person_face'):
                    per_frame = {row['id']: [box for box in row['detections'] if box['class'] == cls]
                                 for row in document['frames']}
                    metrics[mode][cls] = independent_metrics({f['id']: f for f in frames}, per_frame, cls)
                    require(abs(metrics[mode][cls]['ap_at_iou'] - result['metrics'][mode]['classes'][cls]['ap_at_iou']) < 1e-12,
                            'Independent AP differs')
            results[name] = dict(model_state_sha256=before, state_unchanged=True,
                                 independent_metrics=metrics, result=result, transfer_provenance=provenance)
            print(json.dumps(dict(profile=name, AP50={mode: {c: m['ap_at_iou'] for c, m in classes.items()}
                                                       for mode, classes in metrics.items()})), flush=True)
        finally:
            d.close()
    require(all(sha256(BASE/name) == value for name, value in sources.items()), 'Sources changed during diagnosis')
    require(sha256(args.dataset) == MANIFEST_SHA, 'Manifest changed during diagnosis')
    report = dict(status='complete', frozen_plan_sha256=sha256(args.output/'frozen-plan.json'),
        source_hashes=sources, results=results, elapsed_seconds=time.monotonic()-started,
        optimizer_steps=0, validation_inference=False, test_inference=False, firmware_writes=0,
        strategy_changed_after_results=False, ready_for_training=False,
        limitations=['Fixed train32 is diagnostic, not generalization or validation evidence.',
            'Transfer duplicates selected old slots and does not retain every original prediction.',
            'Exact affine identity applies before decoder clamp; original DSP arithmetic remains unverified.',
            'Original-scale quantization of migrated weights has a separately recorded clipping limitation.'])
    write_json(args.output/'report.json', report)
    print(json.dumps(dict(report=str(args.output/'report.json'), sha256=sha256(args.output/'report.json'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_firmware_arguments(parser)
    parser.add_argument('--dataset', type=Path, default=BASE/'private-data/recognition-v2/manifest.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    run(parser.parse_args())
