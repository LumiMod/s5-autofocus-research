#!/usr/bin/env python3
"""Frozen train32 head-swap interventions. No optimizer or held-out inference.

These intentionally mixed checkpoints isolate score and box effects inside the
restored host graph. They are diagnostic interventions, never deployable models.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from extract import MODELS, require, write_json
from pretraining_data import checked_file, sha256
from research_paths import add_firmware_arguments, resolve_source
from train_recognition import (CLASSES, MODEL, Detector, detections, load_image,
                               source_hashes, state_digest)
from train_small_anchor_transfer import measure

BASE = Path(__file__).resolve().parent
MANIFEST_SHA = 'af5459ddebd98947b154580cd176e7846abcceffe5539ac6f0001f6a485a3fc8'
CONDITIONS = {
    'transfer0': ('initial', 'initial'),
    'trained6': ('final', 'final'),
    'trained_scores_initial_boxes': ('final', 'initial'),
    'initial_scores_trained_boxes': ('initial', 'final'),
}


def run(args):
    require(not args.output.exists(), 'Preserve prior diagnostic results')
    require(sha256(args.dataset) == MANIFEST_SHA, 'Unexpected dataset')
    dataset = json.loads(args.dataset.read_text())
    frames = sorted((f for f in dataset['frames'] if f['split'] == 'train'),
                    key=lambda f: f['id'])[:32]
    paths = {'initial': args.run/'baseline_small/checkpoint.pt',
             'final': args.run/'epoch-06/checkpoint.pt'}
    inputs = {k: sha256(p) for k, p in paths.items()}
    states = {k: torch.load(p, map_location='cpu', weights_only=False)['model_state']
              for k, p in paths.items()}
    prefixes = ('convs.86.', 'convs.90.', 'convs.94.')
    require(states['initial'].keys() == states['final'].keys(), 'Graph state keys changed')
    require(all(torch.equal(v, states['final'][k]) for k, v in states['initial'].items()
                if not k.startswith(prefixes)), 'Backbone changed between checkpoints')
    for k, value in states['initial'].items():
        require(value.shape == states['final'][k].shape, 'Parameter shape changed')
    data = (resolve_source(args)/(MODEL+'.bin')).read_bytes()
    require(hashlib.sha256(data).hexdigest() == MODELS[MODEL], 'Unexpected original model')
    anchor_path = args.run/'anchors-small.npy'
    anchors = np.load(anchor_path, allow_pickle=False)
    require(anchors.shape == (9600, 4), 'Unexpected anchors')
    sources = {**source_hashes(), **{n: sha256(BASE/n) for n in (
        Path(__file__).name, 'train_small_anchor_transfer.py', 'verify_external_baselines.py')}}
    args.output.mkdir(parents=True)
    snapshot = args.output/'source-snapshot'
    snapshot.mkdir()
    for name in sources:
        (snapshot/name).write_bytes((BASE/name).read_bytes())
    plan = dict(split='train', frame_ids=[f['id'] for f in frames],
                checkpoint_sha256=inputs, dataset_manifest_sha256=MANIFEST_SHA,
                anchor_file_sha256=sha256(anchor_path), conditions=CONDITIONS,
                class_head_keys=['86', '90'], bbox_head_key='94',
                source_hashes=sources, optimizer_steps=0,
                validation_inference=False, test_inference=False,
                score=.5, iou=.5, pre_nms_top=3000, nms=.3, final_top=30)
    write_json(args.output/'frozen-plan.json', plan)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    detector = Detector(data, 'gemm_k4')
    detector.anchors = anchors
    detector.model.requires_grad_(False)
    detector.model.eval()
    results = {}
    tensor_hashes = {}
    # Cache only these explicitly selected training images, never validation/test.
    tensors = []
    for frame in frames:
        checked_file(args.dataset.parent, frame)
        tensor = load_image(args.dataset.parent, frame)
        tensors.append(tensor)
        tensor_hashes[frame['id']] = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
    with torch.inference_mode():
        try:
            for condition, (cls_source, bbox_source) in CONDITIONS.items():
                state = {k: states[cls_source if k.startswith(('convs.86.', 'convs.90.'))
                                  else bbox_source if k.startswith('convs.94.') else 'initial'][k]
                         for k in states['initial']}
                detector.model.load_state_dict(state)
                before = state_digest(detector.model.state_dict())
                rows = []
                for frame, tensor in zip(frames, tensors):
                    logits, delta = detector(tensor)
                    rows.append(dict(id=frame['id'], detections=detections(
                        logits, delta, anchors, frame)))
                document = dict(dataset_manifest_sha256=MANIFEST_SHA, frames=rows,
                                model_state_sha256=before,
                                plan_sha256=sha256(args.output/'frozen-plan.json'))
                result = measure(dataset, frames, document)
                require(state_digest(detector.model.state_dict()) == before, 'State changed')
                require(all(p.grad is None for p in detector.model.parameters()), 'Unexpected gradient')
                output = args.output/condition
                output.mkdir()
                write_json(output/'predictions.json', document)
                write_json(output/'metrics.json', result)
                results[condition] = dict(metrics=result, state_sha256=before,
                    predictions_sha256=sha256(output/'predictions.json'))
        finally:
            detector.close()
    require(all(sha256(paths[k]) == v for k, v in inputs.items()), 'Checkpoint changed')
    require(all(sha256(BASE/k) == v for k, v in sources.items()), 'Source changed')
    report = dict(plan_sha256=sha256(args.output/'frozen-plan.json'), source_hashes=sources,
                  input_tensor_sha256=tensor_hashes, results=results,
                  checkpoint_files_unchanged=True, graph_and_backbone_unchanged=True,
                  optimizer_steps=0, validation_inference=False, test_inference=False,
                  ready_to_train=False,
                  limitations=['Mixed heads are causal host interventions, not usable camera models.',
                               'Training-set interventions do not establish independent accuracy.',
                               'Both checkpoints use the same new anchors; original-anchor effects are excluded.'])
    write_json(args.output/'report.json', report)
    print(json.dumps({k: {'face': v['metrics']['classes']['person_face'],
                         'small': v['metrics']['small_faces']}
                      for k, v in results.items()}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_firmware_arguments(parser)
    parser.add_argument('--dataset', type=Path, default=BASE/'private-data/recognition-v2/manifest.json')
    parser.add_argument('--run', type=Path, default=BASE/'artifacts/recognition-small-anchor-v2')
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
