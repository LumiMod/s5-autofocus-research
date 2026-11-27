#!/usr/bin/env python3
"""Verify the opt-in loss repair with fixtures and frozen train32 gradients.

Gradients are taken only with respect to detached head-output leaves, never
model parameters. This is loss supervision verification, not model training.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from extract import MODELS, require, write_json
from pretraining_data import checked_file, sha256
from recognition_balanced_loss import (balanced_detection_loss, balanced_sample,
    equal_gt_weights, select_samples, HARD_NEGATIVE_BUDGET)
from research_paths import add_firmware_arguments, resolve_source
from train_recognition import (CLASSES, MODEL, Detector, assign_joint_targets,
    ground_truth, load_image, source_hashes, state_digest)

BASE = Path(__file__).resolve().parent
MANIFEST_SHA = 'af5459ddebd98947b154580cd176e7846abcceffe5539ac6f0001f6a485a3fc8'


def fixtures():
    result = {}
    owners = np.array([0]*20+[1]*3+[2], np.int64)
    picked = balanced_sample(np.arange(24), [0, 20, 23], owners, 9, np.random.default_rng(1))
    require(np.isin([0, 20, 23], picked).all(), 'Required GT coverage lost')
    require([int(np.sum(owners[picked] == i)) for i in range(3)] == [5, 3, 1], 'Scarce-GT water filling failed')
    weights = equal_gt_weights(picked, owners)
    require(np.allclose([weights[owners[picked] == i].sum() for i in range(3)], [3]*3), 'Unequal GT mass')
    result['imbalanced_gt_coverage_and_mass'] = True
    try:
        balanced_sample(np.arange(24), [0, 20, 23], owners, 2, np.random.default_rng(1))
    except ValueError:
        result['coverage_overflow_rejected'] = True
    else:
        raise AssertionError('Coverage overflow accepted')
    # Two equal-size anchors with separate centers, only the first overlaps GT.
    ref = np.asarray([[0, 0, 10, 10], [30, 0, 40, 10], [60, 0, 70, 10]], np.float32)
    gt = {'person': np.empty((0, 4), np.float32), 'person_face': ref[:1].copy()}
    delta = torch.zeros(3, 4, requires_grad=True)
    with torch.no_grad():
        delta[1, 0] = -3  # Static negative now decodes EXACTLY to the true face.
    logits = {c: torch.tensor([[0., 2.], [0., 3.], [0., 4.]], requires_grad=True) for c in CLASSES}
    loss, _, trace = balanced_detection_loss(logits, delta, ref, gt, np.random.default_rng(2), return_trace=True)
    row = trace['classes']['person_face']
    require(row['guarded_negative'].tolist() == [1], 'Correct decoded face not protected')
    require(2 in row['hard_negative'] and 1 not in row['negative'], 'Hard-negative selection/guard wrong')
    loss.backward()
    require(logits['person_face'].grad[0, 1] < 0, 'Positive foreground gradient wrong')
    require(logits['person_face'].grad[2, 1] > 0, 'False positive foreground gradient wrong')
    require(torch.equal(logits['person_face'].grad[1], torch.zeros(2)), 'Protected face pushed down')
    result['good_decoded_negative_guard_and_gradient_sign'] = True
    empty = {c: np.empty((0, 4), np.float32) for c in CLASSES}
    lg = {c: torch.zeros(3, 2, requires_grad=True) for c in CLASSES}
    dd = torch.zeros(3, 4, requires_grad=True)
    loss, _, tr = balanced_detection_loss(lg, dd, ref, empty, np.random.default_rng(3), return_trace=True)
    loss.backward()
    require(not len(tr['bbox_positive']) and torch.equal(dd.grad, torch.zeros_like(dd)), 'Background bbox gradient')
    require(all(torch.all(x.grad[:, 1] > 0) for x in lg.values()), 'Background CE gradient wrong')
    result['empty_gt_background_only'] = True
    # Large negative pool: stable hard tie order, mixed randomness, strict budget.
    many = np.tile(ref[2], (200, 1))
    lg = {c: torch.zeros(200, 2) for c in CLASSES}
    tr = select_samples(lg, torch.zeros(200, 4), many, empty, np.random.default_rng(5))
    for c in CLASSES:
        row = tr['classes'][c]
        require(row['hard_negative'].tolist() == list(range(64)), 'Hard tie order unstable')
        require(len(row['negative']) == len(np.unique(row['negative'])) == 128, 'Negative budget/duplicate')
        require(len(row['random_negative']) == 64, 'Mixed random quota wrong')
    result['negative_budget_ties_no_duplicates'] = True
    # No valid anchor, no GT: differentiable zero, not nan or exception.
    outside = np.asarray([[-40, -40, -30, -30]], np.float32)
    lg = {c: torch.zeros(1, 2, requires_grad=True) for c in CLASSES}
    dd = torch.zeros(1, 4, requires_grad=True)
    loss, _ = balanced_detection_loss(lg, dd, outside, empty, np.random.default_rng(6))
    loss.backward()
    require(loss.item() == 0 and all(torch.equal(x.grad, torch.zeros_like(x)) for x in lg.values()), 'Empty sample path')
    result['empty_sample_differentiable_zero'] = True
    return result


def run(args):
    require(not args.output.exists(), 'Preserve previous verification')
    tests = fixtures()
    require(sha256(args.dataset) == MANIFEST_SHA, 'Unexpected dataset')
    dataset = json.loads(args.dataset.read_text())
    frames = sorted((f for f in dataset['frames'] if f['split'] == 'train'), key=lambda f: f['id'])[:32]
    checkpoint_path = args.run/'epoch-06/checkpoint.pt'
    checkpoint_sha = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    require(checkpoint['completed_epoch'] == 6 and checkpoint['global_step'] == 480, 'Wrong checkpoint')
    data = (resolve_source(args)/(MODEL+'.bin')).read_bytes()
    require(hashlib.sha256(data).hexdigest() == MODELS[MODEL], 'Unexpected original firmware')
    sources = {**source_hashes(), **{n: sha256(BASE/n) for n in
               (Path(__file__).name, 'recognition_balanced_loss.py', 'research_paths.py')}}
    args.output.mkdir(parents=True)
    snapshot = args.output/'source-snapshot'
    snapshot.mkdir()
    for name in sources:
        (snapshot/name).write_bytes((BASE/name).read_bytes())
    write_json(args.output/'frozen-plan.json', dict(frame_ids=[f['id'] for f in frames], split='train',
        checkpoint_sha256=checkpoint_sha, source_hashes=sources, seed=20261127,
        optimizer_steps=0, validation_inference=False, test_inference=False,
        gradients='detached output logits/deltas only; no model parameter gradients',
        negative_guard_iou=.3, hard_negative_budget=64, max_class_samples=128,
        max_positive_samples=32, max_bbox_samples=128))
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    detector = Detector(data, 'gemm_k4')
    detector.model.load_state_dict(checkpoint['model_state'])
    detector.model.requires_grad_(False)
    detector.model.eval()
    detector.anchors = np.load(args.run/'anchors-small.npy', allow_pickle=False)
    before = state_digest(detector.model.state_dict())
    rng = np.random.default_rng(20261127)
    rows = []
    try:
        for frame in frames:
            checked_file(args.dataset.parent, frame)
            with torch.no_grad():
                raw, raw_delta = detector(load_image(args.dataset.parent, frame))
            logits = {c: raw[c].detach().clone().requires_grad_(True) for c in CLASSES}
            delta = raw_delta.detach().clone().requires_grad_(True)
            gt = ground_truth(frame)
            old_assignment = assign_joint_targets(detector.anchors, gt)
            rng_before = copy.deepcopy(rng.bit_generator.state)
            loss, info, trace = balanced_detection_loss(logits, delta, detector.anchors, gt, rng, return_trace=True)
            require(bool(torch.isfinite(loss)), 'Nonfinite loss')
            loss.backward()
            assigned = trace['assigned']
            for key in ('matched_gt', 'gt_boxes', 'gt_classes', 'forced_anchor_by_gt'):
                require(np.array_equal(assigned[key], old_assignment[key]), 'Shared owner changed: '+key)
            owners = assigned['matched_gt']
            checks = {}
            for class_index, c in enumerate(CLASSES):
                item = trace['classes'][c]
                p, n, h, guard = [item[k] for k in ('positive', 'negative', 'hard_negative', 'guarded_negative')]
                require(len(p) <= 32 and len(p)+len(n) <= 128, 'Class budget')
                require(not np.isin(guard, n).any() and np.all(item['decoded_max_iou'][n] < .3), 'Protected candidate sampled')
                require(np.isin(assigned['forced_anchor_by_gt'][assigned['gt_classes'] == class_index], p).all(), 'GT lost')
                require(np.isin(p, trace['bbox_positive']).all(), 'Positive lacks bbox supervision')
                require(bool((logits[c].grad[n, 1] > 0).all()), 'Negative not pushed toward background')
                require(bool((logits[c].grad[p, 1] <= 0).all()), 'Positive gradient wrong')
                require(bool((logits[c].grad[guard] == 0).all()), 'Guarded candidate gradient nonzero')
                positive_mass = [float(item['positive_weights'][owners[p] == g].sum()) for g in np.unique(owners[p])]
                require(not positive_mass or np.allclose(positive_mass, positive_mass[0], atol=1e-5), 'Per-GT class mass differs')
                available = np.flatnonzero(item['labels'] == 0)
                ce = torch.nn.functional.cross_entropy(logits[c][available], torch.zeros(len(available), dtype=torch.long), reduction='none').detach().numpy()
                expected_hard = available[np.argsort(-ce, kind='stable')[:min(64, 128-len(p), len(available))]]
                require(np.array_equal(h, expected_hard), 'Hard mining not largest loss')
                checks[c] = dict(selected_positive=p.tolist(), selected_negative=n.tolist(),
                    hard_negative=h.tolist(), guarded_negative=guard.tolist(),
                    positive_weights=item['positive_weights'].tolist(), per_gt_class_mass=positive_mass,
                    hard_negative_gradient_fg_min=float(logits[c].grad[h, 1].min()) if len(h) else None)
            bbox = trace['bbox_positive']
            mass = [float(trace['bbox_weights'][owners[bbox] == g].sum()) for g in np.unique(owners[bbox])]
            require(np.isin(assigned['forced_anchor_by_gt'], bbox).all(), 'BBox GT lost')
            require(not mass or np.allclose(mass, mass[0], atol=1e-5), 'Unequal bbox GT mass')
            require(all(p.grad is None for p in detector.model.parameters()), 'Model parameter gradient created')
            rows.append(dict(frame_id=frame['id'], rng_before_loss=rng_before, info=info,
                class_checks=checks, bbox_indices=bbox.tolist(), bbox_weights=trace['bbox_weights'].tolist(),
                per_gt_bbox_mass=mass))
    finally:
        detector.close()
    require(state_digest(detector.model.state_dict()) == before and sha256(checkpoint_path) == checkpoint_sha,
            'Model/checkpoint changed')
    require(all(sha256(BASE/n) == h for n, h in sources.items()), 'Source changed during verification')
    report = dict(fixtures=tests, frames=rows, train_images=len(rows), source_hashes=sources,
        plan_sha256=sha256(args.output/'frozen-plan.json'), checkpoint_sha256=checkpoint_sha,
        checkpoint_state_sha256=before, graph_and_weights_unchanged=True,
        positive_and_bbox_coverage_passed=True, gradient_direction_passed=True,
        optimizer_steps=0, validation_inference=False, test_inference=False,
        passed_supervision_checks=True, passed_detection_acceptance=False, ready_to_train=False,
        limitations=['No parameter update or post-repair detection measurement was performed.',
                     'Guarded/ignored candidates get no new classification supervision.',
                     'This repair cannot claim to eliminate localization FPs or recover all small faces.'])
    write_json(args.output/'report.json', report)
    print(json.dumps({k: report[k] for k in ('train_images', 'fixtures', 'passed_supervision_checks', 'ready_to_train')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_firmware_arguments(parser)
    parser.add_argument('--dataset', type=Path, default=BASE/'private-data/recognition-v2/manifest.json')
    parser.add_argument('--run', type=Path, default=BASE/'artifacts/recognition-small-anchor-v2')
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
