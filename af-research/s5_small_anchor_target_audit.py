#!/usr/bin/env python3
"""Independent annotation/gradient audit for a fixed 25-slot S5 anchor design.

Reads only local train annotations. Synthetic logits/deltas exercise the existing
loss; no image, network forward, optimizer, validation/test box or firmware write.
Passing target consistency is distinct from IoU >= .5 geometric coverage.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from decode_parameters import recover
from extract import MODELS, require, write_json
from ops_investigation import analyze
from pretraining_data import sha256
from proposal_hypothesis import anchors
from research_paths import add_firmware_arguments, resolve_source
from train_recognition import (CLASSES, MODEL, assign_joint_targets, box_iou,
                               detection_loss, encode_boxes, ground_truth)

BASE = Path(__file__).resolve().parent
MANIFEST_SHA = 'af5459ddebd98947b154580cd176e7846abcceffe5539ac6f0001f6a485a3fc8'
SEEDS = (0, 1, 20261127)


def empty_gt():
    return {c: np.empty((0, 4), np.float32) for c in CLASSES}


def valid_centers(ref):
    centers = (ref[:, :2] + ref[:, 2:]) / 2
    return ((centers[:, 0] >= 0) & (centers[:, 0] < 384) &
            (centers[:, 1] >= 0) & (centers[:, 1] < 256))


def independent_iou(a, b):
    """Scalar continuous-area implementation; independent of target code."""
    area_a = max(0., float(a[2]-a[0])) * max(0., float(a[3]-a[1]))
    area_b = max(0., float(b[2]-b[0])) * max(0., float(b[3]-b[1]))
    intersection = max(0., min(float(a[2]), float(b[2]))-max(float(a[0]), float(b[0])))
    intersection *= max(0., min(float(a[3]), float(b[3]))-max(float(a[1]), float(b[1])))
    return intersection / max(area_a + area_b - intersection, 1e-12)


def loss_gradient_probe(ref, gt, assigned, seed):
    """Observe actual sampling via gradients, without copying sampling logic."""
    logits = {c: torch.zeros((len(ref), 2), requires_grad=True) for c in CLASSES}
    values = np.zeros((len(ref), 4), np.float32)
    owned = np.flatnonzero(assigned['matched_gt'] >= 0)
    if len(owned):
        targets = encode_boxes(ref[owned], assigned['gt_boxes'][assigned['matched_gt'][owned]])
        # Nonzero known residual in every coordinate exposes every sampled bbox.
        values[owned] = targets + np.array([.25, -.25, .5, -.5], np.float32)
    delta = torch.tensor(values, requires_grad=True)
    total, info = detection_loss(logits, delta, ref, gt, np.random.default_rng(seed))
    require(bool(torch.isfinite(total)), 'Nonfinite synthetic loss')
    total.backward()
    require(bool(torch.isfinite(delta.grad).all()), 'Nonfinite bbox gradient')
    bbox_selected = set(torch.nonzero(delta.grad.abs().sum(1) > 0).flatten().tolist())
    positives = set()
    for ci, c in enumerate(CLASSES):
        gradient = logits[c].grad.numpy()
        require(np.isfinite(gradient).all(), 'Nonfinite class gradient')
        sampled = np.flatnonzero(np.abs(gradient).sum(1) > 0)
        selected_positive = set(np.flatnonzero(gradient[:, 1] < 0).tolist())
        selected_negative = set(np.flatnonzero(gradient[:, 1] > 0).tolist())
        required = set(assigned['forced_anchor_by_gt'][assigned['gt_classes'] == ci].tolist())
        require(required.issubset(selected_positive), 'Sampling dropped a required GT classification anchor')
        require(all(assigned['labels'][c][i] == 1 for i in selected_positive), 'Wrong positive class gradient')
        require(all(assigned['labels'][c][i] == 0 for i in selected_negative), 'Wrong negative class gradient')
        require(np.all(assigned['valid_anchors'][sampled]), 'Invalid anchor entered classification loss')
        require(len(sampled) <= 128 and len(selected_positive) <= 32, 'Changed classification budgets')
        positives.update(selected_positive)
    require(positives.issubset(bbox_selected), 'Classification-positive anchor lost bbox supervision')
    forced = set(assigned['forced_anchor_by_gt'].tolist())
    require(forced.issubset(bbox_selected), 'Sampling dropped a required GT regression anchor')
    require(all(assigned['matched_gt'][i] >= 0 and assigned['valid_anchors'][i] for i in bbox_selected),
            'Background or invalid anchor entered bbox loss')
    require(set(assigned['matched_gt'][list(bbox_selected)]) == set(range(len(assigned['gt_boxes']))),
            'Sampling dropped a GT owner')
    require(len(bbox_selected) <= 128, 'Changed bbox budget')
    if bbox_selected:
        # beta=1/9, residual abs .25,.25,.5,.5: mean SmoothL1=.375-1/18.
        require(abs(info['bbox_smooth_l1'] - (.375-1/18)) < 2e-6, 'Unexpected bbox reduction or targets')
        signs = delta.grad[list(bbox_selected)].sign().numpy()
        require(np.array_equal(signs, np.tile([1, -1, 1, -1], (len(signs), 1))),
                'BBox gradient sign disagrees with owned GT residual')
    return dict(seed=seed, classification_positives=len(positives), bbox_positives=len(bbox_selected),
                forced_gt_retained=len(forced), loss=float(total.detach()))


class TargetTests(unittest.TestCase):
    def test_nested_person_face_use_same_class_bbox_owner(self):
        ref = np.array([[0, 0, 20, 20], [0, 0, 8, 8]], np.float32)
        gt = dict(person=ref[:1], person_face=np.array([[1, 1, 3, 3]], np.float32))
        a = assign_joint_targets(ref, gt)
        np.testing.assert_array_equal(a['matched_gt'], [0, 1])
        np.testing.assert_array_equal(a['labels']['person'], [1, 0])
        np.testing.assert_array_equal(a['labels']['person_face'], [0, 1])
        loss_gradient_probe(ref, gt, a, 0)

    def test_coincident_cross_class_and_duplicate_gt_get_unique_anchors(self):
        ref = np.array([[2, 2, 12, 12]]*3, np.float32)
        gt = dict(person=ref[:1], person_face=ref[:2])
        a = assign_joint_targets(ref, gt)
        self.assertEqual(set(a['forced_anchor_by_gt']), {0, 1, 2})
        for index, owner in enumerate(a['matched_gt']):
            self.assertEqual(sum(a['labels'][c][index] == 1 for c in CLASSES), 1)
            self.assertEqual(a['labels'][CLASSES[a['gt_classes'][owner]]][index], 1)

    def test_zero_iou_is_unmatched_and_loss_rejects_it(self):
        ref = np.array([[20, 20, 30, 30]], np.float32)
        gt = empty_gt(); gt['person_face'] = np.array([[0, 0, 1, 1]], np.float32)
        a = assign_joint_targets(ref, gt)
        np.testing.assert_array_equal(a['forced_anchor_by_gt'], [-1])
        with self.assertRaisesRegex(ValueError, 'Unmatched GTs'):
            detection_loss({c: torch.zeros(1, 2) for c in CLASSES}, torch.zeros(1, 4), ref, gt,
                           np.random.default_rng(0))

    def test_overlapping_anchor_with_outside_center_cannot_force_match(self):
        ref = np.array([[-8, 0, 4, 12], [30, 30, 40, 40]], np.float32)
        gt = empty_gt(); gt['person_face'] = np.array([[0, 0, 2, 2]], np.float32)
        self.assertGreater(independent_iou(ref[0], gt['person_face'][0]), 0)
        a = assign_joint_targets(ref, gt)
        np.testing.assert_array_equal(a['forced_anchor_by_gt'], [-1])
        np.testing.assert_array_equal(a['labels']['person_face'], [-1, 0])

    def test_exact_center_boundary_and_partial_outside_policy(self):
        ref = np.array([[-1, -1, 1, 1], [383, 1, 385, 3], [1, 255, 3, 257], [-3, 1, -1, 3]], np.float32)
        gt = empty_gt(); gt['person_face'] = np.array([[0, 0, .5, .5]], np.float32)
        a = assign_joint_targets(ref, gt)
        np.testing.assert_array_equal(a['valid_anchors'], [True, False, False, False])
        np.testing.assert_array_equal(a['forced_anchor_by_gt'], [0])
        loss_gradient_probe(ref, gt, a, 1)

    def test_zero_area_nan_anchor_or_gt_rejected(self):
        valid = np.array([[0, 0, 10, 10]], np.float32)
        for malformed in (np.array([[0, 0, 0, 1]], np.float32), np.array([[0, 0, np.nan, 1]], np.float32)):
            with self.assertRaises(ValueError):
                assign_joint_targets(malformed, empty_gt())
            gt = empty_gt(); gt['person'] = malformed
            with self.assertRaises(ValueError):
                assign_joint_targets(valid, gt)

    def test_ignore_band_is_not_false_background(self):
        ref = np.array([[0, 0, 10, 10], [0, 0, 4, 10], [30, 30, 40, 40]], np.float32)
        gt = empty_gt(); gt['person'] = ref[:1]
        a = assign_joint_targets(ref, gt)
        np.testing.assert_array_equal(a['labels']['person'], [1, -1, 0])
        np.testing.assert_array_equal(a['labels']['person_face'], [0, -1, 0])

    def test_augmenting_path_recovers_contested_gt_coverage(self):
        ref = np.array([[0, 0, 2, 2], [4, 0, 6, 2], [8, 0, 10, 2]], np.float32)
        gt = empty_gt(); gt['person_face'] = np.array([[0, 0, 6, 2], [4, 0, 10, 2], [0, 0, 6, 2]], np.float32)
        a = assign_joint_targets(ref, gt)
        self.assertEqual(set(a['forced_anchor_by_gt']), {0, 1, 2})
        np.testing.assert_array_equal(a['matched_gt'][a['forced_anchor_by_gt']], [0, 1, 2])

    def test_unavailable_distinct_anchor_is_explicit(self):
        ref = np.array([[0, 0, 10, 10]], np.float32)
        gt = {c: ref.copy() for c in CLASSES}
        self.assertEqual(len(assign_joint_targets(ref, gt)['unmatched_gt_indices']), 1)

    def test_more_than_classification_budget_is_rejected(self):
        ref = np.array([[0, 0, 10, 10]]*33, np.float32)
        gt = empty_gt(); gt['person_face'] = ref.copy()
        with self.assertRaisesRegex(ValueError, 'exceeding the fixed budget'):
            detection_loss({c: torch.zeros(33, 2) for c in CLASSES}, torch.zeros(33, 4), ref, gt,
                           np.random.default_rng(0))

    def test_actual_loss_sampling_retains_all_gt_and_positive_bbox(self):
        ref = np.array([[0, 0, 10, 10]]*180, np.float32)
        gt = dict(person=np.array([[.5, .5, 9, 9]], np.float32),
                  person_face=np.array([[1, 1, 8, 8]], np.float32))
        a = assign_joint_targets(ref, gt)
        for seed in SEEDS:
            loss_gradient_probe(ref, gt, a, seed)

    def test_empty_gt_and_all_invalid_are_finite(self):
        for ref in (np.array([[0, 0, 10, 10]], np.float32), np.array([[-8, -8, -2, -2]], np.float32)):
            gt = empty_gt()
            loss_gradient_probe(ref, gt, assign_joint_targets(ref, gt), 0)

    def test_nontrain_boxes_are_never_accessed(self):
        class Poison(dict):
            def __getitem__(self, key):
                if key != 'split':
                    raise AssertionError('Nontrain record was inspected')
                return super().__getitem__(key)
        rows = train_frames({'classes': list(CLASSES), 'frames': [
            {'id': 'train', 'split': 'train'}, Poison(split='validation'), Poison(split='test')]})
        self.assertEqual(rows, [{'id': 'train', 'split': 'train'}])


def train_frames(dataset):
    require(dataset['classes'] == list(CLASSES), 'Unexpected class ordering')
    return sorted((f for f in dataset['frames'] if f['split'] == 'train'), key=lambda f: f['id'])


def make_anchors(data, scales):
    params, _ = recover(data)
    proposals = [n for n in analyze(data)['layers'] if n['operator_id'] == 24]
    require(len(proposals) == 2, 'Unexpected proposal count')
    first, p = proposals[0], params['proposal_blocks'][0]
    ratios = p['ratio_candidates']
    require(len(ratios) == 5 and len(scales) == 5, 'Keep original 25 anchor slots')
    for node, block in zip(proposals, params['proposal_blocks']):
        require(node['anchor_base_size_candidate'] == 16 and node['feature_stride_candidate'] == [16, 16],
                'Unexpected original base or stride')
        require(block['ratio_candidates'] == ratios, 'Class proposal ratio disagreement')
    ref = anchors(16, ratios, scales, 16, 24, [16, 16]).astype(np.float32)
    ref[:, 2:] += 1
    require(ref.shape == (9600, 4), 'Changed spatial/head shape')
    return ref, dict(base=16, stride=[16, 16], feature_grid=[16, 24], ratios=ratios,
                    scales=list(scales), original_scales=p['scale_candidates'], slots_per_cell=25,
                    slot_order='ratio outer, scale inner; spatial y then x',
                    anchor_sha256=hashlib.sha256(ref.tobytes()).hexdigest())


def audit_train(dataset, ref):
    frames = train_frames(dataset)
    require(len(frames) == 320, 'Expected frozen train320')
    records, frame_rows, failures = [], [], []
    valid = valid_centers(ref)
    fully_inside = ((ref[:, 0] >= 0) & (ref[:, 1] >= 0) & (ref[:, 2] <= 384) & (ref[:, 3] <= 256))
    for frame in frames:
        require(all(b['class'] in CLASSES for b in frame['boxes']), 'Unknown train class')
        require(frame['width'] > 0 and frame['height'] > 0, 'Invalid train dimensions')
        for item in frame['boxes']:
            original = np.asarray(item['bbox_xyxy'], np.float64)
            require(np.isfinite(original).all() and np.all(original[:2] >= 0) and
                    np.all(original[2:] <= [frame['width'], frame['height']]), 'Original GT outside image')
        gt = ground_truth(frame)
        a = assign_joint_targets(ref, gt)
        np.testing.assert_array_equal(a['valid_anchors'], valid)
        boxes, classes, matched, forced = a['gt_boxes'], a['gt_classes'], a['matched_gt'], a['forced_anchor_by_gt']
        # Existing float32 projection can exceed a full-width edge by one ULP.
        # Original annotation bounds above are strict; preserve targets unchanged.
        upper = np.nextafter(np.array([384, 256], np.float32), np.float32(np.inf))
        require(np.all(boxes[:, :2] >= 0) and np.all(boxes[:, 2:] <= upper), 'GT outside image')
        reserved = forced[forced >= 0]
        require(len(set(reserved.tolist())) == len(reserved), 'Forced anchor reuse')
        positive = np.flatnonzero(matched >= 0)
        for ci, c in enumerate(CLASSES):
            cls_positive = np.flatnonzero(a['labels'][c] == 1)
            require(np.all(matched[cls_positive] >= 0) and np.all(classes[matched[cls_positive]] == ci),
                    'Class/bbox owner conflict')
        require(np.all(np.stack([a['labels'][c][positive] == 1 for c in CLASSES]).sum(0) == 1),
                'More than one positive class owns a shared bbox')
        require(all(np.all(a['labels'][c][~valid] == -1) for c in CLASSES), 'Invalid anchor was labelled')
        overlap = box_iou(ref, boxes) if len(boxes) else np.empty((len(ref), 0), np.float32)
        for gi, box in enumerate(boxes):
            anchor = int(forced[gi])
            best = float(overlap[valid, gi].max(initial=0))
            if anchor >= 0:
                require(matched[anchor] == gi and valid[anchor], 'Reserved owner changed')
                scalar_iou = independent_iou(ref[anchor], box)
                require(scalar_iou > 0 and abs(scalar_iou-float(a['forced_match_ious'][gi])) < 2e-6,
                        'Zero-IoU or incorrect forced match')
            else:
                scalar_iou = None
            encoded = encode_boxes(ref[anchor:anchor+1], box[None])[0].tolist() if anchor >= 0 else None
            records.append(dict(frame_id=frame['id'], gt_union_index=gi, target_class=CLASSES[int(classes[gi])],
                reference_box=box.tolist(), sqrt_area=float(np.sqrt(np.prod(box[2:]-box[:2]))),
                best_valid_anchor_iou=best, best_fully_inside_anchor_iou=float(overlap[fully_inside, gi].max(initial=0)),
                forced_anchor=anchor, forced_iou=scalar_iou, forced_delta=encoded,
                valid_overlapping_anchor_count=int(np.count_nonzero(overlap[valid, gi] > 0))))
        if len(a['unmatched_gt_indices']):
            failures.append(dict(frame_id=frame['id'], kind='unmatched_gt', indices=a['unmatched_gt_indices'].tolist()))
            probes = []
        elif any(len(gt[c]) > 32 for c in CLASSES):
            failures.append(dict(frame_id=frame['id'], kind='class_gt_budget'))
            probes = []
        else:
            probes = [loss_gradient_probe(ref, gt, a, seed) for seed in SEEDS]
        frame_rows.append(dict(frame_id=frame['id'], gt_count=len(boxes), probes=probes))
    groups = {}
    for c in CLASSES:
        for size in ('all', 'small_lt32', 'medium_32_96', 'large_ge96'):
            rows = [r for r in records if r['target_class'] == c and (size == 'all' or
                size == 'small_lt32' and r['sqrt_area'] < 32 or
                size == 'medium_32_96' and 32 <= r['sqrt_area'] < 96 or
                size == 'large_ge96' and r['sqrt_area'] >= 96)]
            best = np.asarray([r['best_valid_anchor_iou'] for r in rows])
            forced_iou = np.asarray([r['forced_iou'] for r in rows if r['forced_iou'] is not None])
            groups[c+'/'+size] = dict(gt=len(rows), zero_geometric_overlap=int(np.count_nonzero(best <= 0)),
                geometry_iou_ge_03=int(np.count_nonzero(best >= .3)), geometry_iou_ge_05=int(np.count_nonzero(best >= .5)),
                geometry_iou_below_05=int(np.count_nonzero(best < .5)), forced_matched=len(forced_iou),
                forced_below_01=int(np.count_nonzero(forced_iou < .1)), forced_below_03=int(np.count_nonzero(forced_iou < .3)),
                forced_below_05=int(np.count_nonzero(forced_iou < .5)),
                forced_iou_min=float(forced_iou.min()) if len(forced_iou) else None,
                forced_iou_median=float(np.median(forced_iou)) if len(forced_iou) else None)
    return dict(target_consistency_passed=not failures, geometric_all_gt_iou_ge_05=all(r['best_valid_anchor_iou'] >= .5 for r in records),
        images=len(frames), gt_count=len(records), failures=failures, groups=groups,
        valid_anchors=int(valid.sum()), center_invalid_anchors=int((~valid).sum()),
        fully_inside_anchors=int(fully_inside.sum()), partial_outside_center_valid_anchors=int((valid & ~fully_inside).sum()),
        actual_loss_gradient_probes=sum(len(f['probes']) for f in frame_rows), frames=frame_rows, gt_records=records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_firmware_arguments(parser)
    parser.add_argument('--dataset', type=Path, default=BASE/'private-data/recognition-v2/manifest.json')
    parser.add_argument('--scales', nargs=5, type=float, default=[1, 2, 4, 8, 16])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing to overwrite an audit')
    torch.set_num_threads(1)
    files = [Path(__file__).name, 'train_recognition.py', 'proposal_hypothesis.py', 'decode_parameters.py',
             'ops_investigation.py', 'extract.py', 'pretraining_data.py']
    hashes = {name: sha256(BASE/name) for name in files}
    stream = io.StringIO()
    tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TargetTests))
    require(tests.wasSuccessful(), stream.getvalue())
    require(sha256(args.dataset) == MANIFEST_SHA, 'Unexpected manifest SHA')
    data = (resolve_source(args)/(MODEL+'.bin')).read_bytes()
    require(hashlib.sha256(data).hexdigest() == MODELS[MODEL], 'Unexpected model bytes')
    ref, config = make_anchors(data, args.scales)
    result = audit_train(json.loads(args.dataset.read_text()), ref)
    require(hashes == {name: sha256(BASE/name) for name in files}, 'Sources changed during audit')
    require(sha256(args.dataset) == MANIFEST_SHA, 'Manifest changed during audit')
    report = dict(schema_version=1, source_hashes=hashes, manifest_sha256=MANIFEST_SHA,
        source_model_sha256=MODELS[MODEL], anchor_config=config, semantic_tests=dict(passed=True, count=tests.testsRun, log=stream.getvalue()),
        scope=dict(split='train', image_reads=0, network_forwards=0, optimizer_steps=0, validation_boxes_inspected=False,
                   test_boxes_inspected=False, synthetic_gradient_probes=True, firmware_writes=0),
        boundary_policy='Positive area; centers x in [0,384), y in [0,256); partial-outside anchors remain valid. No clipping before regression encoding.',
        gt_projection_policy='Strict original annotation bounds; unchanged float32 projection allows one ULP at the right/bottom image edge.',
        limitations=['Target consistency does not establish detection accuracy or original DSP equivalence.',
            'A positive-IoU forced match may be far below .5; report geometry and forced supervision separately.',
            'Preserved stride16 limits small-GT alignment even with smaller scales.',
            'No network or optimizer was run; loss gradients are from synthetic predictions only.'],
        **result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(json.dumps({k: report[k] for k in ('target_consistency_passed', 'geometric_all_gt_iou_ge_05', 'images', 'gt_count', 'actual_loss_gradient_probes', 'groups')}, indent=2))
    print(json.dumps(dict(output=str(args.output), sha256=sha256(args.output))))
    raise SystemExit(0 if report['target_consistency_passed'] else 1)


if __name__ == '__main__':
    main()
