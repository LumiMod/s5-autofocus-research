#!/usr/bin/env python3
"""Audit train/validation GT assignment against the unchanged recovered anchors.

Reads annotation metadata only: no image pixels, model inference or optimization.
Test annotations are excluded. Legacy matching is reported as a diagnostic
comparison, never as accepted targets. This is not recognition-accuracy evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from decode_parameters import recover
from extract import MODELS, require, write_json
from ops_investigation import analyze
from pretraining_data import sha256
from proposal_hypothesis import anchors
from research_paths import add_firmware_arguments, resolve_source
from train_recognition import (CLASSES, MODEL, assign_joint_targets, assign_targets,
                               box_iou, ground_truth, sample_with_required, source_hashes)

ALLOWED_SPLITS = ('train', 'validation')


def recovered_anchors(data):
    params, _ = recover(data)
    node = next(n for n in analyze(data)['layers'] if n['operator_id'] == 24)
    p = params['proposal_blocks'][0]
    ref = anchors(node['anchor_base_size_candidate'], p['ratio_candidates'],
                  p['scale_candidates'], 16, 24, node['feature_stride_candidate']).astype(np.float32)
    ref[:, 2:] += 1  # inclusive recovered coordinates to continuous xyxy
    return ref


def audit_dataset(dataset, ref, splits=ALLOWED_SPLITS):
    """Return explicit matching/coverage failures; never inspect a test GT box."""
    require(set(splits).issubset(ALLOWED_SPLITS) and splits, 'Only train/validation target audits are allowed')
    require(dataset['classes'] == list(CLASSES), 'Unexpected class ordering')
    reports = {}
    for split in splits:
        rows = [frame for frame in dataset['frames'] if frame['split'] == split]
        require(rows, 'No frames for '+split)
        report = {'images': len(rows), 'gt_count': 0, 'unmatched_gt_count': 0,
                  'current_class_bbox_owner_conflicts': 0,
                  'forced_anchor_reuse_count': 0, 'sampling_coverage_failures': 0,
                  'budget_violation_images': 0, 'failures': [], 'classes': {}}
        forced_ious = {cls: [] for cls in CLASSES}
        for cls in CLASSES:
            report['classes'][cls] = {'gt_count': 0, 'current_positive_anchors': 0,
                                      'current_class_bbox_owner_conflicts': 0,
                                      'forced_matched_gt_count': 0, 'max_gt_per_image': 0,
                                      'legacy_positive_anchors': 0,
                                      'legacy_cross_class_bbox_owner_anchors': 0,
                                      'legacy_cross_class_bbox_owner_images': 0,
                                      'legacy_cross_class_owner_and_own_iou_below_0_5_anchors': 0,
                                      'legacy_cross_class_owner_and_own_iou_below_0_5_images': 0,
                                      'legacy_bbox_target_iou_with_own_class_below_0_5_anchors': 0,
                                      'legacy_bbox_target_iou_with_own_class_below_0_5_images': 0}
        for frame in rows:
            require(frame['width'] > 0 and frame['height'] > 0, 'Invalid dimensions: '+frame['id'])
            require(all(b['class'] in CLASSES for b in frame['boxes']), 'Unknown class: '+frame['id'])
            gt = ground_truth(frame)
            assignment = assign_joint_targets(ref, gt)
            boxes, classes = assignment['gt_boxes'], assignment['gt_classes']
            matched = assignment['matched_gt']
            forced = assignment['forced_anchor_by_gt']
            missing = assignment['unmatched_gt_indices']
            report['gt_count'] += len(boxes)
            report['unmatched_gt_count'] += len(missing)
            report['forced_anchor_reuse_count'] += len(forced[forced >= 0])-len(np.unique(forced[forced >= 0]))
            if len(missing):
                report['failures'].append({'frame_id': frame['id'], 'kind': 'unmatched_gt',
                                           'gt_indices': missing.tolist()})
            _, legacy_match = assign_targets(ref, boxes)
            budget_failed = False
            for class_index, cls in enumerate(CLASSES):
                current = report['classes'][cls]
                count = len(gt[cls])
                current['gt_count'] += count
                current['max_gt_per_image'] = max(count, current['max_gt_per_image'])
                cls_gt = classes == class_index
                cls_forced = forced[cls_gt]
                cls_iou = assignment['forced_match_ious'][cls_gt]
                current['forced_matched_gt_count'] += int(np.count_nonzero(cls_forced >= 0))
                forced_ious[cls].extend(cls_iou[np.isfinite(cls_iou)].tolist())
                positive = np.flatnonzero(assignment['labels'][cls] == 1)
                conflicts = int(np.count_nonzero((matched[positive] < 0) |
                              (classes[matched[positive]] != class_index))) if len(positive) else 0
                current['current_positive_anchors'] += len(positive)
                current['current_class_bbox_owner_conflicts'] += conflicts
                report['current_class_bbox_owner_conflicts'] += conflicts
                if conflicts:
                    report['failures'].append({'frame_id': frame['id'], 'kind': 'class_bbox_owner_conflict',
                                               'class': cls, 'count': conflicts})
                if count > 32:
                    budget_failed = True
                    report['failures'].append({'frame_id': frame['id'], 'kind': 'classification_gt_budget',
                                               'class': cls, 'gt_count': count, 'budget': 32})
                legacy_labels, _ = assign_targets(ref, gt[cls])
                legacy_positive = np.flatnonzero(legacy_labels == 1)
                current['legacy_positive_anchors'] += len(legacy_positive)
                if len(legacy_positive):
                    wrong_owner = classes[legacy_match[legacy_positive]] != class_index
                    below = box_iou(ref[legacy_positive], gt[cls]).max(1) < .5
                    wrong_count, strict_count = int(wrong_owner.sum()), int((wrong_owner & below).sum())
                    current['legacy_cross_class_bbox_owner_anchors'] += wrong_count
                    current['legacy_cross_class_bbox_owner_images'] += bool(wrong_count)
                    current['legacy_cross_class_owner_and_own_iou_below_0_5_anchors'] += strict_count
                    current['legacy_cross_class_owner_and_own_iou_below_0_5_images'] += bool(strict_count)
                    # Measure the old *regression target box*, not anchor IoU:
                    # would a perfect fit to that target be a same-class AP50 TP?
                    target_iou = box_iou(boxes[legacy_match[legacy_positive]], gt[cls]).max(1)
                    incompatible = int(np.count_nonzero(target_iou < .5))
                    current['legacy_bbox_target_iou_with_own_class_below_0_5_anchors'] += incompatible
                    current['legacy_bbox_target_iou_with_own_class_below_0_5_images'] += bool(incompatible)
            report['budget_violation_images'] += budget_failed
            if len(missing) or budget_failed:
                continue
            # Exercise the same sampling helpers/budgets at multiple RNG seeds.
            for seed in (0, 1, 20261127):
                rng = np.random.default_rng(seed)
                classification_positive = []
                for class_index, cls in enumerate(CLASSES):
                    positive = np.flatnonzero(assignment['labels'][cls] == 1)
                    required = forced[classes == class_index]
                    selected = sample_with_required(positive, required, 32, rng)
                    classification_positive.extend(selected.tolist())
                    if not set(required).issubset(selected):
                        report['sampling_coverage_failures'] += 1
                bbox_selected = sample_with_required(np.flatnonzero(matched >= 0),
                                                      classification_positive, 128, rng)
                if (not set(classification_positive).issubset(bbox_selected) or
                        set(matched[bbox_selected]) != set(range(len(boxes)))):
                    report['sampling_coverage_failures'] += 1
        for cls in CLASSES:
            values = np.asarray(forced_ious[cls], np.float32)
            report['classes'][cls]['forced_match_iou'] = {
                'count': len(values), 'below_0_5': int(np.count_nonzero(values < .5)),
                'below_0_3': int(np.count_nonzero(values < .3)),
                'below_0_1': int(np.count_nonzero(values < .1)),
                'min': float(values.min()) if len(values) else None,
                'p10': float(np.quantile(values, .1)) if len(values) else None,
                'median': float(np.median(values)) if len(values) else None}
        report['passed'] = not any(report[k] for k in (
            'unmatched_gt_count', 'current_class_bbox_owner_conflicts',
            'forced_anchor_reuse_count', 'sampling_coverage_failures', 'budget_violation_images'))
        reports[split] = report
    return {'schema_version': 1, 'scope': 'train/validation annotation metadata only; no test/inference/training',
            'limitations': ['Matching consistency and coverage do not demonstrate recognition accuracy.',
                            'Forced matches below 0.5 IoU are explicit coverage targets, not positive-overlap evidence.',
                            'Legacy cross-class ownership is a diagnostic comparison; current ownership is unique.'],
            'anchor_count': len(ref), 'anchor_sha256': hashlib.sha256(ref.tobytes()).hexdigest(),
            'sampling_seeds': [0, 1, 20261127], 'splits': reports,
            'passed': all(item['passed'] for item in reports.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_firmware_arguments(parser)
    parser.add_argument('--dataset', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--splits', nargs='+', choices=ALLOWED_SPLITS, default=list(ALLOWED_SPLITS))
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing to overwrite an existing target audit')
    data = (resolve_source(args)/(MODEL+'.bin')).read_bytes()
    require(hashlib.sha256(data).hexdigest() == MODELS[MODEL], 'Unknown source model')
    dataset = json.loads(args.dataset.read_text())
    result = audit_dataset(dataset, recovered_anchors(data), tuple(args.splits))
    result.update(dataset_manifest_sha256=sha256(args.dataset), source_model_sha256=MODELS[MODEL],
                  source_hashes={**source_hashes(), Path(__file__).name: sha256(Path(__file__))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps({'passed': result['passed'], 'output': str(args.output),
                      'splits': {name: {k: value[k] for k in ('images', 'gt_count', 'unmatched_gt_count',
                                'current_class_bbox_owner_conflicts', 'budget_violation_images')}
                                 for name, value in result['splits'].items()}}, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
