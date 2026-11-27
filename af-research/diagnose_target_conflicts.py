#!/usr/bin/env python3
"""Label/owner/sampling audit of train320 only. No image reads or model forward."""
import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from pretraining_data import sha256
from recognition_data import FACE, person_subclasses
from train_recognition import CLASSES, assign_joint_targets, box_iou, ground_truth

BASE = Path(__file__).resolve().parent


def independently_sample(candidates, required, limit, rng):
    """Reproduce historical RNG operations without calling historical sampler."""
    required = np.unique(required)
    assert len(required) <= limit and np.isin(required, candidates).all()
    remaining = candidates[~np.isin(candidates, required)]
    count = min(limit-len(required), len(remaining))
    if len(remaining) > count:
        remaining = rng.choice(remaining, count, replace=False)
    return np.concatenate((required, remaining))


def source_annotation_check(root, frames):
    people = person_subclasses(json.loads((root/'raw/hierarchy.json').read_text()))
    selected = {f['source_image_id']: f for f in frames}
    rows = defaultdict(list)
    counts = Counter()
    with (root/'raw/validation-bbox.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            if row['ImageID'] not in selected or row['LabelName'] not in people | {FACE}:
                continue
            rows[row['ImageID']].append(row)
            counts['source_rows'] += 1
            counts['confidence_'+row['Confidence']] += 1
            counts['source_'+row['Source']] += 1
            for name in ('IsOccluded', 'IsTruncated', 'IsGroupOf', 'IsDepiction', 'IsInside'):
                counts[name] += int(row[name])
    mismatch = []
    for sid, frame in selected.items():
        expected = set()
        for row in rows[sid]:
            cls = 'person' if row['LabelName'] in people else 'person_face'
            coordinates = tuple(float(row[n])*frame['width' if i%2 == 0 else 'height']
                                for i, n in enumerate(('XMin','YMin','XMax','YMax')))
            expected.add((cls, coordinates))
        actual = {(b['class'], tuple(b['bbox_xyxy'])) for b in frame['boxes']}
        if expected != actual:
            mismatch.append({'frame_id':frame['id'], 'missing':list(expected-actual), 'extra':list(actual-expected)})
        counts['exact_mapped_duplicates_removed'] += len(rows[sid])-len(expected)
        counts['mapped_annotations'] += len(expected)
    return dict(counts=counts, mismatches=mismatch,
        all_selected_source_target_annotations_preserved=not mismatch,
        source_csv_sha256=sha256(root/'raw/validation-bbox.csv'),
        hierarchy_sha256=sha256(root/'raw/hierarchy.json'),
        completeness_limit='Checks all downloaded target bbox rows; cannot prove every visible face is annotated. No image-level completeness labels were downloaded.')


def analyze_anchors(frames, ref):
    total = Counter()
    classes = {c:Counter() for c in CLASSES}
    forced = {c:Counter() for c in CLASSES}
    per_frame = {}
    examples = []
    for frame in frames:
        gt = ground_truth(frame)
        assigned = assign_joint_targets(ref, gt)
        owners = assigned['matched_gt']
        owner_classes = np.full(len(ref), -1, np.int64)
        owner_classes[owners >= 0] = assigned['gt_classes'][owners[owners >= 0]]
        forced_set = np.zeros(len(ref), bool)
        forced_set[assigned['forced_anchor_by_gt'][assigned['forced_anchor_by_gt']>=0]] = True
        total['frames'] += 1
        total['unmatched_gt'] += len(assigned['unmatched_gt_indices'])
        masks = {}
        for ci, cls in enumerate(CLASSES):
            overlaps = box_iou(ref, gt[cls])
            same = overlaps.max(1) if len(gt[cls]) else np.zeros(len(ref))
            opposite = (assigned['labels'][cls] == 0) & (owner_classes >= 0) & (owner_classes != ci)
            high = opposite & (same >= .5)
            band = opposite & (same >= .3) & (same < .5)
            masks[cls] = {'iou_ge_0_5':high, 'iou_0_3_to_0_5':band}
            for key, mask in masks[cls].items():
                classes[cls][key] += int(mask.sum())
                classes[cls][key+'_forced'] += int((mask & forced_set).sum())
                classes[cls][key+'_frames'] += int(mask.any())
            classes[cls]['all_negative'] += int((assigned['labels'][cls] == 0).sum())
            classes[cls]['opposite_owner_negative'] += int(opposite.sum())
            classes[cls]['ground_truth'] += len(gt[cls])
            classes[cls]['max_gt_per_image'] = max(classes[cls]['max_gt_per_image'], len(gt[cls]))
            for gi in np.flatnonzero(assigned['gt_classes'] == ci):
                ov = assigned['forced_match_ious'][gi]
                size = np.sqrt(np.prod(assigned['gt_boxes'][gi,2:]-assigned['gt_boxes'][gi,:2]))
                bucket = 'below_0_3' if ov < .3 else '0_3_to_0_5' if ov < .5 else 'at_least_0_5'
                forced[cls][bucket] += 1
                if size < 32:
                    forced[cls]['small_'+bucket] += 1
                    forced[cls]['small_total'] += 1
            for ai in np.flatnonzero(high)[:2]:
                if len(examples) >= 24:
                    break
                gi = int(overlaps[ai].argmax())
                examples.append(dict(frame_id=frame['id'], negative_class=cls, anchor_index=int(ai),
                    anchor=ref[ai].tolist(), same_class_gt=gt[cls][gi].tolist(),
                    same_class_iou=float(same[ai]), owner_class=CLASSES[owner_classes[ai]],
                    owner_gt=assigned['gt_boxes'][owners[ai]].tolist(), forced=bool(forced_set[ai])))
        per_frame[frame['id']] = (assigned, masks)
    return dict(total=total, classes=classes, forced_gt_iou=forced, examples=examples), per_frame


def replay_sampling(log_path, per_frame):
    totals = {c:Counter() for c in CLASSES}
    epoch_counts = Counter()
    counts_by_frame = Counter()
    for line in log_path.open():
        record = json.loads(line)
        fid = record['frame_id']
        assigned, masks = per_frame[fid]
        rng = np.random.default_rng()
        rng.bit_generator.state = record['rng_before_loss']
        positive_union = []
        for ci, cls in enumerate(CLASSES):
            labels = assigned['labels'][cls]
            required = assigned['forced_anchor_by_gt'][assigned['gt_classes'] == ci]
            positive = independently_sample(np.flatnonzero(labels == 1), required, 32, rng)
            negative = np.flatnonzero(labels == 0)
            if len(negative) > 128-len(positive):
                negative = rng.choice(negative, 128-len(positive), replace=False)
            assert len(positive) == record[cls+'_positive_anchors']
            assert len(required) == record[cls+'_forced_positive_anchors']
            assert len(np.unique(assigned['matched_gt'][positive])) == record[cls+'_supervised_gt']
            positive_union.extend(positive)
            totals[cls]['sampled_negatives'] += len(negative)
            totals[cls]['sampled_positives'] += len(positive)
            for key, mask in masks[cls].items():
                count = int(mask[negative].sum())
                totals[cls]['sampled_'+key] += count
                totals[cls]['visits_with_'+key] += int(count > 0)
        bbox = independently_sample(np.flatnonzero(assigned['matched_gt'] >= 0),
                                   np.asarray(positive_union), 128, rng)
        assert len(bbox) == record['bbox_positive_anchors']
        assert len(set(positive_union)) == record['bbox_required_classification_positives']
        epoch_counts[record['epoch']] += 1
        counts_by_frame[fid] += 1
    assert len(counts_by_frame) == 320 and set(counts_by_frame.values()) == {6}
    assert set(epoch_counts.values()) == {320} and len(epoch_counts) == 6
    return dict(visits=sum(epoch_counts.values()), epoch_visits=epoch_counts, classes=totals,
        sampled_counts_match_all_historical_log_rows=True,
        inference_performed=False, historical_scores_reconstructed=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=BASE/'private-data/recognition-v2/manifest.json')
    parser.add_argument('--run',type=Path,default=BASE/'artifacts/recognition-small-anchor-v2')
    parser.add_argument('--output',type=Path,default=BASE/'artifacts/recognition-fp-diagnosis/target-conflicts')
    args=parser.parse_args()
    assert not (args.output/'report.json').exists(), 'Preserve previous reports'
    dataset=json.loads(args.dataset.read_text())
    frames=sorted((f for f in dataset['frames'] if f['split']=='train'),key=lambda f:f['id'])
    assert len(frames)==320
    results={}
    for name in ('original','small'):
        ref=np.load(args.run/('anchors-'+name+'.npy'),allow_pickle=False)
        results[name], per_frame=analyze_anchors(frames,ref)
        results[name]['anchor_file_sha256']=sha256(args.run/('anchors-'+name+'.npy'))
        if name=='small':
            results[name]['historical_sampling']=replay_sampling(args.run/'training.jsonl',per_frame)
    source=source_annotation_check(args.dataset.parent,frames)
    nesting=Counter()
    for frame in frames:
        gt=ground_truth(frame)
        for face in gt['person_face']:
            inter=np.maximum(0,np.minimum(gt['person'][:,2:],face[2:])-np.maximum(gt['person'][:,:2],face[:2])).prod(1)
            frac=inter/np.prod(face[2:]-face[:2])
            nesting['faces']+=1
            nesting['face_at_least_95pct_inside_some_person']+=int((frac>=.95).any())
            nesting['face_person_iou_ge_0_5']+=int((box_iou(face[None],gt['person'])>=.5).any())
    report=dict(dataset_sha256=sha256(args.dataset),training_log_sha256=sha256(args.run/'training.jsonl'),
        split='train',frame_count=320,frame_ids=[f['id'] for f in frames],source_annotations=source,
        nesting=nesting,anchors=results,
        source_sha256={n:sha256(BASE/n) for n in ('diagnose_target_conflicts.py','train_recognition.py','recognition_data.py')},
        optimizer_steps=0,model_inference=False,image_reads=False,validation_evaluation=False,test_evaluation=False,
        interpretation=[
            'Cross-class anchor-space overlap is potential semantic competition, not proof of a wrong loss: a shared bbox has exactly one GT owner and its output is trained toward that owner.',
            'Class label 0 with a decoded box already overlapping same-class GT is a separate prediction-dependent conflict; this label-only audit does not reconstruct those scores or boxes.',
            'IoU below .3 is background, .3 to below .5 is ignored, except forced positive owners and the opposite-class negatives they imply; these are IoU thresholds, not confidence thresholds.',
            'All forced GT positives are retained by historical sampling; equal coverage does not imply equal gradient mass.',
            'No background-only images and no annotation completeness verification: these remain dataset limitations, not demonstrated dropped source labels.'
        ])
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:{'classes':v['classes'],'forced':v['forced_gt_iou']} for k,v in results.items()}))
    print(json.dumps(source))
    print(json.dumps(results['small']['historical_sampling']))


if __name__=='__main__':main()
