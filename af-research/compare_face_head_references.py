#!/usr/bin/env python3
"""Compare immutable saved outputs against separate official face/head labels.

Only images with supplied, non-group, non-depiction head labels enter each paired
cohort. Missing head annotations never become background. This does not relabel
historical runs or establish the original S5 training annotation convention.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from verify_external_baselines import independent_metrics, overlap

BASE = Path(__file__).resolve().parent
MANIFEST_SHA = 'af5459ddebd98947b154580cd176e7846abcceffe5539ac6f0001f6a485a3fc8'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def match_flags(frames, predictions):
    truths = {fid: f['boxes'] for fid, f in frames.items()}
    used = defaultdict(set)
    result = {}
    rows = sorted(((p['score'], fid, i, p) for fid, ps in predictions.items()
                   for i, p in enumerate(ps)), key=lambda r: (-r[0], r[1], r[2]))
    for score, fid, i, p in rows:
        choices = [(overlap(p['bbox_xyxy'], g['bbox_xyxy']), j)
                   for j, g in enumerate(truths[fid]) if j not in used[fid]]
        best, j = max(choices, default=(0., -1))
        hit = best >= .5 and j >= 0
        if hit:
            used[fid].add(j)
        result[(fid, i)] = dict(score=score, hit=hit, matched_gt=j if hit else None,
                                matched_iou=best if hit else None)
    return result


def run(args):
    need(not args.output.exists(), 'Preserve historical evidence')
    need(sha(args.dataset) == MANIFEST_SHA, 'Unexpected dataset')
    dataset = json.loads(args.dataset.read_text())
    reference = json.loads(args.head_reference.read_text())
    need(reference['manifest_sha256'] == MANIFEST_SHA, 'Head reference mismatch')
    originals = {f['id']: f for f in dataset['frames'] if f['split'] in ('train', 'validation')}
    heads = {f['id']: f for f in reference['frames']}
    need(set(originals) == set(heads), 'Reference frame identity differs')
    paths = {
        'validation/original': BASE/'artifacts/recognition-small-anchor-v2/baseline_original/predictions.json',
        'validation/transferred0': BASE/'artifacts/recognition-small-anchor-v2/baseline_small/predictions.json',
        'validation/trained6': BASE/'artifacts/recognition-small-anchor-v2/epoch-06/predictions.json',
        'train32/transferred0': BASE/'artifacts/recognition-fp-diagnosis/head-interactions/transfer0/predictions.json',
        'train32/trained6': BASE/'artifacts/recognition-fp-diagnosis/head-interactions/trained6/predictions.json',
    }
    sources = {n: sha(BASE/n) for n in (Path(__file__).name, 'verify_external_baselines.py')}
    args.output.mkdir(parents=True)
    snap = args.output/'source-snapshot'
    snap.mkdir()
    for name in sources:
        (snap/name).write_bytes((BASE/name).read_bytes())
    plan = dict(dataset_manifest_sha256=MANIFEST_SHA, head_reference_sha256=sha(args.head_reference),
        saved_prediction_sha256={k: sha(p) for k, p in paths.items()}, source_sha256=sources,
        score_threshold=.5, iou_threshold=.5,
        cohort='Paired same frames; head annotation exists; exclude whole frame if any head GroupOf/Depiction.',
        semantic_hypotheses=['Human face', 'Human head'],
        prediction_coordinates='Unmodified original-image xyxy, no head shrink or face expansion.',
        model_inference=False, optimizer_steps=0, test_access=False,
        source_labels_changed=False, historical_metrics_changed=False)
    (args.output/'frozen-plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    results = {}
    for name, path in paths.items():
        document = json.loads(path.read_text())
        need(document['dataset_manifest_sha256'] == MANIFEST_SHA, 'Prediction dataset differs')
        prediction_ids = [r['id'] for r in document['frames']]
        need(len(set(prediction_ids)) == len(prediction_ids), 'Duplicate prediction frame')
        need(all(fid in originals for fid in prediction_ids), 'Prediction outside train/validation')
        eligible = {fid for fid in prediction_ids if heads[fid]['head_annotation_available'] and
                    not heads[fid]['head_group_or_depiction_present']}
        split = 'train' if name.startswith('train32/') else 'validation'
        need(all(originals[fid]['split'] == split for fid in prediction_ids), 'Split mismatch')
        predictions = {r['id']: [p for p in r['detections'] if p['class'] == 'person_face']
                       for r in document['frames'] if r['id'] in eligible}
        views = {
            'face': {fid: {**originals[fid], 'boxes': [b for b in originals[fid]['boxes']
                                                     if b['class'] == 'person_face']} for fid in eligible},
            # Class name is adapted only inside this diagnostic metric view.
            # The official reference and original dataset are never modified.
            'head': {fid: {**originals[fid], 'boxes': [dict(class_='person_face', **b)
                for b in heads[fid]['boxes']]} for fid in eligible},
        }
        for f in views['head'].values():
            for b in f['boxes']:
                b['class'] = b.pop('class_')
        metrics = {label: independent_metrics(view, predictions, 'person_face')
                   for label, view in views.items()}
        flags = {label: match_flags(view, predictions) for label, view in views.items()}
        transitions = dict(both=0, face_only=0, head_only=0, neither=0)
        per_prediction = []
        for key in sorted(flags['face']):
            face, head = flags['face'][key], flags['head'][key]
            need(face['score'] == head['score'], 'Prediction score changed')
            category = ('both' if face['hit'] and head['hit'] else 'face_only' if face['hit']
                        else 'head_only' if head['hit'] else 'neither')
            if face['score'] >= .5:
                transitions[category] += 1
            per_prediction.append(dict(frame_id=key[0], prediction_index=key[1],
                score=face['score'], category=category, face=face, head=head))
        for label in views:
            tp = sum(v['hit'] and v['score'] >= .5 for v in flags[label].values())
            count = sum(v['score'] >= .5 for v in flags[label].values())
            need(tp == metrics[label]['tp_at_score'] and count-tp == metrics[label]['fp_at_score'],
                 'Independent matching implementation disagrees')
        results[name] = dict(frames=len(eligible), frame_ids=sorted(eligible),
            excluded_no_head=[fid for fid in prediction_ids if not heads[fid]['head_annotation_available']],
            excluded_head_group_or_depiction=[fid for fid in prediction_ids if heads[fid]['head_group_or_depiction_present']],
            metrics=metrics, same_prediction_matching_at_score05=transitions,
            per_prediction=per_prediction)
    need(results['validation/original']['frame_ids'] == results['validation/transferred0']['frame_ids'] ==
         results['validation/trained6']['frame_ids'], 'Validation cohorts differ')
    need(results['train32/transferred0']['frame_ids'] == results['train32/trained6']['frame_ids'], 'Training cohorts differ')
    need(all(sha(p) == plan['saved_prediction_sha256'][k] for k, p in paths.items()), 'Saved prediction changed')
    need(sha(args.dataset) == MANIFEST_SHA and sha(args.head_reference) == plan['head_reference_sha256'], 'Labels changed')
    report = dict(plan_sha256=sha(args.output/'frozen-plan.json'), results=results,
        model_inference=False, optimizer_steps=0, test_access=False,
        source_labels_changed=False, historical_metrics_changed=False,
        semantic_contract_confirmed=False, ready_to_train=False,
        limitations=['Face/head annotations have no instance pairing; independent matching is not an identity mapping.',
            'This cohort excludes unavailable head labels and flagged frames; it is not the original full80 metric.',
            'A better metric under one annotation convention does not prove factory DSP semantics.',
            'No definition is selected by score; both are retained and old AP/TP/FP remain unchanged.'])
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: dict(frames=v['frames'], metrics={t:{x:m[x] for x in
        ('ground_truth_boxes','ap_at_iou','tp_at_score','fp_at_score')} for t,m in v['metrics'].items()},
        matching=v['same_prediction_matching_at_score05']) for k,v in results.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=BASE/'private-data/recognition-v2/manifest.json')
    parser.add_argument('--head-reference', type=Path, default=BASE/'artifacts/recognition-ranking-repair/box-contract/head-reference.json')
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args())
