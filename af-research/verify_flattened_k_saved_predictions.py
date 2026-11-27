#!/usr/bin/env python3
"""Independent train32 metric recomputation from saved K4/baseline predictions.

Standard library only; no photo, model, validation/test inference, or optimizer.
Reuses the separately authored independent metric accumulator, not the producer.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

from verify_external_baselines import independent_metrics, verify_cyclic_control

BASE = Path(__file__).resolve().parent
CLASSES = ('person', 'person_face')
MODES = ('real', 'same_mean', 'fixed_gray117', 'shuffled')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_predictions(document, frames, manifest_sha):
    require(document['dataset_manifest_sha256'] == manifest_sha, 'Wrong prediction manifest')
    rows = {r['id']: r['detections'] for r in document['frames']}
    require(len(rows) == len(document['frames']) and set(rows) == set(frames), 'Wrong prediction coverage')
    for fid, detections in rows.items():
        frame = frames[fid]
        for row in detections:
            require(row['class'] in CLASSES and math.isfinite(row['score']) and 0 <= row['score'] <= 1, 'Invalid detection class/score')
            box = row['bbox_xyxy']
            require(len(box) == 4 and all(math.isfinite(x) for x in box), 'Nonfinite xyxy')
            x1, y1, x2, y2 = box
            require(-1e-9 <= x1 < x2 <= frame['width']+1e-9 and -1e-9 <= y1 < y2 <= frame['height']+1e-9,
                    'Box outside original image; raw values are preserved, no clipping for metric recomputation')
    return rows


def run(args):
    require(not args.output.exists(), 'Refusing to overwrite an existing verification')
    plan_path = args.run/'frozen-plan.json'
    summary_path = args.run/'summary.json'
    plan = json.loads(plan_path.read_text()); summary = json.loads(summary_path.read_text())
    manifest = json.loads(args.dataset.read_text()); manifest_sha = sha(args.dataset)
    require(plan['manifest_sha256'] == manifest_sha, 'Plan binds different dataset')
    require(summary['frozen_plan_sha256'] == sha(plan_path), 'Summary binds different frozen plan')
    selected = sorted((f for f in manifest['frames'] if f['split'] == 'train'), key=lambda f: f['id'])[:32]
    require(len(selected) == 32 and [f['id'] for f in selected] == [f['id'] for f in plan['frames']], 'Not the fixed first32 train frames')
    frames = {f['id']: f for f in selected}
    require(all(frames[r['id']]['sha256'] == r['sha256'] for r in plan['frames']), 'Plan image hashes differ from manifest')
    snapshot = {}
    for name, expected in plan['source_sha256'].items():
        saved = args.run/'source-snapshot'/name
        require(sha(saved) == expected, 'Source snapshot hash mismatch: '+name)
        snapshot[name] = {'recorded_sha256': expected, 'snapshot_verified': True,
                          'current_source_matches': (BASE/name).is_file() and sha(BASE/name) == expected}
    results = {}
    for profile in plan['profiles']:
        folder = args.run/profile
        report = json.loads((folder/'report.json').read_text())
        require(report['split'] == 'train' and report['images'] == 32, 'Unexpected report split/count')
        with gzip.open(folder/'predictions.json.gz', 'rt') as stream:
            predictions = json.load(stream)
        require(set(predictions) == set(MODES), 'Missing or unexpected controls')
        checked = {mode: check_predictions(predictions[mode], frames, manifest_sha) for mode in MODES}
        metrics, errors = {}, []
        for mode, rows in checked.items():
            metrics[mode] = {}
            for cls in CLASSES:
                filtered = {fid: [d for d in detections if d['class'] == cls] for fid, detections in rows.items()}
                actual = independent_metrics(frames, filtered, cls, .5)
                at_zero = independent_metrics(frames, filtered, cls, 0.)
                saved = report['metrics'][mode]['classes'][cls]
                difference = abs(actual['ap_at_iou']-saved['ap_at_iou'])
                require(difference <= 1e-12, 'Independent AP differs from saved report')
                errors.append(difference)
                for key in ('ground_truth_boxes', 'predictions', 'tp_at_score', 'fp_at_score', 'fn_at_score'):
                    require(at_zero[key] == saved[key], 'Saved score0 counts differ: '+key)
                small = [actual['size_strata'][name] for name in ('below16', '16_to_below32')]
                total = sum(r['ground_truth'] for r in small)
                hit = sum(r['matched_at_score'] for r in small)
                all_hit = sum(r['matched_all_saved_scores'] for r in small)
                actual['small_s5_reference'] = {'ground_truth': total, 'matched_at_score_0_5': hit,
                    'matched_all_saved_scores': all_hit, 'recall_at_score_0_5': hit/total if total else None,
                    'recall_all_saved_scores': all_hit/total if total else None}
                metrics[mode][cls] = actual
            mean_ap = sum(metrics[mode][cls]['ap_at_iou'] for cls in CLASSES)/2
            require(abs(mean_ap-report['metrics'][mode]['mean_ap']) <= 1e-12, 'Saved mean AP differs')
        wrong = verify_cyclic_control(frames, checked['real'], checked['shuffled'], 1)
        wrong['pairing'] = 'Fixed first32 sorted internal train IDs; donor=(target_position+1) mod32.'
        results[profile] = {'metrics': metrics, 'maximum_ap_abs_error': max(errors),
            'wrong_image_mapping': wrong, 'prediction_sha256': sha(folder/'predictions.json.gz'),
            'saved_report_sha256': sha(folder/'report.json'), 'all_saved_ap_and_score0_counts_match': True,
            'size_metrics_status': 'Independently recomputed from all-class one-to-one matches; newly reported here, not a comparison against previously saved size metrics.'}
    result = {'scope': 'Saved train32 predictions only; independent numerical recomputation, no image/model loading.',
              'plan_sha256': sha(plan_path), 'summary_sha256': sha(summary_path),
              'manifest_sha256': manifest_sha, 'source_snapshot_verification': snapshot,
              'verifier_sha256': sha(Path(__file__)),
              'metric_implementation_sha256': sha(BASE/'verify_external_baselines.py'),
              'results': results, 'optimizer_steps': 0, 'photo_inference': False,
              'test_inference': False, 'validation_inference': False,
              'verified_dsp_numerical_parity': False, 'ready_to_resume_s5_training': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({name: {'maximum_ap_abs_error': r['maximum_ap_abs_error'],
          'real': {cls: {'ap50': r['metrics']['real'][cls]['ap_at_iou'],
                         'small': r['metrics']['real'][cls]['small_s5_reference']} for cls in CLASSES}}
                      for name, r in results.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=BASE/'artifacts/recognition-diagnosis-v3/flattened-k4-train32')
    parser.add_argument('--dataset', type=Path, default=BASE/'private-data/recognition-v2/manifest.json')
    parser.add_argument('--output', type=Path, default=BASE/'artifacts/recognition-diagnosis-v3/flattened-k4-train32/independent-verification.json')
    run(parser.parse_args())
