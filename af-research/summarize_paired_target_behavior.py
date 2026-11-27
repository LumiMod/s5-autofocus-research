#!/usr/bin/env python3
"""Saved paired-training evidence audit and supplemental size stratification.

No inference. Paired statistics are replayed using their frozen calculation;
real-image AP/size counts use the separately authored independent accumulator.
"""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

from paired_target_behavior import summarize
from verify_external_baselines import independent_metrics

BASE = Path(__file__).resolve().parent


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_small(frame, box):
    return math.sqrt((box[2]-box[0])*(box[3]-box[1])*384*256/(frame['width']*frame['height'])) < 32


def mean(items):
    return sum(items)/len(items) if items else None


def translation_stats(rows):
    original = [r for r in rows if r['real_match']]
    return {'gt_translation_pairs': len(rows),
            'unique_gt': len({(r['frame_id'], r['gt_index']) for r in rows}),
            'shifted_matched': sum(bool(r['shifted_match']) for r in rows),
            'shifted_matched_fraction': mean([bool(r['shifted_match']) for r in rows]),
            'originally_matched_pairs': len(original),
            'following_given_original_match': sum(bool(r['follows_when_original_detected']) for r in original),
            'following_fraction_given_original_match': mean([bool(r['follows_when_original_detected']) for r in original])}


def compare_original_predictions(args, model, prediction):
    mapping = {'s5_baseline_o4_i4_hw': 'baseline_o4_i4_hw', 's5_flattened_k4': 'flattened_k4'}
    if model not in mapping:
        return None
    path = args.train32/mapping[model]/'predictions.json.gz'
    with gzip.open(path, 'rt') as stream:
        previous = {r['id']: r['detections'] for r in json.load(stream)['real']['frames']}
    max_box, max_score, compared = 0., 0., 0
    for row in prediction['records']:
        if row['variant'] != 'real':
            continue
        a, b = previous[row['frame_id']], row['detections']
        need(len(a) == len(b), 'Original-image S5 detection count differs from train32')
        for first, second in zip(a, b):
            need(first['class'] == second['class'], 'Original-image class ordering differs')
            max_score = max(max_score, abs(first['score']-second['score']))
            max_box = max(max_box, max(abs(x-y) for x, y in zip(first['bbox_xyxy'], second['bbox_xyxy'])))
            compared += 1
    need(max_score <= 1e-7 and max_box <= 1e-6, 'S5 adapter did not reproduce preceding fixed real-image predictions')
    return {'train32_predictions_sha256': sha(path), 'compared_detections': compared,
            'max_score_abs_error': max_score, 'max_coordinate_abs_error': max_box,
            'status': 'Saved real-image predictions agree up to coordinate restoration roundoff; no rerun inference.'}


def run(args):
    need(not args.output.exists(), 'Refusing to overwrite existing summary')
    old_path, new_path = args.root/'prepared/frozen-plan.json', args.root/'prepared-v2/frozen-plan.json'
    old, new = json.loads(old_path.read_text()), json.loads(new_path.read_text())
    old_semantic, new_semantic = dict(old), dict(new)
    old_semantic.pop('source_hashes'); new_semantic.pop('source_hashes')
    need(old_semantic == new_semantic, 'Source-only plan revision altered experimental semantics')
    plans = {sha(old_path): old, sha(new_path): new}
    frames = {r['frame']['id']: r['frame'] for r in new['frames']}
    need(all(f['split'] == 'train' for f in frames.values()) and len(frames) == 16, 'Not16 internal training frames')
    profiles = {}
    for model in new['frozen_profiles']:
        folder = args.root/'runs'/model
        prediction_path, report_path = folder/'predictions.json', folder/'report.json'
        prediction, saved = json.loads(prediction_path.read_text()), json.loads(report_path.read_text())
        plan_hash = prediction['plan_sha256']
        need(plan_hash in plans and saved['plan_sha256'] == plan_hash, 'Unknown/mismatched plan binding')
        need(saved['predictions_sha256'] == sha(prediction_path), 'Saved prediction SHA differs')
        replay = summarize(plans[plan_hash], prediction)
        need(all(saved[k] == value for k, value in replay.items()), 'Saved paired metrics do not replay exactly')
        need(prediction['adapter_state_before'] == prediction['adapter_state_after'], 'Adapter state changed')
        # Preserve each actual source revision. Existing old Pico snapshot stays
        # untouched even though the source-enumeration helper was later fixed.
        snapshots = {}
        for name, expected in prediction['source_hashes'].items():
            source = Path(name); destination = folder/'source-snapshot'/source.name
            if not destination.exists():
                need(source.is_file() and sha(source) == expected, 'Cannot preserve original source: '+name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(source.read_bytes())
            need(sha(destination) == expected, 'Preserved source snapshot SHA differs')
            snapshots[name] = {'sha256': expected, 'snapshot': str(destination.relative_to(args.root)),
                               'current_source_matches': source.is_file() and sha(source) == expected}
        real = {r['frame_id']: r['detections'] for r in prediction['records'] if r['variant'] == 'real'}
        class_results = {}
        for cls in prediction['classes']:
            filtered = {fid: [d for d in rows if d['class'] == cls] for fid, rows in real.items()}
            metrics = independent_metrics(frames, filtered, cls, .5)
            small_rows = [metrics['size_strata'][key] for key in ('below16', '16_to_below32')]
            total = sum(r['ground_truth'] for r in small_rows)
            hits = sum(r['matched_at_score'] for r in small_rows)
            all_hits = sum(r['matched_all_saved_scores'] for r in small_rows)
            metrics['small_s5_reference'] = {'ground_truth': total, 'matched_at_score_0_5': hits,
                'matched_all_saved_scores': all_hits, 'recall_at_score_0_5': hits/total if total else None,
                'recall_all_saved_scores': all_hits/total if total else None}
            strata = {}
            for label, small in [('small_lt32', True), ('not_small_ge32', False)]:
                rows = [r for r in saved['translation_pairs'] if r['class'] == cls
                        and is_small(frames[r['frame_id']], r['original_box']) == small]
                strata[label] = translation_stats(rows)
            class_results[cls] = {'independent_real_image_metrics': metrics,
                                 'translation_size_strata': strata,
                                 'frozen_behavior_metrics': saved['aggregates'][cls]}
        profiles[model] = {'model_metadata': prediction['model_metadata'], 'classes': class_results,
            'plan_sha256': plan_hash, 'predictions_sha256': sha(prediction_path), 'report_sha256': sha(report_path),
            'source_snapshots': snapshots, 'source_snapshot_hashes_verified': True,
            'paired_metrics_replayed_exactly': True, 'adapter_state_unchanged': prediction['adapter_state_after'],
            'original_s5_prediction_parity': compare_original_predictions(args, model, prediction)}
    report = {'scope': 'Train16 saved evidence only; no new inference or training.',
              'fixed_transform_rule_revision': 'Identical across all5 profiles; source-only fix documented separately.',
              'plan_hashes': list(plans), 'split': 'train', 'frames': 16,
              'inferences_in_completed_runs': sum(len(json.loads((args.root/'runs'/m/'predictions.json').read_text())['records']) for m in profiles),
              'profiles': profiles,
              'verification_scope': {'paired_metrics': 'Exact replay with original statistical implementation; not an independent second implementation.',
                  'real_ap_and_size': 'Independent accumulator from verify_external_baselines.py; no labels removed before size stratification.',
                  'source_hashes': 'Actual source revisions preserved and checked; hash checks do not imply model inference rerun.',
                  's5_adapter_parity': 'Compared saved original-image output against preceding train32, no new inference.'},
              'small_size_rule': 'sqrt(GT area mapped to reference384x256)<32; native and384-input Blaze have different costs.',
              'limits': new['limits']+['Local support depends jointly on score and IoU; a score drop does not isolate classification from localization.',
                  'Conditional following applies only to already-detected GT; see all-GT matched fractions and size counts for misses.',
                  'Cross-network confidence magnitudes are uncalibrated; compare within-model perturbation behavior only.',
                  'Paired samples share source images and GT; no independent-sample significance or accuracy gate is claimed.'],
              'optimizer_steps': 0, 'test_inference': False, 'validation_inference': False,
              'ready_to_resume_s5_training': False, 'verified_dsp_numerical_parity': False,
              'summary_source_sha256': sha(Path(__file__)),
              'independent_metric_source_sha256': sha(BASE/'verify_external_baselines.py')}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({model: {cls: {'small': r['independent_real_image_metrics']['small_s5_reference'],
                                  'small_translation': r['translation_size_strata']['small_lt32']}
                             for cls, r in profile['classes'].items()} for model, profile in profiles.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=BASE/'artifacts/recognition-diagnosis-v3/paired-behavior')
    parser.add_argument('--train32', type=Path, default=BASE/'artifacts/recognition-diagnosis-v3/flattened-k4-train32')
    parser.add_argument('--output', type=Path, default=BASE/'artifacts/recognition-diagnosis-v3/paired-behavior/summary.json')
    run(parser.parse_args())
