#!/usr/bin/env python3
"""Read-only K4 validation80 audit; saved predictions plus synthetic geometry.

Never opens dataset photos or loads a detector. Numerical scores and strata
are independently accumulated. Synthetic tensor tests check the exact frozen
preprocessing functions, not recognition or firmware/camera equivalence.
"""
import argparse
import ast
import gzip
import hashlib
import json
import math
from pathlib import Path

from verify_external_baselines import independent_metrics
from verify_flattened_k_saved_predictions import check_predictions

BASE = Path(__file__).resolve().parent
CLASSES = ('person', 'person_face')
MODES = ('real', 'same_mean', 'fixed_gray117', 'shuffled')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_transform(frame, geometry):
    w, h = frame['width'], frame['height']
    if geometry == 'stretch':
        rw, rh, x, y = 384, 256, 0, 0
    else:
        ratio = min(384/w, 256/h)
        rw, rh = max(1, round(w*ratio)), max(1, round(h*ratio))
        x, y = (384-rw)//2, (256-rh)//2
    return {'sx': rw/w, 'sy': rh/h, 'x': x, 'y': y, 'width': rw, 'height': rh,
            'original_width': w, 'original_height': h}


def into_canvas(row, transform):
    return {**row, 'bbox_xyxy': [v*transform['sx' if i % 2 == 0 else 'sy']+transform['x' if i % 2 == 0 else 'y']
                               for i, v in enumerate(row['bbox_xyxy'])]}


def out_of_canvas(rows, transform):
    result = []
    for row in rows:
        box = [min(transform['original_width' if i % 2 == 0 else 'original_height'],
                   max(0., (v-transform['x' if i % 2 == 0 else 'y'])/transform['sx' if i % 2 == 0 else 'sy']))
               for i, v in enumerate(row['bbox_xyxy'])]
        if box[2] > box[0] and box[3] > box[1]:
            result.append({**row, 'bbox_xyxy': box})
    return result


def equal_lists(expected, actual):
    need(len(expected) == len(actual), 'Mapped detection count differs')
    maximum = 0.
    for left, right in zip(expected, actual):
        need(left['class'] == right['class'] and abs(left['score']-right['score']) < 1e-12, 'Mapped class or score differs')
        error = max(abs(a-b) for a, b in zip(left['bbox_xyxy'], right['bbox_xyxy']))
        maximum = max(maximum, error)
        need(error <= 1e-8, 'Mapped detection coordinates differ')
    return maximum


def fully_interior(row, frame):
    x1, y1, x2, y2 = row['bbox_xyxy']
    return x1 > 1e-8 and y1 > 1e-8 and x2 < frame['width']-1e-8 and y2 < frame['height']-1e-8


def is_full_canvas(transform):
    return transform['width'] == 384 and transform['height'] == 256 and transform['x'] == transform['y'] == 0


def require_row(expected, actual):
    matches = [row for row in actual if row['class'] == expected['class'] and abs(row['score']-expected['score']) < 1e-12]
    error = min((max(abs(a-b) for a, b in zip(row['bbox_xyxy'], expected['bbox_xyxy'])) for row in matches), default=math.inf)
    need(error <= 1e-8, 'Recoverable shuffled box does not follow target inverse transform')
    return error


def verify_canvas_controls(frames, transforms, checked, pairs):
    ids = sorted(frames)
    expected_pairs = [{'target_id': target, 'prediction_image_id': ids[(i+1) % len(ids)]} for i, target in enumerate(ids)]
    need(pairs == expected_pairs, 'Saved pairing is not the frozen80-frame cyclic derangement')
    reference_id = next((fid for fid in ids if is_full_canvas(transforms[fid])), None)
    need(reference_id is not None, 'Need an unpadded reference to audit all fixed-gray rows')
    gray_canvas = [into_canvas(row, transforms[reference_id]) for row in checked['fixed_gray117'][reference_id]]
    gray_error = 0.
    for fid in ids:
        expected = out_of_canvas(gray_canvas, transforms[fid])
        gray_error = max(gray_error, equal_lists(expected, checked['fixed_gray117'][fid]))
    exact_frames, recoverable_checks, wrong_error, boundary_boxes = 0, 0, 0., 0
    for pair in pairs:
        target, source = pair['target_id'], pair['prediction_image_id']
        source_rows, wrong_rows = checked['real'][source], checked['shuffled'][target]
        if is_full_canvas(transforms[source]):
            canvas = [into_canvas(row, transforms[source]) for row in source_rows]
            expected = out_of_canvas(canvas, transforms[target])
            wrong_error = max(wrong_error, equal_lists(expected, wrong_rows))
            exact_frames += 1
            continue
        # Cropping removes unknown raw-canvas endpoints. Check every box whose
        # endpoints are recoverable in either saved view, never invent the lost
        # raw extent or claim the entire letterbox shuffle was reconstructed.
        for row in source_rows:
            if not fully_interior(row, frames[source]):
                boundary_boxes += 1
                continue
            expected = out_of_canvas([into_canvas(row, transforms[source])], transforms[target])
            if expected:
                wrong_error = max(wrong_error, require_row(expected[0], wrong_rows))
                recoverable_checks += 1
        for row in wrong_rows:
            if not fully_interior(row, frames[target]):
                continue
            expected = out_of_canvas([into_canvas(row, transforms[target])], transforms[source])
            if expected:
                wrong_error = max(wrong_error, require_row(expected[0], source_rows))
                recoverable_checks += 1
    return {'fixed_gray_all80_frames_remap_verified': True,
            'fixed_gray_reference_frame_unpadded': reference_id,
            'maximum_gray_coordinate_error': gray_error,
            'wrong_pairing_all80_verified': True, 'wrong_complete_reconstruction_frames': exact_frames,
            'wrong_additional_recoverable_directional_box_checks': recoverable_checks,
            'wrong_source_boxes_touching_boundary_not_fully_invertible': boundary_boxes,
            'maximum_wrong_recoverable_coordinate_error': wrong_error,
            'same_author_wrong_pairs': sum(frames[p['target_id']]['group_id'] == frames[p['prediction_image_id']]['group_id'] for p in pairs),
            'wrong_semantics': 'Donor detections in384x256 network canvas are inverted using TARGET padding and resize. Letterbox is not a normalized-original-image wrong-prediction map.',
            'limits': 'All stretch shuffled rows reconstruct exactly. Letterbox clipping can remove raw endpoints/boxes; only full-canvas donors and recoverable interior rows can be reconstructed from saved output. Source logic and pairing inspected for remaining rows; raw canvas detections were not saved.'}


def synthetic_preprocessing_checks(snapshot):
    import numpy as np
    from PIL import Image
    import torch
    syntax = ast.parse(snapshot.read_text())
    wanted = {'prepare', 'matched_mean', 'back_to_photo'}
    nodes = [n for n in syntax.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    need(len(nodes) == 3, 'Missing frozen preprocessing functions')
    namespace = {'np': np, 'Image': Image, 'torch': torch}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(snapshot), 'exec'), namespace)
    count = 0
    for geometry in ('stretch', 'letterbox'):
        for w, h in [(31, 73), (73, 31), (101, 67)]:
            config = {'geometry': geometry, 'color': 'pillow_l', 'mean': 117., 'scale': 1.}
            rgb = Image.fromarray(np.arange(w*h*3, dtype=np.uint8).reshape(h, w, 3))
            tensor, transform = namespace['prepare'](rgb, config)
            need(transform == expected_transform({'width': w, 'height': h}, geometry), 'Frozen prepare transform mismatch')
            x, y, rw, rh = (transform[k] for k in ('x', 'y', 'width', 'height'))
            mean_image = namespace['matched_mean'](tensor, transform)
            tensor_np, mean_np = tensor.numpy()[0, 0], mean_image.numpy()[0, 0]
            mask = np.zeros((256, 384), bool); mask[y:y+rh, x:x+rw] = True
            need(np.array_equal(tensor_np[~mask], mean_np[~mask]), 'Matched-mean changed letterbox padding')
            expected_mean = float(tensor_np[mask].astype(np.float64).mean())
            need(np.allclose(mean_np[mask], expected_mean, rtol=0, atol=1e-5), 'Mean region differs')
            photo_boxes = [{'class': 'person', 'score': .8, 'bbox_xyxy': [0., 0., float(w), float(h)]},
                           {'class': 'person_face', 'score': .7, 'bbox_xyxy': [.2*w, .1*h, .8*w, .7*h]}]
            canvas = [into_canvas(row, transform) for row in photo_boxes]
            equal_lists(photo_boxes, namespace['back_to_photo'](canvas, transform))
            raw_extra = [{'class': 'person', 'score': .6, 'bbox_xyxy': [0., 0., 384., 256.]},
                         {'class': 'person_face', 'score': .5, 'bbox_xyxy': [0., 0., 1., 1.]}]
            equal_lists(out_of_canvas(raw_extra, transform), namespace['back_to_photo'](raw_extra, transform))
            gray_tensor, _ = namespace['prepare'](Image.new('RGB', (w, h), (117,)*3), config)
            need(not np.count_nonzero(gray_tensor.numpy()), 'RGB117 is not all-zero prepared tensor')
            count += 1
    return {'synthetic_shape_policy_cases': count, 'passed': True,
            'checks': ['resize and centered integer padding', 'non-square original-coordinate inverse',
                       'clipping and removal of empty padding boxes', 'mean control retains padding',
                       'uniform117 becomes the fixed zero-input control'],
            'scope': 'Pure synthetic image/tensor operations from frozen functions; no detector instantiated and no photo inference.'}


def run(args):
    need(not args.output.exists(), 'Preserve prior verification')
    plan_path, summary_path = args.run/'frozen-plan.json', args.run/'summary.json'
    plan, summary = json.loads(plan_path.read_text()), json.loads(summary_path.read_text())
    dataset = json.loads(args.dataset.read_text()); manifest_sha = sha(args.dataset)
    selected = sorted((f for f in dataset['frames'] if f['split'] == 'validation'), key=lambda f: f['id'])
    need(len(selected) == 80 and len({f['id'] for f in selected}) == 80, 'Not80 unique validation frames')
    need([f['id'] for f in selected] == [f['id'] for f in plan['frames']], 'Plan does not cover all validation frames')
    need(manifest_sha == plan['manifest_sha256'] and summary['frozen_plan_sha256'] == sha(plan_path), 'Manifest/plan binding mismatch')
    frames = {f['id']: f for f in selected}
    need(all(frames[r['id']]['sha256'] == r['sha256'] for r in plan['frames']), 'Plan image SHA differs from manifest')
    source_audit = {}
    for name, expected in plan['sources'].items():
        path = args.run/'source-snapshot'/name
        need(sha(path) == expected, 'Frozen source snapshot mismatch: '+name)
        source_audit[name] = {'recorded_sha256': expected, 'snapshot_verified': True,
                              'current_source_matches': (BASE/name).is_file() and sha(BASE/name) == expected}
    synthetic = synthetic_preprocessing_checks(args.run/'source-snapshot/preprocessing_investigation.py')
    results = {}
    for policy, config in plan['policies'].items():
        folder = args.run/policy
        report = json.loads((folder/'report.json').read_text())
        need(report['configuration'] == config and report['split'] == 'validation' and report['images'] == 80, 'Report policy/split mismatch')
        prediction_path = folder/'predictions.json.gz'
        need(summary['results'][policy]['prediction_sha256'] == sha(prediction_path), 'Prediction SHA differs from summary')
        with gzip.open(prediction_path, 'rt') as stream:
            prediction = json.load(stream)
        need(set(prediction) == set(MODES), 'Wrong controls')
        checked = {mode: check_predictions(document, frames, manifest_sha) for mode, document in prediction.items()}
        stats = json.loads((folder/'input-stats.json').read_text())
        need(len(stats) == 80 and {r['id'] for r in stats} == set(frames), 'Input transform evidence missing/duplicated')
        transforms = {r['id']: r['transform'] for r in stats}
        need(all(transforms[fid] == expected_transform(frame, config['geometry']) for fid, frame in frames.items()), 'Saved resize/padding differs from declared policy')
        metrics, ap_errors = {}, []
        for mode, rows in checked.items():
            metrics[mode] = {}
            for cls in CLASSES:
                filtered = {fid: [d for d in items if d['class'] == cls] for fid, items in rows.items()}
                value = independent_metrics(frames, filtered, cls, .5)
                all_scores = independent_metrics(frames, filtered, cls, 0.)
                saved = report['metrics'][mode]['classes'][cls]
                error = abs(value['ap_at_iou']-saved['ap_at_iou']); ap_errors.append(error)
                need(error <= 1e-12, 'AP differs from report')
                for key in ('ground_truth_boxes', 'predictions', 'tp_at_score', 'fp_at_score', 'fn_at_score'):
                    need(all_scores[key] == saved[key], 'Score0 counts differ')
                saved_independent = summary['results'][policy]['independent'][mode][cls]
                need(all(saved_independent[k] == v for k, v in value.items()), 'Saved independent size or score.5 metrics differ')
                groups = [value['size_strata'][k] for k in ('below16', '16_to_below32')]
                total = sum(r['ground_truth'] for r in groups)
                hit = sum(r['matched_at_score'] for r in groups)
                all_hit = sum(r['matched_all_saved_scores'] for r in groups)
                value['small_fixed384_reference'] = {'ground_truth': total, 'matched_at_score_0_5': hit,
                    'matched_all_saved_scores': all_hit, 'recall_at_score_0_5': hit/total if total else None,
                    'recall_all_saved_scores': all_hit/total if total else None}
                metrics[mode][cls] = value
            need(abs(sum(metrics[mode][c]['ap_at_iou'] for c in CLASSES)/2-report['metrics'][mode]['mean_ap']) <= 1e-12, 'Mean AP differs')
        controls = verify_canvas_controls(frames, transforms, checked, json.loads((folder/'shuffle-pairing.json').read_text()))
        face = metrics['real']['person_face']['small_fixed384_reference']
        need(face['ground_truth'] == 34 and face['matched_at_score_0_5'] == face['matched_all_saved_scores'] == 0,
             'Requested0/34 small-face conclusion does not follow saved predictions')
        # Label sizes always use original-image dimensions and fixed reference
        # stretch; actual letterbox image area is an explicitly separate number.
        actual_small = sum(math.sqrt((b['bbox_xyxy'][2]-b['bbox_xyxy'][0])*(b['bbox_xyxy'][3]-b['bbox_xyxy'][1])*
                                    transforms[fid]['sx']*transforms[fid]['sy']) < 32
                           for fid, frame in frames.items() for b in frame['boxes'] if b['class'] == 'person_face')
        results[policy] = {'metrics': metrics, 'maximum_ap_abs_error': max(ap_errors),
            'all80_frames_each_control_verified': True, 'all_saved_score0_and_score0_5_counts_match': True,
            'all_saved_size_strata_match': True, 'coordinate_and_control_audit': controls,
            'small_definitions': {'fixed_384x256_reference_count': 34,
                                  'actual_preprocessing_size_count_for_comparison_only': actual_small,
                                  'metric_uses': 'fixed reference, never actual letterbox area'},
            'prediction_sha256': sha(prediction_path), 'report_sha256': sha(folder/'report.json'),
            'input_stats_sha256': sha(folder/'input-stats.json')}
    verification = {'scope': 'Read-only all80 saved-validation predictions; metric recomputation and synthetic preprocessing tests, no model inference.',
        'manifest_sha256': manifest_sha, 'frozen_plan_sha256': sha(plan_path), 'summary_sha256': sha(summary_path),
        'source_snapshot_audit': source_audit, 'synthetic_preprocessing_checks': synthetic, 'results': results,
        'matching_review': 'Class-specific score-ordered one-to-one matching covers every GT before size bucketing; no large GT removal or pseudo small AP.',
        'review_findings': ['No AP, coordinate inverse, or fixed-reference size-definition error found.',
            'Letterbox wrong-image control is a donor network-canvas prediction interpreted through target geometry; it differs from external baseline normalized-original-image cyclic controls.',
            'Because raw canvas boxes were not saved, clipping prevents full numerical reconstruction of every letterbox shuffled endpoint; recoverable endpoints and all saved pairing/source logic are checked.'],
        'limits': ['This validation set was previously used for diagnostics and is not an untouched test.',
                   'Raw DSP arithmetic, camera preprocessing, and runtime deployment remain unverified.',
                   'Synthetic preprocessing tests and hash verification are not repeated detector inference.',
                   'All-score means all saved postNMS boxes; excluded preNMS/topK candidates are not represented.'],
        'optimizer_steps': 0, 'verification_photo_inference': False, 'test_inference': False,
        'ready_to_resume_s5_training': False, 'verified_dsp_numerical_parity': False,
        'verifier_sha256': sha(Path(__file__)), 'independent_metric_source_sha256': sha(BASE/'verify_external_baselines.py'),
        'prediction_validator_source_sha256': sha(BASE/'verify_flattened_k_saved_predictions.py')}
    args.output.write_text(json.dumps(verification, indent=2, allow_nan=False)+'\n')
    print(json.dumps({name: {'person_ap50': r['metrics']['real']['person']['ap_at_iou'],
                           'face_ap50': r['metrics']['real']['person_face']['ap_at_iou'],
                           'small': r['metrics']['real']['person_face']['small_fixed384_reference'],
                           'actual_small_count': r['small_definitions']['actual_preprocessing_size_count_for_comparison_only'],
                           'max_ap_error': r['maximum_ap_abs_error'],
                           'canvas_control_audit': r['coordinate_and_control_audit']} for name, r in results.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=BASE/'artifacts/recognition-diagnosis-v3/k4-validation80')
    parser.add_argument('--dataset', type=Path, default=BASE/'private-data/recognition-v2/manifest.json')
    parser.add_argument('--output', type=Path, default=BASE/'artifacts/recognition-diagnosis-v3/k4-validation80/independent-verification.json')
    run(parser.parse_args())
