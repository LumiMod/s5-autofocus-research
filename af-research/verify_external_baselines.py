#!/usr/bin/env python3
"""Independent, saved-validation-prediction checks; no model/image/test loading.

The inputs use the existing frames/detections JSON schema. This verifier
reports one supported class at a time and never grants S5 training/deployment
readiness. Only standard-library modules are imported.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


CLASSES = ('person', 'person_face')
BINS = ('below16', '16_to_below32', '32_to_below96', '96_or_larger')
RUN_MODES = {'real': 'real', 'mean': 'same_image_RGB_channel_mean',
             'gray': 'fixed_gray117', 'wrong': 'cyclic_wrong'}


def checksum(data):
    return hashlib.sha256(data).hexdigest()


def size_bin(frame, box):
    area = ((box[2] - box[0]) * 384 / frame['width'] *
            (box[3] - box[1]) * 256 / frame['height'])
    side = math.sqrt(area)
    return BINS[0] if side < 16 else BINS[1] if side < 32 else BINS[2] if side < 96 else BINS[3]


def overlap(a, b):
    widths = (min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1]))
    intersection = max(0., widths[0]) * max(0., widths[1])
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.


def check_box(box, frame):
    if len(box) != 4 or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in box):
        raise ValueError('Invalid finite xyxy box')
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= frame['width'] and 0 <= y1 < y2 <= frame['height']):
        raise ValueError('Box is not a positive, original-image, continuous xyxy rectangle')


def read_predictions(path, frames, manifest_sha, target_class):
    raw = path.read_bytes()
    document = json.loads(raw)
    if document.get('dataset_manifest_sha256') != manifest_sha:
        raise ValueError('Prediction manifest SHA does not match supplied dataset')
    by_id = {}
    for row in document['frames']:
        fid = row['id']
        if fid not in frames or fid in by_id:
            raise ValueError('Prediction contains duplicate or non-validation frame')
        by_id[fid] = row['detections']
        for detection in row['detections']:
            if detection['class'] != target_class:
                raise ValueError('Model output contains an unsupported/mis-mapped class')
            score = detection['score']
            if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Invalid prediction confidence')
            check_box(detection['bbox_xyxy'], frames[fid])
    if set(by_id) != set(frames):
        raise ValueError('Predictions must include every validation frame, even with no detections')
    return by_id, {'basename': path.name, 'sha256': checksum(raw), 'bytes': len(raw)}


def independent_metrics(frames, predictions, target_class, threshold=.5):
    """Score-ordered, one-to-one matching, independently accumulated AP50.

    Ties use frame ID then prediction position, with larger GT index breaking
    equal-IoU ties, matching the declared local metric. Size recall partitions
    the full-class matching; it does not erase non-small GT to create false FPs.
    """
    gt = {fid: [b['bbox_xyxy'] for b in frame['boxes'] if b['class'] == target_class]
          for fid, frame in frames.items()}
    total = sum(map(len, gt.values()))
    ordered = sorted(((item['score'], fid, index, item['bbox_xyxy'])
                      for fid, items in predictions.items() for index, item in enumerate(items)),
                     key=lambda item: (-item[0], item[1], item[2]))
    used = defaultdict(set)
    flags, scores, matches = [], [], []
    for score, fid, _, box in ordered:
        choices = [(overlap(box, truth), index) for index, truth in enumerate(gt[fid]) if index not in used[fid]]
        best, index = max(choices, default=(0., -1))
        hit = best >= .5 and index >= 0
        flags.append(int(hit)); scores.append(score)
        if hit:
            used[fid].add(index)
            matches.append((fid, index, score))
    count, precisions = 0, []
    for position, hit in enumerate(flags, 1):
        count += hit
        precisions.append(count / position)
    best_precision, area = 0., 0.
    for hit, precision in zip(reversed(flags), reversed(precisions)):
        best_precision = max(best_precision, precision)
        if hit:
            area += best_precision
    tp_at_threshold = sum(hit for hit, score in zip(flags, scores) if score >= threshold)
    count_at_threshold = sum(score >= threshold for score in scores)
    sizes = {name: {'ground_truth': 0, 'matched_at_score': 0, 'matched_all_saved_scores': 0} for name in BINS}
    for fid, boxes in gt.items():
        for box in boxes:
            sizes[size_bin(frames[fid], box)]['ground_truth'] += 1
    for fid, index, score in matches:
        group = sizes[size_bin(frames[fid], gt[fid][index])]
        group['matched_all_saved_scores'] += 1
        group['matched_at_score'] += score >= threshold
    for row in sizes.values():
        row['recall_at_score'] = row['matched_at_score'] / row['ground_truth'] if row['ground_truth'] else None
        row['recall_all_saved_scores'] = row['matched_all_saved_scores'] / row['ground_truth'] if row['ground_truth'] else None
    return {'ground_truth_boxes': total, 'predictions': len(ordered),
            'ap_at_iou': area / total if total else None, 'tp_at_score': tp_at_threshold,
            'fp_at_score': count_at_threshold - tp_at_threshold,
            'fn_at_score': total - tp_at_threshold,
            'recall_all_saved_scores': count / total if total else None,
            'size_strata': sizes}


def verify_cyclic_control(frames, real, wrong, shift):
    ids = sorted(frames)
    if not shift % len(ids):
        raise ValueError('Wrong-image pairing must be a derangement')
    maximum_error = 0.
    for position, target in enumerate(ids):
        donor = ids[(position + shift) % len(ids)]
        if len(real[donor]) != len(wrong[target]):
            raise ValueError('Wrong-image detection count differs from paired real donor')
        source_frame, target_frame = frames[donor], frames[target]
        for source, destination in zip(real[donor], wrong[target]):
            if source['class'] != destination['class'] or abs(source['score'] - destination['score']) > 1e-12:
                raise ValueError('Wrong-image score/class differs from paired real donor')
            expected = [value * target_frame['width' if i % 2 == 0 else 'height'] /
                        source_frame['width' if i % 2 == 0 else 'height']
                        for i, value in enumerate(source['bbox_xyxy'])]
            error = max(abs(a - b) for a, b in zip(expected, destination['bbox_xyxy']))
            maximum_error = max(maximum_error, error)
            if error > 1e-5:
                raise ValueError('Wrong-image normalized coordinates differ from paired real donor')
    return {'pairing': 'Sorted validation IDs; donor=(target_position+shift) mod N',
            'shift': shift, 'frames': len(ids), 'maximum_coordinate_error': maximum_error,
            'derangement_and_all_saved_rows_verified': True}


def dataset_summary(document):
    output = {}
    for split in ('train', 'validation'):
        frames = [f for f in document['frames'] if f['split'] == split]
        classes = {}
        for cls in CLASSES:
            entries = [(f, b) for f in frames for b in f['boxes'] if b['class'] == cls]
            classes[cls] = {'boxes': len(entries),
                'positive_frames': sum(any(b['class'] == cls for b in f['boxes']) for f in frames),
                'size_counts_384x256_reference': dict(Counter(size_bin(f, b['bbox_xyxy']) for f, b in entries)),
                'occluded_boxes': sum(b['attributes'].get('IsOccluded') == 1 for _, b in entries),
                'truncated_boxes': sum(b['attributes'].get('IsTruncated') == 1 for _, b in entries)}
        output[split] = {'images': len(frames), 'author_groups': len({f['group_id'] for f in frames}), 'classes': classes}
    return output


def check_saved_report(path, metrics, identities, manifest_sha, target_class, threshold):
    raw = path.read_bytes(); report = json.loads(raw)
    if (report['dataset_manifest_sha256'] != manifest_sha or report['target_class'] != target_class or
            report['split'] != 'validation' or report['frames'] != 80 or report['optimizer_steps'] != 0 or
            report['test_inference'] or report['ready_to_resume_s5_training'] or
            not report['all_source_model_manifest_hashes_unchanged']):
        raise ValueError('Saved report scope or identity differs from predeclared validation')
    errors = []
    for mode, values in metrics.items():
        saved = report['metrics'][RUN_MODES[mode]]
        if saved['score_threshold'] != threshold or saved['iou_threshold'] != .5:
            raise ValueError('Saved metric threshold differs')
        for key in ('ground_truth_boxes', 'predictions', 'ap_at_iou', 'tp_at_score', 'fp_at_score', 'fn_at_score'):
            error = abs(values[key] - saved['classes'][target_class][key])
            errors.append(error)
            if error > 1e-12:
                raise ValueError('Independent metric differs from saved report: ' + mode + '/' + key)
        strata = report['size_recall'][RUN_MODES[mode]]['reference_s5_384x256']
        groupings = {'small_lt32': BINS[:2], 'medium_32_to_lt96': BINS[2:3], 'large_ge96': BINS[3:]}
        for label, bins in groupings.items():
            total = sum(values['size_strata'][name]['ground_truth'] for name in bins)
            matched = sum(values['size_strata'][name]['matched_at_score'] for name in bins)
            if total != strata[label]['ground_truth_boxes'] or matched != strata[label]['matched_gt']:
                raise ValueError('Independent size recall differs from saved report')
        relative = 'predictions/' + RUN_MODES[mode] + '.json'
        if report['saved_artifacts'][relative]['sha256'] != identities[mode]['sha256']:
            raise ValueError('Prediction is not the file bound to the saved report')
    source_checks = []
    base = Path(__file__).resolve().parent
    for name, expected in report['source_hashes'].items():
        source = (base / name).resolve()
        if not source.is_relative_to(base) or source.suffix != '.py':
            raise ValueError('Source reference is outside the project Python source scope')
        if checksum(source.read_bytes()) != expected:
            raise ValueError('Saved report source hash no longer matches: ' + name)
        source_checks.append(name)
    result = {'saved_report_sha256': checksum(raw), 'supported_class_metrics_agree': True,
              'maximum_absolute_metric_error': max(errors, default=0.),
              'full_matching_size_strata_agree': True, 'prediction_hashes_match_report': True,
              'recorded_project_sources_match_current': source_checks,
              'all_four_controls_recomputed': set(metrics) == set(RUN_MODES)}
    if set(metrics) == set(RUN_MODES):
        plan_path = path.parent / 'frozen-plan.json'
        plan_raw = plan_path.read_bytes()
        if checksum(plan_raw) != report['frozen_plan_sha256']:
            raise ValueError('Frozen plan SHA differs from report')
        plan = json.loads(plan_raw)
        rules = plan['predeclared_necessary_screen']
        real_ap = metrics['real']['ap_at_iou']
        passed = real_ap >= rules['person_real_ap50_min' if target_class == 'person' else 'face_real_ap50_min']
        passed = passed and all(real_ap - item['ap_at_iou'] >= rules['real_ap50_margin_over_every_control_min'] - 1e-12
                                for mode, item in metrics.items() if mode != 'real')
        if target_class == 'person_face':
            small = metrics['real']['size_strata']
            total = sum(small[name]['ground_truth'] for name in BINS[:2])
            matched = sum(small[name]['matched_at_score'] for name in BINS[:2])
            passed = passed and total > 0 and matched / total >= rules['small_face_recall_min'] and matched >= rules['small_face_matched_gt_min']
        if passed != report['ready_for_external_baseline_use']:
            raise ValueError('Independent external-only screen disagrees with report')
        result['external_only_screen_recomputed_passed'] = passed
    return result


def self_check():
    frame = {'width': 384, 'height': 256, 'boxes': [
        {'class': 'person', 'bbox_xyxy': [0, 0, 10, 10]},
        {'class': 'person', 'bbox_xyxy': [20, 0, 30, 10]}]}
    boxes = [[100, 100, 110, 110], [0, 0, 10, 10], [0, 0, 10, 10], [20, 0, 30, 10]]
    predictions = {'a': [{'score': score, 'bbox_xyxy': box} for score, box in zip((.9, .8, .7, .6), boxes)]}
    m = independent_metrics({'a': frame}, predictions, 'person', .75)
    assert m['ap_at_iou'] == .5 and (m['tp_at_score'], m['fp_at_score'], m['fn_at_score']) == (1, 1, 1)
    assert m['size_strata']['below16']['matched_all_saved_scores'] == 2
    empty = independent_metrics({'a': frame}, {'a': []}, 'person')
    assert empty['ap_at_iou'] == 0 and empty['fn_at_score'] == 2
    assert abs(overlap([0, 0, 10, 10], [5, 0, 15, 10]) - 1/3) < 1e-15
    assert size_bin(frame, [0, 0, 32, 32]) == '32_to_below96'
    try:
        check_box([0, 0, 400, 10], frame)
    except ValueError:
        pass
    else:
        raise AssertionError('Out-of-bounds box guard failed')
    return 6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--class', dest='target_class', choices=CLASSES)
    for mode in ('real', 'gray', 'wrong'):
        parser.add_argument('--' + mode, type=Path)
    parser.add_argument('--mean', type=Path, help='Optional same-image RGB-channel-mean control')
    parser.add_argument('--saved-report', type=Path, help='Compare independently recomputed metrics/source hashes against runner report')
    parser.add_argument('--score-threshold', type=float, default=.5)
    parser.add_argument('--wrong-shift', type=int, default=1)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    checks = self_check()
    if args.self_check:
        print(json.dumps({'independent_metric_and_guard_checks': checks, 'passed': True}))
        return
    if not all((args.dataset, args.target_class, args.real, args.gray, args.wrong)):
        parser.error('--dataset, --class, --real, --gray and --wrong are required')
    if not 0 <= args.score_threshold <= 1:
        parser.error('Score threshold must be in [0,1]')
    manifest = args.dataset.read_bytes(); manifest_sha = checksum(manifest)
    document = json.loads(manifest)
    protocol = dataset_summary(document)
    frames = {f['id']: f for f in document['frames'] if f['split'] == 'validation'}
    del document  # No test row reaches a prediction check or metric.
    if len(frames) != 80:
        raise ValueError('Expected the predeclared 80-image validation set')
    predictions, identities = {}, {}
    modes = ('real', 'mean', 'gray', 'wrong') if args.mean else ('real', 'gray', 'wrong')
    for mode in modes:
        predictions[mode], identities[mode] = read_predictions(getattr(args, mode), frames, manifest_sha, args.target_class)
    pairing = verify_cyclic_control(frames, predictions['real'], predictions['wrong'], args.wrong_shift)
    metrics = {mode: independent_metrics(frames, value, args.target_class, args.score_threshold)
               for mode, value in predictions.items()}
    saved_comparison = (check_saved_report(args.saved_report, metrics, identities, manifest_sha,
                                         args.target_class, args.score_threshold) if args.saved_report else
                        {'status': 'No saved runner report supplied; model/source binding not independently checked'})
    if checksum(args.dataset.read_bytes()) != manifest_sha:
        raise ValueError('Dataset manifest changed during verification')
    for mode, identity in identities.items():
        if checksum(getattr(args, mode).read_bytes()) != identity['sha256']:
            raise ValueError('Predictions changed during verification')
    report = {'saved_prediction_verification_passed': True, 'supported_class': args.target_class,
        'source_script_sha256': checksum(Path(__file__).read_bytes()), 'dataset_manifest_sha256': manifest_sha,
        'prediction_file_identities': identities, 'dataset_protocol': protocol,
        'saved_report_comparison': saved_comparison,
        'metric': 'Independent all-points interpolated AP at IoU0.5; one supported class, not COCO/OI/WIDER official AP.',
        'score_threshold': args.score_threshold, 'metrics': metrics, 'wrong_image_control': pairing,
        'real_minus_strongest_control_ap50': metrics['real']['ap_at_iou'] - max(
            values['ap_at_iou'] for mode, values in metrics.items() if mode != 'real'),
        'independent_self_checks': checks, 'model_inference': False, 'test_inference': False, 'optimizer_steps': 0,
        'original_network_fix_verified': False, 's5_network_ready_to_train': False,
        'on_device_compatibility_verified': False,
        'limits': ['No model identity, preprocessing, tensor-to-box conversion or actual gray-image generation is proved by saved prediction checks alone.',
                   'Positive-only validation does not measure empty-scene false alarms or negative-image specificity.',
                   'AP/recall use all saved detections; native model confidence floors and top-K may truncate attainable recall.',
                   'Size bins use a fixed384x256 reference, not actual native/320/640 model input pixel size.',
                   'No threshold selection, bootstrap, independent final-test inference, training or S5 deployment decision is performed.']}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
