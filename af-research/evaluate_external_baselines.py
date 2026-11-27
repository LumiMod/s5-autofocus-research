#!/usr/bin/env python3
"""Frozen external detector validation with image-independent controls.

No optimizer, firmware network, or test split is used. Model adapters return
original-image continuous xyxy boxes. AP uses every returned target-class box,
subject to the official export's own score floor and NMS, if any.
"""
import argparse
from collections import Counter
import copy
import importlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from pretraining_data import evaluate_detection, iou, need, sha256, validate_box

BASE = Path(__file__).resolve().parent
TARGETS = {'picodet': 'person', 'blazeface': 'person_face', 'blazeface_s5_size': 'person_face'}
CONTROLS = ('real', 'same_image_RGB_channel_mean', 'fixed_gray117', 'cyclic_wrong')
REFERENCE_V2_SHA = 'af5459ddebd98947b154580cd176e7846abcceffe5539ac6f0001f6a485a3fc8'


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


def restore_model_xyxy(box, original_wh, resized_wh, padding_xy=(0, 0)):
    """Invert an explicitly supplied resize/pad affine map; never infer policy.

    Adapters may reuse this helper; a helper self-check does not validate an
    adapter's own coordinate restoration. resized_wh excludes any padding.
    """
    ow, oh = original_wh
    rw, rh = resized_wh
    px, py = padding_xy
    need(min(ow, oh, rw, rh) > 0, 'Invalid resize dimensions')
    values = np.asarray(box, dtype=np.float64)
    need(values.shape == (4,) and np.isfinite(values).all(), 'Invalid model box')
    restored = (values-np.array([px, py, px, py]))/np.array([rw/ow, rh/oh]*2)
    restored[[0, 2]] = restored[[0, 2]].clip(0, ow)
    restored[[1, 3]] = restored[[1, 3]].clip(0, oh)
    return restored.tolist()


def map_box_between_images(box, source_wh, target_wh):
    sw, sh = source_wh
    tw, th = target_wh
    need(min(sw, sh, tw, th) > 0, 'Invalid wrong-image dimensions')
    box = np.asarray(box, dtype=np.float64)
    need(box.shape == (4,) and np.isfinite(box).all(), 'Invalid wrong-image box')
    return (box*np.array([tw/sw, th/sh]*2)).tolist()


def map_predictions_between_images(detections, source_wh, target_wh):
    return [{**row, 'bbox_xyxy': map_box_between_images(row['bbox_xyxy'], source_wh, target_wh)}
            for row in detections]


def image_control(rgb, mode):
    need(isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3,
         'Adapter input must be uint8 HWC RGB')
    if mode == 'real':
        return rgb.copy()
    if mode == 'same_image_RGB_channel_mean':
        means = np.rint(rgb.mean(axis=(0, 1))).clip(0, 255).astype(np.uint8)
        return np.broadcast_to(means, rgb.shape).copy()
    if mode == 'fixed_gray117':
        return np.full_like(rgb, 117)
    raise ValueError('Wrong-image control reuses real predictions; it is not another inference mode')


def canonical_detections(items, width, height, target_class):
    need(isinstance(items, list), 'predict_rgb must return a list')
    result, raw, counts = [], [], Counter()
    for item in items:
        item = plain(item)
        need(isinstance(item, dict) and isinstance(item.get('class'), str), 'Detection class must be a string')
        score = item.get('score')
        need(isinstance(score, (int, float)) and math.isfinite(score) and 0 <= score <= 1, 'Invalid detector score')
        box = item.get('bbox_xyxy')
        need(isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in box),
             'Invalid detector box')
        row = {'class': item['class'], 'score': float(score), 'bbox_xyxy': list(map(float, box))}
        raw.append(row)
        if row['class'] != target_class:
            counts['other_class_filtered'] += 1
            continue
        clipped = np.asarray(box, dtype=float)
        clipped[[0, 2]] = clipped[[0, 2]].clip(0, width)
        clipped[[1, 3]] = clipped[[1, 3]].clip(0, height)
        if not np.array_equal(clipped, box):
            counts['boxes_clipped_to_original_image'] += 1
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            counts['degenerate_target_boxes_removed'] += 1
            continue
        row = {**row, 'bbox_xyxy': clipped.tolist()}
        validate_box(row['bbox_xyxy'], width, height)
        result.append(row)
    counts.update(adapter_returned=len(items), retained_target=len(result))
    return result, raw, dict(counts)


def reference_sizes(box, frame, metadata):
    width, height = frame['width'], frame['height']
    area = max(0, box[2]-box[0])*max(0, box[3]-box[1])
    sizes = {'reference_s5_384x256': math.sqrt(area*384*256/(width*height)),
             'reference_stretch_640x640': math.sqrt(area*640*640/(width*height))}
    transform = metadata.get('input_transform', {})
    mh, mw = transform.get('height'), transform.get('width')
    if transform.get('resize_policy') == 'native':
        sizes['actual_native'] = math.sqrt(area)
    elif transform.get('keep_ratio') is False and isinstance(mh, (int, float)) and isinstance(mw, (int, float)) and min(mh, mw) > 0:
        sizes[f'actual_stretch_{int(mw)}x{int(mh)}'] = math.sqrt(area*mw*mh/(width*height))
    return sizes


def size_label(sqrt_area):
    return 'small_lt32' if sqrt_area < 32 else 'medium_32_to_lt96' if sqrt_area < 96 else 'large_ge96'


def match_and_stratify(frames, prediction, target_class, metadata, score_threshold=.5, iou_threshold=.5):
    """Match all target-class GT first, then stratify matched GT by size.

    One-to-one greedy matching and tie ordering mirror evaluate_detection.
    Unmatched prediction boxes are not assigned to a GT-size bucket; no small AP
    is manufactured by dropping large GT or counting their boxes as false alarms.
    """
    frame_map = {f['id']: f for f in frames}
    truth = {f['id']: [(i, b) for i, b in enumerate(f['boxes']) if b['class'] == target_class] for f in frames}
    rows = prediction['frames']
    need(len(rows) == len(frames) and {r['id'] for r in rows} == set(frame_map), 'Prediction coverage differs')
    seen = {fid: set() for fid in frame_map}
    assignments, fp = {}, 0
    ranked = sorted(((d['score'], r['id'], i, d) for r in rows for i, d in enumerate(r['detections'])
                     if d['class'] == target_class and d['score'] >= score_threshold),
                    key=lambda x: (-x[0], x[1], x[2]))
    for score, fid, pred_index, detection in ranked:
        candidates = [(iou(detection['bbox_xyxy'], box['bbox_xyxy']), local_index)
                      for local_index, (_original_index, box) in enumerate(truth[fid]) if local_index not in seen[fid]]
        overlap, local_index = max(candidates, default=(0., -1))
        if local_index >= 0 and overlap >= iou_threshold:
            seen[fid].add(local_index)
            assignments[(fid, local_index)] = {'prediction_index': pred_index, 'score': float(score), 'iou': overlap}
        else:
            fp += 1
    details, summaries = [], {}
    for frame in frames:
        fid = frame['id']
        for local_index, (original_index, box) in enumerate(truth[fid]):
            matched = (fid, local_index) in assignments
            sizes = reference_sizes(box['bbox_xyxy'], frame, metadata)
            detail = {'frame_id': fid, 'gt_index': original_index, 'class': target_class,
                      'bbox_xyxy': box['bbox_xyxy'], 'matched': matched,
                      'sqrt_area': sizes, 'match': assignments.get((fid, local_index))}
            details.append(detail)
            for scheme, value in sizes.items():
                scheme_rows = summaries.setdefault(scheme, {})
                for bucket in ('all', size_label(value)):
                    counter = scheme_rows.setdefault(bucket, {'ground_truth_boxes': 0, 'matched_gt': 0})
                    counter['ground_truth_boxes'] += 1
                    counter['matched_gt'] += int(matched)
    for scheme in summaries.values():
        for bucket in ('all', 'small_lt32', 'medium_32_to_lt96', 'large_ge96'):
            row = scheme.setdefault(bucket, {'ground_truth_boxes': 0, 'matched_gt': 0})
            row['recall'] = row['matched_gt']/row['ground_truth_boxes'] if row['ground_truth_boxes'] else None
    return {'target_class': target_class, 'score_threshold': score_threshold, 'iou_threshold': iou_threshold,
            'matching': 'All target-class GT, descending score, one-to-one; only afterwards bucket matched GT by size.',
            'tp': len(assignments), 'fp': fp, 'fn': len(details)-len(assignments),
            'size_recall': summaries, 'per_gt_matches': details,
            'limits': 'Size-stratified recall only, not small-object AP. Reference640 is explicitly not the actual model scale unless separately named actual.'}


def screening(metrics, strata, target_class):
    aps = {mode: value['classes'][target_class]['ap_at_iou'] for mode, value in metrics.items()}
    real = aps['real']
    checks = {'real_ap50_ge_0_5': real is not None and real >= .5}
    margins = {}
    for mode in CONTROLS[1:]:
        margins[mode] = None if real is None or aps[mode] is None else real-aps[mode]
        checks['real_minus_'+mode+'_ge_0_15'] = margins[mode] is not None and margins[mode] >= .15-1e-12
    small = strata['real']['size_recall']['reference_s5_384x256']['small_lt32']
    if target_class == 'person_face':
        checks['small_face_recall_ge_0_5'] = small['recall'] is not None and small['recall'] >= .5
        checks['small_face_matched_gt_ge_10'] = small['matched_gt'] >= 10
    return {'scope': 'Individual external model on the frozen validation set; combine neither classes nor models implicitly.',
            'target_class': target_class, 'real_ap50': real, 'control_ap50': {k: aps[k] for k in CONTROLS[1:]},
            'real_minus_control_ap50': margins, 'small_s5_reference': small,
            'small_face_checks_applicable': target_class == 'person_face', 'checks': checks,
            'passed_necessary_external_baseline_screen': all(checks.values()),
            'ready_for_external_baseline_use': all(checks.values()),
            'ready_to_resume_s5_training': False,
            'original_s5_equivalence_established': False, 's5_training_resume_conclusion': 'Not established by an external detector baseline.',
            'limits': 'Predeclared necessary screen, not camera-domain performance, formal model validation, or original firmware equivalence.'}


def selected_dataset(manifest_path, expected_images, expected_hash=None):
    if expected_hash:
        need(sha256(manifest_path) == expected_hash, 'Manifest differs from requested frozen dataset')
    data = json.loads(manifest_path.read_text())
    need(data.get('task') == 'recognition_detection' and data.get('schema_version') == 1, 'Unsupported dataset')
    need(set(data['classes']) == {'person', 'person_face'}, 'Expected person/face dataset')
    frames = sorted((f for f in data['frames'] if f['split'] == 'validation'), key=lambda f: f['id'])
    need(len(frames) == expected_images and len({f['id'] for f in frames}) == len(frames), 'Wrong validation count or duplicate IDs')
    need(all(isinstance(f['width'], int) and isinstance(f['height'], int) and min(f['width'], f['height']) > 0 for f in frames),
         'Invalid image dimensions')
    for frame in frames:
        for box in frame['boxes']:
            need(box['class'] in data['classes'], 'Unknown GT class')
            validate_box(box['bbox_xyxy'], frame['width'], frame['height'])
    return data, frames


def model_inventory(model_dir):
    root = model_dir.resolve()
    need(root.is_dir(), 'Model directory does not exist')
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_file():
            need(path.resolve().is_relative_to(root), 'Model file symlink escapes model directory')
            result[path.relative_to(root).as_posix()] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
    need(result, 'Model directory is empty')
    return result


def loaded_project_sources(adapter_module):
    paths = {Path(__file__).resolve(), BASE/'pretraining_data.py', Path(adapter_module.__file__).resolve()}
    for module in tuple(sys.modules.values()):
        name = getattr(module, '__file__', None)
        if name:
            path = Path(name).resolve()
            if path.suffix == '.py' and path.is_relative_to(BASE):
                directories = path.relative_to(BASE).parts[:-1]
                # Runtime packages can live under this checkout. Their versions
                # are recorded separately; do not duplicate thousands of package
                # hashes in every prediction alongside project-owned sources.
                dependency_directory = any(
                    part.startswith('.venv') or part in
                    {'venv', 'site-packages', 'dist-packages', '.tox', '.nox'}
                    for part in directories)
                if not dependency_directory:
                    paths.add(path)
    return {str(p.relative_to(BASE)) if p.is_relative_to(BASE) else str(p): sha256(p) for p in sorted(paths)}


def verify_source_hashes(values):
    return all(sha256(Path(name) if Path(name).is_absolute() else BASE/name) == value for name, value in values.items())


def run(args):
    need(args.model in TARGETS, 'Unknown model')
    target_class = TARGETS[args.model]
    output = args.output or BASE/'artifacts/external-baselines'/args.model
    need(not output.exists() or not any(output.iterdir()), 'Output directory is not empty; use a new per-model run directory')
    model_dir = args.model_dir.resolve()
    need(not output.resolve().is_relative_to(model_dir), 'Output must be outside model directory')
    data, frames = selected_dataset(args.dataset, args.expected_images, args.expected_manifest_sha)
    inventory = model_inventory(model_dir)
    module_name, separator, factory_name = args.adapter.partition(':')
    need(separator and module_name and factory_name, 'Adapter must be module:factory')
    module = importlib.import_module(module_name)
    detector = getattr(module, factory_name)(args.model, model_dir)
    need(callable(getattr(detector, 'predict_rgb', None)), 'Missing predict_rgb adapter method')
    metadata = plain(copy.deepcopy(detector.metadata))
    need(isinstance(metadata, dict) and metadata.get('model_id'), 'Adapter metadata must identify model')
    need(metadata.get('output_coordinate_space') in ('original_image_xyxy', 'original_image_continuous_xyxy', 'original_image'),
         'Adapter must explicitly identify original-image xyxy output coordinates')
    need(metadata.get('model_file_hashes'), 'Adapter must provide model_file_hashes')
    for name, checksum in metadata['model_file_hashes'].items():
        expected = checksum.get('sha256') if isinstance(checksum, dict) else checksum
        need(name in inventory and inventory[name]['sha256'] == expected, 'Adapter model hash differs from actual file: '+name)
    sources = loaded_project_sources(module)
    manifest_hash = sha256(args.dataset)
    # No detector.predict_rgb call occurs before this frozen plan is written.
    gt_counts = dict(Counter(b['class'] for f in frames for b in f['boxes']))
    gt_size_counts = {}
    for frame in frames:
        for box in frame['boxes']:
            if box['class'] == target_class:
                for scheme, size in reference_sizes(box['bbox_xyxy'], frame, metadata).items():
                    counter = gt_size_counts.setdefault(scheme, Counter())
                    counter['all'] += 1
                    counter[size_label(size)] += 1
    plan = {'schema_version': 1, 'model': args.model, 'target_class': target_class, 'adapter': args.adapter,
            'model_metadata': metadata, 'model_files': inventory, 'source_hashes': sources,
            'dataset_manifest_sha256': manifest_hash, 'split': 'validation',
            'frame_ids': [f['id'] for f in frames], 'frames': len(frames),
            'ground_truth_counts': gt_counts, 'target_ground_truth_size_counts': {k: dict(v) for k, v in gt_size_counts.items()},
            'image_files': [{'id': f['id'], 'path': f['path'], 'sha256': f['sha256'], 'width': f['width'], 'height': f['height']} for f in frames],
            'controls': list(CONTROLS), 'wrong_image_policy': 'Sort validation IDs; target i uses source (i+1)%N real detections, normalized by source width/height and mapped to target width/height.',
            'same_mean_policy': 'Per RGB channel arithmetic mean, numpy.rint to uint8; uniform fill at original dimensions.',
            'pixel_policy': 'Stored pixels RGB; no EXIF rotation or label transform beyond adapter-defined resize.',
            'evaluation': {'ap': 'Unmodified pretraining_data.evaluate_detection; AP50 per target class over all returned target detections.',
                           'score_threshold_for_recall_only': .5, 'iou_threshold': .5,
                           'small_primary': 'sqrt(area after reference stretch to384x256)<32',
                           'extra_size_scheme': 'Reference stretch640x640; actual stretch only for explicit keep_ratio=false, actual_native only for explicit resize_policy=native.',
                           'matching': 'Match all target-class GT first, one-to-one; then count recall by matched-GT size. No smallAP.'},
            'predeclared_necessary_screen': {'person_real_ap50_min': .5, 'face_real_ap50_min': .5,
                'real_ap50_margin_over_every_control_min': .15, 'small_face_recall_min': .5,
                'small_face_matched_gt_min': 10, 'scope': 'Apply person/face checks in their respective independent runs; small-face checks apply only to face.'},
            'optimizer_steps': 0, 'test_inference': False,
            'ready_to_resume_s5_training': False,
            'baseline_role': 'External reference only. Cannot confirm execution on S5; passing never unlocks S5 training or substitutes for the original network.',
            'limits': ['Official/exported-model score floors, topK and NMS remain in effect; AP does not cover omitted low-score detections.',
                       'No additional score cutoff is applied by the evaluator before AP.',
                       'Passing this necessary screen does not validate S5 arithmetic, deployment, low-light AF or camera-domain accuracy.',
                       'Native-resolution BlazeFace and the explicitly resized S5-size profile are separate runs, not equal-compute comparisons.',
                       'Positive-only person+face validation lacks empty-background false-alarm evaluation.',
                       'This validation set has been used in earlier research diagnostics; it is not an untouched test set.']}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/'frozen-plan.json', plan)
    started = time.monotonic()
    predictions = {mode: {'dataset_manifest_sha256': manifest_hash, 'model': args.model,
                         'model_metadata': metadata, 'source_hashes': sources, 'control': mode, 'frames': []} for mode in CONTROLS}
    raw_predictions = {mode: {'dataset_manifest_sha256': manifest_hash, 'model': args.model,
                             'model_metadata': metadata, 'source_hashes': sources, 'control': mode,
                             'coordinate_status': 'Adapter-returned original-image coordinates before evaluator clipping/filtering.',
                             'frames': []} for mode in CONTROLS}
    frame_records, counts = [], {mode: Counter() for mode in CONTROLS}
    for index, frame in enumerate(frames):
        path = (args.dataset.parent/frame['path']).resolve()
        need(path.is_relative_to(args.dataset.parent.resolve()), 'Image path escapes dataset root')
        need(sha256(path) == frame['sha256'], 'Image bytes differ from frozen manifest')
        with Image.open(path) as image:
            need(image.size == (frame['width'], frame['height']), 'Image dimensions differ from manifest')
            rgb = np.asarray(image.convert('RGB'), dtype=np.uint8).copy()
        frame_records.append({'id': frame['id'], 'width': frame['width'], 'height': frame['height'],
                              'RGB_channel_mean': rgb.mean(axis=(0, 1)).tolist(),
                              'rounded_control_RGB': np.rint(rgb.mean(axis=(0, 1))).astype(np.uint8).tolist()})
        for mode in CONTROLS[:3]:
            items = detector.predict_rgb(image_control(rgb, mode))
            filtered, raw, stats = canonical_detections(items, frame['width'], frame['height'], target_class)
            predictions[mode]['frames'].append({'id': frame['id'], 'detections': filtered})
            raw_predictions[mode]['frames'].append({'id': frame['id'], 'detections': raw})
            counts[mode].update(stats)
        if (index+1) % 10 == 0:
            print(json.dumps({'phase': 'validation_inference', 'model': args.model, 'images': index+1, 'total': len(frames)}), flush=True)
    for i, target in enumerate(frames):
        j = (i+1) % len(frames)
        source = frames[j]
        source_wh, target_wh = (source['width'], source['height']), (target['width'], target['height'])
        detections = map_predictions_between_images(predictions['real']['frames'][j]['detections'], source_wh, target_wh)
        detections, _mapped_raw, wrong_stats = canonical_detections(detections, target['width'], target['height'], target_class)
        predictions['cyclic_wrong']['frames'].append({'id': target['id'], 'prediction_source_id': source['id'], 'detections': detections})
        raw = map_predictions_between_images(raw_predictions['real']['frames'][j]['detections'], source_wh, target_wh)
        raw_predictions['cyclic_wrong']['frames'].append({'id': target['id'], 'prediction_source_id': source['id'], 'detections': raw})
        counts['cyclic_wrong'].update(wrong_stats)
    need(sha256(args.dataset) == manifest_hash, 'Manifest changed during run')
    need(verify_source_hashes(sources), 'Evaluation/adapter source changed during run')
    need(model_inventory(model_dir) == inventory, 'Model directory changed during run')
    selected = {'classes': [target_class], 'frames': frames}
    metrics, strata = {}, {}
    for mode in CONTROLS:
        metrics[mode] = evaluate_detection(selected, predictions[mode], 'validation', threshold=.5, score_threshold=.5)
        strata[mode] = match_and_stratify(frames, predictions[mode], target_class, metadata)
        class_metrics = metrics[mode]['classes'][target_class]
        need(strata[mode]['tp'] == class_metrics['tp_at_score'] and strata[mode]['fp'] == class_metrics['fp_at_score'] and
             strata[mode]['fn'] == class_metrics['fn_at_score'], 'Stratified matching disagrees with whole-class metric')
        write_json(output/'predictions'/f'{mode}.json', predictions[mode])
        write_json(output/'raw-adapter-predictions'/f'{mode}.json', raw_predictions[mode])
        write_json(output/'matching'/f'{mode}.json', strata[mode])
    acceptance = screening(metrics, strata, target_class)
    write_json(output/'metrics.json', metrics)
    write_json(output/'frame-control-values.json', frame_records)
    write_json(output/'screening.json', acceptance)
    report = {'status': 'External baseline validation completed; no original S5 equivalence claim.',
              'model': args.model, 'target_class': target_class, 'model_metadata': metadata,
              'dataset_manifest_sha256': manifest_hash, 'source_hashes': sources, 'model_files': inventory,
              'frames': len(frames), 'split': 'validation', 'model_predict_calls': len(frames)*3,
              'optimizer_steps': 0, 'test_inference': False, 'metrics': metrics,
              'ready_for_external_baseline_use': acceptance['ready_for_external_baseline_use'],
              'ready_to_resume_s5_training': False, 'baseline_role': plan['baseline_role'],
              'size_recall': {mode: item['size_recall'] for mode, item in strata.items()},
              'screening': acceptance, 'filtering_counts': {k: dict(v) for k, v in counts.items()},
              'frozen_plan_sha256': sha256(output/'frozen-plan.json'),
              'all_source_model_manifest_hashes_unchanged': True,
              'elapsed_seconds': time.monotonic()-started, 'limits': plan['limits']}
    report['saved_artifacts'] = {str(p.relative_to(output)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
                                 for p in sorted(output.rglob('*')) if p.is_file()}
    write_json(output/'report.json', report)
    print(json.dumps({'model': args.model, 'report': str(output/'report.json'), 'screening': acceptance}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=TARGETS, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--adapter', default='external_baseline_models:load_detector')
    parser.add_argument('--dataset', type=Path, default=BASE/'private-data/recognition-v2/manifest.json')
    parser.add_argument('--expected-manifest-sha', default=REFERENCE_V2_SHA)
    parser.add_argument('--expected-images', type=int, default=80)
    parser.add_argument('--output', type=Path)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
