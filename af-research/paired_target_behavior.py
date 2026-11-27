#!/usr/bin/env python3
"""Train-only, paired target-pixel/translation diagnostics; never training.

Preparation needs only Pillow/NumPy. Inference is an explicitly separate command
using an adapter in its own Torch/Paddle environment. No original inference or
previous validation-evaluation source is modified or imported by this module.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from pretraining_data import iou, need, sha256, validate_box

BASE = Path(__file__).resolve().parent
CLASSES = ('person', 'person_face')
MANIFEST_SHA = 'af5459ddebd98947b154580cd176e7846abcceffe5539ac6f0001f6a485a3fc8'
DEFAULT_OUTPUT = BASE/'artifacts/recognition-diagnosis-v3/paired-behavior'
SCORE = .5
IOU = .5
FILL = 117


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


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


def pixel_sha(rgb):
    height, width = rgb.shape[:2]
    return hashlib.sha256(f'{width}x{height}:RGB:'.encode()+rgb.tobytes()).hexdigest()


def load_rgb(root, frame):
    path = (root/frame['path']).resolve()
    need(path.is_relative_to(root.resolve()), 'Photo path escapes dataset root')
    need(sha256(path) == frame['sha256'], 'Photo file SHA changed: '+frame['id'])
    with Image.open(path) as image:
        need(image.size == (frame['width'], frame['height']), 'Photo dimensions differ')
        rgb = np.asarray(image.convert('RGB'), dtype=np.uint8).copy()
    need(pixel_sha(rgb) == frame['pixel_sha256'], 'Decoded RGB SHA differs')
    return rgb


def integer_rect(box, width, height):
    """Pixel mask encloses a continuous xyxy GT box; half-open integer bounds."""
    validate_box(box, width, height)
    return [max(0, math.floor(box[0])), max(0, math.floor(box[1])),
            min(width, math.ceil(box[2])), min(height, math.ceil(box[3]))]


def annotation_occupancy(frame):
    mask = np.zeros((frame['height'], frame['width']), dtype=np.uint8)
    for box in frame['boxes']:
        x1, y1, x2, y2 = integer_rect(box['bbox_xyxy'], frame['width'], frame['height'])
        mask[y1:y2, x1:x2] = 1
    return mask


def first_free_rectangle(mask, width, height):
    """Exact lexicographic (y,x) search for a same-shape annotation-free patch."""
    h, w = mask.shape
    if not (0 < width <= w and 0 < height <= h):
        return None
    integral = np.pad(mask.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    for y in range(h-height+1):
        sums = (integral[y+height, width:]-integral[y, width:]
                -integral[y+height, :-width]+integral[y, :-width])
        free = np.flatnonzero(sums == 0)
        if len(free):
            x = int(free[0])
            return [x, y, x+width, y+height]
    return None


def occlusion_pair(frame, target_class):
    mask = annotation_occupancy(frame)
    candidates = sorted(((tuple(b['bbox_xyxy']), i) for i, b in enumerate(frame['boxes'])
                         if b['class'] == target_class))
    for _coordinates, index in candidates:
        target = integer_rect(frame['boxes'][index]['bbox_xyxy'], frame['width'], frame['height'])
        x1, y1, x2, y2 = target
        background = first_free_rectangle(mask, x2-x1, y2-y1)
        if background is not None:
            return {'target_class': target_class, 'gt_index': index, 'target_rect': target,
                    'annotation_free_rect': background, 'mask_pixels': (x2-x1)*(y2-y1)}
    return None


def variant_specs(frame):
    width, height = frame['width'], frame['height']
    sx, sy = max(1, width//8), max(1, height//8)
    variants = [{'name': 'real', 'op': 'identity'}]
    for name, dx, dy in [('right', sx, 0), ('left', -sx, 0), ('down', 0, sy), ('up', 0, -sy)]:
        variants.append({'name': 'translate_'+name, 'op': 'translate_crop', 'dx': dx, 'dy': dy, 'fill_rgb': [FILL]*3})
    pairs = {}
    for target_class in CLASSES:
        pair = occlusion_pair(frame, target_class)
        pairs[target_class] = pair
        if pair:
            for kind, key in [('target', 'target_rect'), ('background', 'annotation_free_rect')]:
                variants.append({'name': f'occlude_{target_class}_{kind}', 'op': 'mask_rect',
                                 'rect': pair[key], 'fill_rgb': [FILL]*3,
                                 'target_class': target_class, 'target_gt_index': pair['gt_index']})
    return variants, pairs


def transform_rgb(rgb, spec):
    need(rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3, 'Expected uint8 HWC RGB')
    h, w = rgb.shape[:2]
    op = spec['op']
    if op == 'identity':
        return rgb.copy()
    result = rgb.copy()
    fill = np.asarray(spec['fill_rgb'], dtype=np.uint8)
    need(fill.shape == (3,), 'Invalid RGB fill')
    if op == 'mask_rect':
        x1, y1, x2, y2 = spec['rect']
        need(all(isinstance(v, int) for v in spec['rect']) and 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h,
             'Invalid integer mask rectangle')
        result[y1:y2, x1:x2] = fill
    elif op == 'translate_crop':
        dx, dy = spec['dx'], spec['dy']
        need(isinstance(dx, int) and isinstance(dy, int), 'Translation must use integer pixels')
        result[:] = fill
        x1, y1, x2, y2 = max(0, dx), max(0, dy), min(w, w+dx), min(h, h+dy)
        if x2 > x1 and y2 > y1:
            result[y1:y2, x1:x2] = rgb[y1-dy:y2-dy, x1-dx:x2-dx]
    else:
        raise ValueError('Unknown image transform: '+op)
    return result


def mapped_truth(frame, spec):
    """Include clipped distractor GT in matching, tag only fully retained GT."""
    width, height = frame['width'], frame['height']
    dx = spec.get('dx', 0) if spec['op'] == 'translate_crop' else 0
    dy = spec.get('dy', 0) if spec['op'] == 'translate_crop' else 0
    result = []
    for index, item in enumerate(frame['boxes']):
        box = np.array(item['bbox_xyxy'], dtype=float)+[dx, dy, dx, dy]
        fully_retained = bool(0 <= box[0] and box[2] <= width and 0 <= box[1] and box[3] <= height)
        box[[0, 2]] = box[[0, 2]].clip(0, width)
        box[[1, 3]] = box[[1, 3]].clip(0, height)
        if box[2] > box[0] and box[3] > box[1]:
            result.append({'gt_index': index, 'class': item['class'], 'bbox_xyxy': box.tolist(),
                           'fully_retained': fully_retained})
    return result


def prepare(args):
    need(not args.output.exists() or not any(args.output.iterdir()), 'Preparation output must be empty')
    need(sha256(args.dataset) == args.expected_manifest_sha, 'Dataset manifest SHA differs')
    data = json.loads(args.dataset.read_text())
    need(data.get('task') == 'recognition_detection' and set(data['classes']) == set(CLASSES), 'Unexpected dataset')
    # Only internal split=train is used. IDs may say oi-validation because that
    # is their upstream Open Images partition, not this project's split.
    frames = sorted((f for f in data['frames'] if f['split'] == 'train'), key=lambda f: f['id'])[:16]
    need(len(frames) == 16 and len({f['id'] for f in frames}) == 16, 'Need16 distinct training frames')
    records = []
    for frame in frames:
        rgb = load_rgb(args.dataset.parent, frame)
        specs, pairs = variant_specs(frame)
        variants = [{**spec, 'pixel_sha256': pixel_sha(transform_rgb(rgb, spec))} for spec in specs]
        records.append({'frame': frame, 'variants': variants, 'occlusion_pairs': pairs})
    plan = {'schema_version': 1, 'experiment': 'paired_target_behavior_v1',
            'frozen_profiles': ['s5_baseline_o4_i4_hw', 'picodet', 'blazeface', 'blazeface_s5_size', 's5_flattened_k4'],
            'fifth_profile_basis': 'Flattened K4 layout independently proposed from cross-layer zero-channel structure and separately diagnosed on train32; added before any paired-behavior inference, not chosen from paired results.',
            'dataset_manifest': str(args.dataset.resolve()), 'dataset_manifest_sha256': sha256(args.dataset),
            'split': 'train', 'selection_rule': 'First16 internal train frames lexically sorted by ID; no prediction-based selection.',
            'translation_rule': 'Four integer translations: dx=+-max(1,width//8), dy=+-max(1,height//8), one axis at a time; crop outside, RGB117 fill, no interpolation/wrap.',
            'occlusion_rule': 'For each frame/class, sort GT by (xyxy, original index); choose first with an exact same-shape annotation-free rectangle, lexicographic (y,x); exclude all annotated person/face boxes. Otherwise unavailable. Both masks RGB117.',
            'evaluation': {'score_threshold': SCORE, 'iou_threshold': IOU,
                'translation_eligibility': 'Only fully retained original GT; all visible clipped GT still participate in matching.',
                'matching': 'Class-wise descending score greedy one-to-one IoU>=.5, score>=.5, tie by original prediction index then lowest GT index.',
                'following': 'Conditional on original matched GT: shifted GT matched and prediction-center displacement residual <=0.5*translation magnitude.',
                'occlusion': 'Fixed GT local support=max returned score among same-class boxes IoU>=.5, or0 if absent. Not one-to-one recall; compare target and equal-shape background erasure.',
                'threshold_policy': 'Thresholds fixed before this experiment; descriptive diagnostics, no selection or pass/fail training gate.'},
            'frames': records, 'source_hashes': {Path(__file__).name: sha256(Path(__file__)),
                                              'pretraining_data.py': sha256(BASE/'pretraining_data.py')},
            'optimizer_steps': 0, 'test_inference': False, 'validation_inference': False,
            'ready_to_resume_s5_training': False,
            'limits': ['Training-set mechanistic diagnostic, not independent accuracy/acceptance.',
                       'Annotation-free does not prove absence of unlabeled objects; GT rectangles include context, not segmentation.',
                       'Translations change image boundary context and are not guaranteed invariances of a detector.',
                       'Occlusion changes semantics; no AP is computed against erased-object labels.',
                       'No comparison of internal feature values across architectures or claim of equal compute.',
                       'No camera deployment, firmware arithmetic equivalence or training authorization is established.']}
    write_json(args.output/'frozen-plan.json', plan)
    counts = Counter(v['op'] for record in records for v in record['variants'])
    summary = {'frames': len(records), 'variant_counts': dict(counts),
               'occlusion_pairs_by_class': {c: sum(r['occlusion_pairs'][c] is not None for r in records) for c in CLASSES},
               'plan_sha256': sha256(args.output/'frozen-plan.json'), 'photo_inference_performed': False}
    write_json(args.output/'preparation-summary.json', summary)
    print(json.dumps(summary), flush=True)


def canonical_detections(items, width, height, classes):
    need(isinstance(items, list), 'predict_rgb must return a list')
    result = []
    for index, item in enumerate(plain(items)):
        need(isinstance(item, dict) and isinstance(item.get('class'), str), 'Invalid detection class')
        score, box = item.get('score'), item.get('bbox_xyxy')
        need(isinstance(score, (int, float)) and math.isfinite(score) and 0 <= score <= 1, 'Invalid score')
        need(isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float)) and math.isfinite(v) for v in box), 'Invalid prediction xyxy')
        if item['class'] not in classes:
            continue
        clipped = np.asarray(box, dtype=float)
        clipped[[0, 2]] = clipped[[0, 2]].clip(0, width)
        clipped[[1, 3]] = clipped[[1, 3]].clip(0, height)
        if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
            result.append({'class': item['class'], 'score': float(score), 'bbox_xyxy': clipped.tolist(), 'adapter_index': index})
    return result


def match(truth, detections, target_class):
    candidates = [t for t in truth if t['class'] == target_class]
    assignments = {}
    ranked = sorted(enumerate(detections), key=lambda row: (-row[1]['score'], row[0]))
    for pred_index, detection in ranked:
        if detection['class'] != target_class or detection['score'] < SCORE:
            continue
        scores = [(iou(t['bbox_xyxy'], detection['bbox_xyxy']), t['gt_index']) for t in candidates if t['gt_index'] not in assignments]
        overlap, gt_index = max(scores, key=lambda row: (row[0], -row[1]), default=(0., -1))
        if gt_index >= 0 and overlap >= IOU:
            assignments[gt_index] = {'prediction_index': pred_index, 'score': detection['score'],
                                     'iou': overlap, 'bbox_xyxy': detection['bbox_xyxy']}
    return assignments


def support(detections, box, target_class):
    scores = [d['score'] for d in detections if d['class'] == target_class and iou(d['bbox_xyxy'], box) >= IOU]
    return max(scores, default=0.)


def center(box):
    return np.array([(box[0]+box[2])/2, (box[1]+box[3])/2])


def summarize(plan, prediction):
    need(prediction['split'] == plan['split'] == 'train', 'Only internal training split is supported')
    records = {(r['frame_id'], r['variant']): r for r in prediction['records']}
    expected = {(r['frame']['id'], v['name']) for r in plan['frames'] for v in r['variants']}
    need(len(records) == len(prediction['records']) and set(records) == expected, 'Missing/duplicate/unexpected predictions')
    classes = prediction['classes']
    need(classes and len(set(classes)) == len(classes) and set(classes) <= set(CLASSES), 'Invalid/duplicate target classes')
    for record in plan['frames']:
        frame = record['frame']
        for variant in record['variants']:
            saved = records[(frame['id'], variant['name'])]
            need(saved['input_pixel_sha256'] == variant['pixel_sha256'], 'Prediction input differs from plan')
            for detection in saved['detections']:
                need(detection['class'] in classes and math.isfinite(detection['score']) and 0 <= detection['score'] <= 1,
                     'Invalid saved detection')
                validate_box(detection['bbox_xyxy'], frame['width'], frame['height'])
    translation, occlusion = [], []
    for record in plan['frames']:
        frame = record['frame']; fid = frame['id']
        real = records[(fid, 'real')]['detections']
        original_truth = mapped_truth(frame, {'op': 'identity'})
        for target_class in classes:
            original_matches = match(original_truth, real, target_class)
            for spec in record['variants']:
                if spec['op'] != 'translate_crop':
                    continue
                transformed = records[(fid, spec['name'])]['detections']
                truth = mapped_truth(frame, spec)
                shifted_matches = match(truth, transformed, target_class)
                delta = np.array([spec['dx'], spec['dy']]); magnitude = float(np.linalg.norm(delta))
                for gt in truth:
                    if gt['class'] != target_class or not gt['fully_retained']:
                        continue
                    index = gt['gt_index']; original_box = frame['boxes'][index]['bbox_xyxy']
                    first, second = original_matches.get(index), shifted_matches.get(index)
                    residual = None
                    if first and second:
                        residual = float(np.linalg.norm(center(second['bbox_xyxy'])-center(first['bbox_xyxy'])-delta))
                    translation.append({'frame_id': fid, 'class': target_class, 'variant': spec['name'],
                        'gt_index': index, 'translation_xy': delta.tolist(), 'original_box': original_box,
                        'shifted_box': gt['bbox_xyxy'], 'real_match': first, 'shifted_match': second,
                        'center_displacement_residual_pixels': residual,
                        'residual_over_translation': residual/magnitude if residual is not None else None,
                        'follows_when_original_detected': bool(first and second and residual <= .5*magnitude) if first else None,
                        'support_at_shifted_gt': support(transformed, gt['bbox_xyxy'], target_class),
                        'support_at_original_gt': support(transformed, original_box, target_class)})
            pair = record['occlusion_pairs'][target_class]
            if pair:
                box = frame['boxes'][pair['gt_index']]['bbox_xyxy']
                target = records[(fid, f'occlude_{target_class}_target')]['detections']
                background = records[(fid, f'occlude_{target_class}_background')]['detections']
                r, t, b = (support(items, box, target_class) for items in (real, target, background))
                occlusion.append({'frame_id': fid, 'class': target_class, **pair,
                                  'real_support': r, 'target_erased_support': t, 'background_erased_support': b,
                                  'real_minus_target': r-t, 'real_minus_background': r-b,
                                  'extra_drop_after_target_erasure': b-t,
                                  'originally_supported_at_score_0_5': r >= SCORE})
    aggregates = {}
    for target_class in classes:
        moved = [r for r in translation if r['class'] == target_class]
        eligible = [r for r in moved if r['real_match']]
        pairs = [r for r in occlusion if r['class'] == target_class]
        supported = [r for r in pairs if r['originally_supported_at_score_0_5']]
        residuals = [r['residual_over_translation'] for r in eligible if r['shifted_match']]
        mean = lambda xs: float(np.mean(xs)) if xs else None
        aggregates[target_class] = {'fully_retained_gt_translation_pairs': len(moved),
            'unique_fully_retained_gt': len({(r['frame_id'], r['gt_index']) for r in moved}),
            'original_matched_gt_translation_pairs': len(eligible),
            'shifted_matched_count_all_retained': sum(bool(r['shifted_match']) for r in moved),
            'shifted_matched_fraction_all_retained': mean([bool(r['shifted_match']) for r in moved]),
            'following_count_given_original_matched': sum(r['follows_when_original_detected'] for r in eligible),
            'following_fraction_given_original_matched': mean([r['follows_when_original_detected'] for r in eligible]),
            'mean_residual_over_translation_when_both_matched': mean(residuals),
            'mean_shifted_minus_stationary_support': mean([r['support_at_shifted_gt']-r['support_at_original_gt'] for r in moved]),
            'available_occlusion_pairs': len(pairs), 'unavailable_occlusion_frames': len(plan['frames'])-len(pairs),
            'originally_supported_occlusion_pairs': len(supported),
            'mean_extra_target_drop_all_pairs': mean([r['extra_drop_after_target_erasure'] for r in pairs]),
            'mean_extra_target_drop_originally_supported': mean([r['extra_drop_after_target_erasure'] for r in supported])}
    return {'model': prediction['model'], 'classes': classes, 'split': 'train', 'aggregates': aggregates,
            'translation_pairs': translation, 'occlusion_pairs': occlusion,
            'optimizer_steps': 0, 'test_inference': False, 'validation_inference': False,
            'ready_to_resume_s5_training': False, 'pass_fail_gate_defined': False,
            'limits': plan['limits']+['Several translations reuse one GT/image; pair counts are correlated, not independent samples.',
                                     'Local support is censored by the model export score floor/topK/NMS, and is not calibrated across architectures.',
                                     'A stationary large box may overlap old and moved GT; center displacement is reported in addition to IoU.',
                                     'No original detections gives a null conditional following rate, never a successful empty result.']}


def loaded_sources(module):
    paths = {Path(__file__).resolve(), BASE/'pretraining_data.py', Path(module.__file__).resolve()}
    for item in tuple(sys.modules.values()):
        value = getattr(item, '__file__', None)
        if not value:
            continue
        path = Path(value).resolve()
        if path.suffix == '.py' and path.is_relative_to(BASE) and path.is_file():
            parts = path.relative_to(BASE).parts[:-1]
            if not any(p.startswith('.venv') or p in {'venv', 'site-packages', 'dist-packages', '.tox', '.nox'} for p in parts):
                paths.add(path)
    return {str(p): sha256(p) for p in sorted(paths)}


def verify_model_files(model_dir, metadata):
    for name, item in metadata['model_file_hashes'].items():
        path = (model_dir/name).resolve()
        need(path.is_relative_to(model_dir), 'Model hash path escapes model directory')
        expected = item['sha256'] if isinstance(item, dict) else item
        need(sha256(path) == expected, 'Actual model SHA differs: '+name)


def run(args):
    need(not args.output.exists() or not any(args.output.iterdir()), 'Run output must be empty')
    plan = json.loads(args.plan.read_text()); plan_hash = sha256(args.plan)
    need(plan['split'] == 'train' and len(plan['frames']) == 16, 'Expected frozen16-train plan')
    need(args.model in plan['frozen_profiles'], 'Model profile was not frozen in this plan')
    need(sha256(Path(plan['dataset_manifest'])) == plan['dataset_manifest_sha256'], 'Dataset manifest changed')
    need(all(sha256(BASE/name) == digest for name, digest in plan['source_hashes'].items()), 'Preparation source changed; explicitly re-freeze before inference')
    module_name, sep, factory = args.adapter.partition(':')
    need(sep and module_name and factory, 'Expected module:factory adapter')
    module = importlib.import_module(module_name)
    detector = getattr(module, factory)(args.model, args.model_dir.resolve())
    need(callable(getattr(detector, 'predict_rgb', None)), 'Missing predict_rgb')
    metadata = plain(copy.deepcopy(detector.metadata))
    need(metadata.get('model_id') and metadata.get('model_file_hashes'), 'Adapter must identify model and model file hashes')
    need(metadata.get('output_coordinate_space') in ('original_image_xyxy', 'original_image_continuous_xyxy', 'original_image'), 'Adapter must declare original image xyxy output')
    verify_model_files(args.model_dir.resolve(), metadata)
    state_before = plain(detector.audit_state()) if callable(getattr(detector, 'audit_state', None)) else None
    sources = loaded_sources(module)
    binding = {'model': args.model, 'classes': args.classes, 'split': 'train', 'adapter': args.adapter,
               'plan_sha256': plan_hash, 'model_metadata': metadata, 'source_hashes': sources,
               'adapter_state_before': state_before,
               'optimizer_steps': 0, 'test_inference': False, 'validation_inference': False}
    # No prediction occurs before writing these frozen bindings.
    write_json(args.output/'frozen-run.json', binding)
    prediction = {**binding, 'records': []}
    root = Path(plan['dataset_manifest']).parent
    start = time.monotonic()
    for index, record in enumerate(plan['frames']):
        frame = record['frame']; rgb = load_rgb(root, frame)
        for spec in record['variants']:
            transformed = transform_rgb(rgb, spec)
            need(pixel_sha(transformed) == spec['pixel_sha256'], 'Transform pixels differ from frozen plan')
            raw = plain(detector.predict_rgb(transformed))
            need(pixel_sha(transformed) == spec['pixel_sha256'], 'Adapter mutated input RGB pixels')
            detections = canonical_detections(raw, frame['width'], frame['height'], args.classes)
            prediction['records'].append({'frame_id': frame['id'], 'variant': spec['name'],
                                          'input_pixel_sha256': spec['pixel_sha256'], 'detections': detections,
                                          'adapter_returned_raw': raw})
        print(json.dumps({'model': args.model, 'completed_train_frames': index+1, 'total': len(plan['frames'])}), flush=True)
    need(sha256(args.plan) == plan_hash, 'Plan changed during inference')
    need(all(sha256(Path(path)) == digest for path, digest in sources.items()), 'Project/adapter sources changed during inference')
    verify_model_files(args.model_dir.resolve(), metadata)
    state_after = plain(detector.audit_state()) if callable(getattr(detector, 'audit_state', None)) else None
    need(state_after == state_before, 'Adapter model state changed')
    prediction['adapter_state_after'] = state_after
    prediction['elapsed_seconds'] = time.monotonic()-start
    write_json(args.output/'predictions.json', prediction)
    report = summarize(plan, prediction)
    report.update(plan_sha256=plan_hash, predictions_sha256=sha256(args.output/'predictions.json'))
    write_json(args.output/'report.json', report)
    print(json.dumps(report['aggregates']), flush=True)


def analyze(args):
    plan, prediction = json.loads(args.plan.read_text()), json.loads(args.predictions.read_text())
    need(prediction['plan_sha256'] == sha256(args.plan), 'Saved predictions bind a different plan')
    report = summarize(plan, prediction)
    report.update(plan_sha256=sha256(args.plan), predictions_sha256=sha256(args.predictions),
                  verification='Recomputed numerical diagnostics from saved predictions; no model inference.')
    need(not args.output.exists(), 'Refusing to overwrite an existing analysis')
    write_json(args.output, report)
    print(json.dumps(report['aggregates']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare', help='Freeze16 training images and exact transforms; no model imports/inference')
    p.add_argument('--dataset', type=Path, default=BASE/'private-data/recognition-v2/manifest.json')
    p.add_argument('--expected-manifest-sha', default=MANIFEST_SHA)
    p.add_argument('--output', type=Path, default=DEFAULT_OUTPUT/'prepared')
    p.set_defaults(func=prepare)
    p = sub.add_parser('run', help='Run a frozen adapter independently in its existing runtime; no training')
    p.add_argument('--plan', type=Path, default=DEFAULT_OUTPUT/'prepared/frozen-plan.json')
    p.add_argument('--adapter', default='external_baseline_models:load_detector')
    p.add_argument('--model', required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--classes', nargs='+', choices=CLASSES, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.set_defaults(func=run)
    p = sub.add_parser('analyze', help='Recompute saved prediction diagnostics without inference')
    p.add_argument('--plan', type=Path, default=DEFAULT_OUTPUT/'prepared/frozen-plan.json')
    p.add_argument('--predictions', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.set_defaults(func=analyze)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
