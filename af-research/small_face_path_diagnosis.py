#!/usr/bin/env python3
"""Frozen train32 small-face stage/oracle-crop diagnosis, never training.

The oracle uses GT to position a source-image crop. Its success is not automatic
detection success, a deployable crop strategy, or original-camera equivalence.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from extract import MODELS, digest, require, write_json
from flattened_k_diagnosis import build_detector
from preprocessing_investigation import identity
from pretraining_data import checked_file, sha256
from proposal_hypothesis import nms
from train_recognition import CLASSES, MODEL, box_iou, decode_boxes, detections, load_image, source_hashes
from verify_external_baselines import independent_metrics

BASE = Path(__file__).resolve().parent
MANIFEST_SHA = 'af5459ddebd98947b154580cd176e7846abcceffe5539ac6f0001f6a485a3fc8'
WIDTH, HEIGHT, SCORE, IOU = 384, 256, .5, .5


def plain(value):
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def reference_gts(frame):
    scale = np.asarray([WIDTH/frame['width'], HEIGHT/frame['height']]*2, np.float64)
    return [dict(gt_index=i, target_class=b['class'], box=np.asarray(b['bbox_xyxy'])*scale,
                 original_box=b['bbox_xyxy']) for i, b in enumerate(frame['boxes'])]


def crop_spec(target, factor, source_size, reference_size=(WIDTH, HEIGHT)):
    rw, rh = reference_size
    center = (np.asarray(target[:2])+target[2:])/2
    left, top = center-[rw/(2*factor), rh/(2*factor)]
    sw, sh = source_size
    sx, sy = sw/rw, sh/rh
    return dict(factor=factor, center_reference=center.tolist(),
                reference_bounds=[left, top, left+rw/factor, top+rh/factor],
                inverse_pillow_affine=[sx/factor, 0., left*sx, 0., sy/factor, top*sy],
                source_bounds=[left*sx, top*sy, (left+rw/factor)*sx, (top+rh/factor)*sy],
                source_size=list(source_size), output_size=list(reference_size), fill=117,
                intersects_source_boundary=bool(left < 0 or top < 0 or left+rw/factor > rw or top+rh/factor > rh))


def transform_box(box, spec):
    rw, rh = spec['output_size']
    center = np.asarray(spec['center_reference'])
    return ((np.asarray(box).reshape(2, 2)-center)*spec['factor']+
            np.asarray([rw/2, rh/2])).reshape(4)


def transformed_gts(gts, spec):
    result = []
    for item in gts:
        raw = transform_box(item['box'], spec)
        clipped = raw.copy()
        clipped[[0, 2]] = clipped[[0, 2]].clip(0, WIDTH)
        clipped[[1, 3]] = clipped[[1, 3]].clip(0, HEIGHT)
        if np.all(clipped[2:] > clipped[:2]):
            result.append({**item, 'box': clipped, 'unclipped_box': raw,
                           'clipped_by_crop': not np.array_equal(raw, clipped)})
    return result


def crop_image(gray, spec, resample=Image.Resampling.BILINEAR):
    return np.asarray(gray.transform(tuple(spec['output_size']), Image.Transform.AFFINE,
                                     tuple(spec['inverse_pillow_affine']), resample=resample,
                                     fillcolor=117), dtype=np.float32).copy()


def greedy_matches(boxes, scores, indices, gt_rows):
    remaining = set(range(len(gt_rows)))
    found = {}
    for index in indices:  # Input already in stable descending score order.
        if scores[index] < SCORE or not remaining:
            continue
        overlaps = box_iou(boxes[index:index+1], np.asarray([g['box'] for g in gt_rows]))[0]
        best, j = max((float(overlaps[j]), j) for j in remaining)
        if best >= IOU:
            remaining.remove(j)
            found[gt_rows[j]['gt_index']] = dict(anchor_index=int(index), iou=best, score=float(scores[index]),
                                                bbox_xyxy=boxes[index].tolist())
    return found


def stage_category(row):
    if row['final_one_to_one_match'] is not None:
        return 'detected'
    if row['decoded_all']['max_iou'] < IOU:
        return 'decoded_geometry_insufficient'
    if row['decoded_score_ge_05']['max_iou'] < IOU:
        return 'classification_confidence_insufficient'
    if row['pre_nms_top3000_score_ge_05']['max_iou'] < IOU:
        return 'pre_nms_top3000_excluded'
    if row['final30_score_ge_05']['max_iou'] < IOU:
        return 'nms_or_final30_removed'
    return 'one_to_one_competition'


def maximum_info(boxes, indices, target, scores=None):
    if not len(indices):
        return dict(max_iou=0., anchor_index=None, bbox_xyxy=None, score=None)
    overlaps = box_iou(boxes[indices], np.asarray(target)[None])[..., 0]
    j = int(np.argmax(overlaps)); index = int(indices[j])
    return dict(max_iou=float(overlaps[j]), anchor_index=index, bbox_xyxy=boxes[index].tolist(),
                score=None if scores is None else float(scores[index]))


def measure(detector, tensor, gt_rows, save_path):
    logits, delta = detector(tensor)
    boxes = decode_boxes(detector.anchors, delta.numpy())
    valid = np.flatnonzero(np.all(boxes[:, 2:]-boxes[:, :2] > 0, axis=1))
    all_anchor_indices = np.arange(len(detector.anchors))
    archive = {'decoded_boxes': boxes, 'valid_indices': valid}
    per_gt, final_detections = [], []
    reference_frame = dict(width=WIDTH, height=HEIGHT)
    existing = detections(logits, delta, detector.anchors, reference_frame)
    for cls in CLASSES:
        score = logits[cls].softmax(1)[:, 1].numpy()
        pre = valid[np.argsort(-score[valid], kind='stable')[:3000]]
        final = pre[nms(boxes[pre]-np.asarray([0, 0, 1, 1]), score[pre], .3, 30)]
        archive['scores_'+cls] = score
        archive['pre_nms_indices_'+cls] = pre
        archive['final_indices_'+cls] = final
        class_gts = [g for g in gt_rows if g['target_class'] == cls]
        matches = greedy_matches(boxes, score, final, class_gts)
        score_valid = valid[score[valid] >= SCORE]
        score_pre = pre[score[pre] >= SCORE]
        score_final = final[score[final] >= SCORE]
        final_detections.extend(dict(**{'class': cls}, bbox_xyxy=boxes[i].tolist(), score=float(score[i]),
                                     anchor_index=int(i)) for i in final)
        for item in class_gts:
            target = item['box']
            row = dict(gt_index=item['gt_index'], target_class=cls, target_box=target.tolist(),
                       reference_sqrt_area=float(np.sqrt(np.prod(target[2:]-target[:2]))),
                       original_box=item['original_box'],
                       anchor_geometry=maximum_info(detector.anchors, all_anchor_indices, target),
                       decoded_all=maximum_info(boxes, valid, target, score),
                       decoded_score_ge_05=maximum_info(boxes, score_valid, target, score),
                       pre_nms_top3000_score_ge_05=maximum_info(boxes, score_pre, target, score),
                       final30_score_ge_05=maximum_info(boxes, score_final, target, score),
                       final_one_to_one_match=matches.get(item['gt_index']))
            row['failure_stage'] = stage_category(row)
            row['original_anchor_max_iou_below_05'] = row['anchor_geometry']['max_iou'] < IOU
            row['decoded_geometry_can_match'] = row['decoded_all']['max_iou'] >= IOU
            per_gt.append(row)
    reconstructed = [{k: v for k, v in row.items() if k != 'anchor_index'} for row in final_detections]
    require(len(existing) == len(reconstructed), 'Final decoder count differs from unchanged implementation')
    for actual, expected in zip(reconstructed, existing):
        require(actual['class'] == expected['class'] and actual['score'] == expected['score'] and
                np.allclose(actual['bbox_xyxy'], expected['bbox_xyxy'], rtol=0, atol=1e-6),
                'Final decoder differs from unchanged implementation')
    np.savez_compressed(save_path, **archive)
    return dict(per_gt=per_gt, final_detections=final_detections,
                input_tensor_sha256=digest(tensor.contiguous().numpy().tobytes()),
                input_mean=float(tensor.numpy().astype(np.float64).mean())+117., input_min=float(tensor.min())+117.,
                input_max=float(tensor.max())+117., saved_prediction_sha256=sha256(save_path))


def self_check():
    target = np.asarray([3., 2., 5., 4.])
    spec = crop_spec(target, 2, (8, 6), (8, 6))
    require(np.array_equal(transform_box(target, spec), [2, 1, 6, 5]), 'GT center/scale fixture failed')
    ramp = np.arange(48, dtype=np.uint8).reshape(6, 8)
    actual = crop_image(Image.fromarray(ramp), spec, Image.Resampling.NEAREST)
    a, _, c, _, e, f = spec['inverse_pillow_affine']
    expected = np.full((6, 8), 117, np.float32)
    for y in range(6):
        for x in range(8):
            sx, sy = int(np.floor(a*(x+.5)+c)), int(np.floor(e*(y+.5)+f))
            if 0 <= sx < 8 and 0 <= sy < 6:
                expected[y, x] = ramp[sy, sx]
    require(np.array_equal(expected, actual), 'Affine pixel-center coordinate fixture failed')
    edge = crop_spec(np.asarray([0., 0., 2., 2.]), 2, (8, 6), (8, 6))
    edge_pixels = crop_image(Image.fromarray(np.full((6, 8), 43, np.uint8)), edge)
    require(edge['intersects_source_boundary'] and edge_pixels[0, 0] == 117 and edge_pixels[3, 4] == 43,
            'Out-of-source fill fixture failed')
    for factor in (2, 4):
        spec = crop_spec(np.asarray([30., 20., 50., 40.]), factor, (768, 512))
        box = np.asarray([30., 20., 50., 40.])
        changed = transform_box(box, spec).reshape(2, 2)
        inverse = (changed-np.asarray([192, 128]))/factor+spec['center_reference']
        require(np.array_equal(inverse.reshape(4), box), 'Coordinate inverse failed')
    boxes = np.asarray([[0., 0., 10., 10.], [0., 0., 10., 10.]])
    gts = [dict(gt_index=7, box=boxes[0]), dict(gt_index=8, box=boxes[1])]
    matched = greedy_matches(boxes, np.asarray([.9, .8]), [0, 1], gts)
    require(matched[8]['anchor_index'] == 0 and matched[7]['anchor_index'] == 1,
            'One-to-one/larger-GT-index tie fixture failed')
    prototype = dict(final_one_to_one_match=None, decoded_all={'max_iou': .6},
        decoded_score_ge_05={'max_iou': .6}, pre_nms_top3000_score_ge_05={'max_iou': .6},
        final30_score_ge_05={'max_iou': .6})
    expected_stages = [('decoded_all', 'decoded_geometry_insufficient'),
                      ('decoded_score_ge_05', 'classification_confidence_insufficient'),
                      ('pre_nms_top3000_score_ge_05', 'pre_nms_top3000_excluded'),
                      ('final30_score_ge_05', 'nms_or_final30_removed')]
    for key, label in expected_stages:
        require(stage_category({**prototype, key: {'max_iou': .4}}) == label, 'Stage taxonomy fixture failed')
    require(stage_category(prototype) == 'one_to_one_competition', 'Matching competition fixture failed')
    return dict(gt_affine_center_scale=True, pixel_center_sampling=True, out_of_source_fill=True,
                coordinate_roundtrip=True, one_to_one_tie=True, stage_taxonomy=True)


def summarize(rows):
    result = {}
    for cls in CLASSES:
        for group in ('all', 'small_lt32'):
            subset = [r for r in rows if r['target_class'] == cls and
                      (group == 'all' or r['reference_sqrt_area'] < 32)]
            result[cls+'/'+group] = dict(gt=len(subset),
                failure_stage_counts=dict(Counter(r['failure_stage'] for r in subset)),
                anchor_covered=sum(r['anchor_geometry']['max_iou'] >= IOU for r in subset),
                decoded_all_covered=sum(r['decoded_all']['max_iou'] >= IOU for r in subset),
                decoded_confident_covered=sum(r['decoded_score_ge_05']['max_iou'] >= IOU for r in subset),
                final_one_to_one_matched=sum(r['final_one_to_one_match'] is not None for r in subset))
    return result


def run(args):
    fixtures = self_check()
    require(not args.output.exists(), 'Choose a new output directory; preserve frozen evidence')
    require(sha256(args.dataset) == MANIFEST_SHA, 'Unexpected dataset manifest')
    dataset = json.loads(args.dataset.read_text())
    frames = sorted((f for f in dataset['frames'] if f['split'] == 'train'), key=lambda f: f['id'])[:32]
    require(len(frames) == 32, 'Need fixed train32')
    selected = []
    for frame in frames:
        for gt in reference_gts(frame):
            if gt['target_class'] == 'person_face' and np.sqrt(np.prod(gt['box'][2:]-gt['box'][:2])) < 32:
                selected.append(dict(frame_id=frame['id'], gt_index=gt['gt_index'],
                                     reference_box=gt['box'].tolist()))
    require(len(selected) == 10, 'Frozen train32 must contain exactly10 small faces')
    sources = {**source_hashes(), **{n: sha256(BASE/n) for n in
        ('small_face_path_diagnosis.py', 'flattened_k_diagnosis.py', 'preprocessing_investigation.py',
         'verify_external_baselines.py')}}
    require(sha256(args.firmware) == MODELS[MODEL], 'Unexpected original firmware component')
    args.output.mkdir(parents=True)
    snapshot = args.output/'source-snapshot'; snapshot.mkdir()
    for name in sources:
        (snapshot/name).write_bytes((BASE/name).read_bytes())
    plan = dict(status='Frozen before photograph inference', model=MODEL, model_sha256=MODELS[MODEL],
        manifest_sha256=MANIFEST_SHA, sources=sources, self_checks=fixtures,
        frames=[dict(id=f['id'], sha256=f['sha256']) for f in frames], oracle_small_targets=selected,
        baseline='RGB->Pillow L->BILINEAR384x256 stretch minus117', score_threshold=SCORE, iou_threshold=IOU,
        small_definition='sqrt(original GT area after stretch to384x256)<32',
        stages=['all original anchors', 'all positive-area decoded boxes before NMS',
                'decoded score>=.5', 'score>=.5 among preNMS3000', 'score>=.5 final30', 'final30 one-to-one match'],
        failure_taxonomy=['decoded_geometry_insufficient', 'classification_confidence_insufficient',
                          'pre_nms_top3000_excluded', 'nms_or_final30_removed', 'one_to_one_competition'],
        matching='Per class; descending score, stable anchor order; unused GT with max IoU; larger GT index breaks ties.',
        oracle=dict(factors=[2, 4], center='GT center only', output=[WIDTH, HEIGHT], fill=117,
            sampling='Original stored RGB->Pillow L sampled with affine BILINEAR; no EXIF rotation',
            reference_crop_size='384/factor by256/factor',
            box_mapping="x'=factor*(x-cx)+192; y'=factor*(y-cy)+128",
            controls=['real oracle crop', 'constant float image with same mean L value as oracle crop'],
            limitations=['GT localization is supplied; not automatic detection or deployment.',
                'Cropping changes target size, context and retained source detail simultaneously.',
                'All visible clipped GTs are retained for one-to-one matching; selected target is required not to be clipped.']),
        anchors_changed=False, parameters_changed=False, optimizer_steps=0,
        validation_inference=False, test_inference=False, ready_to_resume_recognition_training=False)
    write_json(args.output/'frozen-plan.json', plan)
    prediction_dir = args.output/'predictions'; prediction_dir.mkdir()
    torch.set_num_threads(args.threads); torch.manual_seed(0); torch.use_deterministic_algorithms(True)
    detector = build_detector(args.firmware.read_bytes())
    before = identity(detector.model.state_dict())
    require(not any(p.requires_grad for p in detector.model.parameters()), 'Parameters are not frozen')
    baseline, oracle, per_gt = [], [], []
    started = time.monotonic()
    try:
        with torch.inference_mode():
            for position, frame in enumerate(frames):
                path = checked_file(args.dataset.parent, frame)
                gts = reference_gts(frame)
                tensor = load_image(args.dataset.parent, frame)
                result = measure(detector, tensor, gts, prediction_dir/f'baseline-{position:02d}.npz')
                result.update(frame_id=frame['id'], source_image_sha256=frame['sha256'],
                              prediction_file=f'predictions/baseline-{position:02d}.npz')
                baseline.append(result)
                per_gt.extend(dict(frame_id=frame['id'], **row) for row in result['per_gt'])
                chosen = [g for g in gts if g['target_class'] == 'person_face' and
                          np.sqrt(np.prod(g['box'][2:]-g['box'][:2])) < 32]
                if chosen:
                    with Image.open(path) as image:
                        gray = image.convert('RGB').convert('L')
                    for gt in chosen:
                        for factor in (2, 4):
                            spec = crop_spec(gt['box'], factor, gray.size)
                            crop = crop_image(gray, spec)
                            transformed = transformed_gts(gts, spec)
                            target = next(g for g in transformed if g['gt_index'] == gt['gt_index'])
                            require(not target['clipped_by_crop'], 'Selected oracle target is clipped')
                            crop_mean = float(crop.astype(np.float64).mean())
                            entry = dict(frame_id=frame['id'], source_image_sha256=frame['sha256'],
                                gt_index=gt['gt_index'], original_reference_box=gt['box'].tolist(),
                                original_reference_sqrt_area=float(np.sqrt(np.prod(gt['box'][2:]-gt['box'][:2]))),
                                crop_spec=spec, crop_mean=crop_mean, transformed_target_box=target['box'].tolist(), modes={})
                            for mode in ('real', 'same_mean'):
                                pixels = crop if mode == 'real' else np.full(crop.shape, crop_mean, np.float32)
                                prepared = torch.from_numpy(pixels.copy()-117.)[None, None]
                                filename = f'oracle-{position:02d}-gt{gt["gt_index"]}-x{factor}-{mode}.npz'
                                measured = measure(detector, prepared, transformed, prediction_dir/filename)
                                measured['target'] = next(r for r in measured['per_gt'] if r['gt_index'] == gt['gt_index'])
                                measured['prediction_file'] = 'predictions/'+filename
                                entry['modes'][mode] = measured
                            require(abs(entry['modes']['real']['input_mean']-entry['modes']['same_mean']['input_mean']) < 2e-5,
                                    'Oracle mean control changed input mean')
                            oracle.append(entry)
                print(json.dumps(dict(completed_train_images=position+1, total=32, oracle_conditions=len(oracle))), flush=True)
    finally:
        detector.close()
    require(identity(detector.model.state_dict()) == before, 'Weights changed during diagnosis')
    require(sha256(args.dataset) == MANIFEST_SHA, 'Manifest changed')
    require(all(sha256(BASE/n) == h for n, h in sources.items()), 'Bound source changed during diagnosis')
    reference_frames = {f['id']: dict(id=f['id'], width=WIDTH, height=HEIGHT,
        boxes=[{'class': g['target_class'], 'bbox_xyxy': g['box'].tolist()} for g in reference_gts(f)]) for f in frames}
    independent = {}
    for cls in CLASSES:
        predictions = {r['frame_id']: [d for d in r['final_detections'] if d['class'] == cls] for r in baseline}
        independent[cls] = independent_metrics(reference_frames, predictions, cls)
        require(independent[cls]['tp_at_score'] == sum(r['final_one_to_one_match'] is not None for r in per_gt if r['target_class'] == cls),
                'Independent one-to-one TP count differs')
    oracle_summary = {}
    for factor in (2, 4):
        subset = [r for r in oracle if r['crop_spec']['factor'] == factor]
        oracle_summary[str(factor)] = {}
        for mode in ('real', 'same_mean'):
            targets = [r['modes'][mode]['target'] for r in subset]
            oracle_summary[str(factor)][mode] = dict(targets=len(targets),
                final_one_to_one_matches=sum(r['final_one_to_one_match'] is not None for r in targets),
                decoded_all_covered=sum(r['decoded_all']['max_iou'] >= IOU for r in targets),
                decoded_confident_covered=sum(r['decoded_score_ge_05']['max_iou'] >= IOU for r in targets),
                stage_counts=dict(Counter(r['failure_stage'] for r in targets)))
    report = dict(status='complete', frozen_plan_sha256=sha256(args.output/'frozen-plan.json'),
        source_sha256=sources, all_sources_unchanged=True, manifest_sha256=MANIFEST_SHA,
        model_sha256=MODELS[MODEL], actual_state_sha256=before, actual_state_unchanged=True,
        baseline_summary=summarize(per_gt), baseline_independent_metrics=independent, oracle_summary=oracle_summary,
        baseline=baseline, oracle=oracle, elapsed_seconds=time.monotonic()-started,
        prediction_files={p.relative_to(args.output).as_posix(): sha256(p) for p in sorted(prediction_dir.iterdir())},
        optimizer_steps=0, validation_inference=False, test_inference=False,
        anchors_changed=False, parameters_changed=False, ready_to_resume_recognition_training=False,
        limitations=['Oracle crops use known GT centers; success cannot count as full-image automatic detection.',
            'Failure categories describe this fixed host decode/score/NMS path, not original DSP causal proof.',
            'A low original-anchor overlap alone does not imply impossibility: bbox regression may recover a GT.',
            'NMS suppression and final30 rank truncation are grouped; one-to-one competition is separate.',
            'Fixed train32 diagnosis is not a generalization estimate or threshold-selection exercise.'])
    write_json(args.output/'report.json', plain(report))
    print(json.dumps(dict(baseline=report['baseline_summary'], oracle=oracle_summary)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--firmware', type=Path)
    parser.add_argument('--dataset', type=Path, default=BASE/'private-data/recognition-v2/manifest.json')
    parser.add_argument('--output', type=Path, default=BASE/'artifacts/recognition-diagnosis-v3/small-face-path-train32')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check()))
    else:
        require(args.firmware is not None, 'Provide pinned original firmware component')
        run(args)
