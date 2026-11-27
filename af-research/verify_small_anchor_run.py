#!/usr/bin/env python3
"""Independent audit of bounded small-anchor training and saved predictions.

The data command reads manifest metadata only. The run command reads trusted
locally produced checkpoints and reconstructs original parameters/GT assignment,
but never performs model forward inference or optimizer updates. Test photographs
and predictions are never read. Host head quantization is not DSP equivalence.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from pretraining_data import evaluate_detection
from verify_external_baselines import independent_metrics, verify_cyclic_control
from verify_flattened_k_saved_predictions import check_predictions

BASE = Path(__file__).resolve().parent
CLASSES = ('person', 'person_face')
HEAD_NAMES = tuple(f'convs.{layer}.{kind}' for layer in (86, 90, 94) for kind in ('weight', 'bias'))
CONTROLS = ('real', 'same_mean', 'fixed_gray117', 'shuffled')
MANIFEST_SHA = 'af5459ddebd98947b154580cd176e7846abcceffe5539ac6f0001f6a485a3fc8'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    need(not path.exists(), 'Refusing to overwrite evidence: '+str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def state_sha(state):
    """Independent implementation of the run's sorted-name/raw-value digest."""
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def backbone_sha(state):
    return state_sha({k: v for k, v in state.items() if k not in HEAD_NAMES})


def anchor_sha(value):
    return hashlib.sha256(np.asarray(value, dtype='<f4').tobytes(order='C')).hexdigest()


def owner_sha(assigned):
    h = hashlib.sha256()
    for key, dtype in [('matched_gt', '<i8'), ('gt_classes', '<i8'), ('gt_boxes', '<f4')]:
        h.update(np.asarray(assigned[key], dtype=dtype).tobytes(order='C'))
    return h.hexdigest()


def dataset_audit(dataset, manifest_sha):
    need(manifest_sha == MANIFEST_SHA, 'Different dataset requires a new preregistration')
    groups = {split: [f for f in dataset['frames'] if f['split'] == split]
              for split in ('train', 'validation', 'test')}
    need(len(groups['train']) == 320 and len(groups['validation']) == 80, 'Unexpected training/validation size')
    summaries = {}
    for split in ('train', 'validation'):
        frames = groups[split]; classes = {}
        for cls in CLASSES:
            sizes = []
            small_ids = set()
            for f in frames:
                for b in f['boxes']:
                    if b['class'] != cls:
                        continue
                    x1, y1, x2, y2 = b['bbox_xyxy']
                    side = math.sqrt((x2-x1)*(y2-y1)*384*256/(f['width']*f['height']))
                    sizes.append(side)
                    if side < 32:
                        small_ids.add(f['id'])
            classes[cls] = {'boxes': len(sizes), 'tiny_below16': sum(x < 16 for x in sizes),
                'small_below32': sum(x < 32 for x in sizes), 'medium_32_to_below96': sum(32 <= x < 96 for x in sizes),
                'large_at_least96': sum(x >= 96 for x in sizes), 'images_with_small': len(small_ids),
                'positive_images': sum(any(b['class'] == cls for b in f['boxes']) for f in frames)}
        summaries[split] = {'images': len(frames), 'author_groups': len({f['group_id'] for f in frames}),
                            'empty_background_images': sum(not f['boxes'] for f in frames), 'classes': classes}
    separation = {}
    for left, right in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        row = {key: len({f[key] for f in groups[left]} & {f[key] for f in groups[right]})
               for key in ('id', 'group_id', 'sha256', 'pixel_sha256', 'source_image_id')}
        need(not any(row.values()), 'Cross-split duplicate identity/author found')
        separation[left+'__'+right] = row
    distances = [(int(a['dhash'], 16)^int(b['dhash'], 16)).bit_count()
                 for a in groups['train'] for b in groups['validation']]
    need(min(distances) > 4, 'Cross train/validation near-duplicate detected')
    return {'manifest_sha256': manifest_sha, 'splits': summaries, 'metadata_identity_overlap': separation,
            'train_validation_minimum_dhash_distance': min(distances),
            'small_rule': 'sqrt(GT area after reference stretch384x256)<32; not actual letterbox size.',
            'test_scope': 'Only ID/group/hash metadata used for isolation; no test labels summarized, photos opened or predictions evaluated.',
            'limitations': ['Positive-only public-image pilot, not S5 camera-domain or empty-scene validation.',
                           'Repeatedly inspected validation is for model development, not final independent acceptance.']}


def recompute_metrics(dataset, document, manifest_sha):
    frames = {f['id']: f for f in dataset['frames'] if f['split'] == 'validation'}
    need(list(frames) or False, 'No validation frames')
    rows = check_predictions(document, frames, manifest_sha)
    need([r['id'] for r in document['frames']] == sorted(frames), 'Prediction order differs from frozen sorted validation IDs')
    metrics = {cls: independent_metrics(frames, {fid: [d for d in values if d['class'] == cls]
                     for fid, values in rows.items()}, cls, .5) for cls in CLASSES}
    legacy = evaluate_detection({'classes': list(CLASSES), 'frames': list(frames.values())},
                                document, 'validation', score_threshold=.5)
    for cls in CLASSES:
        need(abs(metrics[cls]['ap_at_iou']-legacy['classes'][cls]['ap_at_iou']) <= 1e-12,
             'Two AP implementations disagree')
        for key in ('ground_truth_boxes', 'predictions', 'tp_at_score', 'fp_at_score', 'fn_at_score'):
            need(metrics[cls][key] == legacy['classes'][cls][key], 'Two count implementations disagree')
    bins = metrics['person_face']['size_strata']
    small = {key: sum(bins[name][key] for name in ('below16', '16_to_below32'))
             for key in ('ground_truth', 'matched_at_score', 'matched_all_saved_scores')}
    small['recall_at_score'] = small['matched_at_score']/small['ground_truth']
    return {'classes': metrics, 'small_faces': small,
            'mean_ap': sum(metrics[c]['ap_at_iou'] for c in CLASSES)/2,
            'legacy_crosscheck': legacy}


def acceptance(final, original, controls):
    need(set(controls) == set(CONTROLS), 'Acceptance requires all frozen controls')
    small = final['small_faces']; before, after = original['classes'], final['classes']
    checks = {
        'small_face_recall_at_least_half': small['recall_at_score'] >= .5,
        'small_face_matches_at_least_17_of34': small['ground_truth'] == 34 and small['matched_at_score'] >= 17,
        'face_ap_floor_and_improvement': after['person_face']['ap_at_iou'] >= max(.5, before['person_face']['ap_at_iou']+.05),
        'person_ap_drop_at_most_0_05': after['person']['ap_at_iou'] >= before['person']['ap_at_iou']-.05,
        'mean_ap_gain_at_least_0_02': final['mean_ap'] >= original['mean_ap']+.02}
    for cls in CLASSES:
        checks[cls+'_beats_every_control_by_0_15'] = all(after[cls]['ap_at_iou'] >= controls[mode]['classes'][cls]['ap_at_iou']+.15
                                                       for mode in CONTROLS if mode != 'real')
    return {'checks': checks, 'passed': all(checks.values()), 'camera_readiness': False}


def audit_state(initial, current, require_update=False):
    import torch
    need(set(initial) == set(current), 'Checkpoint topology differs from original K4')
    changed = []
    for name in initial:
        left, right = initial[name], current[name]
        need(left.shape == right.shape and left.dtype == right.dtype and bool(torch.isfinite(right).all()), 'Invalid checkpoint tensor: '+name)
        if not torch.equal(left, right):
            need(name in HEAD_NAMES, 'Frozen backbone changed: '+name)
            changed.append(name)
    if require_update:
        need(changed, 'Trained state has no head updates')
    return {'state_sha256': state_sha(current), 'backbone_sha256': backbone_sha(current), 'changed_head_tensors': changed}


def reconstruct_analytic_transfer(original, provenance, old_anchors, new_anchors):
    """Independently derive slot/gain/offset and copy rows; no transfer imports."""
    import torch
    old_scales, new_scales = provenance['original_scales'], provenance['new_scales']
    need(len(old_scales) == len(new_scales) == 5 and all(math.isfinite(x) and x > 0 for x in old_scales+new_scales),
         'Invalid transfer scales')
    scale_indices = []
    for scale in new_scales:
        distances = [abs(math.log(old/scale)) for old in old_scales]
        nearest = min(distances)
        scale_indices.append(next(i for i, distance in enumerate(distances) if abs(distance-nearest) <= 1e-12))
    slots = [ratio*5+scale_indices[j] for ratio in range(5) for j in range(5)]
    need(provenance['old_scale_index_by_new_scale'] == scale_indices and provenance['old_slot_by_new_slot'] == slots,
         'Transfer did not use fixed nearest-log same-ratio slots')
    class_rows = [half*25+slots[slot] for half in range(2) for slot in range(25)]
    bbox_rows = [slots[slot]*4+coordinate for slot in range(25) for coordinate in range(4)]
    need(provenance['class_bg_fg_channel_map'] == class_rows and provenance['bbox_xywh_channel_map'] == bbox_rows,
         'Transfer head channel mappings differ from independently derived rows')
    old, new = np.asarray(old_anchors, np.float64), np.asarray(new_anchors, np.float64)
    need(old.shape == new.shape == (9600, 4) and np.isfinite(old).all() and np.isfinite(new).all(), 'Invalid transfer anchor arrays')
    old_size, new_size = old[:, 2:]-old[:, :2], new[:, 2:]-new[:, :2]
    need(np.all(old_size > 0) and np.all(new_size > 0), 'Nonpositive transfer anchor dimensions')
    for ratio in range(5):
        unit = old_size[ratio*5]/old_scales[0]
        for j in range(5):
            need(np.allclose(old_size[ratio*5+j], unit*old_scales[j], atol=1e-7, rtol=0), 'Original scale metadata inconsistent with anchors')
            need(np.allclose(new_size[ratio*5+j], unit*new_scales[j], atol=1e-7, rtol=0), 'New scale metadata inconsistent with anchors')
    old_centers, new_centers = (old[:, :2]+old[:, 2:])/2, (new[:, :2]+new[:, 2:])/2
    need(np.array_equal(old_centers, new_centers), 'Transfer must preserve grid and anchor centers')
    gains, offsets = [], []
    for slot, old_slot in enumerate(slots):
        width_ratio = old_size[old_slot, 0]/new_size[slot, 0]
        height_ratio = old_size[old_slot, 1]/new_size[slot, 1]
        gains.append([width_ratio, height_ratio, 1., 1.])
        offsets.append([(old_centers[old_slot, 0]-new_centers[slot, 0])/new_size[slot, 0],
                        (old_centers[old_slot, 1]-new_centers[slot, 1])/new_size[slot, 1],
                        math.log(width_ratio), math.log(height_ratio)])
    gain = np.asarray(gains, np.float64); offset = np.asarray(offsets, np.float64)
    need(np.allclose(provenance['bbox_gain_by_slot'], gain, atol=1e-12, rtol=0) and
         np.allclose(provenance['bbox_offset_by_slot'], offset, atol=1e-12, rtol=0), 'Declared affine differs from anchor geometry')
    mapped = np.asarray([(row//25)*25+slots[row % 25] for row in range(9600)], dtype='<i8')
    need(hashlib.sha256(mapped.tobytes()).hexdigest() == provenance['row_mapping_sha256'], 'Spatial slot mapping digest differs')
    need(np.allclose(old_size[mapped]/new_size, np.tile(gain[:, :2], (384, 1)), atol=1e-12, rtol=0), 'Shared affine varies over spatial grid')
    need(np.allclose((old_centers[mapped]-new_centers)/new_size, np.tile(offset[:, :2], (384, 1)), atol=1e-12, rtol=0), 'Shared center offset varies over spatial grid')
    state = {name: tensor.detach().clone() for name, tensor in original.items()}
    for index in ('86', '90'):
        weight_name, bias_name = f'convs.{index}.weight', f'convs.{index}.bias'
        need(original[weight_name].shape[0] == 50 and original[weight_name].shape[2:] == (1, 1) and original[bias_name].shape == (50,),
             'Transfer only supports the frozen50-channel classification heads')
        for half in range(2):
            for slot in range(25):
                destination, source = half*25+slot, half*25+slots[slot]
                state[weight_name][destination].copy_(original[weight_name][source])
                state[bias_name][destination].copy_(original[bias_name][source])
    need(original['convs.94.weight'].shape[0] == 100 and original['convs.94.weight'].shape[2:] == (1, 1)
         and original['convs.94.bias'].shape == (100,), 'Transfer only supports the frozen100-channel bbox head')
    for slot in range(25):
        for coordinate in range(4):
            destination, source = slot*4+coordinate, slots[slot]*4+coordinate
            # Geometry was independently checked above. Use the exact frozen
            # float64 literals for bitwise initialization reproduction rather
            # than assuming libm/numpy log produce identical last bits.
            g = float(provenance['bbox_gain_by_slot'][slot][coordinate])
            o = float(provenance['bbox_offset_by_slot'][slot][coordinate])
            weight = original['convs.94.weight'][source].detach().cpu().numpy().astype(np.float64)
            bias = float(original['convs.94.bias'][source])
            state['convs.94.weight'][destination].copy_(torch.from_numpy((weight*g).astype(np.float32)))
            state['convs.94.bias'][destination] = np.float32(bias*g+o).item()
    check = audit_state(original, state)
    need(check['backbone_sha256'] == backbone_sha(original), 'Independent transfer changed backbone')
    return state, {'method': 'Independent same-ratio nearest-log selection and geometry check, frozen gain/offset literals, explicit per-output-row copying; no apply_transfer import/call.',
                   'old_scale_indices': scale_indices, 'old_slot_by_new_slot': slots,
                   'original_state_sha256': state_sha(original), 'transferred_state_sha256': state_sha(state),
                   'changed_head_tensors': check['changed_head_tensors'], 'backbone_unchanged': True}


def audit_optimizer(optimizer, names, state, expected_steps):
    import torch
    identifiers = [value for group in optimizer['param_groups'] for value in group['params']]
    need(len(identifiers) == len(set(identifiers)) == 6 and set(names) == set(HEAD_NAMES), 'Unexpected optimizer parameter identities')
    for group in optimizer['param_groups']:
        need(group['lr'] == 1e-4 and group['weight_decay'] == 1e-4, 'Optimizer hyperparameters changed')
    values = optimizer['state']
    if expected_steps == 0:
        need(not values, 'Epoch0 optimizer has already updated')
    else:
        need(set(values) == set(identifiers), 'Optimizer moments missing/unexpected')
        for identifier, name in zip(identifiers, names):
            row = values[identifier]
            need(float(row['step']) == expected_steps, 'Optimizer state step mismatch')
            for field in ('exp_avg', 'exp_avg_sq'):
                need(row[field].shape == state[name].shape and bool(torch.isfinite(row[field]).all()), 'Invalid AdamW moment tensor')


def audit_training_logs(run, plan, expected_owner, expected_counts=None):
    images = [json.loads(s) for s in (run/'training.jsonl').read_text().splitlines()]
    steps = [json.loads(s) for s in (run/'steps.jsonl').read_text().splitlines()]
    need(len(images) == 1920 and len(steps) == 480, 'Expected6x320 image visits and480 updates')
    train_ids = plan['train_ids']
    for epoch in range(1, 7):
        rows = [r for r in images if r['epoch'] == epoch]
        need(Counter(r['frame_id'] for r in rows) == Counter(train_ids), 'Missing/repeated training image within epoch')
        need([r['image_index'] for r in rows] == list(range(320)), 'Training order accounting differs')
        for i, row in enumerate(rows):
            need(row['optimizer_steps'] == (epoch-1)*80+(i+1)//4, 'Image log optimizer count differs')
            need(row['target_owner_sha256'] == expected_owner[row['frame_id']], 'Loss target owner differs from GT identity')
            need(row['unmatched_gt_count'] == 0, 'A GT was silently dropped')
            if expected_counts is not None:
                counts = expected_counts[row['frame_id']]
                need(row['target_gt_count'] == row['bbox_supervised_gt'] == sum(counts.values()), 'BBox loss does not cover every GT identity')
                for cls in CLASSES:
                    need(row[cls+'_supervised_gt'] == row[cls+'_forced_positive_anchors'] == counts[cls], 'Classification loss does not cover its class GT')
                    need(counts[cls] <= row[cls+'_positive_anchors'] <= 32, 'Classification positive budget differs')
                required = sum(row[cls+'_positive_anchors'] for cls in CLASSES)
                need(row['bbox_required_classification_positives'] == required and required <= row['bbox_positive_anchors'] <= 128,
                     'Classification-positive anchors were lost from bbox supervision')
            need(math.isfinite(row['loss']), 'Nonfinite training loss')
            parts = sum(row[key] for key in ('person_ce', 'person_face_ce', 'bbox_smooth_l1'))
            need(math.isclose(row['loss'], parts, rel_tol=1e-6, abs_tol=1e-6), 'Loss does not equal frozen component sum')
            need(isinstance(row['rng_before_loss'], dict), 'Missing sampling replay RNG')
    for step, row in enumerate(steps, 1):
        need(row['global_step'] == step and row['epoch'] == (step-1)//80+1, 'Missing/duplicate update log')
        need(math.isfinite(row['grad_norm']) and row['grad_norm'] > 0, 'No finite nonzero loss gradient at optimizer update')
        need(set(row['parameter_names_with_gradient']) == set(HEAD_NAMES), 'Unexpected gradient parameter set')
        changes = row['head_delta_l2']
        need(set(changes) == set(HEAD_NAMES) and all(math.isfinite(x) and x >= 0 for x in changes.values())
             and any(x > 0 for x in changes.values()), 'Missing/invalid actual parameter update')
        need(row['backbone_sha256'] == plan['initial_backbone_sha256'], 'Backbone changed during a step')
    return {'image_visits': len(images), 'optimizer_steps': len(steps), 'all_gt_owner_hashes_match': True,
            'all_step_gradient_and_delta_records_valid': True,
            'limitation': 'Saved logs/moments and tensor differences are cross-checked; optimizer trajectories are not independently re-executed.'}


def expected_quantized_state(state, data):
    import torch
    from decode_parameters import recover
    params, _ = recover(data)
    records = {str(r['index']): r for r in params['convolutions']}
    expected = {name: value.clone() for name, value in state.items()}
    evidence = {}
    for name in HEAD_NAMES:
        _, index, kind = name.split('.')
        scale = records[index][kind+'_scale']; bits = 16 if kind == 'weight' else 32
        value = state[name].numpy().astype(np.float64)
        rounded = np.rint(value/scale)
        low, high = -(2**(bits-1)), 2**(bits-1)-1
        clipped = (rounded < low) | (rounded > high)
        integer = np.clip(rounded, low, high).astype('<i2' if bits == 16 else '<i4')
        dequant = (integer.astype(np.float64)*scale).astype(np.float32)
        expected[name] = torch.from_numpy(dequant.copy())
        evidence[name] = {'integer_sha256': hashlib.sha256(integer.tobytes()).hexdigest(),
            'clipped_elements': int(clipped.sum()), 'elements': int(value.size),
            'max_abs_error': float(np.max(np.abs(dequant-value))),
            'rms_error': float(np.sqrt(np.mean((dequant-value)**2)))}
    return expected, evidence


def verify_quantization_report(document, expected):
    need(document['rounding'] == 'nearest_ties_to_even' and document['float_division_dtype'] == 'float64'
         and document['weight_integer_dtype'] == 'int16' and document['bias_integer_dtype'] == 'int32'
         and document['dequantization_dtype'] == 'float32', 'Quantization arithmetic declaration differs')
    names = {'rpn_cls_score1': '86', 'rpn_cls_score2': '90', 'rpn_bbox_pred': '94'}
    need({r['name'] for r in document['heads']} == set(names), 'Unexpected quantized heads')
    for row in document['heads']:
        for kind in ('weight', 'bias'):
            actual, wanted = row[kind], expected[f"convs.{names[row['name']]}.{kind}"]
            for key, value in wanted.items():
                need(actual[key] == value, 'Quantization clipping/error/hash report differs: '+row['name']+'/'+kind+'/'+key)


def original_prediction_regression(document):
    path = BASE/'artifacts/recognition-diagnosis-v3/k4-validation80/k4_stretch/predictions.json.gz'
    with gzip.open(path, 'rt') as stream:
        old = {r['id']: r['detections'] for r in json.load(stream)['real']['frames']}
    max_score, max_box, total = 0., 0., 0
    need({r['id'] for r in document['frames']} == set(old), 'Original prediction regression frame coverage differs')
    for row in document['frames']:
        previous = old[row['id']]
        need(len(row['detections']) == len(previous), 'Original K4 detection count differs from v3')
        for left, right in zip(row['detections'], previous):
            need(left['class'] == right['class'], 'Original K4 prediction class/order differs')
            max_score = max(max_score, abs(left['score']-right['score']))
            max_box = max(max_box, max(abs(a-b) for a, b in zip(left['bbox_xyxy'], right['bbox_xyxy'])))
            total += 1
    need(max_score <= 1e-7 and max_box <= 1e-6, 'Original K4 prediction regression failed')
    return {'v3_predictions_sha256': sha(path), 'detections_compared': total,
            'maximum_score_error': max_score, 'maximum_coordinate_error': max_box,
            'scope': 'Previously saved model outputs compared; no new forward inference.'}


def repair_boundary_document(document, frames):
    """Independent reconstruction of the only allowed finalization change."""
    fixed = copy.deepcopy(document); changes = []; maximum = 0.
    for row in fixed['frames']:
        frame = frames[row['id']]
        for index, item in enumerate(row['detections']):
            before = item['bbox_xyxy']; after = []
            for coordinate, value in enumerate(before):
                cap = frame['width' if coordinate % 2 == 0 else 'height']
                need(-1e-9 <= value <= cap+1e-9, 'Material coordinate excursion cannot be finalized as roundoff')
                corrected = min(float(cap), max(0., value))
                maximum = max(maximum, abs(corrected-value)); after.append(corrected)
            need(after[0] < after[2] and after[1] < after[3], 'Finalization cannot repair an empty detection')
            if before != after:
                changes.append({'mode': 'shuffled', 'id': row['id'], 'prediction_index': index,
                                'before': before, 'after': after})
                item['bbox_xyxy'] = after
    return fixed, changes, maximum


def audit_finalization(run, plan_hash, dataset, summary):
    path = run/'finalization.json'
    if not path.exists():
        need('reporting_recovery' not in summary, 'Missing declared finalization record')
        return None
    record = read(path)
    need(record['frozen_plan_sha256'] == plan_hash and record['model_updates'] == 0 and
         summary['reporting_recovery'] == record, 'Finalization scope/identity differs')
    source = BASE/'finalize_small_anchor_run.py'
    need(sha(source) == record['finalizer_sha256'], 'Finalizer source does not match recorded SHA')
    source_copy = run/'verification-source-snapshot'/source.name
    if not source_copy.exists():
        source_copy.parent.mkdir(parents=True, exist_ok=True)
        source_copy.write_bytes(source.read_bytes())
    need(sha(source_copy) == record['finalizer_sha256'], 'Preserved finalizer source differs')
    for relative, expected in record['protected_artifacts_before_after'].items():
        file = (run/relative).resolve()
        need(file.is_relative_to(run.resolve()) and sha(file) == expected, 'Protected training artifact changed during finalization')
    frames = {f['id']: f for f in dataset['frames'] if f['split'] == 'validation'}
    checked = {}
    for directory, field in [('epoch6', 'fixed_epoch6_control_changes'), ('final_quantized', 'quantized_control_changes')]:
        before = run/'controls'/directory/'shuffled-before-float-boundary-fix.json'
        after = run/'controls'/directory/'shuffled.json'
        reconstructed, changes, maximum = repair_boundary_document(read(before), frames)
        need(reconstructed == read(after), 'Finalization altered more than bounded bbox coordinates')
        need(changes == record[field], 'Declared finalization changes differ from actual file comparison')
        checked[directory] = {'before_path': str(before.relative_to(run)), 'before_sha256': sha(before),
            'after_path': str(after.relative_to(run)), 'after_sha256': sha(after), 'changed_boxes': len(changes),
            'max_abs_coordinate_change': maximum, 'frozen_boundary_allowance': 1e-9,
            'class_score_order_count_and_model_bindings_unchanged': True}
    return {'record_sha256': sha(path), 'finalizer_source_sha256': record['finalizer_sha256'],
            'finalizer_preserved_source': str(source_copy.relative_to(run)),
            'control_repairs': checked, 'protected_training_artifacts_verified': len(record['protected_artifacts_before_after']),
            'optimizer_updates_in_finalization': 0}


def run_audit(args):
    import torch
    from candidate_network import CandidateNetwork
    from train_recognition import assign_joint_targets, ground_truth
    torch.set_num_threads(2)
    plan, summary = read(args.run/'plan.json'), read(args.run/'summary.json')
    plan_hash = sha(args.run/'plan.json'); manifest_sha = sha(args.dataset)
    dataset = read(args.dataset); data_check = dataset_audit(dataset, manifest_sha)
    finalization_check = audit_finalization(args.run, plan_hash, dataset, summary)
    need(plan['epochs'] == 6 and plan['accumulate'] == 4 and plan['layout'] == 'gemm_k4', 'Training recipe differs')
    need(plan['preprocessing']['geometry'] == 'stretch', 'This frozen audit requires the declared stretch recipe')
    need(plan['manifest_sha256'] == manifest_sha and summary['plan_sha256'] == plan_hash, 'Run identity differs')
    for split, field in [('train', 'train_ids'), ('validation', 'validation_ids')]:
        need(plan[field] == sorted(f['id'] for f in dataset['frames'] if f['split'] == split), 'Frozen split coverage differs')
    source_check, source_current_matches = {}, {}
    for name, expected in plan['source_hashes'].items():
        saved, current = args.run/'source-snapshot'/name, BASE/name
        need(sha(saved) == expected, 'Original source snapshot mismatch')
        # The training runner is never imported or executed by this auditor;
        # its frozen snapshot is authoritative even after a new experiment adds
        # an initialization policy. Actual reconstruction/assignment helpers are
        # imported, so those must still match their exact historical revision.
        matches = current.is_file() and sha(current) == expected
        if name not in {'train_small_anchor.py', 'train_small_anchor_transfer.py', 'anchor_head_transfer.py'}:
            need(matches, 'Reconstruction/metric source changed; use the frozen snapshot for this audit')
        source_check[name] = expected
        source_current_matches[name] = matches
    need(sha(args.firmware) == plan['model_sha256'], 'Original firmware model identity differs')
    data = args.firmware.read_bytes()
    original = CandidateNetwork(data, 'gemm_k4').state_dict()
    original_state_id = plan.get('original_state_sha256', plan['initial_state_sha256'])
    need(state_sha(original) == original_state_id and backbone_sha(original) == plan['initial_backbone_sha256'], 'Original weights are not fresh original K4')
    need(sum(original[n].numel() for n in HEAD_NAMES) == plan['trainable_parameters'] == 89800, 'Unexpected trainable parameter count')
    refs = {}
    for name in ('original', 'small'):
        refs[name] = np.load(args.run/f'anchors-{name}.npy', allow_pickle=False)
        need(refs[name].shape == (9600, 4) and anchor_sha(refs[name]) == plan[name+'_anchor_sha256'], 'Anchor shape/hash differs')
    initialization = plan.get('initialization', 'original_k4')
    need(initialization in ('original_k4', 'analytic_log_nearest_anchor_transfer'), 'Unsupported initialization recipe')
    if initialization == 'analytic_log_nearest_anchor_transfer':
        from decode_parameters import recover
        params, _ = recover(data); provenance = plan['transfer_provenance']
        need(provenance['original_model_sha256'] == plan['model_sha256'] and provenance['input_state_sha256'] == original_state_id,
             'Transfer source model/state binding differs')
        need(provenance['original_scales'] == params['proposal_blocks'][0]['scale_candidates'] and
             provenance['original_ratios'] == params['proposal_blocks'][0]['ratio_candidates'] and
             provenance['new_scales'] == plan['small_anchor_scales'], 'Transfer scales/ratios differ from source/plan')
        need(provenance['original_anchor_sha256'] == plan['original_anchor_sha256'] and provenance['new_anchor_sha256'] == plan['small_anchor_sha256'],
             'Transfer anchor bindings differ')
        initial, initialization_check = reconstruct_analytic_transfer(original, provenance, refs['original'], refs['small'])
        need(state_sha(initial) == provenance['transferred_state_sha256'] == plan['initial_state_sha256'], 'Independent transferred initialization differs')
    else:
        initial = original
        initialization_check = {'method': 'Fresh original K4, without head transfer.', 'original_state_sha256': original_state_id,
                                'initial_state_sha256': state_sha(initial)}
    owners = {f['id']: owner_sha(assign_joint_targets(refs['small'], ground_truth(f)))
              for f in dataset['frames'] if f['split'] == 'train'}
    target_counts = {f['id']: {cls: sum(b['class'] == cls for b in f['boxes']) for cls in CLASSES}
                     for f in dataset['frames'] if f['split'] == 'train'}
    log_check = audit_training_logs(args.run, plan, owners, target_counts)
    checkpoints, state_checks = {}, {}
    for epoch in range(7):
        folder = args.run/'baseline_small' if epoch == 0 else args.run/f'epoch-{epoch:02d}'
        checkpoint = torch.load(folder/'checkpoint.pt', map_location='cpu', weights_only=False)
        need(checkpoint['plan_sha256'] == plan_hash and checkpoint['anchor_sha256'] == plan['small_anchor_sha256'], 'Checkpoint binding differs')
        need(checkpoint['completed_epoch'] == epoch and checkpoint['global_step'] == epoch*80, 'Checkpoint progress differs')
        state_checks[str(epoch)] = audit_state(initial, checkpoint['model_state'], require_update=epoch > 0)
        if epoch == 0:
            need(not state_checks['0']['changed_head_tensors'], 'Epoch0 state differs from independently reconstructed initialization')
        audit_optimizer(checkpoint['optimizer_state'], plan['optimizer_parameter_names'], checkpoint['model_state'], epoch*80)
        checkpoints[epoch] = checkpoint
    metric_checks = {}
    def evaluate_file(path, state_id, anchor_id, saved_metrics=None):
        doc = read(path)
        need(doc['plan_sha256'] == plan_hash and doc['model_state_sha256'] == state_id and doc['anchor_sha256'] == anchor_id,
             'Prediction model/anchor/plan identity differs: '+str(path))
        actual = recompute_metrics(dataset, doc, manifest_sha)
        if saved_metrics is not None:
            need(actual == saved_metrics, 'Saved metrics differ from independently checked predictions: '+str(path))
        metric_checks[str(path.relative_to(args.run))] = sha(path)
        return actual, doc
    metrics = {}
    for label in ('baseline_original', 'baseline_small'):
        anchor_name = 'original' if label.endswith('original') else 'small'
        baseline_state = original_state_id if label == 'baseline_original' else plan['initial_state_sha256']
        metrics[label], _ = evaluate_file(args.run/label/'predictions.json', baseline_state,
                                          plan[anchor_name+'_anchor_sha256'], read(args.run/label/'metrics.json'))
        need(summary[label] == metrics[label], 'Summary baseline differs')
    prediction_regression = original_prediction_regression(read(args.run/'baseline_original/predictions.json'))
    best_epoch, best_ap = 0, metrics['baseline_small']['mean_ap']
    for epoch in range(1, 7):
        directory = args.run/f'epoch-{epoch:02d}'
        metrics[epoch], _ = evaluate_file(directory/'predictions.json', state_checks[str(epoch)]['state_sha256'],
                                           plan['small_anchor_sha256'], read(directory/'metrics.json'))
        logged = summary['epochs'][epoch-1]
        need(logged['epoch'] == epoch and logged['global_step'] == epoch*80 and logged['metrics'] == metrics[epoch], 'Epoch summary differs')
        need(logged['model_state_sha256'] == state_checks[str(epoch)]['state_sha256'] and
             logged['backbone_sha256'] == plan['initial_backbone_sha256'], 'Epoch state summary differs')
        if metrics[epoch]['mean_ap'] > best_ap:
            best_epoch, best_ap = epoch, metrics[epoch]['mean_ap']
    need(summary['auxiliary_best_epoch'] == best_epoch and summary['auxiliary_best_mean_ap'] == best_ap, 'Invalid auxiliary best epoch selection')
    need(summary['optimizer_steps'] == 480, 'Final step count differs')
    quantized = torch.load(args.run/'final_quantized/checkpoint.pt', map_location='cpu', weights_only=False)
    need(quantized['plan_sha256'] == plan_hash and quantized['anchor_sha256'] == plan['small_anchor_sha256'] and quantized['host_quantized_proxy'] is True,
         'Quantized proxy binding differs')
    expected_quant, quant_errors = expected_quantized_state(checkpoints[6]['model_state'], data)
    need(set(quantized['model_state']) == set(expected_quant) and all(torch.equal(expected_quant[n], quantized['model_state'][n]) for n in expected_quant),
         'Saved proxy does not equal independently quantized epoch6 heads')
    quant_state = audit_state(initial, quantized['model_state'])
    verify_quantization_report(read(args.run/'final_quantized/quantization.json'), quant_errors)
    original_quant, original_quant_errors = expected_quantized_state(original, data)
    epoch0_quant = read(args.run/'epoch0-quantization-check.json')
    original_quant_max_error = max(float((original_quant[n]-original[n]).abs().max()) for n in original)
    if initialization == 'analytic_log_nearest_anchor_transfer':
        original_report = epoch0_quant['original_quantization_regression']
        verify_quantization_report(original_report, original_quant_errors)
        need(epoch0_quant['original_max_state_difference'] == original_quant_max_error < 1e-6, 'Original host quantization regression differs')
        initial_quant, initial_quant_errors = expected_quantized_state(initial, data)
        verify_quantization_report(epoch0_quant, initial_quant_errors)
        initial_quant_max_error = max(float((initial_quant[n]-initial[n]).abs().max()) for n in initial)
        need(epoch0_quant['max_state_difference'] == initial_quant_max_error and epoch0_quant['transfer_is_not_original_integer_identity'] is True,
             'Transferred host quantization report differs or incorrectly claims original integers')
        for head in plan['transfer_provenance']['head_parameters']:
            for kind in ('weight', 'bias'):
                actual = initial_quant_errors[head['module']+'.'+kind]['clipped_elements']
                need(actual == head['parameters'][kind]['clipped_elements'], 'Initial transfer overflow count differs from frozen provenance')
        initialization_check['initial_host_quantization_clipped_elements'] = {name: row['clipped_elements'] for name, row in initial_quant_errors.items()}
        initialization_check['initial_host_quantization_max_parameter_error'] = initial_quant_max_error
    else:
        original_report = epoch0_quant
        verify_quantization_report(original_report, original_quant_errors)
        need(epoch0_quant['max_state_difference'] == original_quant_max_error < 1e-6, 'Epoch0 host quantization regression differs')
    need(all(h['weight']['integer_elements_different_from_original'] == 0 for h in original_report['heads']), 'Original integer head weights differ')
    metrics['quantized'], _ = evaluate_file(args.run/'final_quantized/predictions.json', quant_state['state_sha256'],
                                            plan['small_anchor_sha256'], read(args.run/'final_quantized/metrics.json'))
    control_sets = {}
    validation_frames = {f['id']: f for f in dataset['frames'] if f['split'] == 'validation'}
    for name, state_id, real_metric in [('epoch0', plan['initial_state_sha256'], metrics['baseline_small']),
                                      ('epoch6', state_checks['6']['state_sha256'], metrics[6]),
                                      ('final_quantized', quant_state['state_sha256'], metrics['quantized'])]:
        directory = args.run/'controls'/name; saved = read(directory/'metrics.json')
        checked, docs = {}, {}
        for mode in CONTROLS:
            checked[mode], docs[mode] = evaluate_file(directory/(mode+'.json'), state_id, plan['small_anchor_sha256'], saved[mode])
        need(checked['real'] == real_metric, 'Real control does not reproduce corresponding saved real metrics')
        verify_cyclic_control(validation_frames, {r['id']: r['detections'] for r in docs['real']['frames']},
                             {r['id']: r['detections'] for r in docs['shuffled']['frames']}, 1)
        control_sets[name] = checked
    float_gate = acceptance(metrics[6], metrics['baseline_original'], control_sets['epoch6'])
    quant_gate = acceptance(metrics['quantized'], metrics['baseline_original'], control_sets['final_quantized'])
    need(summary['primary_fixed_epoch6']['metrics'] == metrics[6] and summary['primary_fixed_epoch6']['acceptance'] == float_gate, 'Primary epoch6 conclusion differs')
    need(summary['final_quantized']['metrics'] == metrics['quantized'] and summary['final_quantized']['acceptance'] == quant_gate, 'Quantized conclusion differs')
    need(summary['test_inference'] is False and summary['camera_deployment_verified'] is False, 'Unsupported test/deployment claim')
    report = {'passed_integrity_audit': True, 'scope': 'Saved bounded experiment, no forward inference/optimizer execution.',
        'plan_sha256': plan_hash, 'dataset': data_check, 'frozen_sources': source_check,
        'current_source_matches_frozen_revision': source_current_matches,
        'training_logs': log_check, 'checkpoint_state_checks': state_checks,
        'initialization_audit': initialization_check,
        'saved_prediction_hashes_recomputed': metric_checks, 'auxiliary_best_epoch': best_epoch,
        'original_k4_prediction_regression': prediction_regression,
        'finalization_audit': finalization_check,
        'primary_fixed_epoch6_metrics': metrics[6], 'primary_fixed_epoch6_acceptance': float_gate,
        'quantized_proxy_metrics': metrics['quantized'], 'quantized_proxy_acceptance': quant_gate,
        'independent_quantization_error_statistics': quant_errors,
        'original_head_quantization_max_parameter_error': original_quant_max_error,
        'quantization_scope': 'Parameters/rounding/error metadata independently recomputed; no claim of bitwise original bias or original/quantized detector prediction identity.',
        'float_and_host_quantized_screens_pass': float_gate['passed'] and quant_gate['passed'],
        'test_images_read': False, 'test_inference': False, 'camera_deployment_verified': False,
        'limits': ['This is a bounded host-model improvement screen, not original DSP or camera execution validation.',
                   'Head quantization emulates weight/bias storage rounding only; activations/accumulation/runtime are unresolved.',
                   'The positive-only development validation set has been examined repeatedly; the fresh test remains unopened.'],
        'verifier_sha256': sha(Path(__file__)), 'metric_helper_sha256': sha(BASE/'verify_external_baselines.py')}
    write(args.output or args.run/'independent-verification.json', report)
    print(json.dumps({'integrity_passed': True, 'primary_epoch6': float_gate,
                      'host_quantized_proxy': quant_gate, 'auxiliary_best_epoch': best_epoch}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    data = sub.add_parser('data', help='Manifest-only split/size audit, no photographs')
    data.add_argument('--dataset', type=Path, default=BASE/'private-data/recognition-v2/manifest.json')
    data.add_argument('--output', type=Path, default=BASE/'artifacts/recognition-small-anchor-preflight/data-audit.json')
    data.set_defaults(func=lambda a: (write(a.output, dataset_audit(read(a.dataset), sha(a.dataset))), print(str(a.output))))
    audit = sub.add_parser('run', help='Audit a completed local run; no model forward or test use')
    audit.add_argument('--dataset', type=Path, default=BASE/'private-data/recognition-v2/manifest.json')
    audit.add_argument('--run', type=Path, required=True)
    audit.add_argument('--firmware', type=Path, required=True)
    audit.add_argument('--output', type=Path)
    audit.set_defaults(func=run_audit)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
