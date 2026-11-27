#!/usr/bin/env python3
"""Evidence-preserving proposal/head investigation. No training or test inference.

Static bytes and shape constraints are separated from conventional decoder
hypotheses. The fixed first training IDs, original weights, and real/gray/wrong
image controls diagnose candidates; their AP never selects a deployable model.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import struct
import time

import numpy as np
import torch

from candidate_network import CandidateNetwork
from decode_parameters import recover
from extract import MODELS, decode, digest, require, write_json
from ops_investigation import analyze
from packing_investigation import SPATIAL_LAYOUTS, unpack
from pretraining_data import evaluate_detection, sha256
from proposal_hypothesis import anchors, nms
from research_paths import add_firmware_arguments, resolve_source
from train_recognition import box_iou, load_image, state_digest

MODEL = '44_hm_d_nw_sng'
CLASSES = ('person', 'person_face')
HEADS = {'score1': '86', 'score2': '90', 'bbox': '94'}
TRACKED = ('anchor_decode_investigation.py', 'candidate_network.py', 'decode_parameters.py',
           'extract.py', 'ops_investigation.py', 'packing_investigation.py',
           'proposal_hypothesis.py', 'pretraining_data.py', 'train_recognition.py')


def hashes():
    base = Path(__file__).resolve().parent
    return {name: sha256(base / name) for name in TRACKED}


def occurrences(data, needle, cap=12):
    result, cursor, count = [], 0, 0
    while True:
        offset = data.find(needle, cursor)
        if offset < 0:
            break
        if len(result) < cap:
            result.append(offset)
        count += 1
        cursor = offset + 1
    return {'count': count, 'first_offsets': result, 'offsets_truncated': count > cap}


def ordered_anchors(node, parameters, order='ratio_major', scales=None):
    ratios = parameters['ratio_candidates']
    scales = parameters['scale_candidates'] if scales is None else scales
    h, w = node['input_shapes_candidate'][0][2:]
    ref = anchors(node['anchor_base_size_candidate'], ratios, scales,
                  h, w, node['feature_stride_candidate']).reshape(h, w, len(ratios), len(scales), 4)
    if order == 'scale_major':
        ref = ref.transpose(0, 1, 3, 2, 4)
    else:
        require(order == 'ratio_major', 'Unknown anchor order')
    ref = ref.reshape(-1, 4).astype(np.float32)
    ref[:, 2:] += 1  # One continuous xyxy convention for every diagnostic.
    return ref


def inspect_tables(block):
    a = np.frombuffer(block[64:320], '<u2').astype(np.float64)
    b = np.frombuffer(block[320:576], '<u2').astype(np.float64)
    # Continuous least-squares linear approximation of exp(x), interval width
    # 1/16, interval centers -6+(i+1/2)/16, scaled by 2**15.
    x = -6 + (np.arange(128, dtype=np.float64) + .5) / 16
    h = 1 / 32
    slope = np.exp(x) * 3 * (h * np.cosh(h) - np.sinh(h)) / h**3 * 32768
    intercept = np.exp(x) * np.sinh(h) / h * 32768 - slope * x
    predicted_a = np.clip(np.trunc(slope + .5), 0, 65535)
    predicted_positive_b = np.trunc(intercept[:112] + .5)
    # This reproduces observed bytes, but the actual DSP signedness/overflow
    # and indexing implementation is unknown. Do not silently use this as ABI.
    empirical_b = np.mod(np.clip(np.trunc(intercept + .5), -65536, 65535), 65536)
    return {
        'table_a_relative_offset': 64, 'table_b_relative_offset': 320,
        'element_type_observed': '128 little-endian uint16 words each',
        'table_a_words': a.astype(int).tolist(), 'table_b_words': b.astype(int).tolist(),
        'candidate_function': 'piecewise least-squares affine approximation of exp(x)',
        'candidate_domain': [-6, 2], 'interval_width': 1/16, 'coefficient_scale': 32768,
        'formula': 'm=exp(c)*3*(h*cosh(h)-sinh(h))/h^3; b=exp(c)*sinh(h)/h-m*c; c=-6+(i+0.5)/16,h=1/32',
        'slope_exact_word_matches': int(np.count_nonzero(a == predicted_a)),
        'slope_word_count': 128,
        'positive_intercept_exact_word_matches': int(np.count_nonzero(b[:112] == predicted_positive_b)),
        'positive_intercept_word_count': 112,
        'empirical_all_intercept_word_matches': int(np.count_nonzero(b == empirical_b)),
        'empirical_intercept_encoding': 'mod(clip(trunc(32768*b+0.5), -65536, 65535), 65536)',
        'first_slope_saturation_index': int(np.flatnonzero(a == 65535)[0]),
        'index_if_argument_is_direct_real_delta': 'i=floor(16*(delta+6)); this is a conditional mathematical mapping, not recovered DSP instructions',
        'q15_if_argument_is_direct_bbox_output': 'If delta=q/32768, index=floor(q/2048)+96; signed int16 spans indices 80..111.',
        'range_warning': 'The slope table saturates from index107 (argument interval [0.6875,0.75)); direct affine evaluation cannot reproduce exp over the entire nominal [-6,2] domain. Range reduction or scaling may exist.',
        'unknowns': ['Actual DSP use, argument scaling/indexing, coefficient signedness and overflow are not observed.',
                     'Matching table-generation formulas do not validate a floating-point bbox decoder.',
                     'Negative intercept encoding is an empirical byte reproduction, not a recovered ABI.']}


def inspect_source(data, stem):
    require(digest(data) == MODELS[stem], 'Unknown model source')
    raw_graph = decode(data)
    graph = analyze(data)
    params, tensors = recover(data)
    nodes = [n for n in graph['layers'] if n['operator_id'] == 24]
    raw_nodes = {n['index']: n for n in raw_graph['layers']}
    records = []
    for node, param in zip(nodes, params['proposal_blocks']):
        raw = raw_nodes[node['index']]
        block = data[param['offset']:param['offset'] + param['bytes']]
        q = node['tail_offset']
        fields = []
        for label, relative, kind in (
            ('stride_y_candidate', 61, 'I'), ('stride_x_candidate', 65, 'I'),
            ('base_size_candidate', 69, 'I'), ('ratio_count_candidate', 73, 'I'),
            ('scale_count_candidate', 77, 'I'), ('pre_nms_top_n_candidate', 81, 'I'),
            ('post_nms_top_n_candidate', 85, 'I'), ('nms_threshold_candidate', 89, 'f'),
            ('unresolved_q93', 93, 'I'), ('unresolved_q97', 97, 'I')):
            fields.append({'label': label, 'absolute_offset': q + relative,
                           'raw_hex': data[q+relative:q+relative+4].hex(),
                           'decoded_value': struct.unpack_from('<' + kind, data, q+relative)[0]})
        records.append({'node': node, 'parameter_block': param,
                        'record_start': raw['record_start'], 'record_end': raw['record_end'],
                        'record_hex': data[raw['record_start']:raw['record_end']].hex(),
                        'raw_fields': fields, 'parameter_hex': block.hex(),
                        'output_scale_area_offset': q + 11,
                        'output_scale_area_10_floats': list(struct.unpack_from('<10f', data, q + 11)),
                        'ratio_padding_zero': block[20:32] == bytes(12),
                        'scale_padding_zero': block[52:64] == bytes(12),
                        'remaining_512_bytes': inspect_tables(block)})
    node, param = nodes[0], params['proposal_blocks'][0]
    base = ordered_anchors(node, param).reshape(-1, 25, 4)[0]
    searches = []
    for order in ('ratio_major', 'scale_major'):
        candidate = ordered_anchors(node, param, order).reshape(-1, 25, 4)[0]
        for coordinates in ('continuous', 'inclusive'):
            values = candidate.copy()
            if coordinates == 'inclusive':
                values[:, 2:] -= 1
            for layout in ('anchor_major', 'coordinate_major'):
                pattern = values if layout == 'anchor_major' else values.T
                for dtype in ('<f4', '<i2', '<i4'):
                    if dtype != '<f4' and not np.all(pattern == np.rint(pattern)):
                        continue
                    blob = pattern.astype(dtype).tobytes()
                    searches.append({'candidate': f'{order}/{coordinates}/{layout}/{dtype}',
                                     'pattern_bytes': len(blob), **occurrences(data, blob)})
                # Half-pixel coordinates have an exact integer representation
                # at scale two; retain this bounded alternative as well.
                for dtype in ('<i2', '<i4'):
                    doubled = pattern * 2
                    require(np.all(doubled == np.rint(doubled)), 'Unexpected fractional anchor coordinates')
                    blob = doubled.astype(dtype).tobytes()
                    searches.append({'candidate': f'{order}/{coordinates}/{layout}/{dtype}/coordinate_units_half_pixel',
                                     'pattern_bytes': len(blob), **occurrences(data, blob)})
    mean_std = []
    # These are bounded signature checks, not a claim that all normalization
    # encodings have been exhaustively recovered.
    for name, values in (('zero_mean4', [0., 0., 0., 0.]),
                         ('unit_std4', [1., 1., 1., 1.]),
                         ('common_rcnn_std4', [.1, .1, .2, .2])):
        blob = struct.pack('<4f', *values)
        item = {'signature': name, 'values': values,
                'full_component': occurrences(data, blob),
                'proposal_records': [], 'proposal_parameter_blocks': []}
        for record in records:
            item['proposal_records'].append(occurrences(bytes.fromhex(record['record_hex']), blob))
            item['proposal_parameter_blocks'].append(occurrences(bytes.fromhex(record['parameter_hex']), blob))
        mean_std.append(item)
    pairing = []
    for record in params['convolutions']:
        if not record['name'].startswith('rpn_cls_score'):
            continue
        weight = unpack(tensors[record['tensor_key'] + '_weight_i16_packed'], record).reshape(50, -1).astype(float)
        def cosines(left, right):
            return np.sum(left*right, axis=1) / np.maximum(np.linalg.norm(left, axis=1)*np.linalg.norm(right, axis=1), 1e-30)
        separated = cosines(weight[:25], weight[25:])
        adjacent = cosines(weight[::2], weight[1::2])
        pairing.append({'name': record['name'], 'separated_25_pair_cosines': separated.tolist(),
                        'adjacent_pair_cosines': adjacent.tolist(),
                        'separated_pair_max': float(separated.max()),
                        'bias': tensors[record['tensor_key'] + '_bias_f32'].tolist()})
    comparison_keys = ('feature_stride_candidate', 'anchor_base_size_candidate', 'ratio_count_candidate',
                       'scale_count_candidate', 'pre_nms_top_n_candidate', 'post_nms_top_n_candidate',
                       'nms_threshold_candidate', 'raw_q93', 'raw_q97')
    bbox_record = next(r for r in params['convolutions'] if r['name'] == 'rpn_bbox_pred')
    return {'model': stem, 'sha256': digest(data), 'bytes': len(data),
            'proposal_records': records, 'candidate_anchor_table_signature_search': searches,
            'bbox_mean_std_signature_search': mean_std, 'classification_pairing': pairing,
            'both_proposals_share_bbox_layer': len({n['inputs'][1]['layer'] for n in nodes}) == 1,
            'shared_bbox_layer': nodes[0]['inputs'][1]['layer'],
            'bbox_output_scale': bbox_record['output_scale'],
            'bbox_signed_i16_direct_range_conditional': [-32768*bbox_record['output_scale'], 32767*bbox_record['output_scale']],
            'smallest_square_side_after_direct_min_i16_exp_conditional': float(48*np.exp(-32768*bbox_record['output_scale'])),
            'delta_scaling_evidence': 'No independent bbox mean/std or exp-argument multiplier recovered. Proposal output scale values do not establish bbox input scaling. The exp-like table supports an exponential-family operation but does not identify which delta channels or fixed-point conversion feed it.',
            'proposal_numeric_fields_identical': all(nodes[0][k] == n[k] for n in nodes for k in comparison_keys),
            'parameter_blocks_identical': len({p['sha256'] for p in params['proposal_blocks']}) == 1,
            'base_anchor_sizes': (base[:, 2:] - base[:, :2]).tolist(),
            'smallest_square_anchor_side': float(np.min((base[:, 2:] - base[:, :2])[np.isclose(base[:, 2]-base[:, 0], base[:, 3]-base[:, 1]), 0])),
            'metadata_caveat': 'Field offsets and raw values are observed; semantic field names are conventional interpretations.'}, node, param


def geometry(dataset, node, param):
    designs = [('stored_scales', param['scale_candidates'], 'Current metadata-supported values'),
               ('new_small_scales_16px', [1, 2, 4, 8, 16], 'Counterfactual new detector design; no original-head semantic evidence'),
               ('new_small_scales_8px', [.5, 1, 2, 4, 8], 'Counterfactual new detector design; no original-head semantic evidence')]
    results = []
    for label, scales, status in designs:
        ref = ordered_anchors(node, param, scales=scales)
        center = (ref[:, :2] + ref[:, 2:]) / 2
        ref = ref[(center[:, 0] >= 0) & (center[:, 0] < 384) & (center[:, 1] >= 0) & (center[:, 1] < 256)]
        for split in ('train', 'validation'):
            rows = []
            for f in dataset['frames']:
                if f['split'] != split:
                    continue
                for box in f['boxes']:
                    gt = np.asarray(box['bbox_xyxy'], np.float32)*np.array([384/f['width'], 256/f['height']]*2)
                    overlap = box_iou(ref, gt[None])[:, 0]
                    size = gt[2:] - gt[:2]
                    rows.append({'frame_id': f['id'], 'class': box['class'],
                                 'resized_wh': size.tolist(), 'sqrt_area': float(np.sqrt(np.prod(size))),
                                 'max_anchor_iou': float(overlap.max()), 'anchors_iou_ge_05': int(np.count_nonzero(overlap >= .5))})
            summary = {}
            for cls in CLASSES:
                for size_group in ('all', 'sqrt_area_below_32'):
                    selected = [r for r in rows if r['class'] == cls and (size_group == 'all' or r['sqrt_area'] < 32)]
                    summary[cls + '/' + size_group] = {'boxes': len(selected),
                        'covered_at_iou_05': sum(r['max_anchor_iou'] >= .5 for r in selected),
                        'mean_best_iou': float(np.mean([r['max_anchor_iou'] for r in selected])) if selected else None}
            results.append({'design': label, 'scales': scales, 'status': status, 'split': split,
                            'summary': summary, 'per_box': rows})
    return {'scope': 'Geometry only on existing train/validation labels; no validation or test inference.',
            'small_definition': 'sqrt(box area) < 32 after the fixed 384x256 stretch',
            'results': results,
            'warning': 'Improved coverage with smaller scales is not proof of original model repair; old regression/classification channels may not represent these anchors.'}


def predictions_from_heads(raw, ref, frame, candidate, node):
    bbox = raw['bbox'][0]
    if candidate['bbox_grouping'] == 'anchor_major':
        delta = bbox.transpose(1, 2, 0).reshape(-1, 4)
    else:
        delta = bbox.reshape(4, 25, 16, 24).transpose(2, 3, 1, 0).reshape(-1, 4)
    sizes = ref[:, 2:] - ref[:, :2]
    centers = (ref[:, 2:] + ref[:, :2]) / 2 + delta[:, :2]*sizes
    extent = np.exp(np.clip(delta[:, 2:], -8, 8))*sizes
    boxes = np.concatenate((centers-extent/2, centers+extent/2), axis=1)
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, 384)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, 256)
    valid = np.flatnonzero(np.all(boxes[:, 2:] - boxes[:, :2] > 0, axis=1))
    scale = np.array([frame['width']/384, frame['height']/256]*2)
    heads = ('score1', 'score2') if candidate['semantic_head_mapping'] == 'score1_person' else ('score2', 'score1')
    result = []
    for cls, head in zip(CLASSES, heads):
        logits = raw[head][0].reshape(2, 25, 16, 24).transpose(2, 3, 1, 0).reshape(-1, 2)
        logits = logits - logits.max(axis=1, keepdims=True)
        prob = np.exp(logits)
        scores = (prob/prob.sum(axis=1, keepdims=True))[:, candidate['foreground_half']]
        order = valid[np.argsort(-scores[valid], kind='stable')[:node['pre_nms_top_n_candidate']]]
        keep = nms(boxes[order] - np.array([0, 0, 1, 1]), scores[order],
                   node['nms_threshold_candidate'], node['post_nms_top_n_candidate'])
        result.extend({'class': cls, 'bbox_xyxy': (boxes[i]*scale).tolist(), 'score': float(scores[i])} for i in order[keep])
    return result


def self_check():
    raw = np.arange(100*16*24, dtype=np.float32).reshape(1, 100, 16, 24)
    anchor_major = raw[0].transpose(1, 2, 0).reshape(-1, 4)
    coordinate_major = raw[0].reshape(4, 25, 16, 24).transpose(2, 3, 1, 0).reshape(-1, 4)
    require(np.array_equal(anchor_major[0], raw[0, :4, 0, 0]), 'Anchor-major channel identity failed')
    require(np.array_equal(coordinate_major[0], raw[0, [0, 25, 50, 75], 0, 0]), 'Coordinate-major channel identity failed')
    require(np.array_equal(anchor_major[25], raw[0, :4, 0, 1]), 'Spatial traversal identity failed')
    score = np.arange(50*16*24, dtype=np.float32).reshape(50, 16, 24)
    flat_score = score.reshape(2, 25, 16, 24).transpose(2, 3, 1, 0).reshape(-1, 2)
    require(np.array_equal(flat_score[0], score[[0, 25], 0, 0]), 'Class-major pairing failed')
    require(np.array_equal(flat_score[25], score[[0, 25], 0, 1]), 'Score spatial traversal failed')


def run(args):
    self_check()
    started = time.monotonic()
    source = resolve_source(args)
    args.output.mkdir(parents=True, exist_ok=True)
    source_identity = hashes()
    static = {}
    for stem in MODELS:
        data = (source / (stem + '.bin')).read_bytes()
        static[stem], node, param = inspect_source(data, stem)
        if stem == MODEL:
            target_data, target_node, target_param = data, node, param
    write_json(args.output / 'static-evidence.json', static)
    dataset = json.loads(args.dataset.read_text())
    require(dataset['classes'] == list(CLASSES), 'Unexpected class ordering')
    frames = sorted((f for f in dataset['frames'] if f['split'] == 'train'), key=lambda f: f['id'])[:args.samples]
    require(len(frames) == args.samples and args.samples >= 2, 'Insufficient fixed training frames')
    for frame in frames:
        require(sha256(args.dataset.parent / frame['path']) == frame['sha256'], 'Training image content differs')
    controls = ('real', 'gray117', 'wrong_image')
    wrong_shift = next((s for s in range(1, len(frames)) if all(frames[i]['group_id'] != frames[(i+s)%len(frames)]['group_id'] for i in range(len(frames)))), None)
    require(wrong_shift is not None, 'Cannot create different-author wrong-image control')
    candidates = [dict(anchor_order=a, bbox_grouping=b, foreground_half=f, semantic_head_mapping=m)
                  for a, b, f, m in itertools.product(('ratio_major', 'scale_major'),
                    ('anchor_major', 'coordinate_major'), (1, 0), ('score1_person', 'score2_person'))]
    for i, candidate in enumerate(candidates):
        candidate['id'] = f'decoder-{i:02d}'
    plan = {'status': 'Fixed before any inference; diagnostic only, no automatic candidate selection',
            'model': MODEL, 'model_sha256': MODELS[MODEL], 'manifest_sha256': sha256(args.dataset),
            'sources': source_identity, 'layouts': list(SPATIAL_LAYOUTS), 'candidates': candidates,
            'selected_training_frames': [{'id': f['id'], 'sha256': f['sha256'], 'group_id': f['group_id']} for f in frames],
            'selection': 'First N train IDs in lexical order; independent of predictions and labels',
            'controls': list(controls), 'wrong_image_shift': wrong_shift,
            'preprocessing_fixed': 'Existing stored RGB->Pillow L->bilinear384x256 minus117; no search here',
            'decoder_fixed': 'Continuous xyxy; std1 mean0 hypothesis; float exp clamp[-8,8]; metadata NMS/topN',
            'scope_limits': ['Only training images inferred; no validation/test inference or optimizer steps.',
                             'The factorial candidates arise from unresolved 25x4 channel ordering and unlabeled head outputs.',
                             'Small anchors are geometry-only counterfactuals, never used as an old-weight inference candidate.',
                             'No bbox mean/std candidate is invented from an AP improvement.',
                             'Proposal LUT is analyzed statically; its DSP arithmetic is not silently substituted.']}
    write_json(args.output / 'predeclared-plan.json', plan)
    write_json(args.output / 'geometry.json', geometry(dataset, target_node, target_param))
    if args.static_only:
        print(json.dumps({'static_complete': True, 'output': str(args.output)}))
        return
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    small_dataset = {'classes': list(CLASSES), 'frames': frames}
    all_results, head_evidence, frozen = [], [], []
    for layout in SPATIAL_LAYOUTS:
        network = CandidateNetwork(target_data, layout).eval()
        for parameter in network.parameters():
            parameter.requires_grad_(False)
        initial_digest = state_digest(network.state_dict())
        captured, hooks = {}, []
        for name, key in HEADS.items():
            def capture(_module, _inputs, output, name=name):
                captured[name] = output.detach().cpu().numpy().copy()
            hooks.append(network.convs[key].register_forward_hook(capture))
        raw_images = []
        with torch.inference_mode():
            for frame in frames:
                network(load_image(args.dataset.parent, frame))
                raw_images.append({key: value.copy() for key, value in captured.items()})
            network(torch.zeros(1, 1, 256, 384))
            gray = {key: value.copy() for key, value in captured.items()}
        require(state_digest(network.state_dict()) == initial_digest, 'Original weights changed')
        frozen.append({'layout': layout, 'state_sha256': initial_digest, 'unchanged_after_inference': True})
        for hook in hooks:
            hook.remove()
        raw_file = args.output / f'raw-heads-{layout}.npz'
        np.savez_compressed(raw_file, **{f'image_{i}_{key}': value for i, raw in enumerate(raw_images) for key, value in raw.items()},
                            **{'gray_' + key: value for key, value in gray.items()})
        for i, (frame, raw) in enumerate(zip(frames, raw_images)):
            head_evidence.append({'layout': layout, 'frame_id': frame['id'], 'wrong_image_id': frames[(i+wrong_shift)%len(frames)]['id'],
                'raw_heads': {key: {'shape': list(value.shape), 'sha256': digest(value.tobytes()),
                    'minimum': float(value.min()), 'maximum': float(value.max()),
                    'mean_abs_difference_from_gray': float(np.abs(value-gray[key]).mean()),
                    'rms_difference_from_gray': float(np.sqrt(np.square(value-gray[key]).mean()))} for key, value in raw.items()}})
        for candidate in candidates:
            ref = ordered_anchors(target_node, target_param, candidate['anchor_order'])
            metrics = {}
            for control in controls:
                rows = []
                for i, frame in enumerate(frames):
                    raw = raw_images[i] if control == 'real' else gray if control == 'gray117' else raw_images[(i+wrong_shift)%len(frames)]
                    rows.append({'id': frame['id'], 'detections': predictions_from_heads(raw, ref, frame, candidate, target_node)})
                prediction = {'dataset_manifest_sha256': plan['manifest_sha256'], 'model_state_sha256': initial_digest,
                              'source_model_sha256': MODELS[MODEL], 'layout_hypothesis': layout,
                              'candidate': candidate, 'control': control, 'frames': rows}
                metrics[control] = evaluate_detection(small_dataset, prediction, 'train')
                # Keep every candidate's evidence, not only favorable outcomes.
                destination = args.output / 'predictions' / layout
                destination.mkdir(parents=True, exist_ok=True)
                write_json(destination / f"{candidate['id']}-{control}.json", prediction)
            all_results.append({'layout': layout, 'candidate': candidate, 'metrics': metrics,
                'real_minus_gray_mean_ap': metrics['real']['mean_ap'] - metrics['gray117']['mean_ap'],
                'real_minus_wrong_mean_ap': metrics['real']['mean_ap'] - metrics['wrong_image']['mean_ap']})
        del network
        print(json.dumps({'layout_complete': layout, 'train_images': len(frames), 'decoder_candidates': len(candidates),
                          'controls': list(controls), 'elapsed_seconds': time.monotonic()-started}), flush=True)
    require(hashes() == source_identity, 'Source changed during diagnosis')
    write_json(args.output / 'head-response-evidence.json', head_evidence)
    result = {'status': 'Original-weight training-subset diagnosis only; no chosen repair',
              'model_sha256': MODELS[MODEL], 'manifest_sha256': plan['manifest_sha256'],
              'source_hashes': source_identity, 'fixed_train_samples': len(frames), 'frozen_weight_checks': frozen,
              'test_inference_performed': False, 'validation_inference_performed': False, 'optimizer_steps': 0,
              'candidate_results': all_results, 'elapsed_seconds': time.monotonic()-started,
              'observations': [
                  'Proposal blocks contain 64 bytes of ratio/scale data and padding plus 512 bytes of exp-like affine coefficient tables.',
                  'For both pinned models all 128 slope words and all 112 nonnegative intercept words match the explicit formula exactly.',
                  'All known proposal records have identical numeric anchor fields and blocks within their model; no per-class small-anchor field was found.',
                  'The 44 model proposals share layer94 bbox deltas. Class-major paired classification channels have independent weight-negation and explicit-reshape support.',
                  'No matching complete 25x4 candidate anchor signature was found in the bounded full-component search.',
                  'No recovered bbox mean/std or exp-argument multiplier justifies replacing the current decoder with a tuned scale.',
                  'Original-model correspondence remains unresolved; no decoder or anchor candidate is selected by this diagnosis.'],
              'limits': plan['scope_limits'] + ['Small diagnostic subset metrics are not generalization estimates.',
                  'A candidate beating gray or wrong images is not original-DSP equivalence.',
                  'Unresolved upstream preprocessing and convolution packing can dominate every decoder comparison.']}
    write_json(args.output / 'report.json', result)
    print(json.dumps({'completed': True, 'output': str(args.output), 'candidate_conditions': len(all_results)*len(controls),
                      'optimizer_steps': 0, 'test_inference_performed': False}), flush=True)


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    add_firmware_arguments(parser)
    parser.add_argument('--dataset', type=Path, default=base/'private-data/recognition-v2/manifest.json')
    parser.add_argument('--output', type=Path, default=base/'artifacts/recognition-diagnosis/anchors')
    parser.add_argument('--samples', type=int, default=8)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--static-only', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
