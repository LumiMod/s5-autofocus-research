#!/usr/bin/env python3
"""Explicit host initialization transferring original K4 heads to smaller anchors.

This changes head values/channel semantics. It is NOT the original weight order,
training, firmware export, or proof of the camera's proposal/DSP implementation.
Mappings depend only on the fixed original/new anchor scales, never on images,
annotations, validation AP or learned weights.
"""
import argparse
import copy
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from candidate_network import CandidateNetwork
from decode_parameters import recover
from extract import MODELS, require, write_json
from ops_investigation import analyze
from packing_investigation import pack, unpack
from pretraining_data import sha256
from proposal_hypothesis import anchors
from research_paths import add_firmware_arguments, resolve_source
from train_recognition import Detector, HEADS, MODEL, state_digest


def array_digest(value):
    return hashlib.sha256(np.asarray(value, dtype='<f4').tobytes()).hexdigest()


def nearest_log_indices(original_scales, new_scales):
    old = np.asarray(original_scales, np.float64)
    new = np.asarray(new_scales, np.float64)
    require(np.isfinite(old).all() and np.isfinite(new).all() and np.all(old > 0) and np.all(new > 0),
            'Anchor scales must be finite and positive')
    result = []
    for value in new:
        distances = np.abs(np.log(old / value))
        # Resolve numerical representations of an exact logarithmic tie by index.
        result.append(int(np.flatnonzero(np.abs(distances-distances.min()) <= 1e-12)[0]))
    return np.asarray(result, np.int64)


def geometry_affine(old, new):
    """d_new = gain*d_old + offset for unclipped continuous xyxy decode."""
    old = np.asarray(old, np.float64); new = np.asarray(new, np.float64)
    require(old.shape == new.shape and old.shape[-1] == 4, 'Anchor shapes differ')
    old_size, new_size = old[..., 2:]-old[..., :2], new[..., 2:]-new[..., :2]
    require(np.isfinite(old).all() and np.isfinite(new).all() and
            np.all(old_size > 0) and np.all(new_size > 0), 'Invalid anchor geometry')
    old_center, new_center = (old[..., 2:]+old[..., :2])/2, (new[..., 2:]+new[..., :2])/2
    ratio = old_size/new_size
    return (np.concatenate((ratio, np.ones_like(ratio)), axis=-1),
            np.concatenate(((old_center-new_center)/new_size, np.log(ratio)), axis=-1))


def transfer_plan(data, newanchors):
    require(hashlib.sha256(data).hexdigest() == MODELS[MODEL], 'Expected pinned original 44 model')
    params, _ = recover(data)
    nodes = [n for n in analyze(data)['layers'] if n['operator_id'] == 24]
    require(len(nodes) == len(params['proposal_blocks']) == 2, 'Unexpected proposal topology')
    first = params['proposal_blocks'][0]
    ratios, old_scales = first['ratio_candidates'], first['scale_candidates']
    require(len(ratios) == len(old_scales) == 5, 'Expected original 25 slots')
    for n, p in zip(nodes, params['proposal_blocks']):
        require(n['anchor_base_size_candidate'] == 16 and n['feature_stride_candidate'] == [16, 16],
                'Unexpected proposal geometry')
        require(p['ratio_candidates'] == ratios and p['scale_candidates'] == old_scales,
                'Two class proposals differ')
    oldanchors = anchors(16, ratios, old_scales, 16, 24, [16, 16]).astype(np.float32)
    oldanchors[:, 2:] += 1
    newanchors = np.asarray(newanchors, np.float32)
    require(newanchors.shape == oldanchors.shape == (9600, 4), 'Keep original grid/25 slots')
    old_sizes = (oldanchors[:25, 2:]-oldanchors[:25, :2]).reshape(5, 5, 2)
    new_sizes = (newanchors[:25, 2:]-newanchors[:25, :2]).reshape(5, 5, 2)
    base_sizes = old_sizes[:, :1, :].astype(np.float64)/old_scales[0]
    inferred_scales = new_sizes/base_sizes
    new_scales = inferred_scales[0, :, 0]
    require(np.allclose(inferred_scales, new_scales[None, :, None], rtol=0, atol=1e-7),
            'New ratios or slot order differ from original')
    expected = anchors(16, ratios, new_scales, 16, 24, [16, 16]).astype(np.float32)
    expected[:, 2:] += 1
    require(np.array_equal(newanchors, expected), 'New anchors changed centers, grid or slot order')
    scale_map = nearest_log_indices(old_scales, new_scales)
    slot_map = np.asarray([r*5+int(i) for r in range(5) for i in scale_map], np.int64)
    row_map = (np.arange(384)[:, None]*25 + slot_map[None, :]).reshape(-1)
    gains, offsets = geometry_affine(oldanchors[row_map], newanchors)
    require(np.allclose(gains.reshape(384, 25, 4), gains[:25], rtol=0, atol=1e-12) and
            np.allclose(offsets.reshape(384, 25, 4), offsets[:25], rtol=0, atol=1e-12),
            'Spatially varying affine cannot be represented by a shared 1x1 head')
    class_map = np.concatenate((slot_map, slot_map+25))
    bbox_map = (slot_map[:, None]*4+np.arange(4)[None, :]).reshape(-1)
    return dict(schema_version=1, initialization='nearest_log_anchor_semantic_transfer_from_original_K4',
        original_model_sha256=MODELS[MODEL], original_weight_order=False,
        original_scales=old_scales, new_scales=new_scales.tolist(), original_ratios=ratios,
        nearest_log_tie_policy='lower original index; absolute log-distance tie tolerance 1e-12',
        old_scale_index_by_new_scale=scale_map.tolist(), old_slot_by_new_slot=slot_map.tolist(),
        class_bg_fg_channel_map=class_map.tolist(), bbox_xywh_channel_map=bbox_map.tolist(),
        original_anchor_sha256=array_digest(oldanchors), new_anchor_sha256=array_digest(newanchors),
        bbox_gain_by_slot=gains[:25].tolist(), bbox_offset_by_slot=offsets[:25].tolist(),
        row_mapping_sha256=hashlib.sha256(row_map.astype('<i8').tobytes()).hexdigest(),
        mapped_rows=9600, class_channels=50, bbox_channels=100, grid=[16, 24], slots=25,
        formula=dict(dx='(cx_old-cx_new)/w_new + (w_old/w_new)*dx_old',
                     dy='(cy_old-cy_new)/h_new + (h_old/h_new)*dy_old',
                     dw='dw_old + log(w_old/w_new)', dh='dh_old + log(h_old/h_new)'),
        semantics='Unclipped boxes and unbounded finite exp; existing +/-8 delta clamp can break equality when reached.',
        data_selection='No images, annotations, AP or learned checkpoint used to select mappings.')


def transferred_parameters(weight, bias, name, plan):
    """Compute exact host affine in float64; application explicitly casts to f32."""
    weight, bias = np.asarray(weight, np.float64), np.asarray(bias, np.float64)
    require(weight.ndim == 4 and weight.shape[2:] == (1, 1), 'Only original 1x1 heads')
    if name in ('person', 'person_face'):
        indices = np.asarray(plan['class_bg_fg_channel_map'])
        require(weight.shape[0] == len(bias) == 50, 'Unexpected class head')
        return weight[indices].copy(), bias[indices].copy()
    require(name == 'bbox' and weight.shape[0] == len(bias) == 100, 'Unexpected bbox head')
    indices = np.asarray(plan['bbox_xywh_channel_map'])
    gain = np.asarray(plan['bbox_gain_by_slot']).reshape(-1)
    offset = np.asarray(plan['bbox_offset_by_slot']).reshape(-1)
    return weight[indices]*gain[:, None, None, None], bias[indices]*gain+offset


def quantization_range(value, scale, bits):
    value = np.asarray(value, np.float64)
    rounded = np.rint(value/scale)
    low, high = -(2**(bits-1)), 2**(bits-1)-1
    clipped = (rounded < low) | (rounded > high)
    integer = np.clip(rounded, low, high).astype(np.int16 if bits == 16 else np.int32)
    reconstructed = (integer.astype(np.float64)*scale).astype(np.float32)
    axes = tuple(range(1, value.ndim))
    row_count = np.sum(clipped, axis=axes) if axes else clipped.astype(np.int64)
    return dict(bits=bits, original_scale=scale, elements=int(value.size),
        rounded_integer_min=float(rounded.min()), rounded_integer_max=float(rounded.max()),
        integer_min=low, integer_max=high, clipped_elements=int(clipped.sum()),
        clipped_by_output_channel=row_count.tolist(),
        all_values_within_original_range=not bool(clipped.any()),
        max_abs_round_and_clip_error=float(np.max(np.abs(reconstructed-value))),
        rms_round_and_clip_error=float(np.sqrt(np.mean((reconstructed-value)**2)))), integer


def apply_transfer(detector, data, newanchors):
    """Mutate a pristine original K4 Detector in place; return full provenance.

    Refuses previously trained/already-transferred states. Preserves backbone,
    architecture, requires_grad flags and global RNG; sets detector.anchors to the
    supplied small anchors. It does not quantize/clamp transferred FP32 values.
    """
    require(detector.layout == 'gemm_k4', 'Transfer requires explicit K4 initialization')
    plan = transfer_plan(data, newanchors)
    before = state_digest(detector.model.state_dict())
    with torch.random.fork_rng(devices=[]):
        original = CandidateNetwork(data, 'gemm_k4')
    require(before == state_digest(original.state_dict()), 'Transfer requires pristine original K4 state')
    params, _ = recover(data)
    records = {str(r['index']): r for r in params['convolutions']}
    states_before = {k: v.detach().clone() for k, v in detector.model.state_dict().items()}
    flags = {k: p.requires_grad for k, p in detector.model.named_parameters()}
    head_rows, changed_keys = [], set()
    with torch.no_grad():
        for name, index in HEADS.items():
            module = detector.model.convs[index]
            w, b = transferred_parameters(module.weight.detach().numpy(), module.bias.detach().numpy(), name, plan)
            row = dict(head=name, module='convs.'+index, parameters={})
            for kind, value, bits in (('weight', w, 16), ('bias', b, 32)):
                parameter = getattr(module, kind)
                value32 = value.astype(np.float32)
                require(list(value32.shape) == list(parameter.shape) and np.isfinite(value32).all(), 'Head shape/nonfinite change')
                stats, integer = quantization_range(value32, records[index][kind+'_scale'], bits)
                if kind == 'weight':
                    require(np.array_equal(unpack(pack(integer, records[index], 'gemm_k4'), records[index], 'gemm_k4'), integer),
                            'Transferred head integer packing mismatch')
                row['parameters'][kind] = stats
                parameter.copy_(torch.from_numpy(value32))
                changed_keys.add('convs.'+index+'.'+kind)
            head_rows.append(row)
    after = detector.model.state_dict()
    require(set(after) == set(states_before) and all(after[k].shape == states_before[k].shape for k in after),
            'Architecture changed')
    require(all(torch.equal(after[k], states_before[k]) for k in after if k not in changed_keys), 'Backbone changed')
    require(flags == {k: p.requires_grad for k, p in detector.model.named_parameters()}, 'Gradient flags changed')
    detector.anchors = np.asarray(newanchors, np.float32).copy()
    detector.metadata = {**detector.metadata, 'anchor_scales': plan['new_scales'],
                         'head_initialization': plan['initialization'], 'original_weight_order': False}
    return {**plan, 'input_state_sha256': before, 'transferred_state_sha256': state_digest(after),
        'head_parameters': head_rows, 'backbone_bitwise_unchanged': True, 'architecture_and_shapes_unchanged': True,
        'requires_grad_flags_unchanged': True, 'float32_values_clipped': False,
        'original_integer_ranges_fit': all(p['all_values_within_original_range'] for h in head_rows for p in h['parameters'].values()),
        'firmware_written': False, 'optimizer_steps': 0,
        'limitations':['Explicit migrated initialization; not unchanged original weights or original ordering.',
            'Mapped old slots may repeat; scores/boxes equal those selected old slots, not the full original prediction set.',
            'Smallest slots can exceed original integer ranges; quantization report is separate from float transfer.',
            'Original DSP arithmetic and physical-camera execution remain unverified.']}


def unclipped_decode(ref, delta):
    ref, delta = np.asarray(ref, np.float64), np.asarray(delta, np.float64)
    size = ref[:, 2:]-ref[:, :2]
    center = (ref[:, :2]+ref[:, 2:])/2+delta[:, :2]*size
    extent = np.exp(delta[:, 2:])*size
    require(np.isfinite(extent).all(), 'Synthetic exponent overflow')
    return np.concatenate((center-extent/2, center+extent/2), axis=1)


def synthetic_verify(data, scales):
    torch.set_num_threads(1)
    d = Detector(data, 'gemm_k4')
    oldanchors = d.anchors.copy()
    params, _ = recover(data)
    newanchors = anchors(16, params['proposal_blocks'][0]['ratio_candidates'], scales, 16, 24, [16, 16]).astype(np.float32)
    newanchors[:, 2:] += 1
    original = {name: (d.model.convs[index].weight.detach().numpy().copy(), d.model.convs[index].bias.detach().numpy().copy())
                for name, index in HEADS.items()}
    rng = torch.get_rng_state().clone()
    provenance = apply_transfer(d, data, newanchors)
    require(torch.equal(rng, torch.get_rng_state()), 'Transfer consumed global torch RNG')
    slot_map = np.asarray(provenance['old_slot_by_new_slot'])
    row_map = (np.arange(384)[:, None]*25+slot_map[None, :]).reshape(-1)
    require(np.array_equal(row_map//25, np.arange(9600)//25), 'Mapping moved a spatial location')
    require(np.array_equal((row_map % 25)//5, (np.arange(9600) % 25)//5), 'Mapping crossed a ratio')
    require(nearest_log_indices([2., 8.], [4.]).tolist() == [0], 'Log-distance tie fixture failed')
    nonuniform = {}
    class_map = np.asarray(provenance['class_bg_fg_channel_map'])
    bbox_map = np.asarray(provenance['bbox_xywh_channel_map'])
    raw_bbox = None
    for name, index in HEADS.items():
        w, b = original[name]
        grid = np.arange(w.shape[1]*384, dtype=np.float64).reshape(w.shape[1], 384)
        feature = np.sin(grid*.037)*.07+np.cos(grid*.019)*.03
        old = w[:, :, 0, 0].astype(np.float64)@feature+b[:, None]
        exact_w, exact_b = transferred_parameters(w, b, name, provenance)
        exact = exact_w[:, :, 0, 0]@feature+exact_b[:, None]
        actual = d.model.convs[index](torch.from_numpy(feature.astype(np.float32).reshape(1, w.shape[1], 16, 24)))
        actual = actual.detach().numpy().reshape(-1, 384).astype(np.float64)
        if name != 'bbox':
            expected = old[class_map]
            require(np.array_equal(exact, expected), 'Class bg/fg channel semantic mapping failed')
        else:
            gain = np.asarray(provenance['bbox_gain_by_slot']).reshape(-1, 1)
            offset = np.asarray(provenance['bbox_offset_by_slot']).reshape(-1, 1)
            expected = old[bbox_map]*gain+offset
            require(np.allclose(exact, expected, rtol=1e-13, atol=1e-13), 'Exact affine head relation failed')
            old_rows = old.T.reshape(-1, 4)
            exact_rows, actual_rows = exact.T.reshape(-1, 4), actual.T.reshape(-1, 4)
            old_boxes = unclipped_decode(oldanchors[row_map], old_rows[row_map])
            exact_boxes, actual_boxes = unclipped_decode(newanchors, exact_rows), unclipped_decode(newanchors, actual_rows)
            require(np.allclose(exact_boxes, old_boxes, rtol=1e-12, atol=1e-10), 'Exact box affine relation failed')
            require(np.allclose(actual_boxes, old_boxes, rtol=2e-5, atol=5e-4), 'Float32 transfer numerical error too large')
            raw_bbox = dict(exact_box_max_abs_error=float(np.abs(exact_boxes-old_boxes).max()),
                            float32_box_max_abs_error=float(np.abs(actual_boxes-old_boxes).max()),
                            old_delta_max_abs=float(np.abs(old_rows).max()), new_delta_max_abs=float(np.abs(actual_rows).max()))
        nonuniform[name] = dict(shape=list(actual.shape), feature_std=float(feature.std()),
            exact_relation_max_abs_error=float(np.abs(exact-expected).max()),
            float32_raw_max_abs_error=float(np.abs(actual-expected).max()))
    # Separate shifted-center fixture ensures translation terms are not omitted.
    oa = np.array([[2., 3., 22., 33.], [1., 9., 8., 18.]])
    na = np.array([[-1., 4., 9., 19.], [2., -3., 16., 15.]])
    delta = np.array([[.1, -.2, .3, -.4], [-.31, .17, -.2, .1]])
    gain, offset = geometry_affine(oa, na)
    require(np.allclose(unclipped_decode(oa, delta), unclipped_decode(na, delta*gain+offset), rtol=1e-13, atol=1e-12),
            'Shifted-center affine fixture failed')
    # Applying twice must never silently repeat gains/offsets.
    try:
        apply_transfer(d, data, newanchors)
    except ValueError as error:
        require('pristine' in str(error), 'Unexpected duplicate-transfer rejection')
    else:
        raise ValueError('Repeated transfer was not rejected')
    d.close()
    return dict(passed=True, provenance=provenance, original_head_nonuniform_feature_checks=nonuniform,
        decoded_box_checks=raw_bbox, mapped_rows_checked=9600, ratio_and_spatial_mapping_preserved=True,
        logarithmic_tie_fixture_passed=True, shifted_center_fixture_passed=True,
        repeated_transfer_rejected=True, global_torch_rng_preserved=True,
        scope=dict(image_reads=0, annotations_read=0, network_backbone_forwards=0,
                   synthetic_original_head_forwards=3, optimizer_steps=0, firmware_writes=0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_firmware_arguments(parser)
    parser.add_argument('--scales', type=float, nargs=5, default=[1, 2, 4, 8, 16])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Preserve existing evidence; choose a new output')
    files = [Path(__file__).name, 'candidate_network.py', 'train_recognition.py', 'decode_parameters.py',
             'ops_investigation.py', 'packing_investigation.py', 'proposal_hypothesis.py', 'extract.py']
    base = Path(__file__).resolve().parent
    hashes = {name: sha256(base/name) for name in files}
    result = synthetic_verify((resolve_source(args)/(MODEL+'.bin')).read_bytes(), args.scales)
    require(all(sha256(base/n) == h for n, h in hashes.items()), 'Bound source changed')
    result['source_hashes'] = hashes
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print({'passed': result['passed'], 'output': str(args.output), 'sha256': sha256(args.output),
           'original_integer_ranges_fit': result['provenance']['original_integer_ranges_fit'],
           'decoded_box_checks': result['decoded_box_checks']})


if __name__ == '__main__':
    main()
