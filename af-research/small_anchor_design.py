#!/usr/bin/env python3
"""Train-annotation-only anchor design and in-memory parameter patch audit.

No photograph, model forward, optimizer, validation selection, test prediction,
camera access or modified firmware file is involved. The original architecture,
ratios, 16-pixel grid and 25 anchors per location stay fixed.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import struct

import numpy as np

from decode_parameters import recover
from extract import MODELS, decode, digest, require, write_json
from ops_investigation import analyze
from pretraining_data import sha256
from proposal_hypothesis import anchors

BASE = Path(__file__).resolve().parent
MODEL = '44_hm_d_nw_sng'
MANIFEST_SHA = 'af5459ddebd98947b154580cd176e7846abcceffe5539ac6f0001f6a485a3fc8'
DESIGNS = {'original': [3., 6., 9., 16., 32.],
           'small16_keep256': [1., 2., 4., 8., 16.],
           'small8_keep128': [.5, 1., 2., 4., 8.]}


def boxes_for_design(ratios, scales):
    result = anchors(16, ratios, scales, 16, 24, [16, 16])
    result[:, 2:] += 1
    require(result.shape == (9600, 4), 'Changed anchor count')
    return result


def overlap(boxes, target):
    intersection = np.maximum(np.minimum(boxes[:, 2:], target[2:])-
                              np.maximum(boxes[:, :2], target[:2]), 0).prod(1)
    areas = (boxes[:, 2:]-boxes[:, :2]).prod(1)
    target_area = np.prod(target[2:]-target[:2])
    return intersection/np.maximum(areas+target_area-intersection, 1e-30)


def delta_for(anchor, target):
    size = anchor[2:]-anchor[:2]; gt_size = target[2:]-target[:2]
    delta = np.r_[((target[2:]+target[:2])-(anchor[2:]+anchor[:2]))/(2*size),
                  np.log(gt_size/size)]
    center = (anchor[2:]+anchor[:2])/2+delta[:2]*size
    extent = np.exp(delta[2:])*size
    require(np.max(np.abs(np.r_[center-extent/2, center+extent/2]-target)) < 1e-9,
            'Regression delta inverse differs from GT')
    return delta


def percentile(values):
    return None if not values else dict(zip(('p50', 'p90', 'p95', 'max'),
                                            np.percentile(values, [50, 90, 95, 100]).tolist()))


def patch_scale_arrays(data, scales):
    """Return modified bytes in memory only; original input must be pinned."""
    require(digest(data) == MODELS[MODEL], 'Patch audit requires the pinned original44 component')
    require(len(scales) == 5 and all(np.isfinite(x) and x > 0 for x in scales), 'Invalid five scales')
    original_graph = decode(data); before, tensors = recover(data)
    original_analysis = analyze(data)
    blocks = before['proposal_blocks']; require(len(blocks) == 2, 'Expected two proposal blocks')
    work = bytearray(data); windows = []
    replacement = struct.pack('<5f', *scales)
    for block in blocks:
        start = block['offset']+32
        windows.append(dict(proposal_layer=block['layer'], offset=start, offset_hex=hex(start),
                            bytes=20, before_hex=data[start:start+20].hex(), after_hex=replacement.hex()))
        work[start:start+20] = replacement
    modified = bytes(work)
    after, recovered = recover(modified)
    require(decode(modified) == original_graph, 'Modified graph or header structure')
    new_analysis = analyze(modified)
    require({k: v for k, v in new_analysis.items() if k != 'source_sha256'} ==
            {k: v for k, v in original_analysis.items() if k != 'source_sha256'}, 'Modified operator shapes')
    require(after['convolutions'] == before['convolutions'], 'Modified convolution parameter descriptors')
    require(set(recovered) == set(tensors) and all(np.array_equal(recovered[k], v) for k, v in tensors.items()),
            'Modified any recovered input/weight/bias array')
    require(all(p['scale_candidates'] == list(scales) for p in after['proposal_blocks']), 'Scale reparse differs')
    require(all(a['ratio_candidates'] == b['ratio_candidates'] for a, b in zip(after['proposal_blocks'], blocks)),
            'Modified ratios')
    changed = [int(x) for x in np.flatnonzero(np.frombuffer(data, np.uint8) != np.frombuffer(modified, np.uint8))]
    allowed = {offset for w in windows for offset in range(w['offset'], w['offset']+20)}
    require(set(changed) <= allowed, 'Changed bytes outside the two20-byte scale windows')
    restore = bytearray(modified)
    for w in windows:
        restore[w['offset']:w['offset']+20] = bytes.fromhex(w['before_hex'])
    require(bytes(restore) == data, 'Inverse patch did not restore every original byte')
    return modified, dict(original_sha256=digest(data), in_memory_modified_sha256=digest(modified),
        restored_sha256=digest(bytes(restore)), modified_bin_saved=False, patch_windows=windows,
        actual_changed_byte_count=len(changed), actual_changed_offsets=changed,
        graph_headers_and_operator_shapes_identical=True, recovered_input_weights_biases_identical=True,
        ratios_unchanged=True, both_scale_arrays_reparsed=list(scales), complete_byte_inverse_identical=True,
        native_model_allowlist_accepts_modified=digest(modified) in MODELS.values(),
        interpretation='A byte-scope/roundtrip audit only. Patched candidate identity rejection is expected; '
                       'training must load the pinned original and explicitly override host anchors. '
                       'No update-package integrity, DSP execution or camera deployment is established.')


def geometry(frames, ratios, scales):
    ref = boxes_for_design(ratios, scales)
    center = (ref[:, :2]+ref[:, 2:])/2
    full = (ref[:, :2] >= [0, 0]).all(1) & (ref[:, 2:] <= [384, 256]).all(1)
    clipped = np.clip(ref.reshape(-1, 2, 2), [0, 0], [384, 256]).reshape(-1, 4)
    size = ref[:25, 2:]-ref[:25, :2]
    rows = []
    for frame in frames:
        scale = np.asarray([384/frame['width'], 256/frame['height']]*2, np.float64)
        for index, item in enumerate(frame['boxes']):
            target = np.asarray(item['bbox_xyxy'], np.float64)*scale
            gt_center = (target[:2]+target[2:])/2; area_side = float(np.sqrt(np.prod(target[2:]-target[:2])))
            overlaps = overlap(ref, target); best = int(np.argmax(overlaps))
            oracle = np.concatenate((gt_center-size/2, gt_center+size/2), axis=1)
            oracle_best = float(overlap(oracle, target).max())
            require(oracle_best+1e-12 >= overlaps[best], 'Free-center oracle is not an upper bound')
            nearest_x = 8+16*np.clip(np.rint((gt_center[0]-8)/16), 0, 23)
            nearest_y = 8+16*np.clip(np.rint((gt_center[1]-8)/16), 0, 15)
            d = delta_for(ref[best], target)
            fully_inside_best = float(overlap(ref[full], target).max()) if np.any(full) else 0.
            boundary_distance = float(min(target[0], target[1], 384-target[2], 256-target[3]))
            rows.append(dict(frame_id=frame['id'], gt_index=index, target_class=item['class'],
                bbox_reference=target.tolist(), sqrt_area=area_side,
                size_group='small_lt32' if area_side < 32 else 'medium_32_to_lt96' if area_side < 96 else 'large_ge96',
                boundary_distance=boundary_distance, boundary_within_stride16=boundary_distance < 16,
                max_iou=float(overlaps[best]), anchors_ge_03=int(np.count_nonzero(overlaps >= .3)),
                anchors_ge_05=int(np.count_nonzero(overlaps >= .5)), best_anchor_index=best,
                best_anchor=ref[best].tolist(), best_anchor_extends_outside_image=not bool(full[best]),
                best_fitting_delta=d.tolist(), fully_inside_anchor_max_iou=fully_inside_best,
                clipped_anchor_max_iou=float(overlap(clipped, target).max()),
                free_center_same25shapes_max_iou=oracle_best,
                stride_center_gap_iou=oracle_best-float(overlaps[best]),
                nearest_grid_center_error_xy=(np.asarray([nearest_x, nearest_y])-gt_center).tolist()))
    summary = {}
    for cls in ('person', 'person_face'):
        for size_group in ('all', 'small_lt32', 'medium_32_to_lt96', 'large_ge96'):
            subset = [r for r in rows if r['target_class'] == cls and
                      (size_group == 'all' or r['size_group'] == size_group)]
            n = len(subset)
            summary[cls+'/'+size_group] = dict(gt=n,
                max_iou=percentile([r['max_iou'] for r in subset]),
                covered_ge_03=sum(r['max_iou'] >= .3 for r in subset),
                covered_ge_05=sum(r['max_iou'] >= .5 for r in subset),
                coverage_ge_05=None if not n else sum(r['max_iou'] >= .5 for r in subset)/n,
                free_center_covered_ge_05=sum(r['free_center_same25shapes_max_iou'] >= .5 for r in subset),
                free_center_pass_but_grid_fail_ge_05=sum(r['free_center_same25shapes_max_iou'] >= .5 > r['max_iou'] for r in subset),
                fully_inside_anchors_covered_ge_05=sum(r['fully_inside_anchor_max_iou'] >= .5 for r in subset),
                clipped_anchors_covered_ge_05=sum(r['clipped_anchor_max_iou'] >= .5 for r in subset),
                best_anchor_crosses_boundary=sum(r['best_anchor_extends_outside_image'] for r in subset),
                gt_near_boundary=sum(r['boundary_within_stride16'] for r in subset),
                near_boundary_covered_ge_05=sum(r['boundary_within_stride16'] and r['max_iou'] >= .5 for r in subset),
                fitting_abs_xy=percentile([max(abs(x) for x in r['best_fitting_delta'][:2]) for r in subset]),
                fitting_abs_log_wh=percentile([max(abs(x) for x in r['best_fitting_delta'][2:]) for r in subset]),
                fit_any_component_abs_gt1=sum(max(abs(x) for x in r['best_fitting_delta']) > 1 for r in subset),
                fit_any_component_abs_gt2=sum(max(abs(x) for x in r['best_fitting_delta']) > 2 for r in subset))
    return dict(scales=scales, ratios=ratios, anchor_count=len(ref), anchors_per_grid=25,
                fully_inside_anchor_count=int(np.count_nonzero(full)), per_gt=rows, summary=summary)


def self_check():
    target = np.asarray([0., 0., 10., 10.])
    test = np.asarray([[0., 0., 10., 10.], [0., 0., 20., 10.], [20., 20., 30., 30.]])
    require(np.array_equal(overlap(test, target), [1., .5, 0.]), 'IoU fixture failed')
    require(np.array_equal(delta_for(test[0], target), [0., 0., 0., 0.]), 'Delta identity failed')
    delta_for(np.asarray([-10., 20., 40., 70.]), np.asarray([3., 7., 9., 25.]))
    ref = anchors(16, [1.], [1.], 1, 2, [16, 16])
    ref[:, 2:] += 1
    require(np.array_equal(ref, [[0, 0, 16, 16], [16, 0, 32, 16]]), 'Half-open/stride fixture failed')
    # Uniformly move an8px target through one16px cell. Shape is perfect in
    # the free-center oracle, yet a fixed center can miss at IoU.5.
    gt = np.asarray([12., 12., 20., 20.])
    cell = np.asarray([[4., 4., 12., 12.], [20., 4., 28., 12.],
                       [4., 20., 12., 28.], [20., 20., 28., 28.]])
    require(overlap(cell, gt).max() == 0 and overlap(gt[None], gt)[0] == 1, 'Grid-phase fixture failed')
    return dict(iou=True, delta_identity_and_inverse=True, half_open_centers=True, stride_phase_gap=True)


def run(args):
    checks = self_check()
    require(not args.output.exists(), 'Preserve earlier evidence; use a new output directory')
    require(sha256(args.firmware) == MODELS[MODEL], 'Unexpected original model')
    require(sha256(args.dataset) == MANIFEST_SHA, 'Unexpected dataset manifest')
    data = args.firmware.read_bytes(); params, _ = recover(data); graph = analyze(data)
    proposal = next(n for n in graph['layers'] if n['operator_id'] == 24)
    require(proposal['feature_stride_candidate'] == [16, 16] and proposal['anchor_base_size_candidate'] == 16,
            'Original geometry differs')
    ratios = params['proposal_blocks'][0]['ratio_candidates']
    require(params['proposal_blocks'][0]['scale_candidates'] == DESIGNS['original'], 'Original scales differ')
    manifest = json.loads(args.dataset.read_text())
    frames = sorted((f for f in manifest['frames'] if f['split'] == 'train'), key=lambda f: f['id'])
    require(len(frames) == 320, 'Expected train320')
    counts = dict(Counter(b['class'] for f in frames for b in f['boxes']))
    require(counts == {'person': 463, 'person_face': 420}, 'Training annotation counts differ')
    names = ('small_anchor_design.py', 'decode_parameters.py', 'extract.py', 'ops_investigation.py',
             'pretraining_data.py', 'proposal_hypothesis.py')
    sources = {n: sha256(BASE/n) for n in names}
    args.output.mkdir(parents=True)
    snapshot = args.output/'source-snapshot'; snapshot.mkdir()
    for name in sources:
        (snapshot/name).write_bytes((BASE/name).read_bytes())
    plan = dict(status='Frozen before geometry measurements', source_sha256=sources,
        model_sha256=MODELS[MODEL], manifest_sha256=MANIFEST_SHA, designs=DESIGNS, ratios=ratios,
        frames=[dict(id=f['id'], sha256=f['sha256']) for f in frames], classes_and_gt_counts=counts,
        coordinate_policy='Continuous half-open xyxy; stretch stored annotation coordinates to384x256',
        small='sqrt(area)<32', medium='32<=sqrt(area)<96', large='sqrt(area)>=96',
        fixed=dict(grid=[16, 24], stride=[16, 16], base=16, ratios=5, scales=5, anchors_per_grid=25,
                   anchor_count=9600, classifier_channels_per_head=50, shared_bbox_channels=100),
        selection='Compare all classes and all size strata; do not select small-only gains that discard large-target coverage. '
                  'Recommendation is based on train geometry only, not validation scores or old-head detections.',
        center_limit='Free-center oracle keeps the same25 shapes but translates each to the GT center. '
                     'Its gap describes pre-regression matching limits, not a bound on regressed detection.',
        boundary_policy='Primary geometry retains anchors with centers inside; partially outside boxes are not discarded. '
                        'Fully-inside-only and clipped-anchor geometry are reported separately, not substituted.',
        patch_policy='Only two20-byte float32 scale arrays in an in-memory clone; reparse and inverse all bytes; never save BIN.',
        self_checks=checks, photographs_read=False, model_inference=False, optimizer_steps=0,
        validation_selection=False, test_inference=False)
    write_json(args.output/'frozen-plan.json', plan)
    results, patches = {}, {}
    for name, scales in DESIGNS.items():
        _, patches[name] = patch_scale_arrays(data, scales)
        results[name] = geometry(frames, ratios, scales)
        print(name, json.dumps({k: (v['covered_ge_05'], v['gt']) for k, v in results[name]['summary'].items()}), flush=True)
    require(all(sha256(BASE/n) == h for n, h in sources.items()), 'Source changed during audit')
    require(sha256(args.dataset) == MANIFEST_SHA and sha256(args.firmware) == MODELS[MODEL], 'Input source changed')
    comparisons = {}
    old = results['original']['summary']
    for name in ('small16_keep256', 'small8_keep128'):
        comparisons[name] = {key: dict(delta_covered_ge_05=value['covered_ge_05']-old[key]['covered_ge_05'],
                                      delta_covered_ge_03=value['covered_ge_03']-old[key]['covered_ge_03'])
                             for key, value in results[name]['summary'].items()}
    chosen = results['small16_keep256']['summary']
    preserves_large = all(chosen[c+'/large_ge96']['covered_ge_05'] >= old[c+'/large_ge96']['covered_ge_05']
                          for c in ('person', 'person_face'))
    improves_small_face = chosen['person_face/small_lt32']['covered_ge_05'] > old['person_face/small_lt32']['covered_ge_05']
    report = dict(status='complete', frozen_plan_sha256=sha256(args.output/'frozen-plan.json'),
        sources=sources, sources_unchanged=True, model_sha256=MODELS[MODEL], manifest_sha256=MANIFEST_SHA,
        geometry=results, patch_roundtrips=patches, change_from_original=comparisons,
        recommendation=dict(design='small16_keep256' if preserves_large and improves_small_face else None,
            scales=DESIGNS['small16_keep256'] if preserves_large and improves_small_face else None,
            basis='Train-only geometry improves small-face coverage while retaining both classes large-target coverage; '
                  'medium-size losses remain visible in the full comparison. The smaller128-max design is not preferred '
                  'when its extra small coverage costs substantial large coverage.',
            not_recognition_validation=True, no_automatic_training_permission=True),
        no_modified_bin_saved=True, photographs_read=False, model_inference=False,
        optimizer_steps=0, validation_selection=False, test_inference=False,
        ready_to_resume_recognition_training=False, verified_original_dsp_anchor_abi=False,
        limitations=['This is a new anchor design, not a repair of the original parameter interpretation.',
            'Changing anchor semantics invalidates old head channel meanings; geometry alone proves no recognition improvement.',
            'Stride16 remains fixed; small-target center-phase matching gaps cannot be removed by smaller scales alone.',
            'Bbox regression can move centers continuously; matching gaps are not absolute detector recall ceilings.',
            'The fitting-delta statistics describe best-IoU anchors, not a trained predictor or an original quantization limit.',
            'Patched component identity rejection is expected. No identity allowlist or firmware signature is bypassed.'])
    write_json(args.output/'report.json', report)
    print('Complete: train GT geometry and in-memory patch roundtrips only.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--firmware', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=BASE/'private-data/recognition-v2/manifest.json')
    parser.add_argument('--output', type=Path, default=BASE/'artifacts/recognition-diagnosis-v3/small-anchor-design/train320')
    run(parser.parse_args())
