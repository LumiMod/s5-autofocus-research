#!/usr/bin/env python3
"""Static 720-axis-order probe of pinned S5 weights; no photograph inference.

Spatial smoothness can expose omitted layouts, but cannot prove their DSP ABI.
O/I regrouping and H/W exchange create exact ambiguities retained in outputs.
"""
import argparse
import itertools
import json
from pathlib import Path
import time

import numpy as np

from decode_parameters import recover
from extract import MODELS, digest, require, write_json
from packing_investigation import dimensions, unpack
from research_paths import add_firmware_arguments, resolve_source

AXES = ('Ob', 'Ib', 'O4', 'I4', 'H', 'W')
CANONICAL = ('Ob', 'O4', 'Ib', 'I4', 'H', 'W')
LAYERS = ('conv2', 'conv3', 'rpn_5_1_conv', 'rpn_5_2_conv', 'rpn_3_1_conv')
EXISTING = {'o4_i4_hw': ('Ob', 'Ib', 'O4', 'I4', 'H', 'W'),
            'hw_o4_i4': ('Ob', 'Ib', 'H', 'W', 'O4', 'I4'),
            'o4_hw_i4': ('Ob', 'Ib', 'O4', 'H', 'W', 'I4')}


def axis_id(order):
    return '/'.join(order)


def source_hashes():
    base = Path(__file__).resolve().parent
    return {name: digest((base/name).read_bytes()) for name in
            ('weight_outer_axis_probe.py', 'decode_parameters.py', 'extract.py', 'packing_investigation.py')}


def logical_weight(raw, record, order):
    logical, outputs, inputs, kh, kw = dimensions(record)
    require(logical == outputs and outputs % 4 == inputs % 4 == 0, 'Probe requires unpadded O4/I4 layers')
    dims = dict(Ob=outputs//4, Ib=inputs//4, O4=4, I4=4, H=kh, W=kw)
    shaped = raw.reshape(tuple(dims[a] for a in order))
    return shaped.transpose(tuple(order.index(a) for a in CANONICAL)).reshape(outputs, inputs, kh, kw)


def neighbor_metric_oi(weight):
    w = weight.astype(np.float64)
    energy = float(np.mean(w*w))
    h = float(np.mean(np.diff(w, axis=2)**2))
    v = float(np.mean(np.diff(w, axis=3)**2))
    return (h+v)/(2*energy)


def average_ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind='stable')
    ranks = np.empty(len(values), np.float64)
    start = 0
    while start < len(order):
        end = start+1
        while end < len(order) and np.isclose(values[order[end]], values[order[start]], rtol=0, atol=1e-12):
            end += 1
        ranks[order[start:end]] = ((start+1)+end)/2
        start = end
    return ranks


def spearman(left, right):
    return float(np.corrcoef(average_ranks(left), average_ranks(right))[0, 1])


def representative(orders):
    # Keep the existing nonspatial O/I nesting as a deterministic tie-break,
    # not as evidence. Spatial smoothness cannot tell H from W or O from I.
    def key(order):
        nonspatial = tuple(a for a in order if a not in ('H', 'W'))
        return (nonspatial != ('Ob', 'Ib', 'O4', 'I4'), order.index('H') > order.index('W'),
                order[0] != 'Ob', tuple(AXES.index(a) for a in order))
    return min(orders, key=key)


def self_check():
    dims = dict(Ob=2, Ib=3, O4=4, I4=4, H=3, W=3)
    flat = np.arange(np.prod(list(dims.values())), dtype=np.int16)
    for order in itertools.permutations(AXES):
        physical = flat.reshape(tuple(dims[a] for a in order))
        logical = physical.transpose(tuple(order.index(a) for a in CANONICAL))
        back = logical.transpose(tuple(CANONICAL.index(a) for a in order))
        require(np.array_equal(back.reshape(-1), flat), 'Permutation inversion failed')
        expected = neighbor_metric_oi(logical.reshape(8, 12, 3, 3))
        arr = physical.astype(float)
        actual = (np.mean(np.diff(arr, axis=order.index('H'))**2) +
                  np.mean(np.diff(arr, axis=order.index('W'))**2))/(2*np.mean(arr**2))
        require(abs(expected-actual) < 1e-12, 'Fast metric differs from explicit same-OI kernel metric')


def inspect_layer(raw, record, orders):
    logical, outputs, inputs, kh, kw = dimensions(record)
    require(logical == outputs and (kh, kw) == (3, 3), 'Unexpected requested layer geometry')
    dims = dict(Ob=outputs//4, Ib=inputs//4, O4=4, I4=4, H=kh, W=kw)
    values = raw.astype(np.float64).reshape(-1)
    energy = float(np.mean(values*values))
    require(energy > 0, 'Empty weight energy')
    cache, metrics = {}, []
    for order in orders:
        shape = tuple(dims[a] for a in order)
        spatial = tuple(sorted((order.index('H'), order.index('W'))))
        signature = (shape, spatial)
        if signature not in cache:
            arr = values.reshape(shape)
            diffs = [float(np.mean(np.diff(arr, axis=axis)**2)) for axis in spatial]
            cache[signature] = (diffs[0]+diffs[1])/(2*energy)
        metrics.append(cache[signature])
    baseline_checks = {}
    for name, order in EXISTING.items():
        candidate = logical_weight(raw, record, order)
        require(np.array_equal(candidate, unpack(raw, record, name)), 'Existing unpack disagrees: '+name)
        expected = neighbor_metric_oi(candidate)
        actual = metrics[orders.index(order)]
        require(abs(expected-actual) < 1e-12, 'Existing layout statistic differs')
        baseline_checks[name] = actual
    return {'name': record['name'], 'layer': record['index'], 'axis_dimensions': dims,
            'logical_weight_shape': [outputs, inputs, kh, kw],
            'weight_file_offset': record['weight']['data_offset'],
            'stored_weight_sha256': record['weight']['sha256'], 'raw_energy': energy,
            'permutation_count': len(orders), 'cached_exact_geometry_classes': len(cache),
            'normalized_neighbor_difference_energy': metrics, 'existing_layout_checks': baseline_checks}


def run(args):
    self_check()
    started = time.monotonic()
    root = resolve_source(args)
    args.output.mkdir(parents=True, exist_ok=True)
    orders = list(itertools.permutations(AXES))
    source_ids = source_hashes()
    plan = {'status': 'Static plan fixed before measuring weight layouts', 'sources': source_ids,
            'models': MODELS, 'selected_layer_names': list(LAYERS),
            'selection_reason': 'Fixed early conv2/conv3 and all three terminal 3x3 RPN convolutions; no image/label scores used',
            'axis_orders': [list(order) for order in orders], 'existing_orders': EXISTING,
            'metric': '(mean squared same-OI H-neighbor differences + mean squared same-OI W-neighbor differences)/(2*mean squared weights)',
            'model_aggregation': 'Equal weight for each of five layers; equal weight for the two models',
            'candidate_gate': {'max_frozen': 2, 'min_relative_improvement_over_best_existing_macro_each_model': .10,
                'min_layers_improved_over_per_layer_best_existing_each_model': 4,
                'max_any_layer_ratio_to_per_layer_best_existing': 1.15,
                'max_percentile_rank_each_model': .10},
            'tie_policy': 'Group exactly equal score vectors across all ten tensors. Deterministic representative favors existing nonspatial Ob/Ib/O4/I4 order and H-before-W; ties remain unresolved.',
            'inference': False, 'training': False, 'photograph_or_manifest_read': False,
            'limitations': ['Lower spatial difference is a heuristic, not an original-layout proof.',
                           'A spatial-energy statistic cannot identify O vs I labels or distinguish transposed H/W.',
                           'The two models are related architectures and not guaranteed independent evidence.',
                           'No special padded/spatial tiling beyond six rectangular axes is tested.']}
    write_json(args.output/'predeclared-plan.json', plan)
    measured, raw_by_model = {}, {}
    for stem in MODELS:
        data = (root/(stem+'.bin')).read_bytes()
        require(digest(data) == MODELS[stem], 'Unknown source identity')
        params, tensors = recover(data)
        records = {r['name']: r for r in params['convolutions']}
        measured[stem], raw_by_model[stem] = [], {}
        for name in LAYERS:
            record = records[name]
            raw = tensors[record['tensor_key']+'_weight_i16_packed']
            raw_by_model[stem][name] = raw.copy()
            measured[stem].append(inspect_layer(raw, record, orders))
            print(json.dumps({'model': stem, 'layer': name, 'orders': len(orders),
                              'elapsed_seconds': time.monotonic()-started}), flush=True)
    write_json(args.output/'layer-metrics.json', {'order_ids': [axis_id(o) for o in orders], 'models': measured})
    models = list(MODELS)
    arrays = {m: np.asarray([layer['normalized_neighbor_difference_energy'] for layer in measured[m]]) for m in models}
    means = {m: arrays[m].mean(axis=0) for m in models}
    ranks = {m: average_ranks(means[m]) for m in models}
    existing_indices = [orders.index(order) for order in EXISTING.values()]
    baselines = {m: {'best_macro': float(means[m][existing_indices].min()),
                     'best_each_layer': arrays[m][:, existing_indices].min(axis=1)} for m in models}
    correlation = []
    for i, name in enumerate(LAYERS):
        left, right = (raw_by_model[m][name].astype(float).reshape(-1) for m in models)
        correlation.append({'layer': name,
            'raw_weight_bytes_identical': bool(np.array_equal(left, right)),
            'raw_weight_pearson_correlation': float(np.corrcoef(left, right)[0, 1]),
            'layout_rank_spearman': spearman(arrays[models[0]][i], arrays[models[1]][i])})
    grand_mean = np.stack(list(means.values())).mean(axis=0)
    groups = {}
    for i, order in enumerate(orders):
        signature = tuple(round(float(value), 12) for m in models for value in arrays[m][:, i])
        groups.setdefault(signature, []).append(i)
    ranking = []
    for indices in groups.values():
        rep = representative([orders[i] for i in indices])
        i = orders.index(rep)
        summary, qualify = {}, True
        for m in models:
            ratio = arrays[m][:, i]/baselines[m]['best_each_layer']
            improvement = 1-means[m][i]/baselines[m]['best_macro']
            improved_count = int(np.count_nonzero(ratio < 1-1e-12))
            gates = {'macro_improvement_ge_10_percent': bool(improvement >= .10),
                     'at_least_4_layers_improved': improved_count >= 4,
                     'no_layer_worse_than_15_percent': bool(ratio.max() <= 1.15),
                     'top_10_percent_of_orders': bool(ranks[m][i]/len(orders) <= .10)}
            qualify &= all(gates.values())
            summary[m] = {'macro_normalized_difference': float(means[m][i]),
                          'relative_improvement_over_best_existing_macro': float(improvement),
                          'layers_improved': improved_count, 'per_layer_ratio_to_best_existing': ratio.tolist(),
                          'macro_average_tie_rank': float(ranks[m][i]), 'gates': gates}
        ranking.append({'representative': list(rep), 'order_id': axis_id(rep),
                        'equivalent_orders': [list(orders[k]) for k in indices],
                        'equivalence_size': len(indices), 'grand_macro_normalized_difference': float(grand_mean[i]),
                        'qualifies_as_structural_probe_candidate': bool(qualify), 'models': summary})
    ranking.sort(key=lambda row: (row['grand_macro_normalized_difference'], row['order_id']))
    # Take at most two genuinely different score classes, never two axis swaps
    # with identical spatial statistics.
    frozen = [r for r in ranking if r['qualifies_as_structural_probe_candidate']][:2]
    write_json(args.output/'ranked-equivalence-classes.json', ranking)
    write_json(args.output/'frozen-candidates.json', {
        'status': 'Candidates frozen from static structure only; correctness unresolved',
        'count': len(frozen), 'candidates': frozen, 'source_hashes': source_ids,
        'logical_reconstruction': 'raw.reshape(tuple(axis_sizes[a] for a in order)).transpose(order.index(a) for a in [Ob,O4,Ib,I4,H,W]).reshape(O,I,3,3)',
        'scope': 'Only 3x3 convolution weight order. No change justified for 1x1, conv1, input/output activations or proposal decoding.',
        'unknowns': plan['limitations'] + ['The selected representative within every equivalence class is only a fixed tie-break.']})
    report = {'status': 'Static outer-axis probe completed; no layout validated',
              'source_hashes': source_ids, 'models': MODELS, 'layers_per_model': list(LAYERS),
              'axis_orders': len(orders), 'score_equivalence_classes': len(ranking),
              'cross_model_macro_rank_spearman': spearman(means[models[0]], means[models[1]]),
              'cross_model_layer_comparison': correlation,
              'existing_macro_scores': {m: {name: float(means[m][orders.index(order)]) for name, order in EXISTING.items()} for m in models},
              'lowest_macro_score_class': {'representative': ranking[0]['representative'],
                  'equivalence_size': ranking[0]['equivalence_size'],
                  'contains_existing_layout': any(list(order) in ranking[0]['equivalent_orders'] for order in EXISTING.values())},
              'per_layer_best_improvements': {m: [
                  {'layer': name, 'lowest_energy': float(arrays[m][j].min()),
                   'best_existing_energy': float(baselines[m]['best_each_layer'][j]),
                   'relative_improvement': float(1-arrays[m][j].min()/baselines[m]['best_each_layer'][j]),
                   'lowest_score_order_count': int(np.count_nonzero(np.isclose(arrays[m][j], arrays[m][j].min(), atol=1e-12, rtol=0)))}
                  for j, name in enumerate(LAYERS)] for m in models},
              'frozen_candidate_count': len(frozen), 'frozen_order_ids': [r['order_id'] for r in frozen],
              'qualifying_score_classes': sum(r['qualifies_as_structural_probe_candidate'] for r in ranking),
              'permutation_inverse_and_fast_metric_checks': 720,
              'existing_unpack_parity_checks': 2*len(LAYERS)*len(EXISTING),
              'photograph_inference': False, 'training_steps': 0, 'test_access': False,
              'elapsed_seconds': time.monotonic()-started, 'limitations': plan['limitations']}
    require(source_hashes() == source_ids, 'Source changed during measurement')
    write_json(args.output/'report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_firmware_arguments(parser)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'artifacts/recognition-diagnosis/outer-axes')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
