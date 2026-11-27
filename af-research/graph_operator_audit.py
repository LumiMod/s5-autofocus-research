#!/usr/bin/env python3
"""Static original-graph and synthetic host-operator audit; no image inference.

This checks byte preservation and host implementation consistency. It cannot
establish original DSP operator semantics or select a recognition model.
"""
import argparse
import itertools
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from candidate_network import CandidateNetwork, candidate_weight
from decode_parameters import recover, tail_offset
from extract import MODELS, decode, digest, require, write_json
from ops_investigation import analyze
from packing_investigation import dimensions, unpack
from research_paths import add_firmware_arguments, resolve_source


def k4_full(raw, record):
    """Independent implementation of the candidate physical [Ob,Kb,O4,K4]."""
    logical, outputs, inputs, kh, kw = dimensions(record)
    k = inputs * kh * kw
    require(outputs % 4 == k % 4 == 0, 'K4 alignment mismatch')
    full = raw.reshape(outputs // 4, k // 4, 4, 4).transpose(0, 2, 1, 3)
    return full.reshape(outputs, inputs, kh, kw)


def manual_pool(image, mode):
    """Independent valid-ceil 3x3/stride2 reference including partial windows."""
    h, w = image.shape
    out = np.empty(((h - 2) // 2 + 1, (w - 2) // 2 + 1), np.float64)
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            patch = image[2*y:2*y+3, 2*x:2*x+3]
            out[y, x] = patch.max() if mode == 'max' else patch.mean()
    return out


def synthetic_checks():
    # Compare the actual CandidateNetwork.forward with an independently
    # assembled NumPy reference. No recovered weights or photograph is used.
    image = np.arange(6, dtype=np.float32).reshape(1, 1, 2, 3) - 2
    conv = nn.Conv2d(1, 50, 1)
    with torch.no_grad():
        conv.weight.copy_(torch.arange(50).reshape(50, 1, 1, 1) / 100)
        conv.bias.copy_(torch.arange(50).remainder(7) / 10 - .3)

    def node(index, op, inputs, shapes, **kwargs):
        return dict(index=index, name='synthetic_%d' % index, operator_id=op,
                    inputs=[dict(layer=i, port=p) for i, p in inputs],
                    output_shapes_candidate=shapes, **kwargs)

    layers = [node(0, 6, [], [[1, 1, 2, 3]], shape=[1, 1, 2, 3]),
              node(1, 1, [(0, 0)], [[1, 50, 2, 3]], relu_candidate=False),
              node(2, 10, [(1, 0)], [[1, 25, 2, 3]]*2, cuts=[25], axis=1),
              node(3, 9, [(2, 1), (2, 0)], [[1, 50, 2, 3]], axis=1),
              node(4, 8, [(3, 0)], [[1, 2, 50, 3]], shape=[1, 2, 50, 3]),
              node(5, 29, [(4, 0)], [[1, 2, 50, 3]], axis=1),
              node(6, 8, [(5, 0)], [[1, 50, 2, 3]], shape=[1, 50, 2, 3]),
              node(7, 24, [(6, 0), (1, 0)], [])]
    dummy = SimpleNamespace(layers=layers, convs={'1': conv})
    with torch.inference_mode():
        heads, _ = CandidateNetwork.forward(dummy, torch.from_numpy(image))
    raw = (image * (np.arange(50, dtype=np.float32) / 100).reshape(1, 50, 1, 1)
           + (np.arange(50, dtype=np.float32) % 7 / 10 - .3).reshape(1, 50, 1, 1))
    reversed_groups = np.concatenate([raw[:, 25:], raw[:, :25]], axis=1)
    paired = reversed_groups.reshape(1, 2, 50, 3).astype(np.float64)
    exponents = np.exp(paired - paired.max(axis=1, keepdims=True))
    expected = (exponents / exponents.sum(axis=1, keepdims=True)).reshape(1, 50, 2, 3)
    actual = heads['synthetic_7_probabilities'].numpy()
    error = float(np.max(np.abs(expected - actual)))
    require(error < 2e-7, 'Actual forward order/reshape/softmax differs from reference')
    require(np.max(np.abs(raw - heads['bbox_deltas'].numpy())) < 2e-7,
            'Proposal input preservation failed')
    # One Caffe-style logical reshape pairs channel a with channel 25+a.
    paired_by_channel = np.stack([reversed_groups[:, :25], reversed_groups[:, 25:]], axis=1)
    require(np.array_equal(paired_by_channel.reshape(1, 2, 50, 3),
                           reversed_groups.reshape(1, 2, 50, 3)), 'Class grouping failed')
    image2 = np.arange(24, dtype=np.float32).reshape(4, 6)
    t = torch.from_numpy(image2)[None, None]
    maximum = F.max_pool2d(t, 3, 2, ceil_mode=True).numpy()[0, 0]
    average = F.avg_pool2d(t, 3, 2, ceil_mode=True).numpy()[0, 0]
    require(np.array_equal(maximum, manual_pool(image2, 'max')), 'Max pool reference differs')
    require(np.array_equal(average, manual_pool(image2, 'avg')), 'Average pool reference differs')
    # ReLU forward flag is checked with an identity convolution, independent of
    # whether that flag means ReLU in the original CDNN runtime.
    identity = nn.Conv2d(1, 1, 1)
    with torch.no_grad():
        identity.weight.fill_(1); identity.bias.zero_()
    relu_errors = {}
    for flag in (False, True):
        relu_layers = [layers[0], node(1, 1, [(0, 0)], [[1, 1, 2, 3]], relu_candidate=flag),
                       node(2, 24, [(1, 0), (1, 0)], [])]
        with torch.inference_mode():
            value, _ = CandidateNetwork.forward(SimpleNamespace(layers=relu_layers, convs={'1': identity}),
                                                 torch.from_numpy(image))
        expect = np.maximum(image, 0) if flag else image
        relu_errors[str(flag)] = float(np.max(np.abs(value['bbox_deltas'].numpy() - expect)))
        require(relu_errors[str(flag)] == 0, 'Host ReLU flag dispatch differs')
    # Exhaustively establish this K4 permutation is outside the earlier fixed
    # six rectangular-axis permutations; this is a synthetic identity test,
    # not a search over photograph accuracy or a layout correctness proof.
    axes = ('Ob', 'Ib', 'O4', 'I4', 'H', 'W')
    canonical = ('Ob', 'O4', 'Ib', 'I4', 'H', 'W')
    dims = dict(Ob=2, Ib=3, O4=4, I4=4, H=3, W=3)
    marker = np.arange(8*12*3*3).reshape(-1)
    new = marker.reshape(2, 27, 4, 4).transpose(0, 2, 1, 3).reshape(8, 12, 3, 3)
    equal_orders = []
    for order in itertools.permutations(axes):
        old = marker.reshape(tuple(dims[a] for a in order))
        old = old.transpose(tuple(order.index(a) for a in canonical)).reshape(8, 12, 3, 3)
        if np.array_equal(old, new):
            equal_orders.append(list(order))
    require(not equal_orders, 'K4 accidentally duplicates a previous six-axis permutation')
    return dict(actual_forward_concat_reverse_slice_reshape_softmax_max_error=error,
                host_relu_flag_errors=relu_errors, valid_ceil_pool_shape=list(maximum.shape),
                max_pool_ramp_output=maximum.tolist(), average_pool_ramp_output=average.tolist(),
                max_vs_average_max_difference=float(np.max(np.abs(maximum-average))),
                k4_equal_to_previous_720_six_axis_orders=equal_orders,
                k4_novelty_synthetic_dimensions=[8, 12, 3, 3],
                claim='Host consistency and candidate distinction only; not DSP numerical equivalence.')


def inspect_model(data):
    graph = decode(data)
    analysis = analyze(data)
    params, tensors = recover(data)
    records = {r['index']: r for r in params['convolutions']}
    proposal_params = {r['layer']: r for r in params['proposal_blocks']}
    result = dict(source_sha256=digest(data), concat=[], slices=[], softmax=[], pools=[],
                  activation_flags=[], k4_layers=[], upsample=None, proposals=[],
                  header_class_strings=graph['classes'], header_output_pairs=graph['outputs'])
    for node, meta in zip(graph['layers'], analysis['layers']):
        q = tail_offset(node)
        # Decode raw pair bytes independently from their location before tail.
        count = len(node['inputs'])
        pairs = [struct.unpack_from('<II', data, q - 8*count + 8*i) for i in range(count)]
        expected = [(0xffffffff if x['layer'] is None else x['layer'],
                     0xffffffff if x['port'] is None else x['port']) for x in meta['inputs']]
        require(pairs == expected, 'Raw input order differs from propagated graph')
        op = node['operator_id']
        common = dict(index=node['index'], name=node['name'], record_start=node['record_start'],
                      tail_offset=q)
        if op == 9:
            result['concat'].append(dict(common, raw_input_pairs=[list(x) for x in pairs],
                                          axis=meta['axis'], inputs=meta['input_shapes_candidate']))
        elif op == 10:
            consumers = []
            for following in analysis['layers']:
                for inp in following['inputs']:
                    if inp['layer'] == node['index']:
                        consumers.append(dict(layer=following['index'], port=inp['port'],
                                              channel_count=meta['output_shapes_candidate'][inp['port']][1]))
            result['slices'].append(dict(common, axis=meta['axis'], cut=meta['cuts'][0],
                                         output_channels=[s[1] for s in meta['output_shapes_candidate']],
                                         consumers=consumers))
        elif op == 29:
            predecessor = analysis['layers'][meta['inputs'][0]['layer']]
            require(predecessor['operator_id'] == 8, 'Softmax no longer follows reshape')
            result['softmax'].append(dict(common, axis=meta['axis'],
                                          raw_q57=meta['raw_q57'], raw_q58=meta['raw_q58'],
                                          input_shape=meta['input_shapes_candidate'][0],
                                          pre_reshape_shape=predecessor['input_shapes_candidate'][0],
                                          grouping='Under contiguous logical NCHW: [bg25,fg25], same anchor and pixel.'))
        elif op == 5:
            result['pools'].append(dict(common, kernel=meta['kernel'], stride=meta['stride'],
                                       padding=meta['padding'], extra_flag=meta['pool_extra_flag'],
                                       raw_q85=meta['raw_q85'], raw_q86=meta['raw_q86'],
                                       input_shape=meta['input_shapes_candidate'][0],
                                       output_shape=meta['output_shapes_candidate'][0],
                                       original_max_or_average='unresolved'))
        elif op == 24:
            parameter = proposal_params[node['index']]
            block = data[parameter['offset']:parameter['offset']+576]
            ratios = list(struct.unpack_from('<5f', block, 0))
            scales = list(struct.unpack_from('<5f', block, 32))
            require(ratios == parameter['ratio_candidates'] and scales == parameter['scale_candidates'],
                    'Raw proposal ratios/scales differ from recovered metadata')
            result['proposals'].append(dict(common, raw_input_pairs=[list(p) for p in pairs],
                compared_tail_bytes=101, compared_tail_sha256=digest(data[q:q+101]),
                parameter_file_offset=parameter['offset'], parameter_bytes=576,
                parameter_sha256=digest(block), ratio_candidates=ratios, scale_candidates=scales,
                ratio_and_scale_padding_all_zero=block[20:32]+block[52:64] == bytes(24),
                stride_base_ratio_count_scale_count_topn=list(struct.unpack_from('<7I', data, q+61)),
                nms_threshold_candidate=struct.unpack_from('<f', data, q+89)[0],
                unknown_q93_q97=list(struct.unpack_from('<2I', data, q+93)),
                proposal_output_scale_slots=list(struct.unpack_from('<10f', data, q+11))))
        elif op == 1:
            result['activation_flags'].append(dict(common, byte_q100=data[q+100], byte_q101=data[q+101],
                                                   u32_q335=struct.unpack_from('<I', data, q+335)[0]))
            require(data[q+100] in (0, 1) and data[q+101] in (0, 1), 'Nonbinary conv flag')
            record = records[node['index']]
            raw = tensors[record['tensor_key']+'_weight_i16_packed']
            if meta['transposed_candidate']:
                weight, _ = candidate_weight(raw, record, meta, 'o4_i4_hw')
                require(np.all(tensors[record['tensor_key']+'_bias_i32_pairs'] == 0), 'Upsample bias not zero')
                _, _, h, w = meta['input_shapes_candidate'][0]
                ones = torch.ones(1, 1, h, w)
                kernel = torch.from_numpy(weight[:1])
                out = F.conv_transpose2d(ones, kernel, stride=2, padding=1)[0, 0]
                interpolated = F.interpolate(ones, scale_factor=2, mode='bilinear', align_corners=False)[0, 0]
                difference = out-interpolated
                border = torch.ones_like(out, dtype=torch.bool); border[1:-1, 1:-1] = False
                blocks = raw.reshape(256, 4, 4, 4)
                phase = blocks[0, :, 0, :].reshape(2, 2, 2, 2)
                reproduced = np.zeros_like(blocks)
                reproduced[:128, :, 0, :] = phase.reshape(4, 4)
                reproduced[:128, :, 1, :] = phase.reshape(4, 4)
                require(np.array_equal(blocks, reproduced), 'Full upsample phase-table pattern changed')
                result['upsample'] = dict(common, logical_shape=meta['output_shapes_candidate'][0],
                    full_phase_table_pattern_matches=True,
                    input_scale=record['input_scale'], output_scale=record['output_scale'],
                    all_bias_pairs_zero=True, kernel_sum=float(kernel.sum()),
                    constant_input_corner=float(out[0, 0]), constant_input_edge=float(out[0, 1]),
                    constant_input_interior=float(out[1, 1]), resize_constant_value=1.0,
                    interior_max_abs_difference_from_resize=float(difference[~border].abs().max()),
                    border_max_abs_difference_from_resize=float(difference[border].abs().max()),
                    boundary_pixels=int(border.sum()), total_pixels=out.numel(),
                    interpretation='Current zero-extended transposed convolution attenuates the outer border; '
                                   'bilinear resize with replicated borders does not. Original border and channel routing unresolved.')
            else:
                full = k4_full(raw, record)
                logical, outputs, inputs, kh, kw = dimensions(record)
                require(inputs == meta['input_shapes_candidate'][0][1], 'K4/graph channel mismatch')
                require(np.all(full[logical:] == 0), 'K4 nonzero output padding')
                rebuilt = full.reshape(outputs//4, 4, inputs*kh*kw//4, 4).transpose(0, 2, 1, 3)
                require(np.array_equal(rebuilt.reshape(raw.shape), raw), 'K4 inverse failed')
                row = dict(common, logical_shape=[logical, inputs, kh, kw],
                           weight_sha256=record['weight']['sha256'], weight_offset=record['weight']['data_offset'],
                           inverse_roundtrip_identical=True, zero_output_padding=True)
                if node['name'] == 'conv1':
                    how = raw.reshape(8, 4, 4, 4).transpose(0, 2, 1, 3).reshape(32, 1, 4, 4)
                    require(np.array_equal(full, how), 'K4 conv1 differs from HOW')
                    row['exactly_equals_existing_conv1_HOW'] = True
                elif kh == kw == 1:
                    require(np.array_equal(full[:logical], unpack(raw, record)), 'K4 1x1 differs')
                    row['exactly_equals_existing_1x1'] = True
                else:
                    row['equal_to_existing_spatial_layouts'] = {
                        layout: bool(np.array_equal(full[:logical], unpack(raw, record, layout)))
                        for layout in ('o4_i4_hw', 'hw_o4_i4', 'o4_hw_i4')}
                result['k4_layers'].append(row)
    result['checks'] = dict(raw_input_pair_order_preserved_all_layers=True,
        graph_shape_constraints=analysis['checks'], k4_ordinary_convolutions=len(result['k4_layers']),
        k4_equal_1x1_count=sum(r.get('exactly_equals_existing_1x1', False) for r in result['k4_layers']),
        k4_spatial3x3_count=sum(r['logical_shape'][2:] == [3, 3] for r in result['k4_layers']))
    proposal_rows = result['proposals']
    result['proposal_consistency'] = dict(
        compared_first_101_tail_bytes_identical=len({r['compared_tail_sha256'] for r in proposal_rows}) == 1,
        complete_576_byte_parameters_identical=len({r['parameter_sha256'] for r in proposal_rows}) == 1,
        all_use_same_bbox_input=len({tuple(r['raw_input_pairs'][1]) for r in proposal_rows}) == 1,
        bbox_input_pair=proposal_rows[0]['raw_input_pairs'][1],
        semantic_head_to_class_mapping='Header string order and ordered output-pair list support a mapping; '
            'they do not by themselves establish an original-runtime label ABI.',
        conclusion='There is no observed per-class proposal-parameter difference or separate face bbox input. '
            'Using the first proposal block for both classes does not change any recovered parameter bytes. '
            'Anchor/channel order, bbox transformation, exponential argument, and external image-info semantics remain hypotheses.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_firmware_arguments(parser)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent /
                        'artifacts/recognition-diagnosis-v3/graph-operators/static-synthetic-audit.json')
    args = parser.parse_args()
    source = resolve_source(args)
    base = Path(__file__).resolve().parent
    names = ['graph_operator_audit.py', 'candidate_network.py', 'ops_investigation.py',
             'decode_parameters.py', 'extract.py', 'packing_investigation.py']
    before = {name: digest((base/name).read_bytes()) for name in names}
    torch.set_num_threads(1)
    report = dict(schema='s5-graph-static-synthetic-audit-v1', source_hashes=before,
        photographs_or_manifests_read=False, test_inference=False, optimizer_steps=0,
        training=False, verified_recognition_fix=False, ready_to_resume_recognition_training=False,
        original_dsp_execution=False, synthetic_checks=synthetic_checks(), models={})
    for stem, expected in MODELS.items():
        data = (source/(stem+'.bin')).read_bytes()
        require(digest(data) == expected, 'Unexpected original component SHA')
        report['models'][stem] = inspect_model(data)
    after = {name: digest((base/name).read_bytes()) for name in names}
    require(before == after, 'Audited source changed during static run')
    report['source_hashes_unchanged_during_run'] = True
    proposal_rows = [p for model in report['models'].values() for p in model['proposals']]
    report['cross_model_proposal_consistency'] = dict(
        compared_first_101_tail_bytes_identical=len({p['compared_tail_sha256'] for p in proposal_rows}) == 1,
        complete_576_byte_parameters_identical=len({p['parameter_sha256'] for p in proposal_rows}) == 1,
        proposal_count=len(proposal_rows))
    report['conclusions'] = [
        'No demonstrated concat, slice-port, or host reshape/softmax implementation bug in the audited candidate.',
        'K4 is structurally consistent with every ordinary convolution and exact prior 1x1 and conv1 HOW cases; '
        'the 3x3 mapping is outside the prior 720 rectangular-axis permutations. This warrants a bounded independent '
        'control experiment, not recognition or original-DSP correctness claims.',
        'Max-versus-average pooling, precise ceil/border handling, physical tensor storage, ReLU meaning, '
        'upsample channel/phase routing and quantized softmax remain unconfirmed.',
        'The transposed-convolution/resize border distinction is a measurable host hypothesis difference. '
        'No image scores justify replacing the current upsample operator.',
        'All eight proposals across both pinned components have byte-identical first 101 tail bytes and complete '
        '576-byte parameter blocks. Both 44 heads explicitly share bbox94. No recovered evidence justifies '
        'assigning a different small-anchor parameter set or bbox head to person_face.',
        'Shape and inverse roundtrip consistency do not recover the original arithmetic or prove usable recognition.']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, report)
    print(args.output.relative_to(base))
    print({stem: model['checks'] for stem, model in report['models'].items()})


if __name__ == '__main__':
    main()
