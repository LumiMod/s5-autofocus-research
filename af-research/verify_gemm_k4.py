#!/usr/bin/env python3
"""Verify explicit K4 integration after applying the prepared core patch.

Only pinned model bytes and deterministic synthetic tensors are used. These
checks establish implementation parity, not original DSP numerical equivalence.
No dataset, image, AP, optimizer or retrained weights are involved.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from candidate_network import CandidateNetwork
from decode_parameters import recover
from extract import MODELS, digest, require
from flattened_k_diagnosis import build_detector, unpack_flattened_k4
from packing_investigation import PHYSICAL_LAYOUTS, SPATIAL_LAYOUTS, dimensions, pack, unpack
from research_paths import add_firmware_arguments, resolve_source


BASE = Path(__file__).resolve().parent
LEGACY = ('o4_i4_hw', 'hw_o4_i4', 'o4_hw_i4')


def scalar_k4_offset(o, i, y, x, inputs, kh, kw):
    """Independent scalar-address expression, including cross-kernel K4 tiles."""
    k = (i * kh + y) * kw + x
    return ((((o // 4) * ((inputs * kh * kw) // 4) + k // 4) * 4 + o % 4) * 4 + k % 4)


def scalar_legacy_offset(o, i, y, x, inputs, kh, kw, layout, first=False):
    if first:
        return ((o * inputs + i) * kh + y) * kw + x
    outer = (o // 4) * (inputs // 4) + i // 4
    if (kh, kw) == (1, 1) or layout == 'o4_i4_hw':
        return (((outer * 4 + o % 4) * 4 + i % 4) * kh + y) * kw + x
    if layout == 'hw_o4_i4':
        return (((outer * kh + y) * kw + x) * 4 + o % 4) * 4 + i % 4
    require(layout == 'o4_hw_i4', 'Unexpected legacy layout')
    return (((outer * 4 + o % 4) * kh + y) * kw + x) * 4 + i % 4


def expect_rejected(function, label):
    try:
        function()
    except ValueError:
        return label
    raise AssertionError('Expected rejection: ' + label)


def synthetic_address_checks():
    report = []
    for physical, logical, inputs, kh, kw in ((8, 6, 4, 3, 3), (8, 8, 1, 4, 4),
                                             (8, 6, 12, 1, 1), (12, 12, 8, 3, 3)):
        expected = np.arange(physical * inputs * kh * kw, dtype=np.int16).reshape(physical, inputs, kh, kw)
        expected[logical:] = 0
        raw = np.empty(expected.size, dtype=np.int16)
        for o, i, y, x in np.ndindex(expected.shape):
            raw[scalar_k4_offset(o, i, y, x, inputs, kh, kw)] = expected[o, i, y, x]
        record = {'name': 'conv1' if inputs == 1 else 'fixture',
                  'output_channels_candidate': logical, 'kernel_hw_candidate': [kh, kw],
                  'weight': {'stored_shape': [physical, inputs, kh, kw]}}
        require(np.array_equal(unpack(raw, record, 'gemm_k4'), expected[:logical]),
                'K4 unpack differs from scalar-address fixture')
        require(np.array_equal(pack(expected[:logical], record, 'gemm_k4').reshape(-1), raw),
                'K4 pack differs from independently constructed physical bytes')
        if logical != physical:
            corrupt = raw.copy()
            corrupt[scalar_k4_offset(logical, 0, 0, 0, inputs, kh, kw)] = 1
            expect_rejected(lambda: unpack(corrupt, record, 'gemm_k4'), 'nonzero_output_padding')
        report.append({'physical_output': physical, 'logical_output': logical,
                       'inputs': inputs, 'kernel': [kh, kw], 'scalar_address_unpack_pack_equal': True,
                       'nonzero_output_padding_rejected': logical != physical})
    bad = {'name': 'fixture', 'output_channels_candidate': 4,
           'kernel_hw_candidate': [1, 1], 'weight': {'stored_shape': [4, 1, 1, 1]}}
    rejected = [expect_rejected(lambda: unpack(np.zeros(4, dtype=np.int16), bad, 'gemm_k4'),
                               'K_not_multiple_of_four'),
                expect_rejected(lambda: unpack(np.zeros(4, dtype=np.int16), bad, 'unrecognized'),
                                'unknown_layout')]
    return report, rejected


def original_byte_checks(data):
    params, tensors = recover(data)
    rows = []
    for record in params['convolutions']:
        if record['name'] == 'upsample':
            continue
        raw = tensors[record['tensor_key'] + '_weight_i16_packed']
        candidate = unpack(raw, record, 'gemm_k4')
        require(np.array_equal(candidate, unpack_flattened_k4(raw, record)),
                'Core K4 differs from frozen independent root helper: ' + record['name'])
        require(pack(candidate, record, 'gemm_k4').tobytes() == raw.tobytes(),
                'Original physical bytes not restored: ' + record['name'])
        logical, physical, inputs, kh, kw = dimensions(record)
        flat = raw.reshape(-1)
        # Fixed edge/interior samples exercise output-block, input-block and
        # spatial boundaries without reading a single photograph.
        indices = sorted(set((o, i, y, x)
                             for o in (0, min(3, logical - 1), logical - 1)
                             for i in (0, min(3, inputs - 1), inputs - 1)
                             for y in (0, kh - 1) for x in (0, kw - 1)))
        for o, i, y, x in indices:
            require(candidate[o, i, y, x] == flat[scalar_k4_offset(o, i, y, x, inputs, kh, kw)],
                    'Real-byte scalar K4 offset differs')
        for legacy in LEGACY:
            previous = unpack(raw, record, legacy)
            for o, i, y, x in indices:
                old_offset = scalar_legacy_offset(o, i, y, x, inputs, kh, kw, legacy,
                                                  first=record['name'] == 'conv1')
                require(previous[o, i, y, x] == flat[old_offset], 'Historical decoder behavior changed')
            require(pack(previous, record, legacy).tobytes() == raw.tobytes(), 'Historical pack changed')
        if (kh, kw) == (1, 1):
            require(np.array_equal(candidate, unpack(raw, record, 'o4_i4_hw')), 'K4 altered1x1 mapping')
        rows.append({'layer': record['index'], 'name': record['name'],
                     'weight_sha256': digest(raw.tobytes()), 'root_helper_equal': True,
                     'roundtrip_identical': True, 'scalar_samples': len(indices),
                     'legacy_decoders_scalar_samples_and_roundtrips_pass': True})
    return rows


def tensor_fingerprint(value):
    value = value.detach().cpu().contiguous()
    return {'shape': list(value.shape), 'sha256': digest(value.numpy().tobytes())}


def synthetic_network_check(data):
    core = CandidateNetwork(data, 'gemm_k4').eval().requires_grad_(False)
    reference = build_detector(data, 'flattened_k4')
    legacy = CandidateNetwork(data, 'o4_i4_hw').eval().requires_grad_(False)
    handles = []
    try:
        biases = []
        upsample = []
        for node in core.layers:
            if node['operator_id'] != 1:
                continue
            key = str(node['index'])
            require(torch.equal(core.convs[key].bias, legacy.convs[key].bias), 'Bias changed during K4 integration')
            require(torch.equal(core.convs[key].bias, reference.model.convs[key].bias), 'Bias differs from root helper')
            require(torch.equal(core.convs[key].weight, reference.model.convs[key].weight), 'Restored float weight differs')
            biases.append(node['index'])
            if node['name'] == 'upsample':
                require(torch.equal(core.convs[key].weight, legacy.convs[key].weight), 'Upsample changed')
                upsample.append(node['index'])
        del legacy
        captures = {'core': {}, 'reference': {}}
        for label, model in (('core', core), ('reference', reference.model)):
            for key, module in model.convs.items():
                def capture(_module, _inputs, output, label=label, key=key):
                    captures[label][key] = tensor_fingerprint(output)
                handles.append(module.register_forward_hook(capture))
        grid = torch.arange(256 * 384, dtype=torch.float32).reshape(1, 1, 256, 384)
        synthetic = torch.sin(grid * 0.031) * 64 + torch.cos(grid * 0.007) * 23
        with torch.inference_mode():
            actual, trace_actual = core(synthetic, capture_trace=True)
            expected, trace_expected = reference.model(synthetic, capture_trace=True)
        require(captures['core'] == captures['reference'], 'Per-convolution synthetic outputs differ')
        require(trace_actual == trace_expected, 'Per-operator synthetic trace differs')
        require(actual.keys() == expected.keys(), 'RPN head names differ')
        for name in actual:
            require(torch.equal(actual[name], expected[name]), 'Synthetic RPN output differs: ' + name)
        ordinary_layouts = [x for x in core.layout_hypotheses if x['name'] != 'upsample']
        require(all(x['layout'] == 'gemm_k4_IHW_candidate' for x in ordinary_layouts), 'K4 provenance missing')
        return {'input': 'deterministic sin/cos prepared tensor, not a photograph',
                'input_sha256': digest(synthetic.numpy().tobytes()),
                'bias_layers_unchanged': biases, 'upsample_layers_unchanged': upsample,
                'per_convolution_outputs_bitwise_equal': captures['core'],
                'per_operator_trace_equal': True,
                'rpn_outputs_bitwise_equal': {name: tensor_fingerprint(value) for name, value in actual.items()},
                'ordinary_convolution_provenance': 'gemm_k4_IHW_candidate'}
    finally:
        for handle in handles:
            handle.remove()
        reference.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_firmware_arguments(parser)
    parser.add_argument('--output', type=Path, default=BASE / 'artifacts/recognition-diagnosis-v3/core-integration/verification.json')
    args = parser.parse_args()
    require(not args.output.exists(), 'Preserve existing verification; choose a new output path')
    require(SPATIAL_LAYOUTS == LEGACY, 'Historical layout sweep changed')
    require(PHYSICAL_LAYOUTS == LEGACY + ('gemm_k4',), 'Explicit physical choices differ')
    require(inspect.signature(unpack).parameters['spatial_layout'].default == 'o4_i4_hw', 'unpack default changed')
    require(inspect.signature(pack).parameters['spatial_layout'].default == 'o4_i4_hw', 'pack default changed')
    require(inspect.signature(CandidateNetwork).parameters['spatial_layout'].default == 'hw_o4_i4', 'Network default changed')
    source_files = ('verify_gemm_k4.py', 'packing_investigation.py', 'candidate_network.py',
                    'decode_parameters.py', 'extract.py', 'ops_investigation.py', 'flattened_k_diagnosis.py')
    hashes = {name: digest((BASE / name).read_bytes()) for name in source_files}
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    fixtures, rejected = synthetic_address_checks()
    models = {}
    source = resolve_source(args)
    for stem, expected_sha in MODELS.items():
        data = (source / (stem + '.bin')).read_bytes()
        require(digest(data) == expected_sha, 'Wrong pinned component')
        byte_checks = original_byte_checks(data)
        models[stem] = {'source_sha256': expected_sha, 'ordinary_convolution_count': len(byte_checks),
                        'byte_checks': byte_checks, 'synthetic_network_check': synthetic_network_check(data)}
        print(stem, len(byte_checks), 'ordinary convolution byte checks and synthetic forward passed', flush=True)
    require(sum(x['ordinary_convolution_count'] for x in models.values()) == 126, 'Pinned layer count changed')
    require(all(digest((BASE / name).read_bytes()) == value for name, value in hashes.items()), 'Sources changed during verification')
    report = {'schema_version': 1, 'passed': True, 'source_sha256': hashes,
              'scope': {'photos_read': 0, 'AP_evaluations': 0, 'optimizer_steps': 0,
                        'original_DSP_execution_verified': False},
              'legacy_defaults_and_sweep_unchanged': True,
              'physical_layout_choices': list(PHYSICAL_LAYOUTS),
              'scalar_address_fixtures': fixtures, 'rejection_checks': rejected, 'models': models,
              'limitation': 'Byte preservation and synthetic parity with the independent candidate prove implementation consistency, not original XM4 semantics or firmware deployability.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'report': str(args.output), 'sha256': digest(args.output.read_bytes())}, ensure_ascii=False))


if __name__ == '__main__':
    main()
