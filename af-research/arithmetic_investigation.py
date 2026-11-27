#!/usr/bin/env python3
"""Frozen-weight diagnostic of output-range/rounding assumptions, never training.

CDNN output scales are observed fields; int16 saturation and rounding policies
below remain hypotheses. A better image score cannot establish DSP equivalence.
Only the predeclared first 32 training images and image-independent controls run.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import MethodType

import torch
from torch.nn import functional as F

from decode_parameters import output_scale
from extract import MODELS, decode, require, write_json
from pretraining_data import sha256
from research_paths import add_firmware_arguments, resolve_source
from train_recognition import CLASSES, HEADS, MODEL, Detector, flat_deltas, flat_logits
import weight_layout_investigation as packing


MODES = ('float', 'clip_signed16', 'round_signed16', 'round_unsigned_relu16')


def cast_candidate(value, scale, mode, relu=False):
    if mode == 'float':
        return value
    require(mode in MODES, 'Unknown arithmetic hypothesis')
    lo, hi = (0, 65535) if mode == 'round_unsigned_relu16' and relu else (-32768, 32767)
    raw = value / scale
    if mode.startswith('round_'):
        raw = torch.round(raw)  # ties-to-even is explicit, not a claim about XM4
    return raw.clamp(lo, hi) * scale


def candidate_forward(model, prepared_tensor, capture_trace=False):
    require(list(prepared_tensor.shape) == model.layers[0]['shape'], 'Input shape differs')
    buffers, heads = {}, {}
    model.quantized_heads = {}
    for node in model.layers:
        index, op = node['index'], node['operator_id']
        inputs = [buffers[(entry['layer'], entry['port'])] for entry in node['inputs'] if entry['layer'] is not None]
        if op == 24:
            heads[node['name'] + '_probabilities'] = inputs[0]
            heads['bbox_deltas'] = inputs[1]
            continue
        if op == 6:
            outputs = [prepared_tensor]
        elif op == 1:
            value = model.convs[str(index)](inputs[0])
            outputs = [F.relu(value) if node['relu_candidate'] else value]
        elif op == 5:
            outputs = [F.max_pool2d(inputs[0], node['kernel'], node['stride'], node['padding'], ceil_mode=True)]
        elif op == 9:
            outputs = [torch.cat(inputs, dim=node['axis'])]
        elif op == 10:
            cut = node['cuts'][0]
            outputs = list(torch.split(inputs[0], [cut, inputs[0].shape[1] - cut], dim=1))
        elif op == 8:
            outputs = [inputs[0].reshape(node['shape'])]
        elif op == 29:
            outputs = [F.softmax(inputs[0], dim=node['axis'])]
        else:
            raise ValueError('Unknown operator')
        outputs = [cast_candidate(value, model.output_scales[index], model.arithmetic_mode,
                                  bool(node.get('relu_candidate'))) for value in outputs]
        require([list(x.shape) for x in outputs] == node['output_shapes_candidate'], 'Unexpected shape')
        for port, value in enumerate(outputs):
            require(bool(torch.isfinite(value).all()), 'Nonfinite tensor')
            buffers[index, port] = value
        for name, key in HEADS.items():
            if str(index) == key:
                model.quantized_heads[name] = outputs[0]
    return heads, []


class ArithmeticDetector(Detector):
    def __init__(self, data, specification):
        super().__init__(data, 'o4_i4_hw')
        self.model.output_scales = {n['index']: output_scale(data, n) for n in decode(data)['layers']}
        self.model.arithmetic_mode = specification['arithmetic']
        self.model.forward = MethodType(candidate_forward, self.model)

    def __call__(self, tensor):
        self.raw.clear()
        self.model(tensor)
        raw = self.model.quantized_heads
        return {cls: flat_logits(raw[cls]) for cls in CLASSES}, flat_deltas(raw['bbox'])


def self_check(data):
    x = torch.tensor([-100., -2.5, -1.5, -.5, .5, 1.5, 2.5, 100.])
    require(torch.equal(cast_candidate(x, 1., 'round_signed16'), torch.tensor([-100., -2., -2., 0., 0., 2., 2., 100.])),
            'Explicit ties-to-even check failed')
    x = torch.tensor([-1e6, 1e6])
    require(torch.equal(cast_candidate(x, .25, 'clip_signed16'), torch.tensor([-8192., 8191.75])), 'Signed clipping check failed')
    require(torch.equal(cast_candidate(x, .25, 'round_unsigned_relu16', True), torch.tensor([0., 16383.75])), 'Unsigned clipping check failed')
    # Independently compare the no-quantization interpreter with the original
    # CandidateNetwork on a fixed nonuniform tensor before any photo metrics.
    original, candidate = Detector(data, 'o4_i4_hw'), ArithmeticDetector(data, {'arithmetic': 'float'})
    grid = torch.arange(256*384, dtype=torch.float32).reshape(1, 1, 256, 384)
    with torch.inference_mode():
        left, box_left = original(torch.sin(grid * .071) * 80)
        right, box_right = candidate(torch.sin(grid * .071) * 80)
    require(torch.equal(box_left, box_right) and all(torch.equal(left[c], right[c]) for c in CLASSES),
            'Floating interpreter differs from original')
    original.close()
    candidate.close()
    return 4


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_firmware_arguments(parser)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not (args.output / 'runtime-summary.json').exists(), 'Refusing to overwrite results')
    data = (resolve_source(args) / (MODEL + '.bin')).read_bytes()
    require(hashlib.sha256(data).hexdigest() == MODELS[MODEL], 'Unknown source model')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    checks = self_check(data)
    args.output.mkdir(parents=True, exist_ok=True)
    dataset = json.loads(args.dataset.read_text())
    specifications = [{'name': mode, 'arithmetic': mode, 'conv1_order': 'O_H_W',
                       'spatial_order': 'Ob_Ib_O_I_H_W'} for mode in MODES]
    binding = {'arithmetic_script_sha256': sha256(Path(__file__)),
               'packing_script_sha256': sha256(Path(packing.__file__)), 'independent_checks_passed': checks,
               'modes': MODES, 'scope': 'Fixed original weights; train32 only; no optimizer/validation/test.',
               'limitations': ['16-bit output saturation, signedness and ties-to-even are diagnostic hypotheses.',
                              'Observer convolution summaries are pre-quantization; final logits and deltas are quantized.',
                              'No DSP instruction execution or original numerical comparison.']}
    write_json(args.output / 'arithmetic-plan.json', binding)
    prior_builder = packing.build_probe_detector
    try:
        packing.build_probe_detector = ArithmeticDetector
        packing.runtime_probe(data, dataset, args.dataset.parent, sha256(args.dataset), specifications, args.output)
    finally:
        packing.build_probe_detector = prior_builder
    require(binding['arithmetic_script_sha256'] == sha256(Path(__file__)), 'Arithmetic source changed')
    require(binding['packing_script_sha256'] == sha256(Path(packing.__file__)), 'Probe harness changed')


if __name__ == '__main__':
    main()
