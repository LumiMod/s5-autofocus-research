#!/usr/bin/env python3
"""Bounded preprocessing/activation diagnosis, without training or test access.

Development uses the first 32 train IDs. Validation is an explicit second stage
after a frozen development plan. Camera input and arithmetic remain hypotheses.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from decode_parameters import recover
from extract import MODELS, write_json
from pretraining_data import evaluate_detection, sha256
from research_paths import add_firmware_arguments, resolve_source
from train_recognition import Detector, detections

BASE = Path(__file__).resolve().parent
MODEL = '44_hm_d_nw_sng'
FILES = ('preprocessing_investigation.py', 'candidate_network.py', 'decode_parameters.py',
         'ops_investigation.py', 'packing_investigation.py', 'train_recognition.py',
         'proposal_hypothesis.py', 'pretraining_data.py', 'extract.py')


def identity(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def candidates(input_scale):
    # Small one-variable contrasts, fixed before consulting validation scores.
    baseline = dict(color='pillow_l', mean=117., scale=1., geometry='stretch')
    return {
        'baseline_gray117': baseline,
        'float601_gray117': {**baseline, 'color': 'float601'},
        'float709_gray117': {**baseline, 'color': 'float709'},
        'gray_mean0': {**baseline, 'mean': 0.},
        'gray117_times_input_scale': {**baseline, 'scale': float(input_scale)},
        'gray117_letterbox': {**baseline, 'geometry': 'letterbox'},
    }


def prepare(rgb, config):
    width, height = rgb.size
    if config['color'] == 'pillow_l':
        gray = rgb.convert('L')
    else:
        coeff = [0.299, 0.587, 0.114] if config['color'] == 'float601' else [0.2126, 0.7152, 0.0722]
        plane = np.asarray(rgb, np.float32) @ np.asarray(coeff, np.float32)
        gray = Image.fromarray(plane.astype(np.float32))
    if config['geometry'] == 'stretch':
        new_width, new_height, x, y = 384, 256, 0, 0
    else:
        ratio = min(384/width, 256/height)
        new_width, new_height = max(1, round(width*ratio)), max(1, round(height*ratio))
        x, y = (384-new_width)//2, (256-new_height)//2
    pixels = np.full((256, 384), 117., dtype=np.float32)
    pixels[y:y+new_height, x:x+new_width] = np.asarray(
        gray.resize((new_width, new_height), Image.Resampling.BILINEAR), dtype=np.float32)
    prepared = torch.from_numpy((pixels-config['mean'])*config['scale'])[None, None]
    transform = {'sx': new_width/width, 'sy': new_height/height, 'x': x, 'y': y,
                 'width': new_width, 'height': new_height, 'original_width': width, 'original_height': height}
    return prepared, transform


def matched_mean(tensor, transform):
    control = tensor.clone()
    x, y, width, height = (transform[k] for k in ('x', 'y', 'width', 'height'))
    region = control[..., y:y+height, x:x+width]
    region.fill_(float(region.mean()))
    return control


def back_to_photo(items, transform):
    output = []
    for item in items:
        box = np.asarray(item['bbox_xyxy'], dtype=np.float64)
        box[[0, 2]] = ((box[[0, 2]]-transform['x'])/transform['sx']).clip(0, transform['original_width'])
        box[[1, 3]] = ((box[[1, 3]]-transform['y'])/transform['sy']).clip(0, transform['original_height'])
        if np.all(box[2:] > box[:2]):
            output.append({**item, 'bbox_xyxy': box.tolist()})
    return output


def compressed_json(path, value):
    with gzip.GzipFile(filename=str(path), mode='wb', mtime=0) as f:
        f.write((json.dumps(value, ensure_ascii=False, separators=(',', ':'))+'\n').encode())


def self_check():
    count = 0
    cfg = candidates(.006317331455647945)
    red = Image.new('RGB', (20, 10), (255, 0, 0))
    assert abs(float(prepare(red, cfg['float601_gray117'])[0].mean())-(255*.299-117)) < 1e-4
    count += 1
    assert abs(float(prepare(red, cfg['float709_gray117'])[0].mean())-(255*.2126-117)) < 1e-4
    count += 1
    gray = Image.new('RGB', (20, 10), (117, 117, 117))
    assert bool((prepare(gray, cfg['baseline_gray117'])[0] == 0).all())
    count += 1
    tensor, tr = prepare(red, cfg['gray117_letterbox'])
    assert tr['width'] == 384 and tr['height'] == 192 and tr['y'] == 32
    assert bool((tensor[..., :32, :] == 0).all())
    count += 1
    boxes = back_to_photo([{'class': 'person', 'score': .5, 'bbox_xyxy': [0, 32, 384, 224]}], tr)
    assert np.array_equal(boxes[0]['bbox_xyxy'], [0, 0, 20, 10])
    count += 1
    varying = tensor.clone(); varying[..., 50:70, 50:70] += 20
    control = matched_mean(varying, tr)
    assert abs(float(varying.mean()-control.mean())) < 1e-4
    assert bool((control[..., :32, :] == varying[..., :32, :]).all())
    count += 1
    original, _ = prepare(red, cfg['baseline_gray117'])
    scaled, _ = prepare(red, cfg['gray117_times_input_scale'])
    assert torch.allclose(scaled, original*.006317331455647945)
    count += 1
    return count


class LayerObserver:
    def __init__(self, detector, metadata):
        self.real, self.rows, self.hooks = {}, [], []
        self.mode, self.frame_id = 'none', None
        self.info = {n['index']: n for n in detector.model.layers if n['operator_id'] == 1}
        self.scales = {n['index']: n['output_scale'] for n in metadata['convolutions']}
        for key, module in detector.model.convs.items():
            index = int(key)
            def observe(_module, _inputs, output, index=index):
                node = self.info[index]
                value = output.clamp_min(0) if node['relu_candidate'] else output
                if self.mode == 'real':
                    self.real[index] = value.detach().clone()
                elif self.mode == 'matched_mean':
                    reference = self.real[index]
                    limit = self.scales[index]*32767
                    real_rms = float(reference.square().mean().sqrt())
                    signal_rms = float((reference-value).square().mean().sqrt())
                    self.rows.append({'frame_id': self.frame_id, 'index': index, 'name': node['name'],
                        'shape': list(value.shape), 'metadata_output_scale': self.scales[index],
                        'int16_range_hypothesis_limit': limit,
                        'real_rms': real_rms, 'same_mean_rms': float(value.square().mean().sqrt()),
                        'real_max_abs': float(reference.abs().max()), 'same_mean_max_abs': float(value.abs().max()),
                        'signal_rms': signal_rms, 'signal_over_real_rms': signal_rms/max(real_rms, 1e-12),
                        'real_fraction_beyond_int16_range_hypothesis': float((reference.abs() > limit).float().mean()),
                        'same_mean_fraction_beyond_int16_range_hypothesis': float((value.abs() > limit).float().mean()),
                        'real_zero_fraction': float((reference == 0).float().mean())})
            self.hooks.append(module.register_forward_hook(observe))

    def reset_image(self, frame_id):
        self.real.clear(); self.frame_id = frame_id

    def summary(self):
        summaries = []
        for index in sorted(self.info):
            rows = [r for r in self.rows if r['index'] == index]
            summaries.append({k: rows[0][k] for k in ('index', 'name', 'shape', 'metadata_output_scale',
                                                      'int16_range_hypothesis_limit')} |
                {'images': len(rows), **{key: float(np.mean([r[key] for r in rows])) for key in (
                    'real_rms', 'same_mean_rms', 'real_max_abs', 'same_mean_max_abs', 'signal_rms',
                    'signal_over_real_rms', 'real_fraction_beyond_int16_range_hypothesis',
                    'same_mean_fraction_beyond_int16_range_hypothesis', 'real_zero_fraction')}})
        return summaries

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.real.clear()


def run_variant(detector, params, config, name, frames, root, manifest_hash, output):
    split = frames[0]['split']
    observer = LayerObserver(detector, params)
    raw_rows, transforms, input_stats = [], [], []
    modes = ('real', 'same_mean', 'fixed_gray117', 'shuffled')
    predictions = {mode: {'dataset_manifest_sha256': manifest_hash, 'frames': []} for mode in modes}
    try:
        with torch.inference_mode():
            observer.mode = 'none'
            logits, delta = detector(torch.full((1, 1, 256, 384), (117-config['mean'])*config['scale']))
            flat = detections(logits, delta, detector.anchors, {'width': 384, 'height': 256})
            for frame in frames:
                path = root/frame['path']
                assert sha256(path) == frame['sha256']
                with Image.open(path) as im:
                    assert im.size == (frame['width'], frame['height'])
                    tensor, transform = prepare(im.convert('RGB'), config)
                control = matched_mean(tensor, transform)
                observer.reset_image(frame['id'])
                observer.mode = 'real'
                logits, delta = detector(tensor)
                real = detections(logits, delta, detector.anchors, {'width': 384, 'height': 256})
                observer.mode = 'matched_mean'
                mean_logits, mean_delta = detector(control)
                same_mean = detections(mean_logits, mean_delta, detector.anchors, {'width': 384, 'height': 256})
                input_stats.append({'id': frame['id'], 'mean': float(tensor.mean()), 'std': float(tensor.std()),
                    'minimum': float(tensor.min()), 'maximum': float(tensor.max()),
                    'input_signal_rms': float((tensor-control).square().mean().sqrt()), 'transform': transform})
                raw_rows.append(real); transforms.append(transform)
                for mode, dets in [('real', real), ('same_mean', same_mean), ('fixed_gray117', flat)]:
                    predictions[mode]['frames'].append({'id': frame['id'], 'detections': back_to_photo(dets, transform)})
            # Fixed cyclic derangement of the same split/ID ordering; never pairs
            # an image with itself or mixes train and validation.
            pairings = []
            for index, (frame, transform) in enumerate(zip(frames, transforms)):
                donor = (index+1) % len(frames)
                predictions['shuffled']['frames'].append({'id': frame['id'],
                    'detections': back_to_photo(raw_rows[donor], transform)})
                pairings.append({'target_id': frame['id'], 'prediction_image_id': frames[donor]['id']})
        dataset = {'classes': ['person', 'person_face'], 'frames': frames}
        metrics = {mode: evaluate_detection(dataset, pred, split=split, score_threshold=0)
                   for mode, pred in predictions.items()}
        real_ap = metrics['real']['mean_ap']
        control_ap = max(metrics[m]['mean_ap'] for m in modes if m != 'real')
        result = {'variant': name, 'configuration': config, 'split': split, 'images': len(frames),
            'metrics': metrics, 'real_minus_strongest_control_map50': real_ap-control_ap,
            'layers_after_candidate_relu': observer.summary(),
            'limits': ['Layer range bound assumes signed int16 activation times metadata output scale; precision/saturation are unverified.',
                       'Same-mean control retains letterbox geometry, but removes within-image structure.',
                       'Shuffled control pairs different images while retaining dataset image distribution.',
                       'Positive-only small dataset and unverified graph can produce spatial-prior AP.']}
        folder = output/name; folder.mkdir(parents=True, exist_ok=True)
        write_json(folder/'report.json', result)
        write_json(folder/'input-stats.json', input_stats)
        write_json(folder/'shuffle-pairing.json', pairings)
        compressed_json(folder/'per-image-layer-stats.json.gz', observer.rows)
        compressed_json(folder/'predictions.json.gz', predictions)
        return result
    finally:
        observer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_firmware_arguments(parser)
    parser.add_argument('--dataset', type=Path, default=BASE/'private-data/recognition-v2/manifest.json')
    parser.add_argument('--output', type=Path, default=BASE/'artifacts/recognition-diagnosis/preprocessing')
    parser.add_argument('--layout', choices=('o4_i4_hw', 'hw_o4_i4', 'o4_hw_i4'), default='o4_i4_hw')
    parser.add_argument('--conv1-layout', choices=('direct', 'o4_h_o4_w'), default='direct')
    parser.add_argument('--development-report', type=Path,
                        help='Explicit second stage: verify the two frozen development candidates on validation')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    checks = self_check(); print('Preprocessing independent self-checks:', checks, flush=True)
    if args.self_check:
        return
    before = {name: sha256(BASE/name) for name in FILES}
    manifest = args.dataset.read_bytes(); manifest_hash = hashlib.sha256(manifest).hexdigest()
    source = resolve_source(args)/(MODEL+'.bin'); data = source.read_bytes()
    assert hashlib.sha256(data).hexdigest() == MODELS[MODEL]
    params, tensors = recover(data)
    configurations = candidates(params['convolutions'][0]['input_scale'])
    split = 'validation' if args.development_report else 'train'
    document = json.loads(manifest)
    frames = sorted((f for f in document['frames'] if f['split'] == split), key=lambda f: f['id'])
    del document  # test rows never reach an image loader or metric.
    if split == 'train':
        frames = frames[:32]
        selected = list(configurations)
        args.output = args.output/(args.layout+'__'+args.conv1_layout)/'development'
    else:
        frozen = json.loads(args.development_report.read_text())
        assert frozen['status'] == 'complete' and frozen['manifest_sha256'] == manifest_hash
        assert frozen['layout_hypothesis'] == args.layout and frozen['conv1_layout_hypothesis'] == args.conv1_layout
        assert frozen['source_hashes_end'] == before, 'Code changed after development selection; rerun development'
        selected = frozen['frozen_validation_candidates']
        args.output = args.output/(args.layout+'__'+args.conv1_layout)/'validation'
    assert len(frames) >= 2 and all(f['split'] == split for f in frames)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2); torch.manual_seed(20261127); torch.use_deterministic_algorithms(True)
    detector = Detector(data, args.layout)
    if args.conv1_layout != 'direct':
        record = params['convolutions'][0]
        raw = tensors[record['tensor_key']+'_weight_i16_packed']
        # Explicit investigation hypothesis, not silently replacing shared graph.
        weights = raw.reshape(8, 4, 4, 4).transpose(0, 2, 1, 3).reshape(32, 1, 4, 4)
        with torch.no_grad():
            detector.model.convs['1'].weight.copy_(torch.from_numpy(weights.copy()).float()*record['weight_scale'])
    for parameter in detector.model.parameters():
        parameter.requires_grad_(False)
    detector.model.eval(); state_before = identity(detector.model.state_dict())
    facts = {'embedded_array': params['input_candidate'],
             'input_node_output_scale_candidate': params['convolutions'][0]['input_scale'],
             'input_node_output_scale_equals_first_conv_input_scale': True,
             'embedded_value_sign': 'positive 14976, scaled by 1/128 to positive117; subtraction sign not encoded by this value alone',
             'source_model_sha256': MODELS[MODEL],
             'unverified': ['RGB-to-camera-Y conversion and range', 'mean subtraction direction and runtime use of embedded array',
                            'Whether an additional scale applies before first conv', 'crop/resize/letterbox policy',
                            'Runtime padding, zero point, rounding, activation bit width and saturation']}
    write_json(args.output/'plan.json', {'manifest_sha256': manifest_hash, 'split': split,
        'frame_ids': [f['id'] for f in frames], 'variants': selected, 'configurations': configurations,
        'selection_rule': 'Always baseline; one additional candidate maximizing train real AP minus strongest same-mean/fixed117/shuffled control, with fixed candidate order tie break. No validation selection.',
        'original_test_images_or_predictions_accessed': False, 'source_hashes': before})
    started = time.monotonic(); results = {}
    try:
        for name in selected:
            print('Diagnosing', split, name, args.layout, args.conv1_layout, flush=True)
            results[name] = run_variant(detector, params, configurations[name], name, frames,
                                        args.dataset.parent, manifest_hash, args.output)
            print(name, {m: r['mean_ap'] for m, r in results[name]['metrics'].items()}, flush=True)
        assert identity(detector.model.state_dict()) == state_before, 'Model weights changed'
        after = {name: sha256(BASE/name) for name in FILES}
        stable = before == after and sha256(args.dataset) == manifest_hash
        chosen = max((n for n in selected if n != 'baseline_gray117'),
                     key=lambda n: results[n]['real_minus_strongest_control_map50']) if split == 'train' else None
        report = {'status': 'complete' if stable else 'source_changed_rerun_required',
            'scope': 'No training; no test images/predictions; bounded preprocessing and response diagnostics.',
            'split': split, 'frame_ids': [f['id'] for f in frames], 'manifest_sha256': manifest_hash,
            'input_evidence': facts, 'layout_hypothesis': args.layout, 'conv1_layout_hypothesis': args.conv1_layout,
            'model_state_sha256': state_before, 'source_hashes_start': before, 'source_hashes_end': after,
            'independent_self_checks': checks, 'duration_seconds': time.monotonic()-started,
            'metrics': {name: {'metrics': r['metrics'], 'real_minus_strongest_control_map50': r['real_minus_strongest_control_map50']}
                        for name, r in results.items()},
            'frozen_validation_candidates': ['baseline_gray117', chosen] if split == 'train' else selected,
            'selection_limit': 'Frozen diagnostic comparison, not a claim of original preprocessing recovery or deployment validity.'}
        write_json(args.output/'report.json', report)
        if not stable:
            raise RuntimeError('Code or dataset changed during investigation')
    finally:
        detector.close()


if __name__ == '__main__':
    main()
