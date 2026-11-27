#!/usr/bin/env python3
"""Bounded real-data fine-tuning of an explicitly unverified S5 44 candidate.

No camera execution, quantization, firmware output, or physical focus control.
Only original-sized detection heads are trained; all restored backbone values stay
frozen. Layout and checkpoint selection use validation only; test runs at finish.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F

from candidate_network import CandidateNetwork
from decode_parameters import recover
from extract import MODELS, require, write_json
from packing_investigation import SPATIAL_LAYOUTS
from pretraining_data import evaluate_detection, sha256
from proposal_hypothesis import anchors, nms
from research_paths import add_firmware_arguments, resolve_source

MODEL = '44_hm_d_nw_sng'
CLASSES = ('person', 'person_face')
HEADS = {'person': '86', 'person_face': '90', 'bbox': '94'}


def source_hashes():
    base = Path(__file__).resolve().parent
    return {name: sha256(base/name) for name in (
        'train_recognition.py', 'candidate_network.py', 'decode_parameters.py',
        'extract.py', 'ops_investigation.py', 'packing_investigation.py',
        'proposal_hypothesis.py', 'pretraining_data.py', 'pretraining_readiness.py')}


def box_iou(a, b):
    """Continuous half-open xyxy, matching dataset and metric coordinates."""
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.maximum(rb-lt, 0).prod(2)
    area_a = np.maximum(a[:, 2:]-a[:, :2], 0).prod(1)
    area_b = np.maximum(b[:, 2:]-b[:, :2], 0).prod(1)
    return inter / np.maximum(area_a[:, None]+area_b[None, :]-inter, 1e-12)


def encode_boxes(ref, gt):
    size = ref[:, 2:]-ref[:, :2]
    gt_size = gt[:, 2:]-gt[:, :2]
    center = (ref[:, :2]+ref[:, 2:])/2
    gt_center = (gt[:, :2]+gt[:, 2:])/2
    return np.concatenate(((gt_center-center)/size, np.log(gt_size/size)), axis=1)


def decode_boxes(ref, delta):
    size = ref[:, 2:]-ref[:, :2]
    center = (ref[:, :2]+ref[:, 2:])/2 + delta[:, :2]*size
    # Training experiment safeguard, fixed before any validation/test measurement.
    out_size = np.exp(np.clip(delta[:, 2:], -8, 8))*size
    box = np.concatenate((center-out_size/2, center+out_size/2), axis=1)
    box[:, [0, 2]] = box[:, [0, 2]].clip(0, 384)
    box[:, [1, 3]] = box[:, [1, 3]].clip(0, 256)
    return box


def flat_logits(raw):
    return raw[0].reshape(2, 25, 16, 24).permute(2, 3, 1, 0).reshape(-1, 2)


def flat_deltas(raw):
    return raw[0].permute(1, 2, 0).reshape(-1, 4)


class Detector:
    def __init__(self, data, layout):
        self.model = CandidateNetwork(data, layout)
        self.layout = layout
        self.raw = {}
        self.hooks = []
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for name, key in HEADS.items():
            module = self.model.convs[key]
            for parameter in module.parameters():
                parameter.requires_grad_(True)
            def save(_module, _inputs, output, name=name):
                self.raw[name] = output
            self.hooks.append(module.register_forward_hook(save))
        params, _ = recover(data)
        node = next(n for n in self.model.layers if n['operator_id'] == 24)
        p = params['proposal_blocks'][0]
        self.anchors = anchors(node['anchor_base_size_candidate'], p['ratio_candidates'],
                               p['scale_candidates'], 16, 24, node['feature_stride_candidate']).astype(np.float32)
        # Recovered candidate anchors are inclusive: convert once to continuous.
        self.anchors[:, 2:] += 1
        self.metadata = {'class_head_mapping_hypothesis': HEADS,
                         'anchor_ratios': p['ratio_candidates'], 'anchor_scales': p['scale_candidates'],
                         'anchor_stride': [16, 16], 'anchor_count': len(self.anchors),
                         'coordinates': 'inclusive recovered anchor -> continuous half-open xyxy',
                         'frozen_parameter_count': sum(p.numel() for p in self.model.parameters() if not p.requires_grad),
                         'trainable_parameter_count': sum(p.numel() for p in self.model.parameters() if p.requires_grad)}

    def __call__(self, tensor):
        self.raw.clear()
        self.model(tensor)
        return {cls: flat_logits(self.raw[cls]) for cls in CLASSES}, flat_deltas(self.raw['bbox'])

    def close(self):
        for hook in self.hooks:
            hook.remove()
        self.raw.clear()


def load_image(root, frame):
    # Open Images annotations address stored pixels. Do not silently rotate EXIF.
    with Image.open(root/frame['path']) as image:
        require(image.size == (frame['width'], frame['height']), 'Dimensions changed')
        gray = image.convert('RGB').convert('L').resize((384, 256), Image.Resampling.BILINEAR)
        pixels = np.asarray(gray, np.float32).copy()-117.0
    return torch.from_numpy(pixels)[None, None]


def ground_truth(frame):
    scale = np.array([384/frame['width'], 256/frame['height']]*2, np.float32)
    return {cls: np.array([b['bbox_xyxy'] for b in frame['boxes'] if b['class'] == cls],
                          dtype=np.float32).reshape(-1, 4)*scale for cls in CLASSES}


def assign_targets(ref, gt):
    """Legacy single-class assignment, retained only for diagnostic comparison.

    The training loss uses assign_joint_targets instead. Independent calls here
    for each class and the union must never be combined as training targets.
    """
    center = (ref[:, :2]+ref[:, 2:])/2
    valid = (center[:, 0] >= 0)&(center[:, 0] < 384)&(center[:, 1] >= 0)&(center[:, 1] < 256)
    labels = np.full(len(ref), -1, np.int64)
    matched = np.zeros(len(ref), np.int64)
    if not len(gt):
        labels[valid] = 0
        return labels, matched
    overlap = box_iou(ref, gt)
    overlap[~valid] = -1
    matched = overlap.argmax(1)
    best = overlap.max(1)
    labels[valid & (best < .3)] = 0
    labels[valid & (best >= .5)] = 1
    labels[overlap.argmax(0)] = 1
    return labels, matched


def assign_joint_targets(ref, gt):
    """One GT owner per positive anchor, shared by classification and regression.

    Ordinary positives use the greatest union IoU (>= .5); union IoU < .3 is
    background, the intermediate band and out-of-image centers are ignored.
    A positive anchor has exactly one positive class, belonging to its bbox GT.

    Reserve a distinct positive-IoU anchor for every GT whenever the bipartite
    candidate graph permits it. Deterministic augmenting paths maximize coverage;
    candidate anchors rank by IoU and GTs with fewer choices are attempted first.
    This prioritizes coverage, not a globally optimal sum of IoUs. Unmatched GTs
    are returned explicitly, and the training loss rejects them rather than
    silently training a face-positive anchor against a person bbox.
    """
    ref = np.asarray(ref, dtype=np.float32)
    require(ref.ndim == 2 and ref.shape[1] == 4 and np.isfinite(ref).all(), 'Invalid anchors')
    require(np.all(ref[:, 2:] > ref[:, :2]), 'Anchors require positive area')
    grouped = [np.asarray(gt[c], dtype=np.float32).reshape(-1, 4) for c in CLASSES]
    boxes = np.concatenate(grouped, axis=0)
    require(np.isfinite(boxes).all() and np.all(boxes[:, 2:] > boxes[:, :2]), 'Invalid GT boxes')
    classes = np.concatenate([np.full(len(rows), i, np.int64) for i, rows in enumerate(grouped)])
    center = (ref[:, :2]+ref[:, 2:])/2
    valid = (center[:, 0] >= 0)&(center[:, 0] < 384)&(center[:, 1] >= 0)&(center[:, 1] < 256)
    labels = {c: np.full(len(ref), -1, np.int64) for c in CLASSES}
    matched = np.full(len(ref), -1, np.int64)
    forced_by_gt = np.full(len(boxes), -1, np.int64)
    forced_iou = np.full(len(boxes), np.nan, np.float32)
    if len(boxes) and len(ref):
        overlap = box_iou(ref, boxes)
        overlap[~valid] = -1
        best_gt = overlap.argmax(1)
        best_iou = overlap.max(1)
        positive = valid & (best_iou >= .5)
        matched[positive] = best_gt[positive]
        background = valid & (best_iou < .3)
        choices = [np.flatnonzero(overlap[:, i] > 0) for i in range(len(boxes))]
        choices = [indices[np.argsort(-overlap[indices, i], kind='stable')]
                   for i, indices in enumerate(choices)]
        owner = np.full(len(ref), -1, np.int64)

        def reserve(gt_index, visited):
            # Keep earlier scarce/small-GT reservations when a free overlapping
            # anchor exists. Only displace an owner when needed for coverage.
            for anchor in choices[gt_index]:
                if not visited[anchor] and owner[anchor] < 0:
                    owner[anchor] = gt_index
                    forced_by_gt[gt_index] = anchor
                    return True
            for anchor in choices[gt_index]:
                if visited[anchor]:
                    continue
                visited[anchor] = True
                previous = owner[anchor]
                if previous < 0 or reserve(int(previous), visited):
                    owner[anchor] = gt_index
                    forced_by_gt[gt_index] = anchor
                    return True
            return False

        # GT order does not privilege the person class over small nested faces.
        order = sorted(range(len(boxes)), key=lambda i: (len(choices[i]),
                       float(np.prod(boxes[i, 2:]-boxes[i, :2])), i))
        for gt_index in order:
            reserve(gt_index, np.zeros(len(ref), dtype=bool))
        for gt_index, anchor in enumerate(forced_by_gt):
            if anchor >= 0:
                matched[anchor] = gt_index
                forced_iou[gt_index] = overlap[anchor, gt_index]
    else:
        background = valid if not len(boxes) else np.zeros(len(ref), dtype=bool)
    positive = matched >= 0
    for class_index, cls in enumerate(CLASSES):
        labels[cls][background] = 0
        labels[cls][positive] = (classes[matched[positive]] == class_index).astype(np.int64)
    return {'labels': labels, 'matched_gt': matched, 'gt_boxes': boxes,
            'gt_classes': classes, 'forced_anchor_by_gt': forced_by_gt,
            'forced_match_ious': forced_iou,
            'unmatched_gt_indices': np.flatnonzero(forced_by_gt < 0),
            'valid_anchors': valid}


def sample_with_required(candidates, required, limit, rng):
    """Keep all coverage anchors; exceeding the unchanged budget is an error."""
    candidates = np.asarray(candidates, dtype=np.int64)
    required = np.unique(np.asarray(required, dtype=np.int64))
    require(np.isin(required, candidates).all(), 'Required anchor is not a candidate')
    require(len(required) <= limit,
            'GT coverage requires %d anchors, exceeding the fixed budget %d' % (len(required), limit))
    remaining = candidates[~np.isin(candidates, required)]
    count = min(limit-len(required), len(remaining))
    if len(remaining) > count:
        remaining = rng.choice(remaining, count, replace=False)
    return np.concatenate((required, remaining))


def detection_loss(logits, delta, ref, gt, rng):
    assigned = assign_joint_targets(ref, gt)
    missing = assigned['unmatched_gt_indices']
    require(not len(missing), 'Unmatched GTs have no distinct positive-IoU anchor: '+str(missing.tolist()))
    pieces, info, sampled_positives = [], {'target_gt_count': len(assigned['gt_boxes']),
                                         'unmatched_gt_count': len(missing)}, []
    for class_index, cls in enumerate(CLASSES):
        labels = assigned['labels'][cls]
        pos = np.flatnonzero(labels == 1)
        neg = np.flatnonzero(labels == 0)
        required = assigned['forced_anchor_by_gt'][assigned['gt_classes'] == class_index]
        # Preserve every GT's reserved anchor within the original 32/128 budgets.
        pos = sample_with_required(pos, required, 32, rng)
        sampled_positives.extend(pos.tolist())
        if len(neg) > 128-len(pos):
            neg = rng.choice(neg, 128-len(pos), replace=False)
        indices = np.concatenate((pos, neg))
        loss = (F.cross_entropy(logits[cls][indices], torch.from_numpy(labels[indices]))
                if len(indices) else logits[cls].sum()*0)
        pieces.append(loss)
        info[cls+'_ce'] = float(loss.detach())
        info[cls+'_positive_anchors'] = len(pos)
        info[cls+'_forced_positive_anchors'] = len(required)
        info[cls+'_supervised_gt'] = len(np.unique(assigned['matched_gt'][pos]))
    matches, combined = assigned['matched_gt'], assigned['gt_boxes']
    positive = sample_with_required(np.flatnonzero(matches >= 0), sampled_positives, 128, rng)
    if len(positive):
        targets = torch.from_numpy(encode_boxes(ref[positive], combined[matches[positive]]))
        regression = F.smooth_l1_loss(delta[positive], targets, beta=1/9)
    else:
        regression = delta.sum()*0
    info['bbox_smooth_l1'] = float(regression.detach())
    info['bbox_positive_anchors'] = len(positive)
    info['bbox_required_classification_positives'] = len(set(sampled_positives))
    info['bbox_supervised_gt'] = len(np.unique(matches[positive]))
    total = sum(pieces)+regression
    info['loss'] = float(total.detach())
    return total, info


def detections(logits, delta, ref, frame):
    boxes = decode_boxes(ref, delta.detach().numpy())
    valid = np.flatnonzero(np.all(boxes[:, 2:]-boxes[:, :2] > 0, axis=1))
    result = []
    scale = np.array([frame['width']/384, frame['height']/256]*2)
    for cls in CLASSES:
        scores = logits[cls].softmax(1)[:, 1].detach().numpy()
        order = valid[np.argsort(-scores[valid], kind='stable')[:3000]]
        # Legacy NMS uses inclusive extents. Subtract one to exactly reproduce
        # continuous intersection/union while retaining its stable sort.
        keep = nms(boxes[order]-np.array([0, 0, 1, 1]), scores[order], .3, 30)
        for index in order[keep]:
            result.append({'class': cls, 'bbox_xyxy': (boxes[index]*scale).tolist(),
                           'score': float(scores[index])})
    return result


def evaluate(detector, dataset, root, manifest_hash, split, output=None):
    detector.model.eval()
    rows, losses = [], []
    rng = np.random.default_rng(20261127)  # identical evaluation anchor sample
    with torch.inference_mode():
        for frame in dataset['frames']:
            if frame['split'] != split:
                continue
            logits, delta = detector(load_image(root, frame))
            _, info = detection_loss(logits, delta, detector.anchors, ground_truth(frame), rng)
            losses.append(info['loss'])
            rows.append({'id': frame['id'], 'detections': detections(logits, delta, detector.anchors, frame)})
    prediction = {'dataset_manifest_sha256': manifest_hash, 'source_model_sha256': MODELS[MODEL],
                  'model_state_sha256': state_digest(detector.model.state_dict()),
                  'layout_hypothesis': detector.layout, 'frames': rows}
    metrics = evaluate_detection(dataset, prediction, split)
    metrics['sampled_anchor_loss'] = float(np.mean(losses))
    if output:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output/'predictions.json', prediction)
        write_json(output/'metrics.json', metrics)
    return metrics, prediction


def state_digest(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save_checkpoint(path, detector, optimizer, epoch, global_step, best_ap, config, rng):
    checkpoint = {'schema_version': 1, 'model_state': detector.model.state_dict(),
                  'optimizer_state': optimizer.state_dict(), 'completed_epoch': epoch,
                  'global_step': global_step, 'best_validation_ap50': best_ap, 'config': config,
                  'torch_rng_state': torch.get_rng_state(), 'numpy_rng_state': rng.bit_generator.state,
                  'source_hashes': source_hashes()}
    temporary = path.with_suffix('.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def visuals(dataset, root, before, after, path):
    b = {r['id']: r['detections'] for r in before['frames']}
    a = {r['id']: r['detections'] for r in after['frames']}
    # First six test IDs, fixed independently of output quality.
    frames = sorted((f for f in dataset['frames'] if f['split'] == 'test'), key=lambda f: f['id'])[:6]
    canvas = Image.new('RGB', (1152, len(frames)*280), 'white')
    for row, frame in enumerate(frames):
        for col, (label, items) in enumerate((('Ground truth', frame['boxes']),
                                             ('Before (score >= 0.5)', b[frame['id']]),
                                             ('After (score >= 0.5)', a[frame['id']]))):
            with Image.open(root/frame['path']) as im:
                tile = im.convert('RGB').resize((384, 256))
            draw = ImageDraw.Draw(tile)
            for item in items:
                if item.get('score', 1) < .5:
                    continue
                box = np.array(item['bbox_xyxy'])*np.array([384/frame['width'], 256/frame['height']]*2)
                color = '#00ff80' if item['class'] == 'person' else '#ffb000'
                draw.rectangle(tuple(box), outline=color, width=2)
            canvas.paste(tile, (col*384, row*280+24))
            ImageDraw.Draw(canvas).text((col*384+4, row*280+5), label+' | '+frame['id'], fill='black')
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_firmware_arguments(parser)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--accumulate', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=20261127)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--skip-test', action='store_true', help='For interruption/resume verification, no test evaluation')
    args = parser.parse_args()
    require(args.epochs > 0 and args.accumulate > 0 and args.lr > 0, 'Invalid training settings')
    from pretraining_readiness import validate_recognition_dataset
    validation, dataset = validate_recognition_dataset(args.dataset)
    require(not validation['data_gaps'], 'Recognition data coverage gaps: '+str(validation['data_gaps']))
    data = (resolve_source(args)/(MODEL+'.bin')).read_bytes()
    require(hashlib.sha256(data).hexdigest() == MODELS[MODEL], 'Unknown source model')
    require(dataset['classes'] == list(CLASSES), 'First run requires person/person_face class ordering')
    start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(args.seed)
    manifest_hash = sha256(args.dataset)
    root = args.dataset.parent
    config = {'model': MODEL, 'model_sha256': MODELS[MODEL], 'manifest_sha256': manifest_hash,
              'seed': args.seed, 'epochs': args.epochs, 'accumulate': args.accumulate,
              'lr': args.lr, 'threads': args.threads, 'device': 'cpu',
              'optimizer': 'AdamW', 'weight_decay': 0.0001, 'gradient_clip_norm': 10,
              'preprocessing': 'stored pixels RGB->L, bilinear stretch 384x256, subtract117, scale1; unverified',
              'trainable': 'restored rpn_cls_score1/2 and rpn_bbox_pred only; frozen backbone',
              'target': 'unique positive-IoU GT coverage matching; union IoU>=0.5 positives, <0.3 background; classification and bbox share one GT owner; preserve coverage within 128 anchors/class <=32 positives and 128 bbox anchors',
              'loss': 'CE person + CE person_face + SmoothL1 bbox beta=1/9',
              'inference': 'continuous xyxy, exp delta clamp[-8,8], preNMS3000/NMS0.3/top30 per class, no score cutoff for AP',
              'selection': 'best untrained layout by validation meanAP50 (tie: lower loss); best epoch by validation meanAP50',
              'validation': validation, 'source_hashes': source_hashes(),
              'environment': {'python': platform.python_version(), 'torch': torch.__version__,
                              'numpy': np.__version__, 'platform': platform.platform(), 'cpu_count': os.cpu_count(),
                              'mps_available': torch.backends.mps.is_available()}}
    comparison_path = args.output/'layout-comparison.json'
    initial_epoch, global_step, best_ap = 0, 0, -1.
    if args.resume:
        require(args.resume.resolve() == (args.output/'last.pt').resolve(),
                'Resume must use the current last.pt in the original output directory')
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        old = checkpoint['config']
        for key in ('model_sha256', 'manifest_sha256', 'seed', 'accumulate', 'lr', 'threads', 'source_hashes'):
            require(config[key] == old[key], 'Resume configuration mismatch: '+key)
        require(comparison_path.is_file() and (args.output/'best.pt').is_file(), 'Resume requires original run directory')
        comparison = json.loads(comparison_path.read_text())
        layout = old['selected_layout_hypothesis']
        initial_epoch = checkpoint['completed_epoch']
        global_step = checkpoint['global_step']
        best_ap = checkpoint['best_validation_ap50']
        require(args.epochs >= initial_epoch, 'Epoch target precedes checkpoint')
    else:
        require(not (args.output/'last.pt').exists(), 'Output already contains a run; use --resume')
        comparison = []
        for layout in SPATIAL_LAYOUTS:
            detector = Detector(data, layout)
            metrics, _ = evaluate(detector, dataset, root, manifest_hash, 'validation', args.output/'baseline-validation'/layout)
            comparison.append({'layout': layout, 'validation': metrics})
            print(json.dumps({'phase': 'layout_baseline', 'layout': layout, 'ap50': metrics['mean_ap'],
                              'loss': metrics['sampled_anchor_loss']}), flush=True)
            detector.close()
            del detector
        write_json(comparison_path, comparison)
        layout = max(comparison, key=lambda r: (r['validation']['mean_ap'], -r['validation']['sampled_anchor_loss']))['layout']
    detector = Detector(data, layout)
    initial_state = copy.deepcopy(detector.model.state_dict())
    config['selected_layout_hypothesis'] = layout
    config['detector'] = detector.metadata
    write_json(args.output/'config.json', config)
    parameters = [p for p in detector.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=.0001)
    if args.resume:
        detector.model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        torch.set_rng_state(checkpoint['torch_rng_state'])
        rng.bit_generator.state = checkpoint['numpy_rng_state']
        # A crash may leave image logs newer than the last complete-epoch state.
        # Keep those records separately, then replay the epoch from its checkpoint.
        log_path = args.output/'training.jsonl'
        if log_path.is_file():
            complete, interrupted = [], []
            for line in log_path.read_text().splitlines(keepends=True):
                try:
                    row = json.loads(line)
                    target = complete if row['epoch'] <= initial_epoch else interrupted
                except (ValueError, KeyError):
                    target = interrupted
                target.append(line)
            if interrupted:
                with (args.output/'interrupted-records.jsonl').open('a') as saved:
                    saved.writelines(interrupted)
                log_path.write_text(''.join(complete))
    else:
        # Epoch zero is eligible: do not replace it if every trained epoch is worse.
        best_ap = next(r['validation']['mean_ap'] for r in comparison if r['layout'] == layout)
        save_checkpoint(args.output/'best.pt', detector, optimizer, 0, 0, best_ap, config, rng)
    train_frames = [f for f in dataset['frames'] if f['split'] == 'train']
    with (args.output/'training.jsonl').open('a' if args.resume else 'w') as log:
        for epoch in range(initial_epoch+1, args.epochs+1):
            detector.model.train()
            epoch_losses, grad_norms = [], []
            order = rng.permutation(len(train_frames))
            optimizer.zero_grad(set_to_none=True)
            for offset, index in enumerate(order):
                frame = train_frames[index]
                logits, delta = detector(load_image(root, frame))
                loss, info = detection_loss(logits, delta, detector.anchors, ground_truth(frame), rng)
                require(bool(torch.isfinite(loss)), 'Nonfinite loss')
                window_start = offset//args.accumulate*args.accumulate
                window_size = min(args.accumulate, len(order)-window_start)
                (loss/window_size).backward()
                if (offset+1) % args.accumulate == 0 or offset+1 == len(order):
                    norm = torch.nn.utils.clip_grad_norm_(parameters, 10, error_if_nonfinite=True)
                    require(all(p.grad is not None for p in parameters), 'Missing head gradient')
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    grad_norms.append(float(norm))
                epoch_losses.append(info['loss'])
                log.write(json.dumps({'epoch': epoch, 'image_index': offset, 'frame_id': frame['id'],
                                      'optimizer_steps': global_step, **info})+'\n')
                if (offset+1) % 80 == 0:
                    log.flush()
                    print(json.dumps({'phase': 'training', 'epoch': epoch, 'images': offset+1,
                                      'mean_loss_so_far': float(np.mean(epoch_losses))}), flush=True)
            metrics, _ = evaluate(detector, dataset, root, manifest_hash, 'validation', args.output/f'epoch-{epoch:02d}')
            improved = metrics['mean_ap'] > best_ap
            if improved:
                best_ap = metrics['mean_ap']
                save_checkpoint(args.output/'best.pt', detector, optimizer, epoch, global_step, best_ap, config, rng)
            save_checkpoint(args.output/'last.pt', detector, optimizer, epoch, global_step, best_ap, config, rng)
            record = {'epoch': epoch, 'mean_train_loss': float(np.mean(epoch_losses)),
                      'max_gradient_norm_before_clip': max(grad_norms), 'optimizer_steps': global_step,
                      'validation': metrics, 'best_validation_ap50': best_ap, 'selected': improved,
                      'elapsed_seconds': time.monotonic()-start}
            write_json(args.output/f'epoch-{epoch:02d}'/'training-summary.json', record)
            print(json.dumps({'phase': 'epoch_end', **record}), flush=True)
    trained_state = detector.model.state_dict()
    changes = {key: float((value-initial_state[key]).abs().max()) for key, value in trained_state.items()}
    frozen = [key for key in changes if key.split('.')[1] not in HEADS.values()]
    require(all(changes[key] == 0 for key in frozen), 'Frozen backbone unexpectedly changed')
    require(any(v > 0 for v in changes.values()), 'No actual parameter update')
    update_report = {'last_state_sha256': state_digest(trained_state), 'restored_state_sha256': state_digest(initial_state),
                     'max_absolute_parameter_updates': changes, 'frozen_backbone_unchanged': True,
                     'all_final_weights_finite': all(bool(torch.isfinite(v).all()) for v in trained_state.values()),
                     'optimizer_steps': global_step}
    write_json(args.output/'parameter-update-check.json', update_report)
    best = torch.load(args.output/'best.pt', map_location='cpu', weights_only=False)
    report = {'status': 'real public-data candidate detection-head fine-tuning completed',
              'config': config, 'optimizer_steps': global_step, 'training_images_seen': args.epochs*len(train_frames),
              'selected_epoch': best['completed_epoch'], 'test_evaluated': not args.skip_test,
              'best_validation_ap50': best_ap, 'elapsed_seconds_this_invocation': time.monotonic()-start,
              'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'limitations': ['Candidate architecture/layout/preprocessing and class mapping are unverified against S5.',
                             'Only detection heads were fine-tuned. No hardware or quantized export validation.',
                             'Positive-only person+face subset; no claim about absent-person false alarms or all Open Images.',
                             'No true low-light labels, tracking sequences, or physical autofocus evaluation.',
                             'AP50 is this subset continuous-box metric, not the official Open Images challenge metric.']}
    if not args.skip_test:
        # Test images and metrics are first used here, after layout/epoch decisions.
        detector.model.load_state_dict(initial_state)
        baseline, before = evaluate(detector, dataset, root, manifest_hash, 'test', args.output/'test-before')
        detector.model.load_state_dict(best['model_state'])
        final, after = evaluate(detector, dataset, root, manifest_hash, 'test', args.output/'test-after')
        report.update(test_before=baseline, test_after=final,
                      test_mean_ap50_change=final['mean_ap']-baseline['mean_ap'])
        visuals(dataset, root, before, after, args.output/'test-predictions.jpg')
        # Predeclared diagnostic: report the actual final trained weights even
        # if validation retained epoch zero. Never use this test result to select.
        if best['completed_epoch'] != args.epochs:
            last = torch.load(args.output/'last.pt', map_location='cpu', weights_only=False)
            detector.model.load_state_dict(last['model_state'])
            last_metrics, last_predictions = evaluate(detector, dataset, root, manifest_hash, 'test', args.output/'test-last-epoch')
            report['test_last_epoch_diagnostic'] = last_metrics
            report['test_last_epoch_diagnostic_scope'] = 'Fixed final epoch, not used to select layout, checkpoint or hyperparameters.'
            visuals(dataset, root, before, last_predictions, args.output/'test-last-epoch-predictions.jpg')
    report['elapsed_seconds_this_invocation'] = time.monotonic()-start
    report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report['checkpoints'] = {name: {'sha256': sha256(args.output/name), 'bytes': (args.output/name).stat().st_size}
                             for name in ('best.pt', 'last.pt')}
    write_json(args.output/'report.json', report)
    detector.close()
    print(json.dumps({'phase': 'complete', 'report': str(args.output/'report.json'),
                      'elapsed_seconds': report['elapsed_seconds_this_invocation'],
                      'test_ap50_change': report.get('test_mean_ap50_change')}), flush=True)


if __name__ == '__main__':
    main()
