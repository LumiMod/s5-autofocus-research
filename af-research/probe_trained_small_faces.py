#!/usr/bin/env python3
"""Final frozen train32 stage audit of v2 epoch06 float weights, no more training.

Reuses the existing stage decoder/measurement. No oracle crops, model selection,
new validation/test evaluation, optimizer, or extra candidate search.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from extract import MODELS, require, write_json
from pretraining_data import checked_file, sha256
from research_paths import add_firmware_arguments, resolve_source
from small_face_path_diagnosis import measure, plain, reference_gts, summarize
from train_recognition import Detector, MODEL, load_image, source_hashes, state_digest

BASE = Path(__file__).resolve().parent
MANIFEST_SHA = 'af5459ddebd98947b154580cd176e7846abcceffe5539ac6f0001f6a485a3fc8'


def run(args):
    require(not args.output.exists(), 'Preserve existing stage evidence')
    require(sha256(args.dataset) == MANIFEST_SHA, 'Unexpected manifest')
    dataset = json.loads(args.dataset.read_text())
    frames = sorted((f for f in dataset['frames'] if f['split'] == 'train'), key=lambda f: f['id'])[:32]
    del dataset
    require(len(frames) == 32, 'Expected fixed train32')
    train_plan_path = args.run/'plan.json'
    checkpoint_path = args.run/'epoch-06/checkpoint.pt'
    anchor_path = args.run/'anchors-small.npy'
    training_summary_path = args.run/'summary.json'
    inputs = {name: sha256(path) for name, path in (
        ('train_plan', train_plan_path), ('float_checkpoint', checkpoint_path), ('anchors', anchor_path),
        ('training_summary', training_summary_path), ('old_stage_report', args.old_report))}
    train_plan = json.loads(train_plan_path.read_text())
    train_summary = json.loads(training_summary_path.read_text())
    old = json.loads(args.old_report.read_text())
    require(train_plan['layout'] == 'gemm_k4' and train_plan['manifest_sha256'] == MANIFEST_SHA,
            'Unexpected trained layout or dataset')
    require(train_plan['small_anchor_scales'] == [1, 2, 4, 8, 16], 'Unexpected trained anchors')
    require(train_summary['status'] == 'complete' and train_summary['plan_sha256'] == inputs['train_plan'],
            'Incomplete or mismatched training run')
    # Trusted local checkpoint from the completed, hash-bound training run.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    require(checkpoint['completed_epoch'] == 6 and checkpoint['global_step'] == 480 and
            checkpoint['plan_sha256'] == inputs['train_plan'], 'Not the fixed epoch06 float checkpoint')
    require(not checkpoint.get('host_quantized_proxy', False), 'Quantized proxy is not the requested float checkpoint')
    data = (resolve_source(args)/(MODEL+'.bin')).read_bytes()
    require(hashlib.sha256(data).hexdigest() == MODELS[MODEL], 'Unexpected original model')
    torch.set_num_threads(args.threads); torch.manual_seed(0); torch.use_deterministic_algorithms(True)
    detector = Detector(data, 'gemm_k4')
    detector.model.load_state_dict(checkpoint['model_state'])
    detector.anchors = np.load(anchor_path, allow_pickle=False)
    require(detector.anchors.shape == (9600, 4) and detector.anchors.dtype == np.float32, 'Unexpected anchor array')
    anchor_sha = hashlib.sha256(detector.anchors.astype('<f4').tobytes()).hexdigest()
    require(anchor_sha == train_plan['small_anchor_sha256'] == checkpoint['anchor_sha256'], 'Anchor hash mismatch')
    detector.model.requires_grad_(False); detector.model.eval()
    before = state_digest(detector.model.state_dict())
    require(before == train_summary['epochs'][-1]['model_state_sha256'], 'Final float state differs from run summary')
    require(all(not p.requires_grad and p.grad is None for p in detector.model.parameters()), 'Parameters not frozen')
    old_rows = {r['frame_id']: r for r in old['baseline']}
    require(set(old_rows) == {f['id'] for f in frames}, 'Original comparison uses different train frames')
    small_count = sum(g['target_class'] == 'person_face' and np.sqrt(np.prod(g['box'][2:]-g['box'][:2])) < 32
                      for f in frames for g in reference_gts(f))
    require(small_count == 10, 'Expected same ten small faces')
    sources = {**source_hashes(), **{name: sha256(BASE/name) for name in (
        Path(__file__).name, 'small_face_path_diagnosis.py', 'flattened_k_diagnosis.py',
        'preprocessing_investigation.py', 'verify_external_baselines.py', 'research_paths.py')}}
    args.output.mkdir(parents=True)
    snapshot = args.output/'source-snapshot'; snapshot.mkdir()
    for name in sources:
        (snapshot/name).write_bytes((BASE/name).read_bytes())
    plan = dict(status='frozen_before_inference', split='train', images=32, small_face_targets=10,
        frames=[dict(id=f['id'], sha256=f['sha256']) for f in frames], source_hashes=sources,
        input_file_sha256=inputs, manifest_sha256=MANIFEST_SHA, checkpoint_epoch=6, checkpoint_optimizer_steps=480,
        evaluated_state_sha256=before, anchor_sha256=anchor_sha,
        preprocessing='RGB->Pillow L->bilinear stretch384x256->subtract117',
        score_threshold=.5, match_iou=.5, nms_iou=.3, pre_nms_top=3000, final_top=30,
        optimizer_steps_this_probe=0, oracle_inference=False, validation_inference=False, test_inference=False,
        comparison='Same ten train32 small faces versus original K4 original-anchor baseline stage report.',
        final_probe=True, experiment_continuation_authorized=False)
    write_json(args.output/'frozen-plan.json', plan)
    predictions = args.output/'predictions'; predictions.mkdir()
    started = time.monotonic(); rows = []; per_gt = []
    try:
        with torch.inference_mode():
            for i, frame in enumerate(frames):
                checked_file(args.dataset.parent, frame)
                tensor = load_image(args.dataset.parent, frame)
                result = measure(detector, tensor, reference_gts(frame), predictions/f'baseline-{i:02d}.npz')
                require(result['input_tensor_sha256'] == old_rows[frame['id']]['input_tensor_sha256'],
                        'Input pixels differ from original comparison')
                result.update(frame_id=frame['id'], source_image_sha256=frame['sha256'],
                              prediction_file=f'predictions/baseline-{i:02d}.npz')
                rows.append(result)
                per_gt.extend(dict(frame_id=frame['id'], **r) for r in result['per_gt'])
    finally:
        detector.close()
    after = state_digest(detector.model.state_dict())
    require(before == after, 'Model state changed during frozen probe')
    require(sha256(checkpoint_path) == inputs['float_checkpoint'] and sha256(anchor_path) == inputs['anchors'],
            'Checkpoint or anchor file changed')
    require(sha256(args.dataset) == MANIFEST_SHA and all(sha256(BASE/n) == h for n, h in sources.items()),
            'Dataset or source changed')
    summary = summarize(per_gt)
    transitions = []
    for current in per_gt:
        if current['target_class'] != 'person_face' or current['reference_sqrt_area'] >= 32:
            continue
        previous = next(r for r in old_rows[current['frame_id']]['per_gt'] if r['gt_index'] == current['gt_index'])
        require(previous['target_box'] == current['target_box'], 'GT changed between stage comparisons')
        transitions.append(dict(frame_id=current['frame_id'], gt_index=current['gt_index'],
            old_stage=previous['failure_stage'], new_stage=current['failure_stage'],
            old_best_decoded_iou=previous['decoded_all']['max_iou'], new_best_decoded_iou=current['decoded_all']['max_iou'],
            old_best_geometry_score=previous['decoded_all']['score'], new_best_geometry_score=current['decoded_all']['score']))
    report = dict(status='complete', frozen_plan_sha256=sha256(args.output/'frozen-plan.json'),
        source_hashes=sources, manifest_sha256=MANIFEST_SHA, input_file_sha256=inputs,
        checkpoint_sha256_after=sha256(checkpoint_path), checkpoint_state_sha256_before=before,
        checkpoint_state_sha256_after=after, model_frozen_and_unchanged=True,
        baseline_summary=summary, original_baseline_summary=old['baseline_summary'],
        small_face_transitions=transitions, baseline=rows,
        prediction_files={p.relative_to(args.output).as_posix(): sha256(p) for p in sorted(predictions.iterdir())},
        elapsed_seconds=time.monotonic()-started, optimizer_steps_this_probe=0,
        oracle_inference=False, validation_inference=False, test_inference=False,
        final_probe=True, ready_for_further_training=False,
        limitations=['These 32 training images are not an independent accuracy estimate.',
            'The comparison changes both anchor geometry and trained head weights; it does not isolate their separate causal effects.',
            'Stage categories refer only to the frozen host decoder, score and NMS path, not original DSP execution.',
            'No further optimizer, oracle, candidate search, validation or test evaluation is part of this final probe.'])
    write_json(args.output/'report.json', plain(report))
    print(json.dumps(dict(old_small=old['baseline_summary']['person_face/small_lt32'],
        new_small=summary['person_face/small_lt32'], report=str(args.output/'report.json'),
        sha256=sha256(args.output/'report.json'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_firmware_arguments(parser)
    parser.add_argument('--dataset', type=Path, default=BASE/'private-data/recognition-v2/manifest.json')
    parser.add_argument('--run', type=Path, default=BASE/'artifacts/recognition-small-anchor-v2')
    parser.add_argument('--old-report', type=Path,
                        default=BASE/'artifacts/recognition-diagnosis-v3/small-face-path-train32-meanstat-fix/report.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=2)
    run(parser.parse_args())
