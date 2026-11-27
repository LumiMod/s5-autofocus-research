#!/usr/bin/env python3
"""Preflight and explicitly gated frozen-checkpoint recognition diagnostics.

Defaults to restored 44/o4_i4_hw and train/validation only. --run or a hash-bound
--checkpoint diagnoses validation. --final-test additionally requires a completed
run and its frozen best.pt. No optimizer or camera-equivalence claim.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np
import torch

from extract import write_json
from pretraining_data import evaluate_detection, sha256
from research_paths import add_firmware_arguments, resolve_source
import train_recognition as training

CLASSES = ('person', 'person_face')
THRESHOLDS = (.3, .5, .75)
LAYOUT = 'o4_i4_hw'
INFERENCE_FUNCTIONS = ('Detector', 'load_image', 'ground_truth', 'flat_logits',
                       'flat_deltas', 'decode_boxes', 'detections')
DEPENDENCIES = ('train_recognition.py', 'candidate_network.py', 'decode_parameters.py',
                'extract.py', 'ops_investigation.py', 'packing_investigation.py',
                'proposal_hypothesis.py', 'pretraining_data.py', 'research_paths.py')


def iou_matrix(a, b):
    """Independent half-open xyxy geometry; empty inputs are allowed."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    intersection = np.maximum(rb-lt, 0).prod(axis=2)
    area_a = np.maximum(a[:, 2:]-a[:, :2], 0).prod(axis=1)
    area_b = np.maximum(b[:, 2:]-b[:, :2], 0).prod(axis=1)
    return intersection/np.maximum(area_a[:, None]+area_b[None, :]-intersection, 1e-12)


def maximum_iou(candidates, truth):
    if not len(candidates):
        return np.zeros(len(truth), dtype=np.float64)
    return iou_matrix(candidates, truth).max(axis=0)


def size_group(box):
    width, height = np.maximum(np.asarray(box)[2:]-np.asarray(box)[:2], 0)
    side = float(np.sqrt(width*height))
    return ('lt16' if side < 16 else '16to32' if side < 32
            else '32to64' if side < 64 else 'ge64')


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {'ground_truth_boxes': 0}
    return {'ground_truth_boxes': len(values), 'mean_max_iou': float(values.mean()),
            'max_iou_quantiles': {str(q): float(np.quantile(values, q)) for q in (0, .1, .25, .5, .75, .9, 1)},
            'coverage': {str(t): {'count': int((values >= t).sum()), 'fraction': float((values >= t).mean())}
                         for t in THRESHOLDS}}


def coverage_summary(rows, field):
    output = {}
    for cls in CLASSES:
        matching = [row for row in rows if row['class'] == cls]
        output[cls] = summarize([row[field] for row in matching])
        output[cls]['by_gt_sqrt_area_at_384x256'] = {
            group: summarize([r[field] for r in matching if r['size_group'] == group])
            for group in ('lt16', '16to32', '32to64', 'ge64')}
    return output


def code_hashes(base):
    return {name: sha256(base/name) for name in DEPENDENCIES}


def inference_hashes():
    return {name: hashlib.sha256(inspect.getsource(getattr(training, name)).encode()).hexdigest()
            for name in INFERENCE_FUNCTIONS}


def state_sha256(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def checkpoint_binding(args, manifest_hash):
    """Validate completed-run/explicit-hash authorization before deserializing."""
    if args.final_test and args.run is None:
        raise ValueError('--final-test requires a completed --run with frozen best.pt')
    if args.run and args.expected_sha:
        raise ValueError('--expected-sha is only for --checkpoint; --run binds its completed report')
    if args.run:
        report_path = args.run/'report.json'
        report = json.loads(report_path.read_text())
        if report.get('status') != 'real public-data candidate detection-head fine-tuning completed':
            raise ValueError('Run must be complete before checkpoint controls')
        if report['config']['manifest_sha256'] != manifest_hash:
            raise ValueError('Completed run binds another dataset')
        checkpoint = args.run/'best.pt'
        expected = report['checkpoints']['best.pt']['sha256']
        binding = {'kind': 'completed_run_best', 'run_report_sha256': sha256(report_path),
                   'run_directory': str(args.run), 'expected_best_validation_ap50': report['best_validation_ap50'],
                   'selected_epoch_from_report': report.get('selected_epoch')}
    elif args.checkpoint:
        if not args.expected_sha:
            raise ValueError('--checkpoint requires --expected-sha')
        checkpoint, expected = args.checkpoint, args.expected_sha
        binding = {'kind': 'explicit_local_checkpoint_sha256'}
    else:
        if args.expected_sha:
            raise ValueError('--expected-sha requires --checkpoint')
        return None, {'kind': 'restored_original_parameters'}
    actual = sha256(checkpoint)
    if actual != expected:
        raise ValueError('Checkpoint SHA-256 mismatch')
    # Only hash-bound local training outputs are accepted; arbitrary downloaded
    # pickle/checkpoint files are not an intended input to this tool.
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = saved['config']
    if config['manifest_sha256'] != manifest_hash:
        raise ValueError('Checkpoint binds another dataset')
    if config['model'] != training.MODEL or config['model_sha256'] != training.MODELS[training.MODEL]:
        raise ValueError('Checkpoint binds another original model')
    if config['selected_layout_hypothesis'] != LAYOUT:
        raise ValueError('This diagnostic is fixed to 44/o4_i4_hw')
    if binding['kind'] == 'completed_run_best':
        if saved['best_validation_ap50'] != binding['expected_best_validation_ap50']:
            raise ValueError('Best checkpoint does not match completed validation selection')
        selected = binding['selected_epoch_from_report']
        if selected is not None and selected != saved['completed_epoch']:
            raise ValueError('Selected epoch differs from completed report')
    binding.update(checkpoint_sha256=actual, checkpoint_name=checkpoint.name,
                   completed_epoch=saved['completed_epoch'],
                   checkpoint_source_hashes=saved.get('source_hashes', config.get('source_hashes', {})),
                   model_state_sha256=state_sha256(saved['model_state']))
    return saved, binding


def with_recall(dataset, predictions, split='validation'):
    # score_threshold=0 means all retained predictions, NOT an oracle and not
    # the threshold used by the final experiment. AP remains score-ranked.
    result = evaluate_detection(dataset, predictions, split, score_threshold=0.0)
    for row in result['classes'].values():
        total = row['ground_truth_boxes']
        row['max_recall_among_retained_predictions'] = row['tp_at_score']/total if total else None
    return result


def self_check():
    count = 0
    # Hand-computed intersection=50, union=150, independent from detector IoU.
    assert abs(iou_matrix([[0, 0, 10, 10]], [[5, 0, 15, 10]])[0, 0]-1/3) < 1e-12
    count += 1
    assert np.array_equal(maximum_iou([], [[0, 0, 10, 10]]), [0])
    count += 1
    assert np.array_equal(maximum_iou([[0, 0, 10, 10], [0, 0, 5, 10]], [[0, 0, 10, 10]]), [1])
    count += 1
    assert [size_group([0, 0, x, x]) for x in (15, 16, 31, 32, 63, 64)] == [
        'lt16', '16to32', '16to32', '32to64', '32to64', 'ge64']
    count += 1
    summary = summarize([0.1, 0.3, 0.5, 0.75])
    assert [summary['coverage'][str(t)]['count'] for t in THRESHOLDS] == [3, 2, 1]
    count += 1
    # A single true positive followed by one false positive has AP=1 and recall=1.
    fixture = {'classes': list(CLASSES), 'frames': [
        {'id': 'a', 'width': 20, 'height': 20, 'split': 'validation',
         'boxes': [{'class': 'person_face', 'bbox_xyxy': [0, 0, 10, 10]}]}]}
    prediction = {'dataset_manifest_sha256': 'fixture-only', 'frames': [{'id': 'a', 'detections': [
        {'class': 'person_face', 'score': .9, 'bbox_xyxy': [0, 0, 10, 10]},
        {'class': 'person_face', 'score': .8, 'bbox_xyxy': [10, 10, 20, 20]}]}]}
    result = with_recall(fixture, prediction)['classes']['person_face']
    assert result['ap_at_iou'] == 1 and result['max_recall_among_retained_predictions'] == 1
    count += 1
    # State identity is sensitive to parameter bytes, independent of filename.
    assert state_sha256({'a': torch.tensor([1.])}) != state_sha256({'a': torch.tensor([2.])})
    count += 1
    from types import SimpleNamespace
    try:
        checkpoint_binding(SimpleNamespace(final_test=True, run=None), 'fixture')
    except ValueError:
        count += 1
    else:
        raise AssertionError('Ungated test inference must be rejected')
    import tempfile
    with tempfile.TemporaryDirectory(prefix='recognition-preflight-check-') as temporary:
        folder = Path(temporary)
        checkpoint = folder/'best.pt'
        state = {'a': torch.tensor([1.])}
        torch.save({'config': {'model': training.MODEL, 'model_sha256': training.MODELS[training.MODEL],
                    'manifest_sha256': 'fixture', 'selected_layout_hypothesis': LAYOUT},
                    'model_state': state, 'completed_epoch': 2, 'best_validation_ap50': .2}, checkpoint)
        saved_args = SimpleNamespace(final_test=False, run=None, checkpoint=checkpoint,
                                     expected_sha=sha256(checkpoint))
        saved, binding = checkpoint_binding(saved_args, 'fixture')
        assert binding['model_state_sha256'] == state_sha256(state)
        count += 1
        saved_args.expected_sha = 'wrong'
        try:
            checkpoint_binding(saved_args, 'fixture')
        except ValueError:
            count += 1
        else:
            raise AssertionError('Wrong checkpoint hash must be rejected')
        complete = {'status': 'running', 'config': {'manifest_sha256': 'fixture'},
                    'checkpoints': {'best.pt': {'sha256': sha256(checkpoint)}},
                    'best_validation_ap50': .2, 'selected_epoch': 2}
        write_json(folder/'report.json', complete)
        run_args = SimpleNamespace(final_test=True, run=folder, checkpoint=None, expected_sha=None)
        try:
            checkpoint_binding(run_args, 'fixture')
        except ValueError:
            count += 1
        else:
            raise AssertionError('Unfinished run must not authorize test inference')
        complete['status'] = 'real public-data candidate detection-head fine-tuning completed'
        write_json(folder/'report.json', complete)
        saved, binding = checkpoint_binding(run_args, 'fixture')
        assert binding['completed_epoch'] == 2
        count += 1
    print('Preflight independent diagnostic checks passed:', count)
    return count


def run(args):
    base = Path(__file__).resolve().parent
    started = time.monotonic()
    begin_hashes = code_hashes(base)
    begin_inference = inference_hashes()
    manifest_bytes = args.dataset.read_bytes()
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    saved, binding = checkpoint_binding(args, manifest_hash)
    evaluation_split = 'test' if args.final_test else 'validation'
    allowed_splits = ('test',) if args.final_test else ('train', 'validation')
    original = json.loads(manifest_bytes)
    # By default discard test rows before image access. The separate final-test
    # path reached here only after validating a completed, frozen run above.
    dataset = {**original, 'frames': [f for f in original['frames'] if f['split'] in allowed_splits]}
    del original
    assert set(dataset['classes']) == set(CLASSES)
    root = args.dataset.parent
    frames = dataset['frames']
    assert all(f['split'] in allowed_splits for f in frames)
    assert all(any(f['split'] == split for f in frames) for split in allowed_splits)
    # The manifest checksum binds metadata; only these train/validation file
    # checks are performed. No full-dataset validator may open test images.
    for frame in frames:
        path = (root/frame['path']).resolve()
        assert path.is_relative_to(root.resolve())
        assert sha256(path) == frame['sha256'], frame['id']
    default_output = base/'artifacts/recognition-v2-preflight'
    if args.output is None:
        args.output = default_output if saved is None else default_output/(
            'checkpoint-'+evaluation_split)/binding['checkpoint_sha256'][:16]
    if saved is not None and args.output.resolve() == default_output.resolve():
        raise ValueError('Checkpoint diagnostics must not overwrite restored-model baseline')
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output/'run-start.json', {'status': 'running', 'manifest_sha256': manifest_hash,
        'source_hashes_start': begin_hashes, 'inference_hashes_start': begin_inference,
        'model': training.MODEL, 'layout': LAYOUT, 'allowed_splits': list(allowed_splits),
        'checkpoint_binding': binding, 'test_image_inference_permitted': args.final_test})
    torch.set_num_threads(args.threads)
    torch.manual_seed(20261127)
    torch.use_deterministic_algorithms(True)
    source = resolve_source(args)/(training.MODEL+'.bin')
    detector = training.Detector(source.read_bytes(), LAYOUT)
    if saved is not None:
        detector.model.load_state_dict(saved['model_state'])
    detector.model.eval()
    actual_state_sha = state_sha256(detector.model.state_dict())
    if saved is not None:
        assert actual_state_sha == binding['model_state_sha256']
    anchor = detector.anchors.astype(np.float64)
    center = (anchor[:, :2]+anchor[:, 2:])/2
    valid = ((center[:, 0] >= 0)&(center[:, 0] < 384)&(center[:, 1] >= 0)&(center[:, 1] < 256))
    anchor = anchor[valid]
    clipped = anchor.copy()
    clipped[:, [0, 2]] = clipped[:, [0, 2]].clip(0, 384)
    clipped[:, [1, 3]] = clipped[:, [1, 3]].clip(0, 256)
    gt_rows = []
    for frame in frames:
        for cls, gt in training.ground_truth(frame).items():
            plain, bounded = maximum_iou(anchor, gt), maximum_iou(clipped, gt)
            for index, box in enumerate(gt):
                gt_rows.append({'frame_id': frame['id'], 'split': frame['split'], 'class': cls,
                    'gt_index': index, 'box_at_384x256': box.tolist(), 'size_group': size_group(box),
                    'sqrt_area': float(np.sqrt(np.prod(box[2:]-box[:2]))),
                    'unclipped_anchor_max_iou': float(plain[index]),
                    'clipped_anchor_max_iou': float(bounded[index])})
    row_index = {(row['frame_id'], row['class'], row['gt_index']): row for row in gt_rows}
    mappings = ('nominal',) if args.final_test else ('nominal', 'swapped_heads')
    predictions = {mode+'_'+mapping: {'dataset_manifest_sha256': manifest_hash,
                    'source_model_sha256': sha256(source), 'model_state_sha256': actual_state_sha,
                    'layout_hypothesis': LAYOUT, 'frames': []}
                   for mode in ('real', 'constant') for mapping in mappings}
    signal_rows = []
    with torch.inference_mode():
        const_logits, const_delta = detector(torch.zeros(1, 1, 256, 384))
        const_logits = {c: x.clone() for c, x in const_logits.items()}
        const_delta = const_delta.clone()
        const_boxes = training.decode_boxes(detector.anchors, const_delta.numpy())
        for index, frame in enumerate(f for f in frames if f['split'] == evaluation_split):
            logits, delta = detector(training.load_image(root, frame))
            real_boxes = training.decode_boxes(detector.anchors, delta.numpy())
            for mode, values in [('real', logits), ('constant', const_logits)]:
                reg = delta if mode == 'real' else const_delta
                for mapping in mappings:
                    selected = values if mapping == 'nominal' else {
                        'person': values['person_face'], 'person_face': values['person']}
                    dets = training.detections(selected, reg, detector.anchors, frame)
                    predictions[mode+'_'+mapping]['frames'].append({'id': frame['id'], 'detections': dets})
            for cls, gt in training.ground_truth(frame).items():
                all_real = maximum_iou(real_boxes, gt)
                all_const = maximum_iou(const_boxes, gt)
                signal_rows.append({'frame_id': frame['id'], 'class': cls,
                    'mean_absolute_logit_change_vs_constant': float((logits[cls]-const_logits[cls]).abs().mean()),
                    'max_absolute_logit_change_vs_constant': float((logits[cls]-const_logits[cls]).abs().max()),
                    'mean_absolute_foreground_probability_change_vs_constant': float(
                        (logits[cls].softmax(1)[:, 1]-const_logits[cls].softmax(1)[:, 1]).abs().mean())})
                for gi in range(len(gt)):
                    row = row_index[(frame['id'], cls, gi)]
                    row['real_all_decoded_boxes_max_iou'] = float(all_real[gi])
                    row['constant_all_decoded_boxes_max_iou'] = float(all_const[gi])
                    # Oracle ignores the classifier's scores and does not require
                    # one-to-one matching: this is a geometric upper bound only.
            if index % 10 == 0:
                print(evaluation_split+' images examined:', index+1, flush=True)
    detector.close()
    metrics = {name: with_recall(dataset, prediction, evaluation_split) for name, prediction in predictions.items()}
    end_hashes = code_hashes(base)
    changed = [name for name in begin_hashes if begin_hashes[name] != end_hashes[name]]
    manifest_unchanged = sha256(args.dataset) == manifest_hash
    checkpoint_unchanged = (saved is None or sha256(args.run/'best.pt' if args.run else args.checkpoint)
                            == binding['checkpoint_sha256'])
    run_report_unchanged = (args.run is None or sha256(args.run/'report.json') == binding['run_report_sha256'])
    stable = not changed and manifest_unchanged and checkpoint_unchanged and run_report_unchanged
    report = {'status': 'complete' if stable else 'needs_source_stability_review',
        'scope': '44/o4_i4_hw diagnostics; no optimizer or model selection. '+(
            'Explicit final-test control after frozen best selection; must not tune from these results.' if args.final_test
            else 'Train/validation only; no test inference.'),
        'manifest_sha256': manifest_hash, 'manifest_unchanged': manifest_unchanged,
        'diagnostic_script_sha256': sha256(Path(__file__)),
        'independent_self_checks_passed': args.self_checks_passed,
        'source_sha256': sha256(source), 'source_model_sha256': sha256(source),
        'model_state_sha256': actual_state_sha, 'checkpoint_binding': binding,
        'checkpoint_unchanged': checkpoint_unchanged, 'run_report_unchanged': run_report_unchanged,
        'layout_hypothesis': LAYOUT, 'evaluation_split': evaluation_split,
        'head_swap_scope': 'Diagnostic hypothesis only. Does not authorize changing the final class mapping.',
        'constant_input': 'Zero prepared tensor, corresponding to a flat grayscale 117 image before centering.',
        'source_hashes_start': begin_hashes, 'source_hashes_end': end_hashes,
        'inference_hashes_loaded_at_start': begin_inference, 'source_files_changed_during_run': changed,
        'split_frame_counts': {s: sum(f['split'] == s for f in frames) for s in allowed_splits},
        'test_image_inference_performed': args.final_test, 'existing_test_predictions_opened': False,
        'anchor_count': len(detector.anchors),
        'center_valid_anchor_count': len(anchor), 'metrics': metrics,
        'anchor_coverage': {s: {field: coverage_summary([r for r in gt_rows if r['split'] == s], field)
                              for field in ('unclipped_anchor_max_iou', 'clipped_anchor_max_iou')}
                            for s in allowed_splits},
        'decoded_geometry_oracle': {field: coverage_summary([r for r in gt_rows if r['split'] == evaluation_split], field)
                                    for field in ('real_all_decoded_boxes_max_iou', 'constant_all_decoded_boxes_max_iou')},
        'oracle_limit': 'Per-GT maximum IoU over all decoded boxes, regardless of score/class. Shared predictions may cover multiple GT; coverage is an upper bound, not AP or one-to-one recall.',
        'limits': ['Positive-only, small Open Images subset, not an S5 or official benchmark.',
                   'Input mean, image transform, class mapping, packed layouts and decoder are still hypotheses.',
                   'Image-sensitive predictions do not by themselves prove image-relevant detection.',
                   'No parameters were updated. Existing test predictions were never opened.',
                   'Final-test inference is allowed only by explicit flag after completed-run best checkpoint binding.'],
        'duration_seconds': time.monotonic()-started}
    for name, prediction in predictions.items():
        write_json(args.output/(name+'-'+evaluation_split+'-predictions.json'), prediction)
    write_json(args.output/'gt-coverage.json', gt_rows)
    write_json(args.output/'image-signal.json', signal_rows)
    write_json(args.output/'report.json', report)
    print(json.dumps({name: value['mean_ap'] for name, value in metrics.items()}, indent=2))
    print('Source files changed:', changed)
    assert report['manifest_unchanged'], 'Dataset changed during diagnostics'
    assert checkpoint_unchanged and run_report_unchanged, 'Checkpoint/run selection changed during diagnostics'
    return 3 if changed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_firmware_arguments(parser)
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--output', type=Path)
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument('--run', type=Path, help='Completed training run; bind its frozen best.pt')
    checkpoint.add_argument('--checkpoint', type=Path, help='Explicit local checkpoint for validation controls')
    parser.add_argument('--expected-sha', help='Required SHA-256 with --checkpoint')
    parser.add_argument('--final-test', action='store_true', help='Explicit frozen-best test control; requires completed --run')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    args.self_checks_passed = self_check()
    if args.self_check:
        return 0
    parser.error('--dataset is required') if args.dataset is None else None
    assert args.threads > 0
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
