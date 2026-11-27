#!/usr/bin/env python3
"""Read-only integrity audit of a finished, trusted local recognition experiment.

Recomputes saved detection metrics; never selects a new model or changes weights.
This verifies experiment accounting, not original S5 numerical equivalence.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

import torch

from extract import MODELS, require, write_json
from packing_investigation import SPATIAL_LAYOUTS
from pretraining_data import evaluate_detection, sha256
from pretraining_readiness import validate_recognition_dataset
from research_paths import add_firmware_arguments, resolve_source
from train_recognition import Detector, HEADS, MODEL, source_hashes, state_digest


def read_json(path):
    return json.loads(path.read_text())


def audit(args):
    run = args.run
    report = read_json(run / 'report.json')
    config = read_json(run / 'config.json')
    require(report['config'] == config, 'Report/config mismatch')
    require(config['source_hashes'] == source_hashes(), 'Training source changed since run')
    check, dataset = validate_recognition_dataset(args.dataset)
    require(not check['data_gaps'], 'Incomplete dataset')
    require(check['manifest_sha256'] == config['manifest_sha256'], 'Dataset changed')
    require(config['model_sha256'] == MODELS[MODEL], 'Unexpected source model')
    torch.set_num_threads(config['threads'])
    checkpoints = {}
    for name in ('best.pt', 'last.pt'):
        path = run / name
        require(sha256(path) == report['checkpoints'][name]['sha256'], 'Checkpoint changed: ' + name)
        require(path.stat().st_size == report['checkpoints'][name]['bytes'], 'Checkpoint size mismatch')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        old_config = checkpoint['config']
        require({k: v for k, v in old_config.items() if k != 'epochs'} ==
                {k: v for k, v in config.items() if k != 'epochs'}, 'Checkpoint configuration differs')
        require(old_config['epochs'] <= config['epochs'], 'Checkpoint targets a later experiment')
        require(checkpoint['source_hashes'] == config['source_hashes'], 'Checkpoint source differs')
        require(all(bool(torch.isfinite(v).all()) for v in checkpoint['model_state'].values()), 'Nonfinite weights')
        checkpoints[name] = checkpoint

    layout_comparison = read_json(run / 'layout-comparison.json')
    require(Counter(row['layout'] for row in layout_comparison) == Counter(SPATIAL_LAYOUTS),
            'Missing or repeated layout baseline')
    for row in layout_comparison:
        require(row['validation'] == read_json(run / 'baseline-validation' / row['layout'] / 'metrics.json'),
                'Layout selection summary differs from saved metrics')
    selected_layout = max(layout_comparison, key=lambda row: (
        row['validation']['mean_ap'], -row['validation']['sampled_anchor_loss']))['layout']
    require(selected_layout == config['selected_layout_hypothesis'], 'Invalid layout selection')
    chosen_epoch = 0
    best_ap = next(row['validation']['mean_ap'] for row in layout_comparison if row['layout'] == selected_layout)
    train_ids = {f['id'] for f in dataset['frames'] if f['split'] == 'train'}
    logs = [json.loads(line) for line in (run / 'training.jsonl').read_text().splitlines()]
    require(len(logs) == config['epochs'] * len(train_ids), 'Wrong training image count')
    steps_per_epoch = math.ceil(len(train_ids) / config['accumulate'])
    for epoch in range(1, config['epochs'] + 1):
        rows = [row for row in logs if row['epoch'] == epoch]
        require(Counter(row['frame_id'] for row in rows) == Counter({key: 1 for key in train_ids}),
                'Missing/repeated training image')
        require([row['image_index'] for row in rows] == list(range(len(train_ids))), 'Wrong training order accounting')
        for index, row in enumerate(rows):
            completed = (index + 1) // config['accumulate']
            if index + 1 == len(train_ids):
                completed = steps_per_epoch
            require(row['optimizer_steps'] == (epoch - 1) * steps_per_epoch + completed, 'Wrong optimizer step count')
            require(math.isfinite(row['loss']), 'Nonfinite logged training loss')
        summary = read_json(run / f'epoch-{epoch:02d}' / 'training-summary.json')
        require(summary['validation'] == read_json(run / f'epoch-{epoch:02d}' / 'metrics.json'),
                'Epoch selection summary differs from saved metrics')
        require(summary['epoch'] == epoch and summary['optimizer_steps'] == epoch * steps_per_epoch,
                'Epoch summary progress differs')
        improved = summary['validation']['mean_ap'] > best_ap
        if improved:
            chosen_epoch, best_ap = epoch, summary['validation']['mean_ap']
        require(summary['selected'] == improved, 'Invalid checkpoint selection log')
        require(summary['best_validation_ap50'] == best_ap, 'Invalid best validation score')
        require(math.isclose(summary['mean_train_loss'], sum(row['loss'] for row in rows) / len(rows), abs_tol=1e-12),
                'Training loss summary differs')
    require(report['selected_epoch'] == checkpoints['best.pt']['completed_epoch'] == chosen_epoch,
            'Saved model is not the validation-selected epoch')
    require(report['best_validation_ap50'] == best_ap, 'Best score differs')
    require(report['optimizer_steps'] == checkpoints['last.pt']['global_step'] == config['epochs'] * steps_per_epoch,
            'Final optimizer count differs')
    require(checkpoints['last.pt']['completed_epoch'] == config['epochs'], 'Last checkpoint incomplete')
    for name, checkpoint in checkpoints.items():
        expected_step = checkpoint['completed_epoch'] * steps_per_epoch
        require(checkpoint['global_step'] == expected_step, 'Checkpoint progress differs: ' + name)
        require(checkpoint['best_validation_ap50'] == best_ap, 'Checkpoint best score differs: ' + name)
        optimizer = checkpoint['optimizer_state']
        parameter_ids = [identifier for group in optimizer['param_groups'] for identifier in group['params']]
        states = optimizer['state']
        if expected_step == 0:
            require(not states, 'Epoch-zero optimizer has updated state')
        else:
            require(set(states) == set(parameter_ids), 'Optimizer state missing a parameter')
            require(all(float(state['step']) == expected_step for state in states.values()), 'Optimizer step mismatch')
            require(all(bool(torch.isfinite(state[key]).all()) for state in states.values()
                        for key in ('exp_avg', 'exp_avg_sq')), 'Nonfinite optimizer moments')

    source = resolve_source(args) / (MODEL + '.bin')
    require(sha256(source) == config['model_sha256'], 'Original component changed')
    detector = Detector(source.read_bytes(), selected_layout)
    initial = detector.model.state_dict()
    last = checkpoints['last.pt']['model_state']
    require(set(last) == set(initial), 'Network topology changed')
    changed = [key for key in initial if not torch.equal(initial[key], last[key])]
    require(changed and all(key.split('.')[1] in HEADS.values() for key in changed), 'Unexpected or absent parameter updates')
    update = read_json(run / 'parameter-update-check.json')
    require(state_digest(initial) == update['restored_state_sha256'], 'Restored state digest differs')
    require(state_digest(last) == update['last_state_sha256'], 'Final state digest differs')
    initial_digest = state_digest(initial)
    best_digest = state_digest(checkpoints['best.pt']['model_state'])
    best_state = checkpoints['best.pt']['model_state']
    require(set(best_state) == set(initial), 'Best model topology differs')
    require(all(torch.equal(initial[key], best_state[key]) for key in initial if key.split('.')[1] not in HEADS.values()),
            'Best model frozen backbone changed')
    if chosen_epoch == 0:
        require(best_digest == initial_digest, 'Selected epoch zero is not the original state')
    detector.close()

    baseline_digests = {selected_layout: initial_digest}
    for layout in SPATIAL_LAYOUTS:
        if layout != selected_layout:
            restored = Detector(source.read_bytes(), layout)
            baseline_digests[layout] = state_digest(restored.model.state_dict())
            restored.close()

    metric_checks = []
    directories = [(run / 'baseline-validation' / row['layout'], 'validation', baseline_digests[row['layout']], row['layout'])
                   for row in layout_comparison]
    directories += [(run / f'epoch-{epoch:02d}', 'validation',
                     best_digest if epoch == chosen_epoch else state_digest(last) if epoch == config['epochs'] else None,
                     selected_layout) for epoch in range(1, config['epochs'] + 1)]
    if report.get('test_evaluated', 'test_before' in report):
        directories += [(run / 'test-before', 'test', initial_digest, selected_layout),
                        (run / 'test-after', 'test', best_digest, selected_layout)]
    else:
        require(not any((run / name).exists() for name in ('test-before', 'test-after', 'test-last-epoch')),
                'Test outputs exist despite an unevaluated-test report')
    if 'test_last_epoch_diagnostic' in report:
        directories.append((run / 'test-last-epoch', 'test', state_digest(last), selected_layout))
    for directory, split, expected_state, expected_layout in directories:
        prediction = read_json(directory / 'predictions.json')
        metrics = read_json(directory / 'metrics.json')
        require(prediction['dataset_manifest_sha256'] == config['manifest_sha256'], 'Prediction dataset mismatch')
        require(prediction['source_model_sha256'] == config['model_sha256'], 'Prediction source mismatch')
        require(prediction['layout_hypothesis'] == expected_layout, 'Prediction layout differs')
        if expected_state:
            require(prediction['model_state_sha256'] == expected_state, 'Prediction weight identity differs')
        recomputed = evaluate_detection(dataset, prediction, split)
        require(all(metrics[key] == value for key, value in recomputed.items()), 'Saved metrics disagree with predictions')
        metric_checks.append(str(directory.relative_to(run)))
    for directory, field in (('test-before', 'test_before'), ('test-after', 'test_after'),
                             ('test-last-epoch', 'test_last_epoch_diagnostic')):
        if field in report:
            require(read_json(run / directory / 'metrics.json') == report[field], 'Report metrics differ')

    result = {'passed': True, 'scope': 'Finished experiment bookkeeping and saved-prediction evaluation; not original-camera fidelity.',
              'manifest_sha256': config['manifest_sha256'], 'training_source_hashes': source_hashes(),
              'audit_script_sha256': sha256(Path(__file__)), 'optimizer_steps': report['optimizer_steps'],
              'training_image_visits': len(logs), 'selected_epoch': chosen_epoch,
              'frozen_backbone_unchanged': True, 'changed_tensor_names': changed,
              'restored_state_sha256': initial_digest, 'best_state_sha256': best_digest,
              'last_state_sha256': state_digest(last), 'recomputed_metric_directories': metric_checks,
              'test_evaluated': report.get('test_evaluated', 'test_before' in report),
              'best_test_ap50_change': (report['test_after']['mean_ap'] - report['test_before']['mean_ap'])
                  if 'test_after' in report else None}
    output = args.output or run / 'integrity-audit.json'
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_firmware_arguments(parser)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    audit(parser.parse_args())


if __name__ == '__main__':
    main()
