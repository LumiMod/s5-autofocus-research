#!/usr/bin/env python3
"""Post-hoc image-independent control for an already frozen recognition run.

Constant prepared zeros represent flat gray pixels at the mean (117). This is
not a new trained model and must not be used to select the run's checkpoint.
"""
import argparse
import json
from pathlib import Path
import torch

from extract import write_json, require
from pretraining_data import evaluate_detection, sha256
from pretraining_readiness import validate_recognition_dataset
from research_paths import add_firmware_arguments, resolve_source
from train_recognition import Detector, MODEL, detections, source_hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_firmware_arguments(parser)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    validation, dataset = validate_recognition_dataset(args.dataset)
    report = json.loads((args.run/'report.json').read_text())
    config = report['config']
    require(validation['manifest_sha256'] == config['manifest_sha256'], 'Dataset differs from frozen run')
    require(source_hashes() == config['source_hashes'], 'Experiment source changed')
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    data = (resolve_source(args)/(MODEL+'.bin')).read_bytes()
    output = args.run/'constant-input-control'
    output.mkdir(exist_ok=True)
    controls = {}
    for state in ('restored', 'last_epoch'):
        detector = Detector(data, config['selected_layout_hypothesis'])
        if state == 'last_epoch':
            require(sha256(args.run/'last.pt') == report['checkpoints']['last.pt']['sha256'], 'Checkpoint differs')
            checkpoint = torch.load(args.run/'last.pt', map_location='cpu', weights_only=False)
            detector.model.load_state_dict(checkpoint['model_state'])
        detector.model.eval()
        with torch.inference_mode():
            logits, delta = detector(torch.zeros(1, 1, 256, 384))
            for split in ('validation', 'test'):
                predictions = {'dataset_manifest_sha256': validation['manifest_sha256'],
                               'frames': [{'id': f['id'], 'detections': detections(logits, delta, detector.anchors, f)}
                                          for f in dataset['frames'] if f['split'] == split]}
                metrics = evaluate_detection(dataset, predictions, split)
                controls[state+'_'+split] = metrics
                write_json(output/(state+'_'+split+'-predictions.json'), predictions)
        detector.close()
    result = {'status': 'post-hoc diagnostic after all model selection; no changes to selected model',
              'manifest_sha256': config['manifest_sha256'], 'model_sha256': config['model_sha256'],
              'layout_hypothesis': config['selected_layout_hypothesis'],
              'input': 'Same zero prepared tensor for every image; corresponds to flat gray117 before centering. No photo pixels read by the detector.',
              'purpose': 'Measure how much AP can arise from fixed spatial/size predictions on this positive-only subset.',
              'control_script_sha256': sha256(Path(__file__)), 'experiment_source_hashes': source_hashes(),
              'metrics': controls,
              'limits': 'A constant-input control is distribution-specific, not proof of feature correctness, numerical equivalence, or an estimate of all image-independent baselines.'}
    write_json(output/'report.json', result)
    print(json.dumps({k: v['mean_ap'] for k, v in controls.items()}))


if __name__ == '__main__':
    main()
