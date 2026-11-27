#!/usr/bin/env python3
"""Read-only, separately scoped candidate-recognition and camera-fidelity checks.

This command never trains. Synthetic probes cannot satisfy original-runtime
comparison: measured tensor pairs, source identity and capture evidence are needed.
Capture authenticity still requires human scrutiny of the retained run log.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from research_paths import add_firmware_arguments, resolve_source

import numpy as np
import torch
from PIL import Image

from candidate_network import CandidateNetwork
from extract import MODELS, digest
from packing_investigation import SPATIAL_LAYOUTS
from pretraining_data import checked_file, need, sha256, validate_box, validate_dataset, evaluate_detection, evaluate_focus_baseline


def validate_recognition_dataset(manifest_path, verify_files=True):
    """Check a static-image experiment without inventing focus/lighting evidence.

    Structural validity and consistent provenance do not establish authentic
    labels, dataset representativeness, or the absence of every visual duplicate.
    The optional no-file-check mode is for diagnostics; it cannot satisfy the
    experiment readiness gate.
    """
    manifest_path = Path(manifest_path)
    data = json.loads(manifest_path.read_text())
    need(data.get('schema_version') == 1 and data.get('task') == 'recognition_detection',
         'Unsupported recognition dataset schema/task')
    classes = data.get('classes')
    need(isinstance(classes, list) and classes and len(set(classes)) == len(classes)
         and set(classes) <= {'person', 'person_face'}, 'Invalid recognition classes')
    frames = data.get('frames')
    need(isinstance(frames, list) and frames, 'Empty recognition dataset')
    provenance_path = checked_file(manifest_path.parent, data['provenance']) if verify_files else None
    if provenance_path:
        provenance = json.loads(provenance_path.read_text())
        need(isinstance(provenance, dict) and provenance, 'Empty recognition provenance')
    ids, sources, groups, hashes, pixels = set(), {}, {}, {}, {}
    counts, class_counts, negative_counts = Counter(), defaultdict(Counter), Counter()
    visual_hashes = []
    for frame in frames:
        identifier, split, group = frame['id'], frame['split'], frame['group_id']
        need(isinstance(identifier, str) and identifier and identifier not in ids, 'Duplicate/invalid recognition image ID')
        need(split in ('train', 'validation', 'test'), 'Unknown recognition split')
        need(isinstance(group, str) and group, 'Missing recognition grouping ID')
        need(frame.get('source_kind') in ('public_dataset', 'camera_capture'), 'Recognition requires real-image provenance')
        for key in ('source_dataset', 'source_image_id'):
            need(isinstance(frame.get(key), str) and frame[key], 'Missing recognition source: ' + key)
        source_key = (frame['source_dataset'], frame['source_image_id'])
        need(source_key not in sources, 'Repeated recognition source image')
        need(group not in groups or groups[group] == split, 'Recognition source/group leakage across splits')
        need(frame['sha256'] not in hashes or hashes[frame['sha256']] == split, 'Identical recognition image leakage across splits')
        width, height = frame['width'], frame['height']
        need(type(width) is int and type(height) is int and width > 0 and height > 0, 'Invalid recognition image dimensions')
        if verify_files:
            path = checked_file(manifest_path.parent, frame)
            with Image.open(path) as image:
                image.load()
                need(image.size == (width, height), 'Recognition image dimensions mismatch')
                rgb = image.convert('RGB')
                pixel_hash = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                need(pixel_hash not in pixels or pixels[pixel_hash] == split, 'Decoded-pixel leakage across splits')
                pixels[pixel_hash] = split
                small = np.asarray(rgb.convert('L').resize((9, 8), Image.Resampling.BILINEAR))
                difference = small[:, 1:] > small[:, :-1]
                visual_hash = int.from_bytes(np.packbits(difference).tobytes(), 'big')
                for other_id, other_split, other_hash in visual_hashes:
                    need(other_split == split or (visual_hash ^ other_hash).bit_count() > 4,
                         'Potential near-duplicate leakage across splits: ' + other_id + ' / ' + identifier)
                visual_hashes.append((identifier, split, visual_hash))
        boxes = frame.get('boxes')
        need(isinstance(boxes, list), 'Recognition boxes must be a list')
        for box in boxes:
            need(box['class'] in classes, 'Unknown recognition annotation class')
            validate_box(box['bbox_xyxy'], width, height)
            class_counts[split][box['class']] += 1
        if not boxes:
            negative_counts[split] += 1
        ids.add(identifier); sources[source_key] = split; groups[group] = split; hashes[frame['sha256']] = split
        counts[split] += 1
    gaps = []
    for split in ('train', 'validation', 'test'):
        if not counts[split]:
            gaps.append('missing_' + split + '_images')
        for cls in classes:
            if not class_counts[split][cls]:
                gaps.append('missing_' + split + '_class_' + cls)
    return {'manifest_sha256': sha256(manifest_path), 'images': len(frames), 'groups': len(groups),
            'split_counts': dict(counts), 'class_counts': {k: dict(v) for k, v in class_counts.items()},
            'negative_image_counts': dict(negative_counts), 'data_gaps': gaps,
            'structural_validation_passed': True, 'file_content_verified': bool(verify_files),
            'duplicate_checks': ['source_image_id', 'group_id', 'file_sha256'] +
                                (['decoded_RGB_pixels', '64bit_dHash_Hamming_distance_at_most_4'] if verify_files else []),
            'provenance_limit': 'Source metadata and labels remain supplied by the dataset; hash/structure checks do not independently establish authenticity or label completeness.',
            'evaluation_limit': 'This establishes a bounded static-image recognition experiment, not camera-domain, low-light, tracking, or physical-focus validation. dHash does not detect every repeated scene.',
            'warnings': ['no_empty_background_images_in_' + split for split in ('train', 'validation', 'test')
                         if not negative_counts[split]]}, data


def compare_captures(capture_manifest, source):
    manifest = json.loads(capture_manifest.read_text())
    need(manifest.get('schema_version') == 1, 'Unsupported capture schema')
    captures = manifest.get('captures')
    need(isinstance(captures, list) and captures, 'No original capture samples')
    results = []
    unique_inputs = {stem: set() for stem in MODELS}
    ids = set()
    for capture in captures:
        identifier, stem = capture['id'], capture['model']
        need(stem in MODELS and identifier not in ids, 'Unknown model or duplicate capture')
        ids.add(identifier)
        need(capture['capture_kind'] in ('physical_s5', 'original_cdnn_runtime'), 'Synthetic/candidate run is not original evidence')
        need(capture['capture_point'] == 'convolution_pre_activation', 'Capture point must be known and pre-activation')
        for key in ('device_or_runtime_identity', 'capture_method', 'captured_at'):
            need(isinstance(capture[key], str) and capture[key], 'Missing capture metadata ' + key)
        need(capture['model_sha256'] == MODELS[stem], 'Original capture model differs')
        log_path = checked_file(capture_manifest.parent, capture['run_log'])
        need(log_path.stat().st_size > 0, 'Empty original run log')
        reference_path = checked_file(capture_manifest.parent, capture['reference_npz'])
        data = (source/(stem+'.bin')).read_bytes()
        need(digest(data) == MODELS[stem], 'Local model identity differs')
        with np.load(reference_path, allow_pickle=False) as saved:
            reference = {key: saved[key].copy() for key in saved.files}
        need('prepared_input' in reference, 'Capture missing original prepared input')
        original_input = reference['prepared_input']
        need(original_input.shape == (1,1,256,384) and np.issubdtype(original_input.dtype,np.floating), 'Prepared input must be float NCHW in recovered real units')
        need(np.isfinite(original_input).all(), 'Nonfinite original prepared input')
        # The hash of numeric content prevents duplicated inputs being counted
        # twice merely because capture filenames or NPZ compression differ.
        unique_inputs[stem].add(digest(original_input.astype('<f4').tobytes()))
        sample = {'id': identifier, 'model': stem, 'model_sha256': MODELS[stem],
                  'run_log_sha256': sha256(log_path), 'reference_npz_sha256': sha256(reference_path),
                  'capture_point': capture['capture_point'], 'layouts': {}}
        for layout in SPATIAL_LAYOUTS:
            model = CandidateNetwork(data, spatial_layout=layout).eval()
            required = {'conv_'+key for key in model.convs}
            need(set(reference) == required|{'prepared_input'}, 'Reference must include exactly all convolution pre-activation tensors')
            observed = {}
            hooks = []
            for key, module in model.convs.items():
                def hook(_module, _inputs, output, name='conv_'+key):
                    observed[name] = output.detach().cpu().numpy().copy()
                hooks.append(module.register_forward_hook(hook))
            try:
                with torch.no_grad():
                    model(torch.from_numpy(original_input.astype(np.float32)))
            finally:
                for hook in hooks: hook.remove()
            layer_results = []
            for name in sorted(required):
                actual, expected = observed[name], reference[name]
                need(actual.shape == expected.shape and np.isfinite(expected).all(), 'Reference tensor shape/data invalid: '+name)
                # Explicit tolerances are recorded. They are a float-port audit
                # threshold, not a statement of bit-exact fixed-point parity.
                absolute = np.abs(actual.astype(np.float64)-expected.astype(np.float64))
                atol, rtol = 2e-5, 2e-4
                passed = bool(np.all(absolute <= atol + rtol*np.abs(expected)))
                layer_results.append({'tensor':name,'shape':list(actual.shape),
                                      'maximum_absolute_error':float(absolute.max()),
                                      'rmse':float(np.sqrt(np.mean(absolute**2))), 'passed':passed})
            sample['layouts'][layout] = {'all_layers_pass':all(r['passed'] for r in layer_results),
                                         'atol':atol,'rtol':rtol,'layers':layer_results}
        results.append(sample)
    qualified = {}
    for stem in MODELS:
        subset = [r for r in results if r['model']==stem]
        qualified[stem] = [layout for layout in SPATIAL_LAYOUTS
                           if subset and len(unique_inputs[stem])>=2 and all(r['layouts'][layout]['all_layers_pass'] for r in subset)]
    return {'capture_manifest_sha256':sha256(capture_manifest),'comparisons':results,
            'distinct_input_count':{k:len(v) for k,v in unique_inputs.items()},
            'layouts_matching_all_captures':qualified,
            'unique_layout_for_each_model':all(len(v)==1 for v in qualified.values()),
            'scope':'Measured float convolution pre-activation parity. Does not validate camera preprocessing, proposal ABI or all fixed-point behavior.',
            'provenance_limit':'Capture type and device identity are operator-provided; inspect hash-bound logs to establish authenticity.'}


def readiness(source, dataset_path=None, predictions_path=None, captures_path=None,
              recognition_dataset_path=None, recognition_model='44_hm_d_nw_sng'):
    base = Path(__file__).resolve().parent
    tool_validation = base/'artifacts/pretraining-tools-validation.json'
    tools_passed = False
    if tool_validation.is_file():
        evidence = json.loads(tool_validation.read_text())
        tools_passed = (evidence.get('tests_run',0)>=7 and evidence.get('failures')==0 and evidence.get('errors')==0
                        and set(evidence.get('source_hashes',{})) == {'pretraining_data.py','pretraining_readiness.py','verify_pretraining_tools.py'}
                        and all(sha256(base/name)==value for name,value in evidence['source_hashes'].items()))
    source_identities = {}
    for stem, expected in MODELS.items():
        path = source/(stem+'.bin') if source is not None else None
        source_identities[stem] = path is not None and path.is_file() and sha256(path)==expected
    gates = {'source_identities':all(source_identities.values()), 'dataset_structure_and_coverage':False,
             'numeric_original_execution_comparison':False, 'held_out_detection_baseline':False,
             'image_to_input_preprocessing_fidelity':False, 'proposal_decoder_fidelity':False,
             'physical_focus_control_baseline':False}
    need(recognition_model in MODELS, 'Unknown recognition candidate model')
    recognition_gates = {'selected_source_identity': source_identities[recognition_model],
                         'real_image_dataset_structure_and_held_out_coverage': False}
    recognition_scope = {
        'scope': 'Computer candidate-model recognition experiment; inputs are sufficient to begin a bounded experiment, not proof of model performance or camera parity.',
        'selected_model': recognition_model, 'gates': recognition_gates, 'ready': False,
        'not_required_for_this_scope': ['original_DSP_outputs', 'CEVA_CDNN_SDK', 'physical_camera',
                                        'lens_control_baseline', 'low_light_or_tracking_labels'],
        'remaining_experiment_obligations': ['Declare layout/preprocessing/decoder hypotheses.',
                                             'Measure pre-training and held-out post-training performance.',
                                             'Retain train/validation/test separation and data provenance.',
                                             'Verify actual finite gradients and parameter updates.']}
    report = {'source_identities':source_identities, 'gates':gates,
              'source_identity_check':'not_checked_tools_only' if source is None else 'checked_available_inputs',
              'ready_for_formal_training':False,
              'data_and_evaluation_tools_prepared':tools_passed,
              'computer_preparation_scope':'Offline data validation, metrics and original-capture comparison interfaces are implemented; use verify_pretraining_tools.py to validate their fixtures.',
              'scope':'Separate candidate recognition input readiness from original-camera fidelity and physical-focus readiness.',
              'training_started':False,
              'training_started_scope':'This read-only command does not train or infer whether a separate training run has occurred.'}
    if recognition_dataset_path:
        recognition_validation, _ = validate_recognition_dataset(recognition_dataset_path)
        recognition_scope['dataset'] = recognition_validation
        recognition_gates['real_image_dataset_structure_and_held_out_coverage'] = (
            recognition_validation['file_content_verified'] and not recognition_validation['data_gaps'])
    if dataset_path:
        validation, dataset = validate_dataset(dataset_path)
        report['dataset'] = validation
        report['captured_focus_baseline'] = evaluate_focus_baseline(dataset)
        gates['dataset_structure_and_coverage'] = not validation['data_gaps']
        if predictions_path:
            predictions = json.loads(predictions_path.read_text())
            need(predictions['dataset_manifest_sha256']==validation['manifest_sha256'],'Predictions bind another dataset')
            need(predictions.get('model_sha256') in MODELS.values(),'Predictions bind unknown original model')
            report['baseline'] = evaluate_detection(dataset,predictions)
            report['baseline']['predictions_sha256'] = sha256(predictions_path)
            # Evaluating a file proves metric availability, not the origin of
            # its boxes. An original execution/decoder comparison remains needed.
            gates['held_out_detection_baseline'] = report['baseline']['mean_ap'] is not None
    if captures_path:
        need(all(source_identities.values()), 'Original capture comparison requires both pinned source models')
        report['original_comparison'] = compare_captures(captures_path,source)
        gates['numeric_original_execution_comparison'] = report['original_comparison']['unique_layout_for_each_model']
    report['outstanding'] = [name for name,passed in gates.items() if not passed]
    report['not_automatically_verifiable'] = {
        'image_to_input_preprocessing_fidelity':'Need paired original sensor/image frame and original model input, then recompute the exact transform.',
        'proposal_decoder_fidelity':'Need original scores/boxes on held-out frames and candidate decode comparisons, including edge cases.',
        'physical_focus_control_baseline':'Need measured lens commands, positions, image focus ground truth and feedback; recognition labels alone are insufficient.'}
    # These currently unresolved gates intentionally have no --assume-ready or
    # user-supplied boolean escape hatch. Implement evidence validators first.
    report['ready_for_formal_training'] = all(gates.values())
    report['ready_for_formal_training_legacy_scope'] = 'Historical combined original-camera recognition/tracking and physical-focus gates; not a veto on computer candidate-recognition experiments.'
    recognition_scope['ready'] = all(recognition_gates.values())
    recognition_scope['outstanding'] = [name for name, passed in recognition_gates.items() if not passed]
    report['ready_for_computer_recognition_experiment'] = recognition_scope['ready']
    report['ready_for_original_camera_fidelity_and_deployment'] = False
    report['scopes'] = {
        'computer_recognition_experiment': recognition_scope,
        'original_camera_fidelity_and_deployment': {
            'scope': 'Original-runtime numerical fidelity, preprocessing/decoding, and measured physical-focus behavior; actual deployment additionally needs a validated packaging/loading path.',
            'gates': dict(gates, validated_deployment_packaging_and_loading=False),
            'ready': False,
            'outstanding': report['outstanding'] + ['validated_deployment_packaging_and_loading']}}
    return report


def main():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    add_firmware_arguments(parser)
    parser.add_argument('--tools-only',action='store_true',help='Report tool checks without reading firmware inputs')
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--predictions',type=Path)
    parser.add_argument('--captures',type=Path)
    parser.add_argument('--recognition-dataset',type=Path,
                        help='Static-image candidate recognition manifest; separate from physical-focus dataset requirements')
    parser.add_argument('--recognition-model',choices=tuple(MODELS),default='44_hm_d_nw_sng')
    parser.add_argument('--output',type=Path,default=base/'artifacts/pretraining-readiness.json')
    args = parser.parse_args()
    args.source = resolve_source(args)
    need(args.dataset is not None or args.predictions is None,'Predictions require a dataset')
    torch.set_num_threads(2)
    need(not (args.tools_only and args.captures), '--captures cannot be combined with --tools-only')
    report = readiness(None if args.tools_only else args.source,args.dataset,args.predictions,args.captures,
                       args.recognition_dataset,args.recognition_model)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'ready_for_computer_recognition_experiment':report['ready_for_computer_recognition_experiment'],
                      'ready_for_original_camera_fidelity_and_deployment':report['ready_for_original_camera_fidelity_and_deployment'],
                      'ready_for_formal_training':report['ready_for_formal_training'],
                      'recognition_outstanding':report['scopes']['computer_recognition_experiment']['outstanding'],
                      'original_camera_outstanding':report['scopes']['original_camera_fidelity_and_deployment']['outstanding']},ensure_ascii=False))


if __name__=='__main__':
    main()
