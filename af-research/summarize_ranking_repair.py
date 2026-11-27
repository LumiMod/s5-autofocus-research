#!/usr/bin/env python3
"""Bind ranking, annotation-contract and native-ARM replay evidence; no training."""
import argparse
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE/'artifacts/recognition-ranking-repair'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    relative = {
        'annotation_contract': 'box-contract/report.json',
        'head_reference': 'box-contract/head-reference.json',
        'paired_face_head': 'paired-face-head-metrics/report.json',
        'paired_independent': 'paired-face-head-metrics/independent-verification.json',
        'ranking': 'quality-ranking-normalized/report.json',
        'ranking_independent': 'quality-ranking-normalized/independent-verification.json',
        'firmware_contract': 'firmware-contract/report.json',
        'firmware_followup': 'firmware-contract/followup-consumers.json',
        'firmware_callback': 'firmware-contract/callback-route.json',
        'record_parser_fixtures': 'firmware-contract/parser-fixtures.json',
        'native_arm_replay': 'native-converter-qemu-v2/report.json',
        'native_arm_negative_control': 'native-converter-qemu-negative-control/report.json',
        'native_arm_independent': 'firmware-contract/qemu-harness-independent-review.json',
        'preserved_launch_failure': 'native-converter-qemu/report.json',
        'preserved_unicorn_failure': 'firmware-contract/arm-replay-unavailable.json',
    }
    files = {k: ROOT/n for k, n in relative.items()}
    reports = {k: json.loads(p.read_text()) for k, p in files.items()}
    rank = reports['ranking']
    audit = reports['ranking_independent']
    contract = reports['annotation_contract']
    assert rank['passed_supervision_checks'] and rank['toy_tests']['passed']
    assert audit['passed_supervision_math_checks']
    assert rank['checkpoint_sha256_before'] == rank['checkpoint_sha256_after']
    assert rank['checkpoint_state_sha256_before'] == rank['checkpoint_state_sha256_after']
    assert rank['optimizer_steps'] == audit['optimizer_steps'] == 0
    assert not rank['ready_to_train'] and not rank['passed_detection_acceptance']
    assert not contract['raw_DSP_semantic_contract_confirmed']
    assert reports['native_arm_replay']['passed']
    assert reports['native_arm_negative_control']['passed_negative_control']
    assert not reports['preserved_launch_failure']['actual_original_ARM_instruction_execution']
    assert sha(BASE/contract['document']) == contract['document_sha256']
    for name, digest in rank['source_hashes'].items():
        assert sha(BASE/name) == digest, 'Executed source changed: '+name
    assert (BASE/'s5_result_records.py').read_bytes() == (ROOT/'firmware-contract/arm_output_abi.py').read_bytes()
    compact_metrics = {}
    for name, row in reports['paired_face_head']['results'].items():
        compact_metrics[name] = dict(frames=row['frames'], metrics={
            label: {k: metrics[k] for k in ('ground_truth_boxes', 'ap_at_iou', 'tp_at_score', 'fp_at_score', 'size_strata')}
            for label, metrics in row['metrics'].items()})
    for name in ('recognition_quality_ranking.py', 'verify_quality_ranking.py',
                 'compare_face_head_references.py', 'verify_face_head_comparison.py',
                 's5_result_records.py', 'verify_s5_arm_result_converter.py', Path(__file__).name):
        files[name] = BASE/name
    result = dict(schema_version=1, date='2026-11-27',
        status='Ranking supervision checked; original face/head contract unresolved; detection acceptance not passed; training remains paused.',
        optimizer_steps_this_work=0, validation_model_inference=False, test_access=False,
        network_graph_or_existing_weights_changed=False, new_loss_installed_in_training_runner=False,
        dataset_manifest_sha256=contract['manifest_sha256'],
        paired_annotation_metrics=compact_metrics,
        ranking=dict(toy_tests=rank['toy_tests']['count'], frozen_train_images=rank['train_frames'],
            candidate_pairs=rank['total_pairs'], passed_supervision_checks=True,
            independent_gradient_max_error=audit['saved_pair_checks']['max_saved_scalar_gradient_absolute_error'],
            face_head_geometry_counts=audit['saved_pair_checks']['counts'],
            normalization='Each pair uses its GT positive CE weight mass divided by actual class sample count; then detached IoU gap.',
            direct_bbox_gradient_unchanged=True,
            future_bbox_update_unchanged_is_not_proven='Shared gradient clipping may change bbox update scaling if this loss is later integrated.'),
        original_arm_boundary=dict(executed=True, entry_thumb='0x063bd2c3',
            environment='QEMU9.2.4 virt/Cortex-A15, synthetic records only', records=60,
            checked_output_bytes=960, checked_canary_bytes=16,
            negative_control_detected=True, independent_harness_review_passed=True,
            original_DSP_or_network_executed=False, camera_executed=False,
            verified='u16 xywh to xyxy; numeric class0..6 retained; low16 score /65535; gap untouched',
            not_verified='Original producer, proposal-to-class mapping, face/head training semantics, full camera pipeline'),
        semantic_contract=dict(open_images_face_and_head_are_distinct=True,
            open_images_general_box='Minimum visible extent; not amodal completion',
            original_S5_factory_output_semantics_confirmed=False,
            label_name_alone_is_not_a_training_contract=True,
            historical_GT_or_metrics_relabelled=False),
        ready_to_train=False, passed_detection_acceptance=False,
        unresolved=[
            'Seven audited face pairs favor the bad box under head geometry; three cross the head TP threshold in the opposite direction.',
            'No repaired-weight detector was trained or independently evaluated; the last small-face/FP failures remain.',
            'The DSP output producer and class/box semantics are not established by the ARM converter.',
            'No original-camera same-frame tensor/result comparison or SDK execution is available.'],
        artifacts={k: dict(path=p.relative_to(BASE).as_posix(), sha256=sha(p), bytes=p.stat().st_size)
                   for k, p in files.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(dict(path=str(args.output), sha256=sha(args.output), ready_to_train=False)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=BASE/'evidence/2026-11-27/recognition-ranking-repair/summary.json')
    run(p.parse_args())
