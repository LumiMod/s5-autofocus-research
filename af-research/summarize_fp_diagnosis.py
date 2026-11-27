#!/usr/bin/env python3
"""Publish numerical FP diagnosis evidence, never photos, weights, or training."""
import argparse
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE/'artifacts/recognition-fp-diagnosis'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def run(args):
    files = {
        'saved_fp': ROOT/'saved-predictions/report.json',
        'saved_fp_independent': ROOT/'saved-predictions/independent-verification.json',
        'sampling': ROOT/'training-sampling/report.json',
        'targets': ROOT/'target-conflicts/report.json',
        'head_interventions': ROOT/'head-interactions/report.json',
        'loss_verification': ROOT/'balanced-loss-verification/report.json',
        'independent_head_loss': ROOT/'target-conflicts/independent-head-and-loss-review-extended.json',
        'repair_coverage': ROOT/'training-sampling/independent-balanced-loss-coverage.json',
        'nms_preview_selection': ROOT/'nms-evidence/selection.json',
        'box_convention_caveat': ROOT/'target-conflicts/box-convention-caveat.json',
    }
    reports = {k: json.loads(p.read_text()) for k, p in files.items()}
    fp = reports['saved_fp']
    sampling = reports['sampling']
    heads = reports['head_interventions']
    verification = reports['loss_verification']
    targets = reports['targets']
    need(fp['optimizer_steps'] == heads['optimizer_steps'] == verification['optimizer_steps'] == 0,
         'Unexpected optimizer update')
    need(not verification['ready_to_train'] and not heads['ready_to_train'], 'Readiness boundary changed')
    need(verification['passed_supervision_checks'] and not verification['passed_detection_acceptance'],
         'Supervision and accuracy gates conflated')
    need(all(verification['fixtures'].values()), 'Loss fixture failed')
    need(targets['source_annotations']['all_selected_source_target_annotations_preserved'], 'Source GT dropped')
    source_checks = {}
    for scope in ('head_interventions', 'loss_verification', 'sampling'):
        report = reports[scope]
        for name, expected in report['source_hashes'].items():
            need(sha(BASE/name) == expected, 'Frozen source changed: '+name)
            source_checks[name] = expected
    compact_fp = {stage: {cls: dict(ap50=row['ap50'], **row['score_ge_05'])
                          for cls, row in classes.items()}
                  for stage, classes in fp['summaries'].items()}
    compact_heads = {name: {
        'state_sha256': row['state_sha256'],
        'person': {k: row['metrics']['classes']['person'][k] for k in ('ap_at_iou', 'tp_at_score', 'fp_at_score')},
        'face': {k: row['metrics']['classes']['person_face'][k] for k in ('ap_at_iou', 'tp_at_score', 'fp_at_score')},
        'small_faces': row['metrics']['small_faces']}
        for name, row in heads['results'].items()}
    need(compact_fp['epoch-06']['person_face']['fp'] == 215 and
         compact_fp['baseline_original']['person_face']['fp'] == 48, 'FP baseline changed')
    files.update({name: BASE/name for name in (
        'recognition_balanced_loss.py', 'verify_balanced_loss.py',
        'diagnose_head_interactions.py', 'diagnose_saved_false_positives.py',
        'diagnose_training_sampling.py', 'diagnose_target_conflicts.py',
        Path(__file__).name)})
    summary = dict(schema_version=1, date='2026-11-27',
        status='diagnosed; limited loss repair verified; detection acceptance unresolved; training paused',
        dataset_manifest_sha256=targets['dataset_sha256'],
        optimizer_steps_this_work=0, new_validation_inference=False, test_inference=False,
        graph_changed=False, existing_checkpoint_files_changed=False,
        new_loss_installed_in_training_runner=False,
        score_or_nms_thresholds_changed=False,
        saved_validation_metrics=compact_fp, frozen_train32_head_interventions=compact_heads,
        frozen_train32_final_highscore=sampling['final_highscore_summary'],
        historical_train32_sampling=sampling['historical_sampling_summary'],
        historical_train32_per_gt_positive_counts=sampling['per_gt_positive_sampling_summary'],
        frozen_train32_small_face_failures=sampling['small_face_refined_stage_counts'],
        loss_candidate=dict(module='recognition_balanced_loss.py',
            shared_static_gt_owners_unchanged=True, per_gt_positive_and_bbox_mass_equal=True,
            class_positive_budget=32, class_total_budget=128, bbox_budget=128,
            hard_negative_max=64, remaining_negatives='random without replacement',
            ignore_static_negatives_if_decoded_same_class_iou_at_least=.3,
            fixtures=verification['fixtures'], frozen_output_gradient_images=verification['train_images'],
            passed_supervision_checks=True, passed_detection_acceptance=False,
            actual_frozen_train32_face_fp_supervision=dict(total_fp=83, guard_eligible=49,
                selected_hard=37, selected_random=1, eligible_unselected=11,
                dynamically_guarded=24, statically_ignored=10,
                correctly_decoded_static_negative_small_face_candidates_protected=12)),
        ready_to_train=False,
        unresolved=[
            'The guard excludes IoU .3-.5 localization false positives as well as correct boxes.',
            'Three of four highest-scoring NMS suppressors of small faces are guarded; they are not directly penalized.',
            'Five confidence-limited and one geometry-limited train32 small faces have no post-repair improvement evidence.',
            'No background-only data or S5 camera-domain detector validation.',
            'All four NMS examples are occluded; retained boxes resemble larger head regions. Original face/head box convention is unknown.',
            'Host-restored graph arithmetic and deployment are not validated against the S5 DSP.'],
        interpretation=[
            'Saved validation FP taxonomy describes geometry against existing annotations, not confirmed visual background.',
            'Historical selections are exact; their high-score status is scored retrospectively with epoch06 outputs.',
            'Head interventions and supervision checks use training images; they are not independent accuracy.',
            'No new trained detector or firmware was produced.'],
        verified_current_sources=source_checks,
        artifacts={k: dict(path=p.relative_to(BASE).as_posix(), sha256=sha(p), bytes=p.stat().st_size)
                   for k, p in files.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(dict(path=str(args.output), sha256=sha(args.output), ready_to_train=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=BASE/'evidence/2026-11-27/recognition-fp-diagnosis/summary.json')
    run(parser.parse_args())
