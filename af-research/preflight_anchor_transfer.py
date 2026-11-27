#!/usr/bin/env python3
"""Bind the analytic initialization correction before a second finite run."""
import json
from pathlib import Path
from extract import require,write_json
from pretraining_data import sha256

BASE=Path(__file__).resolve().parent


def main():
    oldpath=BASE/'artifacts/recognition-diagnosis-v3/small-anchor-preflight/ready.json'
    root=BASE/'artifacts/recognition-diagnosis-v3/anchor-head-transfer'
    old=json.loads(oldpath.read_text())
    synthetic=json.loads((root/'synthetic-review.json').read_text())
    diagnostic=json.loads((root/'train32/report.json').read_text())
    review=json.loads((root/'runner-review.json').read_text())
    expected=synthetic['provenance']
    original=diagnostic['results']['original_anchors']['independent_metrics']
    transferred=diagnostic['results']['nearest_log_affine_transfer']['independent_metrics']
    gates={
        'original_target_and_geometry_preflight_passed':old['ready_for_bounded_training'],
        'analytic_mapping_tests_passed':synthetic['passed'],
        'graph_backbone_and_shapes_unchanged':expected['backbone_bitwise_unchanged'] and expected['architecture_and_shapes_unchanged'],
        'diagnostic_train_only_no_fitting':diagnostic['optimizer_steps']==0 and not diagnostic['validation_inference'] and not diagnostic['test_inference'] and not diagnostic['strategy_changed_after_results'],
        'same_scales_and_anchor_bytes':expected['new_scales']==old['scales'] and expected['new_anchor_sha256']==old['small_anchor_sha256'],
        'runner_review_passed':review['passed'],
        'runner_source_bound':review['runner_sha256']==sha256(BASE/'train_small_anchor_transfer.py')}
    for cls in ('person','person_face'):
        value=transferred['real'][cls]['ap_at_iou']
        gates[cls+'_avoids_initialization_collapse']=value>=.8*original['real'][cls]['ap_at_iou']
        gates[cls+'_real_signal_retained']=all(value>=transferred[m][cls]['ap_at_iou']+.15
                                              for m in ('same_mean','fixed_gray117','shuffled'))
    evidence=list(old['evidence'])
    paths=[oldpath,root/'synthetic-review.json',root/'train32/report.json',root/'runner-review.json',
           BASE/'train_small_anchor_transfer.py',BASE/'anchor_head_transfer.py',Path(__file__)]
    for p in paths:evidence.append({'path':str(p.relative_to(BASE)),'sha256':sha256(p)})
    for item in evidence:require(sha256(BASE/item['path'])==item['sha256'],'Reviewed evidence changed')
    for record in (synthetic,diagnostic):
        for name,digest in record['source_hashes'].items():require(sha256(BASE/name)==digest,'Diagnostic source differs')
    result={'ready_for_bounded_training':all(gates.values()),'gates':gates,
            'scales':old['scales'],'manifest_sha256':old['manifest_sha256'],
            'small_anchor_sha256':old['small_anchor_sha256'],
            'transferred_state_sha256':expected['transferred_state_sha256'],'evidence':evidence,
            'scope':'Second finite experiment corrects anchor-slot initialization; no expansion of epochs, validation tuning or test use.',
            'known_limits':['Initial affine transfer exceeds original i16 bbox weight range for19 elements; FP32 is not directly deployable. Final host quantized proxy is separately evaluated.',
                            'Train32 control retention validates a noncollapsed initialization, not small-face success.',
                            'Primary result remains fixed epoch6; same acceptance thresholds as the failed first run.'],
            'small_face_recognition_verified':False,'camera_deployment_verified':False}
    output=root/'ready.json';require(not output.exists(),'Preserve existing preflight')
    write_json(output,result)
    print(json.dumps({'passed':result['ready_for_bounded_training'],'output':str(output),'gates':gates}))
    require(result['ready_for_bounded_training'],'Transfer preflight failed')


if __name__=='__main__':main()
