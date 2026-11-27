#!/usr/bin/env python3
"""Summarize both frozen small-anchor trials without distributing private data."""
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(path.read_text())


def compact(metrics):
    return {'person':metrics['classes']['person'],'person_face':metrics['classes']['person_face'],
            'small_faces':metrics['small_faces'],'mean_ap':metrics['mean_ap']}


def main():
    design=read(BASE/'artifacts/recognition-diagnosis-v3/small-anchor-design/train320/report.json')
    targets=read(BASE/'artifacts/recognition-diagnosis-v3/small-anchor-targets/scales-1-2-4-8-16.json')
    transfer=read(BASE/'artifacts/recognition-diagnosis-v3/anchor-head-transfer/synthetic-review.json')
    runs={};identities=[];source_bindings=[]
    for name in ('recognition-small-anchor-v1','recognition-small-anchor-v2'):
        folder=BASE/'artifacts'/name
        plan=read(folder/'plan.json');summary=read(folder/'summary.json');verification=read(folder/'independent-verification.json')
        assert verification['passed_integrity_audit'] and not verification['test_inference']
        assert summary['status']=='complete' and summary['optimizer_steps']==480 and not summary['test_inference']
        assert summary['backbone_unchanged']
        assert summary['plan_sha256']==sha(folder/'plan.json')
        for source,digest in plan['source_hashes'].items():
            path=folder/'source-snapshot'/source
            assert sha(path)==digest
            source_bindings.append({'run':name,'source':source,'sha256':digest,'snapshot':str(path.relative_to(BASE))})
        checkpoints=[]
        for path in [folder/'baseline_small/checkpoint.pt',*sorted(folder.glob('epoch-*/checkpoint.pt')),folder/'final_quantized/checkpoint.pt']:
            checkpoints.append({'path':str(path.relative_to(BASE)),'sha256':sha(path),'bytes':path.stat().st_size})
        runs[name]={
            'initialization':plan.get('initialization','original_weights_direct_anchor_change'),
            'plan_sha256':sha(folder/'plan.json'),'scales':plan['scales'],'layout':plan['layout'],
            'initial_state_sha256':plan['initial_state_sha256'],
            'original_state_sha256':plan.get('original_state_sha256',plan['initial_state_sha256']),
            'trainable_parameters':plan['trainable_parameters'],'epochs':6,'optimizer_steps':480,
            'backbone_unchanged':summary['backbone_unchanged'],
            'original_anchor_baseline':compact(summary['baseline_original']),
            'new_anchor_epoch0':compact(summary['baseline_small']),
            'primary_fixed_epoch6':{'metrics':compact(summary['primary_fixed_epoch6']['metrics']),
                                   'acceptance':summary['primary_fixed_epoch6']['acceptance']},
            'final_quantized_proxy':{'metrics':compact(summary['final_quantized']['metrics']),
                                     'acceptance':summary['final_quantized']['acceptance'],
                                     'quantization':summary['final_quantized']['quantization']},
            'auxiliary_best_epoch':summary['auxiliary_best_epoch'],
            'epoch_metrics':[{'epoch':row['epoch'],'global_step':row['global_step'],
                              'mean_training_loss':row['mean_training_loss'],
                              'metrics':compact(row['metrics'])} for row in summary['epochs']],
            'independent_verification_sha256':sha(folder/'independent-verification.json'),
            'checkpoint_identities':checkpoints,'test_inference':False,'camera_deployment_verified':False,
            'reporting_recovery':name.endswith('v1')}
        for path in sorted(folder.rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                identities.append({'path':str(path.relative_to(BASE)),'sha256':sha(path),'bytes':path.stat().st_size})
    extra=[
        'artifacts/recognition-diagnosis-v3/small-anchor-design/train320/report.json',
        'artifacts/recognition-diagnosis-v3/small-anchor-design/train320/independent-crosscheck.json',
        'artifacts/recognition-diagnosis-v3/small-anchor-targets/scales-1-2-4-8-16.json',
        'artifacts/recognition-diagnosis-v3/small-anchor-preflight/ready.json',
        'artifacts/recognition-diagnosis-v3/anchor-head-transfer/ready.json',
        'artifacts/recognition-diagnosis-v3/anchor-head-transfer/synthetic-review.json',
        'artifacts/recognition-diagnosis-v3/anchor-head-transfer/train32/report.json',
        'artifacts/recognition-diagnosis-v3/anchor-head-transfer/runner-review.json',
        'artifacts/recognition-small-anchor-preflight/data-audit.json',
        'artifacts/recognition-diagnosis-v3/small-face-after-v2-train32/report.json',
        'artifacts/recognition-small-anchor-v2-preview/selection.json',
        'artifacts/recognition-small-anchor-v2-preview/small-face-match-changes.json',
        'artifacts/recognition-small-anchor-v2-preview-label-fix/selection.json',
        'artifacts/recognition-small-anchor-v2-preview-label-fix/label-correction-verification.json']
    for name in extra:
        path=BASE/name;identities.append({'path':name,'sha256':sha(path),'bytes':path.stat().st_size})
    stage=read(BASE/'artifacts/recognition-diagnosis-v3/small-face-after-v2-train32/report.json')
    assert stage['model_frozen_and_unchanged'] and stage['optimizer_steps_this_probe']==0 and not stage['test_inference']
    summary={
        'schema_version':1,'date':'2026-11-27','scope':'Original S5 graph with explicit K4 and five smaller anchor scales; bounded host training.',
        'manifest_sha256':targets['manifest_sha256'],'model_sha256':targets['source_model_sha256'],
        'architecture_changes':False,'feature_grid':[16,24],'stride':[16,16],
        'anchors_per_location':25,'head_channels':[50,50,100],
        'scales':targets['anchor_config']['scales'],'original_scales':targets['anchor_config']['original_scales'],
        'design_train_only':{n:g['summary'] for n,g in design['geometry'].items()},
        'target_consistency':{k:targets[k] for k in ('target_consistency_passed','geometric_all_gt_iou_ge_05','images','gt_count','groups','actual_loss_gradient_probes')},
        'semantic_tests_passed':targets['semantic_tests']['passed'],
        'transfer':{'old_scale_index_by_new_scale':transfer['provenance']['old_scale_index_by_new_scale'],
                    'decoded_box_checks':transfer['decoded_box_checks'],
                    'limit':'Copies selected old slots and analytically adjusts regression; does not retain the entire original proposal set. Initial fixed-scale i16 proxy clips19 bbox weights.'},
        'runs':runs,'total_optimizer_steps':960,'held_out_test_inference':False,
        'current_comparison_preview':'artifacts/recognition-small-anchor-v2-preview-label-fix/comparison.png',
        'post_training_train32_stage_diagnostic':{
            'original':stage['original_baseline_summary'],'trained':stage['baseline_summary'],
            'limit':'Same fixed training32, stage attribution only; not independent accuracy. NMS and final30 removal are grouped.'},
        'original_dsp_numerical_parity':False,'firmware_exported':False,'camera_deployment_verified':False,
        'limits':['Candidate geometry coverage and consistent GT ownership do not establish detection success.',
                  'Validation80 is reused for diagnostics and finite experiments; it is not an independent test result.',
                  'Dataset has no negative-only backgrounds or camera-domain focus/tracking measurements.',
                  'Quantized proxy rounds/clips only heads with original scales; it does not reproduce full DSP arithmetic.',
                  'Fixed epoch6 remains the primary outcome even if an earlier epoch scores higher.'],
        'source_bindings':source_bindings,'artifact_identities':identities,'summarizer_sha256':sha(Path(__file__))}
    output=BASE/'evidence/2026-11-27/small-anchor-training/summary.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(output),'sha256':sha(output),'runs':2,'optimizer_steps':960,'artifacts':len(identities)}))


if __name__=='__main__':main()
