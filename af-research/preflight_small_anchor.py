#!/usr/bin/env python3
"""Bind completed design/target/core reviews before a bounded training run."""
import argparse
import json
from pathlib import Path

from extract import MODELS, require, write_json
from pretraining_data import sha256
from train_recognition import MODEL, source_hashes

BASE=Path(__file__).resolve().parent


def run(output):
    design_path=BASE/'artifacts/recognition-diagnosis-v3/small-anchor-design/train320/report.json'
    target_path=BASE/'artifacts/recognition-diagnosis-v3/small-anchor-targets/scales-1-2-4-8-16.json'
    review_path=BASE/'artifacts/recognition-diagnosis-v3/small-anchor-preflight/runner-review.json'
    design=json.loads(design_path.read_text());target=json.loads(target_path.read_text());review=json.loads(review_path.read_text())
    scales=[1,2,4,8,16]
    original=design['geometry']['original']['summary']
    proposed=design['geometry']['small16_keep256']['summary']
    patch=design['patch_roundtrips']['small16_keep256']
    gates={
        'design_complete':design['status']=='complete' and design['sources_unchanged'],
        'scales_chosen_using_train_only':design['recommendation']['scales']==scales and not design['validation_selection'] and not design['test_inference'],
        'small_geometry_improved':proposed['person_face/small_lt32']['covered_ge_05']>original['person_face/small_lt32']['covered_ge_05'],
        'large_person_and_face_geometry_preserved':all(proposed[c+'/large_ge96']['covered_ge_05']>=original[c+'/large_ge96']['covered_ge_05'] for c in ('person','person_face')),
        'original_graph_and_channels_preserved':patch['graph_headers_and_operator_shapes_identical'] and patch['recovered_input_weights_biases_identical'],
        'proposal_scale_patch_complete_inverse':patch['complete_byte_inverse_identical'] and patch['ratios_unchanged'] and patch['both_scale_arrays_reparsed']==scales,
        'all_training_targets_consistent':target['target_consistency_passed'] and not target['failures'] and target['images']==320 and target['gt_count']==883,
        'semantic_tests_pass':target['semantic_tests']['passed'],
        'same_source_and_manifest':design['model_sha256']==target['source_model_sha256']==MODELS[MODEL] and design['manifest_sha256']==target['manifest_sha256'],
        'target_geometry_matches_design':target['anchor_config']['scales']==scales,
        'runner_review_passed':review['passed'],
        'runner_review_matches_current_source':review['runner_sha256']==sha256(BASE/'train_small_anchor.py')}
    sources={**source_hashes(),**design['sources'],**target['source_hashes']}
    sources.update({n:sha256(BASE/n) for n in ('train_small_anchor.py','preflight_small_anchor.py','verify_external_baselines.py')})
    for name,digest in sources.items():require(sha256(BASE/name)==digest,'Reviewed source differs: '+name)
    evidence=[{'path':str(p.relative_to(BASE)),'sha256':sha256(p)} for p in (design_path,target_path,review_path,
              BASE/'artifacts/recognition-diagnosis-v3/core-integration/verification.json')]
    evidence.extend({'path':n,'sha256':digest} for n,digest in sources.items())
    result={'schema_version':1,'ready_for_bounded_training':all(gates.values()),'gates':gates,
            'scales':scales,'manifest_sha256':target['manifest_sha256'],'model_sha256':MODELS[MODEL],
            'small_anchor_sha256':target['anchor_config']['anchor_sha256'],'evidence':evidence,
            'scope':'Permission-independent technical preflight for the user-authorized finite computer experiment; not recognition success.',
            'training_plan':'Fresh original K4, unchanged graph and head shapes; heads only, 6epochs, 480steps; fixed epoch6 primary; no test inference.',
            'limitations':['Small anchor IoU>=.5 coverage is48/103, not all103. Regression can improve boxes, but recognition remains to be measured.',
                           'Positive-only public-image data, no camera-domain or background-negative acceptance.',
                           'Integer parameter/byte roundtrip does not prove original DSP arithmetic or camera deployment.'],
            'optimizer_steps':0,'small_face_recognition_verified':False,'camera_deployment_verified':False}
    require(not output.exists(),'Preserve earlier preflight')
    write_json(output,result)
    print(json.dumps({'output':str(output),'passed':result['ready_for_bounded_training'],'gates':gates}))
    require(result['ready_for_bounded_training'],'Training preflight failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=BASE/'artifacts/recognition-diagnosis-v3/small-anchor-preflight/ready.json')
    run(p.parse_args().output)
