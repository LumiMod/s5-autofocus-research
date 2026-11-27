#!/usr/bin/env python3
"""Build a compact shareable K4 evidence index without images or weights.

Reads saved reports only, checks their frozen source snapshots, and preserves
separate host packing, small-face, training and camera-equivalence outcomes.
"""
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent
ART=BASE/'artifacts/recognition-diagnosis-v3'
OUT=BASE/'evidence/2026-11-27/recognition-diagnosis-v3/summary.json'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(name):
    return json.loads((ART/name).read_text())


def main():
    required=[
        'static-constraints/static-constraints.json','static-constraints/root-decoder-review.json',
        'static-constraints/quant-metadata.json','graph-operators/static-synthetic-audit.json',
        'flattened-k4-train32/frozen-plan.json','flattened-k4-train32/summary.json',
        'flattened-k4-train32/independent-verification.json',
        'k4-preprocessing-train32/frozen-plan.json','k4-preprocessing-train32/summary.json',
        'k4-decoder-train8/frozen-plan.json','k4-decoder-train8/summary.json',
        'k4-validation80/frozen-plan.json','k4-validation80/summary.json',
        'k4-validation80/independent-verification.json','paired-behavior/summary.json',
        'small-face-path-train32-meanstat-fix/report.json',
        'small-face-path-train32-meanstat-fix/independent-verification.json',
        'core-integration/verification.json','single-photo-smoke/verification.json']
    documents={name:read(name) for name in required}
    # Old code identities remain bound to snapshots after explicit core integration.
    bindings=[]
    for report,doc in documents.items():
        for field in ('source_hashes','source_sha256','sources'):
            mapping=doc.get(field)
            if not isinstance(mapping,dict):continue
            for name,digest in mapping.items():
                if not name.endswith('.py') or not isinstance(digest,str) or len(digest)!=64:continue
                choices=[BASE/name,*ART.glob('**/source-snapshot/'+name)]
                matches=[p for p in choices if p.is_file() and sha(p)==digest]
                if not matches:raise ValueError('No actual frozen source matches '+report+':'+name)
                bindings.append({'report':report,'source':name,'sha256':digest,
                                 'matched':str(matches[0].relative_to(BASE))})
    train=documents['flattened-k4-train32/summary.json']
    val=documents['k4-validation80/summary.json']
    small=documents['small-face-path-train32-meanstat-fix/report.json']
    pair=documents['paired-behavior/summary.json']
    for doc in (train,val,small):
        assert doc['status']=='complete' and doc['optimizer_steps']==0 and not doc['test_inference']
    assert small['actual_state_unchanged'] and small['all_sources_unchanged']
    assert not small['anchors_changed'] and not small['parameters_changed']
    for result in val['results'].values():
        face=result['independent']['real']['person_face']['size_strata']
        assert sum(face[b]['ground_truth'] for b in ('below16','16_to_below32'))==34
        assert sum(face[b]['matched_all_saved_scores'] for b in ('below16','16_to_below32'))==0
    indexed=[]
    for path in sorted(ART.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            indexed.append({'path':str(path.relative_to(BASE)),'bytes':path.stat().st_size,'sha256':sha(path)})
    paired={}
    for name in pair['profiles']:
        report=read('paired-behavior/runs/'+name+'/report.json')
        paired[name]=report['aggregates']
    summary={
        'schema_version':1,'date':'2026-11-27','status':'host K4 layout correction validated; small-face gate still fails',
        'source_model':'44_hm_d_nw_sng','source_model_sha256':small['model_sha256'],
        'manifest_sha256':small['manifest_sha256'],
        'weight_layout':{'name':'gemm_k4','physical_axes':'[O/4,(I*H*W)/4,O4,K4]',
                         'logical_axes':'OIHW','ordinary_convolutions_verified_across_two_models':126,
                         'upsample_excluded':True,'historical_three_layouts_preserved':True},
        'train32':{n:r['metrics'] for n,r in train['results'].items()},
        'validation80':{n:{'metrics':r['metrics'],'independent':r['independent']} for n,r in val['results'].items()},
        'preprocessing_train32':{n:r['metrics'] for n,r in documents['k4-preprocessing-train32/summary.json']['results'].items()},
        'decoder_train8':documents['k4-decoder-train8/summary.json']['results'],
        'small_face_path_train32':{'baseline':small['baseline_summary'],'oracle_centered_crops':small['oracle_summary'],
                                 'oracle_limit':'Uses GT center, changes size and source detail; not automatic detection or deployable fix.'},
        'paired_train16':{'completed_inferences':pair['inferences_in_completed_runs'],'aggregates':paired,
                         'limit':'Conditional following excludes prior misses; all-GT denominators retained. Confidence values are not calibrated across models.'},
        'external_reference':'../external-baselines/summary.json',
        'verification_reports':[n for n in required if 'verification' in n or 'review' in n],
        'core_usable_inference_entry':'infer_s5_k4.py',
        'training_entry_warning':'Historical train_recognition.py CLI still selects only legacy layouts; do not use it to start K4 training or resume v2 checkpoints.',
        'host_k4_implementation_verified':True,'image_dependent_recognition_restored':True,
        'verified_small_face_fix':False,'ready_to_resume_recognition_training':False,
        'verified_dsp_numerical_parity':False,'verified_camera_deployment':False,
        'optimizer_steps_this_iteration':0,'held_out_test_inference':False,
        'remaining':['Small-face decoded geometry and classification confidence fail.',
                     'Exact camera pixel conversion/crop/resize and DSP arithmetic remain unobserved.',
                     'Pool/upsample/activation and proposal ABI retain unverified details.',
                     'Different small-anchor settings would be a new detector configuration requiring fresh training and deployment validation.'],
        'limits':['Positive-only OpenImages subset; not official COCO/OpenImages scores, negative-image or camera-domain validation.',
                  'Letterbox wrong-image mapping uses the network canvas and target inverse transform; it differs from external normalized-original-image mapping. Clipped-away endpoints cannot be fully recovered from saved boxes.',
                  'AP improvements and host implementation checks do not establish original DSP equivalence.',
                  'A failed mean-statistic-only diagnostic attempt is retained separately; the corrected completed run is named explicitly.'],
        'frozen_source_bindings':bindings,'artifact_identities':indexed,
        'summarizer_sha256':sha(Path(__file__))}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'output':str(OUT),'sha256':sha(OUT),'artifacts':len(indexed),'source_bindings':len(bindings),
                      'training_ready':False}))


if __name__=='__main__':main()
