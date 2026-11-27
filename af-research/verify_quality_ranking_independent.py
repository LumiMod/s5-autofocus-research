#!/usr/bin/env python3
"""Independent normalized-ranking toys and saved train32 geometry; no forward."""
import copy
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from recognition_quality_ranking import quality_ranking_loss
from train_recognition import CLASSES, assign_joint_targets, box_iou, ground_truth

BASE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def scalar_iou(a,b):
    overlap=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    area_a=(a[2]-a[0])*(a[3]-a[1]);area_b=(b[2]-b[0])*(b[3]-b[1])
    return overlap/(area_a+area_b-overlap)


def fixture():
    anchors=np.array([[0,0,10,10],[30,0,40,10],[60,0,70,10],
                      [100,0,110,10],[130,0,140,10],[160,0,170,10]],np.float32)
    gt={'person':np.empty((0,4),np.float32),'person_face':anchors[[0,3]].copy()}
    assigned=assign_joint_targets(anchors,gt)
    trace={'assigned':assigned,'classes':{},'bbox_positive':np.array([0,3],np.int64)}
    for cls in CLASSES:
        p=np.array([0,3],np.int64) if cls=='person_face' else np.empty(0,np.int64)
        n=np.array([2,5],np.int64) if cls=='person_face' else np.arange(6)
        trace['classes'][cls]=dict(positive=p,negative=n,positive_weights=np.ones(len(p),np.float32))
    decoded=torch.tensor([[0,0,10,10],[0,0,25,10],[60,0,70,10],
                          [100,0,110,10],[100,0,125,10],[160,0,170,10]],dtype=torch.float64,requires_grad=True)
    logits={c:torch.zeros((6,2),dtype=torch.float64,requires_grad=True) for c in CLASSES}
    logits['person_face']=torch.tensor([[0,1],[0,3],[0,0],[0,1],[0,3],[0,0]],dtype=torch.float64,requires_grad=True)
    return logits,decoded,trace


def toy_checks():
    checks=[]
    logits,decoded,trace=fixture();before=copy.deepcopy(trace)
    loss,details=quality_ranking_loss(logits,decoded,trace,return_trace=True)
    expected=.3*F.softplus(torch.tensor(2.,dtype=torch.float64))
    torch.testing.assert_close(loss,expected,rtol=1e-12,atol=1e-12)
    gradients=torch.autograd.grad(loss,[logits['person_face'],decoded],allow_unused=True,retain_graph=True)
    assert gradients[1] is None
    expected_grad=.15/(1+math.exp(-2))
    for good,bad in ((0,1),(3,4)):
        assert abs(float(gradients[0][good,1])+expected_grad)<1e-12
        assert abs(float(gradients[0][bad,1])-expected_grad)<1e-12
    assert torch.count_nonzero(gradients[0][[2,5]])==0
    assert torch.count_nonzero(gradients[0].sum(1))==0
    # Explicit base CE and independent bbox tensor: ranking must change only
    # selected class logits, never geometry or the bbox gradient.
    row=trace['classes']['person_face'];indices=np.concatenate((row['positive'],row['negative']))
    labels=torch.tensor([1,1,0,0]);weights=torch.tensor([1.,1.,1.,1.],dtype=torch.float64)
    base_ce=(F.cross_entropy(logits['person_face'][indices],labels,reduction='none')*weights).sum()/4
    delta=torch.tensor([[.02,.03,.04,.05],[.05,.04,.03,.02]],dtype=torch.float64,requires_grad=True)
    bbox=F.smooth_l1_loss(delta,torch.zeros_like(delta),beta=1/9)
    original=torch.autograd.grad(base_ce+bbox,[logits['person_face'],delta],retain_graph=True)
    combined=torch.autograd.grad(base_ce+bbox+loss,[logits['person_face'],delta],retain_graph=True)
    assert torch.equal(original[1],combined[1])
    assert all(float(combined[0][i,1])<0 for i in (0,3))
    assert all(float(combined[0][i,1])>0 for i in (1,4))
    checks.append(dict(name='two_GT_exact_normalized_loss_and_net_gradient',passed=True,
                       pair_effective_mass=.15,total_effective_mass=.3,base_positive_CE_target_mass=.5))
    # Removing one eligible pair must not boost the remaining GT's mass.
    one={c:v.detach().clone().requires_grad_(True) for c,v in logits.items()}
    with torch.no_grad():one['person_face'][4,1]=-2.
    one_loss=quality_ranking_loss(one,decoded,trace)
    torch.testing.assert_close(one_loss,expected/2,rtol=1e-12,atol=1e-12)
    checks.append(dict(name='one_pair_removed_no_effective_pair_renormalization',passed=True))
    # No-GT or no-valid-good returns graph-connected zero.
    no_good=decoded.detach().clone();no_good[0]=no_good[2];no_good[3]=no_good[5]
    zero=quality_ranking_loss(logits,no_good,trace)
    assert float(zero)==0
    # Exact IoU .5 cannot be selected as bad; exact .3 overlap with another GT
    # triggers protection. Test both without modifying stored sampling.
    exact=decoded.detach().clone();exact[1]=torch.tensor([0.,0.,20.,10.]);exact[4]=torch.tensor([100.,0.,120.,10.])
    assert float(quality_ranking_loss(logits,exact,trace))==0
    bad_owner=copy.deepcopy(trace);bad_owner['assigned']['matched_gt'][1]=0;bad_owner['assigned']['matched_gt'][4]=1
    assert float(quality_ranking_loss(logits,decoded,bad_owner))==0
    protected=copy.deepcopy(trace);protected['bbox_positive']=np.array([0],np.int64)
    try:quality_ranking_loss(logits,decoded,protected)
    except ValueError:pass
    else:raise AssertionError('Missing bbox supervision was accepted')
    ambiguous=copy.deepcopy(trace)
    new_ref=np.array([[0,0,10,10],[30,0,40,10],[60,0,70,10],
                      [12,0,22,10],[130,0,140,10],[160,0,170,10]],np.float32)
    ambiguous['assigned']=assign_joint_targets(new_ref,{'person':np.empty((0,4),np.float32),
                                                       'person_face':new_ref[[0,3]]})
    ambiguous_decoded=decoded.detach().clone();ambiguous_decoded[3]=torch.tensor([12.,0.,22.,10.])
    assert float(quality_ranking_loss(logits,ambiguous_decoded,ambiguous))==0
    for key in ('matched_gt','gt_classes','gt_boxes','forced_anchor_by_gt'):
        np.testing.assert_array_equal(trace['assigned'][key],before['assigned'][key])
    checks.append(dict(name='no_good_correct_half_IoU_owned_bad_cross_GT_and_missing_bbox_protections',passed=True))
    return checks


def real_checks(report,root):
    dataset_path=BASE/'private-data/recognition-v2/manifest.json'
    dataset=json.loads(dataset_path.read_text())
    frames={f['id']:f for f in dataset['frames'] if f['split']=='train'}
    expected_ids=sorted(frames)[:32]
    assert sorted(r['frame_id'] for r in report['frames'])==expected_ids
    base_path=BASE/'artifacts/recognition-fp-diagnosis/balanced-loss-verification/report.json'
    bases={r['frame_id']:r for r in json.loads(base_path.read_text())['frames']}
    stage=BASE/'artifacts/recognition-diagnosis-v3/small-face-after-v2-train32'
    stage_report=json.loads((stage/'report.json').read_text());stages={r['frame_id']:r for r in stage_report['baseline']}
    anchors=np.load(BASE/'artifacts/recognition-small-anchor-v2/anchors-small.npy',allow_pickle=False)
    hpath=BASE/'artifacts/recognition-ranking-repair/box-contract/head-reference.json'
    href=json.loads(hpath.read_text());heads={f['id']:f for f in href['frames'] if f['split']=='train'}
    old=json.loads((BASE/'artifacts/recognition-fp-diagnosis/quality-ranking-verification/report.json').read_text())
    old_pairs={(r['frame_id'],p['target_class'],p['union_gt_index'],p['good_anchor'],p['bad_anchor'])
               for r in old['frames'] for p in r['ranking']['pairs']}
    pair_ids=set();mass=[];geometry=[];counts=Counter();max_gradient_error=0.;npz_hashes={}
    for row in report['frames']:
        fid=row['frame_id'];frame=frames[fid];base=bases[fid]
        assignment=assign_joint_targets(anchors,ground_truth(frame));owners=assignment['matched_gt']
        npz=stage/stages[fid]['prediction_file'];assert sha(npz)==stages[fid]['saved_prediction_sha256']
        npz_hashes[fid]=sha(npz)
        with np.load(npz,allow_pickle=False) as raw:decoded=raw['decoded_boxes'].copy()
        by_class={c:[] for c in CLASSES}
        actual_loss=0.
        for p in row['ranking']['pairs']:
            cls=p['target_class'];ci=CLASSES.index(cls);gi=p['union_gt_index'];good=p['good_anchor'];bad=p['bad_anchor']
            pair_ids.add((fid,cls,gi,good,bad));by_class[cls].append(p)
            cbase=base['class_checks'][cls];positives=np.array(cbase['selected_positive']);negative=cbase['selected_negative']
            weights=np.array(cbase['positive_weights']);sample_count=len(positives)+len(negative)
            assert good in positives and good in base['bbox_indices'] and bad not in negative
            assert owners[good]==gi and assignment['gt_classes'][gi]==ci and owners[bad]==-1
            assert assignment['labels'][cls][good]==1 and assignment['labels'][cls][bad] in (0,-1)
            quality=scalar_iou(decoded[good],assignment['gt_boxes'][gi]);bad_quality=scalar_iou(decoded[bad],assignment['gt_boxes'][gi])
            assert quality>=.5 and .3<=bad_quality<.5 and scalar_iou(decoded[good],decoded[bad])>.3
            others=[i for i,c in enumerate(assignment['gt_classes']) if c==ci and i!=gi]
            assert all(scalar_iou(decoded[k],assignment['gt_boxes'][i])<.3 for k in (good,bad) for i in others)
            # The normalized real calculation uses float64 NumPy boxes while GT
            # coordinates are the frozen float32 training values.
            q=float(box_iou(decoded[[good]].astype(np.float64),assignment['gt_boxes'][[gi]])[0,0])
            bq=float(box_iou(decoded[[bad]].astype(np.float64),assignment['gt_boxes'][[gi]])[0,0])
            assert abs(q-p['good_iou'])<1e-12 and abs(bq-p['bad_iou'])<1e-12
            owner_mass=float(weights[owners[positives]==gi].sum())/sample_count
            coefficient=owner_mass*(q-bq)
            zg=p['good_log_odds'];zb=p['bad_log_odds'];assert zb>=zg
            sigmoid=1/(1+math.exp(-(zb-zg)))
            expected_ranking=coefficient*sigmoid
            good_index=np.flatnonzero(positives==good);assert len(good_index)==1
            expected_base=float(weights[good_index[0]])/sample_count*(1/(1+math.exp(-zg))-1)
            for target,sign,base_expected in [('good',-1,expected_base),('bad',1,0.)]:
                g=p['foreground_logit_gradients'][target]
                for actual,expected in [(g['base'],base_expected),(g['ranking'],sign*expected_ranking),(g['combined'],base_expected+sign*expected_ranking)]:
                    error=abs(actual-expected);max_gradient_error=max(max_gradient_error,error);assert error<2e-7
            actual_loss+=coefficient*float(np.logaddexp(0.,zb-zg))
            mass.append(dict(frame_id=fid,target_class=cls,gt_index=gi,base_GT_positive_mass=owner_mass,effective_pair_coefficient=coefficient))
            counts[cls+'_pairs']+=1
            if cls!='person_face':continue
            hframe=heads[fid]
            grow=dict(frame_id=fid,union_gt_index=gi,good_anchor=good,bad_anchor=bad,
                face_good_iou=q,face_bad_iou=bq,head_annotation_available=hframe['head_annotation_available'],
                head_group_or_depiction_present=hframe['head_group_or_depiction_present'],
                inference='No source instance links; every same-image head is retained for geometry only.')
            if not hframe['head_annotation_available'] or hframe['head_group_or_depiction_present']:
                counts['face_pairs_head_reference_unavailable_or_flagged']+=1;geometry.append(grow);continue
            comparisons=[]
            for hb in hframe['boxes']:
                box=np.array(hb['normalized_xyxy'])*np.array([384,256,384,256])
                comparisons.append(dict(source_csv_row=hb['source_csv_row'],attributes=hb['attributes'],
                    head_reference_xyxy=box.tolist(),good_head_iou=scalar_iou(decoded[good],box),
                    bad_head_iou=scalar_iou(decoded[bad],box),face_target_head_iou=scalar_iou(assignment['gt_boxes'][gi],box)))
            gm=max(v['good_head_iou'] for v in comparisons);bm=max(v['bad_head_iou'] for v in comparisons)
            reversal=any(v['bad_head_iou']>=.5 and v['good_head_iou']<.5 for v in comparisons)
            grow.update(heads=comparisons,good_max_head_iou=gm,bad_max_head_iou=bm,
                some_same_head_bad_ge_05_good_lt_05=reversal)
            counts['face_pairs_with_eligible_head_reference']+=1
            counts['bad_max_head_iou_greater_than_good_max']+=int(bm>gm)
            counts['good_max_head_iou_greater_than_bad_max']+=int(gm>bm)
            counts['some_same_head_bad_ge_05_good_lt_05']+=int(reversal)
            geometry.append(grow)
        assert abs(actual_loss-row['ranking_loss'])<3e-7
        for cls in CLASSES:
            pairs=by_class[cls];cbase=base['class_checks'][cls]
            budget=sum(cbase['positive_weights'])/(len(cbase['selected_positive'])+len(cbase['selected_negative']))
            rows=[m for m in mass if m['frame_id']==fid and m['target_class']==cls]
            assert len({p['union_gt_index'] for p in pairs})==len(pairs)
            assert len({p['good_anchor'] for p in pairs})==len(pairs) and len({p['bad_anchor'] for p in pairs})==len(pairs)
            total=sum(m['effective_pair_coefficient'] for m in rows)
            assert total<=.7*budget+1e-9
    assert pair_ids==old_pairs and len(pair_ids)==50
    return dict(real_pairs=50,counts=dict(counts),all_pair_membership_preserved=True,
        unique_owner_and_bbox_supervision_passed=True,other_same_class_GT_ambiguity_excluded=True,
        expected_scalar_gradients_recomputed=True,max_saved_scalar_gradient_absolute_error=max_gradient_error,
        normalized_effective_pair_mass_bound='Per class sum coefficients <= .7 * base positive CE target mass (and therefore <= that mass).',
        bound_limitation='This bounds target coefficients, not the ratio to current CE gradients or the unbounded softplus loss value.',
        mass_by_pair=mass,face_head_pair_geometry=geometry,npz_sha256=npz_hashes,
        head_reference_sha256=sha(hpath),base_verification_sha256=sha(base_path),
        real_bbox_gradient_scope='Unchanged follows from detached ranking graph and toy bitwise gradient test; actual report bbox equality is producer-reported, not independently reconstructed from clipped decoded boxes.')


def main():
    root=BASE/'artifacts/recognition-ranking-repair/quality-ranking-normalized'
    output=root/'independent-verification.json';assert not output.exists()
    report=json.loads((root/'report.json').read_text());plan=json.loads((root/'frozen-plan.json').read_text())
    assert sha(root/'frozen-plan.json')==report['frozen_plan_sha256']
    for name,digest in report['source_hashes'].items():
        assert sha(root/'source-snapshot'/name)==digest
        if name in ('recognition_quality_ranking.py','recognition_balanced_loss.py','train_recognition.py'):
            assert sha(BASE/name)==digest
    checks=toy_checks();real=real_checks(report,root)
    final=dict(passed_supervision_math_checks=True,verifier_sha256=sha(Path(__file__)),report_sha256=sha(root/'report.json'),
        frozen_plan_sha256=sha(root/'frozen-plan.json'),toy_checks=checks,saved_pair_checks=real,
        model_forward=False,optimizer_steps=0,validation_evaluation=False,test_access=False,
        raw_factory_face_head_contract_confirmed=False,passed_detection_acceptance=False,ready_to_train=False,
        limitations=['Gradient direction is for the explicitly chosen face target, not verified factory head semantics.',
                    'Saved dual-definition geometry is not new model inference, held-out acceptance, instance correspondence, or a reason to change historical TP/FP.',
                    'No actual optimizer step is performed; output-gradient correctness does not demonstrate post-update ranking or detection improvement.'])
    output.write_text(json.dumps(final,indent=2)+'\n')
    print(json.dumps(dict(report=str(output),sha256=sha(output),counts=real['counts'],maximum_gradient_error=real['max_saved_scalar_gradient_absolute_error'])))


if __name__=='__main__':main()
