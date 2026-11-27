#!/usr/bin/env python3
"""Independent saved head-swap audit and loss-gradient toys; never optimizer."""
import json
from pathlib import Path

import numpy as np
import torch

from pretraining_data import evaluate_detection, sha256
from recognition_balanced_loss import balanced_detection_loss, balanced_sample, equal_gt_weights
from train_recognition import CLASSES, assign_joint_targets, encode_boxes, state_digest
from verify_external_baselines import independent_metrics

BASE=Path(__file__).resolve().parent


def close_tree(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:close_tree(a[k],b[k])
    elif isinstance(a,(float,int)) and not isinstance(a,bool):
        assert abs(a-b)<=1e-12,(a,b)
    else:assert a==b,(a,b)


def head_review():
    root=BASE/'artifacts/recognition-fp-diagnosis/head-interactions'
    run=BASE/'artifacts/recognition-small-anchor-v2'
    report=json.loads((root/'report.json').read_text())
    plan=json.loads((root/'frozen-plan.json').read_text())
    assert sha256(root/'frozen-plan.json')==report['plan_sha256']
    for name,digest in plan['source_hashes'].items():
        assert sha256(root/'source-snapshot'/name)==digest
    manifest=BASE/'private-data/recognition-v2/manifest.json'
    assert sha256(manifest)==plan['dataset_manifest_sha256']
    dataset=json.loads(manifest.read_text())
    frames=sorted((f for f in dataset['frames'] if f['split']=='train'),key=lambda f:f['id'])[:32]
    assert [f['id'] for f in frames]==plan['frame_ids']
    paths={'initial':run/'baseline_small/checkpoint.pt','final':run/'epoch-06/checkpoint.pt'}
    states={}
    for key,path in paths.items():
        assert sha256(path)==plan['checkpoint_sha256'][key]
        states[key]=torch.load(path,map_location='cpu',weights_only=False)['model_state']
    assert sha256(run/'anchors-small.npy')==plan['anchor_file_sha256']
    for key,value in states['initial'].items():
        if not key.startswith(('convs.86.','convs.90.','convs.94.')):
            assert torch.equal(value,states['final'][key])
    results={}
    for condition,(class_source,box_source) in plan['conditions'].items():
        expected={key:states[class_source if key.startswith(('convs.86.','convs.90.'))
                            else box_source if key.startswith('convs.94.') else 'initial'][key]
                  for key in states['initial']}
        pred_path=root/condition/'predictions.json'
        doc=json.loads(pred_path.read_text())
        assert sha256(pred_path)==report['results'][condition]['predictions_sha256']
        assert doc['model_state_sha256']==state_digest(expected)==report['results'][condition]['state_sha256']
        assert doc['dataset_manifest_sha256']==plan['dataset_manifest_sha256']
        assert doc['plan_sha256']==report['plan_sha256']
        assert [f['id'] for f in doc['frames']]==plan['frame_ids']
        saved=json.loads((root/condition/'metrics.json').read_text())
        assert saved==report['results'][condition]['metrics']
        independent={}
        for cls in CLASSES:
            predictions={r['id']:[d for d in r['detections'] if d['class']==cls] for r in doc['frames']}
            independent[cls]=independent_metrics({f['id']:f for f in frames},predictions,cls)
            close_tree(independent[cls],saved['classes'][cls])
        legacy=evaluate_detection({'classes':list(CLASSES),'frames':frames},doc,'train')
        close_tree(legacy,saved['legacy_crosscheck'])
        face=independent['person_face']['size_strata']
        small={k:sum(face[b][k] for b in ('below16','16_to_below32'))
               for k in ('ground_truth','matched_at_score','matched_all_saved_scores')}
        small['recall_at_score']=small['matched_at_score']/small['ground_truth']
        close_tree(small,saved['small_faces'])
        results[condition]=dict(state_sha256=state_digest(expected),
            face_tp=independent['person_face']['tp_at_score'],face_fp=independent['person_face']['fp_at_score'],
            person_tp=independent['person']['tp_at_score'],person_fp=independent['person']['fp_at_score'],small_faces=small)
    return dict(passed=True,report_sha256=sha256(root/'report.json'),results=results,
        scope='Saved predictions recomputed; checkpoint mixed-state digests independently rebuilt; source snapshots hashed. No model forward or input tensor recreation.')


def loss_review():
    checks=[]
    rng=np.random.default_rng(501)
    owners=np.array([0]*40+[1]*9+[2]*2)
    selected=balanced_sample(np.arange(len(owners)),np.array([0,40,49]),owners,32,rng)
    counts=np.bincount(owners[selected],minlength=3)
    assert list(counts)==[21,9,2],counts
    weights=equal_gt_weights(selected,owners)
    mass=[float(weights[owners[selected]==i].sum()) for i in range(3)]
    assert max(mass)-min(mass)<2e-6 and abs(sum(mass)-32)<3e-6
    checks.append(dict(name='scarce_groups_exhausted_then_balanced; equal_GT_mass',counts=counts.tolist(),mass=mass))
    # Two GTs per head, very unequal anchor replication. This is a loss toy,
    # not a synthetic recognition result. Positive anchors are exactly at GTs.
    gt={'person':np.array([[0,0,100,150],[120,0,180,100]],np.float32),
        'person_face':np.array([[25,10,45,30],[135,10,151,26]],np.float32)}
    ref=np.concatenate([np.repeat(gt['person'][0:1],6,0),np.repeat(gt['person'][1:2],2,0),
                        np.repeat(gt['person_face'][0:1],5,0),gt['person_face'][1:2],
                        np.array([[200,150,220,170],[250,180,270,200]],np.float32)])
    assigned=assign_joint_targets(ref,gt)
    delta=np.zeros((len(ref),4),np.float32)
    pos=np.flatnonzero(assigned['matched_gt']>=0)
    delta[pos]=encode_boxes(ref[pos],assigned['gt_boxes'][assigned['matched_gt'][pos]])+.02
    # This static negative has already regressed to the correct first face.
    delta[14]=encode_boxes(ref[14:15],gt['person_face'][0:1])[0]
    predicted_delta=torch.tensor(delta,requires_grad=True)
    logits={c:torch.zeros((len(ref),2),requires_grad=True) for c in CLASSES}
    loss,info,trace=balanced_detection_loss(logits,predicted_delta,ref,gt,np.random.default_rng(17),return_trace=True)
    loss.backward()
    assert torch.isfinite(loss)
    assert 14 in trace['classes']['person_face']['guarded_negative']
    assert torch.count_nonzero(logits['person_face'].grad[14])==0
    assert logits['person_face'].grad[15,1]>0
    class_masses={}
    for ci,cls in enumerate(CLASSES):
        row=trace['classes'][cls]
        assert not set(row['negative']) & set(row['guarded_negative'])
        assert set(row['hard_negative']).isdisjoint(row['random_negative'])
        assert np.isin(assigned['forced_anchor_by_gt'][assigned['gt_classes']==ci],row['positive']).all()
        assert (logits[cls].grad[row['positive'],1]<0).all()
        masses=[float(-logits[cls].grad[row['positive'][assigned['matched_gt'][row['positive']]==gi],1].sum())
                for gi in np.flatnonzero(assigned['gt_classes']==ci)]
        assert abs(masses[0]-masses[1])<1e-7
        class_masses[cls]=masses
    bbox_masses=[float(predicted_delta.grad[trace['bbox_positive'][assigned['matched_gt'][trace['bbox_positive']]==gi]].abs().sum())
                 for gi in range(4)]
    assert max(bbox_masses)-min(bbox_masses)<1e-7
    assert torch.count_nonzero(predicted_delta.grad[14:])==0
    checks.append(dict(name='loss_gradients_equal_GT_and_guard_no_gradient',
        class_positive_gradient_mass=class_masses,bbox_gradient_mass=bbox_masses,guarded_anchor=14,
        true_background_foreground_logit_gradient=float(logits['person_face'].grad[15,1])))
    # Empty GT / all background and an empty valid-anchor set must remain finite.
    for label,boxes in [('all_background',ref),('out_of_bounds',np.array([[400,300,420,320]],np.float32))]:
        empty={c:np.empty((0,4),np.float32) for c in CLASSES}
        lg={c:torch.zeros((len(boxes),2),requires_grad=True) for c in CLASSES}
        dd=torch.zeros((len(boxes),4),requires_grad=True)
        out,_,tr=balanced_detection_loss(lg,dd,boxes,empty,np.random.default_rng(2),return_trace=True)
        out.backward()
        assert torch.isfinite(out) and all(torch.isfinite(v.grad).all() for v in lg.values())
        assert len(tr['bbox_positive'])==0
        checks.append(dict(name=label,passed=True,loss=float(out.detach())))
    # More than 128 negatives forces both mining and random filling. Foreground
    # logits rise strictly with index, so independent expected hard64 is exact.
    many=np.repeat(np.array([[200,150,220,170]],np.float32),200,axis=0)
    empty={c:np.empty((0,4),np.float32) for c in CLASSES}
    lg={c:torch.stack((torch.zeros(200),torch.linspace(-2,2,200)),1).requires_grad_() for c in CLASSES}
    dd=torch.zeros((200,4),requires_grad=True)
    out,_,tr=balanced_detection_loss(lg,dd,many,empty,np.random.default_rng(73),return_trace=True)
    out.backward()
    for cls in CLASSES:
        row=tr['classes'][cls]
        assert row['hard_negative'].tolist()==list(range(199,135,-1))
        assert len(row['random_negative'])==64 and len(set(row['negative']))==128
        assert (lg[cls].grad[row['negative'],1]>0).all()
        unselected=np.setdiff1d(np.arange(200),row['negative'])
        assert torch.count_nonzero(lg[cls].grad[unselected])==0
    checks.append(dict(name='highest_CE_hard64_random64_exact_budget_and_gradient',passed=True))
    return dict(passed=True,checks=checks,source_sha256=sha256(BASE/'recognition_balanced_loss.py'),
        limitations=['Synthetic derivatives verify bookkeeping, not image recognition or future optimizer stability.',
                     'Decoded IoU guard is a training target rule dependent on current detached boxes; it cannot itself remove a false detection from inference.',
                     'Per-GT balancing preserves positive/bbox total sample mass; hard mining intentionally changes which negatives contribute.'])


def main():
    output=BASE/'artifacts/recognition-fp-diagnosis/target-conflicts/independent-head-and-loss-review.json'
    assert not output.exists(),'Preserve prior review'
    report=dict(head_interactions=head_review(),balanced_loss=loss_review(),
        verifier_sha256=sha256(Path(__file__)),optimizer_steps=0,model_forward=False,
        validation_inference=False,test_inference=False,ready_to_train=False)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
