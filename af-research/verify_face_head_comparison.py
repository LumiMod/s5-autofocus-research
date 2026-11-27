#!/usr/bin/env python3
"""Independent saved-only face/head cohort, source-row, AP and match audit."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

from pretraining_data import evaluate_detection

BASE=Path(__file__).resolve().parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def iou(a,b):
    inter=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/union if union else 0.


def size_group(frame,box):
    area=(box[2]-box[0])*(box[3]-box[1])*384/frame['width']*256/frame['height']
    return 'below16' if area<256 else '16_to_below32' if area<1024 else '32_to_below96' if area<9216 else '96_or_larger'


def compare(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:compare(a[k],b[k])
    elif isinstance(a,(int,float)) and not isinstance(a,bool):assert abs(a-b)<1e-12,(a,b)
    else:assert a==b,(a,b)


def direct_matches(view,predictions):
    bins={k:dict(ground_truth=0,matched_at_score=0,matched_all_saved_scores=0)
          for k in ('below16','16_to_below32','32_to_below96','96_or_larger')}
    for frame in view.values():
        for box in frame['boxes']:bins[size_group(frame,box['bbox_xyxy'])]['ground_truth']+=1
    used=defaultdict(set); flags={}
    order=sorted(((-p['score'],fid,i,p) for fid,rows in predictions.items() for i,p in enumerate(rows)),key=lambda x:x[:3])
    for minus_score,fid,index,p in order:
        best=-1.; owner=-1
        for j,gt in enumerate(view[fid]['boxes']):
            if j in used[fid]:continue
            value=iou(p['bbox_xyxy'],gt['bbox_xyxy'])
            if value>=best:best,owner=value,j
        hit=owner>=0 and best>=.5
        if hit:
            used[fid].add(owner)
            row=bins[size_group(view[fid],view[fid]['boxes'][owner]['bbox_xyxy'])]
            row['matched_all_saved_scores']+=1
            row['matched_at_score']+=int(-minus_score>=.5)
        flags[(fid,index)]=dict(score=-minus_score,hit=hit,matched_gt=owner if hit else None,matched_iou=best if hit else None)
    for row in bins.values():
        row['recall_at_score']=row['matched_at_score']/row['ground_truth'] if row['ground_truth'] else None
        row['recall_all_saved_scores']=row['matched_all_saved_scores']/row['ground_truth'] if row['ground_truth'] else None
    return flags,bins


def main():
    root=BASE/'artifacts/recognition-ranking-repair/paired-face-head-metrics'
    output=root/'independent-verification.json'
    assert not output.exists()
    plan=json.loads((root/'frozen-plan.json').read_text());report=json.loads((root/'report.json').read_text())
    assert sha(root/'frozen-plan.json')==report['plan_sha256']
    for name,digest in plan['source_sha256'].items():assert sha(root/'source-snapshot'/name)==digest
    manifest=BASE/'private-data/recognition-v2/manifest.json';assert sha(manifest)==plan['dataset_manifest_sha256']
    ds=json.loads(manifest.read_text());originals={f['id']:f for f in ds['frames'] if f['split'] in ('train','validation')}
    hp=BASE/'artifacts/recognition-ranking-repair/box-contract/head-reference.json';assert sha(hp)==plan['head_reference_sha256']
    ref=json.loads(hp.read_text());heads={f['id']:f for f in ref['frames']};assert originals.keys()==heads.keys()
    source=BASE/ref['source_csv'];assert sha(source)==ref['source_csv_sha256']
    source_ids={f['source_image_id']:f for f in originals.values()};expected=defaultdict(list)
    with source.open(newline='') as stream:
        for n,row in enumerate(csv.DictReader(stream),2):
            if row['ImageID'] not in source_ids or row['LabelName']!='/m/04hgtk':continue
            f=source_ids[row['ImageID']]
            expected[f['id']].append((n,[float(row[k])*f['width' if i%2==0 else 'height']
                                       for i,k in enumerate(('XMin','YMin','XMax','YMax'))],
                                     {k:int(row[k]) for k in ('IsOccluded','IsTruncated','IsGroupOf','IsDepiction','IsInside')}))
    for fid,f in heads.items():
        assert f['split']==originals[fid]['split']
        assert [(b['source_csv_row'],b['bbox_xyxy'],b['attributes']) for b in f['boxes']]==expected[fid]
        assert f['head_annotation_available']==bool(expected[fid])
        assert f['head_group_or_depiction_present']==any(r[2]['IsGroupOf'] or r[2]['IsDepiction'] for r in expected[fid])
    paths={
        'validation/original':BASE/'artifacts/recognition-small-anchor-v2/baseline_original/predictions.json',
        'validation/transferred0':BASE/'artifacts/recognition-small-anchor-v2/baseline_small/predictions.json',
        'validation/trained6':BASE/'artifacts/recognition-small-anchor-v2/epoch-06/predictions.json',
        'train32/transferred0':BASE/'artifacts/recognition-fp-diagnosis/head-interactions/transfer0/predictions.json',
        'train32/trained6':BASE/'artifacts/recognition-fp-diagnosis/head-interactions/trained6/predictions.json'}
    result={}
    for name,p in paths.items():
        assert sha(p)==plan['saved_prediction_sha256'][name]
        doc=json.loads(p.read_text());split='train' if name.startswith('train32') else 'validation'
        base_ids=sorted(f['id'] for f in originals.values() if f['split']==split)
        if split=='train':base_ids=base_ids[:32]
        assert sorted(r['id'] for r in doc['frames'])==base_ids
        eligible=sorted(fid for fid in base_ids if expected[fid] and not any(r[2]['IsGroupOf'] or r[2]['IsDepiction'] for r in expected[fid]))
        item=report['results'][name]
        assert eligible==item['frame_ids'] and len(eligible)==item['frames']
        assert sorted(item['excluded_no_head'])==sorted(fid for fid in base_ids if not expected[fid])
        assert sorted(item['excluded_head_group_or_depiction'])==sorted(fid for fid in base_ids if any(r[2]['IsGroupOf'] or r[2]['IsDepiction'] for r in expected[fid]))
        predictions={r['id']:[d for d in r['detections'] if d['class']=='person_face'] for r in doc['frames'] if r['id'] in eligible}
        flags={}
        for label in ('face','head'):
            view={fid:{**originals[fid],'boxes':([b for b in originals[fid]['boxes'] if b['class']=='person_face'] if label=='face'
                  else [dict(bbox_xyxy=r[1],class_='person_face') for r in expected[fid]])} for fid in eligible}
            if label=='head':
                for f in view.values():
                    for b in f['boxes']:b['class']=b.pop('class_')
            reduced=dict(dataset_manifest_sha256=doc['dataset_manifest_sha256'],frames=[dict(id=fid,detections=predictions[fid]) for fid in eligible])
            metric=evaluate_detection(dict(classes=['person_face'],frames=list(view.values())),reduced,split)['classes']['person_face']
            for k,value in metric.items():compare(value,item['metrics'][label][k])
            flags[label],bins=direct_matches(view,predictions)
            compare(bins,item['metrics'][label]['size_strata'])
            expected_recall=sum(r['matched_all_saved_scores'] for r in bins.values())/metric['ground_truth_boxes']
            compare(expected_recall,item['metrics'][label]['recall_all_saved_scores'])
        categories=Counter(both=0,face_only=0,head_only=0,neither=0)
        saved={(r['frame_id'],r['prediction_index']):r for r in item['per_prediction']}
        assert saved.keys()==flags['face'].keys()==flags['head'].keys()
        for key in saved:
            row=saved[key];f=flags['face'][key];h=flags['head'][key]
            compare(f,row['face']);compare(h,row['head'])
            category='both' if f['hit'] and h['hit'] else 'face_only' if f['hit'] else 'head_only' if h['hit'] else 'neither'
            assert row['category']==category and row['score']==f['score']==h['score']
            if f['score']>=.5:categories[category]+=1
        compare(dict(categories),item['same_prediction_matching_at_score05'])
        result[name]=dict(frames=len(eligible),all_saved_predictions=len(saved),metrics={label:{k:item['metrics'][label][k]
                        for k in ('ground_truth_boxes','ap_at_iou','tp_at_score','fp_at_score')} for label in ('face','head')})
    final=dict(passed=True,source_sha256=sha(Path(__file__)),report_sha256=sha(root/'report.json'),
        plan_sha256=sha(root/'frozen-plan.json'),source_reference_rows_rebuilt=True,all_saved_matches_recomputed=True,
        alternative_AP_accumulation='pretraining_data.evaluate_detection',size_recall_recomputed_from_full_one_to_one_matches=True,
        results=result,model_forward=False,optimizer_steps=0,test_access=False,semantic_contract_confirmed=False,
        ready_to_train=False)
    output.write_text(json.dumps(final,indent=2)+'\n');print(json.dumps(final))


if __name__=='__main__':main()
