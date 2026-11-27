#!/usr/bin/env python3
"""Preserve original Open Images head rows for existing train/validation IDs."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent
HEAD='/m/04hgtk'
FLAGS=('IsOccluded','IsTruncated','IsGroupOf','IsDepiction','IsInside')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def overlap(a,b):
    intersection=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    aa=(a[2]-a[0])*(a[3]-a[1]);bb=(b[2]-b[0])*(b[3]-b[1])
    return dict(iou=intersection/(aa+bb-intersection),fraction_first_covered=intersection/aa,
                fraction_second_covered=intersection/bb)


def main():
    data_root=BASE/'private-data/recognition-v2'
    manifest=data_root/'manifest.json'
    csv_path=data_root/'raw/validation-bbox.csv'
    output=BASE/'artifacts/recognition-ranking-repair/box-contract/head-reference.json'
    assert not output.exists(),'Preserve prior reference'
    data=json.loads(manifest.read_text())
    source={f['source_image_id']:f for f in data['frames'] if f['split'] in ('train','validation')}
    assert len(source)==400 and all(f['source_official_split']=='validation' for f in source.values())
    frames={sid:{k:f[k] for k in ('id','split','width','height','source_image_id','source_official_split','sha256')}
            | {'boxes':[]} for sid,f in source.items()}
    with csv_path.open(newline='') as stream:
        for line,row in enumerate(csv.DictReader(stream),2):
            if row['ImageID'] not in source or row['LabelName']!=HEAD:continue
            frame=frames[row['ImageID']]
            normalized=[float(row[key]) for key in ('XMin','YMin','XMax','YMax')]
            assert 0<=normalized[0]<normalized[2]<=1 and 0<=normalized[1]<normalized[3]<=1
            frame['boxes'].append(dict(class_name='Human head',source_label=HEAD,source=row['Source'],
                confidence=row['Confidence'],source_csv_row=line,normalized_xyxy=normalized,
                bbox_xyxy=[v*frame['width' if i%2==0 else 'height'] for i,v in enumerate(normalized)],
                attributes={flag:int(row[flag]) for flag in FLAGS}))
    counts={}
    for split in ('train','validation'):
        chosen=[f for f in frames.values() if f['split']==split]
        counts[split]=dict(frames=len(chosen),frames_with_head_boxes=sum(bool(f['boxes']) for f in chosen),
            source_head_rows=sum(len(f['boxes']) for f in chosen),
            source_flags={flag:dict(Counter(b['attributes'][flag] for f in chosen for b in f['boxes'])) for flag in FLAGS})
    for frame in frames.values():
        frame['head_annotation_available']=bool(frame['boxes'])
        frame['head_group_or_depiction_present']=any(b['attributes']['IsGroupOf'] or b['attributes']['IsDepiction'] for b in frame['boxes'])
    selection_path=BASE/'artifacts/recognition-fp-diagnosis/nms-evidence/selection.json'
    selected=json.loads(selection_path.read_text())
    comparisons=[]
    for case in selected['cases']:
        frame=frames[case['source_image_id']]
        candidate=case['displayed_candidate']['bbox_xyxy']
        suppressor=case['displayed_candidate']['suppressor']['bbox_xyxy']
        face=case['target_reference_xyxy']
        comparisons.append(dict(case=case['case'],frame_id=case['frame_id'],
            face_gt_reference_xyxy=face,face_gt_attributes=case['gt_attributes'],
            source_head_boxes=[dict(source_csv_row=b['source_csv_row'],attributes=b['attributes'],
                head_reference_xyxy=[v*384 if i%2==0 else v*256 for i,v in enumerate(b['normalized_xyxy'])],
                overlap_with_face=overlap([v*384 if i%2==0 else v*256 for i,v in enumerate(b['normalized_xyxy'])],face),
                overlap_with_candidate=overlap([v*384 if i%2==0 else v*256 for i,v in enumerate(b['normalized_xyxy'])],candidate),
                overlap_with_suppressor=overlap([v*384 if i%2==0 else v*256 for i,v in enumerate(b['normalized_xyxy'])],suppressor))
                for b in frame['boxes']]))
    result=dict(status='source_reference_only',manifest_sha256=sha(manifest),source_csv=str(csv_path.relative_to(BASE)),
        source_csv_sha256=sha(csv_path),class_mapping_file='private-data/recognition-v2/raw/classes.csv',
        class_mapping_sha256=sha(data_root/'raw/classes.csv'),hierarchy_sha256=sha(data_root/'raw/hierarchy.json'),
        script_sha256=sha(Path(__file__)),target_source_label=HEAD,target_name='Human head',
        coordinate_contract='Original normalized XMin,YMin,XMax,YMax multiplied by width,height,width,height; no +1, enlargement, clipping or relabelling.',
        flag_policy='All original head rows and flags retained. Exclude an entire head-evaluation image if head GroupOf or Depiction is present; do not treat unannotated-head frames as verified negatives.',
        no_head_box_semantics='Unknown/unavailable head reference, not a verified head-negative image.',
        source_no_instance_links='Official rows have no person-instance ID linking face to head. NMS comparisons include every head in the same image; geometry alone is not an identity proof.',
        counts=counts,frames=sorted(frames.values(),key=lambda f:f['id']),
        nms_selection_sha256=sha(selection_path),four_NMS_cases=comparisons,
        images_read=False,model_inference=False,optimizer_steps=0,test_rows_selected=False,test_images_or_predictions_read=False,
        original_manifest_modified=False,ready_to_train=False)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(counts))
    print(json.dumps([dict(case=c['case'],head_count=len(c['source_head_boxes']),
        head_suppressor_iou=[round(h['overlap_with_suppressor']['iou'],5) for h in c['source_head_boxes']],
        head_face_coverage=[round(h['overlap_with_face']['fraction_second_covered'],5) for h in c['source_head_boxes']]) for c in comparisons]))
    print(sha(output))


if __name__=='__main__':main()
