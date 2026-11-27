#!/usr/bin/env python3
"""Describe saved face-FP geometry without changing evaluation or reading photos."""
from collections import Counter
import hashlib
import json
from pathlib import Path

BASE=Path(__file__).resolve().parent


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(rows):
    counts=Counter()
    examples={}
    for row in rows:
        covered=row['max_fraction_same_gt_covered']
        inside=row['max_fraction_prediction_inside_same_gt']
        predicates={
            'covers_at_least_90pct_of_some_face_GT':covered>=.9,
            'covers_at_least_90pct_GT_and_less_than_half_prediction_inside_any_face_GT':covered>=.9 and inside<.5,
            'at_least_90pct_prediction_inside_some_face_GT':inside>=.9,
            'covers_at_least_half_of_some_face_GT':covered>=.5,
            'at_least_half_prediction_inside_some_face_GT':inside>=.5,
            'neither_half_GT_coverage_nor_half_prediction_inside_GT':covered<.5 and inside<.5,
            'any_same_class_iou_at_least_0_5_despite_FP_duplicate':row['max_same_class_iou']>=.5,
        }
        for name,yes in predicates.items():
            counts[name]+=int(yes)
            if yes:
                examples.setdefault(name,[])
                if len(examples[name])<5:
                    examples[name].append({k:row[k] for k in (
                        'frame_id','prediction_index','score','bbox_xyxy','max_same_class_iou',
                        'max_fraction_same_gt_covered','max_fraction_prediction_inside_same_gt','fp_category')})
    categories=Counter(r['fp_category'] for r in rows)
    return dict(false_positive_count=len(rows),counts=dict(counts),original_FP_categories=dict(categories),
        fractions={name:count/len(rows) if rows else None for name,count in counts.items()},
        examples_saved_order_first_five=examples)


def main():
    report_path=BASE/'artifacts/recognition-fp-diagnosis/saved-predictions/report.json'
    selection_path=BASE/'artifacts/recognition-fp-diagnosis/nms-evidence/selection.json'
    output=BASE/'artifacts/recognition-fp-diagnosis/target-conflicts/box-convention-caveat.json'
    assert not output.exists(),'Preserve previous evidence'
    data=json.loads(report_path.read_text())
    selection=json.loads(selection_path.read_text())
    results={}
    for run in ('baseline_original','baseline_small','epoch-06'):
        all_rows=data['per_prediction'][run]['person_face']
        fp=[r for r in all_rows if not r['is_tp']]
        high=[r for r in fp if r['score']>=.5]
        assert len(high)==data['summaries'][run]['person_face']['score_ge_05']['fp']
        results[run]=dict(score_at_least_0_5=summarize(high),all_saved_scores=summarize(fp),
            original_AP50=data['summaries'][run]['person_face']['ap50'],
            original_TP_at_score_0_5=data['summaries'][run]['person_face']['score_ge_05']['tp'])
    cases=[dict(case=c['case'],frame_id=c['frame_id'],split=c['split'],
                gt_attributes=c['gt_attributes'],
                suppressor_iou=c['displayed_candidate']['suppressor']['iou_with_target']) for c in selection['cases']]
    assert len(cases)==4 and all(c['gt_attributes']['IsOccluded']==1 for c in cases)
    result=dict(status='complete_descriptive_geometry_only',script_sha256=digest(Path(__file__)),
        saved_report_sha256=digest(report_path),saved_report_manifest_sha256=data['manifest_sha256'],
        selection_sha256=digest(selection_path),
        scope='Existing validation80 saved face predictions, original/transfer0/epoch6; four NMS illustrative cases are a separate train32 subset.',
        conditions={'baseline_original':'original K4 weights and original anchors',
                    'baseline_small':'analytically transferred original K4 heads and small anchors, epoch0',
                    'epoch-06':'v2 epoch6 heads and the same small anchors'},
        descriptive_thresholds={'GT_fraction':.9,'prediction_fraction':.9,'majority_fraction':.5},
        definitions=[
            'GT coverage = intersection / GT area; prediction inside GT = intersection / prediction area. These are existing source fields.',
            'The two stored maxima may come from different GTs. Do not divide them to infer a specific pair area ratio or assume a shared maximizing GT.',
            'If some GT is at least90pct covered AND less than half the prediction lies in every GT, prediction area is strictly greater than1.8 times that covered GT area. This is geometric oversizing relative to annotation, not proof of a wrong camera head box.',
            'Counts are overlapping descriptive predicates, not a replacement mutually-exclusive FP taxonomy. No AP, TP/FP, IoU.5 or score.5 rule is changed.',
            'These descriptive thresholds were requested after visual inspection; this is post-hoc diagnosis, not a preregistered accuracy acceptance test.'
        ],results=results,train32_NMS_illustrations=cases,
        original_camera_face_versus_head_box_definition='unknown; no real-camera output or SDK establishes the original annotation/output convention',
        interpretation=[
            'Four selected NMS GTs are marked IsOccluded=1. The review raised a full-head versus visible-face box-convention hypothesis; this script neither visually validates that hypothesis nor infers it from coverage alone.',
            'A high-confidence geometric FP can cover an annotated face while failing IoU.5 because it extends beyond that annotation. It remains FP under the frozen dataset metric.',
            'A low-overlap box is not thereby proven visual background. Missing annotations and original camera conventions are unresolved.',
            'The four train32 illustrations cannot establish how many validation80 cases are true head boxes; no full visual adjudication was performed.',
            'Do not relabel these cases as TP, retune thresholds, or use this descriptive caveat as authorization to resume optimization.'
        ],images_read=False,model_inference=False,optimizer_steps=0,test_images_or_predictions_read=False,
        predictions_modified=False,matching_rules_changed=False,ready_to_train=False)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({run:v['score_at_least_0_5'] for run,v in results.items()}))
    print(digest(output))


if __name__=='__main__':main()
