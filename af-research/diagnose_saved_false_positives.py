#!/usr/bin/env python3
"""Saved validation prediction FP attribution; no inference, fitting or test.

Categories describe annotation geometry and score-ordered assignment, not proven
visual causes. Thresholds and categories are fixed before reading predictions.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

from verify_external_baselines import independent_metrics

BASE = Path(__file__).resolve().parent
CLASSES = ('person', 'person_face')
RUNS = ('baseline_original', 'baseline_small', 'epoch-06')
CATEGORIES = ('duplicate', 'other_class_overlap', 'localization_iou_01_to_05', 'background_low_overlap')
MANIFEST_SHA = 'af5459ddebd98947b154580cd176e7846abcceffe5539ac6f0001f6a485a3fc8'
SCORE_EDGES = (0., .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.)
HIGH_SCORE_EDGES = (.5, .6, .7, .8, .9, .95, .99, 1.)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def overlap(a, b):
    intersection = max(0., min(a[2], b[2])-max(a[0], b[0]))*max(0., min(a[3], b[3])-max(a[1], b[1]))
    area_a, area_b = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
    union = area_a+area_b-intersection
    return (intersection/union if union > 0 else 0., intersection/area_a if area_a > 0 else 0.,
            intersection/area_b if area_b > 0 else 0.)


def size_bin(frame, box):
    side = math.sqrt((box[2]-box[0])*(box[3]-box[1])*384*256/(frame['width']*frame['height']))
    return side, 'below16' if side < 16 else '16_to_below32' if side < 32 else '32_to_below96' if side < 96 else '96_or_larger'


def category(same_iou, other_iou):
    if same_iou >= .5:
        return 'duplicate'
    if other_iou >= .5:
        return 'other_class_overlap'
    if same_iou >= .1:
        return 'localization_iou_01_to_05'
    return 'background_low_overlap'


def stats(values):
    if not values:
        return None
    ordered = sorted(values)
    def q(fraction):
        i = fraction*(len(ordered)-1); low = math.floor(i); high = math.ceil(i)
        return ordered[low]+(ordered[high]-ordered[low])*(i-low)
    return dict(count=len(values), mean=sum(values)/len(values), min=ordered[0], p25=q(.25),
                median=q(.5), p75=q(.75), p90=q(.9), p95=q(.95), max=ordered[-1])


def bucket_rows(rows, edges):
    output = []
    for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
        selected = [r for r in rows if lo <= r['score'] and (r['score'] < hi or i == len(edges)-2 and r['score'] <= hi)]
        tp = sum(r['is_tp'] for r in selected)
        output.append(dict(score_lower_inclusive=lo, score_upper=hi, upper_inclusive=i == len(edges)-2,
            predictions=len(selected), tp=tp, fp=len(selected)-tp,
            empirical_detection_precision=tp/len(selected) if selected else None,
            mean_score=sum(r['score'] for r in selected)/len(selected) if selected else None,
            fp_categories={name: sum(not r['is_tp'] and r['fp_category'] == name for r in selected) for name in CATEGORIES}))
    return output


def match_and_diagnose(frames, predictions, cls):
    truths = {fid: [(i, b['bbox_xyxy']) for i, b in enumerate(f['boxes']) if b['class'] == cls]
              for fid, f in frames.items()}
    others = {fid: [(i, b) for i, b in enumerate(f['boxes']) if b['class'] != cls]
              for fid, f in frames.items()}
    ordered = sorted(((p['score'], fid, i, p) for fid, items in predictions.items()
                      for i, p in enumerate(items) if p['class'] == cls), key=lambda x: (-x[0], x[1], x[2]))
    used, rows = defaultdict(set), []
    for score, fid, index, prediction in ordered:
        box = prediction['bbox_xyxy']; gt = truths[fid]
        same = [overlap(box, truth) for _, truth in gt]
        best, match_index = max(((values[0], i) for i, values in enumerate(same) if i not in used[fid]), default=(0., -1))
        hit = best >= .5 and match_index >= 0
        same_max, same_index = max(((values[0], i) for i, values in enumerate(same)), default=(0., -1))
        other_max, other_index = max(((overlap(box, b['bbox_xyxy'])[0], i) for i, (_gi, b) in enumerate(others[fid])), default=(0., -1))
        if hit:
            used[fid].add(match_index)
        elif same_max >= .5:
            need(all(i in used[fid] for i, value in enumerate(same) if value[0] >= .5), 'Duplicate category has an available match')
        side, size = size_bin(frames[fid], box)
        row = dict(frame_id=fid, prediction_index=index, target_class=cls, score=score, bbox_xyxy=box,
            is_tp=hit, matched_iou=best if hit else None,
            matched_gt_index=gt[match_index][0] if hit else None,
            max_same_class_iou=same_max, nearest_same_gt_index=gt[same_index][0] if same_index >= 0 else None,
            max_other_class_iou=other_max,
            nearest_other_gt_index=others[fid][other_index][0] if other_index >= 0 else None,
            nearest_other_gt_class=others[fid][other_index][1]['class'] if other_index >= 0 else None,
            max_fraction_prediction_inside_same_gt=max((v[1] for v in same), default=0.),
            max_fraction_same_gt_covered=max((v[2] for v in same), default=0.),
            predicted_reference_sqrt_area=side, predicted_size_bin=size,
            fp_category=None if hit else category(same_max, other_max))
        rows.append(row)
    total_gt = sum(map(len, truths.values()))
    tp_running, precisions = 0, []
    for i, row in enumerate(rows, 1):
        tp_running += row['is_tp']; precisions.append(tp_running/i)
    best_precision, area = 0., 0.
    for row, precision in zip(reversed(rows), reversed(precisions)):
        best_precision = max(best_precision, precision)
        if row['is_tp']:
            area += best_precision
    chosen = [r for r in rows if r['score'] >= .5]
    tp = sum(r['is_tp'] for r in chosen); fp = len(chosen)-tp
    bins = bucket_rows(rows, SCORE_EDGES)
    n = len(rows)
    ece = sum(b['predictions']*abs(b['mean_score']-b['empirical_detection_precision']) for b in bins if b['predictions'])/n
    summary = dict(ground_truth_boxes=total_gt, all_saved_predictions=n, ap50=area/total_gt,
        all_saved_tp=tp_running, all_saved_fp=n-tp_running,
        score_ge_05=dict(predictions=len(chosen), tp=tp, fp=fp, fn=total_gt-tp,
            precision=tp/len(chosen) if chosen else None, recall=tp/total_gt,
            fp_categories={name: sum(r['fp_category'] == name for r in chosen) for name in CATEGORIES}),
        all_saved_score_by_outcome={name: stats([r['score'] for r in rows if r['is_tp'] == truth]) for name, truth in [('tp', True), ('fp', False)]},
        score_ge_05_by_outcome={name: stats([r['score'] for r in chosen if r['is_tp'] == truth]) for name, truth in [('tp', True), ('fp', False)]},
        reliability_bins_all_saved=bins, fixed_high_score_bins=bucket_rows(chosen, HIGH_SCORE_EDGES),
        conditional_detection_correctness_proxy=dict(ece_10_fixed_bins=ece,
            brier=sum((r['score']-int(r['is_tp']))**2 for r in rows)/n,
            scope='PostNMS/top30 saved boxes only. TP assignment depends on score order/other boxes; this is not raw-class probability calibration.'),
        fp_by_predicted_size_at_score05={name: dict(count=sum(not r['is_tp'] and r['predicted_size_bin'] == name for r in chosen),
            categories={cat: sum(r['fp_category'] == cat and r['predicted_size_bin'] == name for r in chosen) for cat in CATEGORIES})
            for name in ('below16', '16_to_below32', '32_to_below96', '96_or_larger')},
        low_overlap_background_subflags_at_score05=dict(
            prediction_majority_inside_a_same_class_gt=sum(r['fp_category'] == 'background_low_overlap' and r['max_fraction_prediction_inside_same_gt'] >= .5 for r in chosen),
            covers_majority_of_a_same_class_gt=sum(r['fp_category'] == 'background_low_overlap' and r['max_fraction_same_gt_covered'] >= .5 for r in chosen)))
    need(sum(summary['score_ge_05']['fp_categories'].values()) == fp, 'FP categories are not exhaustive/disjoint')
    return summary, rows


def geometry_pairs(old_rows, new_rows):
    old, new = defaultdict(list), defaultdict(list)
    for row in old_rows: old[row['frame_id']].append(row)
    for row in new_rows: new[row['frame_id']].append(row)
    paired = []
    for fid in sorted(old):
        choices = []
        for a in old[fid]:
            for b in new[fid]:
                value = overlap(a['bbox_xyxy'], b['bbox_xyxy'])[0]
                if value >= .9:
                    choices.append((-value, a['prediction_index'], b['prediction_index'], a, b))
        used_old, used_new = set(), set()
        for negative_iou, ai, bi, a, b in sorted(choices, key=lambda r: r[:3]):
            if ai in used_old or bi in used_new: continue
            used_old.add(ai); used_new.add(bi)
            paired.append(dict(frame_id=fid, old_prediction_index=ai, new_prediction_index=bi,
                box_iou=-negative_iou, old_score=a['score'], new_score=b['score'],
                old_is_tp=a['is_tp'], new_is_tp=b['is_tp'], old_fp_category=a['fp_category'], new_fp_category=b['fp_category']))
    both_fp = [r for r in paired if not r['old_is_tp'] and not r['new_is_tp']]
    return dict(pairing='Fixed IoU>=.9; descending pair-IoU, then old/new saved index; one-to-one within frame/class.',
        paired_count=len(paired), old_saved_count=len(old_rows), new_saved_count=len(new_rows),
        both_fp_pairs=len(both_fp), both_fp_score_change=stats([r['new_score']-r['old_score'] for r in both_fp]),
        both_fp_crossed_up_score05=sum(r['old_score'] < .5 <= r['new_score'] for r in both_fp),
        both_fp_crossed_down_score05=sum(r['new_score'] < .5 <= r['old_score'] for r in both_fp),
        limitation='Only nearly unchanged saved geometry is paired. Unpaired changed/new boxes prevent attribution of the full FP increase to calibration alone.',
        pairs=paired)


def self_check():
    need(category(.6, .9) == 'duplicate' and category(.4, .6) == 'other_class_overlap' and
         category(.1, .49) == 'localization_iou_01_to_05' and category(.0999, .49) == 'background_low_overlap',
         'Category precedence/boundary fixture failed')
    frame = dict(id='x', width=100, height=100, boxes=[dict(**{'class': 'person'}, bbox_xyxy=[0, 0, 20, 20]),
        dict(**{'class': 'person_face'}, bbox_xyxy=[40, 40, 50, 50])])
    boxes = [[0, 0, 20, 20], [0, 0, 20, 20], [40, 40, 50, 50], [15, 0, 35, 20], [60, 60, 70, 70]]
    pred = {'x': [dict(**{'class': 'person'}, score=s, bbox_xyxy=b) for s, b in zip([.9, .8, .7, .6, .5], boxes)]}
    summary, rows = match_and_diagnose({'x': frame}, pred, 'person')
    need(summary['score_ge_05']['tp'] == 1 and summary['score_ge_05']['fp'] == 4 and
         all(n == 1 for n in summary['score_ge_05']['fp_categories'].values()) and summary['ap50'] == 1,
         'Matching/AP/category fixture failed')
    need(sum(b['predictions'] for b in summary['fixed_high_score_bins']) == 5, 'Score bucket coverage failed')
    need(geometry_pairs(rows, rows)['paired_count'] == 5, 'One-to-one geometry-pair fixture failed')
    return dict(category_precedence=True, score_and_iou_boundaries=True, exact_matching_ap=True,
                score_bucket_partition=True, geometry_pairing=True)


def run(args):
    fixtures = self_check()
    need(not args.output.exists(), 'Preserve prior diagnosis; choose a new output directory')
    need(sha(args.dataset) == MANIFEST_SHA, 'Unexpected fixed dataset')
    data = json.loads(args.dataset.read_text())
    frames = {f['id']: f for f in data['frames'] if f['split'] == 'validation'}
    need(len(frames) == 80 and all(all(any(b['class'] == c for b in f['boxes']) for c in CLASSES) for f in frames.values()),
         'Expected positive-only validation80')
    training_plan = json.loads((args.run/'plan.json').read_text()); plan_hash = sha(args.run/'plan.json')
    need(set(training_plan['validation_ids']) == set(frames) and training_plan['manifest_sha256'] == MANIFEST_SHA,
         'Training plan validation identity differs')
    sources = {name: sha(BASE/name) for name in ('diagnose_saved_false_positives.py', 'verify_external_baselines.py')}
    paths = {name: dict(prediction=args.run/name/'predictions.json', metrics=args.run/name/'metrics.json') for name in RUNS}
    identities = {name: {kind: dict(file=p.relative_to(args.run).as_posix(), sha256=sha(p)) for kind, p in values.items()}
                  for name, values in paths.items()}
    args.output.mkdir(parents=True)
    snapshot = args.output/'source-snapshot'; snapshot.mkdir()
    for name in sources: (snapshot/name).write_bytes((BASE/name).read_bytes())
    plan = dict(status='Frozen before reading prediction scores', manifest_sha256=MANIFEST_SHA, training_plan_sha256=plan_hash,
        source_sha256=sources, saved_input_files=identities, runs=list(RUNS), fixed_score_threshold=.5, match_iou=.5,
        fp_categories_in_priority_order=[
            'duplicate: FP with any same-class GT IoU>=.5, all eligible same-class GT already matched earlier.',
            'other_class_overlap: remaining FP with another-class GT IoU>=.5; geometric overlap only, not proven semantic confusion.',
            'localization_iou_01_to_05: remaining FP with max same-class IoU in[.1,.5).',
            'background_low_overlap: all remaining FP. This is not proof of empty/unlabelled visual background.'],
        score_edges=list(SCORE_EDGES), fixed_high_score_edges=list(HIGH_SCORE_EDGES),
        predicted_size_bins='sqrt(predicted box area in384x256 reference): <16,16..32,32..96,>=96',
        same_geometry_pair_iou=.9, pair_purpose='Describe score changes for near-identical boxes; not threshold/model selection.',
        matching='All saved boxes, per class, score descending, frame ID, saved index; max IoU unmatched GT, larger GT index tie.',
        self_checks=fixtures, images_read=False, model_inference=False, optimizer_steps=0,
        test_images_or_predictions_read=False, thresholds_tuned=False)
    write(args.output/'frozen-plan.json', plan)
    summaries, all_rows, crosschecks = {}, {}, {}
    for name in RUNS:
        doc = json.loads(paths[name]['prediction'].read_text()); saved = json.loads(paths[name]['metrics'].read_text())
        need(doc['dataset_manifest_sha256'] == MANIFEST_SHA and doc['plan_sha256'] == plan_hash, 'Prediction identity differs')
        expected_anchor = training_plan['original_anchor_sha256'] if name == 'baseline_original' else training_plan['small_anchor_sha256']
        need(doc['anchor_sha256'] == expected_anchor, 'Unexpected anchor identity')
        pred = {r['id']: r['detections'] for r in doc['frames']}
        need(len(pred) == len(doc['frames']) == 80 and set(pred) == set(frames), 'Wrong prediction split/coverage')
        for fid, items in pred.items():
            for p in items:
                b = p['bbox_xyxy']; f = frames[fid]
                need(p['class'] in CLASSES and isinstance(p['score'], (int, float)) and math.isfinite(p['score']) and
                     0 <= p['score'] <= 1 and len(b) == 4 and all(math.isfinite(v) for v in b) and
                     -1e-6 <= b[0] < b[2] <= f['width']+1e-6 and -1e-6 <= b[1] < b[3] <= f['height']+1e-6, 'Invalid saved prediction')
        summaries[name], all_rows[name], crosschecks[name] = {}, {}, {}
        identities[name]['model_state_sha256'] = doc['model_state_sha256']; identities[name]['anchor_sha256'] = doc['anchor_sha256']
        for cls in CLASSES:
            summary, rows = match_and_diagnose(frames, pred, cls)
            reference = independent_metrics(frames, {fid: [p for p in items if p['class'] == cls] for fid, items in pred.items()}, cls)
            metric = saved['classes'][cls]
            need(abs(summary['ap50']-metric['ap_at_iou']) < 1e-12 and abs(summary['ap50']-reference['ap_at_iou']) < 1e-12,
                 'Independent AP mismatch')
            for computed, expected in [('tp', 'tp_at_score'), ('fp', 'fp_at_score'), ('fn', 'fn_at_score')]:
                need(summary['score_ge_05'][computed] == metric[expected] == reference[expected], 'TP/FP/FN mismatch')
            summaries[name][cls], all_rows[name][cls] = summary, rows
            crosschecks[name][cls] = dict(saved_metrics_and_independent_evaluator_agree=True,
                                         ap_absolute_error=abs(summary['ap50']-metric['ap_at_iou']))
            print(json.dumps(dict(run=name, target_class=cls, ap50=summary['ap50'], **summary['score_ge_05'])), flush=True)
    comparisons = {}
    for old_name, new_name in [('baseline_original', 'baseline_small'), ('baseline_small', 'epoch-06'), ('baseline_original', 'epoch-06')]:
        key = old_name+'__to__'+new_name; comparisons[key] = {}
        for cls in CLASSES:
            old, new = summaries[old_name][cls], summaries[new_name][cls]
            old5, new5 = old['score_ge_05'], new['score_ge_05']
            comparisons[key][cls] = dict(ap50_delta=new['ap50']-old['ap50'],
                tp_delta=new5['tp']-old5['tp'], fp_delta=new5['fp']-old5['fp'],
                fixed_score_precision_delta=new5['precision']-old5['precision'],
                fp_category_deltas={cat: new5['fp_categories'][cat]-old5['fp_categories'][cat] for cat in CATEGORIES},
                near_same_geometry=geometry_pairs(all_rows[old_name][cls], all_rows[new_name][cls]))
    need(sha(args.dataset) == MANIFEST_SHA and sha(args.run/'plan.json') == plan_hash, 'Bound dataset/plan changed')
    need(all(sha(BASE/n) == value for n, value in sources.items()), 'Diagnosis code changed')
    for name, values in paths.items():
        for kind, p in values.items(): need(sha(p) == identities[name][kind]['sha256'], 'Saved input changed')
    result = dict(status='complete', frozen_plan_sha256=sha(args.output/'frozen-plan.json'), source_sha256=sources,
        manifest_sha256=MANIFEST_SHA, training_plan_sha256=plan_hash, saved_input_files=identities,
        exact_metric_crosschecks=crosschecks, summaries=summaries, comparisons=comparisons, per_prediction=all_rows,
        interpretation_limits=[
            'A higher AP with lower precision at score.5 is compatible with more high-confidence bad boxes; it is not a global ranking collapse.',
            'Changed geometry, counts and ordering mean the full increase cannot be attributed to score calibration alone. AP is invariant only to strictly monotonic score changes on an otherwise fixed prediction set.',
            'ECE/Brier here are descriptive correctness proxies conditional on saved postNMS/top30 detections, not independent probability-calibration validation.',
            'Other-class overlap is a geometric flag. Person/face annotations can be nested; this is not proof of semantic class confusion.',
            'Low-overlap background may include partial objects, oversized boxes, missing/incorrect annotations or unlabelled classes; no images were inspected here.',
            'Validation80 contains person+face positives in every image; empty-scene false-positive performance remains unmeasured.',
            'Local two-class AP50 differs from official Open Images evaluation; retained occlusion/truncation and annotation-scope differences remain relevant.'],
        images_read=False, model_inference=False, optimizer_steps=0, test_images_or_predictions_read=False,
        thresholds_tuned=False, new_model_selected=False)
    write(args.output/'report.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=BASE/'private-data/recognition-v2/manifest.json')
    parser.add_argument('--run', type=Path, default=BASE/'artifacts/recognition-small-anchor-v2')
    parser.add_argument('--output', type=Path, default=BASE/'artifacts/recognition-fp-diagnosis/saved-predictions')
    run(parser.parse_args())
