#!/usr/bin/env python3
"""Read-only train32 FP/stage and exact historical RNG sampling diagnosis.

Scores are epoch06 retrospective scores, NEVER historical training-time scores.
Original saved NPZs supply geometry; synthetic logits expose actual sampled
indices in the frozen loss. No photograph forward, optimizer or core mutation.
"""
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from extract import require, write_json
from pretraining_data import sha256
from small_face_path_diagnosis import greedy_matches, reference_gts
from train_recognition import CLASSES, box_iou, state_digest

BASE = Path(__file__).resolve().parent
MANIFEST_SHA = 'af5459ddebd98947b154580cd176e7846abcceffe5539ac6f0001f6a485a3fc8'


def iou_max(boxes, gt):
    return box_iou(boxes, np.asarray(gt).reshape(-1, 4)).max(1) if len(gt) else np.zeros(len(boxes))


def owner_digest(a):
    h = hashlib.sha256()
    for key, dtype in [('matched_gt', '<i8'), ('gt_classes', '<i8'), ('gt_boxes', '<f4')]:
        h.update(np.asarray(a[key], dtype=dtype).tobytes())
    return h.hexdigest()


def trace_nms(boxes, scores, pre):
    """Independent complete trace; retain the state when the real cap stops."""
    boxes = np.asarray(boxes, np.float64)
    order = pre[np.argsort(-scores[pre], kind='stable')]
    keep = []; suppressed_by = np.full(len(boxes), -1, np.int64)
    suppression_rank = np.full(len(boxes), -1, np.int64)
    pending_at_30 = np.zeros(len(boxes), bool)
    while len(order):
        selected = int(order[0]); keep.append(selected); rest = order[1:]
        overlap = box_iou(boxes[selected:selected+1], boxes[rest])[0]
        removed = rest[overlap > .3]
        suppressed_by[removed] = selected; suppression_rank[removed] = len(keep)
        order = rest[overlap <= .3]
        if len(keep) == 30:
            pending_at_30[order] = True
    return dict(keep=np.asarray(keep, np.int64), suppressed_by=suppressed_by,
                suppression_rank=suppression_rank, pending_at_30=pending_at_30)


def replay_loss(frozen, ref, gt, assigned, record, next_record):
    rng = np.random.default_rng()
    rng.bit_generator.state = copy.deepcopy(record['rng_before_loss'])
    logits = {c: torch.zeros(len(ref), 2, requires_grad=True) for c in CLASSES}
    values = np.zeros_like(ref)
    owned = np.flatnonzero(assigned['matched_gt'] >= 0)
    values[owned] = frozen.encode_boxes(ref[owned], assigned['gt_boxes'][assigned['matched_gt'][owned]]) + .25
    delta = torch.tensor(values, requires_grad=True)
    loss, info = frozen.detection_loss(logits, delta, ref, gt, rng)
    loss.backward()
    for key in ('target_gt_count', 'unmatched_gt_count', 'bbox_positive_anchors',
                'bbox_required_classification_positives', 'bbox_supervised_gt'):
        require(info[key] == record[key], 'Logged sampling count differs: '+key)
    selections = {}
    for c in CLASSES:
        gradient = logits[c].grad[:, 1].numpy()
        selections[c] = dict(positive=np.flatnonzero(gradient < 0), negative=np.flatnonzero(gradient > 0))
        for key in ('positive_anchors', 'forced_positive_anchors', 'supervised_gt'):
            require(info[c+'_'+key] == record[c+'_'+key], 'Logged class sampling differs')
    bbox = np.flatnonzero(delta.grad.abs().sum(1).numpy() > 0)
    union_positive = set(np.concatenate([selections[c]['positive'] for c in CLASSES]).tolist())
    require(union_positive.issubset(set(bbox.tolist())), 'Classification positive lost bbox gradient')
    require(set(assigned['matched_gt'][bbox]) == set(range(len(assigned['gt_boxes']))), 'GT lost bbox supervision')
    if next_record is not None and next_record['epoch'] == record['epoch']:
        require(rng.bit_generator.state == next_record['rng_before_loss'], 'Exact replay does not reach next logged RNG state')
    return selections, bbox


def bin_iou(value):
    return 'below_03' if value < .3 else '03_to_below_05' if value < .5 else 'ge_05'


def jsonable(value):
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    return value


def run(args):
    require(not args.output.exists(), 'Preserve prior diagnosis')
    require(sha256(args.dataset) == MANIFEST_SHA, 'Unexpected manifest')
    dataset = json.loads(args.dataset.read_text())
    frames = sorted((f for f in dataset['frames'] if f['split'] == 'train'), key=lambda f: f['id'])[:32]
    del dataset
    frame_map = {f['id']: f for f in frames}
    stage = json.loads((args.stage/'report.json').read_text())
    require(set(frame_map) == {r['frame_id'] for r in stage['baseline']}, 'Train32 mismatch')
    train_plan = json.loads((args.run/'plan.json').read_text())
    snapshot = args.run/'source-snapshot/train_recognition.py'
    require(sha256(snapshot) == train_plan['source_hashes']['train_recognition.py'], 'Frozen loss source mismatch')
    spec = importlib.util.spec_from_file_location('frozen_s5_v2_loss_sampling_diagnosis', snapshot)
    frozen = importlib.util.module_from_spec(spec); spec.loader.exec_module(frozen)
    logs = [json.loads(line) for line in (args.run/'training.jsonl').read_text().splitlines()]
    require(len(logs) == 1920, 'Expected six complete train320 passes')
    selected_logs = defaultdict(list)
    for i, record in enumerate(logs):
        if record['frame_id'] in frame_map:
            selected_logs[record['frame_id']].append((record, logs[i+1] if i+1 < len(logs) else None))
    require(all([r['epoch'] for r, _ in selected_logs[f]] == list(range(1, 7)) for f in frame_map), 'Missing historical train visit')
    ref = np.load(args.run/'anchors-small.npy', allow_pickle=False)
    require(hashlib.sha256(ref.astype('<f4').tobytes()).hexdigest() == train_plan['small_anchor_sha256'], 'Wrong training anchors')
    checkpoint_path = args.run/'epoch-06/checkpoint.pt'
    checkpoint_sha = sha256(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    checkpoint_state = state_digest(checkpoint['model_state'])
    require(checkpoint_state == stage['checkpoint_state_sha256_before'] and checkpoint['completed_epoch'] == 6,
            'Raw scores are not from this final float checkpoint')
    torch.set_num_threads(1)
    sources = {name: sha256(BASE/name) for name in (Path(__file__).name, 'small_face_path_diagnosis.py',
               'train_recognition.py', 'proposal_hypothesis.py', 'pretraining_data.py')}
    args.output.mkdir(parents=True); (args.output/'nms-traces').mkdir()
    source_snap = args.output/'source-snapshot'; source_snap.mkdir()
    for name in sources: (source_snap/name).write_bytes((BASE/name).read_bytes())
    plan = dict(status='frozen_before_analysis', split='train', frame_ids=sorted(frame_map),
        source_hashes=sources, frozen_loss_sha256=sha256(snapshot), manifest_sha256=MANIFEST_SHA,
        stage_report_sha256=sha256(args.stage/'report.json'), checkpoint_file_sha256=checkpoint_sha,
        checkpoint_state_sha256=checkpoint_state, training_log_sha256=sha256(args.run/'training.jsonl'),
        method='Replay actual frozen loss with logged rng_before_loss and synthetic outputs; evaluate sampled indices retrospectively with epoch06 NPZ scores.',
        hypothetical_negative_guard='static label0 AND decoded max same-class GT IoU < .3; >=.3 ignored in both hard and random pools.',
        score_threshold=.5, iou_match=.5, nms_iou=.3, pretops=3000, final_cap=30,
        image_reads=0, network_forwards=0, optimizer_steps=0, validation_inference=False, test_inference=False)
    write_json(args.output/'frozen-plan.json', plan)
    totals = {c: Counter() for c in CLASSES}; sampling = {c: Counter() for c in CLASSES}
    per_frame = []; small_faces = []; gt_sampling = []; fp_rows = []; replay_rows = []
    for row in stage['baseline']:
        fid = row['frame_id']; frame = frame_map[fid]
        path = args.stage/row['prediction_file']
        require(sha256(path) == row['saved_prediction_sha256'], 'Saved raw predictions changed')
        with np.load(path, allow_pickle=False) as z: raw = {k: z[k].copy() for k in z.files}
        boxes = raw['decoded_boxes']; gt = frozen.ground_truth(frame); assigned = frozen.assign_joint_targets(ref, gt)
        require(all(owner_digest(assigned) == record['target_owner_sha256'] for record, _ in selected_logs[fid]),
                'Actual training GT owners do not reproduce')
        original_gts = reference_gts(frame); traces = {}; class_info = {}; trace_arrays = {}
        for ci, c in enumerate(CLASSES):
            scores = raw['scores_'+c]; pre = raw['pre_nms_indices_'+c]; final = raw['final_indices_'+c]
            expected_pre = raw['valid_indices'][np.argsort(-scores[raw['valid_indices']], kind='stable')[:3000]]
            require(np.array_equal(pre, expected_pre), 'Pre-NMS order differs')
            trace = trace_nms(boxes, scores, pre); traces[c] = trace
            require(np.array_equal(trace['keep'][:30], final), 'Independent NMS differs from saved final30')
            trace_arrays.update({c+'_'+k: v for k, v in trace.items()})
            rows = [g for g in original_gts if g['target_class'] == c]
            overlaps = iou_max(boxes, [g['box'] for g in rows])
            matches = greedy_matches(boxes, scores, final, rows)
            tp = {v['anchor_index'] for v in matches.values()}
            labels = assigned['labels'][c]
            small = [g for g in rows if np.sqrt(np.prod(g['box'][2:]-g['box'][:2])) < 32]
            small_good = iou_max(boxes, [g['box'] for g in small]) >= .5
            negative = labels == 0; good_negative = negative & (overlaps >= .5)
            guarded = negative & (overlaps < .3)
            for tag, mask in [('negative_decoded_good', good_negative), ('negative_decoded_good_highscore', good_negative & (scores >= .5)),
                              ('small_good_negative', negative & small_good), ('small_good_negative_highscore', negative & small_good & (scores >= .5)),
                              ('guarded_negative_pool', guarded), ('guard_ignored_static_negatives', negative & ~guarded)]:
                totals[c][tag] += int(np.count_nonzero(mask))
            frame_fp = []
            for index in final[scores[final] >= .5]:
                index = int(index); hit = index in tp
                totals[c]['TP' if hit else 'FP'] += 1
                totals[c][('TP' if hit else 'FP')+'_label_'+str(int(labels[index]))] += 1
                if not hit:
                    totals[c]['FP_decoded_'+bin_iou(float(overlaps[index]))] += 1
                    if labels[index] == 0: totals[c]['FP_label0_decoded_'+bin_iou(float(overlaps[index]))] += 1
                    item = dict(frame_id=fid, target_class=c, anchor_index=index, score=float(scores[index]),
                        bbox_xyxy=boxes[index].tolist(), static_training_label=int(labels[index]),
                        matched_owner=int(assigned['matched_gt'][index]), decoded_max_same_class_iou=float(overlaps[index]),
                        false_positive_kind='duplicate_matching' if overlaps[index] >= .5 else 'localization_or_background',
                        eligible_for_guarded_hard_negative=bool(guarded[index]))
                    fp_rows.append(item); frame_fp.append(item)
            class_info[c] = dict(scores=scores, labels=labels, overlaps=overlaps, small_good=small_good,
                                  guarded=guarded, negative=negative, good_negative=good_negative, fp=frame_fp)
            for target in small:
                overlap = box_iou(boxes, target['box'][None])[..., 0]
                valid = raw['valid_indices']; geometric = valid[overlap[valid] >= .5]
                confident = geometric[scores[geometric] >= .5]
                pre_rank = {int(a): i+1 for i, a in enumerate(pre)}
                keep_rank = {int(a): i+1 for i, a in enumerate(trace['keep'])}
                details = []
                for index in confident[np.argsort(-scores[confident], kind='stable')]:
                    index = int(index); parent = int(trace['suppressed_by'][index]); rank = int(trace['suppression_rank'][index])
                    cause = ('pretop3000_excluded' if index not in pre_rank else
                             'retained_final30' if keep_rank.get(index, 100000) <= 30 else
                             'nms_suppressed_before_final30' if 0 < rank <= 30 else 'unprocessed_at_final30_cap')
                    suppressor = None
                    if parent >= 0:
                        suppressor = dict(anchor_index=parent, bbox_xyxy=boxes[parent].tolist(), score=float(scores[parent]),
                            selected_rank=rank, iou_with_target=float(overlap[parent]),
                            max_iou_with_any_same_class_gt=float(overlaps[parent]),
                            iou_with_removed_box=float(box_iou(boxes[parent:parent+1], boxes[index:index+1])[0, 0]),
                            static_training_label=int(labels[parent]), owner=int(assigned['matched_gt'][parent]))
                    details.append(dict(anchor_index=index, bbox_xyxy=boxes[index].tolist(), score=float(scores[index]),
                        iou_with_target=float(overlap[index]), static_training_label=int(labels[index]),
                        owner=int(assigned['matched_gt'][index]), pretop_rank=pre_rank.get(index),
                        unlimited_nms_keep_rank=keep_rank.get(index), actual_discard_reason=cause, suppressor=suppressor))
                actual_causes = Counter(d['actual_discard_reason'] for d in details)
                if not len(geometric): category = 'decoded_geometry_insufficient'
                elif not len(confident): category = 'classification_confidence_insufficient'
                elif actual_causes['retained_final30']: category = 'retained_candidate_check_one_to_one'
                elif actual_causes['unprocessed_at_final30_cap']: category = 'final30_cap_has_unprocessed_qualifying_candidate'
                elif actual_causes['nms_suppressed_before_final30']: category = 'all_pretop_qualifying_candidates_suppressed_by_nms'
                else: category = 'all_qualifying_candidates_excluded_by_pretop3000'
                small_faces.append(dict(frame_id=fid, target_class=c, gt_index=target['gt_index'], target_box=target['box'].tolist(),
                    refined_stage=category, all_decoded_qualifying=len(geometric), confident_qualifying=len(confident),
                    best_score_among_geometrically_good=float(scores[geometric].max()) if len(geometric) else None,
                    geometrically_good_static_negative_count=int(np.count_nonzero(labels[geometric] == 0)),
                    confident_candidates=details, unlimited_nms_would_retain_qualifying=bool(set(geometric).intersection(trace['keep']))))
        trace_path = args.output/'nms-traces'/f'{len(per_frame):02d}.npz'
        np.savez_compressed(trace_path, **trace_arrays)
        frequency = {c: np.zeros(len(ref), np.int64) for c in CLASSES}
        per_gt_pos = defaultdict(list); per_gt_bbox = defaultdict(list)
        for record, next_record in selected_logs[fid]:
            selections, bbox = replay_loss(frozen, ref, gt, assigned, record, next_record)
            visit = dict(frame_id=fid, epoch=record['epoch'], replay_matches_logged_counts=True,
                         rng_reaches_next_logged_state=next_record is not None and next_record['epoch'] == record['epoch'], classes={})
            for ci, c in enumerate(CLASSES):
                pos, neg = selections[c]['positive'], selections[c]['negative']; info = class_info[c]; score = info['scores']
                frequency[c][neg] += 1; denominator = len(pos)+len(neg)
                for gi in np.flatnonzero(assigned['gt_classes'] == ci):
                    per_gt_pos[int(gi)].append(int(np.count_nonzero(assigned['matched_gt'][pos] == gi)))
                    per_gt_bbox[int(gi)].append(int(np.count_nonzero(assigned['matched_gt'][bbox] == gi)))
                high_negative = info['negative'] & (score >= .5)
                selected_high = int(np.count_nonzero(score[neg] >= .5))
                # Counterfactual mining is diagnostic only; it does not run a loss or optimizer.
                static_neg = np.flatnonzero(info['negative'])
                unguarded_hard = static_neg[np.argsort(-score[static_neg], kind='stable')[:min(64, len(neg))]]
                eligible = np.flatnonzero(info['guarded'])
                guarded_hard = eligible[np.argsort(-score[eligible], kind='stable')[:min(64, len(neg))]]
                visit['classes'][c] = dict(positive_indices=pos.tolist(), negative_indices=neg.tolist(),
                    positive_count=len(pos), negative_count=len(neg),
                    selected_negative_score_ge05=selected_high,
                    selected_negative_decoded_same_class_iou_ge05=int(np.count_nonzero(info['good_negative'][neg])),
                    selected_negative_decoded_same_class_iou_ge03=int(np.count_nonzero(info['overlaps'][neg] >= .3)),
                    selected_small_good_negative=int(np.count_nonzero(info['small_good'][neg])),
                    final_score_negative_fg_logit_push_down=float(score[neg].sum()/denominator),
                    final_score_positive_fg_logit_push_up=float((1-score[pos]).sum()/denominator),
                    unguarded_top64_correct_decoded_negative_count=int(np.count_nonzero(info['good_negative'][unguarded_hard])),
                    guarded_top64_correct_decoded_negative_count=int(np.count_nonzero(info['good_negative'][guarded_hard])),
                    guarded_top64_score_ge05=int(np.count_nonzero(score[guarded_hard] >= .5)),
                    final_score_label0_high_candidates=int(np.count_nonzero(high_negative)))
                sampling[c]['negative_draws'] += len(neg)
                sampling[c]['selected_highscore_negatives'] += selected_high
                sampling[c]['final_score_highnegative_opportunities'] += int(np.count_nonzero(high_negative))
                sampling[c]['selected_correct_decoded_negatives'] += int(np.count_nonzero(info['good_negative'][neg]))
                sampling[c]['selected_guard_would_ignore'] += int(np.count_nonzero(info['overlaps'][neg] >= .3))
                sampling[c]['selected_small_good_negatives'] += int(np.count_nonzero(info['small_good'][neg]))
                sampling[c]['unguarded_top64_correct_decoded_negatives'] += int(np.count_nonzero(info['good_negative'][unguarded_hard]))
                sampling[c]['guarded_top64_correct_decoded_negatives'] += int(np.count_nonzero(info['good_negative'][guarded_hard]))
            visit['bbox_indices'] = bbox.tolist(); replay_rows.append(visit)
        for c, info in class_info.items():
            for item in info['fp']:
                item['actual_negative_selections_across_six_visits'] = int(frequency[c][item['anchor_index']])
                if item['static_training_label'] == 0:
                    sampling[c]['final_label0_fp'] += 1
                    sampling[c]['final_label0_fp_never_sampled_negative'] += item['actual_negative_selections_across_six_visits'] == 0
                    if item['eligible_for_guarded_hard_negative']:
                        sampling[c]['guard_eligible_final_fp'] += 1
                        sampling[c]['guard_eligible_final_fp_never_sampled_negative'] += item['actual_negative_selections_across_six_visits'] == 0
        for gi, counts in per_gt_pos.items():
            box = assigned['gt_boxes'][gi]
            gt_sampling.append(dict(frame_id=fid, union_gt_index=gi, target_class=CLASSES[int(assigned['gt_classes'][gi])],
                bbox=box.tolist(), sqrt_area=float(np.sqrt(np.prod(box[2:]-box[:2]))),
                class_positive_counts_by_epoch=counts, bbox_counts_by_epoch=per_gt_bbox[gi],
                available_positive_anchors=int(np.count_nonzero(assigned['matched_gt'] == gi)),
                forced_anchor=int(assigned['forced_anchor_by_gt'][gi]), forced_iou=float(assigned['forced_match_ious'][gi])))
        per_frame.append(dict(frame_id=fid, raw_npz_sha256=sha256(path), raw_npz_file=row['prediction_file'],
            nms_trace_file=trace_path.relative_to(args.output).as_posix(), nms_trace_sha256=sha256(trace_path),
            target_owner_sha256=owner_digest(assigned), replays=6))
    require(state_digest(checkpoint['model_state']) == checkpoint_state and sha256(checkpoint_path) == checkpoint_sha,
            'Checkpoint file/state changed')
    require(all(sha256(BASE/n) == h for n, h in sources.items()), 'Source changed during diagnosis')
    require(len(replay_rows) == 192 and all(min(r['class_positive_counts_by_epoch']) >= 1 and
            min(r['bbox_counts_by_epoch']) >= 1 for r in gt_sampling), 'Sampling dropped a GT')
    face_small = [r for r in small_faces if r['target_class'] == 'person_face']
    require(len(face_small) == 10, 'Small face count changed')
    positive_groups = {}
    for c in CLASSES:
        for size in ('small', 'other'):
            rows = [r for r in gt_sampling if r['target_class'] == c and (r['sqrt_area'] < 32) == (size == 'small')]
            counts = [x for r in rows for x in r['class_positive_counts_by_epoch']]
            positive_groups[c+'/'+size] = dict(gt=len(rows), draws=len(counts), minimum=min(counts) if counts else None,
                median=float(np.median(counts)) if counts else None, maximum=max(counts) if counts else None,
                visits_with_one_positive=sum(x == 1 for x in counts))
    report = dict(status='complete', frozen_plan_sha256=sha256(args.output/'frozen-plan.json'),
        source_hashes=sources, checkpoint_file_sha256_before=checkpoint_sha, checkpoint_file_sha256_after=sha256(checkpoint_path),
        checkpoint_state_sha256=checkpoint_state, final_highscore_summary=totals, historical_sampling_summary=sampling,
        per_gt_positive_sampling_summary=positive_groups,
        small_face_refined_stage_counts=dict(Counter(r['refined_stage'] for r in face_small)),
        small_face_details=face_small, final_highscore_false_positives=fp_rows, gt_sampling=gt_sampling,
        exact_sampling_replays=replay_rows, frames=per_frame,
        conclusions=['Separate static anchor labels from current decoded-box quality; current label0 can decode to a correct same-class GT.',
            'Unguarded highest-score negative mining can increase incorrect downward pressure on correct decoded boxes.',
            'A decoded same-class IoU >= .3 ignore guard must apply to both hard and random negative pools, without changing the shared bbox owner.',
            'Per-GT forced supervision is retained, but counts are not balanced; one retained positive is not equal influence.',
            'The stage trace distinguishes actual pre-cap NMS suppressions from candidates never processed after final30.'],
        limitations=['Epoch06 scores applied to historical sampled indices are retrospective, not scores available at those training steps.',
            'Foreground-logit gradient direction is known analytically; aggregate parameter updates or accuracy improvement are not inferred.',
            'Positive-IoU and high score are not proof of complete dataset annotation; conclusions refer to supplied GT.',
            'No optimizer/new model inference, no threshold search, and no validation/test access.'],
        scope=dict(train_frames=32, train_visits_replayed=192, synthetic_loss_backward_calls=192,
                   image_reads=0, network_forwards=0, optimizer_steps=0, validation_inference=False, test_inference=False),
        core_modified=False, ready_for_new_training=False)
    write_json(args.output/'report.json', jsonable(report))
    print(json.dumps(jsonable(dict(final_highscore_summary=totals, historical_sampling_summary=sampling,
        small_face_refined_stage_counts=report['small_face_refined_stage_counts'],
        positive_sampling=positive_groups, output=str(args.output/'report.json'), sha256=sha256(args.output/'report.json'))), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=BASE/'private-data/recognition-v2/manifest.json')
    parser.add_argument('--run', type=Path, default=BASE/'artifacts/recognition-small-anchor-v2')
    parser.add_argument('--stage', type=Path, default=BASE/'artifacts/recognition-diagnosis-v3/small-face-after-v2-train32')
    parser.add_argument('--output', type=Path, default=BASE/'artifacts/recognition-fp-diagnosis/training-sampling')
    run(parser.parse_args())
