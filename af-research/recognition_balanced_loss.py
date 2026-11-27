"""Opt-in training-loss repair candidate; never changes the S5 graph or decoder.

No training entry point imports this module yet. The frozen v1/v2 loss remains
available for exact reproduction. Passing unit/gradient checks is not detection
acceptance and must not automatically authorize an optimizer run.
"""
import numpy as np
import torch
from torch.nn import functional as F

from extract import require
from train_recognition import (CLASSES, assign_joint_targets, box_iou, decode_boxes,
                               encode_boxes)

POSITIVE_BUDGET = 32
CLASS_BUDGET = 128
BBOX_BUDGET = 128
HARD_NEGATIVE_BUDGET = 64
NEGATIVE_DECODED_IOU_LIMIT = .3


def balanced_sample(candidates, required, owners, limit, rng):
    """Keep coverage, then give the least represented available GT the next slot."""
    candidates = np.unique(np.asarray(candidates, np.int64))
    required = np.unique(np.asarray(required, np.int64))
    require(np.isin(required, candidates).all(), 'Coverage anchor outside candidates')
    require(len(required) <= limit, 'Coverage exceeds unchanged sample budget')
    require(np.all(owners[candidates] >= 0), 'Positive without GT owner')
    selected = required.tolist()
    groups = np.unique(owners[candidates])
    pools = {int(g): rng.permutation(candidates[(owners[candidates] == g) &
                                  ~np.isin(candidates, required)]).tolist() for g in groups}
    counts = {int(g): int(np.sum(owners[required] == g)) for g in groups}
    # Fixed RNG, no dependence on GT class order or current predicted score.
    tie_order = rng.permutation(groups).tolist()
    while len(selected) < limit:
        active = [int(g) for g in tie_order if pools[int(g)]]
        if not active:
            break
        minimum = min(counts[g] for g in active)
        g = next(g for g in active if counts[g] == minimum)
        selected.append(pools[g].pop())
        counts[g] += 1
    return np.asarray(selected, np.int64)


def equal_gt_weights(indices, owners):
    """Each represented GT gets equal mass; total mass remains sample count."""
    if not len(indices):
        return np.zeros(0, np.float32)
    unique, inverse, count = np.unique(owners[indices], return_inverse=True, return_counts=True)
    require(np.all(unique >= 0), 'Weighted positive without GT owner')
    return (len(indices)/(len(unique)*count[inverse])).astype(np.float32)


def select_samples(logits, delta, ref, gt, rng):
    assigned = assign_joint_targets(ref, gt)
    require(not len(assigned['unmatched_gt_indices']), 'Unmatched shared GT owner')
    require(delta.shape == (len(ref), 4) and bool(torch.isfinite(delta).all()), 'Invalid bbox deltas')
    decoded = decode_boxes(ref, delta.detach().cpu().numpy())
    owners = assigned['matched_gt']
    trace = {'assigned': assigned, 'classes': {}}
    all_positives = []
    for class_index, cls in enumerate(CLASSES):
        require(logits[cls].shape == (len(ref), 2) and bool(torch.isfinite(logits[cls]).all()),
                'Invalid binary classification logits')
        labels = assigned['labels'][cls].copy()
        class_gt = np.asarray(gt[cls], np.float32).reshape(-1, 4)
        decoded_overlap = (box_iou(decoded, class_gt).max(1) if len(class_gt)
                           else np.zeros(len(ref), np.float32))
        # Anchor-space negatives can already decode to a correct face/person.
        # Ignore them in BOTH mining and random sampling; do not push them down.
        guarded = np.flatnonzero((labels == 0) & (decoded_overlap >= NEGATIVE_DECODED_IOU_LIMIT))
        labels[guarded] = -1
        required = assigned['forced_anchor_by_gt'][assigned['gt_classes'] == class_index]
        positive = balanced_sample(np.flatnonzero(labels == 1), required, owners,
                                   POSITIVE_BUDGET, rng)
        negative = np.flatnonzero(labels == 0)
        budget = min(CLASS_BUDGET-len(positive), len(negative))
        negative_ce = F.cross_entropy(logits[cls][negative],
            torch.zeros(len(negative), dtype=torch.long, device=logits[cls].device),
            reduction='none').detach().cpu().numpy()
        ranked = negative[np.argsort(-negative_ce, kind='stable')]
        hard = ranked[:min(HARD_NEGATIVE_BUDGET, budget)]
        remaining = negative[~np.isin(negative, hard)]
        random_count = budget-len(hard)
        random = (rng.choice(remaining, random_count, replace=False) if len(remaining) > random_count
                  else remaining)
        negative = np.concatenate((hard, random))
        all_positives.extend(positive.tolist())
        trace['classes'][cls] = dict(labels=labels, positive=positive, negative=negative,
            hard_negative=hard, random_negative=random, guarded_negative=guarded,
            positive_weights=equal_gt_weights(positive, owners), decoded_max_iou=decoded_overlap)
    bbox = balanced_sample(np.flatnonzero(owners >= 0), all_positives, owners, BBOX_BUDGET, rng)
    trace['bbox_positive'] = bbox
    trace['bbox_weights'] = equal_gt_weights(bbox, owners)
    require(np.isin(all_positives, bbox).all(), 'Classification positive lost bbox supervision')
    return trace


def balanced_detection_loss(logits, delta, ref, gt, rng, *, return_trace=False):
    trace = select_samples(logits, delta, ref, gt, rng)
    assigned = trace['assigned']
    pieces, info = [], {}
    for cls in CLASSES:
        row = trace['classes'][cls]
        indices = np.concatenate((row['positive'], row['negative']))
        if len(indices):
            labels = torch.as_tensor(row['labels'][indices], device=logits[cls].device)
            weights = torch.as_tensor(np.concatenate((row['positive_weights'],
                np.ones(len(row['negative']), np.float32))), device=logits[cls].device)
            loss = (F.cross_entropy(logits[cls][indices], labels, reduction='none')*weights).sum()/len(indices)
        else:
            loss = logits[cls].sum()*0
        pieces.append(loss)
        info[cls+'_ce'] = float(loss.detach())
        for name in ('positive', 'negative', 'hard_negative', 'random_negative', 'guarded_negative'):
            info[cls+'_'+name+'_count'] = len(row[name])
    indices = trace['bbox_positive']
    if len(indices):
        targets = torch.as_tensor(encode_boxes(ref[indices],
            assigned['gt_boxes'][assigned['matched_gt'][indices]]), device=delta.device)
        weights = torch.as_tensor(trace['bbox_weights'], device=delta.device)
        per_anchor = F.smooth_l1_loss(delta[indices], targets, beta=1/9, reduction='none').mean(1)
        regression = (per_anchor*weights).mean()
    else:
        regression = delta.sum()*0
    total = sum(pieces)+regression
    info.update(bbox_smooth_l1=float(regression.detach()), bbox_positive_count=len(indices),
                loss=float(total.detach()), target_gt_count=len(assigned['gt_boxes']))
    return (total, info, trace) if return_trace else (total, info)
