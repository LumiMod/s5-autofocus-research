"""Opt-in GT-related localization ordering loss; never an inference operation.

Frozen proposal: per GT choose at most one inverted-or-tied pair. Good is an already
sampled class-positive anchor with its unique shared bbox owner; bad is an
unowned static negative/ignore candidate in the existing [.3,.5) ambiguity band.
All selection, IoU quality and geometry are detached. No threshold, NMS, graph,
class target, or bbox owner is modified by this module.
"""
import numpy as np
import torch
from torch.nn import functional as F

from extract import require
from train_recognition import CLASSES, box_iou

GOOD_IOU = .5
BAD_IOU_MIN = .3
PAIR_OVERLAP = .3
OTHER_GT_AMBIGUITY_IOU = .3


def select_quality_pairs(logits, decoded_boxes, base_trace):
    decoded = (decoded_boxes.detach().cpu().numpy() if torch.is_tensor(decoded_boxes)
               else np.asarray(decoded_boxes)).astype(np.float64, copy=True)
    assigned = base_trace['assigned']
    owners = assigned['matched_gt']; boxes = assigned['gt_boxes']; classes = assigned['gt_classes']
    count = len(owners)
    valid = np.asarray(assigned['valid_anchors'], bool)
    require(valid.shape == (count,), 'Invalid anchor validity mask')
    require(decoded.shape == (count, 4) and np.isfinite(decoded).all(), 'Invalid detached decoded boxes')
    require(set(base_trace['bbox_positive']).issuperset(
        np.concatenate([base_trace['classes'][c]['positive'] for c in CLASSES]).tolist()),
        'Base classification positives lack bbox supervision')
    pairs, gt_reasons, budgets = [], [], {}
    for ci, cls in enumerate(CLASSES):
        require(logits[cls].shape == (count, 2) and bool(torch.isfinite(logits[cls]).all()), 'Invalid logits')
        z = (logits[cls][:, 1]-logits[cls][:, 0]).detach().cpu().numpy().astype(np.float64)
        selected = np.asarray(base_trace['classes'][cls]['positive'], np.int64)
        positive_weights = np.asarray(base_trace['classes'][cls]['positive_weights'], np.float64)
        selected_count = len(selected)+len(base_trace['classes'][cls]['negative'])
        require(positive_weights.shape == selected.shape and np.isfinite(positive_weights).all() and
                np.all(positive_weights > 0), 'Invalid base positive weights')
        budgets[cls] = dict(positive_mass=float(positive_weights.sum()), selected_count=selected_count,
            positive_ce_mass_fraction=float(positive_weights.sum()/selected_count) if selected_count else 0.)
        require(np.all(owners[selected] >= 0) and np.all(classes[owners[selected]] == ci), 'Base class/bbox owner mismatch')
        require(np.all(valid[selected]), 'Base positive has invalid anchor center')
        labels = assigned['labels'][cls]  # Static labels, before the base dynamic ignore guard.
        require(np.all(labels[selected] == 1), 'Ranking good pool contains a nonpositive')
        class_gt = np.flatnonzero(classes == ci)
        overlaps = box_iou(decoded, boxes[class_gt]) if len(class_gt) else np.empty((count, 0))
        for local_index, gi in enumerate(class_gt):
            quality = overlaps[:, local_index]
            other = np.delete(overlaps, local_index, axis=1)
            unambiguous = np.all(other < OTHER_GT_AMBIGUITY_IOU, axis=1)
            good_pool = selected[(owners[selected] == gi) & (quality[selected] >= GOOD_IOU) & unambiguous[selected]]
            reason = dict(target_class=cls, union_gt_index=int(gi), sampled_positive_count=int(np.count_nonzero(owners[selected] == gi)),
                          eligible_good_count=len(good_pool))
            if not len(good_pool):
                gt_reasons.append({**reason, 'reason': 'no_sampled_owned_unambiguous_good'}); continue
            # Quality determines the good; its current score does not select it.
            good = int(sorted(good_pool.tolist(), key=lambda index: (-quality[index], index))[0])
            relation = box_iou(decoded, decoded[good:good+1])[:, 0]
            bad_pool = np.flatnonzero(valid & (owners == -1) & np.isin(labels, [0, -1]) &
                (quality >= BAD_IOU_MIN) & (quality < GOOD_IOU) & unambiguous &
                (relation > PAIR_OVERLAP) & (z >= z[good]))
            reason.update(good_anchor=good, good_iou=float(quality[good]), eligible_inverted_bad_count=len(bad_pool))
            if not len(bad_pool):
                gt_reasons.append({**reason, 'reason': 'no_unowned_unambiguous_inverted_bad'}); continue
            bad = int(sorted(bad_pool.tolist(), key=lambda index: (-z[index], index))[0])
            require(good in base_trace['bbox_positive'], 'Ranking-positive gradient lacks bbox supervision')
            # These bads must already be protected from contradictory base negatives.
            require(bad not in base_trace['classes'][cls]['negative'], 'Ranking bad simultaneously sampled as base negative')
            pairs.append(dict(target_class=cls, union_gt_index=int(gi), good_anchor=good, bad_anchor=bad,
                good_iou=float(quality[good]), bad_iou=float(quality[bad]),
                quality_gap=float(quality[good]-quality[bad]), box_pair_iou=float(relation[bad]),
                good_log_odds=float(z[good]), bad_log_odds=float(z[bad]),
                good_static_label=int(labels[good]), bad_static_label=int(labels[bad]),
                good_owner=int(owners[good]), bad_owner=int(owners[bad]),
                base_gt_positive_mass=float(positive_weights[owners[selected] == gi].sum()),
                base_class_selected_count=selected_count,
                base_gt_ce_weight=float(positive_weights[owners[selected] == gi].sum()/selected_count)))
            gt_reasons.append({**reason, 'reason': 'pair_selected', 'bad_anchor': bad})
    for cls, budget in budgets.items():
        mass = sum(p['base_gt_ce_weight'] for p in pairs if p['target_class'] == cls)
        effective = sum(p['base_gt_ce_weight']*p['quality_gap'] for p in pairs if p['target_class'] == cls)
        require(mass <= budget['positive_ce_mass_fraction']+1e-9, 'Pair mass exceeds original positive CE mass')
        require(effective <= .7*budget['positive_ce_mass_fraction']+1e-9, 'Quality-weighted pair mass bound exceeded')
        budget.update(selected_pair_ce_mass=mass, quality_weighted_pair_mass=effective)
    return dict(pairs=pairs, per_gt_selection=gt_reasons, base_class_weight_budgets=budgets,
        selection='One highest-IoU sampled owned good, then one highest-log-odds eligible inverted-or-tied valid unowned bad per GT; index breaks ties.',
        geometry_detached=True, quality_detached=True, hard_selection_detached=True,
        threshold_source='Existing .3 ambiguity/NMS and .5 GT-match boundaries; not tuned on current cases.')


def quality_ranking_loss(logits, decoded_boxes, base_trace, *, return_trace=False):
    """Sum base_GT_CE_weight * ΔIoU * softplus(z_bad-z_good); no margin.

    z is foreground minus background logit. There is no bbox gradient and no
    inference-time GT requirement: callers opt in only while constructing loss.
    Pair membership is discontinuous at detached selection boundaries, as in
    hard mining. It is not a self-calibrated probability or accuracy metric.
    """
    trace = select_quality_pairs(logits, decoded_boxes, base_trace)
    zero = sum(value.sum()*0 for value in logits.values())
    pieces = []
    for pair in trace['pairs']:
        value = logits[pair['target_class']]
        good, bad = pair['good_anchor'], pair['bad_anchor']
        zg = value[good, 1]-value[good, 0]; zb = value[bad, 1]-value[bad, 0]
        pieces.append(float(pair['base_gt_ce_weight'])*float(pair['quality_gap'])*F.softplus(zb-zg))
    loss = zero + (torch.stack(pieces).sum() if pieces else 0)
    trace.update(pair_count=len(pieces), loss=float(loss.detach()),
                 normalization='Each GT retains its base positive CE mass divided by actual class sample count; no redistribution from missing pairs.',
                 network_or_decoder_modified=False, inference_requires_gt=False)
    return (loss, trace) if return_trace else loss
