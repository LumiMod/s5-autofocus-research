#!/usr/bin/env python3
"""Toy and fixed train32 output-gradient checks; NO optimizer or deployment claim."""
import argparse
import copy
import hashlib
import io
import json
from pathlib import Path
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from extract import MODELS, require, write_json
from pretraining_data import checked_file, sha256
from recognition_balanced_loss import balanced_detection_loss, equal_gt_weights
from recognition_quality_ranking import quality_ranking_loss
from research_paths import add_firmware_arguments, resolve_source
from train_recognition import (CLASSES, Detector, MODEL, assign_joint_targets, decode_boxes,
                               ground_truth, load_image, source_hashes, state_digest)

BASE = Path(__file__).resolve().parent
MANIFEST_SHA = 'af5459ddebd98947b154580cd176e7846abcceffe5539ac6f0001f6a485a3fc8'


def fixture(ref=None, gt=None, decoded=None):
    ref = np.asarray(ref if ref is not None else [[0, 0, 10, 10], [30, 0, 40, 10]], np.float32)
    if gt is None: gt = {'person': [], 'person_face': [[0, 0, 10, 10]]}
    gt = {c: np.asarray(gt[c], np.float32).reshape(-1, 4) for c in CLASSES}
    assigned = assign_joint_targets(ref, gt)
    trace = {'assigned': assigned, 'classes': {}}
    for c in CLASSES:
        positive = np.flatnonzero(assigned['labels'][c] == 1)
        trace['classes'][c] = dict(positive=positive, negative=np.empty(0, np.int64),
                                  positive_weights=equal_gt_weights(positive, assigned['matched_gt']))
    trace['bbox_positive'] = np.flatnonzero(assigned['matched_gt'] >= 0)
    logits = {c: torch.zeros(len(ref), 2, dtype=torch.float64, requires_grad=True) for c in CLASSES}
    logits['person_face'] = torch.tensor([[0., 1.], [0., 3.]] + [[0., 3.]]*(len(ref)-2), dtype=torch.float64, requires_grad=True)
    values = decoded if decoded is not None else [[0, 0, 10, 10], [0, 0, 24, 10]]
    boxes = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    return logits, boxes, trace


class QualityTests(unittest.TestCase):
    def test_exact_loss_and_directions_without_geometry_gradient(self):
        logits, boxes, trace = fixture()
        loss, report = quality_ranking_loss(logits, boxes, trace, return_trace=True)
        expected = (1-10/24)*F.softplus(torch.tensor(2., dtype=torch.float64))
        self.assertAlmostEqual(float(loss.detach()), float(expected), places=12)
        gradients = torch.autograd.grad(loss, [logits['person_face'], boxes], allow_unused=True)
        self.assertIsNone(gradients[1])
        self.assertLess(float(gradients[0][0, 1]), 0); self.assertGreater(float(gradients[0][1, 1]), 0)
        torch.testing.assert_close(gradients[0].sum(1), torch.zeros(2, dtype=torch.float64))
        self.assertEqual(report['pairs'][0]['good_anchor'], 0); self.assertEqual(report['pairs'][0]['bad_anchor'], 1)

    def test_common_logit_offset_does_not_change_loss(self):
        logits, boxes, trace = fixture()
        a = quality_ranking_loss(logits, boxes, trace)
        shifted = {c: v+torch.tensor([[100.], [-47.]], dtype=torch.float64) for c, v in logits.items()}
        torch.testing.assert_close(a, quality_ranking_loss(shifted, boxes, trace))

    def test_no_good_does_not_press_down_bad(self):
        logits, boxes, trace = fixture(decoded=[[0, 0, 40, 10], [0, 0, 24, 10]])
        self.assertEqual(float(quality_ranking_loss(logits, boxes, trace).detach()), 0.)

    def test_correct_or_exact_half_iou_bad_is_protected(self):
        for width in (16, 20):
            logits, boxes, trace = fixture(decoded=[[0, 0, 10, 10], [0, 0, width, 10]])
            self.assertEqual(float(quality_ranking_loss(logits, boxes, trace).detach()), 0.)

    def test_correct_order_has_no_ranking_penalty(self):
        logits, boxes, trace = fixture()
        logits['person_face'] = logits['person_face'].flip(0)
        self.assertEqual(float(quality_ranking_loss(logits, boxes, trace).detach()), 0.)

    def test_owned_positive_bad_is_not_pushed_against_ce(self):
        logits, boxes, trace = fixture(ref=[[0, 0, 10, 10], [0, 0, 10, 10]])
        self.assertEqual(float(quality_ranking_loss(logits, boxes, trace).detach()), 0.)

    def test_different_class_owner_is_protected(self):
        logits, boxes, trace = fixture(gt={'person': [[30, 0, 40, 10]], 'person_face': [[0, 0, 10, 10]]})
        self.assertEqual(float(quality_ranking_loss(logits, boxes, trace).detach()), 0.)

    def test_multiple_gt_ambiguity_is_protected(self):
        logits, boxes, trace = fixture(ref=[[0, 0, 10, 10], [30, 0, 40, 10], [12, 0, 22, 10]],
            gt={'person': [], 'person_face': [[0, 0, 10, 10], [12, 0, 22, 10]]},
            decoded=[[0, 0, 10, 10], [0, 0, 24, 10], [12, 0, 22, 10]])
        self.assertEqual(float(quality_ranking_loss(logits, boxes, trace).detach()), 0.)

    def test_empty_gt_returns_differentiable_zero(self):
        logits, boxes, trace = fixture(gt={'person': [], 'person_face': []})
        loss = quality_ranking_loss(logits, boxes, trace)
        gs = torch.autograd.grad(loss, list(logits.values()))
        self.assertTrue(all(torch.count_nonzero(g) == 0 for g in gs))

    def test_bbox_supervision_requirement_is_checked(self):
        logits, boxes, trace = fixture(); trace['bbox_positive'] = np.empty(0, np.int64)
        with self.assertRaisesRegex(ValueError, 'bbox supervision'):
            quality_ranking_loss(logits, boxes, trace)

    def test_tie_uses_lower_anchor_index(self):
        logits, boxes, trace = fixture(ref=[[0, 0, 10, 10], [30, 0, 40, 10], [30, 0, 40, 10]],
            decoded=[[0, 0, 10, 10], [0, 0, 24, 10], [0, 0, 24, 10]])
        _, report = quality_ranking_loss(logits, boxes, trace, return_trace=True)
        self.assertEqual(report['pairs'][0]['bad_anchor'], 1)

    def test_outside_center_bad_cannot_participate(self):
        logits, boxes, trace = fixture(ref=[[0, 0, 10, 10], [-30, 0, -20, 10]])
        self.assertFalse(trace['assigned']['valid_anchors'][1])
        self.assertEqual(trace['assigned']['labels']['person_face'][1], -1)
        self.assertEqual(float(quality_ranking_loss(logits, boxes, trace).detach()), 0.)

    def test_equal_scores_are_an_explicit_tied_pair(self):
        logits, boxes, trace = fixture()
        logits['person_face'] = torch.tensor([[0., 1.], [0., 1.]], dtype=torch.float64, requires_grad=True)
        loss, report = quality_ranking_loss(logits, boxes, trace, return_trace=True)
        self.assertEqual(report['pair_count'], 1)
        self.assertAlmostEqual(float(loss.detach()), (1-10/24)*np.log(2), places=12)

    def test_static_targets_and_base_sampling_are_not_mutated(self):
        logits, boxes, trace = fixture(); before = copy.deepcopy(trace)
        quality_ranking_loss(logits, boxes, trace)
        for key in ('matched_gt', 'gt_classes', 'gt_boxes', 'forced_anchor_by_gt'):
            np.testing.assert_array_equal(trace['assigned'][key], before['assigned'][key])
        for c in CLASSES:
            np.testing.assert_array_equal(trace['assigned']['labels'][c], before['assigned']['labels'][c])
            np.testing.assert_array_equal(trace['classes'][c]['positive'], before['classes'][c]['positive'])

    def test_actual_class_sample_denominator_sets_rank_mass(self):
        logits, boxes, trace = fixture(ref=[[0, 0, 10, 10], [30, 0, 40, 10], [50, 0, 60, 10]],
                                      decoded=[[0, 0, 10, 10], [0, 0, 24, 10], [50, 0, 60, 10]])
        trace['classes']['person_face']['negative'] = np.array([2])
        loss, report = quality_ranking_loss(logits, boxes, trace, return_trace=True)
        expected = .5*(1-10/24)*F.softplus(torch.tensor(2., dtype=torch.float64))
        self.assertAlmostEqual(float(loss.detach()), float(expected), places=12)
        self.assertEqual(report['pairs'][0]['base_gt_ce_weight'], .5)

    def test_missing_pair_does_not_redistribute_another_gt_mass(self):
        logits, boxes, trace = fixture(ref=[[0, 0, 10, 10], [60, 0, 70, 10], [100, 0, 110, 10], [160, 0, 170, 10]],
            gt={'person': [], 'person_face': [[0, 0, 10, 10], [100, 0, 110, 10]]},
            decoded=[[0, 0, 10, 10], [0, 0, 24, 10], [100, 0, 110, 10], [100, 0, 124, 10]])
        logits['person_face'] = torch.tensor([[0., 1.], [0., 3.], [0., 1.], [0., 3.]], dtype=torch.float64, requires_grad=True)
        first, info = quality_ranking_loss(logits, boxes, trace, return_trace=True)
        first_grad = torch.autograd.grad(first, logits['person_face'])[0]
        changed = {c: v.detach().clone().requires_grad_(True) for c, v in logits.items()}
        with torch.no_grad(): changed['person_face'][3, 1] = 0
        second, second_info = quality_ranking_loss(changed, boxes, trace, return_trace=True)
        second_grad = torch.autograd.grad(second, changed['person_face'])[0]
        self.assertEqual(info['pair_count'], 2); self.assertEqual(second_info['pair_count'], 1)
        torch.testing.assert_close(first_grad[:2], second_grad[:2])
        self.assertEqual(info['base_class_weight_budgets']['person_face']['selected_pair_ce_mass'], 1.)
        self.assertEqual(second_info['base_class_weight_budgets']['person_face']['selected_pair_ce_mass'], .5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_firmware_arguments(parser)
    parser.add_argument('--dataset', type=Path, default=BASE/'private-data/recognition-v2/manifest.json')
    parser.add_argument('--run', type=Path, default=BASE/'artifacts/recognition-small-anchor-v2')
    parser.add_argument('--output', type=Path, default=BASE/'artifacts/recognition-ranking-repair/quality-ranking-normalized')
    args = parser.parse_args()
    require(not args.output.exists(), 'Preserve existing ranking evidence')
    torch.set_num_threads(2); torch.manual_seed(20261127); torch.use_deterministic_algorithms(True)
    stream = io.StringIO(); tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(QualityTests))
    require(tests.wasSuccessful(), stream.getvalue())
    require(sha256(args.dataset) == MANIFEST_SHA, 'Unexpected dataset')
    dataset = json.loads(args.dataset.read_text())
    frames = sorted((f for f in dataset['frames'] if f['split'] == 'train'), key=lambda f: f['id'])[:32]
    del dataset
    prior_path = BASE/'artifacts/recognition-fp-diagnosis/balanced-loss-verification/report.json'
    prior = json.loads(prior_path.read_text()); prior_frames = {r['frame_id']: r for r in prior['frames']}
    diagnosis_path = BASE/'artifacts/recognition-fp-diagnosis/training-sampling/report.json'
    diagnosis = json.loads(diagnosis_path.read_text())
    head_reference = BASE/'artifacts/recognition-ranking-repair/box-contract/head-reference.json'
    require(head_reference.exists(), 'Missing explicit head/face semantic reference')
    stage_base = BASE/'artifacts/recognition-diagnosis-v3/small-face-after-v2-train32'
    stage_rows = {r['frame_id']: r for r in json.loads((stage_base/'report.json').read_text())['baseline']}
    require({f['id'] for f in frames} == set(prior_frames) == set(stage_rows), 'Fixed train32 differs')
    checkpoint_path = args.run/'epoch-06/checkpoint.pt'; checkpoint_sha = sha256(checkpoint_path)
    require(checkpoint_sha == prior['checkpoint_sha256'], 'Wrong frozen checkpoint')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    data = (resolve_source(args)/(MODEL+'.bin')).read_bytes(); require(hashlib.sha256(data).hexdigest() == MODELS[MODEL], 'Wrong original model')
    detector = Detector(data, 'gemm_k4'); detector.model.load_state_dict(checkpoint['model_state'])
    detector.anchors = np.load(args.run/'anchors-small.npy', allow_pickle=False)
    detector.model.requires_grad_(False); detector.model.eval(); before = state_digest(detector.model.state_dict())
    require(before == prior['checkpoint_state_sha256'], 'Frozen state differs')
    sources = {**source_hashes(), **{n: sha256(BASE/n) for n in (
        'recognition_balanced_loss.py', 'recognition_quality_ranking.py', Path(__file__).name, 'research_paths.py')}}
    require(sources['recognition_balanced_loss.py'] == prior['source_hashes']['recognition_balanced_loss.py'], 'Base loss changed')
    args.output.mkdir(parents=True); snapshot = args.output/'source-snapshot'; snapshot.mkdir()
    for n in sources: (snapshot/n).write_bytes((BASE/n).read_bytes())
    plan = dict(status='frozen_before_real_output_gradients', manifest_sha256=MANIFEST_SHA,
        checkpoint_sha256=checkpoint_sha, checkpoint_state_sha256=before, source_hashes=sources,
        frame_ids=[f['id'] for f in frames], base_verification_sha256=sha256(prior_path),
        rules={'good': 'Already sampled positive, correct unique owner, bbox supervised, decoded IoU>=.5; select highest IoU.',
               'bad': 'Valid center, static0/-1 and owner==-1; decoded same GT IoU in [.3,.5); overlaps good>.3; inverted or tied bad log-odds>=good.',
               'multi_gt': 'Both candidates must have IoU<.3 to all other same-class GTs.',
               'pair_limit': 'At most one pair per GT; bad highest detached log-odds, ties lower anchor index.',
               'loss': 'Sum base_GT_positive_mass/class_actual_sample_count * detached_quality_gap * softplus(z_bad-z_good); no added margin or redistribution.',
               'combined_coefficient': 1., 'selection_and_geometry_detached': True},
        optimizer_steps=0, validation_inference=False, test_inference=False,
        head_reference_sha256=sha256(head_reference),
        training_blocked_pending_target_contract=True,
        threshold_or_candidate_search=False, inference_needs_gt=False,
        interpretation='Output-gradient verification only; not parameter updates or improved detections.')
    write_json(args.output/'frozen-plan.json', plan)
    rows, small_cases = [], []
    try:
        for frame in frames:
            fid = frame['id']; checked_file(args.dataset.parent, frame)
            with torch.inference_mode(): raw_logits, raw_delta = detector(load_image(args.dataset.parent, frame))
            logits = {c: raw_logits[c].detach().clone().requires_grad_(True) for c in CLASSES}
            delta = raw_delta.detach().clone().requires_grad_(True)
            decoded = decode_boxes(detector.anchors, delta.detach().numpy())
            npz_path = stage_base/stage_rows[fid]['prediction_file']
            require(sha256(npz_path) == stage_rows[fid]['saved_prediction_sha256'], 'Saved NPZ changed')
            with np.load(npz_path, allow_pickle=False) as raw:
                require(np.array_equal(decoded, raw['decoded_boxes']), 'Frozen decoded outputs changed')
                for c in CLASSES:
                    require(np.array_equal(logits[c].softmax(1).detach().numpy()[:, 1], raw['scores_'+c]), 'Frozen scores changed')
            rng = np.random.default_rng(); rng.bit_generator.state = copy.deepcopy(prior_frames[fid]['rng_before_loss'])
            base_loss, base_info, trace = balanced_detection_loss(logits, delta, detector.anchors, ground_truth(frame), rng, return_trace=True)
            for c in CLASSES:
                for now, old in [('positive', 'selected_positive'), ('negative', 'selected_negative'), ('guarded_negative', 'guarded_negative')]:
                    require(np.array_equal(trace['classes'][c][now], prior_frames[fid]['class_checks'][c][old]), 'Base sample changed')
            require(np.array_equal(trace['bbox_positive'], prior_frames[fid]['bbox_indices']), 'Base bbox sample changed')
            # Use a differentiable decoded proxy to explicitly prove detachment.
            decoded_proxy = torch.tensor(decoded, requires_grad=True)
            rank_loss, ranking = quality_ranking_loss(logits, decoded_proxy, trace, return_trace=True)
            targets = [logits[c] for c in CLASSES]+[delta]
            bg = torch.autograd.grad(base_loss, targets, retain_graph=True)
            rg_all = torch.autograd.grad(rank_loss, targets+[decoded_proxy], retain_graph=True, allow_unused=True)
            require(rg_all[2] is None and rg_all[3] is None, 'Rank leaked a bbox/geometry gradient')
            cg = torch.autograd.grad(base_loss+rank_loss, targets)
            require(torch.equal(bg[2], cg[2]), 'Ranking changed bbox gradient')
            gradient_summary = {}
            for ci, c in enumerate(CLASSES):
                require(torch.allclose(cg[ci], bg[ci]+rg_all[ci], rtol=1e-6, atol=1e-7), 'Combined gradient differs')
                allowed = {p[k] for p in ranking['pairs'] if p['target_class'] == c for k in ('good_anchor', 'bad_anchor')}
                changed = set(torch.nonzero(rg_all[ci].abs().sum(1)>0).flatten().tolist())
                require(changed.issubset(allowed), 'Ranking gradient touched an unrelated anchor')
                base_norm=float(bg[ci].norm());rank_norm=float(rg_all[ci].norm())
                gradient_summary[c] = dict(base_l2=base_norm, rank_l2=rank_norm, combined_l2=float(cg[ci].norm()),
                    rank_to_base_l2=rank_norm/base_norm if base_norm else None,
                    base_abs_max=float(bg[ci].abs().max()), rank_abs_max=float(rg_all[ci].abs().max()),
                    combined_abs_max=float(cg[ci].abs().max()))
            for pair in ranking['pairs']:
                ci = CLASSES.index(pair['target_class']); good, bad = pair['good_anchor'], pair['bad_anchor']
                require(float(rg_all[ci][good, 1]) < 0 and float(rg_all[ci][bad, 1]) > 0, 'Ranking directions wrong')
                require(float(cg[ci][good, 1]) < 0 and float(cg[ci][bad, 1]) > 0, 'Combined directions conflict')
                pair['foreground_logit_gradients'] = {name: dict(base=float(bg[ci][index, 1]), ranking=float(rg_all[ci][index, 1]),
                    combined=float(cg[ci][index, 1])) for name, index in [('good', good), ('bad', bad)]}
            original_indices = [i for c in CLASSES for i, b in enumerate(frame['boxes']) if b['class'] == c]
            for old in diagnosis['small_face_details']:
                if old['frame_id'] != fid: continue
                gi = original_indices.index(old['gt_index'])
                selection = next(r for r in ranking['per_gt_selection'] if r['union_gt_index'] == gi)
                item = dict(frame_id=fid, gt_index=old['gt_index'], union_gt_index=gi, previous_stage=old['refined_stage'], selection=selection)
                if old['refined_stage'].startswith('all_pretop'):
                    candidate = old['confident_candidates'][0]; parent = candidate['suppressor']; idx = parent['anchor_index']; ci = CLASSES.index('person_face')
                    item['main_suppressor'] = {**parent, 'foreground_logit_gradient_base': float(bg[ci][idx, 1]),
                        'foreground_logit_gradient_ranking': float(rg_all[ci][idx, 1]), 'foreground_logit_gradient_combined': float(cg[ci][idx, 1]),
                        'selected_as_ranking_bad': any(p['target_class']=='person_face' and p['bad_anchor']==idx for p in ranking['pairs'])}
                small_cases.append(item)
            rows.append(dict(frame_id=fid, base_loss=float(base_loss.detach()), ranking_loss=float(rank_loss.detach()),
                combined_loss=float((base_loss+rank_loss).detach()), base_info=base_info, ranking=ranking,
                class_output_gradient_norms=gradient_summary, base_selection_matches_frozen_verification=True,
                bbox_gradient_unchanged=True, ranking_geometry_gradient_is_none=True))
    finally:
        detector.close()
    require(state_digest(detector.model.state_dict()) == before and sha256(checkpoint_path) == checkpoint_sha, 'Checkpoint/model changed')
    require(all(sha256(BASE/n) == h for n, h in sources.items()), 'Sources changed during verification')
    require(len(small_cases) == 10, 'Expected same ten small faces')
    report = dict(passed_supervision_checks=True, toy_tests=dict(count=tests.testsRun, passed=True, log=stream.getvalue()),
        source_hashes=sources, frozen_plan_sha256=sha256(args.output/'frozen-plan.json'),
        checkpoint_sha256_before=checkpoint_sha, checkpoint_sha256_after=sha256(checkpoint_path),
        checkpoint_state_sha256_before=before, checkpoint_state_sha256_after=state_digest(detector.model.state_dict()),
        model_and_graph_unchanged=True, train_frames=32, frames=rows, small_face_cases=small_cases,
        total_pairs=sum(r['ranking']['pair_count'] for r in rows), optimizer_steps=0,
        validation_inference=False, test_inference=False, inference_requires_gt=False,
        passed_detection_acceptance=False, ready_to_train=False,
        head_reference=dict(path=head_reference.relative_to(BASE).as_posix(), sha256=sha256(head_reference)),
        training_blocked_pending_target_contract=True,
        limitations=['Training-only output-gradient check, with no optimizer or changed weights; outputs and accuracy remain unchanged.',
            'Only inverted-or-tied eligible owned-good/valid-unowned-bad GT pairs are supervised; missing-good, ownership or ambiguity exclusions remain gaps.',
            'Pair weight uses the GT base CE mass; this bounds coefficient mass, not gradient norm relative to current CE.',
            'Quality and pair selection are detached; no differentiable geometry reward can manipulate its own target.',
            'No NMS/score threshold change or GT reinterpretation as whole-head boxes was made.',
            'Visible-face and source-head reference semantics can disagree; original factory box semantics remain unknown and training is blocked pending target-contract resolution.'])
    write_json(args.output/'report.json', report)
    print(json.dumps(dict(report=str(args.output/'report.json'), sha256=sha256(args.output/'report.json'),
        toy_tests=tests.testsRun, total_pairs=report['total_pairs'], small_faces=small_cases)), flush=True)


if __name__ == '__main__': main()
