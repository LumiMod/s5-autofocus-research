#!/usr/bin/env python3
"""Semantic checks for candidate training tools; synthetic fixtures are NOT accuracy data.

The optional source argument exercises the restored 44 candidate with synthetic
inputs. Its optimizer update is a tool check, never a recognition-training result.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from extract import MODELS, write_json
from pretraining_data import evaluate_detection, sha256
from research_paths import add_firmware_arguments, resolve_source
from train_recognition import (CLASSES, HEADS, Detector, assign_joint_targets, box_iou,
                               decode_boxes, detection_loss, detections, evaluate,
                               encode_boxes, flat_deltas, flat_logits,
                               sample_with_required, save_checkpoint, source_hashes, state_digest)


FIRMWARE = None


class TrainingSemantics(unittest.TestCase):
    def test_logits_pair_background_and_foreground_for_each_anchor(self):
        raw = torch.arange(50*16*24).reshape(1, 50, 16, 24)
        paired = flat_logits(raw)
        self.assertEqual(tuple(paired.shape), (9600, 2))
        for y, x, anchor in ((0, 0, 0), (15, 23, 24), (7, 11, 13)):
            index = (y*24+x)*25+anchor
            self.assertEqual(paired[index, 0], raw[0, anchor, y, x])
            self.assertEqual(paired[index, 1], raw[0, anchor+25, y, x])

    def test_bbox_channels_stay_with_their_anchor(self):
        raw = torch.arange(100*16*24).reshape(1, 100, 16, 24)
        grouped = flat_deltas(raw)
        for y, x, anchor in ((0, 0, 0), (15, 23, 24), (7, 11, 13)):
            index = (y*24+x)*25+anchor
            self.assertTrue(torch.equal(grouped[index], raw[0, 4*anchor:4*anchor+4, y, x]))

    def test_continuous_iou_has_no_plus_one_pixel(self):
        a = np.array([[0., 0., 2., 2.], [3., 3., 4., 4.]])
        b = np.array([[1., 1., 3., 3.]])
        np.testing.assert_allclose(box_iou(a, b)[:, 0], [1/7, 0])

    def test_box_coder_exact_inverse_for_fractional_in_bounds_targets(self):
        ref = np.array([[-2., -3., 50., 80.], [100., 20., 140., 60.]], np.float32)
        gt = np.array([[1.25, 2.75, 55.5, 90.5], [110.5, 21.5, 115.75, 30.25]], np.float32)
        np.testing.assert_allclose(decode_boxes(ref, encode_boxes(ref, gt)), gt, atol=1e-5)

    def test_zero_delta_identity_and_exp_clamp_are_finite(self):
        ref = np.array([[2., 4., 8., 12.]], np.float32)
        np.testing.assert_array_equal(decode_boxes(ref, np.zeros_like(ref)), ref)
        result = decode_boxes(ref, np.array([[0., 0., 1000., -1000.]], np.float32))
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(np.all(result >= 0))

    def test_assignment_positive_negative_ignore_and_forced_small_target(self):
        ref = np.array([[0., 0., 10., 10.], [40., 40., 50., 50.],
                        [-50., -50., -40., -40.]], np.float32)
        gt = np.array([[0., 0., 1., 1.]], np.float32)
        targets = {'person': gt, 'person_face': np.empty((0, 4), np.float32)}
        assignment = assign_joint_targets(ref, targets)
        np.testing.assert_array_equal(assignment['labels']['person'], [1, 0, -1])
        np.testing.assert_array_equal(assignment['labels']['person_face'], [0, 0, -1])
        self.assertEqual(assignment['matched_gt'][0], 0)
        self.assertAlmostEqual(float(assignment['forced_match_ious'][0]), .01)

    def test_nested_face_and_person_positive_scores_regress_to_their_own_gt(self):
        ref = np.array([[0., 0., 10., 10.], [0., 0., 5., 5.]], np.float32)
        gt = {'person': ref[:1].copy(), 'person_face': np.array([[1., 1., 2., 2.]], np.float32)}
        assignment = assign_joint_targets(ref, gt)
        np.testing.assert_array_equal(assignment['matched_gt'], [0, 1])
        np.testing.assert_array_equal(assignment['labels']['person'], [1, 0])
        np.testing.assert_array_equal(assignment['labels']['person_face'], [0, 1])
        logits = {name: torch.zeros(2, 2, requires_grad=True) for name in CLASSES}
        # Independent expected deltas: anchor center (2.5,2.5), face center
        # (1.5,1.5), both face dimensions 1 versus anchor dimensions 5.
        delta = torch.tensor([[0., 0., 0., 0.], [-.2, -.2, np.log(.2), np.log(.2)]],
                             dtype=torch.float32, requires_grad=True)
        loss, info = detection_loss(logits, delta, ref, gt, np.random.default_rng(2))
        self.assertEqual(info['bbox_positive_anchors'], 2)
        self.assertEqual(info['bbox_smooth_l1'], 0.)
        self.assertAlmostEqual(float(loss.detach()), 2*np.log(2), places=6)
        loss.backward()
        self.assertTrue(torch.equal(delta.grad, torch.zeros_like(delta)))
        self.assertTrue(all(v.grad is not None and bool(v.grad.isfinite().all()) for v in logits.values()))
        self.assertLess(float(logits['person'].grad[0, 1]), 0)
        self.assertGreater(float(logits['person'].grad[1, 1]), 0)
        self.assertLess(float(logits['person_face'].grad[1, 1]), 0)
        self.assertGreater(float(logits['person_face'].grad[0, 1]), 0)

    def test_same_box_different_classes_receive_distinct_anchors(self):
        ref = np.array([[0., 0., 10., 10.]]*2, np.float32)
        gt = {name: ref[:1].copy() for name in CLASSES}
        assignment = assign_joint_targets(ref, gt)
        self.assertEqual(set(assignment['forced_anchor_by_gt']), {0, 1})
        self.assertEqual(set(assignment['matched_gt']), {0, 1})
        for anchor, owner in enumerate(assignment['matched_gt']):
            self.assertEqual(sum(assignment['labels'][c][anchor] == 1 for c in CLASSES), 1)
            self.assertEqual(assignment['labels'][CLASSES[owner]][anchor], 1)
        self.assertEqual(len(assignment['unmatched_gt_indices']), 0)

    def test_impossible_exclusive_assignment_is_reported_and_loss_rejected(self):
        ref = np.array([[0., 0., 10., 10.]], np.float32)
        gt = {name: ref.copy() for name in CLASSES}
        assignment = assign_joint_targets(ref, gt)
        self.assertEqual(len(assignment['unmatched_gt_indices']), 1)
        with self.assertRaisesRegex(ValueError, 'Unmatched GTs'):
            detection_loss({name: torch.zeros(1, 2) for name in CLASSES},
                           torch.zeros(1, 4), ref, gt, np.random.default_rng(3))

    def test_competing_targets_cover_all_gt_without_reusing_an_anchor(self):
        # Three equal-choice/equal-area GTs form edges AB, BC, AB. A one-pass
        # reservation picks A, B, then strands the third GT: B must move to C.
        ref = np.array([[0, 0, 2, 2], [4, 0, 6, 2], [8, 0, 10, 2]], np.float32)
        gt = {'person': np.array([[0, 0, 6, 2], [4, 0, 10, 2], [0, 0, 6, 2]], np.float32),
              'person_face': np.empty((0, 4), np.float32)}
        assignment = assign_joint_targets(ref, gt)
        self.assertEqual(len(set(assignment['forced_anchor_by_gt'])), 3)
        self.assertTrue(np.all(assignment['forced_anchor_by_gt'] >= 0))
        self.assertEqual(len(assignment['unmatched_gt_indices']), 0)
        np.testing.assert_array_equal(assignment['matched_gt'][assignment['forced_anchor_by_gt']], np.arange(3))

    def test_zero_overlap_and_invalid_anchor_never_force_a_positive(self):
        ref = np.array([[20, 20, 30, 30], [-10, -10, -1, -1]], np.float32)
        gt = {'person': np.array([[0, 0, 1, 1]], np.float32),
              'person_face': np.empty((0, 4), np.float32)}
        assignment = assign_joint_targets(ref, gt)
        np.testing.assert_array_equal(assignment['forced_anchor_by_gt'], [-1])
        np.testing.assert_array_equal(assignment['labels']['person'], [0, -1])
        np.testing.assert_array_equal(assignment['unmatched_gt_indices'], [0])

    def test_union_iou_ambiguous_band_is_ignored_for_both_classes(self):
        ref = np.array([[0, 0, 10, 10], [0, 0, 4, 10], [20, 0, 30, 10]], np.float32)
        gt = {'person': ref[:1].copy(), 'person_face': np.empty((0, 4), np.float32)}
        assignment = assign_joint_targets(ref, gt)
        np.testing.assert_array_equal(assignment['labels']['person'], [1, -1, 0])
        np.testing.assert_array_equal(assignment['labels']['person_face'], [0, -1, 0])

    def test_sampling_keeps_required_anchors_or_explicitly_rejects_budget(self):
        for seed in range(4):
            selected = sample_with_required(np.arange(80), [3, 17, 79], 8, np.random.default_rng(seed))
            self.assertEqual(len(selected), 8)
            self.assertEqual(len(set(selected)), 8)
            self.assertTrue({3, 17, 79}.issubset(selected))
        with self.assertRaisesRegex(ValueError, 'exceeding the fixed budget'):
            sample_with_required(np.arange(80), np.arange(33), 32, np.random.default_rng(1))

    def test_every_sampled_class_positive_receives_bbox_gradient(self):
        ref = np.array([[0., 0., 10., 10.]]*160, np.float32)
        gt = {'person': np.array([[.5, .5, 9., 9.]], np.float32),
              'person_face': np.array([[1., 1., 8., 8.]], np.float32)}
        for seed in range(4):
            logits = {name: torch.zeros(len(ref), 2, requires_grad=True) for name in CLASSES}
            delta = torch.zeros(len(ref), 4, requires_grad=True)
            loss, info = detection_loss(logits, delta, ref, gt, np.random.default_rng(seed))
            loss.backward()
            classification_positive = set()
            for cls in CLASSES:
                classification_positive.update(torch.nonzero(logits[cls].grad[:, 1] < 0).flatten().tolist())
                self.assertEqual(info[cls+'_forced_positive_anchors'], 1)
                self.assertEqual(info[cls+'_supervised_gt'], 1)
            regression_supervised = set(torch.nonzero(delta.grad.abs().sum(1) > 0).flatten().tolist())
            self.assertTrue(classification_positive.issubset(regression_supervised))
            self.assertEqual(info['bbox_required_classification_positives'], len(classification_positive))
            self.assertEqual(info['bbox_supervised_gt'], 2)

    def test_empty_scene_has_finite_classification_and_zero_bbox_loss(self):
        ref = np.array([[0., 0., 10., 10.], [40., 40., 50., 50.]], np.float32)
        gt = {name: np.empty((0, 4), np.float32) for name in CLASSES}
        logits = {name: torch.zeros(2, 2, requires_grad=True) for name in CLASSES}
        delta = torch.zeros(2, 4, requires_grad=True)
        loss, info = detection_loss(logits, delta, ref, gt, np.random.default_rng(3))
        self.assertTrue(bool(loss.isfinite()))
        self.assertEqual(info['bbox_positive_anchors'], 0)
        loss.backward()
        self.assertTrue(torch.equal(delta.grad, torch.zeros_like(delta)))

    def test_all_invalid_anchors_and_empty_scene_give_finite_zero_loss(self):
        ref = np.array([[-10., -10., -5., -5.]], np.float32)
        gt = {name: np.empty((0, 4), np.float32) for name in CLASSES}
        logits = {name: torch.zeros(1, 2, requires_grad=True) for name in CLASSES}
        delta = torch.zeros(1, 4, requires_grad=True)
        loss, info = detection_loss(logits, delta, ref, gt, np.random.default_rng(3))
        self.assertEqual(float(loss.detach()), 0.)
        loss.backward()
        self.assertTrue(torch.equal(delta.grad, torch.zeros_like(delta)))
        self.assertTrue(all(torch.equal(value.grad, torch.zeros_like(value)) for value in logits.values()))

    def test_target_audit_skips_test_annotations_and_rejects_test_requests(self):
        from audit_recognition_targets import audit_dataset
        ref = np.array([[0, 0, 10, 10], [0, 0, 5, 5]], np.float32)
        dataset = {'classes': list(CLASSES), 'frames': [
            {'id': 'train-fixture', 'split': 'train', 'width': 384, 'height': 256,
             'boxes': [{'class': 'person', 'bbox_xyxy': [0, 0, 10, 10]},
                       {'class': 'person_face', 'bbox_xyxy': [1, 1, 2, 2]}]},
            {'split': 'test', 'boxes': object(), 'width': None}]}
        result = audit_dataset(dataset, ref, ('train',))
        self.assertTrue(result['passed'])
        self.assertEqual(result['splits']['train']['gt_count'], 2)
        self.assertEqual(result['splits']['train']['classes']['person_face']['legacy_cross_class_bbox_owner_anchors'], 1)
        self.assertEqual(result['splits']['train']['current_class_bbox_owner_conflicts'], 0)
        with self.assertRaisesRegex(ValueError, 'Only train/validation'):
            audit_dataset(dataset, ref, ('test',))

    def test_predictions_bind_source_model_and_actual_mutable_state_separately(self):
        class FixtureDetector:
            def __init__(self):
                self.model = torch.nn.Linear(1, 1)
                self.layout = 'synthetic-fixture-only'
                self.anchors = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], np.float32)

            def __call__(self, _tensor):
                return {cls: torch.zeros(2, 2) for cls in CLASSES}, torch.zeros(2, 4)

        dataset = {'classes': list(CLASSES), 'frames': [
            {'id': 'fixture', 'split': 'validation', 'width': 384, 'height': 256,
             'path': 'fixture.png', 'boxes': [
                 {'class': 'person', 'bbox_xyxy': [0, 0, 10, 10]},
                 {'class': 'person_face', 'bbox_xyxy': [20, 20, 30, 30]}]}]}
        detector = FixtureDetector()
        with tempfile.TemporaryDirectory(prefix='s5-prediction-provenance-') as temporary:
            root = Path(temporary)
            Image.new('RGB', (384, 256)).save(root/'fixture.png')
            _, before = evaluate(detector, dataset, root, 'synthetic-manifest', 'validation')
            with torch.no_grad():
                detector.model.weight.add_(.25)
            _, after = evaluate(detector, dataset, root, 'synthetic-manifest', 'validation')
        self.assertEqual(before['source_model_sha256'], MODELS['44_hm_d_nw_sng'])
        self.assertEqual(before['source_model_sha256'], after['source_model_sha256'])
        self.assertNotEqual(before['model_state_sha256'], after['model_state_sha256'])
        self.assertEqual(after['model_state_sha256'], state_digest(detector.model.state_dict()))

    def test_nms_uses_continuous_extents_even_for_subpixel_boxes(self):
        ref = np.array([[0., 0., .4, .4], [0., 0., .4, .4], [10., 10., 11., 11.]], np.float32)
        logits = {name: torch.tensor([[0., 5.], [0., 4.], [0., 3.]]) for name in CLASSES}
        rows = detections(logits, torch.zeros(3, 4), ref, {'width': 384, 'height': 256})
        self.assertEqual(len(rows), 4)
        for cls in CLASSES:
            self.assertEqual(sum(r['class'] == cls for r in rows), 2)

    def test_ap_counts_empty_frame_false_alarm_and_duplicate_detection(self):
        dataset = {'classes': list(CLASSES), 'frames': [
            {'id': 'object', 'split': 'test', 'width': 100, 'height': 100,
             'boxes': [{'class': 'person', 'bbox_xyxy': [10, 10, 30, 30]}]},
            {'id': 'background', 'split': 'test', 'width': 100, 'height': 100, 'boxes': []}]}
        predictions = {'dataset_manifest_sha256': 'synthetic_fixture_only', 'frames': [
            {'id': 'object', 'detections': [
                {'class': 'person', 'bbox_xyxy': [10, 10, 30, 30], 'score': .8},
                {'class': 'person', 'bbox_xyxy': [10, 10, 30, 30], 'score': .7}]},
            {'id': 'background', 'detections': [
                {'class': 'person', 'bbox_xyxy': [10, 10, 30, 30], 'score': .9}]}]}
        metrics = evaluate_detection(dataset, predictions, 'test')
        self.assertAlmostEqual(metrics['mean_ap'], .5)
        result = metrics['classes']['person']
        self.assertEqual((result['tp_at_score'], result['fp_at_score'], result['fn_at_score']), (1, 2, 0))
        self.assertIsNone(metrics['classes']['person_face']['ap_at_iou'])

    def test_checkpoint_restores_optimizer_and_both_rng_streams_exactly(self):
        def build():
            torch.manual_seed(123)
            detector = SimpleNamespace(model=torch.nn.Linear(3, 2))
            optimizer = torch.optim.AdamW(detector.model.parameters(), lr=.001)
            return detector, optimizer, np.random.default_rng(77)

        def step(detector, optimizer, rng):
            x = torch.from_numpy(rng.normal(size=(4, 3)).astype(np.float32))+torch.rand(4, 3)
            y = torch.from_numpy(rng.normal(size=(4, 2)).astype(np.float32))
            optimizer.zero_grad(set_to_none=True)
            (detector.model(x)-y).square().mean().backward()
            optimizer.step()

        detector, optimizer, rng = build()
        for _ in range(4):
            step(detector, optimizer, rng)
        uninterrupted = state_digest(detector.model.state_dict())
        detector, optimizer, rng = build()
        for _ in range(2):
            step(detector, optimizer, rng)
        with tempfile.TemporaryDirectory(prefix='s5-checkpoint-check-') as temporary:
            path = Path(temporary)/'last.pt'
            save_checkpoint(path, detector, optimizer, 1, 2, .1, {'fixture': True}, rng)
            checkpoint = torch.load(path, weights_only=False)
            detector, optimizer, rng = build()
            detector.model.load_state_dict(checkpoint['model_state'])
            optimizer.load_state_dict(checkpoint['optimizer_state'])
            torch.set_rng_state(checkpoint['torch_rng_state'])
            rng.bit_generator.state = checkpoint['numpy_rng_state']
            for _ in range(2):
                step(detector, optimizer, rng)
        self.assertEqual(state_digest(detector.model.state_dict()), uninterrupted)
        self.assertEqual(checkpoint['completed_epoch'], 1)
        self.assertEqual(checkpoint['global_step'], 2)

    def test_restored_candidate_updates_only_heads_with_finite_gradients(self):
        if FIRMWARE is None:
            self.skipTest('Pass --firmware-root or --source for restored-candidate tool check')
        torch.manual_seed(9)
        detector = Detector(FIRMWARE, 'o4_i4_hw')
        try:
            before = {k: v.clone() for k, v in detector.model.state_dict().items()}
            parameters = [p for p in detector.model.parameters() if p.requires_grad]
            self.assertEqual(sum(p.numel() for p in parameters), 89800)
            tensor = torch.rand(1, 1, 256, 384)*255-117
            logits, delta = detector(tensor)
            gt = {'person': np.array([[40, 30, 150, 240]], np.float32),
                  'person_face': np.array([[60, 40, 120, 100]], np.float32)}
            loss, _ = detection_loss(logits, delta, detector.anchors, gt, np.random.default_rng(9))
            loss.backward()
            self.assertTrue(all(p.grad is not None and bool(p.grad.isfinite().all()) for p in parameters))
            self.assertTrue(all(p.grad is None for p in detector.model.parameters() if not p.requires_grad))
            torch.nn.utils.clip_grad_norm_(parameters, 10, error_if_nonfinite=True)
            torch.optim.AdamW(parameters, lr=.0001).step()
            changed = []
            for key, value in detector.model.state_dict().items():
                if key.split('.')[1] in HEADS.values():
                    changed.append(not torch.equal(value, before[key]))
                else:
                    self.assertTrue(torch.equal(value, before[key]), key)
            self.assertTrue(all(changed))
        finally:
            detector.close()


def main():
    global FIRMWARE
    parser = argparse.ArgumentParser(description=__doc__)
    add_firmware_arguments(parser)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'artifacts/recognition-tool-validation.json')
    args = parser.parse_args()
    source = resolve_source(args)
    path = source/'44_hm_d_nw_sng.bin'
    if path.is_file():
        if sha256(path) != MODELS['44_hm_d_nw_sng']:
            raise ValueError('Restored candidate source digest mismatch')
        FIRMWARE = path.read_bytes()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(TrainingSemantics))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, {'status': 'synthetic tool semantics only; not recognition accuracy or a real training run',
                            'tests_run': result.testsRun, 'failures': len(result.failures),
                            'errors': len(result.errors), 'skipped': len(result.skipped),
                            'restored_candidate_exercised': FIRMWARE is not None,
                            'source_hashes': {**source_hashes(), Path(__file__).name: sha256(Path(__file__)),
                                              'audit_recognition_targets.py': sha256(Path(__file__).with_name('audit_recognition_targets.py'))}})
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
