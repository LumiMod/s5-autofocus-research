#!/usr/bin/env python3
"""Synthetic geometry/statistics fixtures only; never evidence of recognition."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

import paired_target_behavior as p


def example_frame(boxes=None, width=64, height=48):
    return {'id': 'synthetic_train', 'split': 'train', 'width': width, 'height': height,
            'boxes': boxes if boxes is not None else [{'class': 'person_face', 'bbox_xyxy': [10., 10., 30., 30.]}]}


def detection(box, score=.9, cls='person_face'):
    return {'class': cls, 'bbox_xyxy': list(box), 'score': score}


def synthetic_plan_and_predictions(stationary=False, empty=False):
    frame = example_frame()
    variants = [{'name': 'real', 'op': 'identity', 'pixel_sha256': 'synthetic-real'},
                {'name': 'translate_right', 'op': 'translate_crop', 'dx': 5, 'dy': 0, 'pixel_sha256': 'synthetic-shift'},
                {'name': 'occlude_person_face_target', 'op': 'mask_rect', 'pixel_sha256': 'synthetic-target'},
                {'name': 'occlude_person_face_background', 'op': 'mask_rect', 'pixel_sha256': 'synthetic-background'}]
    pair = {'target_class': 'person_face', 'gt_index': 0, 'target_rect': [10, 10, 30, 30],
            'annotation_free_rect': [40, 0, 60, 20], 'mask_pixels': 400}
    plan = {'split': 'train', 'frames': [{'frame': frame, 'variants': variants,
             'occlusion_pairs': {'person': None, 'person_face': pair}}], 'limits': []}
    boxes = [[10, 10, 30, 30], [10 if stationary else 15, 10, 30 if stationary else 35, 30],
             [10, 10, 30, 30], [10, 10, 30, 30]]
    records = [{'frame_id': frame['id'], 'variant': v['name'], 'input_pixel_sha256': v['pixel_sha256'],
                'detections': [] if empty else [detection(b, score)]}
               for v, b, score in zip(variants, boxes, [.9, .8, .1, .8])]
    prediction = {'model': 'synthetic_fixture_only', 'split': 'train', 'classes': ['person_face'], 'records': records}
    return plan, prediction


class GeometryTests(unittest.TestCase):
    def test_translation_exact_pixels_four_directions_and_diagonal(self):
        image = np.arange(7*11*3, dtype=np.uint8).reshape(7, 11, 3)
        before = image.copy()
        for dx, dy in [(2, 0), (-2, 0), (0, 3), (0, -3), (2, -3), (11, 0), (0, -7)]:
            actual = p.transform_rgb(image, {'op': 'translate_crop', 'dx': dx, 'dy': dy, 'fill_rgb': [117]*3})
            expected = np.full_like(image, 117)
            for y in range(7):
                for x in range(11):
                    if 0 <= x-dx < 11 and 0 <= y-dy < 7:
                        expected[y, x] = image[y-dy, x-dx]
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(image, before)

    def test_mask_is_exact_half_open_and_original_unchanged(self):
        image = np.zeros((7, 11, 3), np.uint8)
        result = p.transform_rgb(image, {'op': 'mask_rect', 'rect': [2, 1, 7, 3], 'fill_rgb': [117]*3})
        self.assertEqual(np.count_nonzero(np.any(result != image, axis=2)), 10)
        self.assertEqual(int(image.sum()), 0)
        self.assertEqual(result[3, 2, 0], 0)

    def test_fractional_gt_mask_enclosure(self):
        self.assertEqual(p.integer_rect([1.2, 2.8, 5.0, 6.1], 8, 9), [1, 2, 5, 7])

    def test_free_search_against_independent_bruteforce(self):
        rng = np.random.default_rng(725)
        for _ in range(25):
            mask = rng.integers(0, 2, (7, 11), dtype=np.uint8)
            width, height = int(rng.integers(1, 5)), int(rng.integers(1, 4))
            expected = next(([x, y, x+width, y+height] for y in range(8-height)
                             for x in range(12-width) if not mask[y:y+height, x:x+width].any()), None)
            self.assertEqual(p.first_free_rectangle(mask, width, height), expected)

    def test_background_excludes_all_classes_and_keeps_shape(self):
        frame = example_frame([{'class': 'person', 'bbox_xyxy': [0., 0., 8., 8.]},
                               {'class': 'person_face', 'bbox_xyxy': [3.2, 2.3, 6.8, 5.1]}], 12, 8)
        pair = p.occlusion_pair(frame, 'person_face')
        self.assertEqual(pair['target_rect'], [3, 2, 7, 6])
        self.assertEqual(pair['annotation_free_rect'], [8, 0, 12, 4])
        self.assertEqual(pair['mask_pixels'], 16)

    def test_unavailable_never_shrinks_or_overlaps(self):
        frame = example_frame([{'class': 'person', 'bbox_xyxy': [0., 0., 10., 10.]},
                               {'class': 'person_face', 'bbox_xyxy': [2., 2., 4., 4.]}], 10, 10)
        self.assertIsNone(p.occlusion_pair(frame, 'person'))
        self.assertIsNone(p.occlusion_pair(frame, 'person_face'))

    def test_translated_truth_clipping_and_full_eligibility(self):
        frame = example_frame([{'class': 'person_face', 'bbox_xyxy': [0., 0., 2., 2.]},
                               {'class': 'person_face', 'bbox_xyxy': [3., 1., 7., 5.]},
                               {'class': 'person_face', 'bbox_xyxy': [8., 1., 10., 5.]}], 10, 7)
        truth = p.mapped_truth(frame, {'op': 'translate_crop', 'dx': -2, 'dy': 0})
        self.assertEqual([r['gt_index'] for r in truth], [1, 2])
        self.assertTrue(all(r['fully_retained'] for r in truth))
        self.assertEqual(truth[0]['bbox_xyxy'], [1., 1., 5., 5.])
        truth = p.mapped_truth(frame, {'op': 'translate_crop', 'dx': -1, 'dy': 0})
        self.assertEqual(truth[0]['bbox_xyxy'], [0., 0., 1., 2.])
        self.assertFalse(truth[0]['fully_retained'])

    def test_spec_sizes_use_original_non_square_shape(self):
        specs, _pairs = p.variant_specs(example_frame(width=65, height=49))
        translations = [(s['dx'], s['dy']) for s in specs if s['op'] == 'translate_crop']
        self.assertEqual(translations, [(8, 0), (-8, 0), (0, 6), (0, -6)])


class StatisticsTests(unittest.TestCase):
    def test_matching_one_to_one_and_class_specific(self):
        truth = p.mapped_truth(example_frame(), {'op': 'identity'})
        items = [detection([10, 10, 30, 30], .99, 'person'), detection([10, 10, 30, 30], .8),
                 detection([10, 10, 30, 30], .9)]
        result = p.match(truth, items, 'person_face')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['prediction_index'], 2)

    def test_follows_perfect_shift(self):
        report = p.summarize(*synthetic_plan_and_predictions())
        row = report['aggregates']['person_face']
        self.assertEqual(row['following_fraction_given_original_matched'], 1.)
        self.assertEqual(row['mean_residual_over_translation_when_both_matched'], 0.)

    def test_stationary_overlap_is_not_following(self):
        report = p.summarize(*synthetic_plan_and_predictions(stationary=True))
        row = report['aggregates']['person_face']
        self.assertEqual(row['shifted_matched_fraction_all_retained'], 1.)
        self.assertEqual(row['following_fraction_given_original_matched'], 0.)
        self.assertEqual(row['mean_residual_over_translation_when_both_matched'], 1.)

    def test_empty_detections_no_vacuous_success(self):
        report = p.summarize(*synthetic_plan_and_predictions(empty=True))
        row = report['aggregates']['person_face']
        self.assertIsNone(row['following_fraction_given_original_matched'])
        self.assertEqual(row['shifted_matched_fraction_all_retained'], 0.)
        self.assertIsNone(row['mean_extra_target_drop_originally_supported'])

    def test_occlusion_support_is_continuous_not_success_recall(self):
        report = p.summarize(*synthetic_plan_and_predictions())
        pair = report['occlusion_pairs'][0]
        self.assertAlmostEqual(pair['extra_drop_after_target_erasure'], .7)
        self.assertAlmostEqual(pair['real_minus_background'], .1)
        self.assertEqual(pair['target_erased_support'], .1)
        self.assertFalse(report['pass_fail_gate_defined'])
        self.assertFalse(report['ready_to_resume_s5_training'])

    def test_missing_or_duplicate_predictions_rejected(self):
        plan, prediction = synthetic_plan_and_predictions()
        for replacement in [prediction['records'][:-1], prediction['records']+[prediction['records'][0]]]:
            changed = {**prediction, 'records': replacement}
            with self.assertRaises(ValueError):
                p.summarize(plan, changed)

    def test_wrong_pixels_or_split_rejected(self):
        plan, prediction = synthetic_plan_and_predictions()
        for change in ['pixels', 'split']:
            changed = copy.deepcopy(prediction)
            if change == 'pixels':
                changed['records'][0]['input_pixel_sha256'] = 'wrong'
            else:
                changed['split'] = 'test'
            with self.assertRaises(ValueError):
                p.summarize(plan, changed)

    def test_adapter_filter_and_clipping_before_matching(self):
        result = p.canonical_detections([detection([-1, -2, 80, 60]),
            detection([1, 1, 2, 2], cls='person'), detection([90, 90, 100, 100])], 64, 48, ['person_face'])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['bbox_xyxy'], [0., 0., 64., 48.])


class S5PreprocessingTests(unittest.TestCase):
    def test_adapter_preprocessing_equals_existing_on_synthetic_png(self):
        from PIL import Image
        from paired_s5_baseline import prepare_rgb
        from train_recognition import load_image
        rng = np.random.default_rng(27)
        rgb = rng.integers(0, 256, (17, 29, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            Image.fromarray(rgb).save(root/'synthetic.png')
            expected = load_image(root, {'path': 'synthetic.png', 'width': 29, 'height': 17})
            np.testing.assert_array_equal(prepare_rgb(rgb).numpy(), expected.numpy())
        self.assertEqual(float(prepare_rgb(np.full((7, 11, 3), 117, np.uint8)).abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main(verbosity=2)
