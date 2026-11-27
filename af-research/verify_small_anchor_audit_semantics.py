#!/usr/bin/env python3
"""Synthetic audit-logic fixtures, not training or recognition evidence."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import verify_small_anchor_run as v


def metric(person=.65, face=.6, small=17):
    return {'classes': {'person': {'ap_at_iou': person}, 'person_face': {'ap_at_iou': face}},
            'mean_ap': (person+face)/2,
            'small_faces': {'ground_truth': 34, 'matched_at_score': small, 'recall_at_score': small/34}}


class AcceptanceTests(unittest.TestCase):
    def test_fixed_absolute_face_floor_and_17_of34(self):
        original, final = metric(.6, .34, 0), metric()
        controls = {name: metric(.2, .1, 0) for name in v.CONTROLS}
        self.assertTrue(v.acceptance(final, original, controls)['passed'])
        self.assertFalse(v.acceptance(metric(small=16), original, controls)['passed'])
        self.assertFalse(v.acceptance(metric(face=.49), original, controls)['passed'])

    def test_all_controls_required_and_quant_can_fail_independently(self):
        original, final = metric(.6, .34, 0), metric()
        controls = {name: metric(.2, .1, 0) for name in v.CONTROLS}
        controls['shuffled'] = metric(.51, .1, 0)
        self.assertFalse(v.acceptance(final, original, controls)['passed'])
        with self.assertRaises(ValueError):
            v.acceptance(final, original, {'real': final})


class MetricTests(unittest.TestCase):
    def test_only_tiny_boundary_overshoot_can_be_repaired(self):
        frames = {'f': {'width': 1024, 'height': 768}}
        doc = {'model_state_sha256': 'frozen', 'frames': [{'id': 'f', 'detections': [
            {'class': 'person_face', 'score': .9, 'bbox_xyxy': [0., 0., 20., 768.0000000000001]}]}]}
        fixed, changes, maximum = v.repair_boundary_document(doc, frames)
        self.assertEqual(fixed['frames'][0]['detections'][0]['bbox_xyxy'][-1], 768.)
        self.assertEqual(fixed['frames'][0]['detections'][0]['score'], .9)
        self.assertEqual(doc['frames'][0]['detections'][0]['bbox_xyxy'][-1], 768.0000000000001)
        self.assertEqual(len(changes), 1)
        self.assertLess(maximum, 1e-9)
        doc['frames'][0]['detections'][0]['bbox_xyxy'][-1] = 768.001
        with self.assertRaises(ValueError):
            v.repair_boundary_document(doc, frames)

    def test_all_gt_matched_before_small_stratification(self):
        frame = {'id': 'synthetic', 'split': 'validation', 'width': 64, 'height': 64,
            'boxes': [{'class': 'person', 'bbox_xyxy': [0., 0., 32., 64.]},
                      {'class': 'person_face', 'bbox_xyxy': [20., 20., 24., 24.]},
                      {'class': 'person_face', 'bbox_xyxy': [0., 0., 40., 40.]}]}
        predictions = [{'class': b['class'], 'bbox_xyxy': b['bbox_xyxy'], 'score': .9-i*.1} for i, b in enumerate(frame['boxes'])]
        predictions.append({**predictions[1], 'score': .55})
        doc = {'dataset_manifest_sha256': 'fixture', 'frames': [{'id': 'synthetic', 'detections': predictions}]}
        result = v.recompute_metrics({'frames': [frame]}, doc, 'fixture')
        self.assertEqual(result['small_faces']['ground_truth'], 1)
        self.assertEqual(result['small_faces']['matched_at_score'], 1)
        self.assertEqual(result['classes']['person_face']['tp_at_score'], 2)
        self.assertEqual(result['classes']['person_face']['fp_at_score'], 1)
        self.assertEqual(result['classes']['person_face']['ap_at_iou'], 1.)

    def test_duplicate_prediction_frame_is_rejected(self):
        frame = {'id': 'synthetic', 'split': 'validation', 'width': 64, 'height': 64, 'boxes': []}
        doc = {'dataset_manifest_sha256': 'fixture', 'frames': [{'id': 'synthetic', 'detections': []}]*2}
        with self.assertRaises(ValueError):
            v.recompute_metrics({'frames': [frame]}, doc, 'fixture')


class StateTests(unittest.TestCase):
    def setUp(self):
        self.original = {name: torch.zeros(2) for name in v.HEAD_NAMES}
        self.original['convs.1.weight'] = torch.ones(2)

    def test_epoch0_and_head_only_change(self):
        self.assertFalse(v.audit_state(self.original, self.original)['changed_head_tensors'])
        changed = copy.deepcopy(self.original); changed[v.HEAD_NAMES[0]][0] = .1
        self.assertEqual(v.audit_state(self.original, changed, True)['changed_head_tensors'], [v.HEAD_NAMES[0]])

    def test_backbone_change_and_absent_update_rejected(self):
        changed = copy.deepcopy(self.original); changed['convs.1.weight'][0] += 1
        with self.assertRaises(ValueError):
            v.audit_state(self.original, changed)
        with self.assertRaises(ValueError):
            v.audit_state(self.original, self.original, True)

    def test_optimizer_steps_and_moments(self):
        optimizer = {'param_groups': [{'params': list(range(6)), 'lr': .0001, 'weight_decay': .0001}],
                     'state': {i: {'step': torch.tensor(80.), 'exp_avg': torch.ones(2), 'exp_avg_sq': torch.ones(2)} for i in range(6)}}
        v.audit_optimizer(optimizer, v.HEAD_NAMES, self.original, 80)
        optimizer['state'][3]['step'] += 1
        with self.assertRaises(ValueError):
            v.audit_optimizer(optimizer, v.HEAD_NAMES, self.original, 80)
        optimizer['state'] = {}
        v.audit_optimizer(optimizer, v.HEAD_NAMES, self.original, 0)

    def test_quantization_round_even_clip_and_backbone_unchanged(self):
        records = [{'index': i, 'weight_scale': .125, 'bias_scale': .25} for i in (86, 90, 94)]
        state = {name: torch.tensor([.0625, .1875]) if name.endswith('weight') else torch.tensor([.125, .375]) for name in v.HEAD_NAMES}
        state['convs.1.weight'] = torch.tensor([123.])
        state['convs.94.weight'] = torch.tensor([4096., -4096.])
        with patch('decode_parameters.recover', return_value=({'convolutions': records}, {})):
            quant, evidence = v.expected_quantized_state(state, b'fixture')
        self.assertEqual(quant['convs.86.weight'].tolist(), [0., .25])
        self.assertEqual(quant['convs.86.bias'].tolist(), [0., .5])
        self.assertEqual(evidence['convs.94.weight']['clipped_elements'], 1)
        self.assertEqual(quant['convs.94.weight'].tolist(), [4095.875, -4096.])
        self.assertTrue(torch.equal(quant['convs.1.weight'], state['convs.1.weight']))


class LogTests(unittest.TestCase):
    def test_complete_logs_then_reject_duplicate_and_zero_gradient(self):
        names = [f'synthetic-{i:03}' for i in range(320)]
        expected = {name: 'owner-'+name for name in names}
        images = [{'epoch': epoch, 'frame_id': name, 'image_index': i,
                   'optimizer_steps': (epoch-1)*80+(i+1)//4, 'target_owner_sha256': expected[name],
                   'unmatched_gt_count': 0, 'rng_before_loss': {}, 'loss': 1.6,
                   'person_ce': .5, 'person_face_ce': .8, 'bbox_smooth_l1': .3}
                  for epoch in range(1, 7) for i, name in enumerate(names)]
        steps = [{'epoch': (i-1)//80+1, 'global_step': i, 'grad_norm': .5,
                  'parameter_names_with_gradient': list(v.HEAD_NAMES),
                  'head_delta_l2': {n: .01 for n in v.HEAD_NAMES}, 'backbone_sha256': 'frozen'} for i in range(1, 481)]
        plan = {'train_ids': names, 'initial_backbone_sha256': 'frozen'}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            def save():
                (root/'training.jsonl').write_text('\n'.join(map(json.dumps, images))+'\n')
                (root/'steps.jsonl').write_text('\n'.join(map(json.dumps, steps))+'\n')
            save(); self.assertEqual(v.audit_training_logs(root, plan, expected)['optimizer_steps'], 480)
            images[0]['frame_id'] = images[1]['frame_id']; save()
            with self.assertRaises(ValueError):
                v.audit_training_logs(root, plan, expected)
            images[0]['frame_id'] = names[0]; steps[0]['grad_norm'] = 0.; save()
            with self.assertRaises(ValueError):
                v.audit_training_logs(root, plan, expected)


class TransferTests(unittest.TestCase):
    def fixture(self):
        old_scales, new_scales = [3., 6., 9., 16., 32.], [1., 2., 4., 8., 16.]
        choice = [0, 0, 0, 2, 3]
        slots = [r*5+j for r in range(5) for j in choice]
        def boxes(scales):
            values = []
            for y in range(16):
                for x in range(24):
                    cx, cy = 8+16*x, 8+16*y
                    for ratio in range(5):
                        for scale in scales:
                            w, h = (2+ratio)*scale, (7-ratio)*scale
                            values.append([cx-w/2, cy-h/2, cx+w/2, cy+h/2])
            return np.array(values, np.float32)
        gains, offsets = [], []
        for _ratio in range(5):
            for j, scale in enumerate(new_scales):
                gain = old_scales[choice[j]]/scale
                gains.append([gain, gain, 1., 1.]); offsets.append([0., 0., float(np.log(gain)), float(np.log(gain))])
        mapped = np.asarray([row//25*25+slots[row % 25] for row in range(9600)], dtype='<i8')
        provenance = {'original_scales': old_scales, 'new_scales': new_scales,
            'old_scale_index_by_new_scale': choice, 'old_slot_by_new_slot': slots,
            'class_bg_fg_channel_map': slots+[s+25 for s in slots],
            'bbox_xywh_channel_map': [s*4+j for s in slots for j in range(4)],
            'bbox_gain_by_slot': gains, 'bbox_offset_by_slot': offsets,
            'row_mapping_sha256': hashlib.sha256(mapped.tobytes()).hexdigest()}
        state = {'convs.1.weight': torch.ones(3)}
        for index, outputs in [(86, 50), (90, 50), (94, 100)]:
            state[f'convs.{index}.weight'] = torch.arange(outputs*2, dtype=torch.float32).reshape(outputs, 2, 1, 1)
            state[f'convs.{index}.bias'] = torch.arange(outputs, dtype=torch.float32)
        return state, provenance, boxes(old_scales), boxes(new_scales)

    def test_class_halves_and_bbox_affine_are_separate(self):
        original, provenance, old, new = self.fixture()
        before = v.state_sha(original)
        initial, report = v.reconstruct_analytic_transfer(original, provenance, old, new)
        self.assertEqual(v.state_sha(original), before)
        self.assertTrue(torch.equal(initial['convs.86.weight'][3], original['convs.86.weight'][2]))
        self.assertEqual(float(initial['convs.90.bias'][28]), 27.)
        self.assertTrue(torch.equal(initial['convs.94.weight'][0], original['convs.94.weight'][0]*3))
        self.assertAlmostEqual(float(initial['convs.94.bias'][2]), 2+np.log(3), places=6)
        self.assertTrue(torch.equal(initial['convs.94.weight'][2], original['convs.94.weight'][2]))
        self.assertTrue(report['backbone_unchanged'])

    def test_wrong_slot_offset_and_center_are_rejected(self):
        original, provenance, old, new = self.fixture()
        for mode in ('slot', 'offset', 'class_half', 'center'):
            changed = copy.deepcopy(provenance); new_boxes = new.copy()
            if mode == 'slot': changed['old_slot_by_new_slot'][0] = 5
            if mode == 'offset': changed['bbox_offset_by_slot'][0][2] += .1
            if mode == 'class_half': changed['class_bg_fg_channel_map'][0] = 25
            if mode == 'center': new_boxes[:, [0, 2]] += 1
            with self.assertRaises(ValueError):
                v.reconstruct_analytic_transfer(original, changed, old, new_boxes)


if __name__ == '__main__':
    unittest.main(verbosity=2)
