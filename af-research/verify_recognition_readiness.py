#!/usr/bin/env python3
"""Synthetic contract tests, never recognition-performance or camera evidence."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from pretraining_data import sha256
from pretraining_readiness import MODELS, readiness, validate_recognition_dataset


class RecognitionReadinessTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='s5-recognition-contract-')
        self.base = Path(self.temp.name)
        self.path = self.base / 'manifest.json'
        provenance = self.base / 'provenance.json'
        provenance.write_text(json.dumps({'fixture_only': True, 'not_real_training_data': True}))
        rng = np.random.default_rng(7201)
        frames = []
        for index, split in enumerate(('train', 'validation', 'test')):
            image_path = self.base / f'{index}.png'
            pixels = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(image_path)
            frames.append({'id': str(index), 'path': image_path.name, 'sha256': sha256(image_path),
                           'width': 32, 'height': 32, 'split': split,
                           'source_kind': 'public_dataset', 'source_dataset': 'unit-test-fixture-only',
                           'source_image_id': str(index), 'group_id': str(index),
                           'boxes': [{'class': 'person', 'bbox_xyxy': [1, 1, 30, 30]},
                                     {'class': 'person_face', 'bbox_xyxy': [10, 2, 20, 12]}]})
        self.data = {'schema_version': 1, 'task': 'recognition_detection',
                     'classes': ['person', 'person_face'], 'frames': frames,
                     'provenance': {'path': provenance.name, 'sha256': sha256(provenance)}}
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        self.path.write_text(json.dumps(self.data))

    def test_static_image_contract_requires_no_invented_camera_fields(self):
        report, _ = validate_recognition_dataset(self.path)
        self.assertEqual(report['data_gaps'], [])
        self.assertTrue(report['file_content_verified'])
        self.assertEqual(report['split_counts'], {'train': 1, 'validation': 1, 'test': 1})
        self.assertIn('no_empty_background_images_in_test', report['warnings'])

    def test_candidate_scope_does_not_claim_camera_parity(self):
        hashes = {}
        for stem in MODELS:
            source = self.base / (stem + '.bin')
            source.write_bytes(('fixture-only-' + stem).encode())
            hashes[stem] = sha256(source)
        with patch.dict(MODELS, hashes, clear=True):
            report = readiness(self.base, recognition_dataset_path=self.path)
        self.assertTrue(report['ready_for_computer_recognition_experiment'])
        self.assertFalse(report['ready_for_formal_training'])
        self.assertFalse(report['ready_for_original_camera_fidelity_and_deployment'])
        self.assertFalse(report['gates']['numeric_original_execution_comparison'])
        self.assertFalse(report['gates']['image_to_input_preprocessing_fidelity'])
        self.assertFalse(report['gates']['proposal_decoder_fidelity'])
        self.assertFalse(report['gates']['physical_focus_control_baseline'])

    def test_tools_only_cannot_satisfy_source_identity(self):
        report = readiness(None, recognition_dataset_path=self.path)
        self.assertFalse(report['ready_for_computer_recognition_experiment'])
        self.assertEqual(report['scopes']['computer_recognition_experiment']['outstanding'], ['selected_source_identity'])

    def test_only_selected_original_model_is_needed_for_candidate_scope(self):
        source = self.base / '44_hm_d_nw_sng.bin'
        source.write_bytes(b'fixture-only-selected-model')
        with patch.dict(MODELS, {'44_hm_d_nw_sng': sha256(source)}):
            report = readiness(self.base, recognition_dataset_path=self.path)
        self.assertTrue(report['ready_for_computer_recognition_experiment'])
        self.assertFalse(report['gates']['source_identities'])

    def test_source_group_leak_is_rejected(self):
        self.data['frames'][1]['group_id'] = self.data['frames'][0]['group_id']
        self.save()
        with self.assertRaisesRegex(ValueError, 'group leakage'):
            validate_recognition_dataset(self.path)

    def test_identical_content_leak_is_rejected(self):
        self.data['frames'][1]['sha256'] = self.data['frames'][0]['sha256']
        self.save()
        with self.assertRaisesRegex(ValueError, 'Identical recognition image leakage'):
            validate_recognition_dataset(self.path)

    def test_decoded_pixel_leak_is_rejected(self):
        with Image.open(self.base / '0.png') as image:
            image.save(self.base / '1.png', compress_level=0)
        self.data['frames'][1]['sha256'] = sha256(self.base / '1.png')
        self.save()
        with self.assertRaisesRegex(ValueError, 'Decoded-pixel leakage'):
            validate_recognition_dataset(self.path)

    def test_near_duplicate_leak_is_rejected(self):
        with Image.open(self.base / '0.png') as image:
            pixels = np.asarray(image).copy()
        pixels[0, 0, 0] = (int(pixels[0, 0, 0]) + 1) % 256
        Image.fromarray(pixels).save(self.base / '1.png')
        self.data['frames'][1]['sha256'] = sha256(self.base / '1.png')
        self.save()
        with self.assertRaisesRegex(ValueError, 'near-duplicate leakage'):
            validate_recognition_dataset(self.path)

    def test_bad_box_is_rejected(self):
        self.data['frames'][0]['boxes'][0]['bbox_xyxy'] = [0, 0, 99, 99]
        self.save()
        with self.assertRaisesRegex(ValueError, 'Box outside'):
            validate_recognition_dataset(self.path)

    def test_missing_test_is_an_explicit_gap(self):
        self.data['frames'].pop()
        self.save()
        report, _ = validate_recognition_dataset(self.path)
        self.assertIn('missing_test_images', report['data_gaps'])

    def test_synthetic_data_kind_is_rejected(self):
        self.data['frames'][0]['source_kind'] = 'synthetic'
        self.save()
        with self.assertRaisesRegex(ValueError, 'real-image provenance'):
            validate_recognition_dataset(self.path)

    def test_provenance_hash_is_verified(self):
        (self.base / 'provenance.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Hash mismatch'):
            validate_recognition_dataset(self.path)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RecognitionReadinessTest)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    base = Path(__file__).resolve().parent
    report = {'fixture_only': True, 'tests_run': result.testsRun, 'failures': len(result.failures),
              'errors': len(result.errors), 'successful': result.wasSuccessful(),
              'source_hashes': {name: sha256(base / name) for name in
                                ('pretraining_readiness.py', 'verify_recognition_readiness.py')}}
    (base / 'artifacts').mkdir(exist_ok=True)
    (base / 'artifacts' / 'recognition-readiness-tests.json').write_text(json.dumps(report, indent=2) + '\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
