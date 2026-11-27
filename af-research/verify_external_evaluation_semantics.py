#!/usr/bin/env python3
"""Numerical fixtures only: no model, real photos, training or recognition claim."""
import copy
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace

import numpy as np
from PIL import Image

from evaluate_external_baselines import (canonical_detections, image_control, map_box_between_images,
    map_predictions_between_images, match_and_stratify, reference_sizes, restore_model_xyxy, screening, run,
    loaded_project_sources, BASE)
from pretraining_data import evaluate_detection


def main():
    checks = []
    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)
    check('non_square_stretch_inverse', np.allclose(restore_model_xyxy([32, 64, 160, 256], (1000, 500), (320, 320)), [100, 100, 500, 400]))
    check('letterbox_padding_inverse', np.allclose(restore_model_xyxy([80, 200, 400, 360], (800, 400), (640, 320), (0, 160)), [100, 50, 500, 250]))
    check('inverse_box_clips_padding', np.allclose(restore_model_xyxy([-10, 100, 650, 600], (800, 400), (640, 320), (0, 160)), [0, 0, 800, 400]))
    check('wrong_non_square_normalized_map', np.allclose(map_box_between_images([100, 50, 900, 450], (1000, 500), (400, 1000)), [40, 100, 360, 900]))
    original = [{'class': 'person_face', 'score': .7, 'bbox_xyxy': [100, 50, 900, 450]}]
    mapped = map_predictions_between_images(original, (1000, 500), (400, 1000))
    check('wrong_map_preserves_score_class_without_mutating', mapped[0]['class'] == 'person_face' and mapped[0]['score'] == .7 and original[0]['bbox_xyxy'] == [100, 50, 900, 450])
    detections = [{'class': 'person', 'score': .4, 'bbox_xyxy': [-1, 0, 50, 50]},
                  {'class': 'person_face', 'score': .9, 'bbox_xyxy': [10, 10, 20, 20]},
                  {'class': 'dog', 'score': .8, 'bbox_xyxy': [2, 2, 9, 9]},
                  {'class': 'person', 'score': .1, 'bbox_xyxy': [60, 0, 80, 20]}]
    kept, raw, counts = canonical_detections(detections, 50, 50, 'person')
    check('class_filter_and_no_extra_score_floor', len(kept) == 1 and kept[0]['score'] == .4 and kept[0]['bbox_xyxy'] == [0., 0., 50., 50.] and len(raw) == 4 and counts['other_class_filtered'] == 2 and counts['degenerate_target_boxes_removed'] == 1)
    try:
        canonical_detections([{'class': 'person', 'score': float('nan'), 'bbox_xyxy': [0, 0, 2, 2]}], 10, 10, 'person')
    except ValueError:
        check('nonfinite_score_rejected', True)
    else:
        check('nonfinite_score_rejected', False)
    rgb = np.array([[[0, 100, 200], [20, 120, 220]]], np.uint8)
    check('RGB_channel_mean_control', np.array_equal(image_control(rgb, 'same_image_RGB_channel_mean'), [[[10, 110, 210], [10, 110, 210]]]))
    check('fixed_gray117_control', np.array_equal(image_control(rgb, 'fixed_gray117'), np.full_like(rgb, 117)))
    real = image_control(rgb, 'real'); real[0, 0, 0] = 99
    check('real_control_preserves_original', rgb[0, 0, 0] == 0)
    frame = {'id': 'fixture', 'split': 'validation', 'width': 384, 'height': 256,
             'boxes': [{'class': 'person_face', 'bbox_xyxy': [10, 10, 44, 44]},
                       {'class': 'person_face', 'bbox_xyxy': [12, 12, 42, 42]},
                       {'class': 'person', 'bbox_xyxy': [0, 0, 100, 200]}]}
    metadata = {'input_transform': {'width': 640, 'height': 640, 'keep_ratio': False}}
    prediction = {'dataset_manifest_sha256': 'synthetic-fixture-not-real-data',
                  'frames': [{'id': 'fixture', 'detections': [{'class': 'person_face', 'score': .9, 'bbox_xyxy': [10, 10, 44, 44]}]}]}
    result = match_and_stratify([frame], prediction, 'person_face', metadata)
    sizes = result['size_recall']['reference_s5_384x256']
    check('match_large_GT_before_stratifying_small', sizes['small_lt32']['ground_truth_boxes'] == 1 and sizes['small_lt32']['matched_gt'] == 0 and sizes['medium_32_to_lt96']['matched_gt'] == 1)
    prediction2 = copy.deepcopy(prediction)
    prediction2['frames'][0]['detections'] += [{'class': 'person_face', 'score': .8, 'bbox_xyxy': [12, 12, 42, 42]},
                                              {'class': 'person_face', 'score': .7, 'bbox_xyxy': [12, 12, 42, 42]}]
    result2 = match_and_stratify([frame], prediction2, 'person_face', metadata)
    overall = evaluate_detection({'classes': ['person_face'], 'frames': [frame]}, prediction2, 'validation')
    check('one_to_one_duplicates_and_metric_parity', result2['tp'] == 2 and result2['fp'] == 1 and result2['fn'] == 0 and result2['tp'] == overall['classes']['person_face']['tp_at_score'])
    low = copy.deepcopy(prediction2); low['frames'][0]['detections'][1]['score'] = .49; low['frames'][0]['detections'] = low['frames'][0]['detections'][:2]
    result3 = match_and_stratify([frame], low, 'person_face', metadata)
    check('recall_score_threshold_does_not_remove_GT', result3['tp'] == 1 and result3['fn'] == 1 and result3['size_recall']['reference_s5_384x256']['small_lt32']['ground_truth_boxes'] == 1)
    dimensions = reference_sizes([0, 0, 100, 100], {'width': 1000, 'height': 1000}, {'input_transform': {'width': 320, 'height': 320, 'keep_ratio': False}})
    check('reference640_distinct_from_actual320', math.isclose(dimensions['reference_s5_384x256'], math.sqrt(983.04)) and dimensions['reference_stretch_640x640'] == 64 and dimensions['actual_stretch_320x320'] == 32)
    unknown = reference_sizes([0, 0, 100, 100], {'width': 1000, 'height': 500}, {'input_transform': {'width': 640, 'height': 640, 'keep_ratio': True}})
    check('unknown_keep_ratio_not_mislabeled_actual', set(unknown) == {'reference_s5_384x256', 'reference_stretch_640x640'})
    native = reference_sizes([0, 0, 100, 100], {'width': 1000, 'height': 500}, {'input_transform': {'resize_policy': 'native'}})
    check('native_area_declared_explicitly', native['actual_native'] == 100 and 'actual_stretch_640x640' not in native)
    metrics = {k: {'classes': {'person_face': {'ap_at_iou': .65 if k == 'real' else .5}}} for k in ('real', 'same_image_RGB_channel_mean', 'fixed_gray117', 'cyclic_wrong')}
    strata = {'real': {'size_recall': {'reference_s5_384x256': {'small_lt32': {'ground_truth_boxes': 20, 'matched_gt': 10, 'recall': .5}}}}}
    check('predeclared_face_screen_boundary', screening(metrics, strata, 'person_face')['passed_necessary_external_baseline_screen'])
    passed = screening(metrics, strata, 'person_face')
    check('external_success_does_not_unlock_S5_training', passed['ready_for_external_baseline_use'] and not passed['ready_to_resume_s5_training'])
    strata['real']['size_recall']['reference_s5_384x256']['small_lt32'].update(ground_truth_boxes=10, matched_gt=9, recall=.9)
    check('minimum_ten_small_hits_required', not screening(metrics, strata, 'person_face')['passed_necessary_external_baseline_screen'])
    # Exercise the full persistence/freeze path with an explicitly artificial
    # adapter. No official model or real dataset is loaded by this test.
    with tempfile.TemporaryDirectory(prefix='s5-external-evaluation-fixture-') as temp:
        root = Path(temp)
        models = root/'fixture-model'; models.mkdir()
        (models/'fixture.bin').write_bytes(b'synthetic fixture identity; not model weights')
        model_sha = hashlib.sha256((models/'fixture.bin').read_bytes()).hexdigest()
        output = root/'output'
        frames = []
        for index, (width, height) in enumerate(((400, 200), (200, 400))):
            pixels = np.zeros((height, width, 3), np.uint8)
            pixels[:, :width//2] = [10+index, 100, 240]
            pixels[:, width//2:] = [240, 20+index, 100]
            path = root/f'fixture-{index}.png'; Image.fromarray(pixels).save(path)
            frames.append({'id': f'synthetic-{index}', 'split': 'validation', 'width': width, 'height': height,
                           'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                           'boxes': [{'class': 'person', 'bbox_xyxy': [width*.1, height*.2, width*.6, height*.8]}]})
        manifest = root/'manifest.json'
        manifest.write_text(json.dumps({'task': 'recognition_detection', 'schema_version': 1,
                                        'classes': ['person', 'person_face'], 'frames': frames}))
        fixture_module = ModuleType('_external_semantic_fixture_adapter')
        fixture_module.__file__ = __file__
        dependency_modules = []
        for index, directory in enumerate(('.venv', '.venv-paddle', 'venv', 'vendor/site-packages')):
            dependency = ModuleType(f'_external_fixture_dependency_{index}')
            dependency.__file__ = str(BASE/directory/'unavailable_fixture_dependency.py')
            sys.modules[dependency.__name__] = dependency
            dependency_modules.append(dependency)
        try:
            bound_sources = loaded_project_sources(fixture_module)
            check('runtime_dependency_directories_excluded_from_source_binding',
                  all('unavailable_fixture_dependency' not in name for name in bound_sources) and
                  'evaluate_external_baselines.py' in bound_sources and
                  'pretraining_data.py' in bound_sources)
        finally:
            for dependency in dependency_modules:
                del sys.modules[dependency.__name__]
        class FixtureDetector:
            metadata = {'model_id': 'SYNTHETIC_FIXTURE_NOT_A_RECOGNITION_MODEL',
                        'model_file_hashes': {'fixture.bin': model_sha},
                        'output_coordinate_space': 'original_image_continuous_xyxy',
                        'input_transform': {'height': 320, 'width': 320, 'keep_ratio': False}}
            def predict_rgb(self, rgb):
                if not (output/'frozen-plan.json').is_file():
                    raise AssertionError('Predict called before frozen plan existed')
                h, w = rgb.shape[:2]
                return [{'class': 'person', 'score': .9, 'bbox_xyxy': [w*.1, h*.2, w*.6, h*.8]}]
        fixture_module.load_detector = lambda model, model_dir: FixtureDetector()
        sys.modules[fixture_module.__name__] = fixture_module
        args = SimpleNamespace(model='picodet', output=output, model_dir=models, dataset=manifest,
                               expected_images=2, expected_manifest_sha=None,
                               adapter=fixture_module.__name__+':load_detector')
        with redirect_stdout(io.StringIO()):
            run(args)
        del sys.modules[fixture_module.__name__]
        saved = json.loads((output/'report.json').read_text())
        check('frozen_plan_precedes_every_fixture_predict_call', saved['model_predict_calls'] == 6)
        check('all_four_control_predictions_are_saved', len(list((output/'predictions').glob('*.json'))) == 4 and len(list((output/'raw-adapter-predictions').glob('*.json'))) == 4)
        check('persisted_run_hashes_and_no_training', saved['all_source_model_manifest_hashes_unchanged'] and saved['optimizer_steps'] == 0 and not saved['test_inference'] and not saved['ready_to_resume_s5_training'])
        check('perfect_location_prior_fails_external_screen', saved['metrics']['real']['classes']['person']['ap_at_iou'] == 1 and not saved['ready_for_external_baseline_use'])
    print(json.dumps({'passed': True, 'checks': checks, 'check_count': len(checks),
                      'real_model_inference': False, 'photographs_read': False, 'synthetic_fixture_images': 2, 'optimizer_steps': 0,
                      'recognition_quality_established': False}, indent=2))


if __name__ == '__main__':
    main()
