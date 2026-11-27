#!/usr/bin/env python3
"""Verify the PicoDet bridge against official preprocessing and graph geometry.

Only the first sorted training image is opened, never validation or test.
This is a bridge check, not detection-performance evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from external_picodet import PicoDetDetector, file_hash

BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=BASE/'private-data/recognition-v2/manifest.json')
    parser.add_argument('--model-dir', type=Path, default=BASE/'artifacts/external-baselines/models/picodet')
    parser.add_argument('--vendor', type=Path, default=BASE/'artifacts/external-baselines/vendor/PaddleDetection')
    parser.add_argument('--output', type=Path, default=BASE/'artifacts/external-baselines/picodet-adapter-check.json')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Refusing to overwrite frozen checks')
    detector = PicoDetDetector(args.model_dir)
    sys.path.insert(0, str(args.vendor/'deploy/python'))
    import preprocess as official
    pattern = (np.arange(173*349*3, dtype=np.uint32).reshape(173, 349, 3)*17 % 256).astype(np.uint8)
    operators = [getattr(official, op['type'])(**{k: v for k, v in op.items() if k != 'type'})
                 for op in detector.cfg['Preprocess']]
    expected, information = official.preprocess(pattern, operators)
    feeds = detector.prepare(pattern)
    assert np.array_equal(feeds['image'][0], expected), 'Official tensor mismatch'
    assert np.array_equal(feeds['scale_factor'][0], information['scale_factor']), 'Official scale mismatch'
    assert detector.person_index == 0 and detector.cfg['label_list'][0] == 'person'
    nms = detector.metadata['graph_nms']
    assert abs(nms['score_threshold']-.025) < 1e-7 and abs(nms['nms_threshold']-.6) < 1e-7
    data = json.loads(args.dataset.read_text())
    frame = sorted((f for f in data['frames'] if f['split'] == 'train'), key=lambda f: f['id'])[0]
    path = args.dataset.parent/frame['path']
    assert file_hash(path) == frame['sha256']
    with Image.open(path) as image:
        rgb = np.asarray(image.convert('RGB'))
    assert rgb.shape[:2] == (frame['height'], frame['width'])
    real_feeds = detector.prepare(rgb)
    actual = detector.predict_raw(real_feeds)
    unit = detector.predict_raw({**real_feeds, 'scale_factor': np.ones((1, 2), dtype=np.float32)})
    assert len(actual) > 0 and actual.shape == unit.shape, 'No boxes to check output geometry'
    assert np.array_equal(actual[:, :2], unit[:, :2]), 'Class/score affected by output transform'
    expected_boxes = unit[:, 2:] / real_feeds['scale_factor'][0][[1, 0, 1, 0]]
    coordinate_error = float(np.abs(actual[:, 2:]-expected_boxes).max())
    assert coordinate_error < .002, 'Graph original-photo coordinate mapping mismatch'
    detections = detector.predict_rgb(rgb)
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox_xyxy']
        assert 0 <= x1 < x2 <= frame['width'] and 0 <= y1 < y2 <= frame['height']
        assert detection['class'] == 'person' and 0 <= detection['score'] <= 1
    report = {'passed': True, 'scope': 'Official preprocessing parity and graph scale-factor geometry; not performance acceptance.',
              'train_image_id': frame['id'], 'train_image_sha256': frame['sha256'],
              'model_metadata': detector.metadata, 'raw_detections': len(actual),
              'person_detections': len(detections), 'max_coordinate_error_pixels': coordinate_error,
              'official_preprocess_tensor_exact': True, 'official_scale_factor_exact': True,
              'optimizer_steps': 0, 'validation_inference': False, 'test_inference': False,
              'source_hashes': {p.name: file_hash(p) for p in (Path(__file__), BASE/'external_picodet.py',
                                args.vendor/'deploy/python/preprocess.py', args.vendor/'deploy/python/keypoint_preprocess.py')}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('passed','train_image_id','raw_detections','person_detections','max_coordinate_error_pixels')}))


if __name__ == '__main__':
    main()
