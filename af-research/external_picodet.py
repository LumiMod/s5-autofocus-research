#!/usr/bin/env python3
"""Official exported PicoDet inference, with original-image continuous boxes.

No optimizer, dataset access, or S5 weight conversion. The packaged graph owns
box decoding and NMS. YAML NMS values may be stale; record actual graph attrs.
"""
import hashlib
from pathlib import Path

import cv2
import numpy as np
import paddle
from paddle.base.proto import framework_pb2
from paddle.inference import Config, create_predictor
import yaml


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PicoDetDetector:
    def __init__(self, model_dir):
        model_dir = Path(model_dir)
        if not (model_dir / 'infer_cfg.yml').is_file():
            candidates = list(model_dir.glob('*/infer_cfg.yml'))
            if len(candidates) != 1:
                raise ValueError('Expected one exported PicoDet inference directory')
            model_dir = candidates[0].parent
        self.model_dir = model_dir
        self.cfg = yaml.safe_load((model_dir / 'infer_cfg.yml').read_text())
        ops = self.cfg['Preprocess']
        if [op['type'] for op in ops] != ['Resize', 'NormalizeImage', 'Permute']:
            raise ValueError('Unsupported official preprocessing sequence')
        resize, normalization = ops[:2]
        if resize['keep_ratio'] or resize['target_size'] != [320, 320]:
            raise ValueError('This frozen baseline requires the official S320 input')
        self.height, self.width = resize['target_size']
        # Match official NormalizeImage's float64 coefficient arrays with
        # in-place float32 outputs, including its rounding sequence.
        self.mean = np.asarray(normalization['mean']).reshape(1, 1, 3)
        self.std = np.asarray(normalization['std']).reshape(1, 1, 3)
        self.person_index = self.cfg['label_list'].index('person')
        program = framework_pb2.ProgramDesc()
        program.ParseFromString((model_dir / 'model.pdmodel').read_bytes())
        nms = [op for block in program.blocks for op in block.ops if op.type == 'multiclass_nms3']
        if len(nms) != 1:
            raise ValueError('Expected one graph-owned multiclass NMS')
        graph_nms = {}
        for attr in nms[0].attrs:
            if attr.name in ('score_threshold', 'nms_threshold'):
                graph_nms[attr.name] = attr.f
            elif attr.name in ('nms_top_k', 'keep_top_k', 'background_label'):
                graph_nms[attr.name] = attr.i
            elif attr.name == 'normalized':
                graph_nms[attr.name] = attr.b
        config = Config(str(model_dir / 'model.pdmodel'), str(model_dir / 'model.pdiparams'))
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(1)
        config.disable_glog_info()
        self.predictor = create_predictor(config)
        if set(self.predictor.get_input_names()) != {'image', 'scale_factor'}:
            raise ValueError('Unexpected inference inputs')
        self.metadata = {
            'model_id': 'official_picodet_s_320_coco_lcnet',
            'runtime': {'paddle': paddle.__version__, 'numpy': np.__version__,
                        'opencv': cv2.__version__, 'device': 'CPU', 'threads': 1},
            'model_file_hashes': {name: file_hash(model_dir / name) for name in
                                 ('model.pdmodel', 'model.pdiparams', 'infer_cfg.yml')},
            'official_source_urls': [
                'https://paddledet.bj.bcebos.com/deploy/Inference/picodet_s_320_coco_lcnet.tar',
                'https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/deploy/python/preprocess.py',
                'https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/deploy/python/infer.py'],
            'input_transform': {'height': self.height, 'width': self.width, 'keep_ratio': False,
                                'resize_policy': 'stretch', 'color_order': 'RGB',
                                'interp': resize['interp'], **{k: normalization[k] for k in
                                                             ('mean', 'std', 'is_scale')}},
            'supported_evaluation_class': 'person', 'model_person_index': self.person_index,
            'graph_nms': graph_nms, 'packaged_yaml_nms_for_comparison_only': self.cfg.get('NMS'),
            'output_coordinate_space': 'original_image_continuous_xyxy',
            'coordinate_note': 'Graph applies scale_factor once; no adapter-side inverse resize.',
            'postprocessing': 'Keep all graph-emitted person scores; clip to original frame; no extra score cutoff.',
            'limits': 'Graph NMS retains at most 100 detections across all COCO classes and filters scores below its threshold. No S5 compatibility claim.'}

    def prepare(self, rgb):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('Expected uint8 HWC RGB')
        h, w = rgb.shape[:2]
        sy, sx = self.height / h, self.width / w
        image = cv2.resize(rgb, None, fx=sx, fy=sy, interpolation=self.cfg['Preprocess'][0]['interp'])
        image = image.astype(np.float32, copy=False)
        if self.cfg['Preprocess'][1]['is_scale']:
            image *= 1.0 / 255.0
        image -= self.mean
        image /= self.std
        return {'image': np.ascontiguousarray(image.transpose(2, 0, 1)[None]),
                'scale_factor': np.asarray([[sy, sx]], dtype=np.float32)}

    def predict_raw(self, feeds):
        for name, value in feeds.items():
            self.predictor.get_input_handle(name).copy_from_cpu(value)
        self.predictor.run()
        outputs = [self.predictor.get_output_handle(name).copy_to_cpu()
                   for name in self.predictor.get_output_names()]
        boxes = [value for value in outputs if value.ndim == 2 and value.shape[1] == 6]
        if len(boxes) != 1 or not np.isfinite(boxes[0]).all():
            raise ValueError('Malformed exported detections')
        return boxes[0]

    def predict_rgb(self, rgb):
        raw = self.predict_raw(self.prepare(rgb))
        h, w = rgb.shape[:2]
        results = []
        for row in raw:
            if row[0] != self.person_index:
                continue
            if not 0 <= row[1] <= 1:
                raise ValueError('Invalid score')
            box = row[2:].astype(np.float64)
            box[[0, 2]] = box[[0, 2]].clip(0, w)
            box[[1, 3]] = box[[1, 3]].clip(0, h)
            if box[2] > box[0] and box[3] > box[1]:
                results.append({'class': 'person', 'score': float(row[1]), 'bbox_xyxy': box.tolist()})
        return results

