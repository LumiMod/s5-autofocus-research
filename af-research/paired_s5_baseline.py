"""Independent inference-only adapters for the two predeclared S5 hypotheses.

No core network, preprocessing, training or decoder source is changed. The K4
profile is a structural candidate, not proven firmware/CEVA equivalence.
"""
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from extract import MODELS, require
from pretraining_data import sha256
from train_recognition import Detector, MODEL, detections, state_digest


def prepare_rgb(rgb):
    require(isinstance(rgb, np.ndarray) and rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3,
            'Expected uint8 HWC RGB')
    gray = Image.fromarray(rgb).convert('L').resize((384, 256), Image.Resampling.BILINEAR)
    pixels = np.asarray(gray, np.float32).copy()-117.0
    return torch.from_numpy(pixels)[None, None]


class S5HypothesisDetector:
    def __init__(self, model, model_dir):
        require(model in ('s5_baseline_o4_i4_hw', 's5_flattened_k4'), 'Unknown fixed S5 profile')
        source = Path(model_dir)/(MODEL+'.bin')
        require(sha256(source) == MODELS[MODEL], 'Original model identity differs')
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(0)
        data = source.read_bytes()
        if model == 's5_flattened_k4':
            from flattened_k_diagnosis import build_detector
            self.detector = build_detector(data, 'flattened_k4')
        else:
            self.detector = Detector(data, 'o4_i4_hw')
            self.detector.model.requires_grad_(False)
            self.detector.model.eval()
        self.metadata = {'model_id': MODEL, 'profile': model, 'framework': 'torch',
            'torch_version': torch.__version__, 'cpu_threads': 2,
            'model_file_hashes': {source.name: MODELS[MODEL]},
            'input_transform': {'color_order': 'RGB->Pillow L', 'height': 256, 'width': 384,
                'keep_ratio': False, 'interp': 'Pillow bilinear', 'mean': 117., 'scale': 1.,
                'layout': 'NCHW', 'dtype': 'float32', 'camera_input_equivalence': False},
            'output_coordinate_space': 'original_image_continuous_xyxy',
            'decoder': 'Unmodified train_recognition.detections: current anchor_major delta, class_major foreground1 score, ratio-major anchors, floating exp clamp[-8,8], preNMS3000/NMS.3/postNMS30.',
            'packing_profile': 'o4_i4_hw' if model == 's5_baseline_o4_i4_hw' else '[O/4,(I*H*W)/4,O4,K4] ordinary conv; upsample unchanged',
            'additional_score_filter': None, 'state_before': self.audit_state(),
            'verified_dsp_numerical_parity': False, 'training_or_optimizer': False}

    def audit_state(self):
        return {'state_sha256': state_digest(self.detector.model.state_dict()),
                'parameters_require_grad': sum(p.numel() for p in self.detector.model.parameters() if p.requires_grad),
                'model_training': self.detector.model.training}

    def predict_rgb(self, rgb):
        height, width = rgb.shape[:2]
        with torch.inference_mode():
            logits, delta = self.detector(prepare_rgb(rgb))
            return detections(logits, delta, self.detector.anchors, {'width': width, 'height': height})


def load_detector(model, model_dir):
    return S5HypothesisDetector(model, model_dir)
