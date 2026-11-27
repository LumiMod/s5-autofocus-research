"""Official PaddleDetection BlazeFace-FPN-SSH inference-only adapter.

The release/2.9 exported graph includes SSDBox decoding and multiclass NMS.
Preprocessing follows its infer_cfg.yml and deploy/python/preprocess.py.
No weights, anchors, score threshold or NMS settings are changed here.
An optional (width, height) stretch is an explicitly labelled experiment;
the graph's scale_factor input maps those results back to the original image.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import yaml


DEFAULT_MODEL_DIR = (Path(__file__).resolve().parent / "artifacts" /
                     "external-baselines/models/blazeface/export" /
                     "blazeface_fpn_ssh_1000e")
SOURCE_BASE = "https://github.com/PaddlePaddle/PaddleDetection/blob/release/2.9/"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _graph_nms(model_file: Path) -> dict:
    from paddle.base.proto.framework_pb2 import ProgramDesc

    graph = ProgramDesc()
    graph.ParseFromString(model_file.read_bytes())
    found = []
    for block in graph.blocks:
        for op in block.ops:
            if op.type != "multiclass_nms3":
                continue
            attrs = {a.name: a for a in op.attrs}
            found.append({
                "operator": op.type,
                "score_threshold": float(attrs["score_threshold"].f),
                "nms_threshold": float(attrs["nms_threshold"].f),
                "nms_top_k": int(attrs["nms_top_k"].i),
                "keep_top_k": int(attrs["keep_top_k"].i),
                "background_label": int(attrs["background_label"].i),
                "normalized": bool(attrs["normalized"].b),
            })
    if len(found) != 1:
        raise ValueError(f"Expected one exported multiclass_nms3, got {len(found)}")
    return found[0]


class BlazeFaceDetector:
    """Return person_face detections in original-image pixel coordinates."""

    def __init__(self, model_dir=DEFAULT_MODEL_DIR, input_size=None, cpu_threads=1):
        import paddle
        from paddle.inference import Config, create_predictor

        self.model_dir = Path(model_dir).resolve()
        self.input_size = tuple(input_size) if input_size is not None else None
        if self.input_size is not None and (
                len(self.input_size) != 2 or
                any(not isinstance(x, int) or x <= 0 for x in self.input_size)):
            raise ValueError("input_size must be positive integer (width, height)")
        cfg = yaml.safe_load((self.model_dir / "infer_cfg.yml").read_text())
        if cfg.get("arch") != "Face" or cfg.get("label_list") != ["face"]:
            raise ValueError("Expected official one-class Face inference config")
        transforms = cfg["Preprocess"]
        if [x["type"] for x in transforms] != ["NormalizeImage", "Permute"]:
            raise ValueError("This adapter requires the default no-resize BlazeFace config")
        norm = transforms[0]
        if norm.get("is_scale") is not False or norm.get("norm_type", "mean_std") != "mean_std":
            raise ValueError("Unexpected normalization in exported BlazeFace config")
        # Upstream creates default-dtype numpy constants (float64 for std),
        # then performs in-place operations on the float32 image.
        self.mean = np.asarray(norm["mean"]).reshape(1, 1, 3)
        self.std = np.asarray(norm["std"]).reshape(1, 1, 3)
        if not np.isfinite(self.mean).all() or not (self.std > 0).all():
            raise ValueError("Invalid normalization constants")

        model_file = self.model_dir / "model.pdmodel"
        params_file = self.model_dir / "model.pdiparams"
        nms = _graph_nms(model_file)
        if nms["background_label"] != 1:
            raise ValueError("Expected foreground face=0 and background=1")
        config = Config(str(model_file), str(params_file))
        config.disable_gpu()
        config.disable_glog_info()
        config.set_cpu_math_library_num_threads(cpu_threads)
        self.predictor = create_predictor(config)
        self.input_names = self.predictor.get_input_names()
        self.output_names = self.predictor.get_output_names()
        if set(self.input_names) != {"image", "im_shape", "scale_factor"}:
            raise ValueError(f"Unexpected graph inputs: {self.input_names}")
        if len(self.output_names) != 2:
            raise ValueError(f"Expected bbox and bbox_num outputs: {self.output_names}")
        transform = {"color_order": "RGB", "dtype": "float32", "layout": "NCHW",
                     "mean": norm["mean"], "std": norm["std"], "is_scale": False,
                     "resize_policy": "native", "keep_ratio": True}
        if self.input_size is not None:
            transform.update(resize_policy="stretch", width=self.input_size[0],
                             height=self.input_size[1], keep_ratio=False,
                             interpolation="cv2.INTER_LINEAR",
                             resize_before_normalization=True)
        self.metadata = {
            "name": "blazeface_native" if self.input_size is None else "blazeface_s5_size",
            "model_id": "blazeface_fpn_ssh_1000e",
            "model": "BlazeFace-FPN-SSH", "framework": "Paddle Inference",
            "paddle_version": paddle.__version__, "cpu_threads": cpu_threads,
            "class_map": {"0": "person_face"}, "input_transform": transform,
            "output_format": "class_id,score,xmin,ymin,xmax,ymax; original-image pixels",
            "output_coordinate_space": "original_image_continuous_xyxy",
            "box_decode": "official exported SSDBox and multiclass_nms3; no external decode or NMS",
            "graph_nms": nms, "additional_score_filter": None,
            "model_file_hashes": {name: _sha256(self.model_dir / name) for name in
                                  ("model.pdmodel", "model.pdiparams", "infer_cfg.yml")},
            "adapter_sha256": _sha256(Path(__file__)),
            "source_urls": [SOURCE_BASE + "configs/face_detection/_base_/face_reader.yml",
                            SOURCE_BASE + "deploy/python/preprocess.py",
                            SOURCE_BASE + "ppdet/modeling/architectures/blazeface.py",
                            SOURCE_BASE + "ppdet/modeling/layers.py"],
        }

    def preprocess_rgb(self, rgb: np.ndarray) -> dict[str, np.ndarray]:
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
            raise TypeError("Expected a uint8 RGB numpy array")
        if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1:
            raise ValueError("Expected nonempty HWC RGB input")
        orig_h, orig_w = rgb.shape[:2]
        image = rgb if self.input_size is None else cv2.resize(
            rgb, self.input_size, interpolation=cv2.INTER_LINEAR)
        height, width = image.shape[:2]
        # Same order and arithmetic as official NormalizeImage + Permute.
        image = image.astype(np.float32, copy=True)
        image -= self.mean
        image /= self.std
        return {
            "image": np.ascontiguousarray(image.transpose(2, 0, 1)[None]),
            "im_shape": np.array([[height, width]], dtype=np.float32),
            "scale_factor": np.array([[height / orig_h, width / orig_w]], dtype=np.float32),
        }

    def predict_rgb(self, rgb: np.ndarray) -> list[dict]:
        inputs = self.preprocess_rgb(rgb)
        for name in self.input_names:
            handle = self.predictor.get_input_handle(name)
            handle.reshape(inputs[name].shape)
            handle.copy_from_cpu(inputs[name])
        self.predictor.run()
        boxes, counts = [self.predictor.get_output_handle(name).copy_to_cpu()
                         for name in self.output_names]
        if counts.size != 1 or int(counts.reshape(-1)[0]) < 0:
            raise ValueError(f"Unexpected single-image box count {counts}")
        count = int(counts.reshape(-1)[0])
        if count == 0:
            return []
        if boxes.ndim != 2 or boxes.shape[1] != 6 or boxes.shape[0] != count:
            raise ValueError(f"Unexpected box output {boxes.shape}, count={count}")
        if not np.isfinite(boxes).all():
            raise ValueError("Non-finite BlazeFace predictions")
        detections = []
        for row in boxes:
            if row[0] < 0:  # Paddle's no-detection sentinel, if present.
                continue
            if row[0] != 0 or not 0 <= row[1] <= 1:
                raise ValueError(f"Unexpected class or confidence: {row[:2]}")
            if row[4] <= row[2] or row[5] <= row[3]:
                raise ValueError(f"Degenerate BlazeFace box: {row[2:]}")
            detections.append({"class": "person_face", "score": float(row[1]),
                               "bbox_xyxy": row[2:6].astype(float).tolist()})
        # SSDBox already multiplies by im_shape / scale_factor. Do not rescale again.
        return detections
