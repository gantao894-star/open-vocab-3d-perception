#!/usr/bin/env python3
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmpose.apis import init_model


def summarize(lat_ms):
    arr = np.asarray(lat_ms, dtype=np.float64)
    return {
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "fps": float(1000.0 / arr.mean()),
    }


def write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_cfg(config_path):
    return Config.fromfile(config_path)


def infer_input_size(cfg):
    if hasattr(cfg, "codec") and "input_size" in cfg.codec:
        size = cfg.codec["input_size"]
        return int(size[0]), int(size[1])
    pipelines = []
    for attr in ["test_dataloader", "val_dataloader", "train_dataloader"]:
        if hasattr(cfg, attr):
            dataset = getattr(cfg, attr).dataset
            if hasattr(dataset, "pipeline"):
                pipelines.extend(dataset.pipeline)
    for step in pipelines:
        if step.get("type") == "TopdownAffine" and "input_size" in step:
            size = step["input_size"]
            return int(size[0]), int(size[1])
    raise RuntimeError("Could not infer MMPose input size from config")


def preprocess_image(image_path, input_size):
    raw = cv2.imread(image_path)
    if raw is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
    img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size, interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    mean = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 1, 3)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)


class TensorModeWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x, None, mode="tensor")
        if isinstance(out, tuple):
            return out
        return (out,)


def init_pose_model(config, checkpoint, device):
    model = init_model(config, checkpoint, device=device)
    model.eval()
    return model
