#!/usr/bin/env python3
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch

from depth_anything_v2.dpt import DepthAnythingV2


MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


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


def preprocess_fixed(raw_bgr, input_size):
    img = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/data/ganyw/3D/models/depth_anything_v2")
    parser.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--image", default="/data/ganyw/3D/models/depth_anything_v2/assets/DA-2K.png")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--output", default="/data/ganyw/3D/benchmarks/depth/pytorch_fp32_vits_518.json")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")

    ckpt = os.path.join(args.repo_root, "checkpoints", f"depth_anything_v2_{args.encoder}.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    raw = cv2.imread(args.image)
    if raw is None:
        raise FileNotFoundError(f"Image not found or unreadable: {args.image}")

    model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder])
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(args.device).eval()

    inp_np = preprocess_fixed(raw, args.input_size)
    inp = torch.from_numpy(inp_np).to(args.device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model(inp)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()

        lat_ms = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            _ = model(inp)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            lat_ms.append((time.perf_counter() - t0) * 1000.0)

    metrics = summarize(lat_ms)
    result = {
        "backend": "pytorch_fp32",
        "encoder": args.encoder,
        "input_size": args.input_size,
        "device": args.device,
        "warmup": args.warmup,
        "runs": args.runs,
        "metrics": metrics,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
