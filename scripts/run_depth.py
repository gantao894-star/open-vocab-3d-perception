#!/usr/bin/env python3
"""
Depth Anything V2 - PyTorch inference script.
Usage: python scripts/run_depth.py --image data/demo_inputs/sample.jpg
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'depth_anything_v2'))

import argparse
import time
import numpy as np
import torch
import cv2
from pathlib import Path

# DepthAnythingV2 model from the cloned repo
from depth_anything_v2.dpt import DepthAnythingV2

CKPT_DIR = Path(__file__).parent.parent / "weights"
OUTDIR = Path(__file__).parent.parent / "visualizations" / "depth"

MODEL_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
    "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
    "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
}


def load_model(encoder: str = "vits", device: torch.device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = MODEL_CONFIGS[encoder]
    model = DepthAnythingV2(**cfg)
    ckpt_path = CKPT_DIR / f"depth_anything_v2_{encoder}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model = model.to(device).eval()
    print(f"[Depth] Loaded {encoder} from {ckpt_path}")
    return model, device


def run_inference(model, img_path: str, device, input_size: int = 518):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Warmup
    with torch.no_grad():
        depth = model.infer_image(img_rgb, input_size)

    # Timed run (5 iterations)
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        with torch.no_grad():
            depth = model.infer_image(img_rgb, input_size)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = float(np.mean(times))
    p95_ms = float(np.percentile(times, 95))
    fps = 1000.0 / avg_ms
    print(f"[Depth] avg={avg_ms:.1f}ms  p95={p95_ms:.1f}ms  fps={fps:.1f}")

    return depth, avg_ms, fps


def save_visualization(img_path: str, depth: np.ndarray, encoder: str, avg_ms: float):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    stem = Path(img_path).stem

    # Normalize depth to [0, 255]
    depth_norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)

    # Side-by-side with original
    orig = cv2.imread(img_path)
    h, w = orig.shape[:2]
    depth_resized = cv2.resize(depth_colored, (w, h))
    combo = np.hstack([orig, depth_resized])

    # Add text overlay
    cv2.putText(combo, f"Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
    cv2.putText(combo, f"Depth Anything V2 ({encoder}) | {avg_ms:.0f}ms",
                (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    out_path = OUTDIR / f"{stem}_depth_{encoder}.jpg"
    cv2.imwrite(str(out_path), combo)
    print(f"[Depth] Saved → {out_path}")

    # Also save raw depth as npy for downstream use
    np.save(str(OUTDIR / f"{stem}_depth_{encoder}.npy"), depth)
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Depth Anything V2 inference")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--input-size", type=int, default=518)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Depth] Device: {device}")
    if torch.cuda.is_available():
        print(f"[Depth] GPU: {torch.cuda.get_device_name(0)}")

    model, device = load_model(args.encoder, device)
    depth, avg_ms, fps = run_inference(model, args.image, device, args.input_size)
    save_visualization(args.image, depth, args.encoder, avg_ms)

    print(f"\n[Depth] Summary:")
    print(f"  Encoder: {args.encoder}")
    print(f"  Avg latency: {avg_ms:.1f} ms")
    print(f"  FPS: {fps:.1f}")
    print(f"  Depth shape: {depth.shape}")
    print(f"  Depth min/max: {depth.min():.3f} / {depth.max():.3f}")


if __name__ == "__main__":
    main()
