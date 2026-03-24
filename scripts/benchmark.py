#!/usr/bin/env python3
"""
Benchmark script: run the full pipeline across N KITTI images and produce a CSV report.

Usage:
    python scripts/benchmark.py --n 10
    python scripts/benchmark.py --dir data/kitti/training/image_2 --n 20
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'depth_anything_v2'))

import argparse, time, csv, json
from pathlib import Path
import numpy as np
import torch
import cv2

from depth_anything_v2.dpt import DepthAnythingV2
from groundingdino.util.inference import load_model as load_gdino, load_image, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

ROOT = Path(__file__).parent.parent
WEIGHTS = ROOT / "weights"
DEPTH_CONFIG = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
GDINO_CONFIG = ROOT / "models" / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
BENCH_DIR = ROOT / "benchmarks"


def load_models(device):
    depth_model = DepthAnythingV2(**DEPTH_CONFIG)
    depth_model.load_state_dict(torch.load(WEIGHTS / "depth_anything_v2_vits.pth", map_location="cpu"))
    depth_model = depth_model.to(device).eval()

    gdino_model = load_gdino(str(GDINO_CONFIG), str(WEIGHTS / "groundingdino_swint_ogc.pth"), device=str(device))
    gdino_model.eval()

    sam2_model = build_sam2(SAM2_CONFIG, str(WEIGHTS / "sam2.1_hiera_small.pt"), device=str(device))
    predictor = SAM2ImagePredictor(sam2_model)
    return depth_model, gdino_model, predictor


def run_one(img_path, text, depth_model, gdino_model, sam2_pred, device, box_thr=0.35, text_thr=0.25):
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]

    # Depth
    t0 = time.perf_counter()
    with torch.no_grad():
        depth = depth_model.infer_image(img_rgb, 518)
    t_depth = (time.perf_counter() - t0) * 1000

    # GDINO
    t0 = time.perf_counter()
    img_src, img_tensor = load_image(str(img_path))
    with torch.no_grad():
        boxes, logits, phrases = predict(
            model=gdino_model, image=img_tensor, caption=text,
            box_threshold=box_thr, text_threshold=text_thr, device=str(device))
    t_gdino = (time.perf_counter() - t0) * 1000

    # SAM2
    t0 = time.perf_counter()
    if len(boxes) > 0:
        b = boxes.cpu().numpy()
        boxes_px = np.column_stack([
            (b[:, 0] - b[:, 2] / 2) * w, (b[:, 1] - b[:, 3] / 2) * h,
            (b[:, 0] + b[:, 2] / 2) * w, (b[:, 1] + b[:, 3] / 2) * h,
        ]).astype(np.float32)
        sam2_pred.set_image(img_rgb)
        masks, _, _ = sam2_pred.predict(
            point_coords=None, point_labels=None,
            box=boxes_px, multimask_output=False)
    t_sam2 = (time.perf_counter() - t0) * 1000

    return {
        "image": img_path.name,
        "n_detections": len(boxes),
        "depth_ms": round(t_depth, 1),
        "gdino_ms": round(t_gdino, 1),
        "sam2_ms": round(t_sam2, 1),
        "total_ms": round(t_depth + t_gdino + t_sam2, 1),
        "fps": round(1000 / (t_depth + t_gdino + t_sam2), 2),
        "phrases": " | ".join(set(phrases)) if phrases else "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/kitti/training/image_2", help="Image directory")
    parser.add_argument("--n", type=int, default=10, help="Number of images to benchmark")
    parser.add_argument("--text", default="car . person . truck")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Bench] Device: {device}")

    img_dir = ROOT / args.dir
    imgs = sorted(img_dir.glob("*.png"))[:args.n] + sorted(img_dir.glob("*.jpg"))[:args.n]
    imgs = imgs[:args.n]
    if not imgs:
        print(f"[Bench] No images found in {img_dir}"); return

    print(f"[Bench] Loading models...")
    depth_model, gdino_model, sam2_pred = load_models(device)

    # Warmup
    print("[Bench] Warmup run...")
    run_one(imgs[0], args.text, depth_model, gdino_model, sam2_pred, device)

    print(f"[Bench] Benchmarking {len(imgs)} images: '{args.text}'")
    results = []
    for i, img_path in enumerate(imgs):
        r = run_one(img_path, args.text, depth_model, gdino_model, sam2_pred, device)
        results.append(r)
        print(f"  [{i+1}/{len(imgs)}] {r['image']}: {r['n_detections']} det | {r['total_ms']}ms | {r['fps']} FPS")

    # Stats
    total_ms = [r["total_ms"] for r in results]
    det_ms = [r["gdino_ms"] for r in results]
    seg_ms = [r["sam2_ms"] for r in results]
    dep_ms = [r["depth_ms"] for r in results]

    summary = {
        "n_images": len(results),
        "query": args.text,
        "total_ms": {"mean": round(np.mean(total_ms), 1), "p50": round(np.percentile(total_ms, 50), 1), "p95": round(np.percentile(total_ms, 95), 1)},
        "depth_ms": {"mean": round(np.mean(dep_ms), 1)},
        "gdino_ms": {"mean": round(np.mean(det_ms), 1)},
        "sam2_ms": {"mean": round(np.mean(seg_ms), 1)},
        "mean_fps": round(np.mean([r["fps"] for r in results]), 2),
        "mean_detections": round(np.mean([r["n_detections"] for r in results]), 1),
    }

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = BENCH_DIR / f"benchmark_{len(results)}imgs.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    json_path = BENCH_DIR / f"benchmark_{len(results)}imgs_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[Bench] Summary:")
    print(f"  N images:        {summary['n_images']}")
    print(f"  Mean total:      {summary['total_ms']['mean']}ms  (p50={summary['total_ms']['p50']}ms, p95={summary['total_ms']['p95']}ms)")
    print(f"  Mean FPS:        {summary['mean_fps']}")
    print(f"  Mean detections: {summary['mean_detections']}")
    print(f"  Breakdown:  Depth={summary['depth_ms']['mean']}ms  GDINO={summary['gdino_ms']['mean']}ms  SAM2={summary['sam2_ms']['mean']}ms")
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")


if __name__ == "__main__":
    main()
