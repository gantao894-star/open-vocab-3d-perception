#!/usr/bin/env python3
"""
Combined Grounded Scene Understanding Pipeline.
Input: RGB image + text query  
Output: Detection boxes + segmentation masks + depth map, all visualized together.

Usage:
    python scripts/run_grounded_scene.py --image data/demo_inputs/kitti_000010.png --text "car . person"
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'depth_anything_v2'))

import argparse, time, json
from pathlib import Path
import numpy as np
import torch
import cv2

# ---- Depth Anything V2 ----
from depth_anything_v2.dpt import DepthAnythingV2

# ---- GroundingDINO ----
from groundingdino.util.inference import load_model as load_gdino, load_image, predict

# ---- SAM 2 ----
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# ---- Paths ----
ROOT = Path(__file__).parent.parent
WEIGHTS = ROOT / "weights"
OUTDIR = ROOT / "visualizations" / "pipeline"
BENCHMARKS = ROOT / "benchmarks" / "pipeline"

DEPTH_CONFIGS = {
    "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
}

GDINO_CONFIG = ROOT / "models" / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"


def load_models(device):
    print("[Pipeline] Loading models...")
    # Depth
    depth_model = DepthAnythingV2(**DEPTH_CONFIGS["vits"])
    depth_model.load_state_dict(torch.load(WEIGHTS / "depth_anything_v2_vits.pth", map_location="cpu"))
    depth_model = depth_model.to(device).eval()
    print("  ✓ Depth Anything V2 ViT-S")

    # GroundingDINO
    gdino_model = load_gdino(str(GDINO_CONFIG), str(WEIGHTS / "groundingdino_swint_ogc.pth"), device=str(device))
    gdino_model.eval()
    print("  ✓ Grounding DINO SwinT")

    # SAM 2
    sam2_model = build_sam2(SAM2_CONFIG, str(WEIGHTS / "sam2.1_hiera_small.pt"), device=str(device))
    predictor = SAM2ImagePredictor(sam2_model)
    print("  ✓ SAM 2 hiera-small")

    return depth_model, gdino_model, predictor


def run_pipeline(image_path: str, text_query: str, depth_model, gdino_model, sam2_predictor,
                 device, box_thr=0.35, text_thr=0.25):
    # --- Load image ---
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]

    timings = {}

    # --- Step 1: Depth Estimation ---
    t0 = time.perf_counter()
    with torch.no_grad():
        depth = depth_model.infer_image(img_rgb, 518)
    timings["depth_ms"] = (time.perf_counter() - t0) * 1000
    print(f"  [Depth] {timings['depth_ms']:.0f}ms  min/max: {depth.min():.2f}/{depth.max():.2f}")

    # --- Step 2: Grounding DINO ---
    img_src, img_tensor = load_image(image_path)
    t0 = time.perf_counter()
    with torch.no_grad():
        boxes, logits, phrases = predict(
            model=gdino_model, image=img_tensor, caption=text_query,
            box_threshold=box_thr, text_threshold=text_thr, device=str(device)
        )
    timings["gdino_ms"] = (time.perf_counter() - t0) * 1000
    print(f"  [GDINO] {timings['gdino_ms']:.0f}ms  {len(boxes)} detections")

    # Convert boxes from [cx, cy, w, h] normalized to [x1, y1, x2, y2] pixel
    if len(boxes) > 0:
        boxes_xyxy = boxes.cpu().numpy().copy()
        boxes_xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2) * w
        boxes_xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2) * h
        boxes_xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2) * w
        boxes_xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2) * h
    else:
        boxes_xyxy = np.zeros((0, 4), dtype=np.float32)

    # --- Step 3: SAM 2 Segmentation ---
    t0 = time.perf_counter()
    if len(boxes_xyxy) > 0:
        sam2_predictor.set_image(img_rgb)
        masks, scores, _ = sam2_predictor.predict(
            point_coords=None, point_labels=None,
            box=boxes_xyxy[:, :4].astype(np.float32),
            multimask_output=False,
        )
        if masks.ndim == 4:  # [N, 1, H, W] -> [N, H, W]
            masks = masks[:, 0]
    else:
        masks = np.zeros((0, h, w), dtype=bool)
    timings["sam2_ms"] = (time.perf_counter() - t0) * 1000
    print(f"  [SAM2] {timings['sam2_ms']:.0f}ms  {len(masks)} masks")

    timings["total_ms"] = timings["depth_ms"] + timings["gdino_ms"] + timings["sam2_ms"]

    return img_bgr, depth, boxes_xyxy, logits, phrases, masks, timings


def visualize(image_path, img_bgr, depth, boxes_xyxy, logits, phrases, masks, timings):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    h, w = img_bgr.shape[:2]

    # Color palette
    COLORS = [
        (255, 80, 80), (80, 255, 80), (80, 80, 255),
        (255, 255, 80), (255, 80, 255), (80, 255, 255),
        (200, 150, 80), (80, 200, 150), (150, 80, 200),
    ]

    # -- Panel 1: Detection + Segmentation overlay --
    panel_det = img_bgr.copy()
    mask_overlay = img_bgr.copy()
    for i, (box, logit, phrase, mask) in enumerate(zip(boxes_xyxy, logits, phrases, masks)):
        c = COLORS[i % len(COLORS)]
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(panel_det, (x1, y1), (x2, y2), c, 2)
        label = f"{phrase} {logit:.2f}"
        cv2.putText(panel_det, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        if mask.dtype != bool:
            mask = mask.astype(bool)
        mask_overlay[mask] = np.array(c, dtype=np.uint8)
    panel_det = cv2.addWeighted(img_bgr, 0.5, mask_overlay, 0.5, 0)
    for i, (box, logit, phrase) in enumerate(zip(boxes_xyxy, logits, phrases)):
        c = COLORS[i % len(COLORS)]
        x1, y1, x2, y2 = box.astype(int)
        cv2.rectangle(panel_det, (x1, y1), (x2, y2), c, 2)

    # -- Panel 2: Depth colormap --
    depth_norm = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
    depth_colored = cv2.resize(depth_colored, (w, h))

    # -- Combine panels --
    gap = np.ones((h, 4, 3), dtype=np.uint8) * 60
    combo = np.hstack([img_bgr, gap, panel_det, gap, depth_colored])

    # -- Header text --
    info = (f"Query: '{' | '.join(set(phrases))}' | Det: {timings['gdino_ms']:.0f}ms  "
            f"Seg: {timings['sam2_ms']:.0f}ms  Depth: {timings['depth_ms']:.0f}ms  "
            f"Total: {timings['total_ms']:.0f}ms")
    cv2.putText(combo, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(combo, "Original", (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(combo, "Grounded Segmentation", (w + 10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(combo, "Depth Estimation", (2 * w + 10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    out_path = OUTDIR / f"{stem}_pipeline.jpg"
    cv2.imwrite(str(out_path), combo)
    print(f"  [Pipeline] Saved → {out_path}")

    # -- Save benchmark JSON --
    BENCHMARKS.mkdir(parents=True, exist_ok=True)
    bench_path = BENCHMARKS / f"{stem}_timings.json"
    with open(bench_path, "w") as f:
        json.dump({
            "image": image_path,
            "query": " | ".join(set(phrases)),
            "n_detections": len(boxes_xyxy),
            "timings_ms": timings,
        }, f, indent=2)
    print(f"  [Benchmark] Saved → {bench_path}")
    return str(out_path)


def main():
    parser = argparse.ArgumentParser(description="Open-Vocabulary 3D Grounded Scene Pipeline")
    parser.add_argument("--image", required=True)
    parser.add_argument("--text", default="car . person", help="Labels separated by ' . '")
    parser.add_argument("--box-thr", type=float, default=0.35)
    parser.add_argument("--text-thr", type=float, default=0.25)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Pipeline] Device: {device}")
    if torch.cuda.is_available():
        print(f"[Pipeline] GPU: {torch.cuda.get_device_name(0)}")

    depth_model, gdino_model, sam2_predictor = load_models(device)

    print(f"\n[Pipeline] Running on: {args.image}")
    print(f"[Pipeline] Query: '{args.text}'")
    print()

    img_bgr, depth, boxes_xyxy, logits, phrases, masks, timings = run_pipeline(
        args.image, args.text, depth_model, gdino_model, sam2_predictor, device,
        args.box_thr, args.text_thr
    )

    out_path = visualize(args.image, img_bgr, depth, boxes_xyxy, logits, phrases, masks, timings)

    print(f"\n[Pipeline] Summary:")
    print(f"  Query: '{args.text}'")
    print(f"  Detections: {len(boxes_xyxy)} ({', '.join(set(phrases))})")
    print(f"  Depth: {timings['depth_ms']:.0f}ms")
    print(f"  Detection: {timings['gdino_ms']:.0f}ms")
    print(f"  Segmentation: {timings['sam2_ms']:.0f}ms")
    print(f"  Total: {timings['total_ms']:.0f}ms ({1000/timings['total_ms']:.2f} FPS pipeline)")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
