#!/usr/bin/env python3
"""
SAM 2 - Segmentation from bounding boxes.
Usage: python scripts/run_sam2.py --image path.jpg --boxes "x1,y1,x2,y2"
Or called programmatically from run_grounded_scene.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'sam2'))

import argparse, time
import numpy as np
import torch
import cv2
from pathlib import Path

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

CKPT = Path(__file__).parent.parent / "weights" / "sam2.1_hiera_small.pt"
# SAM2.1 hiera_s config
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
OUTDIR = Path(__file__).parent.parent / "visualizations" / "segmentation"


def load_sam2(device="cuda"):
    model = build_sam2(SAM2_CONFIG, str(CKPT), device=device)
    predictor = SAM2ImagePredictor(model)
    print(f"[SAM2] Loaded sam2.1-hiera-s from {CKPT}")
    return predictor


def segment_from_boxes(predictor, img_bgr: np.ndarray, boxes_xyxy: np.ndarray):
    """
    boxes_xyxy: np.ndarray shape [N,4] in absolute pixel coords
    returns: masks [N, H, W] bool
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    predictor.set_image(img_rgb)

    t0 = time.perf_counter()
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=boxes_xyxy,
        multimask_output=False,
    )
    ms = (time.perf_counter() - t0) * 1000
    print(f"[SAM2] {len(masks)} masks | {ms:.1f}ms")
    return masks, scores, ms


def draw_masks(img_bgr: np.ndarray, masks: np.ndarray, alpha=0.45):
    overlay = img_bgr.copy()
    colors = [
        (255, 100, 100), (100, 255, 100), (100, 100, 255),
        (255, 255, 100), (255, 100, 255), (100, 255, 255),
    ]
    for i, mask in enumerate(masks):
        if mask.ndim == 3:
            mask = mask[0]
        color = colors[i % len(colors)]
        overlay[mask.astype(bool)] = color
    return cv2.addWeighted(img_bgr, 1 - alpha, overlay, alpha, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--boxes", default=None,
                        help="Comma-separated x1,y1,x2,y2 or multiple boxes as x1,y1,x2,y2;x1,y1,x2,y2")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SAM2] Device: {device}")

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    h, w = img.shape[:2]

    # Parse boxes or use full image
    if args.boxes:
        raw_boxes = [[float(v) for v in b.split(",")] for b in args.boxes.split(";")]
        boxes = np.array(raw_boxes, dtype=np.float32)
    else:
        boxes = np.array([[0, 0, w, h]], dtype=np.float32)

    predictor = load_sam2(device)
    masks, scores, ms = segment_from_boxes(predictor, img, boxes)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    masked = draw_masks(img, masks)
    out_path = OUTDIR / f"{Path(args.image).stem}_sam2.jpg"
    cv2.imwrite(str(out_path), masked)
    print(f"[SAM2] Saved → {out_path}")


if __name__ == "__main__":
    main()
