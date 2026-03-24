#!/usr/bin/env python3
"""
Grounding DINO - Open-vocabulary detection script.
Usage: python scripts/run_grounding_dino.py --image path.jpg --text "cup . bottle"
"""
import sys, os
import argparse, time
import numpy as np
import torch
import cv2
from pathlib import Path

from groundingdino.util.inference import load_model, load_image, annotate, predict

CKPT = Path(__file__).parent.parent / "weights" / "groundingdino_swint_ogc.pth"
CONFIG = Path(__file__).parent.parent / "models" / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
OUTDIR = Path(__file__).parent.parent / "visualizations" / "detection"


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[GDINO] Device: {device}")

    model = load_model(str(CONFIG), str(CKPT), device=device)
    img_src, img_tensor = load_image(args.image)

    # Warmup
    predict(model=model, image=img_tensor, caption=args.text,
            box_threshold=args.box_thr, text_threshold=args.text_thr, device=device)

    # Timed
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        boxes, logits, phrases = predict(
            model=model, image=img_tensor, caption=args.text,
            box_threshold=args.box_thr, text_threshold=args.text_thr, device=device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = float(np.mean(times))
    print(f"[GDINO] avg={avg_ms:.1f}ms  fps={1000/avg_ms:.1f}  detections={len(boxes)}")
    for ph, logit in zip(phrases, logits):
        print(f"  - {ph}: {logit:.3f}")

    # Visualize
    OUTDIR.mkdir(parents=True, exist_ok=True)
    annotated = annotate(image_source=img_src, boxes=boxes, logits=logits, phrases=phrases)
    out_path = OUTDIR / f"{Path(args.image).stem}_gdino.jpg"
    cv2.imwrite(str(out_path), annotated)
    print(f"[GDINO] Saved → {out_path}")

    print(f"\n[GDINO] Summary: {len(boxes)} detections for '{args.text}' | {avg_ms:.1f}ms avg")
    return boxes, logits, phrases, avg_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--text", default="person . car . cup . bottle", help="Labels separated by ' . '")
    parser.add_argument("--box-thr", type=float, default=0.35)
    parser.add_argument("--text-thr", type=float, default=0.25)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
