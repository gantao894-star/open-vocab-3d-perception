#!/usr/bin/env python3
"""
2D → 3D Depth Unprojection + Point Cloud Generation
Takes RGB image, depth map, and segmentation masks → outputs colored 3D point cloud
with detected objects highlighted.

Usage:
    python scripts/lift_to_3d.py --image data/demo_inputs/kitti_000010.png --text "car . person"

Intrinsics:
    If running on KITTI, real camera matrix P2 is used from calib files.
    Otherwise falls back to a reasonable default (FOV ~70°).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models', 'depth_anything_v2'))

import argparse, json
from pathlib import Path
import numpy as np
import torch
import cv2
import open3d as o3d

# ---- Models ----
from depth_anything_v2.dpt import DepthAnythingV2
from groundingdino.util.inference import load_model as load_gdino, load_image, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

ROOT = Path(__file__).parent.parent
WEIGHTS = ROOT / "weights"
OUTDIR = ROOT / "visualizations" / "3d"

DEPTH_CONFIG = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}
GDINO_CONFIG = ROOT / "models" / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"

# KITTI Camera P2 intrinsics (standard left color camera)
KITTI_K = np.array([
    [721.5377, 0.0,      609.5593],
    [0.0,      721.5377, 172.8540],
    [0.0,      0.0,      1.0     ]
], dtype=np.float32)


def get_intrinsics(image_path: str, img_w: int, img_h: int) -> np.ndarray:
    """Get camera intrinsics: use KITTI calib if available, else estimate from FOV."""
    # Try KITTI calib
    calib_path = Path(image_path).parent.parent / "calib" / (Path(image_path).stem + ".txt")
    if calib_path.exists():
        with open(calib_path) as f:
            for line in f:
                if line.startswith("P2:"):
                    vals = [float(v) for v in line.split()[1:]]
                    K = np.array(vals).reshape(3, 4)[:3, :3].astype(np.float32)
                    return K
    # Fallback: estimate from image size assuming 70° horizontal FOV
    fx = img_w / (2 * np.tan(np.radians(70) / 2))
    cx, cy = img_w / 2, img_h / 2
    return np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], dtype=np.float32)


def depth_to_pointcloud(depth: np.ndarray, K: np.ndarray,
                         img_rgb: np.ndarray,
                         max_depth: float = 50.0,
                         stride: int = 2) -> tuple:
    """
    Unproject depth map to 3D point cloud.
    Returns:
        points: [N, 3] float32  (X, Y, Z in camera frame)
        colors: [N, 3] float32  (R, G, B in [0,1])
    """
    h, w = depth.shape
    # Normalize depth to a reasonable metric scale (relative depth → ~0..50m)
    d_min, d_max = depth.min(), depth.max()
    depth_metric = (depth - d_min) / (d_max - d_min + 1e-8) * max_depth + 0.5

    # Pixel grid
    u = np.arange(0, w, stride)
    v = np.arange(0, h, stride)
    uu, vv = np.meshgrid(u, v)
    dd = depth_metric[::stride, ::stride]

    # Back-project: X = (u - cx) * Z / fx
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (uu - cx) * dd / fx
    Y = (vv - cy) * dd / fy
    Z = dd

    # Filter far/invalid
    valid = (Z > 0.1) & (Z < max_depth)
    points = np.stack([X[valid], Y[valid], Z[valid]], axis=1).astype(np.float32)
    colors_arr = (img_rgb[::stride, ::stride][valid] / 255.0).astype(np.float32)

    return points, colors_arr


def colorize_masks(masks: np.ndarray, phrases: list, depth_metric: np.ndarray,
                   K: np.ndarray, stride: int) -> dict:
    """For each detected object mask, compute its 3D centroid and return mask→label."""
    COLORS = [
        [1.0, 0.3, 0.3], [0.3, 1.0, 0.3], [0.3, 0.3, 1.0],
        [1.0, 1.0, 0.3], [1.0, 0.3, 1.0], [0.3, 1.0, 1.0],
    ]
    results = []
    h, w = depth_metric.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    for i, (mask, phrase) in enumerate(zip(masks, phrases)):
        if mask.dtype != bool:
            mask = mask.astype(bool)
        ys, xs = np.where(mask)
        if len(ys) == 0:
            continue
        zs = depth_metric[ys, xs]
        z_med = float(np.median(zs))
        x_med = float((np.median(xs) - cx) * z_med / fx)
        y_med = float((np.median(ys) - cy) * z_med / fy)
        results.append({
            "phrase": phrase,
            "mask": mask,
            "color": COLORS[i % len(COLORS)],
            "centroid_3d": [x_med, y_med, z_med],
            "pixel_count": int(len(ys)),
        })
    return results


def run_all(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[3D] Device: {device}")

    # Load models
    print("[3D] Loading models...")
    depth_model = DepthAnythingV2(**DEPTH_CONFIG)
    depth_model.load_state_dict(torch.load(WEIGHTS / "depth_anything_v2_vits.pth", map_location="cpu"))
    depth_model = depth_model.to(device).eval()

    gdino_model = load_gdino(str(GDINO_CONFIG), str(WEIGHTS / "groundingdino_swint_ogc.pth"), device=str(device))
    gdino_model.eval()

    sam2_model = build_sam2(SAM2_CONFIG, str(WEIGHTS / "sam2.1_hiera_small.pt"), device=str(device))
    predictor = SAM2ImagePredictor(sam2_model)
    print("  All models loaded.")

    # Load image
    img_bgr = cv2.imread(args.image)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]
    K = get_intrinsics(args.image, w, h)
    print(f"[3D] Image: {w}×{h}  K[0,0]={K[0,0]:.1f}")

    # Depth
    with torch.no_grad():
        depth = depth_model.infer_image(img_rgb, 518)

    # Normalize depth to metric scale
    d_min, d_max = depth.min(), depth.max()
    depth_metric = (depth - d_min) / (d_max - d_min + 1e-8) * args.max_depth + 0.5

    # GroundingDINO
    img_src, img_tensor = load_image(args.image)
    with torch.no_grad():
        boxes, logits, phrases = predict(
            model=gdino_model, image=img_tensor, caption=args.text,
            box_threshold=args.box_thr, text_threshold=args.text_thr, device=str(device)
        )
    print(f"[3D] Detected {len(boxes)} objects: {phrases}")

    # Convert boxes to pixel coords
    boxes_xyxy = np.zeros((len(boxes), 4), dtype=np.float32)
    if len(boxes) > 0:
        b = boxes.cpu().numpy()
        boxes_xyxy[:, 0] = (b[:, 0] - b[:, 2] / 2) * w
        boxes_xyxy[:, 1] = (b[:, 1] - b[:, 3] / 2) * h
        boxes_xyxy[:, 2] = (b[:, 0] + b[:, 2] / 2) * w
        boxes_xyxy[:, 3] = (b[:, 1] + b[:, 3] / 2) * h

    # SAM 2
    masks = np.zeros((0, h, w), dtype=bool)
    if len(boxes_xyxy) > 0:
        predictor.set_image(img_rgb)
        masks_raw, _, _ = predictor.predict(
            point_coords=None, point_labels=None,
            box=boxes_xyxy.astype(np.float32),
            multimask_output=False,
        )
        if masks_raw.ndim == 4:
            masks_raw = masks_raw[:, 0]
        masks = masks_raw.astype(bool)

    # 2D → 3D lift
    print("[3D] Unprojecting depth to point cloud...")
    points, colors = depth_to_pointcloud(depth, K, img_rgb, args.max_depth, stride=args.stride)
    print(f"[3D] Base cloud: {len(points):,} points")

    # Colorize detected objects (highlight them with object-class colors)
    obj_info = colorize_masks(masks, phrases, depth_metric, K, args.stride)

    # Create a LUT: pixel (sampled) → point index in `points`
    colors_highlight = colors.copy()
    valid_ds = (depth_metric[::args.stride, ::args.stride] > 0.1) & \
               (depth_metric[::args.stride, ::args.stride] < args.max_depth)
    # point_lut[i,j] = index in points[], or -1 if invalid
    cumvalid = np.cumsum(valid_ds.flatten()) - 1   # 0-indexed running count
    point_lut = np.where(valid_ds.flatten(), cumvalid, -1).reshape(valid_ds.shape)

    for obj in obj_info:
        mask_ds = obj["mask"][::args.stride, ::args.stride]
        obj_pix_in_valid = mask_ds & valid_ds
        idxs = point_lut[obj_pix_in_valid]
        idxs = idxs[idxs >= 0]
        if len(idxs) > 0:
            colors_highlight[idxs] = np.array(obj["color"], dtype=np.float32)
        print(f"  {obj['phrase']}: {obj['pixel_count']} px  3D centroid: "
              f"({obj['centroid_3d'][0]:.1f}, {obj['centroid_3d'][1]:.1f}, {obj['centroid_3d'][2]:.1f})m")

    # Save point clouds
    OUTDIR.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem

    # Base cloud (colored by RGB)
    pcd_base = o3d.geometry.PointCloud()
    pcd_base.points = o3d.utility.Vector3dVector(points)
    pcd_base.colors = o3d.utility.Vector3dVector(colors)
    ply_base = str(OUTDIR / f"{stem}_cloud_rgb.ply")
    o3d.io.write_point_cloud(ply_base, pcd_base)
    print(f"[3D] Saved base cloud → {ply_base}")

    # Highlighted cloud (objects in vivid colors)
    pcd_highlight = o3d.geometry.PointCloud()
    pcd_highlight.points = o3d.utility.Vector3dVector(points)
    pcd_highlight.colors = o3d.utility.Vector3dVector(colors_highlight)
    ply_highlight = str(OUTDIR / f"{stem}_cloud_objects.ply")
    o3d.io.write_point_cloud(ply_highlight, pcd_highlight)
    print(f"[3D] Saved object-highlighted cloud → {ply_highlight}")

    # Save a top-down 2D projection (bird's-eye view) as image
    bev = make_bev_image(points, colors_highlight, scale=5.0, img_size=800)
    bev_path = str(OUTDIR / f"{stem}_bev.jpg")
    cv2.imwrite(bev_path, cv2.cvtColor((bev * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    print(f"[3D] Saved bird's-eye view → {bev_path}")

    # Save per-object 3D report
    report = {
        "image": args.image,
        "query": args.text,
        "n_points": int(len(points)),
        "objects": [
            {"phrase": o["phrase"],
             "centroid_3d_m": o["centroid_3d"],
             "pixel_count": o["pixel_count"]}
            for o in obj_info
        ]
    }
    report_path = str(OUTDIR / f"{stem}_3d_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[3D] Saved 3D report → {report_path}")
    return report


def make_bev_image(points: np.ndarray, colors: np.ndarray,
                   scale: float = 5.0, img_size: int = 800) -> np.ndarray:
    """Top-down (bird's-eye) view: X→right, Z→up in image."""
    # Project onto X-Z plane (bird's-eye)
    x = points[:, 0]
    z = points[:, 2]  # depth forward

    # Normalize to image coords
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-6) * (img_size - 1)
    z_norm = (1 - (z - z.min()) / (z.max() - z.min() + 1e-6)) * (img_size - 1)

    bev = np.zeros((img_size, img_size, 3), dtype=np.float32)
    xi = x_norm.astype(int).clip(0, img_size - 1)
    zi = z_norm.astype(int).clip(0, img_size - 1)
    # Draw from far to near so near objects are on top
    order = np.argsort(-z)
    bev[zi[order], xi[order]] = colors[order]
    return bev


def main():
    parser = argparse.ArgumentParser(description="Depth → 3D Point Cloud with object highlighting")
    parser.add_argument("--image", required=True)
    parser.add_argument("--text", default="car . person", help="Query labels separated by ' . '")
    parser.add_argument("--max-depth", type=float, default=50.0, help="Max depth in meters")
    parser.add_argument("--stride", type=int, default=2, help="Pixel stride for point cloud density")
    parser.add_argument("--box-thr", type=float, default=0.35)
    parser.add_argument("--text-thr", type=float, default=0.25)
    args = parser.parse_args()
    run_all(args)


if __name__ == "__main__":
    main()
