#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


def load_velodyne(bin_path: Path) -> np.ndarray:
    return np.fromfile(str(bin_path), dtype=np.float32).reshape(-1, 4)


def load_calib(calib_path: Path):
    with open(calib_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    def parse_line(idx: int) -> np.ndarray:
        return np.array(lines[idx].split(" ")[1:], dtype=np.float32)

    p2 = parse_line(2).reshape(3, 4)
    r0 = parse_line(4).reshape(3, 3)
    tr_velo_to_cam = parse_line(5).reshape(3, 4)
    return p2, r0, tr_velo_to_cam


def cart_to_hom(points: np.ndarray) -> np.ndarray:
    return np.hstack((points, np.ones((points.shape[0], 1), dtype=np.float32)))


def lidar_to_rect(points_lidar: np.ndarray, r0: np.ndarray, tr_velo_to_cam: np.ndarray) -> np.ndarray:
    return cart_to_hom(points_lidar) @ tr_velo_to_cam.T @ r0.T


def fit_plane_from_points(points: np.ndarray) -> np.ndarray | None:
    if points.shape[0] < 3:
        return None
    centroid = points.mean(axis=0)
    centered = points - centroid
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    normal = vh[-1]
    norm = np.linalg.norm(normal)
    if norm < 1e-6:
        return None
    normal = normal / norm
    d = -float(normal @ centroid)
    plane = np.array([normal[0], normal[1], normal[2], d], dtype=np.float32)
    if plane[1] > 0:
        plane = -plane
    plane /= np.linalg.norm(plane[:3])
    return plane


def ransac_ground_plane(points_rect: np.ndarray, rng: np.random.Generator, iterations: int, thresh: float):
    if points_rect.shape[0] < 3:
        return None, 0

    best_inliers = None
    best_count = 0
    for _ in range(iterations):
        sample = points_rect[rng.choice(points_rect.shape[0], 3, replace=False)]
        plane = fit_plane_from_points(sample)
        if plane is None:
            continue
        # Ground plane in rect camera coord should be close to horizontal.
        if abs(plane[1]) < 0.95:
            continue
        distances = np.abs(points_rect @ plane[:3] + plane[3])
        inliers = distances < thresh
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < 100:
        return None, best_count

    refined = fit_plane_from_points(points_rect[best_inliers])
    return refined, best_count


def select_ground_candidates(points_rect: np.ndarray) -> np.ndarray:
    x = points_rect[:, 0]
    y = points_rect[:, 1]
    z = points_rect[:, 2]

    # KITTI rect camera coordinates: x right, y down, z forward.
    mask = (
        (z > 5.0) & (z < 60.0) &
        (np.abs(x) < 20.0) &
        (y > 1.0) & (y < 2.5)
    )
    return points_rect[mask]


def write_plane(plane_path: Path, plane: np.ndarray) -> None:
    plane_path.parent.mkdir(parents=True, exist_ok=True)
    with open(plane_path, "w", encoding="utf-8") as f:
        f.write("# Plane\n")
        f.write("# Automatically generated\n")
        f.write("# a b c d in rect camera coordinate\n")
        f.write("{:.8f} {:.8f} {:.8f} {:.8f}\n".format(*plane.tolist()))


def generate_planes(root: Path, split: str, iterations: int, thresh: float, overwrite: bool) -> None:
    split_dir = root / "training"
    image_set = root / "ImageSets" / f"{split}.txt"
    plane_dir = split_dir / "planes"
    velodyne_dir = split_dir / "velodyne"
    calib_dir = split_dir / "calib"

    sample_ids = [line.strip() for line in image_set.read_text(encoding="utf-8").splitlines() if line.strip()]
    rng = np.random.default_rng(20260323)
    generated = 0
    fallback = 0
    skipped = 0

    for index, sample_id in enumerate(sample_ids, start=1):
        plane_path = plane_dir / f"{sample_id}.txt"
        if plane_path.exists() and not overwrite:
            skipped += 1
            continue

        points = load_velodyne(velodyne_dir / f"{sample_id}.bin")[:, :3]
        _, r0, tr_velo_to_cam = load_calib(calib_dir / f"{sample_id}.txt")
        points_rect = lidar_to_rect(points, r0, tr_velo_to_cam)
        candidates = select_ground_candidates(points_rect)
        plane, inlier_count = ransac_ground_plane(candidates, rng, iterations=iterations, thresh=thresh)

        if plane is None:
            # Fallback to a canonical camera-height plane when fitting is unreliable.
            plane = np.array([0.0, -1.0, 0.0, 1.65], dtype=np.float32)
            fallback += 1

        write_plane(plane_path, plane)
        generated += 1

        if index % 500 == 0 or index == len(sample_ids):
            print(
                f"[{index}/{len(sample_ids)}] generated={generated} skipped={skipped} "
                f"fallback={fallback} last_inliers={inlier_count}"
            )

    print(f"Done. generated={generated}, skipped={skipped}, fallback={fallback}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate KITTI road plane files for OpenPCDet.")
    parser.add_argument("--root", type=Path, default=Path("data/kitti"), help="KITTI root directory")
    parser.add_argument("--split", type=str, default="train", help="ImageSets split to generate for")
    parser.add_argument("--iterations", type=int, default=300, help="RANSAC iterations per sample")
    parser.add_argument("--thresh", type=float, default=0.08, help="Inlier threshold in meters")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing plane files")
    args = parser.parse_args()

    generate_planes(
        root=args.root,
        split=args.split,
        iterations=args.iterations,
        thresh=args.thresh,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
