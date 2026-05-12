#!/usr/bin/env python3
"""
generate_skip_small.py

This is the file used to generate the data used for NeRF Inpainting.

Generate Points2NeRF-style training data (with depth) from Seg2Tunnel point clouds
using GLOBAL (full-scene) rendering mode and RANDOM sampling.

Modifications:
  - Uses strictly RANDOM sampling (no adaptive/curvature logic).
  - Skips files with fewer points than --num_points.
  - Logs skipped files to 'skipped_files.xlsx'.

Camera and projection helpers are imported from preprocessing/src/.
Default hyperparameters come from preprocessing/configs/default.json
(override the path with --config; override individual values with their CLI flags).

Input:
  - Directory containing .txt files, each with 5 columns: x y z intensity label

Outputs (inside --output_dir):
  - images/         (grayscale PNGs rendered from intensity; bg = black, for debug)
  - depths/         (16-bit PNG depth maps; bg = 0, for debug)
  - transforms.json (camera-to-world poses + intrinsics; optional debug)
  - <scene_name>.npz with:
        images:    [V, H, W, 3] float32 in [0,1], bg = white
        cam_poses: [V, 4, 4]   float32 (camera-to-world)
        data:      [N_pts, 6]  float32 [x, y, z, r, g, b]
        depths:    [V, H, W]   float32 depth in meters, 0 = invalid
  - pointcloud_centers.xlsx (centers of processed clouds)
  - skipped_files.xlsx (list of files skipped due to insufficient points)
"""

import sys
from pathlib import Path

# Make `src.*` importable when this script is run directly:
# scripts/ and src/ are siblings under preprocessing/, so prepend the parent.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import math
import hashlib
from typing import List, Optional

import numpy as np
from PIL import Image
from openpyxl import Workbook

from src.camera import random_sphere, look_at_c2w
from src.projection import load_txt_point_cloud, project_points_fullscene


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.json"


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------
# Scene processing (GLOBAL)
# ---------------------------
def process_scene_global(
    txt_paths: List[Path],
    out_dir: Path,
    num_views: int,
    img_h: int,
    img_w: int,
    fov_x: float,
    radius_scale: float,
    num_points: int,
    seed: int = 0,
    write_pngs: bool = True,
    num_variants: int = 1,
    npz_root_dir=None,
    save_downsampled_txt: bool = True,
    downsampled_txt_dir=None,
    txt_intensity: str = "raw",      # "raw" or "normalized"
    txt_coords: str = "centered",    # "centered" or "raw"
) -> Optional[np.ndarray]:
    """
    Returns center_raw (np.ndarray) if successful.
    Returns None if the file was skipped (too few points).
    """

    # Load & concatenate all point clouds in this folder
    pts_list, intens_list, labels_list = [], [], []
    for p in txt_paths:
        P, I, L = load_txt_point_cloud(p)
        pts_list.append(P)
        intens_list.append(I)
        labels_list.append(L)

    pts = np.concatenate(pts_list, axis=0).astype(np.float32)
    intens = np.concatenate(intens_list, axis=0).astype(np.float32)
    labels = np.concatenate(labels_list, axis=0).astype(np.int32)

    N_total = pts.shape[0]

    # --- CHECK: Skip if not enough points ---
    if N_total < num_points:
        print(f"[SKIP] File has {N_total} points, which is less than required {num_points}.")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    if write_pngs:
        (out_dir / "images").mkdir(exist_ok=True)
        (out_dir / "depths").mkdir(exist_ok=True)

    # --- Compute center in ORIGINAL coordinates ---
    mins_raw = pts.min(axis=0)
    maxs_raw = pts.max(axis=0)
    center_raw = 0.5 * (mins_raw + maxs_raw)

    # --- Center at origin ---
    pts = pts - center_raw[None, :]

    # --- Global intensity normalisation ---
    if np.ptp(intens) > 0:
        intens_norm = (intens - intens.min()) / (intens.max() - intens.min())
    else:
        intens_norm = np.zeros_like(intens, dtype=np.float32)

    # Scene bbox & center (centered coords)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    half_diag = 0.5 * np.linalg.norm(maxs - mins)

    # Camera settings
    distance = (half_diag / math.tan(fov_x / 2.0)) * radius_scale
    fx = fy = 0.5 * img_w / math.tan(fov_x / 2.0)
    cx = img_w / 2.0
    cy = img_h / 2.0
    camera_angle_x = fov_x

    # --- Generate Views ---
    all_images = []
    all_depths = []
    all_poses = []
    frames = []

    if num_variants < 1:
        num_variants = 1

    ss = np.random.SeedSequence(seed)
    children = ss.spawn(1 + num_variants)
    rng_views = np.random.default_rng(children[0])
    variant_rngs = [np.random.default_rng(s) for s in children[1:]]

    max_attempts = num_views * 20
    attempts = 0

    while len(all_images) < num_views and attempts < max_attempts:
        d = random_sphere(1, rng_views)[0]
        attempts += 1

        origin = center + distance * d
        c2w = look_at_c2w(
            origin.astype(np.float32),
            center.astype(np.float32),
            up_hint=np.array([0, 1, 0], dtype=np.float32),
        )

        rgb, depth, hit = project_points_fullscene(
            pts, intens_norm, c2w, fx, fy, cx, cy, img_w, img_h,
            depth_clip=(0.0, distance * 2.0),
        )

        if not hit.any():
            continue

        view_idx = len(all_images)
        img_name = f"{view_idx:05d}.png"

        img_f = rgb.astype(np.float32) / 255.0
        img_rgb = np.stack([img_f, img_f, img_f], axis=-1)
        bg_mask = ~hit
        if bg_mask.any():
            img_rgb[bg_mask] = 1.0

        depth_m = depth.astype(np.float32) / 1000.0

        all_images.append(img_rgb.astype(np.float32))
        all_depths.append(depth_m.astype(np.float32))
        all_poses.append(c2w.astype(np.float32))

        if write_pngs:
            Image.fromarray(rgb, mode="L").save(out_dir / "images" / img_name)
            Image.fromarray(depth, mode="I;16").save(out_dir / "depths" / img_name)

        frames.append({
            "file_path": f"./images/{img_name}",
            "transform_matrix": c2w.tolist(),
        })

    if len(all_images) == 0:
        print("[WARN] No valid views produced for this scene.")
        raise RuntimeError("Global mode: no valid views produced.")

    # Save transforms.json
    if write_pngs:
        transforms = {
            "fl_x": fx,
            "fl_y": fy,
            "cx": cx,
            "cy": cy,
            "w": img_w,
            "h": img_h,
            "camera_angle_x": camera_angle_x,
            "frames": frames,
        }
        with open(out_dir / "transforms.json", "w") as f:
            json.dump(transforms, f, indent=2)

    images_arr = np.stack(all_images, axis=0).astype(np.float32)
    depths_arr = np.stack(all_depths, axis=0).astype(np.float32)
    poses_arr  = np.stack(all_poses, axis=0).astype(np.float32)

    if npz_root_dir is None:
        npz_root_dir = out_dir
    npz_root_dir = Path(npz_root_dir)

    scene_name = out_dir.name

    # --- Random Downsampling Variants ---
    for variant_idx in range(num_variants):
        rng_points = variant_rngs[variant_idx]

        # Random Sample without replacement since we verified N_total >= num_points
        idx = rng_points.choice(N_total, size=num_points, replace=False)

        # --- Optionally write downsampled TXT ---
        if save_downsampled_txt:
            txt_coords_mode = str(txt_coords).lower().strip()
            if txt_coords_mode not in ["centered", "raw"]:
                txt_coords_mode = "centered"

            txt_int_mode = str(txt_intensity).lower().strip()
            if txt_int_mode not in ["raw", "normalized"]:
                txt_int_mode = "raw"

            if downsampled_txt_dir is None:
                txt_out_dir = Path(npz_root_dir) / "downsampled_txt"
            else:
                txt_out_dir = Path(downsampled_txt_dir)
            txt_out_dir.mkdir(parents=True, exist_ok=True)

            txt_filename = f"{scene_name}.txt" if num_variants == 1 else f"{scene_name}_v{variant_idx}.txt"
            txt_path = txt_out_dir / txt_filename

            if txt_coords_mode == "raw":
                xyz_out = (pts[idx] + center_raw[None, :]).astype(np.float32)
            else:
                xyz_out = pts[idx].astype(np.float32)

            inten_out = intens_norm[idx] if txt_int_mode == "normalized" else intens[idx]
            lab_out = labels[idx].astype(np.float32)

            out_arr = np.column_stack([xyz_out, inten_out, lab_out]).astype(np.float32)

            np.savetxt(
                str(txt_path),
                out_arr,
                fmt=["%.6f", "%.6f", "%.6f", "%.6f", "%.0f"],
            )
            print(f"Saved downsampled TXT variant {variant_idx} to {txt_path}")

        # --- Save NPZ ---
        data_pts = pts[idx].astype(np.float32)
        data_int = intens_norm[idx].astype(np.float32)
        colors = np.stack([data_int, data_int, data_int], axis=-1)
        data = np.concatenate([data_pts, colors], axis=1)

        npz_filename = f"{scene_name}.npz" if num_variants == 1 else f"{scene_name}_v{variant_idx}.npz"
        npz_path = npz_root_dir / npz_filename

        np.savez_compressed(
            npz_path,
            images=images_arr,
            cam_poses=poses_arr,
            data=data,
            depths=depths_arr,
        )

        print(f"Saved Points2NeRF dataset variant {variant_idx} to {npz_path}")

    return center_raw


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    # Pre-parse only --config so we know which JSON to load before building the real parser.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    pre_args, _ = pre.parse_known_args()

    cfg = _load_config(pre_args.config)

    ap = argparse.ArgumentParser(
        description="Generate Points2NeRF-style RGBD dataset using RANDOM sampling and skipping small files."
    )
    ap.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH,
        help="Path to a JSON config providing defaults for most flags. Defaults to preprocessing/configs/default.json.",
    )
    ap.add_argument("--input_dir",    type=str,   default=cfg["input_dir"],
                    help="Directory with .txt point clouds.")
    ap.add_argument("--output_dir",   type=str,   default=cfg["output_dir"],
                    help="Output directory.")
    ap.add_argument("--num_views",    type=int,   default=cfg["num_views"],
                    help="Number of virtual camera views.")
    ap.add_argument("--img_hw",       type=int,   nargs=2,
                    default=[cfg["img_h"], cfg["img_w"]],
                    help="Image height and width.")
    ap.add_argument("--fov",          type=float, default=cfg["fov_x_radians"],
                    help="Horizontal FOV in radians.")
    ap.add_argument("--radius_scale", type=float, default=cfg["radius_scale"],
                    help="Distance margin factor.")
    ap.add_argument("--num_points",   type=int,   default=cfg["num_points"],
                    help="Required points. If file has less, it is SKIPPED.")
    ap.add_argument("--seed",         type=int,   default=cfg["seed"],
                    help="Random seed.")
    ap.add_argument("--num_variants", type=int,   default=cfg["num_variants"],
                    help="Number of random downsampled variants per file.")

    # Script-specific knobs not in default.json — keep CLI-only.
    ap.add_argument("--no_pngs", action="store_true",
                    help="Do NOT write debug PNGs, only .npz.")
    ap.add_argument("--save_downsampled_txt", action="store_true", default=True,
                    help="Save downsampled TXT files.")
    ap.add_argument("--downsampled_txt_dir", type=str, default=None,
                    help="Custom dir for downsampled TXT files.")
    ap.add_argument("--txt_intensity", choices=["raw", "normalized"], default="raw",
                    help="TXT output intensity mode.")
    ap.add_argument("--txt_coords", choices=["centered", "raw"], default="centered",
                    help="TXT output coordinate mode.")
    return ap.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.input_dir)
    root_out = Path(args.output_dir)
    root_out.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")

    txts = sorted(in_dir.glob("*.txt"))
    if len(txts) == 0:
        raise FileNotFoundError(f"No .txt point clouds found in {in_dir}")

    H, W = args.img_hw
    centers_rows = []
    skipped_files = []  # Track skipped files

    for txt_path in txts:
        scene_name = txt_path.stem
        h = hashlib.md5(scene_name.encode("utf-8")).hexdigest()
        scene_seed = args.seed + (int(h[:8], 16) % 2_000_000_000)

        debug_dir = root_out / scene_name

        print(f"\n=== Processing {txt_path.name} as scene '{scene_name}' ===")

        center_raw = process_scene_global(
            txt_paths=[txt_path],
            out_dir=debug_dir,
            num_views=args.num_views,
            img_h=H,
            img_w=W,
            fov_x=args.fov,
            radius_scale=args.radius_scale,
            num_points=args.num_points,
            seed=scene_seed,
            write_pngs=not args.no_pngs,
            num_variants=args.num_variants,
            npz_root_dir=root_out,
            save_downsampled_txt=bool(args.save_downsampled_txt),
            downsampled_txt_dir=args.downsampled_txt_dir,
            txt_intensity=args.txt_intensity,
            txt_coords=args.txt_coords,
        )

        if center_raw is not None:
            centers_rows.append([
                txt_path.name,
                float(center_raw[0]),
                float(center_raw[1]),
                float(center_raw[2]),
            ])
        else:
            # Track skipped file
            skipped_files.append([txt_path.name])

    # --- Write Centers Excel ---
    excel_path = root_out / "pointcloud_centers.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "centers"
    ws.append(["file_name", "x_centre", "y_centre", "z_centre"])
    for r in centers_rows:
        ws.append(r)
    wb.save(excel_path)
    print(f"\nSaved centers Excel to {excel_path}")

    # --- Write Skipped Files Excel ---
    if skipped_files:
        skipped_excel_path = root_out / "skipped_files.xlsx"
        wb_skip = Workbook()
        ws_skip = wb_skip.active
        ws_skip.title = "skipped"
        ws_skip.append(["file_name"])  # Header
        for r in skipped_files:
            ws_skip.append(r)
        wb_skip.save(skipped_excel_path)
        print(f"Saved SKIPPED files list to {skipped_excel_path}")
        print(f"Total skipped: {len(skipped_files)}")
    else:
        print("No files were skipped.")


if __name__ == "__main__":
    main()
