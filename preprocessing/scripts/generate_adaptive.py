#!/usr/bin/env python3
"""
generate.py

Generate Points2NeRF-style training data (with depth) from Seg2Tunnel point clouds
using the GLOBAL (full-scene) rendering mode.

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

Depth will be used for a squared-error depth loss in NeRF:
    L_depth = (D_pred - D_gt)^2  (masked where depth == 0).

Usage example:
  python generate.py \
      --input_dir /path/to/ring_txts \
      --output_dir ./p2n_dataset/ring_001 \
      --num_views 50 \
      --img_hw 200 200 \
      --radius_scale 0.85 \
      --num_points 2048
"""

import argparse
import json
import math
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
try:
    import open3d as o3d
except Exception:
    o3d = None  # open3d is not required for this script
import hashlib
from openpyxl import Workbook


# ---------------------------
# Adaptive downsampling (Multi-scale curvature + Feature-FPS + FPS-fill)
# ---------------------------
# This is adapted from Adaptive_multiscale_curv_fps.py and operates directly on in-memory XYZ arrays.
try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


def _estimate_mean_nn_distance(xyz: np.ndarray, sample_n: int = 10000, seed: int = 0) -> float:
    """Estimate mean nearest-neighbor distance (excluding self) using a subset."""
    if cKDTree is None:
        raise ImportError("scipy is required for adaptive downsampling (pip install scipy)")

    N = int(xyz.shape[0])
    rng = np.random.default_rng(int(seed))
    idx = np.arange(N)
    if N > sample_n:
        idx = rng.choice(idx, size=sample_n, replace=False)
    pts = xyz[idx]

    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=2, workers=-1)
    nn = d[:, 1]
    nn = nn[np.isfinite(nn)]
    if nn.size == 0:
        return 0.0
    return float(np.mean(nn))


def _voxel_unique_indices(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    """Keep one representative per voxel (first occurrence). Returns indices into xyz."""
    if voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")
    mn = xyz.min(axis=0)
    coords = np.floor((xyz - mn) / voxel_size).astype(np.int32)
    _, idx = np.unique(coords, axis=0, return_index=True)
    return idx.astype(np.int64)


def _prethin_points(
    xyz: np.ndarray,
    method: str = "voxel",
    threshold: int = 200000,
    target: int = 150000,
    seed: int = 0,
    verbose: bool = True,
) -> np.ndarray:
    """
    Pre-thin very dense clouds before curvature.

    Returns indices into the original xyz.
    """
    N = int(xyz.shape[0])
    method = str(method).lower()

    if method == "none" or N <= int(threshold):
        return np.arange(N, dtype=np.int64)

    rng = np.random.default_rng(int(seed))

    if method == "random":
        if N <= int(target):
            return np.arange(N, dtype=np.int64)
        idx = rng.choice(N, size=int(target), replace=False)
        if verbose:
            print(f"[Prethin-random] {N} -> {len(idx)}")
        return idx.astype(np.int64)

    if method != "voxel":
        raise ValueError(f"Unknown prethin method: {method}")

    d_nn = _estimate_mean_nn_distance(xyz, sample_n=min(20000, N), seed=int(seed))
    if d_nn <= 0:
        if verbose:
            print("[Prethin-voxel] failed to estimate NN spacing; fallback to random.")
        if N <= int(target):
            return np.arange(N, dtype=np.int64)
        return rng.choice(N, size=int(target), replace=False).astype(np.int64)

    if verbose:
        print(f"[Prethin-voxel] N={N}, target={int(target)}, estimated mean_nn={d_nn:.6f}")

    low = 0.25 * d_nn
    high = 8.0 * d_nn

    def count_for(v):
        return len(_voxel_unique_indices(xyz, v))

    cnt_low = count_for(low)
    cnt_high = count_for(high)

    tries = 0
    while cnt_high > int(target) and tries < 8:
        high *= 2.0
        cnt_high = count_for(high)
        tries += 1

    tries = 0
    while cnt_low < int(target) and tries < 8:
        low *= 0.5
        if low < 1e-9:
            break
        cnt_low = count_for(low)
        tries += 1

    if not (cnt_low >= int(target) >= cnt_high):
        if verbose:
            print(
                f"[Prethin-voxel] could not bracket target. cnt_low={cnt_low}, cnt_high={cnt_high}. "
                "Fallback to random."
            )
        if N <= int(target):
            return np.arange(N, dtype=np.int64)
        return rng.choice(N, size=int(target), replace=False).astype(np.int64)

    best_idx = None
    best_diff = float("inf")
    best_cnt = None
    best_vox = None

    for it in range(10):
        mid = 0.5 * (low + high)
        idx_mid = _voxel_unique_indices(xyz, mid)
        cnt_mid = len(idx_mid)
        diff = abs(cnt_mid - int(target))

        if diff < best_diff:
            best_diff = diff
            best_idx = idx_mid
            best_cnt = cnt_mid
            best_vox = mid

        if verbose:
            print(f"[Prethin-voxel] iter {it+1}/10: voxel={mid:.6f}, count={cnt_mid}")

        if cnt_mid > int(target):
            low = mid
        else:
            high = mid

    idx = best_idx
    if verbose:
        print(f"[Prethin-voxel] best voxel={best_vox:.6f} -> {best_cnt} points")

    if len(idx) > int(target):
        idx = rng.choice(idx, size=int(target), replace=False)
        if verbose:
            print(f"[Prethin-voxel] trim {best_cnt} -> {len(idx)}")

    return np.asarray(idx, dtype=np.int64)


def _choose_k_auto(N: int):
    """Heuristic k selection based on point count after prethin."""
    N = int(N)
    if N <= 60000:
        return 32, 120
    if N <= 200000:
        return 24, 80
    return 24, 80


def _curvature_knn(xyz: np.ndarray, k: int, chunk: int = 20000, verbose: bool = True) -> np.ndarray:
    """Curvature proxy per point via local PCA on kNN: curv = λ0 / (λ0+λ1+λ2)."""
    if cKDTree is None:
        raise ImportError("scipy is required for adaptive downsampling (pip install scipy)")

    N = int(xyz.shape[0])
    k = int(k)
    if k < 5:
        raise ValueError("k should be >= 5 for stable curvature estimation.")
    if k >= N:
        k = max(N - 1, 5)

    tree = cKDTree(xyz)
    curv = np.zeros(N, dtype=np.float32)

    for start in range(0, N, int(chunk)):
        end = min(N, start + int(chunk))
        pts = xyz[start:end]

        _, idx = tree.query(pts, k=k, workers=-1)
        neigh = xyz[idx]

        mu = neigh.mean(axis=1, keepdims=True)
        X = neigh - mu
        cov = np.einsum("mki,mkj->mij", X, X) / max(k - 1, 1)
        evals = np.linalg.eigvalsh(cov).astype(np.float32)

        denom = (evals[:, 0] + evals[:, 1] + evals[:, 2]) + 1e-12
        curv[start:end] = evals[:, 0] / denom

        if verbose and (start == 0 or (start // int(chunk)) % 10 == 0 or end == N):
            print(f"[Curvature k={k}] processed {end}/{N}")

    return curv


def _fps_indices(xyz: np.ndarray, n_samples: int, start_idx: int = 0, verbose: bool = True) -> np.ndarray:
    """Classic FPS O(N*n_samples) on xyz (N,3). Returns indices in selection order."""
    N = int(xyz.shape[0])
    n_samples = int(n_samples)
    if n_samples <= 0:
        return np.array([], dtype=np.int64)
    if n_samples >= N:
        return np.arange(N, dtype=np.int64)

    start_idx = int(start_idx)
    if not (0 <= start_idx < N):
        start_idx = 0

    inds = np.empty(n_samples, dtype=np.int64)
    inds[0] = start_idx

    diff = xyz - xyz[start_idx]
    dists = np.einsum("ij,ij->i", diff, diff).astype(np.float32)

    for i in range(1, n_samples):
        idx = int(np.argmax(dists))
        inds[i] = idx

        diff = xyz - xyz[idx]
        new_d = np.einsum("ij,ij->i", diff, diff).astype(np.float32)
        dists = np.minimum(dists, new_d)

        if verbose and (i % 500 == 0 or i == n_samples - 1):
            print(f"[Feature-FPS] selected {i+1}/{n_samples}")

    return inds


def _fps_fill_with_seeds(
    xyz: np.ndarray,
    target_n: int,
    seed_indices: np.ndarray,
    seed: int = 0,
    verbose: bool = True,
) -> np.ndarray:
    """FPS-fill on full xyz starting from seed_indices (keeps them, then fills to target_n)."""
    if cKDTree is None:
        raise ImportError("scipy is required for adaptive downsampling (pip install scipy)")

    N = int(xyz.shape[0])
    target_n = int(target_n)
    if target_n >= N:
        return np.arange(N, dtype=np.int64)

    seed_indices = np.unique(seed_indices.astype(np.int64))
    seed_indices = seed_indices[(seed_indices >= 0) & (seed_indices < N)]

    selected = np.zeros(N, dtype=bool)
    selected[seed_indices] = True
    sel_list = list(seed_indices.tolist())

    rng = np.random.default_rng(int(seed))
    if len(sel_list) == 0:
        start = int(rng.integers(0, N))
        selected[start] = True
        sel_list = [start]

    seed_xyz = xyz[np.array(sel_list, dtype=np.int64)]
    tree = cKDTree(seed_xyz)
    d, _ = tree.query(xyz, k=1, workers=-1)
    dists = (d.astype(np.float32) ** 2)
    dists[selected] = -1.0

    while len(sel_list) < target_n:
        idx = int(np.argmax(dists))
        selected[idx] = True
        sel_list.append(idx)

        diff = xyz - xyz[idx]
        new_d = np.einsum("ij,ij->i", diff, diff).astype(np.float32)
        dists = np.minimum(dists, new_d)
        dists[selected] = -1.0

        if verbose and (len(sel_list) % 500 == 0 or len(sel_list) == target_n):
            print(f"[FPS-fill] selected {len(sel_list)}/{target_n}")

    return np.array(sel_list, dtype=np.int64)


class AdaptiveMultiScaleCurvFPSSampler:
    """Cache-heavy parts (prethin + curvature + score) once, then sample indices many times."""

    def __init__(
        self,
        xyz: np.ndarray,
        seed: int = 0,
        prethin_method: str = "voxel",
        prethin_threshold: int = 200000,
        prethin_target: int = 150000,
        auto_k: bool = True,
        k_small: int = None,
        k_large: int = None,
        chunk: int = 20000,
        w_small: float = 0.7,
        w_large: float = 0.3,
        clip_pct: float = 99.5,
        verbose: bool = True,
    ):
        if cKDTree is None:
            raise ImportError("scipy is required for adaptive downsampling (pip install scipy)")

        self.xyz_all = xyz.astype(np.float32, copy=False)
        self.N_raw = int(self.xyz_all.shape[0])
        self.verbose = bool(verbose)

        # Pre-thin indices into raw
        self.pre_idx = _prethin_points(
            xyz=self.xyz_all,
            method=prethin_method,
            threshold=int(prethin_threshold),
            target=int(prethin_target),
            seed=int(seed),
            verbose=self.verbose,
        )

        self.xyz = self.xyz_all[self.pre_idx]
        self.N_pre = int(self.xyz.shape[0])
        if self.verbose:
            print(f"[AdaptiveDownsample] raw N={self.N_raw} -> pre N={self.N_pre} (method={prethin_method})")

        # Choose k
        if (k_small is not None) and (k_large is not None):
            self.k_small, self.k_large = int(k_small), int(k_large)
        else:
            if bool(auto_k) or (k_small is None and k_large is None):
                self.k_small, self.k_large = _choose_k_auto(self.N_pre)
            else:
                ks, kl = _choose_k_auto(self.N_pre)
                self.k_small = int(k_small) if k_small is not None else int(ks)
                self.k_large = int(k_large) if k_large is not None else int(kl)

        if self.verbose:
            print(f"[AdaptiveDownsample] k_small={self.k_small}, k_large={self.k_large}")

        # Curvature + score
        curv_s = _curvature_knn(self.xyz, k=self.k_small, chunk=int(chunk), verbose=self.verbose)
        curv_l = _curvature_knn(self.xyz, k=self.k_large, chunk=int(chunk), verbose=self.verbose)

        score = float(w_small) * curv_s + float(w_large) * curv_l

        clip_pct = float(clip_pct)
        if clip_pct > 0:
            hi = np.percentile(score, clip_pct)
            score = np.clip(score, 0.0, hi)
            if self.verbose:
                print(f"[AdaptiveDownsample] score clip at p{clip_pct} -> {hi:.6e}")

        self.score = score.astype(np.float32, copy=False)

    def sample(
        self,
        target_n: int,
        seed: int = 0,
        feature_frac: float = 0.55,
        feature_pool_mult: float = 6.0,
        sort_indices: bool = False,
        topk_start: int = 64,
    ) -> np.ndarray:
        """Return indices into the ORIGINAL xyz array (self.xyz_all)."""
        target_n = int(target_n)
        if target_n <= 0:
            return np.array([], dtype=np.int64)

        # If target exceeds raw, match old behavior: pad with replacement
        if self.N_raw < target_n:
            rng = np.random.default_rng(int(seed))
            return rng.choice(self.N_raw, size=target_n, replace=True).astype(np.int64)

        if self.N_raw == target_n:
            return np.arange(self.N_raw, dtype=np.int64)

        # If pre-thinned cloud is smaller than target, fall back to random on raw
        # (rare; can happen if prethin_target < target_n)
        if self.N_pre < target_n:
            rng = np.random.default_rng(int(seed))
            return rng.choice(self.N_raw, size=target_n, replace=False).astype(np.int64)

        rng = np.random.default_rng(int(seed))

        # Feature budget
        feature_frac = max(0.0, min(1.0, float(feature_frac)))
        n_feat = int(round(feature_frac * target_n))
        n_feat = max(0, min(n_feat, target_n))

        if self.verbose:
            print(f"[AdaptiveDownsample] budget total={target_n}, feature={n_feat}, fill={target_n - n_feat}")

        # Candidate pool (highest-score points)
        if n_feat > 0:
            pool_mult = max(1.0, float(feature_pool_mult))
            pool_size = int(round(pool_mult * n_feat))
            pool_size = min(pool_size, self.N_pre)

            pool_idx = np.argpartition(self.score, -pool_size)[-pool_size:]
            pool_xyz = self.xyz[pool_idx]

            # Randomize start among top-K in pool for variant diversity
            k = min(int(topk_start), pool_size)
            top_local = np.argpartition(self.score[pool_idx], -k)[-k:]
            start_local = int(rng.choice(top_local))

            local_sel = _fps_indices(pool_xyz, n_samples=n_feat, start_idx=start_local, verbose=self.verbose)
            feat_idx = pool_idx[local_sel].astype(np.int64)

            # Unique, then ensure we still have n_feat
            feat_idx = np.unique(feat_idx)
            if len(feat_idx) < n_feat:
                missing = n_feat - len(feat_idx)
                remaining = np.setdiff1d(pool_idx, feat_idx, assume_unique=False)
                extra = remaining[np.argsort(self.score[remaining])[-missing:]]
                feat_idx = np.unique(np.concatenate([feat_idx, extra]).astype(np.int64))

            if self.verbose:
                print(f"[AdaptiveDownsample] feature seeds (pre-space): {len(feat_idx)}")
        else:
            feat_idx = np.array([], dtype=np.int64)

        # FPS-fill (pre-space)
        sel_pre = _fps_fill_with_seeds(
            xyz=self.xyz,
            target_n=target_n,
            seed_indices=feat_idx,
            seed=int(seed),
            verbose=self.verbose,
        )

        sel_raw = self.pre_idx[sel_pre]
        if sort_indices:
            sel_raw = np.sort(sel_raw)

        return sel_raw.astype(np.int64)




# ------------------------------
# Sphere sampling for viewpoints
# ------------------------------
def fibonacci_sphere(n: int) -> np.ndarray:
    """Return n roughly-uniform directions on the unit sphere."""
    dirs = []
    golden = (1 + 5 ** 0.5) / 2
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = (1 - z * z) ** 0.5
        phi = 2 * math.pi * i / golden
        dirs.append(np.array([r * math.cos(phi), r * math.sin(phi), z], dtype=np.float32))
    return np.stack(dirs, axis=0)


# --------------------------------------
# Virtual camera + full-scene projection
# --------------------------------------
def look_at_c2w(
    origin: np.ndarray,
    target: np.ndarray,
    up_hint: np.ndarray = np.array([0, 1, 0], dtype=np.float32),
) -> np.ndarray:
    """
    Construct camera-to-world transform matrix (Blender/NeRF style):
      - camera looks along -Z in camera space
      - we store a matrix that maps camera coords -> world coords
      - columns are [right, up, forward, origin]
    """
    origin = origin.astype(np.float32)
    target = target.astype(np.float32)
    up_hint = up_hint.astype(np.float32)

    forward = target - origin
    f = forward / (np.linalg.norm(forward) + 1e-8)

    # right = normalize(cross(f, up_hint))
    r = np.cross(f, up_hint)
    r_norm = np.linalg.norm(r)
    if r_norm < 1e-6:
        up_hint = np.array([0, 0, 1], dtype=np.float32)
        r = np.cross(f, up_hint)
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-6:
            up_hint = np.array([1, 0, 0], dtype=np.float32)
            r = np.cross(f, up_hint)
            r_norm = np.linalg.norm(r)
    r = r / (r_norm + 1e-8)

    u = np.cross(r, f)
    r = r / (np.linalg.norm(r) + 1e-8)
    u = u / (np.linalg.norm(u) + 1e-8)
    f = f / (np.linalg.norm(f) + 1e-8)

    c2w = np.eye(4, dtype=np.float32)
    c2w[0:3, 0] = r
    c2w[0:3, 1] = u
    c2w[0:3, 2] = -f  # camera +Z axis is “back”; camera looks along -Z
    c2w[0:3, 3] = origin
    return c2w


def project_points_fullscene(
    pts: np.ndarray,
    intens: np.ndarray,
    c2w: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    img_w: int,
    img_h: int,
    depth_clip: Tuple[float, float] = (0.0, 1000.0),
):
    """
    Project the entire point cloud into the image plane with a Z-buffer.

    Returns:
      rgb:   (H,W) uint8 grayscale [0,255], bg = 0
      depth: (H,W) uint16 depth in mm, bg = 0
      hit:   (H,W) bool mask where we saw at least one point
    """
    # world->camera
    w2c = np.linalg.inv(c2w)
    Pw = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)], axis=1)
    Pc = (w2c @ Pw.T).T[:, :3]
    z = -Pc[:, 2]
    valid = z > 1e-4
    if not np.any(valid):
        rgb = np.zeros((img_h, img_w), dtype=np.uint8)
        depth = np.zeros((img_h, img_w), dtype=np.uint16)
        hit = np.zeros((img_h, img_w), dtype=bool)
        return rgb, depth, hit

    Pc = Pc[valid]
    z = z[valid]
    I = intens[valid]

    u = fx * (Pc[:, 0] / z) + cx
    v = -fy * (Pc[:, 1] / z) + cy

    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    in_img = (ui >= 0) & (ui < img_w) & (vi >= 0) & (vi < img_h)
    if not np.any(in_img):
        rgb = np.zeros((img_h, img_w), dtype=np.uint8)
        depth = np.zeros((img_h, img_w), dtype=np.uint16)
        hit = np.zeros((img_h, img_w), dtype=bool)
        return rgb, depth, hit

    ui = ui[in_img]
    vi = vi[in_img]
    z = z[in_img]
    I = I[in_img]

    # Normalize intensity per view to [0,255]
    I_clamped = np.clip(I, 0.0, 1.0)
    Iu8 = np.round(I_clamped * 255.0).astype(np.uint8)

    H, W = img_h, img_w
    rgb = np.zeros((H, W), dtype=np.uint8)
    depth = np.zeros((H, W), dtype=np.uint16)
    hit = np.zeros((H, W), dtype=bool)

    zmin, zmax = depth_clip
    for px, py, pz, val in zip(ui, vi, z, Iu8):
        if not (zmin <= pz <= zmax):
            continue
        # Z-buffer: keep nearest
        if (not hit[py, px]) or pz < (depth[py, px] / 1000.0):
            hit[py, px] = True
            rgb[py, px] = val
            depth[py, px] = int(np.clip(round(pz * 1000.0), 0, 65535))  # mm

    return rgb, depth, hit


# ---------------------------
# IO: read Seg2Tunnel .txt
# ---------------------------
def load_txt_point_cloud(path: Path):
    """
    Returns:
      pts:      (N,3) float32 XYZ
      intensity:(N,)  float32
      label:    (N,)  int (unused here)
    """
    arr = np.loadtxt(str(path), dtype=np.float32)
    if arr.shape[1] != 5:
        raise ValueError(f"{path} must have 5 columns: x y z intensity label")
    pts = arr[:, :3].astype(np.float32)
    intensity = arr[:, 3].astype(np.float32)
    label = arr[:, 4].astype(np.int32)
    return pts, intensity, label


# ---------------------------
# Scene processing (GLOBAL)
# ---------------------------
def process_scene_global(
    txt_paths,
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
    downsample_method: str = "adaptive",
    downsample_cfg: dict = None,
    save_downsampled_txt: bool = True,
    downsampled_txt_dir=None,
    txt_intensity: str = "raw",      # "raw" or "normalized"
    txt_coords: str = "centered",    # "centered" or "raw"
):
    """
    Global mode: render the full scene from cameras on a sphere and
    produce a Points2NeRF-style .npz with depth.

    - fov_x: horizontal FOV in radians (e.g. 0.6911112070083618)
    - radius_scale: how far the camera is from the scene (relative to bbox size)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if write_pngs:
        (out_dir / "images").mkdir(exist_ok=True)
        (out_dir / "depths").mkdir(exist_ok=True)

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

    # --- Compute center in ORIGINAL coordinates (per txt / per scene) ---
    mins_raw = pts.min(axis=0)                 # (3,)
    maxs_raw = pts.max(axis=0)                 # (3,)
    center_raw = 0.5 * (mins_raw + maxs_raw)   # (3,) bbox midpoint

    # --- Center at origin (remove translation in x,y,z) ---
    pts = pts - center_raw[None, :]



    # --- NEW: global intensity normalisation (shared by images and data) ---
    if np.ptp(intens) > 0:
        intens_norm = (intens - intens.min()) / (intens.max() - intens.min())
    else:
        intens_norm = np.zeros_like(intens, dtype=np.float32)


    # Scene bbox & center
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    center = (mins + maxs) / 2.0
    half_diag = 0.5 * np.linalg.norm(maxs - mins)  # conservative "radius"

    # Camera distance from center
    distance = (half_diag / math.tan(fov_x / 2.0)) * radius_scale

    # Intrinsics from FOV (match training: focal = 0.5*W / tan(FOV/2))
    fx = fy = 0.5 * img_w / math.tan(fov_x / 2.0)
    cx = img_w / 2.0
    cy = img_h / 2.0

    camera_angle_x = fov_x



    # For .npz
    all_images = []
    all_depths = []
    all_poses = []
    frames = []

    # View directions on unit sphere
 
    if num_variants < 1:
        num_variants = 1

    ss = np.random.SeedSequence(seed)
    children = ss.spawn(1 + num_variants)          # [0]=views, [1:]=variants
    rng_views = np.random.default_rng(children[0])
    variant_rngs = [np.random.default_rng(s) for s in children[1:]]

    max_attempts = num_views * 20
    attempts = 0

    while len(all_images) < num_views and attempts < max_attempts:
        # sample ONE direction at a time so we can resample failures
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
        img_name = f"{view_idx:05d}.png"   # <-- define regardless of write_pngs

        # training image (white background)
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

            # optional depth visualisation (keep your existing code if you want)

        frames.append({
            "file_path": f"./images/{img_name}",
            "transform_matrix": c2w.tolist(),
        })

    if len(all_images) < num_views:
        raise RuntimeError(
            f"Only got {len(all_images)}/{num_views} valid views after {attempts} attempts. "
            f"Increase max_attempts or check camera distance / point cloud scale."
        )


    if len(all_images) == 0:
        raise RuntimeError("Global mode: no valid views produced. Check FOV/radius_scale.")

    # ---------- Save transforms.json (debug / optional) ----------
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

    # ---------- Build 'data' array [N_pts,6] for Points2NeRF ----------
    # First stack images / depths / poses (shared for all variants)
    images_arr = np.stack(all_images, axis=0).astype(np.float32)   # [V,H,W,3]
    depths_arr = np.stack(all_depths, axis=0).astype(np.float32)   # [V,H,W]
    poses_arr  = np.stack(all_poses, axis=0).astype(np.float32)    # [V,4,4]

    print(f"[global] wrote {images_arr.shape[0]} views to {out_dir}")

    # Where to save .npz files
    if npz_root_dir is None:
        npz_root_dir = out_dir
    npz_root_dir = Path(npz_root_dir)

    scene_name = out_dir.name
    N_total = pts.shape[0]

    # --- Downsampling setup (for NPZ 'data' only) ---
    ds_method = (downsample_method or "random").lower()
    ds_cfg = downsample_cfg or {}

    adaptive_sampler = None
    if ds_method in ["adaptive", "adaptive_ms_curv_fps", "mscurv_fps"] and num_points is not None:
        adaptive_sampler = AdaptiveMultiScaleCurvFPSSampler(
            xyz=pts,
            seed=int(seed),
            prethin_method=ds_cfg.get("prethin_method", "voxel"),
            prethin_threshold=int(ds_cfg.get("prethin_threshold", 200000)),
            prethin_target=int(ds_cfg.get("prethin_target", 150000)),
            auto_k=bool(ds_cfg.get("auto_k", True)),
            k_small=ds_cfg.get("k_small", None),
            k_large=ds_cfg.get("k_large", None),
            chunk=int(ds_cfg.get("chunk", 20000)),
            w_small=float(ds_cfg.get("w_small", 0.7)),
            w_large=float(ds_cfg.get("w_large", 0.3)),
            clip_pct=float(ds_cfg.get("clip_pct", 99.5)),
            verbose=bool(ds_cfg.get("verbose", True)),
        )

    if adaptive_sampler is None and ds_method not in ["random", "adaptive", "adaptive_ms_curv_fps", "mscurv_fps"]:
        print(f"[Warn] Unknown downsample_method='{downsample_method}'. Falling back to random.")

    for variant_idx in range(num_variants):
        rng_points = variant_rngs[variant_idx]

        # --- Downsample for this variant ---
        if num_points is None:
            idx = np.arange(N_total, dtype=np.int64)
        else:
            if adaptive_sampler is not None and ds_method in ["adaptive", "adaptive_ms_curv_fps", "mscurv_fps"]:
                # Derive a stable integer seed per variant
                variant_seed = int(rng_points.integers(0, 2_000_000_000))
                idx = adaptive_sampler.sample(
                    target_n=int(num_points),
                    seed=variant_seed,
                    feature_frac=float(ds_cfg.get("feature_frac", 0.55)),
                    feature_pool_mult=float(ds_cfg.get("feature_pool_mult", 6.0)),
                    sort_indices=bool(ds_cfg.get("sort_indices", False)),
                    topk_start=int(ds_cfg.get("topk_start", 64)),
                )
            else:
                # Original random downsample
                if N_total >= num_points:
                    idx = rng_points.choice(N_total, size=num_points, replace=False)
                else:
                    idx = rng_points.choice(N_total, size=num_points, replace=True)


        # --- Optionally write a downsampled TXT for this variant ---
        if save_downsampled_txt and num_points is not None:
            txt_coords_mode = str(txt_coords).lower().strip()
            if txt_coords_mode not in ["centered", "raw"]:
                txt_coords_mode = "centered"

            txt_int_mode = str(txt_intensity).lower().strip()
            if txt_int_mode not in ["raw", "normalized"]:
                txt_int_mode = "raw"

            # Choose output directory
            if downsampled_txt_dir is None:
                txt_out_dir = Path(npz_root_dir) / "downsampled_txt"
            else:
                txt_out_dir = Path(downsampled_txt_dir)
            txt_out_dir.mkdir(parents=True, exist_ok=True)

            if num_variants == 1:
                txt_filename = f"{scene_name}.txt"
            else:
                txt_filename = f"{scene_name}_v{variant_idx}.txt"
            txt_path = txt_out_dir / txt_filename

            # Assemble 5-column output: x y z intensity label
            if txt_coords_mode == "raw":
                xyz_out = (pts[idx] + center_raw[None, :]).astype(np.float32)
            else:
                xyz_out = pts[idx].astype(np.float32)

            if txt_int_mode == "normalized":
                inten_out = intens_norm[idx].astype(np.float32)
            else:
                inten_out = intens[idx].astype(np.float32)

            # NOTE: np.savetxt applies python '%' formatting per column.
            # If we mix float+int via column_stack, NumPy upcasts to float and '%d' will error.
            # So we store labels as float and format them as integers via '%.0f'.
            lab_out = labels[idx].astype(np.float32)
            out_arr = np.column_stack([xyz_out, inten_out, lab_out]).astype(np.float32)

            # Format: x y z intensity label
            np.savetxt(
                str(txt_path),
                out_arr,
                fmt=["%.6f", "%.6f", "%.6f", "%.6f", "%.0f"],
            )
            print(f"Saved downsampled TXT variant {variant_idx} to {txt_path}")


        data_pts = pts[idx].astype(np.float32)            # (N_pts,3)
        data_int = intens_norm[idx].astype(np.float32)    # (N_pts,)

        # Use the same globally-normalised intensity as three identical channels
        colors = np.stack([data_int, data_int, data_int], axis=-1)  # (N_pts,3) in [0,1]


        data = np.concatenate(
            [data_pts.astype(np.float32), colors.astype(np.float32)], axis=1
        )  # (N_pts,6) = [x,y,z,r,g,b]

        # --- name of this variant's .npz ---
        if num_variants == 1:
            npz_filename = f"{scene_name}.npz"
        else:
            npz_filename = f"{scene_name}_v{variant_idx}.npz"

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



def random_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    u = rng.random(n, dtype=np.float32)
    v = rng.random(n, dtype=np.float32)
    theta = 2.0 * math.pi * u
    z = 2.0 * v - 1.0
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=1).astype(np.float32)

# ---------------------------
# CLI
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Generate Points2NeRF-style RGBD dataset from Seg2Tunnel point clouds (global mode)."
    )
    ap.add_argument(
        "--input_dir",
        type=str, 
        default =  "/mnt/c/Users/zy349/Documents/Points2NeRF/Seg2Tunnel/T1/1-9_1-17",
        # default =  "./exp/T1",
        help="Directory with .txt point clouds (x y z intensity label).",
    )
    ap.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/c/Users/zy349/Documents/Points2NeRF/Seg2Tunnel_multi_culv/T1",
        help="Output directory for images, depths, transforms.json, and .npz",
    )
    ap.add_argument(
        "--num_views",
        type=int,
        default=50,
        help="Number of virtual camera views to generate.",
    )
    ap.add_argument(
        "--img_hw",
        type=int,
        nargs=2,
        default=[650, 650],
        help="Image height and width, e.g. --img_hw 200 200 (Points2NeRF uses 200x200).",
    )
    ap.add_argument(
        "--fov",
        type=float,
        default=0.6911112070083618,
        help="Horizontal field-of-view in radians (must match training FOV).",
    )
    ap.add_argument(
        "--radius_scale",
        type=float,
        default=0.85,
        help="Distance margin factor (camera distance from scene center).",
    )
    ap.add_argument(
        "--num_points",
        type=int,
        default=8192,
        help="Number of points to sample for 'data' in the .npz.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for view sampling and point sampling.",
    )
    ap.add_argument(
        "--no_pngs",
        action="store_true",
        # default= True,
        help="If set, do NOT write debug PNGs, only the .npz.",
    )
    # --- Downsampling method for NPZ 'data' ---
    ap.add_argument(
        "--downsample",
        choices=["random", "adaptive"],
        default="adaptive",
        help="Downsampling method for the NPZ 'data' points. 'adaptive' = multi-scale curvature + FPS.",
    )

    # Adaptive downsampling options (used when --downsample adaptive)
    ap.add_argument("--prethin_method", choices=["none", "random", "voxel"], default="voxel",
                    help="Adaptive: pre-thin method for huge rings")
    ap.add_argument("--prethin_threshold", type=int, default=200000,
                    help="Adaptive: if N_raw > threshold, apply pre-thinning")
    ap.add_argument("--prethin_target", type=int, default=150000,
                    help="Adaptive: target count after pre-thinning")

    ap.add_argument("--no_auto_k", action="store_true",
                    help="Adaptive: disable auto k selection and use --k_small/--k_large")
    ap.add_argument("--k_small", type=int, default=None, help="Adaptive: kNN for small-scale curvature")
    ap.add_argument("--k_large", type=int, default=None, help="Adaptive: kNN for large-scale curvature")
    ap.add_argument("--curv_chunk", type=int, default=20000,
                    help="Adaptive: chunk size for curvature computation")

    ap.add_argument("--w_small", type=float, default=0.7, help="Adaptive: weight for small-scale curvature")
    ap.add_argument("--w_large", type=float, default=0.3, help="Adaptive: weight for large-scale curvature")
    ap.add_argument("--clip_pct", type=float, default=99.5,
                    help="Adaptive: percentile clip for score (0 disables)")

    ap.add_argument("--feature_frac", type=float, default=0.55,
                    help="Adaptive: fraction reserved for feature points")
    ap.add_argument("--feature_pool_mult", type=float, default=6.0,
                    help="Adaptive: candidate pool size multiplier")
    ap.add_argument("--topk_start", type=int, default=64,
                    help="Adaptive: randomize Feature-FPS start among top-K score points in the pool")

    ap.add_argument("--adaptive_sort_indices", action="store_true",
                    help="Adaptive: sort selected indices to preserve original order")
    ap.add_argument("--no_adaptive_verbose", action="store_true",
                    help="Adaptive: disable progress printing")
    ap.add_argument(
        "--num_variants",
        type=int,
        default=8,
        help="Number of different downsampled point clouds (variants) per .txt file.",
    )

    # --- Optional: also save downsampled point clouds as TXT ---
    ap.add_argument(
        "--save_downsampled_txt",
        action="store_true",
        default=True,
        help="If set, also save each variant's downsampled point cloud as a .txt (x y z intensity label).",
    )
    ap.add_argument(
        "--downsampled_txt_dir",
        type=str,
        default=None,
        help="Directory to write downsampled TXT files. Default: <output_dir>/downsampled_txt",
    )
    ap.add_argument(
        "--txt_intensity",
        choices=["raw", "normalized"],
        default="raw",
        help="When saving TXT, write raw intensity or normalized intensity (0-1).",
    )
    ap.add_argument(
        "--txt_coords",
        choices=["centered", "raw"],
        default="centered",
        help="When saving TXT, write centered XYZ (used by NPZ) or raw XYZ (original coordinates).",
    )
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
    for txt_path in txts:
        scene_name = txt_path.stem  # e.g. "1-2-61_1"


        h = hashlib.md5(scene_name.encode("utf-8")).hexdigest()
        scene_seed = args.seed + (int(h[:8], 16) % 2_000_000_000)


        # This folder is only for debug PNGs + transforms.json for this scene
        debug_dir = root_out / scene_name

        print(f"\n=== Processing {txt_path.name} as scene '{scene_name}' ===")
        print(f"scene_name={scene_name}, scene_seed={scene_seed}")


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
            write_pngs= not args.no_pngs,
            num_variants=args.num_variants,
            npz_root_dir=root_out,   # <-- ALL .npz files go here
            downsample_method=args.downsample,
            downsample_cfg={
                'prethin_method': args.prethin_method,
                'prethin_threshold': args.prethin_threshold,
                'prethin_target': args.prethin_target,
                'auto_k': (not args.no_auto_k),
                'k_small': args.k_small,
                'k_large': args.k_large,
                'chunk': args.curv_chunk,
                'w_small': args.w_small,
                'w_large': args.w_large,
                'clip_pct': args.clip_pct,
                'feature_frac': args.feature_frac,
                'feature_pool_mult': args.feature_pool_mult,
                'topk_start': args.topk_start,
                'sort_indices': args.adaptive_sort_indices,
                'verbose': (not args.no_adaptive_verbose),
            },
            save_downsampled_txt=bool(args.save_downsampled_txt),
            downsampled_txt_dir=args.downsampled_txt_dir,
            txt_intensity=args.txt_intensity,
            txt_coords=args.txt_coords,
        )

        centers_rows.append([
            txt_path.name,
            float(center_raw[0]),
            float(center_raw[1]),
            float(center_raw[2]),
        ])
    
    # --- Write all centers to one Excel file ---
    excel_path = root_out / "pointcloud_centers.xlsx"

    wb = Workbook()
    ws = wb.active
    ws.title = "centers"
    ws.append(["file_name", "x_centre", "y_centre", "z_centre"])

    for r in centers_rows:
        ws.append(r)

    wb.save(excel_path)
    print(f"Saved centers Excel to {excel_path}")




if __name__ == "__main__":
    main()