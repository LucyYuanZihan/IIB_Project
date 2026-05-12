"""Point-cloud → image projection utilities.

`project_points_fullscene` renders a point cloud into an RGBD image through a
virtual camera (Z-buffer, intensity-based grayscale). `load_txt_point_cloud`
reads the Seg2Tunnel 5-column format `(x, y, z, intensity, label)`.

The dataset-generation scripts in `preprocessing/scripts/` currently inline
equivalent implementations; this module is the canonical place to import from
in new code.
"""

from pathlib import Path
from typing import Tuple
import numpy as np


def load_txt_point_cloud(path: Path):
    """Return (xyz[N,3] float32, intensity[N] float32, label[N] int32)."""
    arr = np.loadtxt(str(path), dtype=np.float32)
    if arr.shape[1] != 5:
        raise ValueError(f"{path} must have 5 columns: x y z intensity label")
    pts = arr[:, :3].astype(np.float32)
    intensity = arr[:, 3].astype(np.float32)
    label = arr[:, 4].astype(np.int32)
    return pts, intensity, label


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
    """Project the entire point cloud to (rgb, depth, hit) via a Z-buffer.

    Returns:
      rgb   (H,W) uint8 grayscale [0,255], bg = 0
      depth (H,W) uint16 depth in mm, bg = 0
      hit   (H,W) bool mask
    """
    w2c = np.linalg.inv(c2w)
    Pw = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)], axis=1)
    Pc = (w2c @ Pw.T).T[:, :3]
    z = -Pc[:, 2]
    valid = z > 1e-4
    if not np.any(valid):
        return (np.zeros((img_h, img_w), dtype=np.uint8),
                np.zeros((img_h, img_w), dtype=np.uint16),
                np.zeros((img_h, img_w), dtype=bool))

    Pc = Pc[valid]
    z = z[valid]
    I = intens[valid]

    u = fx * (Pc[:, 0] / z) + cx
    v = -fy * (Pc[:, 1] / z) + cy

    ui = np.round(u).astype(np.int32)
    vi = np.round(v).astype(np.int32)
    in_img = (ui >= 0) & (ui < img_w) & (vi >= 0) & (vi < img_h)
    if not np.any(in_img):
        return (np.zeros((img_h, img_w), dtype=np.uint8),
                np.zeros((img_h, img_w), dtype=np.uint16),
                np.zeros((img_h, img_w), dtype=bool))

    ui = ui[in_img]; vi = vi[in_img]; z = z[in_img]; I = I[in_img]
    Iu8 = np.round(np.clip(I, 0.0, 1.0) * 255.0).astype(np.uint8)

    rgb = np.zeros((img_h, img_w), dtype=np.uint8)
    depth = np.zeros((img_h, img_w), dtype=np.uint16)
    hit = np.zeros((img_h, img_w), dtype=bool)

    zmin, zmax = depth_clip
    for px, py, pz, val in zip(ui, vi, z, Iu8):
        if not (zmin <= pz <= zmax):
            continue
        if (not hit[py, px]) or pz < (depth[py, px] / 1000.0):
            hit[py, px] = True
            rgb[py, px] = val
            depth[py, px] = int(np.clip(round(pz * 1000.0), 0, 65535))

    return rgb, depth, hit
