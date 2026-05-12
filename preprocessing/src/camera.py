"""Shared virtual-camera helpers used by the dataset-generation scripts.

Defines:
  - fibonacci_sphere(n)       : roughly-uniform directions on S^2 (deterministic).
  - random_sphere(n, rng)     : i.i.d. random directions on S^2.
  - look_at_c2w(origin, target, up_hint=[0,1,0])
                              : 4x4 camera-to-world matrix in Blender/NeRF convention.

The corresponding scripts in `preprocessing/scripts/` currently inline equivalent
implementations; this module is the canonical place to import them from in new
code (e.g. when extending preprocessing).
"""

import math
import numpy as np


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


def random_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    """Return n i.i.d. uniform directions on the unit sphere."""
    u = rng.random(n, dtype=np.float32)
    v = rng.random(n, dtype=np.float32)
    theta = 2.0 * math.pi * u
    z = 2.0 * v - 1.0
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def look_at_c2w(
    origin: np.ndarray,
    target: np.ndarray,
    up_hint: np.ndarray = np.array([0, 1, 0], dtype=np.float32),
) -> np.ndarray:
    """Camera-to-world transform (Blender/NeRF style: camera looks along -Z)."""
    origin = origin.astype(np.float32)
    target = target.astype(np.float32)
    up_hint = up_hint.astype(np.float32)

    forward = target - origin
    f = forward / (np.linalg.norm(forward) + 1e-8)

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
    c2w[0:3, 2] = -f
    c2w[0:3, 3] = origin
    return c2w
