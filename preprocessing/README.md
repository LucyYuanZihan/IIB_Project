# preprocessing/

Geometric preprocessing for the tunnel inpainter: normalise raw point clouds, generate virtual camera viewpoints, render RGB + depth supervision, and package per-scene `.npz` files consumed by `inpainter/`.

## Layout

```
preprocessing/
├── configs/
│   └── default.json
├── src/
│   ├── camera.py           fibonacci_sphere · random_sphere · look_at_c2w
│   └── projection.py       project_points_fullscene · load_txt_point_cloud
└── scripts/
    ├── normalise.py        unit-sphere XYZ + 1/99-percentile intensity → [0,1]
    ├── make_splits.py      80/10/10 train/val/test split (deterministic seed)
    ├── downsample_one.py   one-shot random downsample of a single .txt
    ├── generate_random.py  full-pipeline (Fibonacci/random sphere views; random downsample)
    ├── generate_adaptive.py adaptive multi-scale curvature + FPS downsample
    └── generate_skip_small.py random sphere views, skip files with too few points
```

## Output `.npz` schema

Each `<scene>.npz` (or `<scene>_v<i>.npz` for variant `i`) contains:

- `images` `[V, H, W, 3]` float32 — rendered grayscale RGB (white background)
- `depths` `[V, H, W]` float32 — depth in meters, 0 = invalid
- `cam_poses` `[V, 4, 4]` float32 — camera-to-world (Blender/NeRF convention)
- `data` `[N_pts, 6]` float32 — `(x, y, z, r, g, b)` with `r=g=b=intensity`

`V` defaults to 50 (`num_views`); points are scene-centered and intensity is normalised to [0, 1] across the scene.

## One-line examples

```bash
python scripts/normalise.py            # set INPUT_FOLDER / OUTPUT_FOLDER inline
python scripts/make_splits.py          # set NORMALIZED_FOLDER / OUTPUT_BASE_FOLDER
python scripts/generate_skip_small.py --input_dir <txt_dir> --output_dir <npz_dir>
```
