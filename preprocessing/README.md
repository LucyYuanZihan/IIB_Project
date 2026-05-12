# preprocessing/

Geometric preprocessing for the tunnel inpainter: normalise raw point clouds, generate virtual camera viewpoints, render RGB + depth supervision, and package per-scene `.npz` files consumed by `inpainter/`.

## Layout

```
preprocessing/
├── configs/
│   └── default.json          shared hyperparameters (image size, FOV, view count, point count, ...)
├── src/                      canonical helpers; imported by the scripts
│   ├── camera.py             fibonacci_sphere · random_sphere · look_at_c2w
│   └── projection.py         project_points_fullscene · load_txt_point_cloud
└── scripts/
    ├── normalise.py          unit-sphere XYZ + 1/99-percentile intensity → [0,1]
    ├── make_splits.py        80/10/10 train/val/test split (deterministic seed)
    └── generate_skip_small.py  ← canonical pipeline: random sphere views,
                                 random downsample, skip files with too few points
```

## Pipeline

Typical flow for inpainter data:

1. *(optional)* `normalise.py` — pre-condition raw `.txt` scans (unit-sphere XYZ + percentile-clipped intensity).
2. *(optional)* `make_splits.py` — produce `train/`, `val/`, `test/` subdirs.
3. `generate_skip_small.py` — render multi-view RGBD `.npz` consumed by `inpainter/`.

Steps 1–2 are optional; `generate_skip_small.py` does its own bbox-centring and global min-max intensity scaling, so it can run directly on raw `.txt`.

## Configuration

`generate_skip_small.py` reads its defaults from [configs/default.json](configs/default.json):
`input_dir`, `output_dir`, `num_views` (50), `img_h` / `img_w` (650), `fov_x_radians`
(≈ 39.6°), `radius_scale` (0.85), `num_points` (230 000 — also the skip threshold),
`seed` (0), `num_variants` (8). Override with the matching CLI flags, or pass
`--config path/to/other.json` to load a different file.

Script-specific knobs stay CLI-only: `--no_pngs`, `--save_downsampled_txt`,
`--txt_coords {centered|raw}`, `--txt_intensity {raw|normalized}`, `--downsampled_txt_dir`.

The `splits.*` and `intensity_normalisation.*` blocks in `default.json` are
consumed by `make_splits.py` and `normalise.py` respectively.

## Output `.npz` schema

Each `<scene>.npz` (or `<scene>_v<i>.npz` for variant `i`) contains:

- `images` `[V, H, W, 3]` float32 — rendered grayscale RGB (white background)
- `depths` `[V, H, W]` float32 — depth in meters, 0 = invalid
- `cam_poses` `[V, 4, 4]` float32 — camera-to-world (Blender/NeRF convention)
- `data` `[N_pts, 6]` float32 — `(x, y, z, r, g, b)` with `r=g=b=intensity`

`V` defaults to 50 (`num_views`); points are scene-centered and intensity is normalised to [0, 1] across the scene.

## Side-outputs from `generate_skip_small.py`

Alongside the per-scene `.npz`, the script writes into `--output_dir`:

- `pointcloud_centers.xlsx` — bbox midpoint of each scene in original coordinates (lets you reverse a scene-centred cloud back to world coords later).
- `skipped_files.xlsx` — names of `.txt` files that fell below `--num_points` and were skipped.
- `downsampled_txt/<scene>[_v<i>].txt` — the variant's downsampled cloud as 5-column text (controlled by `--txt_coords` / `--txt_intensity`); useful as a drop-in replacement for the source `.txt`.
- Per-scene debug PNGs (`images/`, `depths/`) and `transforms.json` — unless `--no_pngs` is set.

## One-line examples

```bash
# 1. (optional) normalise raw .txt scans
python scripts/normalise.py            # set INPUT_FOLDER / OUTPUT_FOLDER inline

# 2. (optional) build train/val/test split
python scripts/make_splits.py          # set NORMALIZED_FOLDER / OUTPUT_BASE_FOLDER inline

# 3. generate inpainter-ready .npz dataset (defaults from configs/default.json)
python scripts/generate_skip_small.py
python scripts/generate_skip_small.py --input_dir <txt_dir> --output_dir <npz_dir>
python scripts/generate_skip_small.py --config configs/my_alt_config.json
```
