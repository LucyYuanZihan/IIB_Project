# Pipeline — Tunnel Point-Cloud Inpainting

## 1. Task

Given a sparse / partially-scanned tunnel point cloud, recover a dense, complete one. Quality is assessed both directly (geometric / photometric) and indirectly (semantic preservation via a frozen Seg2Lining classifier).

## 2. Components

### preprocessing/
- Normalise raw `.txt` clouds (unit-sphere XYZ + percentile intensity rescale).
- Generate per-scene virtual cameras (Fibonacci or random sphere; `look_at_c2w` Blender/NeRF convention).
- Render RGB + 16-bit depth + accumulated-opacity maps per virtual view.
- Package per-scene `.npz` (`images`, `depths`, `cam_poses`, `data`).

### inpainter/  (the inpainter)
- **Encoder** — Conv1d stack (6→512) + max-pool + FC → `(μ, log σ)`, `z = 4096`. Variant: PointNet++ (FPS+kNN, 8192→2048→512→128).
- **Hypernetwork** — `hypnettorch.ChunkedHMLP` produces all NeRF weights from `z`.
- **NeRF MLP** — depth 8, width 256, skip @ 4; PE multires=10 (63D coords) + 4 (27D dirs).
- **Volumetric render** — `N_samples=256`, `N_rand=1024`, `near/far=0.1/15.0`.
- **Loss** — `L = L_rgb + λ_depth · L_depth` (MSE per channel; depth MSE on valid pixels).
- **Inpainting inference** — `depth_to_pointcloud(img, disp, acc, K, c2w, acc_threshold=0.99)` unprojects per-view depth to 3D and aggregates over views.

### evaluator/  (downstream evaluator)
- **Architecture** — hierarchical encoder–decoder; pluggable LFA / RFA / GFA_S / GFA_L modules selected at config time.
- **Output regimes** — one-hot softmax (`ohe`) or spherical-encoding (`se`).
- **Inpainting eval** — `prepare_inpainted.py` reformats the inpainter output, `test_inpainted.py` runs the frozen seg net, `IoUCalculator` reports IoU / OA / confusion matrix. The IoU gap between GT and inpainted clouds is the semantic-fidelity proxy.

## 3. Dataflow contract

| File type | Producer | Consumer |
|---|---|---|
| `<scene>.txt` (5-col xyz/i/label) | external scanner | `preprocessing/` |
| `<scene>.npz` (images/depths/cam_poses/data) | `preprocessing/` | `inpainter/` train + render |
| Inpainted point cloud (`.txt` or `.ply`) | `inpainter/scripts/inpaint.py` | `evaluator/scripts/prepare_inpainted.py` |
| `<station>.npy` + `<station>_KDTree.pkl` | `evaluator/scripts/prepare_*.py` | `evaluator/scripts/test*.py` |
| IoU / OA / confusion matrix | `evaluator/src/metrics/iou.py` | logs / paper tables |

## 4. Configurations

- **Inpainter** (`inpainter/configs/seg2tunnel*.json`): `n_points`, `z_size`, `max_epochs`, `near`/`far`, `lambda_depth`, encoder choice, NeRF MLP width/depth.
- **Evaluator** (`evaluator/configs/{default,ring}.py`): `subset` (selects 7-class scene-wise or 8-class ring-wise), `num_points`, station IDs (train/val/test), `lfa`/`rfa`/`gfa_s`/`gfa_l` strings.

## 5. Reproducibility hooks

- `inpainter/scripts/train.py` — `set_seed(config.seed)`; default tensor type forced to `cuda.FloatTensor`.
- `preprocessing/scripts/generate_*.py` — per-scene seed derived as `seed + (md5(scene_name)[0:8] mod 2e9)`.
- `evaluator/configs/*.py` — explicit train/val/test station lists.

## 6. Out of scope (excluded from this tree)

- `SNAP/`, `SNAP_new/` — interactive 3D segmentation (SAM lineage); separate research strand.
- `Seg2Lining_wanru/` — HPC / pretrain+finetune fork; superseded by the canonical `evaluator/` for the inpainting pipeline.
- `points2nerf` ablation variants `pts2nerf_seg2tunnel.py`, `_new_bias.py`, `pts2nerf_nodep.py` — kept in the user backup.
