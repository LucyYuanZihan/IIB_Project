# TunnelInpainting

Tunnel point-cloud **inpainting** pipeline. Given a sparse or partially-occluded scan of a tunnel, the system produces a dense, infilled point cloud, evaluated both directly (PSNR / Chamfer / depth-MSE) and through a downstream semantic-segmentation proxy.

## Pipeline

```
   sparse / partial point cloud
              │
              ▼
   ┌──────────────────────────┐
   │ preprocessing/           │   normalise, generate virtual camera poses,
   │ (ex-virtual_camera)      │   render multi-view RGB+depth, package .npz
   └─────────────┬────────────┘
                 │
                 ▼
   ┌──────────────────────────┐
   │ inpainter/               │   Encoder → VAE → ChunkedHMLP → NeRF MLP
   │ (ex-points2nerf)         │   L = L_rgb + λ_d · L_depth
   └─────────────┬────────────┘
                 │  vol-render + depth_to_pointcloud (acc>0.99)
                 ▼
        DENSE INPAINTED CLOUD
                 │
       ┌─────────┴──────────┐
       ▼                    ▼
 direct geometric eval   ┌────────────────────────────┐
 (PSNR, Chamfer,         │ evaluator/                 │
  depth-MSE)             │ (ex-Seg2Lining, frozen)    │
                         │ IoU / mIoU / OA / conf.    │
                         └────────────────────────────┘
```

## Layout

| Path | Role |
|---|---|
| `preprocessing/` | Normalise raw clouds, generate Fibonacci-sphere virtual cameras, build train/val/test splits, package `.npz` for the inpainter |
| `inpainter/` | Trains the points-to-NeRF inpainter; renders dense RGB + depth + point clouds; computes Chamfer + PSNR |
| `evaluator/` | A frozen Seg2Lining (`Network`) ingests an inpainted cloud and reports IoU / OA — the semantic-fidelity proxy for inpainting |
| `data/` | Symlink → original raw data location (read-only by convention) |
| `docs/` | Architectural notes |

## Quick start

```bash
# 1. Preprocessing — produce per-scene .npz with images, depths, cam_poses, data
python preprocessing/scripts/generate_skip_small.py --input_dir <raw_txt_dir> --output_dir <p2n_npz_dir>

# 2. Train the inpainter (standard encoder)
PYTHONPATH=inpainter python inpainter/scripts/train.py inpainter/configs/seg2tunnel.json

# 2b. Train the PointNet++ encoder variant
PYTHONPATH=inpainter python inpainter/scripts/train_pn2.py inpainter/configs/seg2tunnel_pn2.json

# 3. Inpaint — render and unproject NeRF depth to a dense point cloud
PYTHONPATH=inpainter python inpainter/scripts/inpaint.py inpainter/configs/seg2tunnel.json --ckpt <path/to/ckpt>

# 4a. Direct geometric eval
PYTHONPATH=inpainter python inpainter/scripts/eval_chamfer.py inpainter/configs/seg2tunnel.json
PYTHONPATH=inpainter python inpainter/scripts/eval_psnr.py    inpainter/configs/seg2tunnel.json

# 4b. Semantic eval — segment the inpainted cloud and report IoU/OA
PYTHONPATH=evaluator python evaluator/scripts/prepare_inpainted.py
PYTHONPATH=evaluator python evaluator/scripts/test_inpainted.py
```

## Setup after clone

The repository assumes four sibling directories alongside it, holding data and training output. The four in-tree paths are **relative symlinks** pointing at them:

```
<parent>/                              ← clone into this directory
├── data/             raw point clouds           ← TunnelInpainting/data           → ../data
├── p2n_data/         primary inpainter dataset  ← TunnelInpainting/inpainter/data → ../../p2n_data
├── p2n_data1/        alternative inpainter ds   ← TunnelInpainting/inpainter/data_alt → ../../p2n_data1
├── p2n_results/      training output / ckpt     ← TunnelInpainting/inpainter/experiments → ../../p2n_results
└── TunnelInpainting/ ← this repo
```

If any link is missing or broken, recreate from inside the repo:

```bash
cd TunnelInpainting
ln -sfn ../data           data
ln -sfn ../../p2n_data    inpainter/data
ln -sfn ../../p2n_data1   inpainter/data_alt
ln -sfn ../../p2n_results inpainter/experiments
```

## Conventions

- All scripts use absolute imports off the per-component package root (`src/...`, `configs/...`). Run them with `PYTHONPATH=<component>` set, or `cd <component> && python scripts/<x>.py`.
- Configs are JSON for the inpainter and Python modules for the evaluator (the latter contain runtime branching that JSON can't express).
- The `data/` directory is a symlink — never modify its contents through this tree.


