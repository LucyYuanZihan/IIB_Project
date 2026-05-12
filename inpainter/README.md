# inpainter/

The tunnel point-cloud inpainter: encodes a sparse cloud into a latent, generates a per-cloud NeRF via a chunked hypernetwork, then unprojects rendered depth maps back to a dense 3D point cloud.

## Architecture

```
   Sparse cloud (B, 6, N=8192)
          │
          ▼
   Encoder (default Conv1d-stack VAE; alt PointNet++ via train_pn2.py)
          │
          ▼ (μ, log σ) → z ∼ N(μ, σ²)   [z_size = 4096]
          │
          ▼
   ChunkedHMLP hypernetwork ([4096, 8192], chunk 16384)
          │
          ▼
   NeRF MLP (D=8, W=256, skip @ 4)
          │
          ▼ vol-render rays (N_samples=256, N_rand=1024, near/far=0.1/15.0)
          │
          ▼
   per-view RGB · depth · acc maps
          │
          ▼ depth_to_pointcloud(acc > 0.99)
          │
          ▼
   DENSE INPAINTED POINT CLOUD
```

## Layout

```
inpainter/
├── configs/
│   ├── seg2tunnel.json       standard encoder
│   └── seg2tunnel_pn2.json   PointNet++ encoder
├── src/
│   ├── models/
│   │   ├── encoder.py        Encoder + PointNet2VAEEncoder
│   │   ├── nerf.py
│   │   └── resnet.py
│   ├── datasets/
│   │   └── seg2tunnel.py
│   ├── render/
│   │   └── nerf_helpers.py
│   ├── losses/               (depth + RGB MSE inlined in trainers)
│   └── utils.py
├── scripts/
│   ├── train.py              standard-encoder trainer
│   ├── train_pn2.py          PointNet++ trainer
│   ├── inpaint.py            depth-to-point inpainting renderer (canonical)
│   ├── render_rgb.py         RGB-only rendering
│   ├── render_depth.py       depth-only rendering
│   ├── eval_chamfer.py       Chamfer distance + F-score
│   └── eval_psnr.py          PSNR
├── third_party/
│   └── ChamferDistancePytorch/
├── data/                     → symlink → /home/zy349/Project/points2nerf/data
└── experiments/              → symlink → /home/zy349/Project/points2nerf/results
```

## Loss

```
L = L_rgb + λ_depth · L_depth
L_rgb   = MSE over rendered vs. ground-truth pixels
L_depth = MSE over pixels with valid ground-truth depth
```

Edit `configs/seg2tunnel.json:lambda_depth` to enable depth supervision (default `0.0`).

## Run

```bash
PYTHONPATH=. python scripts/train.py     configs/seg2tunnel.json
PYTHONPATH=. python scripts/train_pn2.py configs/seg2tunnel_pn2.json
PYTHONPATH=. python scripts/inpaint.py   configs/seg2tunnel.json --ckpt <path>
PYTHONPATH=. python scripts/eval_chamfer.py configs/seg2tunnel.json
PYTHONPATH=. python scripts/eval_psnr.py    configs/seg2tunnel.json
```
