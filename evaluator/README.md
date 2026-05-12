# evaluator/

Frozen Seg2Lining segmentation network used as a **downstream semantic evaluator** for the inpainted point cloud. The training pathway (`train.py`, `train_ring.py`) trains the segmentation model on real ground-truth tunnel data; the inpainting-eval pathway (`prepare_inpainted.py` → `test_inpainted.py`) freezes that model and reports IoU/OA on the inpainter's output.

## Architecture (`src/models/network.py`)

```
   features (4-D: XYZ + scalar)
          │
          ▼
   fc0 (Conv1d, 4 → 8, BN)
          │
          ▼
   [LFA × num_layers]  (downsampling) ──► [optional RFA × num_layers]
          │
          ▼
   GFA_S
          │
          ▼
   [Conv2d decoder × num_layers] ◄── skip from each LFA stage
          │
          ▼
   fc1(64) → fc2(32) → Dropout(0.5) → fc3
          │
          ▼  ('ohe' → C class logits, 'se' → 3-D unit vector)
          ▼
   per-point label
```

Pluggable modules (`importlib`, selected by config string):

- **LFA** — `lfa_{cheng2023, fan2021, hu2019, jing2022, lin_v1, zhan2023, zhao2021}` (7)
- **RFA** — `rfa_{lin_v1, lin_v2, lin_v3}` (3)
- **GFA_S** / **GFA_L** — `{deng2021, li2022, liu2022, liu2023, ren2022}` (5 + 5)

## Layout

```
evaluator/
├── configs/
│   ├── default.py        scene-wise (subset='seg2tunnel', 7 classes)
│   └── ring.py           ring-wise (subset='seg2tunnel_ring', 8 classes)
├── src/
│   ├── models/
│   │   ├── network.py
│   │   ├── pytorch_utils.py
│   │   ├── lfa/, rfa/, gfa_s/, gfa_l/
│   ├── datasets/{seg2tunnel,ring}.py
│   ├── losses/losses.py
│   └── metrics/iou.py    IoUCalculator (mIoU, OA, confusion)
└── scripts/
    ├── prepare.py / prepare_ring.py            preprocess GT clouds
    ├── prepare_inpainted.py                    preprocess INPAINTED clouds for eval
    ├── train.py / train_ring.py                train the seg net
    ├── test.py / test_ring.py                  GT-IoU baseline
    ├── test_inpainted.py                       semantic eval of inpainter output
    ├── demo.py                                 FLOPs / parameter counts
    ├── visualise.py                            visualisation
    └── restore.py                              restore from a checkpoint
```

## Loss / metric

- **Loss (training):** weighted CE (`'ohe'` mode) or spherical-encoding distance (`'se'` mode). Optional multi-level supervision (`flag_ml`).
- **Metric (eval):** per-class IoU, mIoU, OA, confusion matrix (`metric.IoUCalculator`).

## Run (semantic eval of inpainter output)

```bash
PYTHONPATH=. python scripts/prepare_inpainted.py     # preprocess inpainted .txt → .npy + KDTree
PYTHONPATH=. python scripts/test_inpainted.py        # frozen Seg2Lining → IoU/OA on inpainted
```
