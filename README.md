# Coffee Leaf Disease Detection under Long-Tailed Class Distribution

Official code for the paper **"Decoupled Training Breaks the Recall–Precision Trade-off in Long-Tailed Coffee Leaf Disease Detection"**.

We study class re-balancing for object detection when the training annotations are severely long-tailed (a 31:1 imbalance in the BRACOL coffee leaf dataset), and systematically compare three strategy families on a YOLO26 detector:

1. **Loss-level re-balancing** — Class-Balanced (CB) loss, CB-Focal, Seesaw Loss.
2. **Data-level re-balancing** — Repeat-Factor-Sampling (RFS) and instance-level Copy-Paste augmentation with Poisson fusion.
3. **Decoupled training** — freeze the BCE-learned backbone and re-train only the detection head with the CB loss.

The key finding: decoupled training is the **only** strategy that improves rare-class recall (+65%, *p* < 0.01 over five seeds) **without** sacrificing precision, while significantly lifting mAP50 above the baseline (*p* < 0.001).

---

## Key Results (BRACOL REVIEWED, YOLO26s @ 640)

| Strategy | Cercospora Recall | Precision | mAP50 |
| --- | --- | --- | --- |
| BCE (baseline) | 0.180 | 0.480 | 0.568 |
| Class-Balanced loss | **0.410** | 0.428 | 0.549 |
| Copy-Paste (×1) | 0.385 | 0.478 | 0.559 |
| **Decoupled training** | 0.297 ± 0.034 | 0.408 ± 0.074 | **0.585 ± 0.004** |

Full per-class metrics for every experiment are in [`results/experiment_log.csv`](results/experiment_log.csv).

---

## Repository Structure

```
├── train_baseline.py        # Training script (BCE / focal / cb / cbfocal / seesaw, --freeze for decoupled training)
├── losses.py                # Long-tail loss injection (CB, CB-Focal, Seesaw, β-annealing, λ trade-off)
├── build_copypaste.py       # Instance-level Copy-Paste augmentation (Poisson fusion + leaf-surface filtering)
├── build_oversample.py      # Image-level Repeat-Factor-Sampling (RFS)
├── eval_robustness.py       # Robustness evaluation under 5 distribution shifts
├── convert_rocole.py        # RoCoLe → YOLO detection format
├── run_sci_experiments.py   # Batch driver for the full experiment matrix
├── run_decouple_seeds.py    # 5-seed decoupled-training significance runs
├── data/
│   ├── data_reviewed.yaml   # BRACOL REVIEWED (dense expert annotation)
│   └── data_original.yaml   # BRACOL ORIGINAL (sparse annotation)
└── results/
    ├── experiment_log.csv   # Complete experiment ledger
    └── robustness_results.csv
```

---

## Installation

```bash
# Python 3.9+ recommended
pip install -r requirements.txt
```

We used `torch 2.7.0+cu118` with `ultralytics>=8.4` (YOLO26 requires the E2ELoss API). The code is designed to run on a single RTX 3090 (24 GB).

## Data Preparation

Experiments use the public **BRACOL** dataset (coffee leaf diseases):

- BRACOL images & annotations: Mendeley Data, `doi:10.17632/yy2k5y8mxg.1`

Download and organize the YOLO-format annotations into `data/`:

```
data/
├── BRACOL_REVIEWED/         # dense expert-reviewed annotations (main experiments)
│   ├── train/{images,labels}
│   ├── valid/{images,labels}
│   └── test/{images,labels}
├── BRACOL_ORIGINAL/         # original sparse annotations (cross-annotation study)
│   └── ...
└── RoCoLe/                  # independent robustness dataset (optional)
    └── ...
```

The four classes are: `Cercospora`, `Miner`, `Phoma`, `Rust` (training counts 133 / 171 / 1310 / 4184).

Pretrained backbones (`yolo26s.pt`, `yolo26n.pt`) are auto-downloaded by Ultralytics; place them under `weights/` if you have a local copy (offline machines).

## Reproducing Experiments

All commands are run from the repository root.

**1. BCE baseline**

```bash
python train_baseline.py --model weights/yolo26s.pt --data data/data_reviewed.yaml \
    --epochs 100 --batch 32 --name B0_yolo26s_640 --tag B0
```

**2. Loss-level re-balancing (Class-Balanced loss)**

```bash
python train_baseline.py --model weights/yolo26s.pt --data data/data_reviewed.yaml \
    --epochs 100 --batch 32 --loss cb --beta 0.9999 --name M2_cb_640 --tag M2
```

**3. Data-level re-balancing (Copy-Paste augmentation)**

```bash
# build the augmented dataset (instance-level Copy-Paste with Poisson fusion)
python build_copypaste.py --data data/data_reviewed.yaml --out data/copypaste_cp --per-img 1

# train on the augmented split
python train_baseline.py --model weights/yolo26s.pt --data data/copypaste_cp/data.yaml \
    --epochs 100 --batch 32 --name CP1_cp_bce_640 --tag CP1
```

**4. Decoupled training (the proposed method)**

```bash
# first train a BCE baseline (B0), then freeze backbone+neck and retrain only the detection head:
python train_baseline.py --model runs/B0_yolo26s_640/weights/best.pt \
    --data data/data_reviewed.yaml --epochs 50 --batch 32 --loss cb --freeze 23 \
    --name D1_decouple_cb_640 --tag D1
```

**5. Five-seed significance (decoupled training)**

```bash
python run_decouple_seeds.py
```

**6. Full experiment matrix (all ablations)**

```bash
python run_sci_experiments.py --dry-run   # preview commands
python run_sci_experiments.py             # run everything
```

Every run appends per-class AP / Precision / Recall to `results/experiment_log.csv`.

---

## Citation

If you use this code, please cite:

```bibtex
@article{dao2026decoupled,
  title={Decoupled Training Breaks the Recall--Precision Trade-off in Long-Tailed Coffee Leaf Disease Detection},
  author={Dao, Jiaxu and Li, Zhuoying}
}
```

## License

MIT License. See [LICENSE](LICENSE).
