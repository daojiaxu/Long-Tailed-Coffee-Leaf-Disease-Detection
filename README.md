# Coffee Leaf Disease Detection under Long-Tailed Class Distribution

Official code for the paper **"An Empirical Study of Class Re-balancing Strategies for Long-Tailed Coffee Leaf Disease Detection"**.

We study class re-balancing for object detection when the training annotations are severely long-tailed (a 31:1 imbalance in the BRACOL coffee leaf dataset), and systematically compare three strategy families on a YOLO26 detector:

1. **Loss-level re-balancing** — Class-Balanced (CB) loss, CB-Focal, Seesaw Loss.
2. **Data-level re-balancing** — Repeat-Factor-Sampling (RFS) and instance-level Copy-Paste augmentation with Poisson fusion.
3. **Decoupled training** — freeze the BCE-learned backbone and re-train only the detection head with the CB loss.

The key finding: decoupled training is the **only** strategy among those compared that improves rare-class recall (+65%, *p* < 0.01 over five seeds) **without** sacrificing precision, while significantly lifting mAP50 above the baseline (*p* < 0.001).

---

## Key Results (BRACOL REVIEWED, YOLO26s @ 640)

| Strategy | Cercospora Recall | Precision | mAP50 |
| --- | --- | --- | --- |
| BCE (baseline) | 0.180 | 0.391 | 0.568 |
| Class-Balanced loss | **0.410** | 0.349 | 0.549 |
| Copy-Paste (×1) | 0.385 | 0.252 | 0.555 |
| **Decoupled training** | 0.297 ± 0.034 | 0.408 ± 0.074 | **0.585 ± 0.004** |

Full per-class metrics for every experiment are in [`results/experiment_log.csv`](results/experiment_log.csv).

---

## Methodology

The study systematically compares three families of class re-balancing strategies for long-tailed object detection, using a YOLO26s detector on the BRACOL REVIEWED split (31:1 imbalance):

1. **Loss-level re-balancing** — replace the BCE classification loss with the Class-Balanced (CB) loss, optionally stacked with Focal and Seesaw variants.
2. **Data-level re-balancing** — oversample rare classes via image-level Repeat-Factor-Sampling (RFS) or instance-level Copy-Paste augmentation with Poisson fusion.
3. **Decoupled training** — freeze the BCE-learned backbone/neck and re-train only the detection head with the CB loss.

Key comparisons are repeated over multiple random seeds and assessed with one-sample t-tests. Every run logs per-class AP, precision, and recall to `results/experiment_log.csv`. No preprocessing beyond resizing images to 640 px was applied; the expert-annotated YOLO-format labels are used as released.

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

Experiments use the public **BRACOL** dataset (coffee leaf diseases), distributed under the Creative Commons Attribution 4.0 (CC BY 4.0) license:

- BRACOL images & annotations: Mendeley Data, `doi:10.17632/yy2k5y8mxg.1` (https://data.mendeley.com/datasets/yy2k5y8mxg)
- License: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)

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

**4. Decoupled training (our key finding)**

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
  title={An Empirical Study of Class Re-balancing Strategies for Long-Tailed Coffee Leaf Disease Detection},
  author={Dao, Jiaxu and Li, Zhuoying}
}
```

## License

MIT License. See [LICENSE](LICENSE).

## Contribution

Bug reports and pull requests are welcome. Please open an issue to discuss any substantive change before submitting a pull request.
