# An Empirical Study of Class Re-balancing Strategies for Long-Tailed Coffee Leaf Disease Detection

Official code and documentation for this paper.

## Description

This repository provides the complete source code used to reproduce all experiments reported in the paper. We study class re-balancing for object detection when the training annotations are severely long-tailed (a 31:1 imbalance in the BRACOL coffee leaf dataset), and systematically compare three strategy families on a YOLO26 detector:

1. **Loss-level re-balancing** — Class-Balanced (CB) loss, CB-Focal, Seesaw Loss.
2. **Data-level re-balancing** — Repeat-Factor-Sampling (RFS) and instance-level Copy-Paste augmentation with Poisson fusion.
3. **Decoupled training** — freeze the BCE-learned backbone and re-train only the detection head with the CB loss.

The key finding: decoupled training is the **only** strategy among those compared that improves rare-class recall (+65%, *p* < 0.01 over five seeds) **without** sacrificing precision, while significantly lifting mAP50 above the baseline (*p* < 0.001).

### Key Results (BRACOL REVIEWED, YOLO26s @ 640)

| Strategy | Cercospora Recall | Precision | mAP50 |
| --- | --- | --- | --- |
| BCE (baseline) | 0.180 | 0.391 | 0.568 |
| Class-Balanced loss | **0.410** | 0.349 | 0.549 |
| Copy-Paste (×1) | 0.385 | 0.252 | 0.555 |
| **Decoupled training** | 0.297 ± 0.034 | 0.408 ± 0.074 | **0.585 ± 0.004** |

Full per-class metrics for every experiment are in [`results/experiment_log.csv`](results/experiment_log.csv).

## Dataset Information

The experiments use the public **BRACOL** dataset (Brazilian Arabica Coffee Leaf images), a third-party dataset distributed under the Creative Commons Attribution 4.0 (CC BY 4.0) license:

- **Name:** BRACOL — Brazilian Arabica Coffee Leaf images dataset
- **Source:** Mendeley Data
- **DOI:** `10.17632/yy2k5y8mxg.1`
- **URL:** https://data.mendeley.com/datasets/yy2k5y8mxg
- **License:** CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)

The four annotated classes are `Cercospora`, `Miner`, `Phoma`, and `Rust`, with training annotation counts of 133 / 171 / 1310 / 4184 (a 31:1 imbalance ratio). Two annotation splits are used:

- **BRACOL REVIEWED** — expert-reviewed, densely annotated bounding boxes (the primary benchmark).
- **BRACOL ORIGINAL** — the initial, sparser annotations (used for the cross-annotation comparison).

Download the dataset and organize the YOLO-format annotations into `data/` as follows:

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

The dataset images and annotations are **not** redistributed in this repository (see the license above); they must be downloaded from Mendeley Data.

## Code Information

The repository is organized as follows:

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

The long-tail losses (CB, CB-Focal, Seesaw) are implemented in `losses.py` and injected into the Ultralytics YOLO26 end-to-end (E2E) loss via a training hook. `train_baseline.py` is the single training entry point for all loss-level and decoupled-training runs; `build_copypaste.py` and `build_oversample.py` generate the data-level variants offline.

## Requirements

- Python 3.9+
- Dependencies are listed in [`requirements.txt`](requirements.txt):

```
torch>=2.0
torchvision>=0.15
ultralytics>=8.4
opencv-python>=4.8
numpy>=1.24
pyyaml>=6.0
matplotlib>=3.7
pillow>=9.5
pandas>=2.0
scipy>=1.10
```

Install with:

```bash
pip install -r requirements.txt
```

The experiments were run with `torch 2.7.0+cu118` and `ultralytics 8.4.173` (YOLO26 requires the E2ELoss API) on a single NVIDIA RTX 3090 (24 GB). Pretrained backbones (`yolo26s.pt`, `yolo26n.pt`) are auto-downloaded by Ultralytics; place them under `weights/` if you have a local copy (offline machines).

## Usage Instructions

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

**4. Decoupled training (key finding)**

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

## Methodology

The study systematically compares three families of class re-balancing strategies for long-tailed object detection, using a YOLO26s detector on the BRACOL REVIEWED split (31:1 imbalance):

1. **Loss-level re-balancing** — replace the BCE classification loss with the Class-Balanced (CB) loss, optionally stacked with Focal and Seesaw variants.
2. **Data-level re-balancing** — oversample rare classes via image-level Repeat-Factor-Sampling (RFS) or instance-level Copy-Paste augmentation with Poisson fusion.
3. **Decoupled training** — freeze the BCE-learned backbone/neck and re-train only the detection head with the CB loss.

Key comparisons are repeated over multiple random seeds and assessed with one-sample t-tests. Every run logs per-class AP, precision, and recall to `results/experiment_log.csv`. No preprocessing beyond resizing images to 640 px was applied; the expert-annotated YOLO-format labels are used as released.

## Citations

If you use this code, please cite the paper:

```bibtex
@article{dao2026decoupled,
  title={An Empirical Study of Class Re-balancing Strategies for Long-Tailed Coffee Leaf Disease Detection},
  author={Dao, Jiaxu and Li, Zhuoying},
  year={2026}
}
```

If you use the BRACOL dataset, please cite the original paper:

```bibtex
@article{Esgario2020,
  title={Deep learning for classification and severity estimation of coffee leaf biotic stress},
  author={Esgario, Jos{\'e} G. M. and Krohling, Renato A. and Ventura, Jos{\'e} A.},
  journal={Computers and Electronics in Agriculture},
  volume={169},
  pages={105162},
  year={2020}
}
```

The dataset itself can be cited via its Mendeley Data DOI: `10.17632/yy2k5y8mxg.1`.

## License & Contribution Guidelines

The code in this repository is released under the **MIT License** — see [LICENSE](LICENSE) for details. The BRACOL dataset used in the experiments is licensed separately under **CC BY 4.0** (see [Dataset Information](#dataset-information)).

Contributions are welcome. Bug reports and pull requests are appreciated; please open an issue to discuss any substantive change before submitting a pull request.
