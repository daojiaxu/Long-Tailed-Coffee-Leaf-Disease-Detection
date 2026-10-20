# -*- coding: utf-8 -*-
"""
Batch runner for the extended experiments

Runs the extended experiment matrix sequentially; results are automatically
appended to runs/experiment_log.csv. Already-completed experiments
(runs/<name>/weights/best.pt exists) are skipped automatically.

Usage (from the project root):
  python run_sci_experiments.py            # run all sequentially
  python run_sci_experiments.py --epochs 100 --batch 32
  python run_sci_experiments.py --dry-run  # print commands only, do not execute
  python run_sci_experiments.py --only M4   # run only tag=M4

Notes:
  - 100 epochs by default (formal experiments); override with --epochs for a quick check
  - all experiments at 640 resolution, batch 32
  - the already-run B0 (baseline) / E1b (CB-Focal gamma=2) are not in the list, to avoid duplication
"""
import argparse
import os
import subprocess
import sys

PY = sys.executable
ROOT = os.path.dirname(os.path.abspath(__file__))

REVIEWED = "data/data_reviewed.yaml"
ORIGINAL = "data/data_original.yaml"
S26 = "weights/yolo26s.pt"
N26 = "weights/yolo26n.pt"
ROCOLE = "data/RoCoLe/data.yaml"

# (tag, name, model, data, extra_args, note)
EXPERIMENTS = [
    # ---- group A: loss-method comparison (REVIEWED @640, YOLO26s) ----
    ("M1", "M1_focal_640",  S26, REVIEWED, ["--loss", "focal"],  "Focal Loss alone (gamma=2)"),
    ("M2", "M2_cb_640",     S26, REVIEWED, ["--loss", "cb"],     "CB Loss alone (beta=0.9999)"),
    ("M4", "M4_seesaw_640", S26, REVIEWED, ["--loss", "seesaw"], "Seesaw Loss (p=2)"),
    # ---- group B: gamma sensitivity (CB-Focal, beta=0.9999) ----
    ("G1", "G1_cbfocal_g1", S26, REVIEWED, ["--loss", "cbfocal", "--gamma", "1.0"], "CB-Focal γ=1"),
    ("G3", "G3_cbfocal_g3", S26, REVIEWED, ["--loss", "cbfocal", "--gamma", "3.0"], "CB-Focal γ=3"),
    # ---- group C: backbone size (YOLO26n) ----
    ("N0", "N0_yolo26n_bce",     N26, REVIEWED, [],                              "YOLO26n baseline"),
    ("N1", "N1_yolo26n_cbfocal", N26, REVIEWED, ["--loss", "cbfocal"],           "YOLO26n + CB-Focal"),
    # ---- group D: cross-dataset (BRACOL ORIGINAL sparse annotations) ----
    ("O0", "O0_original_bce",     S26, ORIGINAL, [],                             "ORIGINAL sparse-annotation baseline"),
    ("O1", "O1_original_cbfocal", S26, ORIGINAL, ["--loss", "cbfocal",
                                                  "--cb-samples", "200", "311", "355", "413"],
     "ORIGINAL + CB-Focal"),
    # ---- group E: cross-dataset generalization on an independent dataset (RoCoLe coffee leaf, 6 classes, 26:1 long tail) ----
    ("R0", "R0_rocole_bce",     S26, ROCOLE, [],                          "RoCoLe independent-dataset baseline"),
    ("R1", "R1_rocole_cbfocal", S26, ROCOLE, ["--loss", "cbfocal",
                                              "--cb-samples", "554", "241", "116", "44", "21", "117"],
     "RoCoLe + CB-Focal"),
]


def done(name: str) -> bool:
    return os.path.exists(os.path.join(ROOT, "runs", name, "weights", "best.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true", help="print commands only, do not execute")
    ap.add_argument("--only", default="", help="run only the specified tag")
    ap.add_argument("--skip-done", action="store_true", default=True, help="skip experiments that are already completed")
    args = ap.parse_args()

    todo = [e for e in EXPERIMENTS if (not args.only or e[0] == args.only)]

    for tag, name, model, data, extra, note in todo:
        if args.skip_done and done(name):
            print(f"[skip] {tag} ({name}) best.pt already exists")
            continue

        cmd = [
            PY, "train_baseline.py",
            "--model", model,
            "--data", data,
            "--epochs", str(args.epochs),
            "--batch", str(args.batch),
            "--name", name,
            "--tag", tag,
            "--note", note,
        ] + extra

        print("\n" + "=" * 70)
        print(f"[{tag}] {name} —— {note}")
        print("cmd:", " ".join(cmd))
        print("=" * 70)

        if args.dry_run:
            continue

        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print(f"[failed] {tag} returned code {r.returncode}; stopping subsequent experiments")
            break

    print("\nAll experiments processed.")


if __name__ == "__main__":
    main()
