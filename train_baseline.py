# -*- coding: utf-8 -*-
"""
YOLO26 coffee leaf disease detection -- training script (with automatic experiment logging)

Example:
  python train_baseline.py --model weights/yolo26s.pt \
      --data data/data_reviewed.yaml --epochs 100 --batch 32 \
      --name B0_yolo26s_640 --tag B0

Notes:
  - project uses a relative path (run from the repo root); output goes to runs/<name>/
  - after training, best.pt is explicitly evaluated on the validation split and
    per-class AP/Recall are printed and saved
  - key metrics are appended to runs/experiment_log.csv for paper aggregation
  - ultralytics automatically writes results.csv and plots under runs/<name>/
    (results.png / confusion_matrix.png / PR_curve.png / F1_curve.png, etc.)
"""
import argparse
import csv
import os
from datetime import datetime

from ultralytics import YOLO


def tolist(x):
    """tensor/list -> python list[float]"""
    if hasattr(x, "tolist"):
        x = x.tolist()
    return [float(v) for v in x]


def main():
    parser = argparse.ArgumentParser(description="YOLO26 training with auto experiment logging")
    parser.add_argument("--model", default="weights/yolo26n.pt", help="path to the pretrained weights")
    parser.add_argument("--data", default="data/data_reviewed.yaml", help="dataset yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="training resolution")
    parser.add_argument("--epochs", type=int, default=100, help="training epochs")
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--name", default="baseline", help="output directory name (under runs/)")
    parser.add_argument("--tag", default="", help="experiment tag, e.g. B0/E1/E2/E3; logged to the experiment log")
    parser.add_argument("--note", default="", help="experiment note; logged to the experiment log")
    parser.add_argument("--device", default="0", help="device, 0=GPU")
    parser.add_argument("--workers", type=int, default=4, help="dataloader workers; set 0 on Windows if errors occur")
    parser.add_argument("--seed", type=int, default=42, help="random seed, for multi-seed repeated experiments")
    parser.add_argument("--loss", default="bce", choices=["bce", "focal", "cb", "cbfocal", "seesaw"],
                        help="classification loss type: bce/focal/cb/cbfocal/seesaw")
    parser.add_argument("--cbfocal", action="store_true", help="[compat] equivalent to --loss cbfocal")
    parser.add_argument("--gamma", type=float, default=2.0, help="focal gamma")
    parser.add_argument("--beta", type=float, default=0.9999, help="beta of the CB weighting")
    parser.add_argument("--beta-start", type=float, default=None, help="beta annealing start value (curriculum weighting, e.g. 0.999)")
    parser.add_argument("--beta-end", type=float, default=None, help="beta annealing end value (e.g. 0.9999)")
    parser.add_argument("--lam", type=float, default=1.0, help="CB balance strength lambda (>1 favors high recall for screening, <1 favors high precision, =1 pure CB)")
    parser.add_argument("--seesaw-p", type=float, default=2.0, help="Seesaw mitigation exponent p")
    parser.add_argument("--cb-samples", type=int, nargs="+", default=None,
                        help="per-class sample counts (in names order); defaults to the BRACOL distribution [133,171,1310,4184]")
    parser.add_argument("--freeze", type=int, default=None,
                        help="freeze the first N layers (for decoupled training). YOLO26s has 24 layers; freeze=23 freezes the backbone+neck and trains only the Detect head")
    args = parser.parse_args()

    # project uses a relative path (run from the repo root); output goes to runs/<name>/
    project = "runs"

    model = YOLO(args.model)

    # ---- optional: inject the long-tail loss ----
    loss_type = "cbfocal" if args.cbfocal else args.loss
    if loss_type != "bce":
        from losses import BRACOL_SAMPLES_PER_CLASS, apply_loss

        samples = args.cb_samples if args.cb_samples is not None else BRACOL_SAMPLES_PER_CLASS
        apply_loss(
            model,
            loss_type=loss_type,
            samples_per_class=samples,
            gamma=args.gamma,
            beta=args.beta,
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            lam=args.lam,
            seesaw_p=args.seesaw_p,
        )
        anneal_note = ""
        if args.beta_start is not None and args.beta_end is not None:
            anneal_note = f", anneal={args.beta_start}->{args.beta_end}"
        print(f"[loss] enabled: {loss_type}, gamma={args.gamma}, beta={args.beta}, lam={args.lam}{anneal_note}, samples={samples}")

    # ---- training ----
    train_kwargs = dict(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        name=args.name,
        device=args.device,
        workers=args.workers,
        project=project,
        exist_ok=True,
        plots=True,
        val=True,
        seed=args.seed,
        pretrained=True,
        verbose=True,
    )
    if args.freeze is not None:
        train_kwargs["freeze"] = args.freeze
        print(f"[freeze] frozen first {args.freeze} layers (decoupled training: only the detection head is re-trained)")
    model.train(**train_kwargs)

    # ---- explicit post-training evaluation of best.pt; print and save per-class metrics ----
    print("\n" + "=" * 60)
    print("Evaluating best.pt on the validation set (per-class metrics)")
    print("=" * 60)
    metrics = model.val(data=args.data, split="val", device=args.device, verbose=False)
    names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)

    ap50 = tolist(metrics.box.ap50)
    ap = tolist(metrics.box.ap)
    precision = tolist(metrics.box.p)   # per-class
    recall = tolist(metrics.box.r)      # per-class
    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)

    print(f"{'class':<12}{'AP50':>10}{'AP50-95':>10}{'Precision':>12}{'Recall':>10}")
    for i, n in enumerate(names):
        print(f"{n:<12}{ap50[i]:>10.4f}{ap[i]:>10.4f}{precision[i]:>12.4f}{recall[i]:>10.4f}")
    print(f"{'mAP':<12}{map50:>10.4f}{map5095:>10.4f}")

    # ---- write the experiment log CSV ----
    log_path = os.path.join(project, "experiment_log.csv")
    header = ["experiment", "model", "imgsz", "epochs", "batch", "date",
              "mAP50", "mAP50-95"]
    for n in names:
        header += [f"{n}_AP50", f"{n}_AP50-95", f"{n}_Precision", f"{n}_Recall"]
    header += ["note"]

    row = {
        "experiment": args.tag or args.name,
        "model": os.path.basename(args.model),
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "batch": args.batch,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "mAP50": round(map50, 4),
        "mAP50-95": round(map5095, 4),
    }
    for i, n in enumerate(names):
        row[f"{n}_AP50"] = round(ap50[i], 4)
        row[f"{n}_AP50-95"] = round(ap[i], 4)
        row[f"{n}_Precision"] = round(precision[i], 4)
        row[f"{n}_Recall"] = round(recall[i], 4)
    row["note"] = ((f"[{loss_type}] " if loss_type != "bce" else "") + args.note)

    new_file = not os.path.exists(log_path)
    with open(log_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if new_file:
            writer.writeheader()
        writer.writerow(row)

    print(f"\nDone. Metrics written to log: {log_path}")
    print(f"Training output dir: {project}/{args.name}/ (contains results.csv, results.png, confusion matrix, and other plots)")


if __name__ == "__main__":
    main()
