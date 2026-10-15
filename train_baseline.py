# -*- coding: utf-8 -*-
"""
YOLO26 咖啡叶病害检测 —— 训练脚本（带实验台账自动记录）

用法示例:
  python experiments/train_baseline.py --model weights/yolo26s.pt \
      --data experiments/data_reviewed.yaml --epochs 100 --batch 32 \
      --name B0_yolo26s_640 --tag B0

说明:
  - project 用绝对路径，输出落在 runs/<name>/ 下（避免 8.4 的嵌套）
  - 训练结束后在验证集上显式评估 best.pt，打印并落盘 per-class AP/Recall
  - 关键指标自动 append 到 runs/experiment_log.csv，供写论文汇总使用
  - ultralytics 会自动在 runs/<name>/ 下生成 results.csv 及各类图表
    （results.png / confusion_matrix.png / PR_curve.png / F1_curve.png 等）
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
    parser.add_argument("--model", default="weights/yolo26n.pt", help="预训练权重路径")
    parser.add_argument("--data", default="experiments/data_reviewed.yaml", help="数据集 yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="训练分辨率")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--batch", type=int, default=32, help="batch size")
    parser.add_argument("--name", default="baseline", help="输出目录名(位于 runs/ 下)")
    parser.add_argument("--tag", default="", help="实验标签, 如 B0/E1/E2/E3，写入台账")
    parser.add_argument("--note", default="", help="实验备注，写入台账")
    parser.add_argument("--device", default="0", help="设备, 0=GPU")
    parser.add_argument("--workers", type=int, default=4, help="数据加载进程数, Windows 报错可设 0")
    parser.add_argument("--seed", type=int, default=42, help="随机种子, 用于多 seed 重复实验")
    parser.add_argument("--loss", default="bce", choices=["bce", "focal", "cb", "cbfocal", "seesaw"],
                        help="分类损失类型: bce/focal/cb/cbfocal/seesaw")
    parser.add_argument("--cbfocal", action="store_true", help="[兼容] 等价于 --loss cbfocal")
    parser.add_argument("--gamma", type=float, default=2.0, help="focal gamma")
    parser.add_argument("--beta", type=float, default=0.9999, help="CB 权重的 beta")
    parser.add_argument("--beta-start", type=float, default=None, help="β 退火起始值(课程式加权，如 0.999)")
    parser.add_argument("--beta-end", type=float, default=None, help="β 退火终止值(如 0.9999)")
    parser.add_argument("--lam", type=float, default=1.0, help="CB 平衡强度 λ(>1 偏筛查高召回，<1 偏精确，=1 纯 CB)")
    parser.add_argument("--seesaw-p", type=float, default=2.0, help="Seesaw mitigation 指数 p")
    parser.add_argument("--cb-samples", type=int, nargs="+", default=None,
                        help="各类别样本数(按 names 顺序)，默认用 BRACOL 分布 [133,171,1310,4184]")
    parser.add_argument("--freeze", type=int, default=None,
                        help="冻结前 N 层(解耦训练用)。yolo26s 共24层，freeze=23 冻结 backbone+neck 只训 Detect head")
    args = parser.parse_args()

    # project 用相对路径（在仓库根目录运行），输出落在 runs/<name>/ 下
    project = "runs"

    model = YOLO(args.model)

    # ---- 可选：注入长尾损失 ----
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
        print(f"[loss] 已启用: {loss_type}, gamma={args.gamma}, beta={args.beta}, lam={args.lam}{anneal_note}, samples={samples}")

    # ---- 训练 ----
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
        print(f"[freeze] 冻结前 {args.freeze} 层（解耦训练：只重训检测头）")
    model.train(**train_kwargs)

    # ---- 训练后显式评估 best.pt，打印并落盘 per-class 指标 ----
    print("\n" + "=" * 60)
    print("评估 best.pt 于验证集（per-class 指标）")
    print("=" * 60)
    metrics = model.val(data=args.data, split="val", device=args.device, verbose=False)
    names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)

    ap50 = tolist(metrics.box.ap50)
    ap = tolist(metrics.box.ap)
    precision = tolist(metrics.box.p)   # per-class
    recall = tolist(metrics.box.r)      # per-class
    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)

    print(f"{'类别':<12}{'AP50':>10}{'AP50-95':>10}{'Precision':>12}{'Recall':>10}")
    for i, n in enumerate(names):
        print(f"{n:<12}{ap50[i]:>10.4f}{ap[i]:>10.4f}{precision[i]:>12.4f}{recall[i]:>10.4f}")
    print(f"{'mAP':<12}{map50:>10.4f}{map5095:>10.4f}")

    # ---- 写实验台账 CSV ----
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

    print(f"\n完成。指标已写入台账: {log_path}")
    print(f"训练产物目录: {project}/{args.name}/ （含 results.csv、results.png、混淆矩阵等图表）")


if __name__ == "__main__":
    main()
