# -*- coding: utf-8 -*-
"""
SCI 扩展实验批量运行脚本

按顺序执行扩展实验矩阵，结果自动 append 到 runs/experiment_log.csv。
已完成的实验（runs/<name>/weights/best.pt 存在）会自动跳过。

用法（在项目根目录）:
  python experiments/run_sci_experiments.py            # 顺序跑全部
  python experiments/run_sci_experiments.py --epochs 100 --batch 32
  python experiments/run_sci_experiments.py --dry-run  # 只打印命令不执行
  python experiments/run_sci_experiments.py --only M4   # 只跑 tag=M4

说明:
  - 默认 100 epochs（正式实验），可用 --epochs 覆盖做快速验证
  - 所有实验 640 分辨率、batch 32
  - 已跑的 B0(基线)/E1b(CB-Focal γ=2) 不在列表内，避免重复
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
    # ---- 组A：损失方法对比（REVIEWED @640, YOLO26s）----
    ("M1", "M1_focal_640",  S26, REVIEWED, ["--loss", "focal"],  "Focal Loss 单独(γ=2)"),
    ("M2", "M2_cb_640",     S26, REVIEWED, ["--loss", "cb"],     "CB Loss 单独(β=0.9999)"),
    ("M4", "M4_seesaw_640", S26, REVIEWED, ["--loss", "seesaw"], "Seesaw Loss(p=2)"),
    # ---- 组B：γ 敏感性（CB-Focal, β=0.9999）----
    ("G1", "G1_cbfocal_g1", S26, REVIEWED, ["--loss", "cbfocal", "--gamma", "1.0"], "CB-Focal γ=1"),
    ("G3", "G3_cbfocal_g3", S26, REVIEWED, ["--loss", "cbfocal", "--gamma", "3.0"], "CB-Focal γ=3"),
    # ---- 组C：backbone 尺寸（YOLO26n）----
    ("N0", "N0_yolo26n_bce",     N26, REVIEWED, [],                              "YOLO26n 基线"),
    ("N1", "N1_yolo26n_cbfocal", N26, REVIEWED, ["--loss", "cbfocal"],           "YOLO26n + CB-Focal"),
    # ---- 组D：跨数据集（BRACOL ORIGINAL 稀疏标注）----
    ("O0", "O0_original_bce",     S26, ORIGINAL, [],                             "ORIGINAL 稀疏标注 基线"),
    ("O1", "O1_original_cbfocal", S26, ORIGINAL, ["--loss", "cbfocal",
                                                  "--cb-samples", "200", "311", "355", "413"],
     "ORIGINAL + CB-Focal"),
    # ---- 组E：独立数据集跨域泛化（RoCoLe 咖啡叶，6 类长尾 26:1）----
    ("R0", "R0_rocole_bce",     S26, ROCOLE, [],                          "RoCoLe 独立数据集 基线"),
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
    ap.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    ap.add_argument("--only", default="", help="只跑指定 tag")
    ap.add_argument("--skip-done", action="store_true", default=True, help="跳过已完成的实验")
    args = ap.parse_args()

    todo = [e for e in EXPERIMENTS if (not args.only or e[0] == args.only)]

    for tag, name, model, data, extra, note in todo:
        if args.skip_done and done(name):
            print(f"[跳过] {tag} ({name}) 已存在 best.pt")
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
            print(f"[失败] {tag} 返回码 {r.returncode}，停止后续实验")
            break

    print("\n全部实验处理完毕。")


if __name__ == "__main__":
    main()
