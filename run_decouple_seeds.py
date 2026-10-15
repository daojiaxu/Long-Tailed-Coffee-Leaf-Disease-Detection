# -*- coding: utf-8 -*-
"""串行驱动 D1 解耦训练 5-seed 显著性 + D3 解耦×Copy-Paste 组合实验。"""
import subprocess
import sys

PY = sys.executable
BASE = ["train_baseline.py", "--epochs", "50", "--batch", "32",
        "--loss", "cb", "--freeze", "23"]


def run(cmd, label):
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}", flush=True)
    print("CMD:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    print(f"[返回码] {r.returncode} <- {label}", flush=True)
    return r.returncode


def main():
    # ---- 1. D1 解耦训练 5-seed 显著性 ----
    for s in range(5):
        cmd = [PY] + BASE + [
            "--model", "runs/B0_yolo26s_640/weights/best.pt",
            "--data", "data/data_reviewed.yaml",
            "--name", f"D1s{s}_decouple_cb_640",
            "--tag", f"D1s{s}",
            "--seed", str(s),
            "--note", f"[decouple] D1 5seed显著性 seed={s}",
        ]
        rc = run(cmd, f"D1 5-seed seed={s}")
        if rc != 0:
            print(f"[警告] seed={s} 失败(rc={rc})，继续下一个", flush=True)

    # ---- 2. D3 解耦训练 + Copy-Paste 组合 ----
    cmd = [PY] + BASE + [
        "--model", "runs/CP1_cp_bce_640/weights/best.pt",
        "--data", "data/copypaste_cp/data.yaml",
        "--name", "D3_cp_decouple_cb_640",
        "--tag", "D3",
        "--cb-samples", "845", "672", "1310", "4184",
        "--note", "[decouple+CP] CP1骨干+freeze23+CB重训head",
    ]
    run(cmd, "D3 解耦训练 + Copy-Paste 组合")

    print("\n[全部完成]", flush=True)


if __name__ == "__main__":
    main()
