# -*- coding: utf-8 -*-
"""Serial driver for the D1 decoupled-training 5-seed significance test and the D3 decoupled x Copy-Paste combination experiment."""
import subprocess
import sys

PY = sys.executable
BASE = ["train_baseline.py", "--epochs", "50", "--batch", "32",
        "--loss", "cb", "--freeze", "23"]


def run(cmd, label):
    print(f"\n{'=' * 60}\n{label}\n{'=' * 60}", flush=True)
    print("CMD:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    print(f"[return code] {r.returncode} <- {label}", flush=True)
    return r.returncode


def main():
    # ---- 1. D1 decoupled-training 5-seed significance test ----
    for s in range(5):
        cmd = [PY] + BASE + [
            "--model", "runs/B0_yolo26s_640/weights/best.pt",
            "--data", "data/data_reviewed.yaml",
            "--name", f"D1s{s}_decouple_cb_640",
            "--tag", f"D1s{s}",
            "--seed", str(s),
            "--note", f"[decouple] D1 5-seed significance seed={s}",
        ]
        rc = run(cmd, f"D1 5-seed seed={s}")
        if rc != 0:
            print(f"[warning] seed={s} failed (rc={rc}); continuing", flush=True)

    # ---- 2. D3 decoupled training + Copy-Paste combination ----
    cmd = [PY] + BASE + [
        "--model", "runs/CP1_cp_bce_640/weights/best.pt",
        "--data", "data/copypaste_cp/data.yaml",
        "--name", "D3_cp_decouple_cb_640",
        "--tag", "D3",
        "--cb-samples", "845", "672", "1310", "4184",
        "--note", "[decouple+CP] CP1 backbone + freeze=23 + CB head retrain",
    ]
    run(cmd, "D3 decoupled training + Copy-Paste combination")

    print("\n[all done]", flush=True)


if __name__ == "__main__":
    main()
