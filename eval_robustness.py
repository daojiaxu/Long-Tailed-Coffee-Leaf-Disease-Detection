# -*- coding: utf-8 -*-
"""
Robustness evaluation under distribution shift

Checks whether the CB recall gain on Cercospora holds under five distribution
shifts. Uses the trained best.pt of B0 (BCE baseline) and M2 (pure CB),
evaluated on the BRACOL validation set after applying perturbations.

Perturbation types:
  1. brightness_contrast : brightness/contrast scaling (different exposures)
  2. color_shift         : BGR three-channel color-temperature shift (different white balance / device)
  3. gaussian_noise      : Gaussian noise (sensor noise)
  4. gaussian_blur       : Gaussian blur (focus deviation)
  5. rotate_scale        : rotation +-10 deg + scale 0.9-1.1 (shooting angle/distance, bboxes transformed in sync)

Usage:
  python eval_robustness.py
"""
import os
import shutil
import csv
from datetime import datetime

import numpy as np
import cv2
from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
VAL_IMAGES = os.path.join(BASE, "data/BRACOL_REVIEWED/valid/images")
VAL_LABELS = os.path.join(BASE, "data/BRACOL_REVIEWED/valid/labels")
DATA_YAML = os.path.join(BASE, "data/data_reviewed.yaml")
TMP_ROOT = os.path.join(BASE, "runs/robustness_tmp")
OUT_CSV = os.path.join(BASE, "runs/robustness_results.csv")

MODELS = {
    "BCE (B0)": os.path.join(BASE, "runs/B0_yolo26s_640/weights/best.pt"),
    "CB  (M2)": os.path.join(BASE, "runs/M2_cb_640/weights/best.pt"),
}

# (disturbance name, whether geometric [bbox transform needed])
DISTURBANCES = [
    ("clean", False),
    ("brightness_contrast", False),
    ("color_shift", False),
    ("gaussian_noise", False),
    ("gaussian_blur", False),
    ("rotate_scale", True),
]


def tolist(x):
    if hasattr(x, "tolist"):
        x = x.tolist()
    return [float(v) for v in x]


def _clip(img):
    return np.clip(img, 0, 255).astype(np.uint8)


# ---------- perturbation functions ----------
def perturb_brightness_contrast(img, rng):
    alpha = rng.uniform(0.6, 1.4)
    beta = rng.uniform(-40, 40)
    return _clip(alpha * img.astype(np.float32) + beta)


def perturb_color_shift(img, rng):
    shift = rng.randint(-30, 31, size=3).astype(np.float32)
    return _clip(img.astype(np.float32) + shift)


def perturb_gaussian_noise(img, rng):
    noise = rng.normal(0, 15, img.shape).astype(np.float32)
    return _clip(img.astype(np.float32) + noise)


def perturb_gaussian_blur(img, rng):
    return cv2.GaussianBlur(img, (5, 5), 0)


def perturb_rotate_scale(img, boxes, rng):
    """Rotation + scaling with the bboxes transformed in sync. boxes: list of normalized (cls, cx, cy, w, h)."""
    h, w = img.shape[:2]
    angle = rng.uniform(-10, 10)
    scale = rng.uniform(0.9, 1.1)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    new_boxes = []
    for cls, cx, cy, bw, bh in boxes:
        px, py = cx * w, cy * h
        bw_px, bh_px = bw * w, bh * h
        corners = np.array([
            [px - bw_px / 2, py - bh_px / 2],
            [px + bw_px / 2, py - bh_px / 2],
            [px + bw_px / 2, py + bh_px / 2],
            [px - bw_px / 2, py + bh_px / 2],
        ])
        pts = np.hstack([corners, np.ones((4, 1))])
        new_pts = (M @ pts.T).T
        x1, y1 = new_pts.min(axis=0)
        x2, y2 = new_pts.max(axis=0)
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(w), x2), min(float(h), y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue  # bbox out of bounds after rotation: discard
        new_boxes.append((cls, ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h,
                          (x2 - x1) / w, (y2 - y1) / h))
    return rotated, new_boxes


def read_yolo_labels(label_path):
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:5])
            boxes.append((cls, cx, cy, bw, bh))
    return boxes


def build_disturbed_split(dist_name, seed):
    """Build the perturbed validation set at TMP_ROOT/<dist_name>/ and return the data.yaml path. clean returns the original yaml directly."""
    if dist_name == "clean":
        return DATA_YAML

    out_dir = os.path.join(TMP_ROOT, dist_name)
    img_out = os.path.join(out_dir, "images")
    lbl_out = os.path.join(out_dir, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    rng = np.random.RandomState(seed)
    is_geo = dict(DISTURBANCES)[dist_name]

    for fname in sorted(os.listdir(VAL_IMAGES)):
        img = cv2.imread(os.path.join(VAL_IMAGES, fname))
        if img is None:
            continue
        stem = os.path.splitext(fname)[0]
        label_path = os.path.join(VAL_LABELS, stem + ".txt")
        boxes = read_yolo_labels(label_path)

        if dist_name == "brightness_contrast":
            out_img = perturb_brightness_contrast(img, rng)
        elif dist_name == "color_shift":
            out_img = perturb_color_shift(img, rng)
        elif dist_name == "gaussian_noise":
            out_img = perturb_gaussian_noise(img, rng)
        elif dist_name == "gaussian_blur":
            out_img = perturb_gaussian_blur(img, rng)
        elif dist_name == "rotate_scale":
            out_img, new_boxes = perturb_rotate_scale(img, boxes, rng)
            cv2.imwrite(os.path.join(img_out, fname), out_img)
            with open(os.path.join(lbl_out, stem + ".txt"), "w") as f:
                for cls, ncx, ncy, nbw, nbh in new_boxes:
                    f.write(f"{cls} {ncx:.6f} {ncy:.6f} {nbw:.6f} {nbh:.6f}\n")
            continue
        else:
            out_img = img

        cv2.imwrite(os.path.join(img_out, fname), out_img)
        shutil.copy(label_path, os.path.join(lbl_out, stem + ".txt"))

    yaml_path = os.path.join(out_dir, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {out_dir}\n")
        f.write("train: images\nval: images\ntest: images\n")
        f.write("nc: 4\n")
        f.write("names: ['Cercospora', 'Miner', 'Phoma', 'Rust']\n")
    return yaml_path


def main():
    print("=" * 80)
    print("Robustness evaluation under distribution shift: does the CB gain hold under the 5 shifts?")
    print("=" * 80)

    rows = []
    header = ["disturbance", "model", "Cercospora_Recall", "Cercospora_Precision",
              "Cercospora_AP50", "mAP50", "mAP50-95"]

    for dist_name, _ in DISTURBANCES:
        yaml_path = build_disturbed_split(dist_name, seed=42)
        for model_name, model_path in MODELS.items():
            model = YOLO(model_path)
            metrics = model.val(data=yaml_path, split="val", device=0, imgsz=640,
                                save=False, plots=False, verbose=False,
                                project=TMP_ROOT, name=f"val_{dist_name}_{model_name[:3]}")
            names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
            ap50 = tolist(metrics.box.ap50)
            p = tolist(metrics.box.p)
            r = tolist(metrics.box.r)
            map50 = float(metrics.box.map50)
            map5095 = float(metrics.box.map)
            i = names.index("Cercospora") if "Cercospora" in names else 0
            rows.append([dist_name, model_name, round(r[i], 4), round(p[i], 4),
                         round(ap50[i], 4), round(map50, 4), round(map5095, 4)])
            print(f"{dist_name:<20} {model_name:<10} Cer_R={r[i]:.4f} "
                  f"Cer_P={p[i]:.4f} Cer_AP50={ap50[i]:.4f} mAP50={map50:.4f}")
            del model

    # write the CSV
    with open(OUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    # gain-retention summary table
    print("\n" + "=" * 80)
    print("Cercospora recall gain-retention analysis (CB - BCE)")
    print("=" * 80)
    by_dist = {}
    for dist, model, cer_r, cer_p, cer_ap, m50, m5095 in rows:
        by_dist.setdefault(dist, {})[model] = cer_r
    print(f"{'disturbance':<20}{'BCE_R':>10}{'CB_R':>10}{'gain(CB-BCE)':>14}{'verdict':>8}")
    base_gain = by_dist["clean"]["CB  (M2)"] - by_dist["clean"]["BCE (B0)"]
    for dist, _ in DISTURBANCES:
        bce_r = by_dist[dist]["BCE (B0)"]
        cb_r = by_dist[dist]["CB  (M2)"]
        gain = cb_r - bce_r
        if dist == "clean":
            verdict = "baseline"
        else:
            ratio = gain / base_gain if base_gain else 0
            verdict = "kept" if ratio >= 0.7 else ("partial decay" if ratio >= 0.4 else "lost")
        print(f"{dist:<20}{bce_r:>10.4f}{cb_r:>10.4f}{gain:>+14.4f}{verdict:>8}")

    shutil.rmtree(TMP_ROOT, ignore_errors=True)
    print(f"\nResults written to: {OUT_CSV}")
    print(f"Baseline gain (clean): {base_gain:+.4f}")


if __name__ == "__main__":
    main()
