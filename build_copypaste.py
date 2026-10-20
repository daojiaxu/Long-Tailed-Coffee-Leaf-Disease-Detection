# -*- coding: utf-8 -*-
"""
Instance-level Copy-Paste oversampling (rare-class lesion pasting augmentation)

For the long-tailed BRACOL distribution, rare-class (Cercospora=133 / Miner=171)
lesion instances are cropped from the source images, randomly scaled / flipped
horizontally, and pasted onto training images, with bbox labels generated in
sync, raising the number of rare-class instances.

This is the state-of-the-art data augmentation for long-tailed detection
(Simple Copy-Paste, 2021), finer-grained than RFS whole-image duplication.

Usage:
  # small-scale validation (only the first 20 images)
  python build_copypaste.py --limit 20 --out data/copypaste_demo

  # full generation
  python build_copypaste.py --out data/copypaste_cp

Parameters:
  --per-img N        number of rare-class lesions pasted per image (default: 1)
  --scale-range a b  scale range of the pasted patch (default: 0.8 1.2)
  --iou-thresh t     maximum IoU with existing bboxes (default: 0.3; resample if exceeded)
  --pad-ratio r      crop padding ratio (keeps a little leaf context, default: 0.15)
  --rare-classes ... rare-class ids (default: 0 1)
  --seed N           random seed (default: 42)
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_yolo_labels(path):
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        boxes.append([int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4])])
    return boxes


def iou(a, b):
    """a/b are xyxy pixel coordinates"""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    denom = aa + bb - inter
    return inter / denom if denom > 0 else 0.0


def extract_instance(img, box_xyxy, pad_ratio):
    """Crop a lesion patch from an image (with padding); returns (patch, rel_bbox). box_xyxy is in pixel coordinates."""
    H, W = img.shape[:2]
    x1 = max(0, int(round(box_xyxy[0])))
    y1 = max(0, int(round(box_xyxy[1])))
    x2 = min(W, int(round(box_xyxy[2])))
    y2 = min(H, int(round(box_xyxy[3])))
    bw, bh = x2 - x1, y2 - y1
    if bw < 2 or bh < 2:
        return None, None
    pad_w = int(round(bw * pad_ratio))
    pad_h = int(round(bh * pad_ratio))
    px1 = max(0, x1 - pad_w); py1 = max(0, y1 - pad_h)
    px2 = min(W, x2 + pad_w); py2 = min(H, y2 + pad_h)
    patch = img[py1:py2, px1:px2].copy()
    ph, pw = patch.shape[:2]
    if ph < 2 or pw < 2:
        return None, None
    # lesion bbox relative to the patch (normalized to the patch)
    rx1 = (x1 - px1) / pw; ry1 = (y1 - py1) / ph
    rx2 = (x2 - px1) / pw; ry2 = (y2 - py1) / ph
    rel_cx = (rx1 + rx2) / 2; rel_cy = (ry1 + ry2) / 2
    rel_w = rx2 - rx1; rel_h = ry2 - ry1
    return patch, (rel_cx, rel_cy, rel_w, rel_h)


def leaf_ratio(img, px1, py1, npw, nph):
    """Green-leaf-pixel ratio of the target region (HSV: H 25-95 and S>30)"""
    region = img[py1:py1 + nph, px1:px1 + npw]
    if region.size == 0:
        return 0.0
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    h, s = hsv[..., 0], hsv[..., 1]
    return float(((h >= 25) & (h <= 95) & (s > 30)).mean())


def _blend(img, patch, px1, py1):
    """Poisson seamless blending (seamlessClone); removes rectangular color borders"""
    mask = 255 * np.ones(patch.shape[:2], dtype=np.uint8)
    center = (px1 + patch.shape[1] // 2, py1 + patch.shape[0] // 2)
    img[:] = cv2.seamlessClone(patch, img, mask, center, cv2.NORMAL_CLONE)


def paste_instance(img, inst, rng, scale_range, iou_thresh, existing_px):
    """Paste the instance onto img; returns the new bbox (normalized xywh) or None.
    existing_px is the list of existing bboxes in xyxy pixel coordinates.
    Positions are preferably sampled on the green leaf surface (leaf_ratio >= 0.5);
    if no leaf-surface position is found within 50 attempts, fall back to any
    position satisfying the IoU constraint."""
    patch = inst["patch"]
    rel_cx, rel_cy, rel_w, rel_h = inst["rel_bbox"]

    scale = rng.uniform(scale_range[0], scale_range[1])
    if rng.random() < 0.5:
        patch = patch[:, ::-1].copy()
        rel_cx = 1.0 - rel_cx

    ph, pw = patch.shape[:2]
    npw = max(2, int(round(pw * scale)))
    nph = max(2, int(round(ph * scale)))
    patch = cv2.resize(patch, (npw, nph), interpolation=cv2.INTER_LINEAR)

    nbw = rel_w * npw
    nbh = rel_h * nph
    if nbw < 2 or nbh < 2:
        return None

    H, W = img.shape[:2]
    cx_lo = npw / 2.0
    cx_hi = max(cx_lo, W - npw / 2.0)
    cy_lo = nph / 2.0
    cy_hi = max(cy_lo, H - nph / 2.0)

    fallback = None  # positions satisfying IoU but not on the leaf surface
    for _ in range(50):
        cx = rng.uniform(cx_lo, cx_hi)
        cy = rng.uniform(cy_lo, cy_hi)
        px1 = int(round(cx - npw / 2.0))
        py1 = int(round(cy - nph / 2.0))
        bx_c = px1 + rel_cx * npw
        by_c = py1 + rel_cy * nph
        bbox = [bx_c - nbw / 2.0, by_c - nbh / 2.0, bx_c + nbw / 2.0, by_c + nbh / 2.0]
        if any(iou(bbox, e) >= iou_thresh for e in existing_px):
            continue
        if fallback is None:
            fallback = (px1, py1, bbox)
        if leaf_ratio(img, px1, py1, npw, nph) >= 0.5:
            _blend(img, patch, px1, py1)
            return [bx_c / W, by_c / H, nbw / W, nbh / H]

    if fallback is not None:
        px1, py1, bbox = fallback
        _blend(img, patch, px1, py1)
        return [bbox[0] / W, bbox[1] / H, nbw / W, nbh / H]
    return None


def main():
    ap = argparse.ArgumentParser(description="Copy-Paste rare-class instance oversampling")
    ap.add_argument("--data", default="data/data_reviewed.yaml")
    ap.add_argument("--out", default="data/copypaste_cp")
    ap.add_argument("--per-img", type=int, default=1)
    ap.add_argument("--scale-range", type=float, nargs=2, default=[0.8, 1.2])
    ap.add_argument("--iou-thresh", type=float, default=0.3)
    ap.add_argument("--pad-ratio", type=float, default=0.08)
    ap.add_argument("--min-leaf", type=float, default=0.35, help="minimum leaf-surface ratio of a patch; discarded if below")
    ap.add_argument("--rare-classes", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="process only the first N images (0 = all)")
    args = ap.parse_args()

    src_yaml = Path(args.data).resolve()
    cfg = yaml.safe_load(src_yaml.read_text(encoding="utf-8"))
    src_root = Path(cfg["path"]).resolve()
    train_rel = cfg["train"]
    labels_rel = train_rel.replace("images", "labels")
    src_imgs = src_root / train_rel
    src_lbls = src_root / labels_rel

    out_root = Path(args.out).resolve()
    out_imgs = out_root / "train" / "images"
    out_lbls = out_root / "train" / "labels"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_lbls.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    rare = set(args.rare_classes)

    # ---------- 1. build the rare-class instance library ----------
    instances = []
    img_paths = sorted(src_imgs.glob("*.jpg")) or sorted(src_imgs.glob("*.png"))
    print(f"[1/3] building instance library: scanning {len(img_paths)} training images ...")
    for ip in img_paths:
        lbl = src_lbls / (ip.stem + ".txt")
        boxes = load_yolo_labels(lbl)
        if not boxes:
            continue
        img = cv2.imdecode(np.fromfile(str(ip), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        for b in boxes:
            if b[0] not in rare:
                continue
            cx, cy, w, h = b[1], b[2], b[3], b[4]
            H, W = img.shape[:2]
            x1 = (cx - w / 2) * W; y1 = (cy - h / 2) * H
            x2 = (cx + w / 2) * W; y2 = (cy + h / 2) * H
            patch, rel = extract_instance(img, [x1, y1, x2, y2], args.pad_ratio)
            if patch is None:
                continue
            # discard patches with dirty background (white background / large non-leaf areas) to avoid pasting artifacts
            ph, pw = patch.shape[:2]
            if leaf_ratio(patch, 0, 0, pw, ph) < args.min_leaf:
                continue
            instances.append({"cls": b[0], "patch": patch, "rel_bbox": rel})

    # per-class instance counts + inverse-frequency sampling weights
    from collections import Counter
    cnt = Counter(i["cls"] for i in instances)
    print(f"     instance library: " + ", ".join(f"cls{c}={cnt.get(c, 0)}" for c in sorted(rare)))
    if not instances:
        print("[error] instance library is empty; check the rare-class ids")
        return
    cls_list = sorted(cnt.keys())
    weights = np.array([1.0 / cnt[c] for c in cls_list])
    weights /= weights.sum()

    # ---------- 2. paste image by image ----------
    targets = img_paths if args.limit <= 0 else img_paths[:args.limit]
    print(f"[2/3] pasting augmentation: processing {len(targets)} images, {args.per_img} lesions per image ...")
    pasted = Counter()
    for idx, ip in enumerate(targets):
        lbl = src_lbls / (ip.stem + ".txt")
        boxes = load_yolo_labels(lbl)
        img = cv2.imdecode(np.fromfile(str(ip), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        existing_px = []
        for b in boxes:
            x1 = (b[1] - b[3] / 2) * W; y1 = (b[2] - b[4] / 2) * H
            x2 = (b[1] + b[3] / 2) * W; y2 = (b[2] + b[4] / 2) * H
            existing_px.append([x1, y1, x2, y2])

        new_boxes = []
        for _ in range(args.per_img):
            c = rng.choice(cls_list, p=weights)
            inst = rng.choice([i for i in instances if i["cls"] == c])
            nb = paste_instance(img, inst, rng, args.scale_range, args.iou_thresh, existing_px)
            if nb is not None:
                new_boxes.append([c] + nb)
                # add the new bbox to existing to avoid overlapping repeat pastes within the same image
                cx_n, cy_n, w_n, h_n = nb
                x1 = (cx_n - w_n / 2) * W; y1 = (cy_n - h_n / 2) * H
                x2 = (cx_n + w_n / 2) * W; y2 = (cy_n + h_n / 2) * H
                existing_px.append([x1, y1, x2, y2])
                pasted[c] += 1

        # write the image + merged labels
        cv2.imencode('.jpg', img)[1].tofile(str(out_imgs / ip.name))
        lines = [f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in boxes]
        lines += [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in new_boxes]
        (out_lbls / (ip.stem + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")

        if (idx + 1) % 200 == 0:
            print(f"     processed {idx + 1}/{len(targets)}")

    # ---------- 3. write data.yaml ----------
    new_yaml = {
        "path": str(out_root),
        "train": "train/images",
        "val": str(src_root / cfg["val"]),
        "test": str(src_root / cfg["test"]),
        "nc": cfg["nc"],
        "names": cfg["names"],
    }
    out_yaml = out_root / "data.yaml"
    out_yaml.write_text(yaml.safe_dump(new_yaml, allow_unicode=True), encoding="utf-8")

    print(f"[3/3] done")
    print(f"     pasted lesions: " + ", ".join(f"cls{c}=+{pasted.get(c, 0)}" for c in sorted(rare)))
    print(f"     data.yaml written to: {out_yaml}")
    print(f"     training command: --data {out_yaml}")


if __name__ == "__main__":
    main()
