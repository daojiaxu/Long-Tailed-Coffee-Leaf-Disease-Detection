# -*- coding: utf-8 -*-
"""
Copy-Paste 实例级过采样（稀有类病斑粘贴增广）

针对 BRACOL 长尾分布，把稀有类(Cercospora=133 / Miner=171)病斑实例从源图抠出，
随机缩放/水平翻转后粘贴到训练图上，同步生成 bbox 标签，抬升稀有类实例数。

这是长尾检测的 SOTA 数据增广（Simple Copy-Paste, 2021），粒度比 RFS 整图重复采样更细。

用法：
  # 小规模验证（只处理前 20 张图）
  python experiments/build_copypaste.py --limit 20 --out experiments/copypaste_demo

  # 全量生成
  python experiments/build_copypaste.py --out experiments/copypaste_cp

参数：
  --per-img N        每张图粘贴的稀有病斑数（默认 1）
  --scale-range a b  粘贴病斑缩放范围（默认 0.8 1.2）
  --iou-thresh t     与现有 bbox 的最大 IoU（默认 0.3，超过则换位置）
  --pad-ratio r      抠图外扩比例（保留少量叶面上下文，默认 0.15）
  --rare-classes ... 稀有类 id（默认 0 1）
  --seed N           随机种子（默认 42）
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
    """a/b 为 xyxy 像素坐标"""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    denom = aa + bb - inter
    return inter / denom if denom > 0 else 0.0


def extract_instance(img, box_xyxy, pad_ratio):
    """从图中抠出病斑 patch（外扩 padding），返回 (patch, rel_bbox)。box_xyxy 为像素坐标。"""
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
    # 病斑在 patch 内的相对 bbox（归一化到 patch）
    rx1 = (x1 - px1) / pw; ry1 = (y1 - py1) / ph
    rx2 = (x2 - px1) / pw; ry2 = (y2 - py1) / ph
    rel_cx = (rx1 + rx2) / 2; rel_cy = (ry1 + ry2) / 2
    rel_w = rx2 - rx1; rel_h = ry2 - ry1
    return patch, (rel_cx, rel_cy, rel_w, rel_h)


def leaf_ratio(img, px1, py1, npw, nph):
    """目标区域的绿色叶面像素占比（HSV: H 25~95 且 S>30）"""
    region = img[py1:py1 + nph, px1:px1 + npw]
    if region.size == 0:
        return 0.0
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    h, s = hsv[..., 0], hsv[..., 1]
    return float(((h >= 25) & (h <= 95) & (s > 30)).mean())


def _blend(img, patch, px1, py1):
    """Poisson 无缝融合（seamlessClone），消除矩形色差边界"""
    mask = 255 * np.ones(patch.shape[:2], dtype=np.uint8)
    center = (px1 + patch.shape[1] // 2, py1 + patch.shape[0] // 2)
    img[:] = cv2.seamlessClone(patch, img, mask, center, cv2.NORMAL_CLONE)


def paste_instance(img, inst, rng, scale_range, iou_thresh, existing_px):
    """把实例粘贴到 img 上，返回新 bbox（归一化 xywh）或 None。existing_px 为现有 bbox 像素 xyxy 列表。
    位置优先选在绿色叶面上（leaf_ratio>=0.5）；50 次尝试内没有叶面位置时退回任意满足 IoU 的位置。"""
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

    fallback = None  # 满足 IoU 但不在叶面上的位置
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
    ap = argparse.ArgumentParser(description="Copy-Paste 稀有类实例过采样")
    ap.add_argument("--data", default="data/data_reviewed.yaml")
    ap.add_argument("--out", default="data/copypaste_cp")
    ap.add_argument("--per-img", type=int, default=1)
    ap.add_argument("--scale-range", type=float, nargs=2, default=[0.8, 1.2])
    ap.add_argument("--iou-thresh", type=float, default=0.3)
    ap.add_argument("--pad-ratio", type=float, default=0.08)
    ap.add_argument("--min-leaf", type=float, default=0.35, help="patch 叶面占比下限，低于则剔除")
    ap.add_argument("--rare-classes", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 张图（0=全量）")
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

    # ---------- 1. 建立稀有类实例库 ----------
    instances = []
    img_paths = sorted(src_imgs.glob("*.jpg")) or sorted(src_imgs.glob("*.png"))
    print(f"[1/3] 建立实例库：扫描 {len(img_paths)} 张训练图 ...")
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
            # 过滤背景不干净的 patch（含白色背景/大面积非叶面），避免粘贴伪影
            ph, pw = patch.shape[:2]
            if leaf_ratio(patch, 0, 0, pw, ph) < args.min_leaf:
                continue
            instances.append({"cls": b[0], "patch": patch, "rel_bbox": rel})

    # 各类实例数 + inverse-frequency 采样权重
    from collections import Counter
    cnt = Counter(i["cls"] for i in instances)
    print(f"     实例库: " + ", ".join(f"cls{c}={cnt.get(c, 0)}" for c in sorted(rare)))
    if not instances:
        print("[错误] 实例库为空，检查稀有类 id 是否正确")
        return
    cls_list = sorted(cnt.keys())
    weights = np.array([1.0 / cnt[c] for c in cls_list])
    weights /= weights.sum()

    # ---------- 2. 逐图粘贴 ----------
    targets = img_paths if args.limit <= 0 else img_paths[:args.limit]
    print(f"[2/3] 粘贴增广：处理 {len(targets)} 张图，每图 {args.per_img} 个病斑 ...")
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
                # 把新 bbox 加入 existing，避免同一张图内重复重叠
                cx_n, cy_n, w_n, h_n = nb
                x1 = (cx_n - w_n / 2) * W; y1 = (cy_n - h_n / 2) * H
                x2 = (cx_n + w_n / 2) * W; y2 = (cy_n + h_n / 2) * H
                existing_px.append([x1, y1, x2, y2])
                pasted[c] += 1

        # 写图 + 合并标签
        cv2.imencode('.jpg', img)[1].tofile(str(out_imgs / ip.name))
        lines = [f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in boxes]
        lines += [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in new_boxes]
        (out_lbls / (ip.stem + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")

        if (idx + 1) % 200 == 0:
            print(f"     已处理 {idx + 1}/{len(targets)}")

    # ---------- 3. 写 data.yaml ----------
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

    print(f"[3/3] 完成")
    print(f"     粘贴病斑: " + ", ".join(f"cls{c}=+{pasted.get(c, 0)}" for c in sorted(rare)))
    print(f"     输出 data.yaml: {out_yaml}")
    print(f"     训练命令: --data {out_yaml}")


if __name__ == "__main__":
    main()
