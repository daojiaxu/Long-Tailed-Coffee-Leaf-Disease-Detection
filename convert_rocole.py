# -*- coding: utf-8 -*-
"""
RoCoLe -> YOLO detection format conversion

The original RoCoLe annotations are "leaf segmentation polygons + image-level
classification labels" (Labelbox export), not YOLO detection boxes. This script
does the following:
  1. converts each image's leaf segmentation polygon -> its bounding box
  2. uses the image-level classification label (6 classes: healthy / rust 4
     levels / red spider mite) as the box class, forming YOLO txt annotations
  3. splits train/val/test stratified 70/20/10 and writes the standard
     ultralytics directory layout

Output: archive/RoCoLe/{images,labels}/{train,val,test}/ + data.yaml
"""
import csv
import json
import os
import random
import shutil
from collections import Counter, defaultdict

from PIL import Image

SRC = "RoCoLe A robusta coffee leaf images dataset"
DST = "data/RoCoLe"
CLASSES = ["healthy", "rust_level_1", "rust_level_2", "rust_level_3", "rust_level_4", "red_spider_mite"]

random.seed(42)


def main():
    # ---- 1. read the CSV to extract (filename, class_idx, polygon) ----
    records = []
    unknown = Counter()
    with open(os.path.join(SRC, "Annotations", "RoCoLE-csv.csv"), encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row["External ID"]
            label = json.loads(row["Label"])
            cls_name = label.get("classification", "?")
            if cls_name not in CLASSES:
                unknown[cls_name] += 1
                continue
            cls_idx = CLASSES.index(cls_name)
            leaf = label["Leaf"][0]
            poly = [(p["x"], p["y"]) for p in leaf["geometry"]]
            records.append((fn, cls_idx, poly))

    print(f"Parsed {len(records)} annotations")
    if unknown:
        print(f"Unknown classes: {dict(unknown)}")

    # ---- 2. read image sizes and compute normalized bboxes ----
    size_dist = Counter()
    norm_records = []
    for fn, cls_idx, poly in records:
        img_path = os.path.join(SRC, "Photos", fn)
        with Image.open(img_path) as im:
            w, h = im.size
        size_dist[(w, h)] += 1
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        # normalize to [0,1] and clip the boundaries
        xc = ((xmin + xmax) / 2) / w
        yc = ((ymin + ymax) / 2) / h
        bw = (xmax - xmin) / w
        bh = (ymax - ymin) / h
        xc = max(0.0, min(1.0, xc))
        yc = max(0.0, min(1.0, yc))
        bw = max(0.0, min(1.0, bw))
        bh = max(0.0, min(1.0, bh))
        norm_records.append((fn, cls_idx, xc, yc, bw, bh))

    print(f"Image size distribution: {dict(size_dist)}")

    # ---- 3. stratified split (per class, 70/20/10) ----
    by_cls = defaultdict(list)
    for r in norm_records:
        by_cls[r[1]].append(r)

    splits = {"train": [], "val": [], "test": []}
    for cls_idx, items in by_cls.items():
        random.shuffle(items)
        n = len(items)
        n_test = max(1, round(n * 0.10))
        n_val = max(1, round(n * 0.20))
        n_train = n - n_val - n_test
        splits["train"] += items[:n_train]
        splits["val"] += items[n_train:n_train + n_val]
        splits["test"] += items[n_train + n_val:]

    for s in splits:
        print(f"{s}: {len(splits[s])}")

    # ---- 4. write the YOLO directory structure ----
    for s in ["train", "val", "test"]:
        os.makedirs(os.path.join(DST, "images", s), exist_ok=True)
        os.makedirs(os.path.join(DST, "labels", s), exist_ok=True)

    for s in ["train", "val", "test"]:
        for fn, cls_idx, xc, yc, bw, bh in splits[s]:
            # copy the image
            shutil.copy2(os.path.join(SRC, "Photos", fn), os.path.join(DST, "images", s, fn))
            # write the label
            base = os.path.splitext(fn)[0]
            lbl_path = os.path.join(DST, "labels", s, base + ".txt")
            with open(lbl_path, "w") as f:
                f.write(f"{cls_idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

    # ---- 5. data.yaml ----
    yaml_content = (
        "# RoCoLe (Robusta Coffee Leaf) leaf-condition detection - 6 classes (long-tailed)\n"
        f"path: data/RoCoLe\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "nc: 6\n"
        f"names: {CLASSES}\n"
    )
    with open(os.path.join(DST, "data.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml_content)

    print(f"\nConversion complete -> {DST}")
    print(f"Class distribution (train): {dict(Counter(r[1] for r in splits['train']))}")


if __name__ == "__main__":
    main()
