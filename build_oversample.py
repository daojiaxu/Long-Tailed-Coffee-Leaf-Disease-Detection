# -*- coding: utf-8 -*-
"""
Long-tail handling -- rare-class oversampling (data-level approximation of Repeat-Factor-Sampling)

For the class long tail of BRACOL (Cercospora=133 / Miner=171 are the rare classes),
training images that contain rare classes are duplicated by a repeat factor,
raising the frequency of rare classes in training.

Usage:
  python build_oversample.py \
      --data data/data_reviewed.yaml \
      --out data/oversampled_r3 \
      --repeat 3 \
      --rare-classes 0 1

Notes:
  - Only train is oversampled; val/test are reused as-is (pointing at the source dataset).
  - A new data.yaml is generated; point `--data` at it during training.
"""
import argparse
import shutil
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="rare-class oversampling dataset builder")
    parser.add_argument("--data", required=True, help="source data.yaml")
    parser.add_argument("--out", required=True, help="output directory (will contain train/images, train/labels, data.yaml)")
    parser.add_argument("--repeat", type=int, default=3, help="number of extra copies of rare-class-containing images")
    parser.add_argument("--rare-classes", type=int, nargs="+", default=[0, 1],
                        help="rare-class ids, default 0(Cercospora) 1(Miner)")
    args = parser.parse_args()

    src_yaml = Path(args.data).resolve()
    cfg = yaml.safe_load(src_yaml.read_text(encoding="utf-8"))
    src_root = Path(cfg["path"]).resolve()
    train_rel = cfg["train"]            # e.g. train/images
    labels_rel = train_rel.replace("images", "labels")
    src_imgs = src_root / train_rel
    src_lbls = src_root / labels_rel

    out_root = Path(args.out).resolve()
    out_imgs = out_root / "train" / "images"
    out_lbls = out_root / "train" / "labels"
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_lbls.mkdir(parents=True, exist_ok=True)

    rare = set(args.rare_classes)
    total_orig = 0
    total_rare = 0
    copied = 0

    for lbl in sorted(src_lbls.glob("*.txt")):
        lines = lbl.read_text(encoding="utf-8").strip().splitlines()
        cls_ids = {int(ln.split()[0]) for ln in lines if ln.strip()}
        is_rare = bool(cls_ids & rare)

        stem = lbl.stem
        img = src_imgs / (stem + ".jpg")
        if not img.exists():
            # fall back to other extensions
            cand = list(src_imgs.glob(stem + ".*"))
            if not cand:
                print(f"[skip] image not found: {stem}")
                continue
            img = cand[0]

        # original copy
        shutil.copy2(img, out_imgs / img.name)
        shutil.copy2(lbl, out_lbls / lbl.name)
        total_orig += 1
        copied += 1

        # extra repeat copies for rare-class images
        if is_rare:
            total_rare += 1
            ext = img.suffix
            for k in range(1, args.repeat + 1):
                shutil.copy2(img, out_imgs / f"{stem}_r{k}{ext}")
                shutil.copy2(lbl, out_lbls / f"{stem}_r{k}.txt")
                copied += 1

    # write the new data.yaml
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

    print(f"Done: {total_orig} original images, {total_rare} with rare classes, {copied} total after oversampling")
    print(f"data.yaml written to: {out_yaml}")
    print(f"training command example: --data {out_yaml}")


if __name__ == "__main__":
    main()
