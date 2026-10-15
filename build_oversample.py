# -*- coding: utf-8 -*-
"""
长尾分布处理 —— 稀有类过采样（Repeat-Factor-Sampling 的数据级近似）

针对 BRACOL 的类别长尾（Cercospora=69 / Miner=86 为稀有类），
对「包含稀有类」的训练图像按倍率复制，抬升稀有类在训练中的出现频率。

用法：
  python experiments/build_oversample.py \
      --data experiments/data_reviewed.yaml \
      --out experiments/oversampled_r3 \
      --repeat 3 \
      --rare-classes 0 1

说明：
  - 只对 train 做过采样，val/test 原样复用（指向源数据集）。
  - 生成新的 data.yaml，训练时用 `--data` 指向它即可。
"""
import argparse
import shutil
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="稀有类过采样数据构建")
    parser.add_argument("--data", required=True, help="源 data.yaml")
    parser.add_argument("--out", required=True, help="输出目录（将生成 train/images、train/labels、data.yaml）")
    parser.add_argument("--repeat", type=int, default=3, help="含稀有类图像的额外复制份数")
    parser.add_argument("--rare-classes", type=int, nargs="+", default=[0, 1],
                        help="稀有类 id，默认 0(Cercospora) 1(Miner)")
    args = parser.parse_args()

    src_yaml = Path(args.data).resolve()
    cfg = yaml.safe_load(src_yaml.read_text(encoding="utf-8"))
    src_root = Path(cfg["path"]).resolve()
    train_rel = cfg["train"]            # 如 train/images
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
            # 兼容其它扩展名
            cand = list(src_imgs.glob(stem + ".*"))
            if not cand:
                print(f"[跳过] 找不到图像: {stem}")
                continue
            img = cand[0]

        # 原始 1 份
        shutil.copy2(img, out_imgs / img.name)
        shutil.copy2(lbl, out_lbls / lbl.name)
        total_orig += 1
        copied += 1

        # 稀有类额外 repeat 份
        if is_rare:
            total_rare += 1
            ext = img.suffix
            for k in range(1, args.repeat + 1):
                shutil.copy2(img, out_imgs / f"{stem}_r{k}{ext}")
                shutil.copy2(lbl, out_lbls / f"{stem}_r{k}.txt")
                copied += 1

    # 写新 data.yaml
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

    print(f"完成：原始图像 {total_orig} 张，含稀有类 {total_rare} 张，过采样后共 {copied} 张")
    print(f"输出 data.yaml: {out_yaml}")
    print(f"训练命令示例: --data {out_yaml}")


if __name__ == "__main__":
    main()
