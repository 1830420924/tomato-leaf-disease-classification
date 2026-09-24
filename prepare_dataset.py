"""将番茄三分类原始图片划分为 train/val/test。"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def split_counts(total: int) -> tuple[int, int, int]:
    train_count = int(total * 0.70)
    val_count = int(total * 0.15)
    test_count = total - train_count - val_count
    return train_count, val_count, test_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("archive_38class/data_38class/raw/color"))
    parser.add_argument("--output", type=Path, default=Path("data_tomato"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=0)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["Tomato___healthy", "Tomato___Early_blight", "Tomato___Late_blight"],
        help="只保留指定的番茄类别",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.source.exists():
        raise FileNotFoundError(f"找不到原始数据目录：{args.source}")

    class_dirs = sorted(path for path in args.source.iterdir() if path.is_dir())
    if not class_dirs:
        raise RuntimeError("原始数据目录中没有类别文件夹。")

    if args.classes:
        requested = set(args.classes)
        available = {path.name for path in class_dirs}
        missing = sorted(requested - available)
        if missing:
            raise RuntimeError(f"source中找不到指定类别：{', '.join(missing)}")
        class_dirs = [path for path in class_dirs if path.name in requested]

    split_dirs = [args.output / name for name in ("train", "val", "test")]
    if any(path.exists() and any(path.rglob("*")) for path in split_dirs) and not args.overwrite:
        raise RuntimeError("目标目录已有数据。如需重新划分，请添加 --overwrite。")

    if args.overwrite:
        for split_dir in split_dirs:
            if split_dir.exists():
                shutil.rmtree(split_dir)

    for split_dir in split_dirs:
        split_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    total_copied = 0
    for class_dir in class_dirs:
        images = sorted(
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        rng.shuffle(images)
        if args.max_per_class > 0:
            images = images[:args.max_per_class]

        train_count, val_count, _ = split_counts(len(images))
        assignments = (
            [("train", image) for image in images[:train_count]]
            + [("val", image) for image in images[train_count:train_count + val_count]]
            + [("test", image) for image in images[train_count + val_count:]]
        )

        for split, image_path in assignments:
            destination = args.output / split / class_dir.name
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, destination / image_path.name)
            total_copied += 1

        print(f"{class_dir.name}: {len(images)} 张")

    print(f"完成：共复制 {total_copied} 张图片。")


if __name__ == "__main__":
    main()
