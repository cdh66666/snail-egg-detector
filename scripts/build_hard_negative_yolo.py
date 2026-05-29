from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def copy_tree_split(src_root: Path, dst_root: Path, split: str) -> tuple[int, int]:
    src_img = src_root / "images" / split
    src_lab = src_root / "labels" / split
    dst_img = dst_root / "images" / split
    dst_lab = dst_root / "labels" / split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lab.mkdir(parents=True, exist_ok=True)
    image_count = 0
    box_count = 0
    for image_path in sorted(p for p in src_img.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        target_name = f"pos_{image_path.name}"
        shutil.copy2(image_path, dst_img / target_name)
        label_src = src_lab / f"{image_path.stem}.txt"
        label_dst = dst_lab / f"{Path(target_name).stem}.txt"
        text = label_src.read_text(encoding="utf-8") if label_src.exists() else ""
        label_dst.write_text(text, encoding="utf-8")
        image_count += 1
        box_count += len([line for line in text.splitlines() if line.strip()])
    return image_count, box_count


def copy_negative(image_path: Path, dst_root: Path, split: str, index: int, max_dim: int) -> bool:
    image = cv2.imread(str(image_path))
    if image is None:
        return False
    h, w = image.shape[:2]
    if min(h, w) < 64:
        return False
    scale = min(1.0, max_dim / float(max(h, w)))
    if scale < 0.999:
        image = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    dst_img = dst_root / "images" / split
    dst_lab = dst_root / "labels" / split
    dst_img.mkdir(parents=True, exist_ok=True)
    dst_lab.mkdir(parents=True, exist_ok=True)
    name = f"neg_{index:04d}_{image_path.stem[:70]}.jpg"
    ok = cv2.imwrite(str(dst_img / name), image, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        return False
    (dst_lab / f"{Path(name).stem}.txt").write_text("", encoding="utf-8")
    return True


def write_yaml(dst_root: Path) -> None:
    (dst_root / "pinkeggs_hardneg.yaml").write_text(
        (
            f"path: {dst_root.resolve().as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "names:\n"
            "  0: eggs\n"
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a YOLO dataset with real hard-negative background images.")
    parser.add_argument("--positive-root", type=Path, default=Path("data/yolo_pinkeggs_clean_500"))
    parser.add_argument("--negative-dir", type=Path, default=Path("data/hard_negatives_wikimedia/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/yolo_pinkeggs_hardneg_v2"))
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--train-ratio", type=float, default=0.76)
    parser.add_argument("--val-ratio", type=float, default=0.12)
    parser.add_argument("--max-negative-dim", type=int, default=960)
    args = parser.parse_args()

    if args.clean and args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {"positive": {}, "negative": {}}
    for split in ["train", "val", "test"]:
        images, boxes = copy_tree_split(args.positive_root, args.output_dir, split)
        summary["positive"][split] = {"images": images, "boxes": boxes}

    negatives = sorted(p for p in args.negative_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    rng = random.Random(args.seed)
    rng.shuffle(negatives)
    n = len(negatives)
    train_end = int(n * args.train_ratio)
    val_end = train_end + int(n * args.val_ratio)
    splits = {
        "train": negatives[:train_end],
        "val": negatives[train_end:val_end],
        "test": negatives[val_end:],
    }
    neg_index = 0
    for split, paths in splits.items():
        count = 0
        for path in paths:
            if copy_negative(path, args.output_dir, split, neg_index, args.max_negative_dim):
                count += 1
                neg_index += 1
        summary["negative"][split] = {"images": count}

    write_yaml(args.output_dir)
    summary["output"] = str(args.output_dir)
    summary["yaml"] = str(args.output_dir / "pinkeggs_hardneg.yaml")
    (args.output_dir / "dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
