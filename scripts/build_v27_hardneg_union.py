"""Add training-only hard negatives to the deduplicated v26 union dataset.

The v26 validation and test splits are copied unchanged. This keeps the
evaluation holdouts independent while increasing the variety of negative
textures seen during training.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def digest(path: Path) -> str | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return hashlib.sha1(image.tobytes()).hexdigest()


def copy_tree(source: Path, target: Path, split: str, seen: set[str], prefix: str) -> tuple[int, int]:
    image_dir = source / "images" / split
    label_dir = source / "labels" / split
    copied = skipped = 0
    if not image_dir.exists():
        return copied, skipped
    for image in sorted(image_dir.iterdir()):
        if image.suffix.lower() not in EXTS:
            continue
        key = digest(image)
        if key is None or key in seen:
            skipped += 1
            continue
        label = label_dir / f"{image.stem}.txt"
        seen.add(key)
        out_image = target / "images" / split / f"{prefix}_{image.name}"
        out_label = target / "labels" / split / f"{prefix}_{image.stem}.txt"
        out_image.parent.mkdir(parents=True, exist_ok=True)
        out_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, out_image)
        if label.exists():
            shutil.copy2(label, out_label)
        else:
            out_label.write_text("", encoding="ascii")
        copied += 1
    return copied, skipped


def augment_training_negatives(target: Path, seed: int = 20260719) -> int:
    """Create one deterministic camera-like variant per training negative."""
    rng = random.Random(seed)
    image_dir = target / "images" / "train"
    label_dir = target / "labels" / "train"
    negatives = [
        image
        for image in sorted(image_dir.iterdir())
        if image.suffix.lower() in EXTS
        and not (label_dir / f"{image.stem}.txt").read_text(encoding="utf-8").strip()
    ]
    written = 0
    for index, source in enumerate(negatives):
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            continue
        h, w = image.shape[:2]
        alpha = rng.uniform(0.55, 1.45)
        beta = rng.uniform(-20, 24)
        variant = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        hsv = cv2.cvtColor(variant, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + rng.uniform(-8, 8)) % 180
        hsv[:, :, 1] *= rng.uniform(0.65, 1.35)
        variant = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        angle = rng.uniform(-8, 8)
        scale = rng.uniform(0.88, 1.12)
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        matrix[:, 2] += (rng.uniform(-0.04, 0.04) * w, rng.uniform(-0.04, 0.04) * h)
        variant = cv2.warpAffine(variant, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
        if rng.random() < 0.45:
            kernel = rng.choice((3, 5))
            variant = cv2.GaussianBlur(variant, (kernel, kernel), rng.uniform(0.2, 1.1))
        out_stem = f"negaug_{index:05d}_{source.stem}"
        out_image = image_dir / f"{out_stem}.jpg"
        if cv2.imwrite(str(out_image), variant, [cv2.IMWRITE_JPEG_QUALITY, 88]):
            (label_dir / f"{out_stem}.txt").write_text("", encoding="ascii")
            written += 1
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=Path("data/yolo_pinkeggs_union_v26_640x480"))
    parser.add_argument("--hardneg", type=Path, default=Path("data/yolo_pinkeggs_hardneg_v8_field_light_640x480"))
    parser.add_argument("--output", type=Path, default=Path("data/yolo_pinkeggs_union_v27_hardneg_640x480"))
    parser.add_argument("--augment-negatives", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")

    seen: set[str] = set()
    report = {}
    for split in ("train", "val", "test"):
        report[f"base_{split}"] = copy_tree(args.base, args.output, split, seen, "base")
    # Only training images from the extra source are used. Its val/test remain
    # untouched to prevent hidden evaluation leakage.
    report["hardneg_train"] = copy_tree(args.hardneg, args.output, "train", seen, "hardneg")
    if args.augment_negatives:
        report["augmented_training_negatives"] = augment_training_negatives(args.output)
    yaml_path = args.output / "pinkeggs_union_v27.yaml"
    yaml_path.write_text(
        f"path: {args.output.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: snail_eggs\n",
        encoding="ascii",
    )
    print(report)
    print(yaml_path)


if __name__ == "__main__":
    main()
