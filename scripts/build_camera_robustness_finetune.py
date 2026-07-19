from __future__ import annotations

import argparse
import hashlib
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def copy_dataset(source: Path, output: Path) -> None:
    for kind in ("images", "labels"):
        for split in ("train", "val", "test"):
            for path in (source / kind / split).glob("*"):
                if path.is_file() and path.suffix.lower() != ".cache":
                    link_or_copy(path, output / kind / split / path.name)


def decoded_sha1(frame: np.ndarray) -> str:
    return hashlib.sha1(frame.tobytes()).hexdigest()


def transform(frame: np.ndarray, condition: str, rng: random.Random) -> np.ndarray:
    result = frame.astype(np.float32)
    if condition == "cool":
        result[:, :, 0] *= rng.uniform(1.10, 1.25)
        result[:, :, 2] *= rng.uniform(0.72, 0.90)
    elif condition == "warm":
        result[:, :, 2] *= rng.uniform(1.08, 1.24)
        result[:, :, 0] *= rng.uniform(0.72, 0.92)
    elif condition == "dark":
        result = result * rng.uniform(0.42, 0.68) + rng.uniform(-8, 2)
    elif condition == "bright":
        result = result * rng.uniform(1.18, 1.48) + rng.uniform(8, 28)
    elif condition == "motion":
        result = np.clip(result, 0, 255).astype(np.uint8)
        kernel_size = rng.choice((3, 5, 7))
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        if rng.random() < 0.5:
            kernel[kernel_size // 2, :] = 1.0 / kernel_size
        else:
            np.fill_diagonal(kernel, 1.0 / kernel_size)
        return cv2.filter2D(result, -1, kernel)
    elif condition == "low_res":
        height, width = frame.shape[:2]
        scale = rng.uniform(0.42, 0.70)
        small = cv2.resize(frame, (max(16, round(width * scale)), max(16, round(height * scale))), interpolation=cv2.INTER_AREA)
        return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)
    return np.clip(result, 0, 255).astype(np.uint8)


def labeled_images(root: Path) -> tuple[list[Path], list[Path]]:
    positives, negatives = [], []
    for image in (root / "images" / "train").iterdir():
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTS:
            continue
        label = root / "labels" / "train" / f"{image.stem}.txt"
        (positives if label.exists() and label.stat().st_size else negatives).append(image)
    return positives, negatives


def unique_images(paths: list[Path]) -> list[Path]:
    unique = []
    hashes = set()
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        digest = decoded_sha1(frame)
        if digest not in hashes:
            hashes.add(digest)
            unique.append(path)
    return unique


def write_variant(source_root: Path, image: Path, output: Path, stem: str, condition: str, rng: random.Random) -> bool:
    frame = cv2.imread(str(image))
    if frame is None:
        return False
    variant = transform(frame, condition, rng)
    destination = output / "images" / "train" / f"{stem}.jpg"
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), variant, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(78, 95)])
    label = source_root / "labels" / "train" / f"{image.stem}.txt"
    label_destination = output / "labels" / "train" / f"{stem}.txt"
    label_destination.parent.mkdir(parents=True, exist_ok=True)
    if label.exists():
        shutil.copy2(label, label_destination)
    else:
        label_destination.write_text("", encoding="ascii")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Build whole-frame camera robustness fine-tuning data.")
    parser.add_argument("--source", type=Path, default=Path("data/yolo_pinkeggs_dual_capture_v20_640x480"))
    parser.add_argument("--small-source", type=Path, default=Path("data/yolo_pinkeggs_real_card_v21_640x480"))
    parser.add_argument("--output", type=Path, default=Path("data/yolo_pinkeggs_camera_balanced_v23_640x480"))
    parser.add_argument("--small-count", type=int, default=300)
    parser.add_argument("--base-positive-count", type=int, default=180)
    parser.add_argument("--negative-count", type=int, default=360)
    parser.add_argument("--camera-variants-per-image", type=int, default=2)
    parser.add_argument("--negative-repeat-count", type=int, default=600)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")

    rng = random.Random(args.seed)
    copy_dataset(args.source, args.output)
    positives, negatives = labeled_images(args.source)
    camera_positives = unique_images([path for path in positives if path.stem.startswith("cam")])
    base_positives = unique_images([path for path in positives if not path.stem.startswith("cam")])
    negatives = unique_images(negatives)
    rng.shuffle(base_positives)
    rng.shuffle(negatives)

    counts = {"camera_variants": 0, "base_variants": 0, "negative_variants": 0, "negative_repeats": 0, "small_examples": 0}
    conditions = ("cool", "warm", "dark", "bright", "motion", "low_res")
    for index, image in enumerate(camera_positives):
        start = index % len(conditions)
        selected_conditions = [conditions[(start + offset) % len(conditions)] for offset in range(args.camera_variants_per_image)]
        for condition in selected_conditions:
            stem = f"robust_camera_{index:04d}_{condition}"
            counts["camera_variants"] += int(write_variant(args.source, image, args.output, stem, condition, rng))

    for index, image in enumerate(base_positives[: args.base_positive_count]):
        condition = conditions[index % len(conditions)]
        stem = f"robust_base_{index:04d}_{condition}"
        counts["base_variants"] += int(write_variant(args.source, image, args.output, stem, condition, rng))

    for index, image in enumerate(negatives[: args.negative_count]):
        condition = conditions[index % len(conditions)]
        stem = f"robust_negative_{index:04d}_{condition}"
        counts["negative_variants"] += int(write_variant(args.source, image, args.output, stem, condition, rng))

    # Preserve hard-negative weight without inventing new visual content. Exact
    # repeats are explicit dataset weighting and remain training-only.
    repeat_pool = negatives or []
    for index in range(args.negative_repeat_count):
        image = repeat_pool[index % len(repeat_pool)]
        label = args.source / "labels" / "train" / f"{image.stem}.txt"
        stem = f"weighted_negative_{index:04d}_{image.stem}"
        link_or_copy(image, args.output / "images" / "train" / f"{stem}{image.suffix.lower()}")
        destination = args.output / "labels" / "train" / f"{stem}.txt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if label.exists():
            link_or_copy(label, destination)
        else:
            destination.write_text("", encoding="ascii")
        counts["negative_repeats"] += 1

    small_images = sorted((args.small_source / "images" / "train").glob("realcard_*.jpg"))
    rng.shuffle(small_images)
    for image in small_images[: args.small_count]:
        label = args.small_source / "labels" / "train" / f"{image.stem}.txt"
        link_or_copy(image, args.output / "images" / "train" / image.name)
        link_or_copy(label, args.output / "labels" / "train" / label.name)
        counts["small_examples"] += 1

    yaml_path = args.output / "pinkeggs_camera_robust.yaml"
    yaml_path.write_text(
        f"path: {args.output.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: eggs\n",
        encoding="ascii",
    )
    print(counts)
    print(yaml_path)


if __name__ == "__main__":
    main()
