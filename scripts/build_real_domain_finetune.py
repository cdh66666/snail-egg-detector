from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from pathlib import Path


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def add_pair(image: Path, label: Path, output: Path, split: str, stem: str) -> None:
    link_or_copy(image, output / "images" / split / f"{stem}{image.suffix.lower()}")
    link_or_copy(label, output / "labels" / split / f"{stem}.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a real-MaixCAM-first fine-tuning dataset.")
    parser.add_argument("--camera", type=Path, default=Path("runs/camera_domain_v10"))
    parser.add_argument("--base", type=Path, default=Path("data/yolo_pinkeggs_multi_v14_mined_640x480"))
    parser.add_argument("--output", type=Path, default=Path("data/yolo_pinkeggs_real_domain_v16_640x480"))
    parser.add_argument("--camera-repeat", type=int, default=4)
    parser.add_argument("--base-positive", type=int, default=400)
    parser.add_argument("--base-negative", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")

    manifest = list(csv.DictReader((args.camera / "manifest.csv").open(newline="", encoding="utf-8")))
    counts = {"camera_train": 0, "camera_val": 0, "base_positive": 0, "base_negative": 0, "test": 0}
    for row in manifest:
        split = row["split"]
        if split not in {"train", "val"}:
            continue
        source_image = args.camera / "images" / split / f'{row["stem"]}.jpg'
        source_label = args.camera / "labels" / split / f'{row["stem"]}.txt'
        repeats = args.camera_repeat if split == "train" else 1
        for repeat in range(repeats):
            stem = f'realcam_r{repeat:02d}_{row["stem"]}'
            add_pair(source_image, source_label, args.output, split, stem)
            counts[f"camera_{split}"] += 1

    rng = random.Random(args.seed)
    positives, negatives = [], []
    for image in (args.base / "images" / "train").iterdir():
        if not image.is_file() or image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if image.stem.startswith("camera_"):
            continue
        label = args.base / "labels" / "train" / f"{image.stem}.txt"
        target = positives if label.exists() and label.stat().st_size > 0 else negatives
        target.append((image, label))
    rng.shuffle(positives)
    rng.shuffle(negatives)
    for kind, pool, limit in (
        ("base_positive", positives, args.base_positive),
        ("base_negative", negatives, args.base_negative),
    ):
        for image, label in pool[:limit]:
            add_pair(image, label, args.output, "train", f"base_{image.stem}")
            counts[kind] += 1

    for image in (args.base / "images" / "test").iterdir():
        if not image.is_file() or image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        label = args.base / "labels" / "test" / f"{image.stem}.txt"
        add_pair(image, label, args.output, "test", image.stem)
        counts["test"] += 1

    yaml_path = args.output / "pinkeggs_real_domain.yaml"
    yaml_path.write_text(
        f"path: {args.output.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: eggs\n",
        encoding="ascii",
    )
    print(counts)
    print(yaml_path)


if __name__ == "__main__":
    main()
