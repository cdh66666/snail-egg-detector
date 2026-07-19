from __future__ import annotations

import argparse
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
    link_or_copy(image, output / "images" / split / f"{stem}.jpg")
    link_or_copy(label, output / "labels" / split / f"{stem}.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine independent MaixCAM captures without val-source leakage.")
    parser.add_argument("--camera-a", type=Path, default=Path("runs/camera_domain_v10"))
    parser.add_argument("--camera-b", type=Path, default=Path("runs/camera_domain_v9"))
    parser.add_argument("--base", type=Path, default=Path("data/yolo_pinkeggs_multi_v14_mined_640x480"))
    parser.add_argument("--output", type=Path, default=Path("data/yolo_pinkeggs_dual_capture_v20_640x480"))
    parser.add_argument("--capture-repeat", type=int, default=2)
    parser.add_argument("--base-positive", type=int, default=400)
    parser.add_argument("--base-negative", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")

    counts = {"capture_train": 0, "val": 0, "base_positive": 0, "base_negative": 0, "test": 0}
    # Only the train split of both physical capture sessions is used for adaptation.
    # camera-b val/holdout remain independent by source identity and physical capture.
    for capture_name, camera in (("a", args.camera_a), ("b", args.camera_b)):
        for image in sorted((camera / "images" / "train").glob("*.jpg")):
            label = camera / "labels" / "train" / f"{image.stem}.txt"
            for repeat in range(args.capture_repeat):
                add_pair(image, label, args.output, "train", f"cam{capture_name}_r{repeat:02d}_{image.stem}")
                counts["capture_train"] += 1

    # Keep v9 validation separate by source identity and capture.
    for image in sorted((args.camera_b / "images" / "val").glob("*.jpg")):
        label = args.camera_b / "labels" / "val" / f"{image.stem}.txt"
        add_pair(image, label, args.output, "val", f"v9val_{image.stem}")
        counts["val"] += 1

    rng = random.Random(args.seed)
    positives, negatives = [], []
    for image in (args.base / "images" / "train").iterdir():
        if not image.is_file() or image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if image.stem.startswith("camera_"):
            continue
        label = args.base / "labels" / "train" / f"{image.stem}.txt"
        (positives if label.exists() and label.stat().st_size > 0 else negatives).append((image, label))
    rng.shuffle(positives); rng.shuffle(negatives)
    for kind, pool, limit in (("base_positive", positives, args.base_positive), ("base_negative", negatives, args.base_negative)):
        for image, label in pool[:limit]:
            add_pair(image, label, args.output, "train", f"base_{image.stem}")
            counts[kind] += 1

    for image in sorted((args.base / "images" / "test").iterdir()):
        if not image.is_file() or image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        add_pair(image, args.base / "labels" / "test" / f"{image.stem}.txt", args.output, "test", image.stem)
        counts["test"] += 1

    yaml = args.output / "pinkeggs_dual_capture.yaml"
    yaml.write_text(
        f"path: {args.output.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: eggs\n",
        encoding="ascii",
    )
    print(counts)
    print(yaml)


if __name__ == "__main__":
    main()
