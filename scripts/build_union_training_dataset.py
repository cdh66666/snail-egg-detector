"""Build a deduplicated union of broad v6 data and camera-domain training data."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import cv2


EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def content_hash(path: Path) -> str | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return hashlib.sha1(image.tobytes()).hexdigest()


def copy_split(sources: list[Path], output: Path, split: str) -> dict[str, int]:
    seen: set[str] = set()
    counts = {"images": 0, "positive": 0, "negative": 0, "skipped_duplicate": 0}
    for source_index, source in enumerate(sources):
        image_dir = source / "images" / split
        label_dir = source / "labels" / split
        if not image_dir.exists():
            continue
        for image in sorted(image_dir.iterdir()):
            if not image.is_file() or image.suffix.lower() not in EXTS:
                continue
            digest = content_hash(image)
            if digest is None:
                continue
            if digest in seen:
                counts["skipped_duplicate"] += 1
                continue
            seen.add(digest)
            label = label_dir / f"{image.stem}.txt"
            stem = f"s{source_index}_{image.stem}"
            destination = output / "images" / split / f"{stem}{image.suffix.lower()}"
            destination_label = output / "labels" / split / f"{stem}.txt"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination_label.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, destination)
            if label.exists():
                shutil.copy2(label, destination_label)
            else:
                destination_label.write_text("", encoding="ascii")
            counts["images"] += 1
            if destination_label.stat().st_size:
                counts["positive"] += 1
            else:
                counts["negative"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--broad", type=Path, default=Path("data/yolo_pinkeggs_multi_v14_mined_640x480"))
    parser.add_argument("--camera", type=Path, default=Path("data/yolo_pinkeggs_camera_balanced_v23_640x480"))
    parser.add_argument("--output", type=Path, default=Path("data/yolo_pinkeggs_union_v26_640x480"))
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    sources = [args.broad.resolve(), args.camera.resolve()]
    report = {split: copy_split(sources, args.output, split) for split in ("train", "val", "test")}
    yaml_path = args.output / "pinkeggs_union.yaml"
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
