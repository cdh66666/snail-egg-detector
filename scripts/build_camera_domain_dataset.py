"""Build a leak-free YOLO dataset with selected MaixCam domain frames."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-positive-brightness", type=float, default=0.75)
    return parser.parse_args()


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def copy_split(source_root: Path, output_root: Path, split: str) -> int:
    count = 0
    for kind in ("images", "labels"):
        for source in (source_root / kind / split).glob("*"):
            if source.is_file():
                link_or_copy(source, output_root / kind / split / source.name)
                count += 1
    return count


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit("output already exists: %s" % args.output)
    for split in ("train", "val", "test"):
        copy_split(args.base, args.output, split)

    selected = 0
    rejected = 0
    with (args.camera / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            split = row["split"]
            if split not in ("train", "val"):
                continue
            if row["kind"] == "positive" and float(row["brightness"]) < args.min_positive_brightness:
                rejected += 1
                continue
            stem = row["stem"]
            source_image = args.camera / "images" / split / (stem + ".jpg")
            source_label = args.camera / "labels" / split / (stem + ".txt")
            prefix = "camera_%s" % stem
            link_or_copy(source_image, args.output / "images" / split / (prefix + ".jpg"))
            link_or_copy(source_label, args.output / "labels" / split / (prefix + ".txt"))
            selected += 1

    yaml_path = args.output / "pinkeggs_camera_domain.yaml"
    yaml_path.write_text(
        "path: %s\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: eggs\n"
        % args.output.resolve().as_posix(),
        encoding="ascii",
    )
    print("selected=%d rejected_dark_positive=%d yaml=%s" % (selected, rejected, yaml_path))


if __name__ == "__main__":
    main()
