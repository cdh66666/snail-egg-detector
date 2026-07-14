"""Add leak-free, multi-instance small-object scenes to a YOLO dataset.

Positive patches and backgrounds are sampled only from the requested split.
The generated labels retain every pasted egg mass, which directly targets the
multi-object failure mode seen when several small masses share one camera frame.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_boxes(path: Path, width: int, height: int) -> list[tuple[int, int, int, int]]:
    boxes = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        _, cx, cy, bw, bh = map(float, fields[:5])
        x1 = max(0, round((cx - bw / 2) * width))
        y1 = max(0, round((cy - bh / 2) * height))
        x2 = min(width, round((cx + bw / 2) * width))
        y2 = min(height, round((cy + bh / 2) * height))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def yolo_line(box: tuple[int, int, int, int], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    return "0 %.6f %.6f %.6f %.6f" % (
        (x1 + x2) / (2 * width),
        (y1 + y2) / (2 * height),
        (x2 - x1) / width,
        (y2 - y1) / height,
    )


def collect(root: Path, split: str):
    positives = []
    negatives = []
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    for path in sorted(image_dir.iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        try:
            with Image.open(path) as opened:
                size = opened.size
        except Exception:
            continue
        boxes = read_boxes(label_dir / (path.stem + ".txt"), *size)
        if boxes:
            positives.append((path, boxes))
        else:
            negatives.append(path)
    return positives, negatives


def intersect_ratio(a, b) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / max(1, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))


def extract_patch(path: Path, boxes, chosen, rng: random.Random):
    image = Image.open(path).convert("RGB")
    x1, y1, x2, y2 = chosen
    bw, bh = x2 - x1, y2 - y1
    # Retain substantial natural context. These patches become photo panels,
    # matching the monitor/collage domain instead of floating cut-out objects.
    context = rng.uniform(1.5, 2.8)
    px = round(bw * context)
    py = round(bh * context)
    left, top = max(0, x1 - px), max(0, y1 - py)
    right, bottom = min(image.width, x2 + px), min(image.height, y2 + py)
    patch = image.crop((left, top, right, bottom))
    mapped = []
    for bx1, by1, bx2, by2 in boxes:
        ix1, iy1 = max(left, bx1), max(top, by1)
        ix2, iy2 = min(right, bx2), min(bottom, by2)
        visible = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area = max(1, (bx2 - bx1) * (by2 - by1))
        if visible / area >= 0.70:
            mapped.append((ix1 - left, iy1 - top, ix2 - left, iy2 - top))
    chosen_local = (x1 - left, y1 - top, x2 - left, y2 - top)
    return patch, mapped, chosen_local


def make_background(path: Path, size: tuple[int, int], rng: random.Random) -> Image.Image:
    image = Image.open(path).convert("RGB")
    target_ratio = size[0] / size[1]
    source_ratio = image.width / image.height
    if source_ratio > target_ratio:
        crop_w = round(image.height * target_ratio)
        left = rng.randint(0, max(0, image.width - crop_w))
        image = image.crop((left, 0, left + crop_w, image.height))
    else:
        crop_h = round(image.width / target_ratio)
        top = rng.randint(0, max(0, image.height - crop_h))
        image = image.crop((0, top, image.width, top + crop_h))
    return image.resize(size, Image.Resampling.LANCZOS)


def paste_scene(positives, negatives, count: int, size, rng: random.Random):
    background_path = rng.choice(negatives) if negatives else rng.choice(positives)[0]
    canvas = make_background(background_path, size, rng)
    labels = []
    panels = []
    attempts = 0
    # Deliberately cover tiny through medium targets. The first bin is the
    # failure region observed when several PPT images share the camera frame.
    target_long_sides = [rng.randint(18, 30), rng.randint(28, 48), rng.randint(42, 82)]
    while len(labels) < count and attempts < count * 30:
        attempts += 1
        source_path, source_boxes = rng.choice(positives)
        patch, local_boxes, chosen_local = extract_patch(
            source_path, source_boxes, rng.choice(source_boxes), rng
        )
        target_long = rng.choice(target_long_sides)
        scale = target_long / max(chosen_local[2] - chosen_local[0], chosen_local[3] - chosen_local[1])
        out_w = max(8, round(patch.width * scale))
        out_h = max(8, round(patch.height * scale))
        if out_w >= size[0] or out_h >= size[1]:
            continue
        patch = patch.resize((out_w, out_h), Image.Resampling.LANCZOS)
        patch = ImageEnhance.Brightness(patch).enhance(rng.uniform(0.55, 1.35))
        patch = ImageEnhance.Contrast(patch).enhance(rng.uniform(0.75, 1.25))
        patch = ImageEnhance.Color(patch).enhance(rng.uniform(0.70, 1.30))
        mapped_boxes = []
        for box in local_boxes:
            mapped = tuple(round(v * scale) for v in box)
            if max(mapped[2] - mapped[0], mapped[3] - mapped[1]) >= 14:
                mapped_boxes.append(mapped)
        x = rng.randint(0, size[0] - out_w)
        y = rng.randint(0, size[1] - out_h)
        placed_boxes = [(x + a, y + b, x + c, y + d) for a, b, c, d in mapped_boxes]
        panel = (x, y, x + out_w, y + out_h)
        if any(intersect_ratio(box, existing) > 0.08 for box in placed_boxes for existing in labels):
            continue
        if any(intersect_ratio(panel, existing) > 0.10 for existing in panels):
            continue
        canvas.paste(patch, (x, y))
        labels.extend(placed_boxes)
        panels.append(panel)
    return canvas, labels


def build_split(base: Path, output: Path, split: str, scenes: int, seed: int, size):
    positives, negatives = collect(base, split)
    if not positives:
        raise RuntimeError(f"No positive samples in {split}")
    rng = random.Random(seed)
    image_dir = output / "images" / split
    label_dir = output / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    histogram = {}
    for index in range(scenes):
        count = rng.randint(2, 6)
        image, boxes = paste_scene(positives, negatives, count, size, rng)
        if len(boxes) < 2:
            continue
        name = f"multi_{split}_{index:05d}"
        image.save(image_dir / f"{name}.jpg", quality=92)
        (label_dir / f"{name}.txt").write_text(
            "\n".join(yolo_line(box, *size) for box in boxes) + "\n", encoding="utf-8"
        )
        histogram[str(len(boxes))] = histogram.get(str(len(boxes)), 0) + 1
    return {"positive_sources": len(positives), "negative_sources": len(negatives), "counts": histogram}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--source", type=Path, help="Clean split-safe source for composites; defaults to --base.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-scenes", type=int, default=1600)
    parser.add_argument("--val-scenes", type=int, default=240)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    source = args.source or args.base
    if args.clean and args.output.exists():
        shutil.rmtree(args.output)
    shutil.copytree(args.base, args.output, dirs_exist_ok=True)
    size = (args.width, args.height)
    summary = {
        "train": build_split(source, args.output, "train", args.train_scenes, args.seed, size),
        "val": build_split(source, args.output, "val", args.val_scenes, args.seed + 1, size),
    }
    yaml_path = args.output / "pinkeggs_multi.yaml"
    yaml_path.write_text(
        "path: %s\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: snail_eggs\n"
        % args.output.resolve().as_posix(), encoding="utf-8"
    )
    (args.output / "multi_instance_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
