from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import cv2
from ultralytics import YOLO

from safety_filter import pass_laser_safe_filter


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure false positives on negative-only image folders.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--negative-dir", type=Path, action="append", required=True)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.2)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int, default=0, help="0 means all images")
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--safe-filter", action="store_true")
    parser.add_argument("--exclude-yolo-root", type=Path, help="Skip source stems already present in this YOLO dataset.")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def iter_images(paths: list[Path]) -> list[Path]:
    images: list[Path] = []
    for root in paths:
        if root.is_file() and root.suffix.lower() in IMAGE_EXTS:
            images.append(root)
            continue
        if root.is_dir():
            images.extend(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    return sorted(images)


def source_stems_already_in_dataset(root: Path | None) -> set[str]:
    if root is None:
        return set()
    stems: set[str] = set()
    pattern = re.compile(r"(\d{12})")
    for split in ("train", "val", "test"):
        image_dir = root / "images" / split
        if not image_dir.exists():
            continue
        for image_path in image_dir.iterdir():
            stem = image_path.stem
            stems.add(stem)
            for match in pattern.findall(stem):
                stems.add(match)
    return stems


def main() -> None:
    args = parse_args()
    used_stems = source_stems_already_in_dataset(args.exclude_yolo_root)
    images = [p for p in iter_images(args.negative_dir) if p.stem not in used_stems]
    rng = random.Random(args.seed)
    rng.shuffle(images)
    if args.limit > 0:
        images = images[: args.limit]

    model = YOLO(str(args.model), task="detect")
    images_scanned = 0
    images_with_fp = 0
    fp_count = 0
    max_conf = 0.0
    examples = []

    for image_path in images:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        images_scanned += 1
        result = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )[0]
        kept = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            for box, score in zip(xyxy, scores):
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                if args.safe_filter:
                    ok, pink_ratio = pass_laser_safe_filter(
                        frame,
                        x1,
                        y1,
                        x2,
                        y2,
                        float(score),
                        min_conf=args.conf,
                    )
                    if not ok:
                        continue
                else:
                    pink_ratio = None
                kept.append(
                    {
                        "score": float(score),
                        "box": [x1, y1, x2, y2],
                        "pink_ratio": pink_ratio,
                    }
                )
                max_conf = max(max_conf, float(score))
        if kept:
            images_with_fp += 1
            fp_count += len(kept)
            if len(examples) < 30:
                examples.append({"image": str(image_path), "detections": kept})

    summary = {
        "model": str(args.model),
        "images_scanned": images_scanned,
        "images_with_fp": images_with_fp,
        "fp_count": fp_count,
        "image_fp_rate": images_with_fp / max(1, images_scanned),
        "detections_per_image": fp_count / max(1, images_scanned),
        "max_conf": max_conf,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "safe_filter": args.safe_filter,
        "examples": examples,
    }
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
