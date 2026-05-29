from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a base YOLO dataset, then mine model false positives from negative image "
            "directories and add them back as empty-label hard negatives."
        )
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--base-data-root", type=Path, required=True)
    parser.add_argument("--output-data-root", type=Path, required=True)
    parser.add_argument("--negative-dir", type=Path, action="append", required=True)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--max-source-images", type=int, default=0, help="0 means all source images")
    parser.add_argument("--max-full-images", type=int, default=1200)
    parser.add_argument("--max-crops", type=int, default=1800)
    parser.add_argument("--crop-margin", type=float, default=1.5)
    parser.add_argument("--max-full-dim", type=int, default=960)
    parser.add_argument("--clean", action="store_true")
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


def source_stems_already_in_dataset(root: Path) -> set[str]:
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


def copy_base_dataset(src: Path, dst: Path, clean: bool) -> None:
    if clean and dst.exists():
        resolved = dst.resolve()
        if len(resolved.parts) < 4 or "snail-egg-detector" not in str(resolved):
            raise SystemExit(f"Refusing to remove unexpected path: {resolved}")
        shutil.rmtree(dst)
    if dst.exists():
        raise SystemExit(f"Output already exists; pass --clean to recreate it: {dst}")
    ignore = shutil.ignore_patterns("*.cache")
    shutil.copytree(src, dst, ignore=ignore)


def resize_max_dim(img: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(1.0, max_dim / float(max(h, w)))
    if scale >= 0.999:
        return img
    return cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


def expanded_crop(img: np.ndarray, xyxy: np.ndarray, margin: float) -> np.ndarray | None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    bw = x2 - x1
    bh = y2 - y1
    if bw < 6 or bh < 6:
        return None
    pad = margin * max(bw, bh)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    nx1 = max(0, int(round(cx - bw * 0.5 - pad)))
    ny1 = max(0, int(round(cy - bh * 0.5 - pad)))
    nx2 = min(w, int(round(cx + bw * 0.5 + pad)))
    ny2 = min(h, int(round(cy + bh * 0.5 + pad)))
    if nx2 - nx1 < 32 or ny2 - ny1 < 32:
        return None
    return img[ny1:ny2, nx1:nx2]


def write_empty_label(label_dir: Path, stem: str) -> None:
    (label_dir / f"{stem}.txt").write_text("", encoding="utf-8")


def write_yaml(root: Path, base_yaml: Path | None) -> None:
    target = root / "pinkeggs_hardneg.yaml"
    if base_yaml and base_yaml.exists():
        text = base_yaml.read_text(encoding="utf-8")
        lines = []
        for line in text.splitlines():
            if line.startswith("path:"):
                lines.append(f"path: {root.resolve().as_posix()}")
            else:
                lines.append(line)
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    target.write_text(
        (
            f"path: {root.resolve().as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "names:\n"
            "  0: eggs\n"
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    copy_base_dataset(args.base_data_root, args.output_data_root, args.clean)
    write_yaml(args.output_data_root, args.base_data_root / "pinkeggs_hardneg.yaml")

    rng = random.Random(args.seed)
    used_stems = source_stems_already_in_dataset(args.base_data_root)
    source_images = [p for p in iter_images(args.negative_dir) if p.stem not in used_stems]
    rng.shuffle(source_images)
    if args.max_source_images > 0:
        source_images = source_images[: args.max_source_images]

    train_img = args.output_data_root / "images" / "train"
    train_lab = args.output_data_root / "labels" / "train"
    train_img.mkdir(parents=True, exist_ok=True)
    train_lab.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model), task="detect")
    full_count = 0
    crop_count = 0
    detection_count = 0
    scanned_count = 0
    records = []

    for image_path in source_images:
        if full_count >= args.max_full_images and crop_count >= args.max_crops:
            break
        img = cv2.imread(str(image_path))
        if img is None or min(img.shape[:2]) < 48:
            continue
        scanned_count += 1
        result = model.predict(
            img,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            continue

        xyxy = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        detection_count += len(scores)

        record = {
            "source": str(image_path),
            "scores": [float(v) for v in scores],
            "boxes": [[float(v) for v in box] for box in xyxy],
        }
        records.append(record)

        if full_count < args.max_full_images:
            out_img = resize_max_dim(img, args.max_full_dim)
            stem = f"minefp_full_{full_count:05d}"
            cv2.imwrite(str(train_img / f"{stem}.jpg"), out_img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            write_empty_label(train_lab, stem)
            full_count += 1

        order = np.argsort(-scores)
        for idx in order:
            if crop_count >= args.max_crops:
                break
            crop = expanded_crop(img, xyxy[idx], args.crop_margin)
            if crop is None:
                continue
            crop = resize_max_dim(crop, args.max_full_dim)
            stem = f"minefp_crop_{crop_count:05d}"
            cv2.imwrite(str(train_img / f"{stem}.jpg"), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            write_empty_label(train_lab, stem)
            crop_count += 1

    summary = {
        "model": str(args.model),
        "base_data_root": str(args.base_data_root),
        "output_data_root": str(args.output_data_root),
        "source_images": len(source_images),
        "scanned_images": scanned_count,
        "images_with_false_positive": len(records),
        "raw_detection_count": detection_count,
        "added_full_images": full_count,
        "added_crops": crop_count,
        "conf": args.conf,
        "imgsz": args.imgsz,
    }
    (args.output_data_root / "mined_false_positives.json").write_text(
        json.dumps({"summary": summary, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
