from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np

from augment_yolo_snail_data import (
    IMAGE_EXTS,
    collect_crops,
    concrete_background,
    draw_hard_distractors,
    read_yolo_boxes,
    to_yolo_line,
)


def read_image(path: Path) -> np.ndarray | None:
    return cv2.imread(str(path))


def load_yolo_lines(label_path: Path, width: int, height: int) -> list[tuple[int, int, int, int]]:
    return read_yolo_boxes(label_path, width, height)


def clip_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(0, min(width - 1, int(round(x2))))
    y2 = max(0, min(height - 1, int(round(y2))))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return x1, y1, x2, y2


def write_sample(image_dir: Path, label_dir: Path, name: str, img: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> None:
    height, width = img.shape[:2]
    image_path = image_dir / f"{name}.jpg"
    label_path = label_dir / f"{name}.txt"
    cv2.imwrite(str(image_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    labels = [to_yolo_line(box, width, height) for box in boxes]
    label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")


def photometric_variant(img: np.ndarray, rng: random.Random) -> np.ndarray:
    out = img.astype(np.float32)
    contrast = rng.uniform(0.62, 1.42)
    brightness = rng.uniform(-42, 38)
    out = (out - 128.0) * contrast + 128.0 + brightness
    out = np.clip(out, 0, 255).astype(np.uint8)

    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] *= rng.uniform(0.68, 1.35)
    hsv[:, :, 2] *= rng.uniform(0.65, 1.38)
    hsv[:, :, 0] += rng.uniform(-3, 3)
    hsv[:, :, 0] = np.mod(hsv[:, :, 0], 180)
    hsv[:, :, 1:] = np.clip(hsv[:, :, 1:], 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    gamma = rng.uniform(0.62, 1.65)
    lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.uint8)
    out = cv2.LUT(out, lut)
    if rng.random() < 0.45:
        k = rng.choice([3, 5])
        out = cv2.GaussianBlur(out, (k, k), rng.uniform(0.4, 1.2))
    if rng.random() < 0.40:
        noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0, rng.uniform(3, 10), out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def distance_variant(img: np.ndarray, boxes: list[tuple[int, int, int, int]], rng: random.Random) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    height, width = img.shape[:2]
    out_w, out_h = rng.choice([(640, 480), (768, 432), (640, 384), (512, 512)])
    scale = rng.uniform(0.34, 0.82)
    resized_w = max(32, int(round(width * scale)))
    resized_h = max(32, int(round(height * scale)))
    fit = min((out_w - 4) / resized_w, (out_h - 4) / resized_h, 1.0)
    resized_w = max(32, int(round(resized_w * fit)))
    resized_h = max(32, int(round(resized_h * fit)))
    scaled = cv2.resize(img, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    canvas = concrete_background(out_w, out_h, rng)
    if rng.random() < 0.65:
        draw_hard_distractors(canvas, rng)
    ox = rng.randint(0, out_w - resized_w)
    oy = rng.randint(0, out_h - resized_h)
    canvas[oy : oy + resized_h, ox : ox + resized_w] = photometric_variant(scaled, rng)
    sx = resized_w / width
    sy = resized_h / height
    new_boxes: list[tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in boxes:
        clipped = clip_box((ox + x1 * sx, oy + y1 * sy, ox + x2 * sx, oy + y2 * sy), out_w, out_h)
        if clipped is not None:
            new_boxes.append(clipped)
    return canvas, new_boxes


def closeup_variant(img: np.ndarray, boxes: list[tuple[int, int, int, int]], rng: random.Random) -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    if not boxes:
        return None
    height, width = img.shape[:2]
    bx = rng.choice(boxes)
    x1, y1, x2, y2 = bx
    bw = x2 - x1
    bh = y2 - y1
    margin = int(rng.uniform(0.45, 1.2) * max(bw, bh))
    cx1 = max(0, x1 - margin)
    cy1 = max(0, y1 - margin)
    cx2 = min(width - 1, x2 + margin)
    cy2 = min(height - 1, y2 + margin)
    if cx2 - cx1 < 32 or cy2 - cy1 < 32:
        return None
    crop = img[cy1:cy2, cx1:cx2]
    out_w, out_h = rng.choice([(640, 480), (512, 512), (768, 432)])
    resized = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    sx = out_w / max(1, cx2 - cx1)
    sy = out_h / max(1, cy2 - cy1)
    new_boxes: list[tuple[int, int, int, int]] = []
    for bx1, by1, bx2, by2 in boxes:
        ix1, iy1 = max(bx1, cx1), max(by1, cy1)
        ix2, iy2 = min(bx2, cx2), min(by2, cy2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        visible = ((ix2 - ix1) * (iy2 - iy1)) / max(1, (bx2 - bx1) * (by2 - by1))
        if visible < 0.45:
            continue
        clipped = clip_box(((ix1 - cx1) * sx, (iy1 - cy1) * sy, (ix2 - cx1) * sx, (iy2 - cy1) * sy), out_w, out_h)
        if clipped is not None:
            new_boxes.append(clipped)
    return photometric_variant(resized, rng), new_boxes


def paste_small_crop(canvas: np.ndarray, crop, rng: random.Random) -> tuple[int, int, int, int] | None:
    src = cv2.imread(str(crop.image_path))
    if src is None:
        return None
    x1, y1, x2, y2 = crop.crop_xyxy
    patch = src[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    ch, cw = patch.shape[:2]
    scale = rng.uniform(0.16, 0.58) if rng.random() < 0.78 else rng.uniform(0.65, 1.18)
    nw = max(7, int(round(cw * scale)))
    nh = max(7, int(round(ch * scale)))
    H, W = canvas.shape[:2]
    if nw >= W or nh >= H:
        return None
    patch = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    patch = photometric_variant(patch, rng)
    px = rng.randint(0, W - nw)
    py = rng.randint(0, H - nh)
    alpha = rng.uniform(0.78, 1.0)
    roi = canvas[py : py + nh, px : px + nw]
    cv2.addWeighted(patch, alpha, roi, 1.0 - alpha, rng.uniform(-8, 8), dst=roi)

    ex1, ey1, ex2, ey2 = crop.egg_xyxy_in_crop
    sx = nw / max(1, cw)
    sy = nh / max(1, ch)
    return clip_box((px + ex1 * sx, py + ey1 * sy, px + ex2 * sx, py + ey2 * sy), W, H)


def add_field_variants(root: Path, rng: random.Random, per_positive: int, distance_count: int, close_count: int, montage_count: int, negative_count: int) -> None:
    image_dir = root / "images" / "train"
    label_dir = root / "labels" / "train"
    positives: list[tuple[Path, np.ndarray, list[tuple[int, int, int, int]]]] = []
    negatives: list[Path] = []
    for image_path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        img = read_image(image_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        boxes = load_yolo_lines(label_dir / f"{image_path.stem}.txt", w, h)
        if boxes:
            positives.append((image_path, img, boxes))
        else:
            negatives.append(image_path)

    if not positives:
        raise SystemExit("No positive train images found.")

    index = 0
    for image_path, img, boxes in positives:
        for _ in range(per_positive):
            out = photometric_variant(img, rng)
            write_sample(image_dir, label_dir, f"field_light_{index:05d}_{image_path.stem[:24]}", out, boxes)
            index += 1

    for i in range(distance_count):
        image_path, img, boxes = rng.choice(positives)
        out, new_boxes = distance_variant(img, boxes, rng)
        if new_boxes:
            write_sample(image_dir, label_dir, f"field_distance_{i:05d}_{image_path.stem[:24]}", out, new_boxes)

    made_close = 0
    attempts = 0
    while made_close < close_count and attempts < close_count * 5:
        attempts += 1
        image_path, img, boxes = rng.choice(positives)
        result = closeup_variant(img, boxes, rng)
        if result is None:
            continue
        out, new_boxes = result
        if new_boxes:
            write_sample(image_dir, label_dir, f"field_close_{made_close:05d}_{image_path.stem[:24]}", out, new_boxes)
            made_close += 1

    crops = collect_crops(root, "train", rng)
    for i in range(montage_count):
        width, height = rng.choice([(640, 480), (768, 432), (640, 384), (512, 512)])
        canvas = concrete_background(width, height, rng)
        if rng.random() < 0.72:
            draw_hard_distractors(canvas, rng)
        labels: list[tuple[int, int, int, int]] = []
        for _ in range(rng.randint(3, 10)):
            box = paste_small_crop(canvas, rng.choice(crops), rng)
            if box is not None:
                labels.append(box)
        write_sample(image_dir, label_dir, f"field_montage_{i:05d}", canvas, labels)

    for i in range(negative_count):
        width, height = rng.choice([(640, 480), (768, 432), (640, 384), (512, 512)])
        canvas = concrete_background(width, height, rng)
        draw_hard_distractors(canvas, rng)
        if negatives and rng.random() < 0.35:
            neg = read_image(rng.choice(negatives))
            if neg is not None:
                neg = cv2.resize(neg, (width, height), interpolation=cv2.INTER_AREA)
                canvas = cv2.addWeighted(canvas, 0.45, photometric_variant(neg, rng), 0.55, 0)
        write_sample(image_dir, label_dir, f"field_hardneg_{i:05d}", canvas, [])


def update_yaml(output_root: Path) -> None:
    yaml_path = output_root / "pinkeggs_hardneg.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {output_root.resolve().as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: eggs",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create train-only field robustness variants for snail egg YOLO data.")
    parser.add_argument("--base-root", type=Path, default=Path("data/yolo_pinkeggs_hardneg_v5_640x480"))
    parser.add_argument("--output-root", type=Path, default=Path("data/yolo_pinkeggs_hardneg_v6_field_640x480"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--per-positive", type=int, default=1)
    parser.add_argument("--distance-count", type=int, default=520)
    parser.add_argument("--close-count", type=int, default=220)
    parser.add_argument("--montage-count", type=int, default=520)
    parser.add_argument("--negative-count", type=int, default=260)
    args = parser.parse_args()

    if args.output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"{args.output_root} exists; pass --overwrite to replace it.")
        shutil.rmtree(args.output_root)
    shutil.copytree(args.base_root, args.output_root)
    rng = random.Random(args.seed)
    add_field_variants(
        args.output_root,
        rng,
        args.per_positive,
        args.distance_count,
        args.close_count,
        args.montage_count,
        args.negative_count,
    )
    update_yaml(args.output_root)
    counts = {}
    for split in ("train", "val", "test"):
        counts[split] = len(list((args.output_root / "images" / split).glob("*")))
    print({"output_root": str(args.output_root), "counts": counts})


if __name__ == "__main__":
    main()
