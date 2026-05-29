from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass
class EggCrop:
    image_path: Path
    crop_xyxy: tuple[int, int, int, int]
    egg_xyxy_in_crop: tuple[int, int, int, int]


def read_yolo_boxes(label_path: Path, width: int, height: int) -> list[tuple[int, int, int, int]]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = map(float, parts[:5])
        x1 = int(round((cx - bw / 2.0) * width))
        y1 = int(round((cy - bh / 2.0) * height))
        x2 = int(round((cx + bw / 2.0) * width))
        y2 = int(round((cy + bh / 2.0) * height))
        boxes.append((max(0, x1), max(0, y1), min(width - 1, x2), min(height - 1, y2)))
    return boxes


def to_yolo_line(box: tuple[int, int, int, int], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = max(1, x2 - x1) / width
    bh = max(1, y2 - y1) / height
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def collect_crops(root: Path, split: str, rng: random.Random) -> list[EggCrop]:
    crops: list[EggCrop] = []
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    for image_path in sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS):
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        for box in read_yolo_boxes(label_dir / f"{image_path.stem}.txt", w, h):
            x1, y1, x2, y2 = box
            bw = x2 - x1
            bh = y2 - y1
            if bw < 8 or bh < 8:
                continue
            margin = int(rng.uniform(0.10, 0.28) * max(bw, bh))
            cx1 = max(0, x1 - margin)
            cy1 = max(0, y1 - margin)
            cx2 = min(w - 1, x2 + margin)
            cy2 = min(h - 1, y2 + margin)
            crops.append(
                EggCrop(
                    image_path=image_path,
                    crop_xyxy=(cx1, cy1, cx2, cy2),
                    egg_xyxy_in_crop=(x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1),
                )
            )
    return crops


def concrete_background(width: int, height: int, rng: random.Random) -> np.ndarray:
    base = np.zeros((height, width, 3), dtype=np.uint8)
    tone = rng.randint(95, 180)
    tint = np.array(
        [tone + rng.randint(-18, 18), tone + rng.randint(-18, 18), tone + rng.randint(-18, 18)],
        dtype=np.int16,
    )
    noise = np.random.default_rng(rng.randint(0, 2**31 - 1)).normal(0, rng.uniform(12, 30), base.shape)
    base[:] = np.clip(tint + noise, 0, 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), rng.uniform(1.0, 2.8))
    for _ in range(rng.randint(4, 12)):
        x1 = rng.randint(0, width - 1)
        y1 = rng.randint(0, height - 1)
        x2 = min(width - 1, max(0, x1 + rng.randint(-width // 3, width // 3)))
        y2 = min(height - 1, max(0, y1 + rng.randint(-height // 3, height // 3)))
        color = tuple(int(max(0, min(255, tone + rng.randint(-55, 35)))) for _ in range(3))
        cv2.line(base, (x1, y1), (x2, y2), color, rng.randint(1, 3), cv2.LINE_AA)
    return base


def draw_hard_distractors(img: np.ndarray, rng: random.Random) -> None:
    h, w = img.shape[:2]
    colors = [
        (65, 65, 210),
        (85, 95, 230),
        (110, 135, 245),
        (130, 105, 220),
        (55, 120, 235),
        (100, 80, 180),
    ]
    for _ in range(rng.randint(4, 12)):
        color = colors[rng.randrange(len(colors))]
        cx = rng.randint(0, w - 1)
        cy = rng.randint(0, h - 1)
        rw = rng.randint(max(10, w // 35), max(18, w // 8))
        rh = rng.randint(max(10, h // 35), max(18, h // 8))
        shape = rng.choice(["ellipse", "rect", "line", "smooth_blob"])
        if shape == "ellipse":
            cv2.ellipse(img, (cx, cy), (rw, rh), rng.randint(0, 180), 0, 360, color, -1, cv2.LINE_AA)
        elif shape == "rect":
            cv2.rectangle(
                img,
                (max(0, cx - rw), max(0, cy - rh)),
                (min(w - 1, cx + rw), min(h - 1, cy + rh)),
                color,
                -1,
            )
        elif shape == "line":
            cv2.line(
                img,
                (max(0, cx - rw), max(0, cy - rh)),
                (min(w - 1, cx + rw), min(h - 1, cy + rh)),
                color,
                rng.randint(4, 14),
                cv2.LINE_AA,
            )
        else:
            points = []
            for i in range(rng.randint(5, 8)):
                a = 2 * np.pi * i / 7.0
                rad = rng.uniform(0.45, 1.0)
                points.append((int(cx + np.cos(a) * rw * rad), int(cy + np.sin(a) * rh * rad)))
            pts = np.array(points, dtype=np.int32)
            cv2.fillPoly(img, [pts], color, cv2.LINE_AA)


def paste_crop(canvas: np.ndarray, crop: EggCrop, rng: random.Random) -> tuple[int, int, int, int] | None:
    src = cv2.imread(str(crop.image_path))
    if src is None:
        return None
    x1, y1, x2, y2 = crop.crop_xyxy
    patch = src[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    ch, cw = patch.shape[:2]
    scale = rng.uniform(0.35, 1.10)
    nw = max(8, int(round(cw * scale)))
    nh = max(8, int(round(ch * scale)))
    H, W = canvas.shape[:2]
    if nw >= W or nh >= H:
        return None
    patch = cv2.resize(patch, (nw, nh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    px = rng.randint(0, W - nw)
    py = rng.randint(0, H - nh)

    alpha = rng.uniform(0.86, 1.0)
    roi = canvas[py : py + nh, px : px + nw]
    cv2.addWeighted(patch, alpha, roi, 1.0 - alpha, rng.uniform(-5, 5), dst=roi)

    ex1, ey1, ex2, ey2 = crop.egg_xyxy_in_crop
    sx = nw / max(1, cw)
    sy = nh / max(1, ch)
    return (
        max(0, int(round(px + ex1 * sx))),
        max(0, int(round(py + ey1 * sy))),
        min(W - 1, int(round(px + ex2 * sx))),
        min(H - 1, int(round(py + ey2 * sy))),
    )


def generate_split(root: Path, split: str, montage_count: int, negative_count: int, seed: int) -> None:
    rng = random.Random(seed)
    crops = collect_crops(root, split, rng)
    if not crops:
        raise SystemExit(f"No egg crops found for split {split}.")
    image_dir = root / "images" / split
    label_dir = root / "labels" / split

    for idx in range(montage_count):
        width, height = rng.choice([(640, 384), (640, 480), (512, 512), (768, 432)])
        canvas = concrete_background(width, height, rng)
        if rng.random() < 0.55:
            draw_hard_distractors(canvas, rng)
        labels = []
        for _ in range(rng.randint(5, 12)):
            box = paste_crop(canvas, rng.choice(crops), rng)
            if box is not None and box[2] - box[0] >= 4 and box[3] - box[1] >= 4:
                labels.append(to_yolo_line(box, width, height))
        name = f"aug_montage_{idx:04d}.jpg"
        cv2.imwrite(str(image_dir / name), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        (label_dir / f"{Path(name).stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")

    for idx in range(negative_count):
        width, height = rng.choice([(640, 384), (640, 480), (512, 512), (768, 432)])
        canvas = concrete_background(width, height, rng)
        draw_hard_distractors(canvas, rng)
        name = f"aug_hardneg_{idx:04d}.jpg"
        cv2.imwrite(str(image_dir / name), canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        (label_dir / f"{Path(name).stem}.txt").write_text("", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/yolo_pinkeggs_full_500"))
    parser.add_argument("--train-montage", type=int, default=180)
    parser.add_argument("--train-negatives", type=int, default=160)
    parser.add_argument("--val-montage", type=int, default=24)
    parser.add_argument("--val-negatives", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260528)
    args = parser.parse_args()

    generate_split(args.data_root, "train", args.train_montage, args.train_negatives, args.seed)
    generate_split(args.data_root, "val", args.val_montage, args.val_negatives, args.seed + 1)
    print(
        {
            "data_root": str(args.data_root),
            "train_montage": args.train_montage,
            "train_negatives": args.train_negatives,
            "val_montage": args.val_montage,
            "val_negatives": args.val_negatives,
        }
    )


if __name__ == "__main__":
    main()
