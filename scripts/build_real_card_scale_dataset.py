from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

import cv2


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


def read_box(label: Path, width: int, height: int):
    for line in label.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 5:
            cx, cy, bw, bh = map(float, fields[1:5])
            return ((cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate scaled real-camera image-card examples.")
    parser.add_argument("--source", type=Path, default=Path("data/yolo_pinkeggs_dual_capture_v20_640x480"))
    parser.add_argument("--camera", type=Path, nargs="+", default=(Path("runs/camera_domain_v9"), Path("runs/camera_domain_v10")))
    parser.add_argument("--output", type=Path, default=Path("data/yolo_pinkeggs_real_card_v21_640x480"))
    parser.add_argument("--count", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    copy_dataset(args.source, args.output)

    cards = []
    backgrounds = []
    for camera in args.camera:
        for image_path in (camera / "images" / "train").glob("*.jpg"):
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            height, width = frame.shape[:2]
            box = read_box(camera / "labels" / "train" / f"{image_path.stem}.txt", width, height)
            if box is None:
                backgrounds.append(frame)
                continue
            x1, y1, x2, y2 = box
            bw, bh = x2 - x1, y2 - y1
            margin_x, margin_y = bw * 1.6, bh * 1.6
            left, top = max(0, round(x1 - margin_x)), max(0, round(y1 - margin_y))
            right, bottom = min(width, round(x2 + margin_x)), min(height, round(y2 + margin_y))
            crop = frame[top:bottom, left:right]
            if crop.size:
                cards.append((crop, (x1 - left, y1 - top, x2 - left, y2 - top)))
    if not cards or not backgrounds:
        raise SystemExit("cards or backgrounds are empty")

    rng = random.Random(args.seed)
    image_out = args.output / "images" / "train"
    label_out = args.output / "labels" / "train"
    for index in range(args.count):
        frame = rng.choice(backgrounds).copy()
        height, width = frame.shape[:2]
        crop, target = rng.choice(cards)
        target_width = target[2] - target[0]
        desired_target_width = rng.randint(12, 48)
        scale = desired_target_width / max(1.0, target_width)
        card_w = max(12, round(crop.shape[1] * scale))
        card_h = max(12, round(crop.shape[0] * scale))
        if card_w >= width - 8 or card_h >= height - 8:
            continue
        resized = cv2.resize(crop, (card_w, card_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        resized = cv2.convertScaleAbs(resized, alpha=rng.uniform(0.70, 1.25), beta=rng.uniform(-10, 10))
        if rng.random() < 0.3:
            resized = cv2.GaussianBlur(resized, (3, 3), rng.uniform(0.3, 0.9))
        x = rng.randint(4, width - card_w - 4)
        y = rng.randint(8, height - card_h - 4)
        frame[y : y + card_h, x : x + card_w] = resized
        tx1, ty1, tx2, ty2 = (value * scale for value in target)
        x1, y1, x2, y2 = x + tx1, y + ty1, x + tx2, y + ty2
        stem = f"realcard_{index:05d}"
        cv2.imwrite(str(image_out / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(75, 95)])
        (label_out / f"{stem}.txt").write_text(
            f"0 {(x1+x2)/(2*width):.7f} {(y1+y2)/(2*height):.7f} {(x2-x1)/width:.7f} {(y2-y1)/height:.7f}\n",
            encoding="ascii",
        )

    yaml = args.output / "pinkeggs_real_card.yaml"
    yaml.write_text(
        f"path: {args.output.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: eggs\n",
        encoding="ascii",
    )
    print(f"cards={len(cards)} backgrounds={len(backgrounds)} generated={args.count}")
    print(yaml)


if __name__ == "__main__":
    main()
