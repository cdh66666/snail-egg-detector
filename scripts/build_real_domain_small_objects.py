from __future__ import annotations

import argparse
import os
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def copy_dataset(source: Path, output: Path) -> None:
    for kind in ("images", "labels"):
        for split in ("train", "val", "test"):
            directory = source / kind / split
            if not directory.exists():
                continue
            for path in directory.iterdir():
                if path.is_file() and path.suffix.lower() != ".cache":
                    link_or_copy(path, output / kind / split / path.name)


def boxes(label_path: Path, width: int, height: int):
    result = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        cx, cy, bw, bh = map(float, fields[1:5])
        result.append(
            (
                max(0, round((cx - bw / 2) * width)),
                max(0, round((cy - bh / 2) * height)),
                min(width, round((cx + bw / 2) * width)),
                min(height, round((cy + bh / 2) * height)),
            )
        )
    return result


def feather_mask(height: int, width: int) -> np.ndarray:
    mask = np.ones((height, width), dtype=np.float32)
    edge = max(1, min(5, min(height, width) // 6))
    ramp = np.linspace(0.15, 1.0, edge, dtype=np.float32)
    mask[:edge] *= ramp[:, None]
    mask[-edge:] *= ramp[::-1, None]
    mask[:, :edge] *= ramp[None, :]
    mask[:, -edge:] *= ramp[None, ::-1]
    return mask[:, :, None]


def visible_pink_ratio(crop: np.ndarray) -> float:
    blue, green, red = cv2.split(crop.astype(np.float32))
    chroma = np.maximum(np.maximum(red, green), blue) - np.minimum(np.minimum(red, green), blue)
    pink = (red > green * 1.06) & (red > blue * 0.92) & (chroma > 14) & (red > 55)
    return float(np.mean(pink))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate real-camera small-object composites.")
    parser.add_argument("--source", type=Path, default=Path("data/yolo_pinkeggs_real_domain_v16_640x480"))
    parser.add_argument("--camera", type=Path, default=Path("runs/camera_domain_v10"))
    parser.add_argument("--output", type=Path, default=Path("data/yolo_pinkeggs_real_domain_v17_small_640x480"))
    parser.add_argument("--count", type=int, default=1400)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    copy_dataset(args.source, args.output)

    patches = []
    backgrounds = []
    image_dir = args.camera / "images" / "train"
    label_dir = args.camera / "labels" / "train"
    for image_path in image_dir.glob("*.jpg"):
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        found = boxes(label_dir / f"{image_path.stem}.txt", width, height)
        if not found:
            backgrounds.append(frame)
            continue
        for x1, y1, x2, y2 in found:
            margin = max(2, round(max(x2 - x1, y2 - y1) * 0.08))
            crop = frame[max(0, y1 - margin) : min(height, y2 + margin), max(0, x1 - margin) : min(width, x2 + margin)]
            if crop.size and visible_pink_ratio(crop) >= 0.12:
                patches.append(crop)
    if not patches or not backgrounds:
        raise SystemExit("camera patches or backgrounds are empty")

    rng = random.Random(args.seed)
    image_output = args.output / "images" / "train"
    label_output = args.output / "labels" / "train"
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)
    for index in range(args.count):
        frame = rng.choice(backgrounds).copy()
        height, width = frame.shape[:2]
        labels = []
        target_count = rng.choices((1, 2, 3), weights=(0.65, 0.25, 0.10), k=1)[0]
        for _ in range(target_count):
            patch = rng.choice(patches).copy()
            target_width = rng.randint(16, 64)
            scale = target_width / max(1, patch.shape[1])
            target_height = max(8, round(patch.shape[0] * scale))
            if target_height >= height - 8:
                continue
            patch = cv2.resize(patch, (target_width, target_height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
            patch = cv2.convertScaleAbs(patch, alpha=rng.uniform(0.65, 1.30), beta=rng.uniform(-12, 12))
            if rng.random() < 0.35:
                kernel = rng.choice((3, 5))
                patch = cv2.GaussianBlur(patch, (kernel, kernel), rng.uniform(0.3, 1.2))
            x = rng.randint(4, width - target_width - 4)
            y = rng.randint(12, height - target_height - 4)
            region = frame[y : y + target_height, x : x + target_width].astype(np.float32)
            alpha = feather_mask(target_height, target_width)
            blended = patch.astype(np.float32) * alpha + region * (1.0 - alpha)
            frame[y : y + target_height, x : x + target_width] = np.clip(blended, 0, 255).astype(np.uint8)
            labels.append((x, y, x + target_width, y + target_height))
        if rng.random() < 0.25:
            frame = cv2.convertScaleAbs(frame, alpha=rng.uniform(0.72, 1.25), beta=rng.uniform(-10, 15))
        stem = f"realcam_small_{index:05d}"
        cv2.imwrite(str(image_output / f"{stem}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(72, 95)])
        lines = []
        for x1, y1, x2, y2 in labels:
            lines.append(f"0 {(x1+x2)/(2*width):.7f} {(y1+y2)/(2*height):.7f} {(x2-x1)/width:.7f} {(y2-y1)/height:.7f}")
        (label_output / f"{stem}.txt").write_text("\n".join(lines), encoding="ascii")

    yaml_path = args.output / "pinkeggs_real_domain_small.yaml"
    yaml_path.write_text(
        f"path: {args.output.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: eggs\n",
        encoding="ascii",
    )
    print(f"patches={len(patches)} backgrounds={len(backgrounds)} generated={args.count}")
    print(yaml_path)


if __name__ == "__main__":
    main()
