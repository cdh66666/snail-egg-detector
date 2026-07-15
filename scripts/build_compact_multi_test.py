"""Build test-only compact multi-target panels from the untouched test split."""

import argparse
import random
from pathlib import Path

from PIL import Image

from build_multi_instance_dataset import read_boxes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    samples = []
    for path in sorted((args.dataset / "images/test").iterdir()):
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            continue
        boxes = read_boxes(args.dataset / "labels/test" / (path.stem + ".txt"), *image.size)
        for box in boxes:
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            pad_x, pad_y = round(w * 0.35), round(h * 0.35)
            crop = image.crop((max(0, x1 - pad_x), max(0, y1 - pad_y),
                               min(image.width, x2 + pad_x), min(image.height, y2 + pad_y)))
            pixels = list(crop.resize((32, 32), Image.Resampling.BILINEAR).getdata())
            pink = sum(1 for r, g, b in pixels if r > 120 and r > g * 1.05 and r > b * 0.90)
            if pink < 20:
                continue
            samples.append(crop)
    rng = random.Random(20260715)
    rng.shuffle(samples)
    index = 0
    for count in (2, 3, 4, 5):
        for variant in range(3):
            canvas = Image.new("RGB", (420, 315), (205, 205, 200))
            cols = 2 if count <= 4 else 3
            rows = (count + cols - 1) // cols
            cell_w, cell_h = 420 // cols, 315 // rows
            for item in range(count):
                crop = samples[index % len(samples)].copy()
                index += 1
                scale = min((cell_w - 12) / crop.width, (cell_h - 12) / crop.height)
                crop = crop.resize(
                    (max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                    Image.Resampling.LANCZOS,
                )
                x = (item % cols) * cell_w + (cell_w - crop.width) // 2
                y = (item // cols) * cell_h + (cell_h - crop.height) // 2
                canvas.paste(crop, (x, y))
            canvas.save(args.output / f"compact_{count}_{variant}.jpg", quality=94)


if __name__ == "__main__":
    main()
