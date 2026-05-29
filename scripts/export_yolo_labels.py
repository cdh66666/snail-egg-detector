from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def convert_box(box: list[int], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2.0) / width
    cy = ((y1 + y2) / 2.0) / height
    bw = abs(x2 - x1) / width
    bh = abs(y2 - y1) / height
    return f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/pinkeggs_100/annotations.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/yolo_pinkeggs_100"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    base_dir = args.dataset.parent

    if args.clean and args.output_dir.exists():
        shutil.rmtree(args.output_dir)

    items = list(dataset["images"])
    rng = random.Random(args.seed)
    rng.shuffle(items)
    n = len(items)
    train_end = int(n * args.train_ratio)
    val_end = train_end + int(n * args.val_ratio)
    splits = {
        "train": items[:train_end],
        "val": items[train_end:val_end],
        "test": items[val_end:],
    }

    for split, split_items in splits.items():
        images_dir = args.output_dir / "images" / split
        labels_dir = args.output_dir / "labels" / split
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for item in split_items:
            source = base_dir / item["file"]
            target = images_dir / Path(item["file"]).name
            shutil.copy2(source, target)
            lines = [convert_box(box, item["width"], item["height"]) for box in item["boxes"]]
            (labels_dir / f"{target.stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    yaml = args.output_dir / "pinkeggs.yaml"
    yaml.write_text(
        (
            f"path: {args.output_dir.resolve().as_posix()}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "names:\n"
            "  0: eggs\n"
        ),
        encoding="utf-8",
    )
    print(
        {
            "images": len(dataset["images"]),
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
            "output": str(args.output_dir),
            "yaml": str(yaml),
        }
    )


if __name__ == "__main__":
    main()
