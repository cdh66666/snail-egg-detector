"""Build a conservative v29 fine-tuning set.

Keep the v26 train/val/test split intact and add only a deterministic,
bounded sample of field-style hard negatives to the training split. The
evaluation holdouts are never copied into training.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
from pathlib import Path

import cv2


EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def digest(path: Path) -> str | None:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        return None
    return hashlib.sha1(image.tobytes()).hexdigest()


def copy_base(base: Path, out: Path) -> set[str]:
    seen: set[str] = set()
    for split in ("train", "val", "test"):
        for image in sorted((base / "images" / split).iterdir()):
            if image.suffix.lower() not in EXTS:
                continue
            key = digest(image)
            if key is None or key in seen:
                continue
            seen.add(key)
            target = out / "images" / split / image.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, target)
            label = base / "labels" / split / f"{image.stem}.txt"
            target_label = out / "labels" / split / f"{image.stem}.txt"
            target_label.parent.mkdir(parents=True, exist_ok=True)
            if label.exists():
                shutil.copy2(label, target_label)
            else:
                target_label.write_text("", encoding="ascii")
    return seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--hardneg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260720)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")

    seen = copy_base(args.base, args.output)
    hardneg_images = args.hardneg / "images" / "train"
    if not hardneg_images.exists():
        hardneg_images = args.hardneg / "images"
    candidates = [
        p for p in sorted(hardneg_images.iterdir())
        if p.suffix.lower() in EXTS
    ]
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    selected = 0
    for image in candidates:
        if selected >= args.limit:
            break
        key = digest(image)
        if key is None or key in seen:
            continue
        seen.add(key)
        stem = f"targeted_neg_{selected:04d}_{image.stem}"
        target = args.output / "images" / "train" / f"{stem}{image.suffix.lower()}"
        target_label = args.output / "labels" / "train" / f"{stem}.txt"
        shutil.copy2(image, target)
        target_label.write_text("", encoding="ascii")
        selected += 1

    yaml_path = args.output / "pinkeggs_union_v29.yaml"
    yaml_path.write_text(
        f"path: {args.output.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: snail_eggs\n",
        encoding="ascii",
    )
    print({"base": str(args.base), "hardneg": str(args.hardneg), "selected_hardneg": selected})
    print(yaml_path)


if __name__ == "__main__":
    main()
