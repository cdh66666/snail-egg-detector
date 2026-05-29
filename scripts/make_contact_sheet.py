from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make a simple image contact sheet.")
    p.add_argument("--input-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--max-images", type=int, default=16)
    p.add_argument("--tile-w", type=int, default=420)
    p.add_argument("--tile-h", type=int, default=240)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(
        p for p in args.input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )[: args.max_images]
    if not paths:
        raise SystemExit(f"No images in {args.input_dir}")

    tiles = []
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = min(args.tile_w / w, args.tile_h / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        tile = np.full((args.tile_h, args.tile_w, 3), 245, dtype=np.uint8)
        x = (args.tile_w - nw) // 2
        y = (args.tile_h - nh) // 2
        tile[y : y + nh, x : x + nw] = resized
        cv2.putText(tile, path.stem, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
        tiles.append(tile)

    cols = max(1, args.cols)
    rows = int(np.ceil(len(tiles) / cols))
    blank = np.full((args.tile_h, args.tile_w, 3), 245, dtype=np.uint8)
    while len(tiles) < rows * cols:
        tiles.append(blank.copy())
    sheet_rows = [np.hstack(tiles[i * cols : (i + 1) * cols]) for i in range(rows)]
    sheet = np.vstack(sheet_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), sheet)
    print(args.output)


if __name__ == "__main__":
    main()
