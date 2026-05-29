from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from safety_filter import pass_laser_safe_filter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create annotated result collages.")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--images-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--prefix", default="test_samples")
    p.add_argument("--count", type=int, default=36)
    p.add_argument("--per-sheet", type=int, default=12)
    p.add_argument("--cols", type=int, default=4)
    p.add_argument("--tile-w", type=int, default=420)
    p.add_argument("--tile-h", type=int, default=300)
    p.add_argument("--conf", type=float, default=0.65)
    p.add_argument("--iou", type=float, default=0.35)
    p.add_argument("--imgsz", type=int, default=320)
    return p.parse_args()


def draw(frame, detections):
    for idx, (x1, y1, x2, y2, score) in enumerate(detections, start=1):
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 80), 2, cv2.LINE_AA)
        cv2.line(frame, (cx - 8, cy), (cx + 8, cy), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - 8), (cx, cy + 8), (0, 0, 255), 2, cv2.LINE_AA)
        label = f"{idx} {score:.2f}"
        cv2.rectangle(frame, (x1, max(0, y1 - 24)), (x1 + 84, max(24, y1)), (0, 220, 80), -1)
        cv2.putText(frame, label, (x1 + 4, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)


def annotate_image(model, path: Path, args) -> np.ndarray:
    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(path)
    res = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, verbose=False)[0]
    detections = []
    if res.boxes is not None:
        for box, conf in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = map(int, np.round(box))
            ok, _ = pass_laser_safe_filter(frame, x1, y1, x2, y2, float(conf), min_conf=args.conf)
            if ok:
                detections.append((x1, y1, x2, y2, float(conf)))
    detections.sort(key=lambda d: (((d[1] + d[3]) // 2) // max(24, frame.shape[0] // 8), (d[0] + d[2]) // 2))
    draw(frame, detections)
    cv2.putText(frame, f"{path.stem}  count={len(detections)}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, f"{path.stem}  count={len(detections)}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def to_tile(img, w, h):
    ih, iw = img.shape[:2]
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    tile = np.full((h, w, 3), 245, dtype=np.uint8)
    x = (w - nw) // 2
    y = (h - nh) // 2
    tile[y : y + nh, x : x + nw] = resized
    return tile


def save_sheet(images, output, args):
    cols = args.cols
    rows = int(np.ceil(len(images) / cols))
    blank = np.full((args.tile_h, args.tile_w, 3), 245, dtype=np.uint8)
    tiles = [to_tile(img, args.tile_w, args.tile_h) for img in images]
    while len(tiles) < rows * cols:
        tiles.append(blank.copy())
    sheet_rows = [np.hstack(tiles[i * cols : (i + 1) * cols]) for i in range(rows)]
    sheet = np.vstack(sheet_rows)
    cv2.imwrite(str(output), sheet)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model), task="detect")
    paths = sorted(p for p in args.images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if len(paths) > args.count:
        idxs = np.linspace(0, len(paths) - 1, args.count).round().astype(int)
        paths = [paths[i] for i in idxs]
    annotated = [annotate_image(model, p, args) for p in paths]
    for sheet_idx, start in enumerate(range(0, len(annotated), args.per_sheet), start=1):
        save_sheet(
            annotated[start : start + args.per_sheet],
            args.output_dir / f"{args.prefix}_{sheet_idx}.jpg",
            args,
        )
    print(args.output_dir)


if __name__ == "__main__":
    main()
