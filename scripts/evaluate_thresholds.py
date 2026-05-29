from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from safety_filter import pass_laser_safe_filter


def parse_imgsz(value: str) -> int | list[int]:
    parts = [part.strip() for part in str(value).replace("x", ",").split(",") if part.strip()]
    if len(parts) == 1:
        return int(parts[0])
    if len(parts) == 2:
        return [int(parts[0]), int(parts[1])]
    raise argparse.ArgumentTypeError("imgsz must be an int or HEIGHT,WIDTH, for example 640 or 480,640")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate detection thresholds on a YOLO-format split.")
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("data/yolo_pinkeggs_full_960"))
    p.add_argument("--split", default="test")
    p.add_argument("--imgsz", type=parse_imgsz, default=320)
    p.add_argument("--iou-match", type=float, default=0.5)
    p.add_argument("--confs", default="0.35,0.50,0.60,0.65,0.70,0.75,0.80")
    p.add_argument("--safe-filter", action="store_true")
    p.add_argument("--output", type=Path)
    return p.parse_args()


def load_labels(label_path: Path, width: int, height: int) -> list[tuple[float, float, float, float]]:
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = map(float, parts[:5])
        x1 = (cx - bw / 2.0) * width
        y1 = (cy - bh / 2.0) * height
        x2 = (cx + bw / 2.0) * width
        y2 = (cy + bh / 2.0) * height
        boxes.append((x1, y1, x2, y2))
    return boxes


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-9, area_a + area_b - inter)


def score_image(gt_boxes, pred_boxes, iou_match: float) -> tuple[int, int, int]:
    matched = set()
    tp = 0
    fp = 0
    for pred in sorted(pred_boxes, key=lambda p: p[4], reverse=True):
        best_iou = 0.0
        best_idx = -1
        for idx, gt in enumerate(gt_boxes):
            if idx in matched:
                continue
            val = iou(pred[:4], gt)
            if val > best_iou:
                best_iou = val
                best_idx = idx
        if best_iou >= iou_match:
            tp += 1
            matched.add(best_idx)
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched)
    return tp, fp, fn


def main() -> None:
    args = parse_args()
    confs = [float(x) for x in args.confs.split(",") if x.strip()]
    image_dir = args.data_root / "images" / args.split
    label_dir = args.data_root / "labels" / args.split
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    model = YOLO(str(args.model), task="detect")
    rows = []

    all_preds = {}
    for image_path in images:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        result = model.predict(frame, imgsz=args.imgsz, conf=min(confs), iou=0.45, verbose=False)[0]
        preds = []
        if result.boxes is not None:
            xyxy = result.boxes.xyxy.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            for box, score in zip(xyxy, scores):
                x1, y1, x2, y2 = map(int, np.round(box))
                if args.safe_filter:
                    ok, _ = pass_laser_safe_filter(frame, x1, y1, x2, y2, float(score), min_conf=min(confs))
                    if not ok:
                        continue
                preds.append((float(x1), float(y1), float(x2), float(y2), float(score)))
        all_preds[image_path] = preds

    for conf in confs:
        total_tp = total_fp = total_fn = 0
        for image_path, preds in all_preds.items():
            frame = cv2.imread(str(image_path))
            h, w = frame.shape[:2]
            gt = load_labels(label_dir / f"{image_path.stem}.txt", w, h)
            selected = [p for p in preds if p[4] >= conf]
            tp, fp, fn = score_image(gt, selected, args.iou_match)
            total_tp += tp
            total_fp += fp
            total_fn += fn
        precision = total_tp / max(1, total_tp + total_fp)
        recall = total_tp / max(1, total_tp + total_fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        rows.append(
            {
                "conf": conf,
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    text = json.dumps(rows, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
