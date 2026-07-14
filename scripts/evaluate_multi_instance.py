"""Report multi-instance recall and exact-count accuracy by target count."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

from build_multi_instance_dataset import read_boxes


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(1, union)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.35)
    parser.add_argument("--match-iou", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model = YOLO(str(args.model), task="detect")
    image_dir = args.dataset / "images" / args.split
    label_dir = args.dataset / "labels" / args.split
    stats = defaultdict(lambda: {"images": 0, "gt": 0, "matched": 0, "pred": 0, "exact": 0})
    for image_path in sorted(image_dir.glob("multi_*")):
        with Image.open(image_path) as opened:
            gt = read_boxes(label_dir / (image_path.stem + ".txt"), *opened.size)
        result = model.predict(str(image_path), imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                               device=args.device, verbose=False)[0]
        pred = result.boxes.xyxy.cpu().numpy().tolist() if result.boxes is not None else []
        pairs = sorted(
            ((iou(g, p), gi, pi) for gi, g in enumerate(gt) for pi, p in enumerate(pred)),
            reverse=True,
        )
        used_gt, used_pred = set(), set()
        for score, gi, pi in pairs:
            if score < args.match_iou:
                break
            if gi not in used_gt and pi not in used_pred:
                used_gt.add(gi)
                used_pred.add(pi)
        row = stats[len(gt)]
        row["images"] += 1
        row["gt"] += len(gt)
        row["matched"] += len(used_gt)
        row["pred"] += len(pred)
        row["exact"] += int(len(pred) == len(gt) and len(used_gt) == len(gt))

    report = {}
    total = {key: sum(row[key] for row in stats.values()) for key in ("images", "gt", "matched", "pred", "exact")}
    for count, row in sorted(stats.items()):
        report[str(count)] = row | {
            "recall": row["matched"] / max(1, row["gt"]),
            "exact_count_accuracy": row["exact"] / max(1, row["images"]),
        }
    report["all"] = total | {
        "recall": total["matched"] / max(1, total["gt"]),
        "precision": total["matched"] / max(1, total["pred"]),
        "exact_count_accuracy": total["exact"] / max(1, total["images"]),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
