from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from safety_filter import pass_laser_safe_filter  # noqa: E402


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    ab = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / max(1e-9, aa + ab - inter)


def labels(path: Path):
    result = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) >= 5:
                result.append(tuple(map(float, p[1:5])))
    return result


def gt_pixels(norm, w, h):
    cx, cy, bw, bh = norm
    return [
        (cx - bw / 2) * w,
        (cy - bh / 2) * h,
        (cx + bw / 2) * w,
        (cy + bh / 2) * h,
    ]


def choose(dataset: Path):
    images = dataset / "images/test"
    label_dir = dataset / "labels/test"
    pos, multi, neg = [], [], []
    for path in sorted(images.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        gt = labels(label_dir / f"{path.stem}.txt")
        row = (path, gt)
        if not gt:
            neg.append(row)
        elif len(gt) >= 2:
            multi.append(row)
        else:
            pos.append(row)
    rng = random.Random(20260719)
    rng.shuffle(pos); rng.shuffle(multi); rng.shuffle(neg)
    small = sorted(pos + multi, key=lambda row: min((b[2] * b[3] for b in row[1]), default=1.0))
    return {
        "single target": pos[:4],
        "multiple targets": multi[:4],
        "small or difficult targets": small[:4],
        "difficult negatives": neg[:4],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "runs/detect/runs_yolo/snail_eggs_yolo11n_robust_v6_mined/weights/best.pt")
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/yolo_pinkeggs_multi_v14_mined_640x480")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/pc_native_gallery.mp4")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/pc_native_gallery_report.json")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    scenes = choose(args.dataset)
    model = YOLO(str(args.model), task="detect")
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (640, 480))
    report = {"conf": args.conf, "model_input": "320x224", "display_output": "640x480", "fps": args.fps, "scenes": {}}
    for name, samples in scenes.items():
        stats = {"frames": 0, "gt": 0, "tp": 0, "fp": 0, "files": [p.name for p, _ in samples]}
        for path, normalized_gt in samples:
            raw = cv2.imread(str(path))
            if raw is None:
                continue
            frame = cv2.resize(raw, (640, 480), interpolation=cv2.INTER_AREA)
            gt = [gt_pixels(box, 640, 480) for box in normalized_gt]
            # Match the MaixCAM MUD input exactly: width 320, height 224.
            result = model.predict(frame, imgsz=(224, 320), conf=args.conf, iou=0.45, device=0, verbose=False)[0]
            pred = []
            if result.boxes is not None:
                for box, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                    x1, y1, x2, y2 = map(float, box)
                    ok, _ = pass_laser_safe_filter(frame, round(x1), round(y1), round(x2), round(y2), float(score), min_conf=args.conf)
                    if ok:
                        pred.append([x1, y1, x2, y2, float(score)])
            matched = set(); tp = 0
            for box in sorted(pred, key=lambda x: x[4], reverse=True):
                scores = [(iou(box, target), index) for index, target in enumerate(gt) if index not in matched]
                if scores and max(scores)[0] >= 0.5:
                    matched.add(max(scores)[1]); tp += 1
            fp = len(pred) - tp
            for index, box in enumerate(pred, 1):
                x1, y1, x2, y2, score = box
                cv2.rectangle(frame, (round(x1), round(y1)), (round(x2), round(y2)), (0, 220, 70), 2)
                cv2.putText(frame, str(index), (round(x1), max(18, round(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 220, 70), 2)
            for index, box in enumerate(gt):
                if index not in matched:
                    cv2.rectangle(frame, (round(box[0]), round(box[1])), (round(box[2]), round(box[3])), (0, 0, 255), 2)
            cv2.rectangle(frame, (0, 0), (640, 30), (15, 15, 15), -1)
            cv2.putText(frame, f"{name}  GT {len(gt)}  DET {len(pred)}  TP {tp}  FP {fp}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, .52, (240, 240, 240), 1, cv2.LINE_AA)
            for _ in range(max(1, round(args.fps / 3))):
                writer.write(frame)
            stats["frames"] += max(1, round(args.fps / 3)); stats["gt"] += len(gt) * max(1, round(args.fps / 3)); stats["tp"] += tp * max(1, round(args.fps / 3)); stats["fp"] += fp * max(1, round(args.fps / 3))
        stats["recall"] = stats["tp"] / max(1, stats["gt"])
        stats["precision"] = stats["tp"] / max(1, stats["tp"] + stats["fp"])
        report["scenes"][name] = stats
    writer.release()
    # Re-scan the output summary from accumulated frame-weighted counts.
    # The per-image rows are intentionally preserved in the video and filenames.
    # Re-run a lightweight summary by reading the generated scenes is unnecessary.
    # The report records the selected source files for reproducibility.
    report["legend"] = {"green": "prediction", "red": "missed ground-truth box"}
    args.out.parent.mkdir(parents=True, exist_ok=True); args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
