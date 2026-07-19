from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from safety_filter import pass_laser_safe_filter  # noqa: E402


@dataclass
class Sample:
    image_path: Path
    boxes: list[tuple[float, float, float, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and evaluate a PC YOLO11 stress demo.")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT
        / "runs/detect/runs_yolo/snail_eggs_yolo11n_robust_v6_mined/weights/best.pt",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "data/yolo_pinkeggs_multi_v14_mined_640x480",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/pc_yolo11_v6_stress_demo.mp4")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/pc_yolo11_v6_stress_report.json")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--seconds-per-scene", type=float, default=5.0)
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def read_sample(image_path: Path, label_path: Path) -> Sample:
    boxes: list[tuple[float, float, float, float]] = []
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 5:
                boxes.append(tuple(float(v) for v in parts[1:5]))
    return Sample(image_path=image_path, boxes=boxes)


def load_split(dataset: Path) -> tuple[list[Sample], list[Sample]]:
    images = dataset / "images" / "test"
    labels = dataset / "labels" / "test"
    positives: list[Sample] = []
    negatives: list[Sample] = []
    for image_path in sorted(images.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        sample = read_sample(image_path, labels / f"{image_path.stem}.txt")
        (positives if sample.boxes else negatives).append(sample)
    return positives, negatives


def letterbox_tile(sample: Sample, width: int, height: int) -> tuple[np.ndarray, list[list[float]]]:
    image = cv2.imread(str(sample.image_path))
    if image is None:
        raise RuntimeError(f"cannot read {sample.image_path}")
    src_h, src_w = image.shape[:2]
    scale = min(width / src_w, height / src_h)
    resized_w = max(1, round(src_w * scale))
    resized_h = max(1, round(src_h * scale))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    tile = np.full((height, width, 3), 114, dtype=np.uint8)
    ox = (width - resized_w) // 2
    oy = (height - resized_h) // 2
    tile[oy : oy + resized_h, ox : ox + resized_w] = resized

    boxes: list[list[float]] = []
    for cx, cy, bw, bh in sample.boxes:
        x1 = ox + (cx - bw / 2) * src_w * scale
        y1 = oy + (cy - bh / 2) * src_h * scale
        x2 = ox + (cx + bw / 2) * src_w * scale
        y2 = oy + (cy + bh / 2) * src_h * scale
        boxes.append([x1, y1, x2, y2])
    return tile, boxes


def make_canvas(samples: list[Sample], frame_index: int, fps: int) -> tuple[np.ndarray, list[list[float]]]:
    width, height = 640, 448
    if len(samples) == 1:
        layout = [(0, 0, width, height)]
    else:
        layout = [(0, 0, 320, 224), (320, 0, 320, 224), (0, 224, 320, 224), (320, 224, 320, 224)]
    canvas = np.full((height, width, 3), 114, dtype=np.uint8)
    ground_truth: list[list[float]] = []
    for sample, (x, y, w, h) in zip(samples, layout):
        tile, boxes = letterbox_tile(sample, w, h)
        canvas[y : y + h, x : x + w] = tile
        for box in boxes:
            ground_truth.append([box[0] + x, box[1] + y, box[2] + x, box[3] + y])

    t = frame_index / max(1, fps)
    gain = 0.72 + 0.30 * math.sin(t * 1.17) + 0.10 * math.sin(t * 0.37)
    bias = 10.0 * math.sin(t * 0.73)
    canvas = cv2.convertScaleAbs(canvas, alpha=max(0.45, min(1.18, gain)), beta=bias)
    blur_phase = frame_index % 48
    if 16 <= blur_phase <= 22:
        canvas = cv2.GaussianBlur(canvas, (3, 3), 0.8)
    return canvas, ground_truth


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(1e-9, area_a + area_b - inter)


def match_boxes(predictions: list[list[float]], ground_truth: list[list[float]]) -> tuple[int, int, list[int]]:
    pairs: list[tuple[float, int, int]] = []
    for pi, pred in enumerate(predictions):
        for gi, gt in enumerate(ground_truth):
            pairs.append((iou(pred, gt), pi, gi))
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    for overlap, pi, gi in sorted(pairs, reverse=True):
        if overlap < 0.5:
            break
        if pi not in matched_pred and gi not in matched_gt:
            matched_pred.add(pi)
            matched_gt.add(gi)
    missed = [index for index in range(len(ground_truth)) if index not in matched_gt]
    return len(matched_gt), len(predictions) - len(matched_pred), missed


def predict(model: YOLO, frame: np.ndarray, conf: float, device: str) -> list[list[float]]:
    result = model.predict(frame, imgsz=(224, 320), conf=conf, iou=0.45, device=device, verbose=False)[0]
    predictions: list[list[float]] = []
    if result.boxes is None:
        return predictions
    for box, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = [float(value) for value in box]
        accepted, _ = pass_laser_safe_filter(
            frame,
            round(x1),
            round(y1),
            round(x2),
            round(y2),
            float(score),
            min_conf=conf,
            min_pink_ratio=0.03,
        )
        if accepted:
            predictions.append([x1, y1, x2, y2, float(score)])
    return predictions


def main() -> None:
    args = parse_args()
    random.seed(20260719)
    positives, negatives = load_split(args.dataset)
    if len(positives) < 9 or len(negatives) < 4:
        raise SystemExit("test split does not contain enough positive/negative samples")
    chosen_pos = random.sample(positives, 9)
    chosen_neg = random.sample(negatives, 4)
    scenes = [
        ("single target", [chosen_pos[0]]),
        ("four targets", chosen_pos[1:5]),
        ("targets and distractors", [chosen_pos[5], chosen_neg[0], chosen_pos[6], chosen_neg[1]]),
        ("new targets and distractors", [chosen_neg[2], chosen_pos[7], chosen_neg[3], chosen_pos[8]]),
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (640, 448))
    model = YOLO(str(args.model), task="detect")
    scene_frames = max(1, round(args.seconds_per_scene * args.fps))
    totals = {"frames": 0, "gt": 0, "tp": 0, "fp": 0, "inference_ms": 0.0}
    scene_reports = []
    longest_dropout = 0
    current_dropout = 0

    for scene_index, (name, samples) in enumerate(scenes):
        scene_stats = {"name": name, "frames": 0, "gt": 0, "tp": 0, "fp": 0}
        for local_frame in range(scene_frames):
            frame, ground_truth = make_canvas(samples, scene_index * scene_frames + local_frame, args.fps)
            started = time.perf_counter()
            predictions = predict(model, frame, args.conf, args.device)
            inference_ms = (time.perf_counter() - started) * 1000.0
            tp, fp, missed = match_boxes([box[:4] for box in predictions], ground_truth)

            if ground_truth and tp == 0:
                current_dropout += 1
                longest_dropout = max(longest_dropout, current_dropout)
            else:
                current_dropout = 0

            for index, box in enumerate(predictions, start=1):
                x1, y1, x2, y2, score = box
                cv2.rectangle(frame, (round(x1), round(y1)), (round(x2), round(y2)), (0, 220, 70), 2)
                cv2.putText(frame, str(index), (round(x1), max(18, round(y1) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 70), 2)
            for index in missed:
                x1, y1, x2, y2 = ground_truth[index]
                cv2.rectangle(frame, (round(x1), round(y1)), (round(x2), round(y2)), (0, 0, 255), 2)

            recall = tp / max(1, len(ground_truth))
            cv2.rectangle(frame, (0, 0), (640, 30), (15, 15, 15), -1)
            status = f"{name}  GT {len(ground_truth)}  DET {len(predictions)}  R {recall:.0%}  FP {fp}  {inference_ms:.1f}ms"
            cv2.putText(frame, status, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (240, 240, 240), 1, cv2.LINE_AA)
            writer.write(frame)

            scene_stats["frames"] += 1
            scene_stats["gt"] += len(ground_truth)
            scene_stats["tp"] += tp
            scene_stats["fp"] += fp
            totals["frames"] += 1
            totals["gt"] += len(ground_truth)
            totals["tp"] += tp
            totals["fp"] += fp
            totals["inference_ms"] += inference_ms
        scene_stats["recall"] = scene_stats["tp"] / max(1, scene_stats["gt"])
        scene_stats["precision"] = scene_stats["tp"] / max(1, scene_stats["tp"] + scene_stats["fp"])
        scene_reports.append(scene_stats)
    writer.release()

    report = {
        "model": str(args.model.resolve()),
        "dataset": str(args.dataset.resolve()),
        "output": str(args.output.resolve()),
        "conf": args.conf,
        "fps": args.fps,
        "frames": totals["frames"],
        "ground_truth_instances": totals["gt"],
        "true_positives": totals["tp"],
        "false_positives": totals["fp"],
        "recall": totals["tp"] / max(1, totals["gt"]),
        "precision": totals["tp"] / max(1, totals["tp"] + totals["fp"]),
        "mean_inference_ms": totals["inference_ms"] / max(1, totals["frames"]),
        "longest_complete_dropout_frames": longest_dropout,
        "longest_complete_dropout_seconds": longest_dropout / args.fps,
        "scenes": scene_reports,
        "legend": {"green": "prediction", "red": "missed ground-truth box"},
    }
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
