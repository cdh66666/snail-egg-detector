from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from pc_realworld_stress import make_frame, source_pool  # noqa: E402
from safety_filter import pass_laser_safe_filter  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a deterministic side-by-side detector comparison.")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/yolo_pinkeggs_multi_v14_mined_640x480")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--frames-per-condition", type=int, default=30)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/v6_vs_v13_pressure_comparison.mp4")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/v6_vs_v13_pressure_comparison.json")
    return parser.parse_args()


def overlap(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(1e-9, area_a + area_b - intersection)


def detections(model: YOLO, frame, conf: float, device: str) -> list[list[float]]:
    result = model.predict(frame, imgsz=(224, 320), conf=conf, iou=0.45, device=device, verbose=False)[0]
    accepted = []
    if result.boxes is None:
        return accepted
    for box, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = (float(value) for value in box)
        passed, _ = pass_laser_safe_filter(
            frame,
            round(x1),
            round(y1),
            round(x2),
            round(y2),
            float(score),
            min_conf=conf,
            min_pink_ratio=0.035,
            max_red_bad_ratio=0.55,
            red_bad_dominance=2.4,
            strong_conf=0.35,
        )
        if passed:
            accepted.append([x1, y1, x2, y2, float(score)])
    return accepted


def match(predictions: list[list[float]], truth: list[list[float]]) -> tuple[set[int], set[int]]:
    matched_predictions: set[int] = set()
    matched_truth: set[int] = set()
    for prediction_index in sorted(range(len(predictions)), key=lambda index: predictions[index][4], reverse=True):
        choices = [
            (overlap(predictions[prediction_index], target), truth_index)
            for truth_index, target in enumerate(truth)
            if truth_index not in matched_truth
        ]
        if choices:
            score, truth_index = max(choices)
            if score >= 0.5:
                matched_predictions.add(prediction_index)
                matched_truth.add(truth_index)
    return matched_predictions, matched_truth


def draw_panel(frame, predictions, truth, title: str, condition: str, stats: dict) -> None:
    matched_predictions, matched_truth = match(predictions, truth)
    for index, target in enumerate(truth):
        color = (255, 170, 0) if index in matched_truth else (0, 0, 255)
        cv2.rectangle(frame, (round(target[0]), round(target[1])), (round(target[2]), round(target[3])), color, 2)
    for index, prediction in enumerate(predictions):
        color = (0, 220, 70) if index in matched_predictions else (0, 180, 255)
        cv2.rectangle(frame, (round(prediction[0]), round(prediction[1])), (round(prediction[2]), round(prediction[3])), color, 2)
        cv2.putText(frame, f"{prediction[4]:.2f}", (round(prediction[0]), max(28, round(prediction[1]) - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    tp = len(matched_truth)
    fp = len(predictions) - len(matched_predictions)
    fn = len(truth) - tp
    stats["tp"] += tp
    stats["fp"] += fp
    stats["fn"] += fn
    cv2.rectangle(frame, (0, 0), (320, 42), (18, 18, 18), -1)
    cv2.putText(frame, title, (7, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(frame, f"{condition}  TP {tp}  FP {fp}  FN {fn}", (7, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)


def main() -> None:
    args = parse_args()
    rng = random.Random(20260719)
    positives, multis, negatives = source_pool(args.dataset)
    pools = {
        "normal": positives,
        "dark": positives,
        "bright": positives,
        "warm": positives,
        "cool": positives,
        "motion": multis or positives,
        "noise": positives,
        "angle": multis or positives,
        "occlusion": positives,
        "negative": negatives,
    }
    baseline = YOLO(str(args.baseline), task="detect")
    candidate = YOLO(str(args.candidate), task="detect")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (1280, 448))
    report = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "conf": args.conf,
        "frames": 0,
        "baseline_stats": {"tp": 0, "fp": 0, "fn": 0, "inference_ms": []},
        "candidate_stats": {"tp": 0, "fp": 0, "fn": 0, "inference_ms": []},
        "conditions": {},
    }

    for condition, pool in pools.items():
        condition_stats = {"baseline": {"tp": 0, "fp": 0, "fn": 0}, "candidate": {"tp": 0, "fp": 0, "fn": 0}}
        for _ in range(args.frames_per_condition):
            frame, truth = make_frame(rng.choice(pool), condition, rng)
            started = time.perf_counter()
            baseline_predictions = detections(baseline, frame, args.conf, args.device)
            report["baseline_stats"]["inference_ms"].append((time.perf_counter() - started) * 1000.0)
            started = time.perf_counter()
            candidate_predictions = detections(candidate, frame, args.conf, args.device)
            report["candidate_stats"]["inference_ms"].append((time.perf_counter() - started) * 1000.0)

            left = frame.copy()
            right = frame.copy()
            before_baseline = report["baseline_stats"].copy()
            before_candidate = report["candidate_stats"].copy()
            draw_panel(left, baseline_predictions, truth, "V6 BASELINE", condition, report["baseline_stats"])
            draw_panel(right, candidate_predictions, truth, "V13 CANDIDATE", condition, report["candidate_stats"])
            for key in ("tp", "fp", "fn"):
                condition_stats["baseline"][key] += report["baseline_stats"][key] - before_baseline[key]
                condition_stats["candidate"][key] += report["candidate_stats"][key] - before_candidate[key]
            combined = cv2.hconcat([left, right])
            writer.write(cv2.resize(combined, (1280, 448), interpolation=cv2.INTER_NEAREST))
            report["frames"] += 1
        report["conditions"][condition] = condition_stats

    writer.release()
    for key in ("baseline_stats", "candidate_stats"):
        stats = report[key]
        times = stats.pop("inference_ms")
        stats["recall"] = stats["tp"] / max(1, stats["tp"] + stats["fn"])
        stats["precision"] = stats["tp"] / max(1, stats["tp"] + stats["fp"])
        stats["mean_inference_ms"] = sum(times) / max(1, len(times))
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(args.output.resolve())


if __name__ == "__main__":
    main()
