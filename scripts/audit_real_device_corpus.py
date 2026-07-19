from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
import re

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from safety_filter import pass_laser_safe_filter  # noqa: E402

SOURCE_DIRS = (
    "live_record_probe",
    "demo_retest_frames",
    "retest_device_yolo_overlay",
)
RAW_PREFIXES = ("once_", "manual_", "trace_raw_", "current")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit historical raw MaixCAM frames for temporal detector stability."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT
        / "runs/detect/runs_yolo/snail_eggs_yolo11n_robust_v6_mined/weights/best.pt",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--report-conf", type=float, nargs="+", default=(0.15, 0.20))
    parser.add_argument("--strong-conf", type=float, default=0.35)
    parser.add_argument(
        "--report",
        type=Path,
        default=ROOT / "outputs/real_device_corpus_audit.json",
    )
    parser.add_argument(
        "--frames-csv",
        type=Path,
        default=ROOT / "outputs/real_device_corpus_frames.csv",
    )
    parser.add_argument(
        "--contact-sheet",
        type=Path,
        default=ROOT / "outputs/real_device_corpus_contact_sheet.jpg",
    )
    parser.add_argument("--contact-count", type=int, default=96)
    return parser.parse_args()


def discover_frames():
    rows = []
    for directory_name in SOURCE_DIRS:
        directory = ROOT / "runs" / directory_name
        if not directory.exists():
            continue
        candidates = []
        for path in directory.iterdir():
            if path.suffix.lower() not in IMAGE_EXTS:
                continue
            if not path.stem.startswith(RAW_PREFIXES):
                continue
            match = re.match(r"once_(\d+)$", path.stem)
            frame_id = int(match.group(1)) if match else None
            candidates.append((frame_id, path.name, path))
        # The folders contain several capture sessions. Lexical filename order
        # is not a temporal sequence; only `once_<device-frame-id>` can be used
        # for continuity statistics, and large id gaps start a new segment.
        candidates.sort(key=lambda item: (item[0] is None, item[0] if item[0] is not None else item[1]))
        previous_id = None
        segment = 0
        for frame_id, _, path in candidates:
            if frame_id is None:
                segment += 1
            elif previous_id is not None and not (0 < frame_id - previous_id <= 250):
                segment += 1
            rows.append((directory_name, path, frame_id, segment))
            if frame_id is not None:
                previous_id = frame_id
    return rows


def image_sha1(frame):
    return hashlib.sha1(frame.tobytes()).hexdigest()


def motion_score(previous, current):
    if previous is None:
        return 0.0
    a = cv2.resize(previous, (160, 120), interpolation=cv2.INTER_AREA)
    b = cv2.resize(current, (160, 120), interpolation=cv2.INTER_AREA)
    return float(np.mean(cv2.absdiff(a, b)))


def filtered_detections(frame, result, threshold, strong_conf):
    detections = []
    if result.boxes is None:
        return detections
    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    for box, score in zip(boxes, scores):
        score = float(score)
        if score < threshold:
            continue
        x1, y1, x2, y2 = (float(value) for value in box)
        passed, reason = pass_laser_safe_filter(
            frame,
            round(x1),
            round(y1),
            round(x2),
            round(y2),
            score,
            min_conf=threshold,
            min_pink_ratio=0.035,
            max_red_bad_ratio=0.55,
            red_bad_dominance=2.4,
            strong_conf=strong_conf,
        )
        if passed:
            detections.append(
                {
                    "box": [x1, y1, x2, y2],
                    "confidence": score,
                    "filter_reason": reason,
                }
            )
    return detections


def percentile(values, q):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def temporal_metrics(rows):
    valid_pairs = 0
    detection_disagreements = 0
    dropout_after_detection = 0
    center_jumps = []
    by_source = {}
    for source in SOURCE_DIRS:
        source_rows = [row for row in rows if row["source"] == source and row.get("frame_id") is not None]
        if not source_rows:
            continue
        source_rows.sort(key=lambda row: (row["segment"], row["frame_id"]))
        source_pairs = 0
        source_disagreements = 0
        source_dropouts = 0
        for previous, current in zip(source_rows, source_rows[1:]):
            if previous["segment"] != current["segment"] or current["motion"] > 25.0:
                continue
            source_pairs += 1
            valid_pairs += 1
            previous_detected = previous["count"] > 0
            current_detected = current["count"] > 0
            if previous_detected != current_detected:
                source_disagreements += 1
                detection_disagreements += 1
            if previous_detected and not current_detected:
                source_dropouts += 1
                dropout_after_detection += 1
            if previous_detected and current_detected:
                a, b = previous["primary_center"], current["primary_center"]
                if a is not None and b is not None:
                    diagonal = max(1.0, math.hypot(current["width"], current["height"]))
                    center_jumps.append(math.hypot(b[0] - a[0], b[1] - a[1]) / diagonal)

        by_source[source] = {
            "continuous_pairs": source_pairs,
            "detection_disagreement_rate": source_disagreements / max(1, source_pairs),
            "dropout_after_detection_rate": source_dropouts / max(1, source_pairs),
        }
    return {
        "metric_definition": "Adjacent once_<frame-id> pairs in the same source segment with image motion <= 25; this is a continuity proxy, not labeled recall.",
        "continuous_pairs": valid_pairs,
        "detection_disagreement_rate": detection_disagreements / max(1, valid_pairs),
        "dropout_after_detection_rate": dropout_after_detection / max(1, valid_pairs),
        "center_jump_p50_frame_diagonal": percentile(center_jumps, 50),
        "center_jump_p95_frame_diagonal": percentile(center_jumps, 95),
        "by_source": by_source,
    }


def draw_contact_sheet(samples, output):
    if not samples:
        return
    tile_w, tile_h = 320, 224
    columns = 8
    rows = math.ceil(len(samples) / columns)
    sheet = np.full((rows * tile_h, columns * tile_w, 3), 28, dtype=np.uint8)
    for index, sample in enumerate(samples):
        frame = cv2.imread(sample["path"])
        if frame is None:
            continue
        frame = cv2.resize(frame, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
        sx, sy = tile_w / sample["width"], tile_h / sample["height"]
        for detection in sample["detections"]:
            x1, y1, x2, y2 = detection["box"]
            cv2.rectangle(
                frame,
                (round(x1 * sx), round(y1 * sy)),
                (round(x2 * sx), round(y2 * sy)),
                (0, 220, 60),
                2,
            )
        label = f'{sample["source"][:12]} n={sample["count"]} m={sample["motion"]:.1f}'
        cv2.rectangle(frame, (0, 0), (tile_w, 20), (0, 0, 0), -1)
        cv2.putText(frame, label, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
        row, column = divmod(index, columns)
        sheet[row * tile_h : (row + 1) * tile_h, column * tile_w : (column + 1) * tile_w] = frame
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    args = parse_args()
    sources = discover_frames()
    if not sources:
        raise SystemExit("No historical raw MaixCAM frames found.")

    model = YOLO(str(args.model), task="detect")
    thresholds = sorted(set(float(value) for value in args.report_conf))
    rows_by_threshold = {threshold: [] for threshold in thresholds}
    previous_by_source = {}
    hashes = Counter()
    inference_ms = []

    for index, (source, path, frame_id, segment) in enumerate(sources, start=1):
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        height, width = frame.shape[:2]
        hashes[image_sha1(frame)] += 1
        previous = previous_by_source.get(source)
        motion = motion_score(previous["frame"], frame) if previous and previous["segment"] == segment else 0.0
        previous_by_source[source] = {"frame": frame, "segment": segment}

        started = time.perf_counter()
        result = model.predict(
            frame,
            imgsz=(224, 320),
            conf=args.conf,
            iou=0.45,
            device=args.device,
            verbose=False,
        )[0]
        inference_ms.append((time.perf_counter() - started) * 1000.0)

        for threshold in thresholds:
            detections = filtered_detections(frame, result, threshold, args.strong_conf)
            detections.sort(key=lambda item: item["confidence"], reverse=True)
            primary = detections[0] if detections else None
            primary_center = None
            primary_area = None
            if primary is not None:
                x1, y1, x2, y2 = primary["box"]
                primary_center = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]
                primary_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            rows_by_threshold[threshold].append(
                {
                    "source": source,
                    "path": str(path),
                    "frame_id": frame_id,
                    "segment": segment,
                    "width": width,
                    "height": height,
                    "motion": motion,
                    "count": len(detections),
                    "max_confidence": detections[0]["confidence"] if detections else 0.0,
                    "primary_center": primary_center,
                    "primary_area": primary_area,
                    "detections": detections,
                }
            )
        if index % 200 == 0:
            print(f"processed {index}/{len(sources)}")

    report = {
        "scope": "historical raw MaixCAM captures; overlay-prefixed files excluded",
        "limitations": (
            "Most captures show a monitor rather than an outdoor field scene. Detection ratios are response-rate proxies, not recall. Temporal metrics use only contiguous once_<frame-id> segments and are not labeled tracking accuracy."
        ),
        "model": str(args.model),
        "model_input": "320x224",
        "source_frames": len(sources),
        "exact_unique_frames": len(hashes),
        "exact_duplicate_frames": sum(value - 1 for value in hashes.values()),
        "inference_ms": {
            "mean": float(np.mean(inference_ms)),
            "p50": percentile(inference_ms, 50),
            "p95": percentile(inference_ms, 95),
        },
        "thresholds": {},
    }

    for threshold, rows in rows_by_threshold.items():
        confidences = [row["max_confidence"] for row in rows if row["count"]]
        moving = [row for row in rows if row["motion"] >= 4.0]
        still = [row for row in rows if row["motion"] < 4.0]
        report["thresholds"][str(threshold)] = {
            "frames": len(rows),
            "frames_with_detection": sum(row["count"] > 0 for row in rows),
            "detection_frame_ratio": sum(row["count"] > 0 for row in rows) / max(1, len(rows)),
            "detection_count_distribution": dict(Counter(row["count"] for row in rows)),
            "max_confidence_p50": percentile(confidences, 50),
            "max_confidence_p10": percentile(confidences, 10),
            "moving_frames": len(moving),
            "moving_detection_ratio": sum(row["count"] > 0 for row in moving) / max(1, len(moving)),
            "still_frames": len(still),
            "still_detection_ratio": sum(row["count"] > 0 for row in still) / max(1, len(still)),
            "temporal": temporal_metrics(rows),
        }

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    selected_threshold = thresholds[0]
    selected_rows = rows_by_threshold[selected_threshold]
    with args.frames_csv.open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ("source", "path", "width", "height", "motion", "count", "max_confidence")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selected_rows:
            writer.writerow({field: row[field] for field in fields})

    count = min(args.contact_count, len(selected_rows))
    indices = np.linspace(0, len(selected_rows) - 1, count, dtype=int)
    draw_contact_sheet([selected_rows[index] for index in indices], args.contact_sheet)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
