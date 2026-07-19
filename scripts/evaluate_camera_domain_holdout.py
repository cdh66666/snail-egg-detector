from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from safety_filter import pass_laser_safe_filter  # noqa: E402

THRESHOLDS = (0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.35)


def iou(a: list[float], b: list[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(1e-9, area_a + area_b - intersection)


def load_boxes(label_path: Path, width: int, height: int) -> list[list[float]]:
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 5:
            continue
        cx, cy, bw, bh = map(float, fields[1:5])
        boxes.append(
            [
                (cx - bw / 2) * width,
                (cy - bh / 2) * height,
                (cx + bw / 2) * width,
                (cy + bh / 2) * height,
            ]
        )
    return boxes


def source_group(stem: str) -> str:
    return re.sub(r"_v\d+$", "", stem)


def raw_predictions(result) -> list[list[float]]:
    if result.boxes is None:
        return []
    return [
        [*(float(value) for value in box), float(score)]
        for box, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())
    ]


def filter_predictions(
    frame,
    predictions: list[list[float]],
    threshold: float,
    strong_conf: float,
    min_pink_ratio: float,
) -> list[list[float]]:
    accepted = []
    for prediction in predictions:
        x1, y1, x2, y2, score = prediction
        if score < threshold:
            continue
        passed, _ = pass_laser_safe_filter(
            frame,
            round(x1),
            round(y1),
            round(x2),
            round(y2),
            score,
            min_conf=threshold,
            min_pink_ratio=min_pink_ratio,
            max_red_bad_ratio=0.55,
            red_bad_dominance=2.4,
            strong_conf=strong_conf,
        )
        if passed:
            accepted.append(prediction)
    return accepted


def match(predictions: list[list[float]], truth: list[list[float]]) -> tuple[int, int, int]:
    used = set()
    true_positives = 0
    for prediction in sorted(predictions, key=lambda row: row[4], reverse=True):
        choices = [(iou(prediction, target), index) for index, target in enumerate(truth) if index not in used]
        if choices:
            overlap, index = max(choices)
            if overlap >= 0.5:
                used.add(index)
                true_positives += 1
    return true_positives, len(predictions) - true_positives, len(truth) - true_positives


def group_summary(group_hits: dict[str, list[bool]]) -> dict[str, float | int]:
    groups = list(group_hits.values())
    return {
        "source_groups": len(groups),
        "groups_with_any_detection": sum(any(values) for values in groups),
        "any_variant_detection_rate": sum(any(values) for values in groups) / max(1, len(groups)),
        "groups_with_all_variants_detected": sum(all(values) for values in groups),
        "all_variants_detection_rate": sum(all(values) for values in groups) / max(1, len(groups)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a detector on labeled MaixCAM-domain captures.")
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "runs/detect/runs_yolo/snail_eggs_yolo11n_robust_v6_mined/weights/best.pt",
    )
    parser.add_argument("--camera", type=Path, default=ROOT / "runs/camera_domain_v10")
    parser.add_argument("--device", default="0")
    parser.add_argument("--report", type=Path, default=ROOT / "outputs/camera_domain_holdout_report.json")
    parser.add_argument("--min-pink-ratio", type=float, default=0.035)
    parser.add_argument("--strong-conf", type=float, default=0.35)
    parser.add_argument("--thresholds", type=float, nargs="+", default=THRESHOLDS)
    args = parser.parse_args()

    manifest_path = args.camera / "manifest.csv"
    manifest = list(csv.DictReader(manifest_path.open(newline="", encoding="utf-8")))
    selected = [row for row in manifest if row["split"] in ("val", "holdout")]
    thresholds = sorted(set(float(value) for value in args.thresholds))
    inference_floor = min(thresholds)
    model = YOLO(str(args.model), task="detect")

    samples = []
    missing_or_corrupt = []
    for row in selected:
        split = row["split"]
        image_path = args.camera / "images" / split / f'{row["stem"]}.jpg'
        frame = cv2.imread(str(image_path))
        if frame is None:
            missing_or_corrupt.append(str(image_path))
            continue
        truth = [] if split == "holdout" else load_boxes(
            args.camera / "labels" / split / f'{row["stem"]}.txt', frame.shape[1], frame.shape[0]
        )
        result = model.predict(
            frame,
            imgsz=(224, 320),
            conf=inference_floor,
            iou=0.45,
            device=args.device,
            verbose=False,
        )[0]
        samples.append({"row": row, "frame": frame, "truth": truth, "raw": raw_predictions(result)})

    report = {
        "schema_version": 2,
        "model": str(args.model.resolve()),
        "inference_input": "320x224",
        "camera_domain": str(args.camera.resolve()),
        "manifest_rows_selected": len(selected),
        "frames_evaluated": len(samples),
        "missing_or_corrupt_files": missing_or_corrupt,
        "metric_notes": {
            "annotated_box_recall": "IoU >= 0.5 against the single projected label in each positive validation frame.",
            "proxy_box_precision": "Diagnostic only. Positive captures are not exhaustively labeled, so extra detections cannot be trusted as false positives.",
            "negative_frame_false_positive_rate": "Authoritative for the labeled negative validation frames.",
            "holdout_frame_detection_rate": "Frame-level acquisition only; holdout frames do not contain box labels.",
        },
        "thresholds": {},
    }

    for threshold in thresholds:
        totals = {
            "matched_boxes": 0,
            "unmatched_predictions_on_positive_frames": 0,
            "missed_annotated_boxes": 0,
            "positive_frames": 0,
            "positive_detected_frames": 0,
            "negative_frames": 0,
            "negative_detected_frames": 0,
            "holdout_frames": 0,
            "holdout_detected_frames": 0,
        }
        details = {"validation_misses": [], "negative_false_positives": [], "holdout_misses": []}
        positive_groups: dict[str, list[bool]] = defaultdict(list)
        holdout_groups: dict[str, list[bool]] = defaultdict(list)

        for sample in samples:
            row, frame, truth = sample["row"], sample["frame"], sample["truth"]
            predictions = filter_predictions(
                frame, sample["raw"], threshold, args.strong_conf, args.min_pink_ratio
            )
            detected = bool(predictions)
            if row["split"] == "holdout":
                totals["holdout_frames"] += 1
                totals["holdout_detected_frames"] += int(detected)
                holdout_groups[source_group(row["stem"])].append(detected)
                if not detected:
                    details["holdout_misses"].append(row["stem"])
            elif truth:
                matched, unmatched, missed = match(predictions, truth)
                totals["matched_boxes"] += matched
                totals["unmatched_predictions_on_positive_frames"] += unmatched
                totals["missed_annotated_boxes"] += missed
                totals["positive_frames"] += 1
                totals["positive_detected_frames"] += int(detected)
                positive_groups[source_group(row["stem"])].append(matched > 0)
                if missed:
                    details["validation_misses"].append(row["stem"])
            else:
                totals["negative_frames"] += 1
                totals["negative_detected_frames"] += int(detected)
                if detected:
                    details["negative_false_positives"].append(row["stem"])

        matched = totals["matched_boxes"]
        missed = totals["missed_annotated_boxes"]
        unmatched = totals["unmatched_predictions_on_positive_frames"]
        totals["annotated_box_recall"] = matched / max(1, matched + missed)
        totals["proxy_box_precision"] = matched / max(1, matched + unmatched)
        totals["positive_frame_detection_rate"] = totals["positive_detected_frames"] / max(
            1, totals["positive_frames"]
        )
        totals["negative_frame_false_positive_rate"] = totals["negative_detected_frames"] / max(
            1, totals["negative_frames"]
        )
        totals["holdout_frame_detection_rate"] = totals["holdout_detected_frames"] / max(
            1, totals["holdout_frames"]
        )
        totals["positive_source_groups"] = group_summary(positive_groups)
        totals["holdout_source_groups"] = group_summary(holdout_groups)
        totals["details"] = details
        report["thresholds"][str(threshold)] = totals

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
