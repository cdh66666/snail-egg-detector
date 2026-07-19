from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from safety_filter import pass_laser_safe_filter  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def args():
    p = argparse.ArgumentParser(description="1000-frame realistic 320x224 YOLO11 stress test.")
    p.add_argument("--model", type=Path, default=ROOT / "runs/detect/runs_yolo/snail_eggs_yolo11n_robust_v6_mined/weights/best.pt")
    p.add_argument("--dataset", type=Path, default=ROOT / "data/yolo_pinkeggs_multi_v14_mined_640x480")
    p.add_argument("--frames-per-condition", type=int, default=100)
    p.add_argument("--conf", type=float, default=0.20)
    p.add_argument("--safe-filter", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="0")
    p.add_argument("--report", type=Path, default=ROOT / "outputs/pc_realworld_1000frame_report.json")
    p.add_argument("--preview", type=Path, default=ROOT / "outputs/pc_realworld_1000frame_preview.mp4")
    return p.parse_args()


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(1e-9, aa + ab - inter)


def read_boxes(label_path: Path):
    boxes = []
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) >= 5:
                boxes.append(tuple(float(v) for v in p[1:5]))
    return boxes


def source_pool(dataset: Path):
    images = dataset / "images/test"
    labels = dataset / "labels/test"
    positives, multis, negatives = [], [], []
    for path in sorted(images.iterdir()):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        boxes = read_boxes(labels / f"{path.stem}.txt")
        row = (path, boxes)
        if not boxes:
            negatives.append(row)
        elif len(boxes) >= 2:
            multis.append(row)
        else:
            positives.append(row)
    return positives, multis, negatives


def normalized_to_pixels(box, w, h):
    cx, cy, bw, bh = box
    return np.array([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h], dtype=np.float32)


def warp_box(box, matrix, w, h):
    x1, y1, x2, y2 = normalized_to_pixels(box, w, h)
    corners = np.float32([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]])
    transformed = cv2.perspectiveTransform(corners, matrix).reshape(-1, 2)
    min_xy, max_xy = transformed.min(axis=0), transformed.max(axis=0)
    return [float(max(0, min_xy[0])), float(max(0, min_xy[1])), float(min(319, max_xy[0])), float(min(223, max_xy[1]))]


def make_frame(row, condition, rng):
    path, boxes = row
    source = cv2.imread(str(path))
    if source is None:
        raise RuntimeError(path)
    h, w = source.shape[:2]
    # Camera aspect is 320x224. Use a random crop/zoom, then a mild projective warp.
    target_ratio = 320 / 224
    crop_h = min(h, max(32, int(w / target_ratio)))
    crop_w = min(w, max(32, int(crop_h * target_ratio)))
    zoom = rng.uniform(0.82, 1.10)
    crop_w = min(w, max(32, int(crop_w / zoom)))
    crop_h = min(h, max(32, int(crop_w / target_ratio)))
    left = rng.randint(0, max(0, w - crop_w))
    top = rng.randint(0, max(0, h - crop_h))
    crop = source[top : top + crop_h, left : left + crop_w]
    cropped_boxes = []
    for cx, cy, bw, bh in boxes:
        px = normalized_to_pixels((cx, cy, bw, bh), w, h)
        px[[0, 2]] -= left; px[[1, 3]] -= top
        cropped_boxes.append((px[0] / crop_w, px[1] / crop_h, px[2] / crop_w, px[3] / crop_h))
    frame = cv2.resize(crop, (320, 224), interpolation=cv2.INTER_AREA if zoom <= 1 else cv2.INTER_CUBIC)
    sx, sy = 320 / crop_w, 224 / crop_h
    gt = [[b[0] * crop_w * sx, b[1] * crop_h * sy, b[2] * crop_w * sx, b[3] * crop_h * sy] for b in cropped_boxes]

    angle = rng.uniform(-5, 5) if condition in {"angle", "motion"} else rng.uniform(-1.5, 1.5)
    dx, dy = rng.uniform(-5, 5), rng.uniform(-4, 4)
    rot = cv2.getRotationMatrix2D((160, 112), angle, 1.0)
    rot[:, 2] += (dx, dy)
    frame = cv2.warpAffine(frame, rot, (320, 224), borderMode=cv2.BORDER_REFLECT)
    affine = np.vstack([rot, [0, 0, 1]]).astype(np.float32)
    gt = [warp_box(((b[0] + b[2]) / 640, (b[1] + b[3]) / 448, (b[2] - b[0]) / 320, (b[3] - b[1]) / 224), affine, 320, 224) for b in gt]

    if condition == "dark":
        frame = cv2.convertScaleAbs(frame, alpha=rng.uniform(.38, .65), beta=rng.uniform(-8, 4))
    elif condition == "bright":
        frame = cv2.convertScaleAbs(frame, alpha=rng.uniform(1.15, 1.55), beta=rng.uniform(12, 35))
    elif condition == "warm":
        frame = frame.astype(np.float32); frame[:, :, 2] *= rng.uniform(1.05, 1.28); frame[:, :, 0] *= rng.uniform(.70, .95); frame = np.clip(frame, 0, 255).astype(np.uint8)
    elif condition == "cool":
        frame = frame.astype(np.float32); frame[:, :, 0] *= rng.uniform(1.05, 1.25); frame[:, :, 2] *= rng.uniform(.72, .95); frame = np.clip(frame, 0, 255).astype(np.uint8)
    elif condition == "motion":
        k = rng.choice([3, 5, 7]); kernel = np.zeros((k, k)); kernel[k // 2, :] = 1 / k; frame = cv2.filter2D(frame, -1, kernel)
    elif condition == "noise":
        np_rng = np.random.default_rng(rng.randrange(0, 2**32))
        noise = np_rng.normal(0, rng.uniform(3, 12), frame.shape).astype(np.float32); frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        if rng.random() < .5:
            encode_quality = rng.randint(35, 75); ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, encode_quality]); frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else frame
    if condition == "occlusion" and gt:
        # Small edge occlusion, leaving the main egg mass visible in most cases.
        x = rng.randint(0, 270); y = rng.randint(0, 180); frame[y:y+rng.randint(5, 16), x:x+rng.randint(5, 22)] = (80, 80, 80)
    return frame, [b for b in gt if b[2] - b[0] >= 2 and b[3] - b[1] >= 2]


def match(pred, gt):
    used, tp = set(), 0
    for p in sorted(pred, key=lambda x: x[4], reverse=True):
        choices = [(iou(p, g), i) for i, g in enumerate(gt) if i not in used]
        if choices:
            score, index = max(choices)
            if score >= .5:
                used.add(index); tp += 1
    return tp, len(pred) - tp, len(gt) - tp


def main():
    a = args(); rng = random.Random(20260719); pos, multi, neg = source_pool(a.dataset)
    pools = {"normal": pos, "dark": pos, "bright": pos, "warm": pos, "cool": pos, "motion": multi or pos, "noise": pos, "angle": multi or pos, "occlusion": pos, "negative": neg}
    conditions = list(pools)
    model = YOLO(str(a.model), task="detect")
    preview_writer = cv2.VideoWriter(str(a.preview), cv2.VideoWriter_fourcc(*"mp4v"), 20, (320, 224))
    report = {"model_input": "320x224", "frames": 0, "conf": a.conf, "conditions": {}, "source_split": "test only; no training images"}
    longest_dropout = 0
    for condition in conditions:
        stats = {"frames": 0, "gt": 0, "tp": 0, "fp": 0, "fn": 0, "max_consecutive_zero_detection": 0}
        run = 0
        for index in range(a.frames_per_condition):
            pool = pools[condition]
            if not pool: continue
            frame, gt = make_frame(rng.choice(pool), condition, rng)
            result = model.predict(frame, imgsz=(224, 320), conf=a.conf, iou=.45, device=a.device, verbose=False)[0]
            pred = []
            if result.boxes is not None:
                for box, score in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                    x1, y1, x2, y2 = map(float, box)
                    ok = True
                    if a.safe_filter:
                        ok, _ = pass_laser_safe_filter(
                            frame,
                            round(x1), round(y1), round(x2), round(y2),
                            float(score),
                            min_conf=a.conf,
                            min_pink_ratio=0.035,
                            max_red_bad_ratio=0.55,
                            red_bad_dominance=2.4,
                            strong_conf=0.35,
                        )
                    if ok: pred.append([x1, y1, x2, y2, float(score)])
            tp, fp, fn = match(pred, gt); stats["frames"] += 1; stats["gt"] += len(gt); stats["tp"] += tp; stats["fp"] += fp; stats["fn"] += fn
            if gt and tp == 0: run += 1; stats["max_consecutive_zero_detection"] = max(stats["max_consecutive_zero_detection"], run)
            else: run = 0
            if index < 10:
                preview = frame.copy()
                for p in pred: cv2.rectangle(preview, (round(p[0]), round(p[1])), (round(p[2]), round(p[3])), (0, 220, 70), 1)
                for i, g in enumerate(gt):
                    if i >= tp: cv2.rectangle(preview, (round(g[0]), round(g[1])), (round(g[2]), round(g[3])), (0, 0, 255), 1)
                cv2.putText(preview, condition, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, .45, (255,255,255), 1)
                preview_writer.write(preview)
        stats["recall"] = stats["tp"] / max(1, stats["tp"] + stats["fn"]); stats["precision"] = stats["tp"] / max(1, stats["tp"] + stats["fp"]); report["conditions"][condition] = stats; report["frames"] += stats["frames"]
        longest_dropout = max(longest_dropout, stats["max_consecutive_zero_detection"])
    preview_writer.release(); report["safe_filter"] = a.safe_filter; report["recall"] = sum(v["tp"] for v in report["conditions"].values()) / max(1, sum(v["gt"] for v in report["conditions"].values())); report["precision"] = sum(v["tp"] for v in report["conditions"].values()) / max(1, sum(v["tp"]+v["fp"] for v in report["conditions"].values())); report["longest_zero_detection_frames"] = longest_dropout
    a.report.parent.mkdir(parents=True, exist_ok=True); a.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"); print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
