from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
from ultralytics import YOLO

from safety_filter import pass_laser_safe_filter


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}
DEFAULT_MODEL = Path("runs/detect/runs_yolo/pinkeggs_yolov8n_320/weights/best.pt")


@dataclass
class Detection:
    id: int
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    cx: int
    cy: int


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def is_video(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTS


def default_output(source: Path, out_dir: Path | None) -> Path:
    base_dir = out_dir or source.parent
    suffix = ".jpg" if is_image(source) else ".mp4"
    return base_dir / f"{source.stem}_yolo{suffix}"


def shrink_xyxy(x1: float, y1: float, x2: float, y2: float, shrink: float) -> tuple[int, int, int, int]:
    shrink = max(0.0, min(0.45, shrink))
    if shrink <= 0:
        return round(x1), round(y1), round(x2), round(y2)
    w = x2 - x1
    h = y2 - y1
    dx = w * shrink * 0.5
    dy = h * shrink * 0.5
    return round(x1 + dx), round(y1 + dy), round(x2 - dx), round(y2 - dy)


def sort_detections(raw: list[dict], image_h: int, row_height: int) -> list[dict]:
    if row_height <= 0:
        row_height = max(18, image_h // 10)
    return sorted(raw, key=lambda d: (d["cy"] // row_height, d["cx"]))


def detections_from_result(result, names: dict[int, str], frame, args) -> list[Detection]:
    image_h = int(result.orig_shape[0])
    boxes = result.boxes
    raw: list[dict] = []
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)

    for box, conf, class_id in zip(xyxy, confs, classes):
        x1, y1, x2, y2 = shrink_xyxy(float(box[0]), float(box[1]), float(box[2]), float(box[3]), args.box_shrink)
        if x2 <= x1 or y2 <= y1:
            continue
        if args.safe_filter:
            ok, _ = pass_laser_safe_filter(
                frame,
                x1,
                y1,
                x2,
                y2,
                float(conf),
                min_conf=args.conf,
                min_pink_ratio=args.min_pink_ratio,
            )
            if not ok:
                continue
        raw.append(
            {
                "class_id": int(class_id),
                "class_name": names.get(int(class_id), str(class_id)),
                "confidence": float(conf),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": (x1 + x2) // 2,
                "cy": (y1 + y2) // 2,
            }
        )

    ordered = sort_detections(raw, image_h=image_h, row_height=args.row_height)
    return [Detection(id=i + 1, **d) for i, d in enumerate(ordered)]


def draw_cross(frame, cx: int, cy: int, color: tuple[int, int, int], size: int = 8) -> None:
    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, 2, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, 2, cv2.LINE_AA)


def draw_detections(frame, detections: list[Detection]) -> None:
    for det in detections:
        box_color = (0, 220, 80)
        text_color = (0, 0, 0)
        fill_color = (0, 220, 80)
        cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), box_color, 2, cv2.LINE_AA)
        draw_cross(frame, det.cx, det.cy, (0, 0, 255))

        label = f"{det.id}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        lx = max(0, min(det.x1, frame.shape[1] - tw - 6))
        ly = max(th + baseline + 2, det.y1 - 4)
        cv2.rectangle(frame, (lx, ly - th - baseline - 4), (lx + tw + 6, ly + 2), fill_color, -1)
        cv2.putText(frame, label, (lx + 3, ly - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2, cv2.LINE_AA)


def predict_frame(model: YOLO, frame, args) -> list[Detection]:
    results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device, verbose=False)
    return detections_from_result(results[0], model.names, frame, args)


def save_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def process_image(args, model: YOLO, output_path: Path) -> dict:
    frame = cv2.imread(str(args.source))
    if frame is None:
        raise SystemExit(f"Cannot read image: {args.source}")
    start = time.perf_counter()
    detections = predict_frame(model, frame, args)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    draw_detections(frame, detections)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)

    result = {
        "type": "image",
        "source": str(args.source),
        "output": str(output_path),
        "model": str(args.model),
        "detections": [asdict(d) for d in detections],
        "count": len(detections),
        "elapsed_ms": elapsed_ms,
    }
    if not args.no_json:
        save_json(output_path.with_suffix(".json"), result)
    return result


def process_video(args, model: YOLO, output_path: Path) -> dict:
    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise SystemExit(f"Cannot read video: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_seconds is not None:
        total = min(total, int(args.max_seconds * fps))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    jsonl_path = output_path.with_suffix(".jsonl")
    jsonl = None if args.no_json else jsonl_path.open("w", encoding="utf-8")

    frame_idx = 0
    infer_count = 0
    last_detections: list[Detection] = []
    start = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_seconds is not None and frame_idx >= total:
                break

            if frame_idx % args.video_stride == 0:
                last_detections = predict_frame(model, frame, args)
                infer_count += 1

            draw_detections(frame, last_detections)
            writer.write(frame)

            if jsonl is not None:
                jsonl.write(
                    json.dumps(
                        {
                            "frame": frame_idx,
                            "time_s": frame_idx / fps,
                            "detections": [asdict(d) for d in last_detections],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"processed {frame_idx}/{total or '?'} frames")
    finally:
        cap.release()
        writer.release()
        if jsonl is not None:
            jsonl.close()

    elapsed = time.perf_counter() - start
    return {
        "type": "video",
        "source": str(args.source),
        "output": str(output_path),
        "model": str(args.model),
        "frames": frame_idx,
        "inference_frames": infer_count,
        "elapsed_s": elapsed,
        "processed_fps": frame_idx / elapsed if elapsed > 0 else 0.0,
        "jsonl": None if args.no_json else str(jsonl_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO detector for Pomacea canaliculata egg masses.")
    parser.add_argument("source", type=Path, help="Image or video path.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/yolo_media"))
    parser.add_argument("--conf", type=float, default=0.65)
    parser.add_argument("--iou", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--video-stride", type=int, default=1, help="Infer every N frames and reuse boxes between inference frames.")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--row-height", type=int, default=0, help="Row bucket size for top-left to bottom-right IDs. 0 = auto.")
    parser.add_argument("--box-shrink", type=float, default=0.0, help="Optional visual box shrink ratio, 0.0 to 0.45.")
    parser.add_argument("--safe-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-pink-ratio", type=float, default=0.03)
    parser.add_argument("--no-json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.source = args.source.resolve()
    args.model = args.model.resolve()
    if not args.source.exists():
        raise SystemExit(f"Source not found: {args.source}")
    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    output_path = (args.output or default_output(args.source, args.out_dir)).resolve()
    model = YOLO(str(args.model), task="detect")

    if is_image(args.source):
        result = process_image(args, model, output_path)
    elif is_video(args.source):
        result = process_video(args, model, output_path)
    else:
        raise SystemExit(f"Unsupported file type: {args.source.suffix}")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
