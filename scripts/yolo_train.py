from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and test a YOLO model for apple snail eggs.")
    parser.add_argument("--data", type=Path, default=Path("data/yolo_pinkeggs_full_960/pinkeggs.yaml"))
    parser.add_argument("--base-model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--project", default="runs_yolo")
    parser.add_argument("--name", default="pinkeggs_yolov8n_320")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--export-onnx", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.base_model, task="detect")
    train_result = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        cache=False,
        verbose=True,
    )

    best_path = Path(train_result.save_dir) / "weights" / "best.pt"
    best_model = YOLO(str(best_path), task="detect")
    test_metrics = best_model.val(
        data=str(args.data),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=f"{args.name}_test",
        plots=True,
    )

    exported = None
    if args.export_onnx:
        exported = best_model.export(format="onnx", imgsz=args.imgsz, dynamic=False, simplify=True, opset=17)

    summary = {
        "best_model": str(best_path),
        "train_dir": str(train_result.save_dir),
        "test_map50": float(test_metrics.box.map50),
        "test_map50_95": float(test_metrics.box.map),
        "exported": exported,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
