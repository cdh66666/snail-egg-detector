from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def parse_imgsz(value: str) -> int | list[int]:
    parts = [part.strip() for part in str(value).replace("x", ",").split(",") if part.strip()]
    if len(parts) == 1:
        return int(parts[0])
    if len(parts) == 2:
        return [int(parts[0]), int(parts[1])]
    raise argparse.ArgumentTypeError("imgsz must be an int or HEIGHT,WIDTH, for example 640 or 480,640")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and test a YOLO model for apple snail eggs.")
    parser.add_argument("--data", type=Path, default=Path("data/yolo_pinkeggs_full_960/pinkeggs.yaml"))
    parser.add_argument("--base-model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=parse_imgsz, default=320)
    parser.add_argument("--export-imgsz", type=parse_imgsz, help="Fixed ONNX export size. Use HEIGHT,WIDTH for rectangular models.")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--project", default="runs_yolo")
    parser.add_argument("--name", default="pinkeggs_yolov8n_320")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--rect", action="store_true", help="Use rectangular batches during training.")
    parser.add_argument("--hsv-h", type=float, default=0.015, help="Hue augmentation strength.")
    parser.add_argument("--hsv-s", type=float, default=0.7, help="Saturation augmentation strength.")
    parser.add_argument("--hsv-v", type=float, default=0.4, help="Value/brightness augmentation strength.")
    parser.add_argument("--degrees", type=float, default=0.0, help="Rotation augmentation in degrees.")
    parser.add_argument("--translate", type=float, default=0.1, help="Translate augmentation fraction.")
    parser.add_argument("--scale", type=float, default=0.5, help="Scale augmentation fraction.")
    parser.add_argument("--shear", type=float, default=0.0, help="Shear augmentation in degrees.")
    parser.add_argument("--perspective", type=float, default=0.0, help="Perspective augmentation strength.")
    parser.add_argument("--flipud", type=float, default=0.0, help="Vertical flip probability.")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Horizontal flip probability.")
    parser.add_argument("--multi-scale", type=float, default=0.0, help="Random training-size range, for example 0.25.")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation probability.")
    parser.add_argument("--mixup", type=float, default=0.0, help="MixUp augmentation probability.")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Disable mosaic for final N epochs.")
    parser.add_argument("--freeze", type=int, default=0, help="Freeze the first N model layers during fine-tuning.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing an existing run directory.")
    parser.add_argument("--export-onnx", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_imgsz = args.imgsz
    # Ultralytics train/val accepts one maximum side length. Supplying [H, W]
    # is silently collapsed to max(H, W), so make that behavior explicit and
    # reserve the exact rectangular shape for fixed-size ONNX export.
    train_imgsz = max(requested_imgsz) if isinstance(requested_imgsz, list) else requested_imgsz
    export_imgsz = args.export_imgsz or requested_imgsz
    model = YOLO(args.base_model, task="detect")
    train_result = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=train_imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        weight_decay=args.weight_decay,
        project=args.project,
        name=args.name,
        patience=args.patience,
        rect=args.rect,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        shear=args.shear,
        perspective=args.perspective,
        flipud=args.flipud,
        fliplr=args.fliplr,
        multi_scale=args.multi_scale,
        mosaic=args.mosaic,
        mixup=args.mixup,
        close_mosaic=args.close_mosaic,
        freeze=args.freeze,
        cache=False,
        verbose=True,
        exist_ok=args.exist_ok,
    )

    best_path = Path(train_result.save_dir) / "weights" / "best.pt"
    best_model = YOLO(str(best_path), task="detect")
    test_metrics = best_model.val(
        data=str(args.data),
        split="test",
        imgsz=train_imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=args.project,
        name=f"{args.name}_test",
        plots=True,
        exist_ok=args.exist_ok,
    )

    exported = None
    if args.export_onnx:
        exported = best_model.export(
            format="onnx",
            imgsz=export_imgsz,
            dynamic=False,
            simplify=True,
            opset=17,
        )

    summary = {
        "best_model": str(best_path),
        "train_dir": str(train_result.save_dir),
        "requested_imgsz": requested_imgsz,
        "effective_train_imgsz": train_imgsz,
        "fixed_export_imgsz": export_imgsz if args.export_onnx else None,
        "rectangular_batches": args.rect,
        "test_map50": float(test_metrics.box.map50),
        "test_map50_95": float(test_metrics.box.map),
        "exported": exported,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
