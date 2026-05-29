from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "dist" / "maixcam_snail_eggs_yolo"


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise SystemExit(f"Missing file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a MaixCam YOLO deployment package.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--calibration-images", type=int, default=200)
    parser.add_argument("--model-name", default="snail_eggs_yolov8n_640x480")
    parser.add_argument("--onnx", type=Path, default=ROOT / "models" / "snail_eggs_yolov8n_640x480.onnx")
    parser.add_argument("--mud", type=Path, default=ROOT / "maixcam" / "snail_eggs_yolov8n_640x480.mud")
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--input-height", type=int, default=480)
    args = parser.parse_args()

    out = args.out.resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    copy_file(args.onnx.resolve(), out / "convert" / f"{args.model_name}.onnx")
    copy_file(ROOT / "scripts" / "maixcam_convert_snail_eggs_yolov8.sh", out / "convert" / "maixcam_convert_snail_eggs_yolov8.sh")
    copy_file(args.mud.resolve(), out / "device" / f"{args.model_name}.mud")
    copy_file(ROOT / "maixcam" / "main.py", out / "device" / "main.py")
    copy_file(ROOT / "README.md", out / "README_DEPLOY.md")

    env_file = out / "convert" / "convert.env"
    env_file.write_text(
        "\n".join(
            [
                f"export NET_NAME={args.model_name}",
                f"export INPUT_W={args.input_width}",
                f"export INPUT_H={args.input_height}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    image_dir = ROOT / "data" / "yolo_pinkeggs_hardneg_v2" / "images" / "train"
    images = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    if not images:
        raise SystemExit(f"No calibration images found in {image_dir}")

    cali_dir = out / "convert" / "cali_images"
    cali_dir.mkdir(parents=True, exist_ok=True)
    selected = images[: args.calibration_images]
    for img in selected:
        shutil.copy2(img, cali_dir / img.name)
    shutil.copy2(selected[0], out / "convert" / "image.jpg")

    print(f"Prepared: {out}")
    print(f"Calibration images: {len(selected)}")
    print("Next: copy the convert folder into the TPU-MLIR Docker shared directory and run the .sh script.")


if __name__ == "__main__":
    main()
