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
    args = parser.parse_args()

    out = args.out.resolve()
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    copy_file(ROOT / "models" / "snail_eggs_yolov8n_320x320.onnx", out / "convert" / "snail_eggs_yolov8n_320x320.onnx")
    copy_file(ROOT / "scripts" / "maixcam_convert_snail_eggs_yolov8.sh", out / "convert" / "maixcam_convert_snail_eggs_yolov8.sh")
    copy_file(ROOT / "maixcam" / "snail_eggs_yolov8n_320x320.mud", out / "device" / "snail_eggs_yolov8n_320x320.mud")
    copy_file(ROOT / "maixcam" / "main.py", out / "device" / "main.py")
    copy_file(ROOT / "README.md", out / "README_DEPLOY.md")

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
