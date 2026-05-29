#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONVERT_DIR="$PROJECT_DIR/dist/maixcam_snail_eggs_yolo/convert"
RELEASE_DIR="$PROJECT_DIR/release/maixcam_copy_to_device"
OUT_DEVICE="$RELEASE_DIR/root/models"

export CONDA_PREFIX="$HOME/miniforge3/envs/tpu310"
export PATH="$CONDA_PREFIX/bin:$PATH"

cd "$CONVERT_DIR/workspace"

python "$CONDA_PREFIX/lib/python3.10/site-packages/tpu_mlir/python/tools/model_deploy.py" \
  --mlir snail_eggs_yolov8n_320x320.mlir \
  --quantize INT8 \
  --quant_input \
  --calibration_table snail_eggs_yolov8n_320x320_cali_table \
  --processor cv181x \
  --skip_validation \
  --model snail_eggs_yolov8n_320x320.cvimodel

mkdir -p "$OUT_DEVICE"
cp snail_eggs_yolov8n_320x320.cvimodel "$OUT_DEVICE/snail_eggs_yolov8n_320x320.cvimodel"
cp "$PROJECT_DIR/maixcam/snail_eggs_yolov8n_320x320.mud" "$OUT_DEVICE/snail_eggs_yolov8n_320x320.mud"
cp "$PROJECT_DIR/maixcam/main.py" "$RELEASE_DIR/main.py"

cd "$PROJECT_DIR/release"
rm -f maixcam_copy_to_device.zip
python - <<'PY'
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

root = Path("maixcam_copy_to_device")
with ZipFile("maixcam_copy_to_device.zip", "w", ZIP_DEFLATED) as zf:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            zf.write(path, path.relative_to(root))
PY

ls -lh "$RELEASE_DIR/main.py" "$OUT_DEVICE"
