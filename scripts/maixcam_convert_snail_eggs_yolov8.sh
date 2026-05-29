#!/bin/bash
set -euxo pipefail

# Run this inside Sipeed/SOPHGO's TPU-MLIR Docker environment.
# Expected files next to this script:
#   ${NET_NAME}.onnx
#   image.jpg
#   cali_images/*.jpg

net_name="${NET_NAME:-snail_eggs_yolov8n_640x480}"
input_w="${INPUT_W:-640}"
input_h="${INPUT_H:-480}"

# YOLOv8 detection output nodes for the exported Ultralytics YOLOv8n graph.
# If your ONNX graph differs, inspect it with Netron and update these names.
output_names="/model.22/dfl/conv/Conv_output_0,/model.22/Sigmoid_output_0"

TPUC_ROOT="${TPUC_ROOT:-$(python -c 'import importlib.util, pathlib; spec=importlib.util.find_spec("tpu_mlir"); print(pathlib.Path(spec.origin).resolve().parent)')}"
export TPUC_ROOT
export PATH="${TPUC_ROOT}/bin:${TPUC_ROOT}/python/tools:${PATH}"
export PYTHONPATH="${TPUC_ROOT}:${TPUC_ROOT}/python:${PYTHONPATH:-}"

run_tool() {
  local tool="$1"
  shift
  python "${TPUC_ROOT}/python/tools/${tool}" "$@"
}

mkdir -p workspace
cd workspace

transform_args=(
  --model_name ${net_name}
  --model_def ../${net_name}.onnx
  --input_shapes "[[1,3,${input_h},${input_w}]]"
  --mean "0,0,0"
  --scale "0.00392156862745098,0.00392156862745098,0.00392156862745098"
  --keep_aspect_ratio
  --pixel_format rgb
  --channel_format nchw
  --output_names "${output_names}"
  --mlir ${net_name}.mlir
)

if [ "${VALIDATE_TRANSFORM:-1}" = "1" ]; then
  transform_args+=(
    --test_input ../image.jpg
    --test_result ${net_name}_top_outputs.npz
    --tolerance 0.99,0.99
  )
fi

run_tool model_transform.py "${transform_args[@]}"

run_tool run_calibration.py ${net_name}.mlir \
  --dataset ../cali_images \
  --input_num 200 \
  -o ${net_name}_cali_table

run_tool model_deploy.py \
  --mlir ${net_name}.mlir \
  --quantize INT8 \
  --quant_input \
  --calibration_table ${net_name}_cali_table \
  --processor cv181x \
  --skip_validation \
  --model ${net_name}.cvimodel

cp ${net_name}.cvimodel ..
echo "Done: ${net_name}.cvimodel"
