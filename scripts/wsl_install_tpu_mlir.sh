#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HOME"

if [ ! -x "$HOME/miniforge3/bin/conda" ]; then
  curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o Miniforge3-Linux-x86_64.sh
  bash Miniforge3-Linux-x86_64.sh -b -p "$HOME/miniforge3"
fi

if ! "$HOME/miniforge3/bin/conda" env list | awk '{print $1}' | grep -qx tpu310; then
  "$HOME/miniforge3/bin/conda" create -y -n tpu310 python=3.10 pip
fi

"$HOME/miniforge3/bin/conda" install -y -n tpu310 gcc_linux-64 gxx_linux-64 cython

mkdir -p "$HOME/tpu_mlir_install"
cd "$HOME/tpu_mlir_install"
if [ ! -f tpu_mlir-1.28.1-py3-none-any.whl ]; then
  curl -L https://github.com/sophgo/tpu-mlir/releases/download/v1.28.1/tpu_mlir-1.28.1-py3-none-any.whl -o tpu_mlir-1.28.1-py3-none-any.whl
fi

export CONDA_PREFIX="$HOME/miniforge3/envs/tpu310"
export PATH="$CONDA_PREFIX/bin:$PATH"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc"
export CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

"$CONDA_PREFIX/bin/pip" install ./tpu_mlir-1.28.1-py3-none-any.whl
"$CONDA_PREFIX/bin/pip" install "setuptools<81"
"$CONDA_PREFIX/bin/pip" install "protobuf==3.20.3" "onnx==1.14.1" onnxruntime onnxsim psutil flatbuffers
"$CONDA_PREFIX/bin/pip" install --index-url https://download.pytorch.org/whl/cpu "torch==2.4.1"
"$CONDA_PREFIX/bin/python" "$SCRIPT_DIR/wsl_patch_tpu_mlir_shell.py"

python -c 'import tpu_mlir; print("tpu_mlir import ok")'
which model_transform.py
which run_calibration.py
which model_deploy.py
