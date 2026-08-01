#!/bin/bash
#SBATCH --job-name=thesis-build-container
#SBATCH --partition=frida
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=%x-%j.out

# One-time container build: NGC PyTorch base + this project + CUDA llama-cpp-python,
# saved as a squashfs so benchmark jobs never recompile.
#
#   sbatch deploy/build_container.sbatch
#
# A GPU is requested so the CUDA toolchain in the image matches a real device during the
# build (and lets you smoke-test inside the same job).

set -euo pipefail

PROJECT="$WORK/thesis"
IMAGE_OUT="$PROJECT/containers/thesis-eval.sqsh"
BASE_IMAGE="nvcr.io#nvidia/pytorch:25.06-py3"   # CUDA + torch preinstalled, Python >= 3.12

mkdir -p "$PROJECT/containers"

srun --container-image "$BASE_IMAGE" \
     --container-save "$IMAGE_OUT" \
     --container-mounts "$PROJECT:/work" \
     --container-workdir /work/code \
     bash -exc '
       # llama-cpp-python must be COMPILED with CUDA first (the PyPI wheel is CPU-only on
       # Linux); installing it up front stops the project install from grabbing that wheel.
       CMAKE_ARGS="-DGGML_CUDA=on" FORCE_CMAKE=1 \
           pip install --no-cache-dir --no-binary llama-cpp-python "llama-cpp-python>=0.3.20"

       # Project + metrics extra from pyproject. torch/transformers already ship with the
       # NGC image and satisfy the constraints, so pip reuses them.
       pip install --no-cache-dir -e ".[metrics]"

       python - <<EOF
import llama_cpp, torch
print("llama-cpp-python", llama_cpp.__version__)
print("torch", torch.__version__, "cuda:", torch.cuda.is_available())
EOF
     '

echo "Container saved to $IMAGE_OUT"
