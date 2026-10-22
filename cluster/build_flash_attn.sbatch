#!/bin/bash
#SBATCH --job-name=sar-flash-attn
#SBATCH --partition=short
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/sar/logs/%x_%j.out
#
# Builds a flash-attn wheel for torch 2.11 + CUDA 12, which upstream does not
# publish (its newest wheels stop at torch 2.8). Compiling needs no GPU, only
# nvcc; kernels are built for H200 (sm_90) and A100 (sm_80) only, which cuts
# the build time several-fold.
#   sbatch /scratch/$USER/sar/code/cluster/build_flash_attn.sbatch
set -euo pipefail
source "${SAR_CODE:-/scratch/$USER/sar/code}/cluster/common.sh"
FA_VERSION=${FA_VERSION:-2.8.3}

module load anaconda3/2024.06 >/dev/null 2>&1 || true
module load cuda/12.8.0
CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
export CUDA_HOME
nvcc --version | tail -2 >&2

BUILD=/tmp/sar-fa-build-$SLURM_JOB_ID
export UV_CACHE_DIR=$BUILD/uv-cache UV_LINK_MODE=copy UV_NO_PROGRESS=1
trap 'rm -rf "$BUILD"' EXIT
mkdir -p "$BUILD"
python3 -m venv "$BUILD/venv"
"$BUILD/venv/bin/pip" install -q --upgrade pip uv
"$BUILD/venv/bin/uv" pip install --python "$BUILD/venv/bin/python" --torch-backend "$CUDA_TAG" \
  "torch==2.11.0" ninja packaging psutil setuptools wheel einops

export FLASH_ATTN_CUDA_ARCHS="80;90" MAX_JOBS=${MAX_JOBS:-16} NVCC_THREADS=2
export FLASH_ATTENTION_FORCE_BUILD=TRUE
t0=$SECONDS
"$BUILD/venv/bin/pip" wheel --no-build-isolation --no-deps -w "$BUILD/out" "flash-attn==$FA_VERSION"
cp "$BUILD"/out/flash_attn-*.whl "$WHEELS/"
log "built $(ls "$BUILD"/out) in $((SECONDS - t0))s"
