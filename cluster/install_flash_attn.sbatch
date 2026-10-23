#!/bin/bash
#SBATCH --job-name=sar-install-fa
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/sar/logs/%x_%j.out
#
# Installs the flash-attn wheel from build_flash_attn.sbatch into its own
# directory, $ROOT/envs/fa-$ENV_REV, which jobs add with SAR_USE_FA=1.
#   sbatch cluster/install_flash_attn.sbatch
set -euo pipefail
source "${SAR_CODE:-/scratch/$USER/sar/code}/cluster/common.sh"
stage_env
whl=$(ls "$WHEELS"/flash_attn-*.whl | tail -1)
dst=$ROOT/envs/fa-$ENV_REV
rm -rf "$dst.partial"
"$ENV_DIR/bin/uv" pip install --python "$ENV_DIR/bin/python" --target "$dst.partial" --no-deps "$whl"
rm -rf "$dst" && mv "$dst.partial" "$dst"
ls "$dst" >&2
# No GPU here: importing the CUDA extension may fail for lack of libcuda;
# the Python package and the shared object must at least be present.
ls "$dst"/flash_attn_2_cuda*.so >&2
PYTHONPATH="$dst" python -P -c "import importlib.util as u; print('flash_attn found at', u.find_spec('flash_attn').origin)"
log "installed $whl -> $dst"
