#!/bin/bash
#SBATCH --job-name=sar-sft-overlay
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/%u/sar/logs/%x_%j.out
#
# LLaMA-Factory 0.9.5 checks for versions older than the training env has
# (transformers<=5.6.0, datasets<=4.0.0, peft<=0.20.0, trl<=0.24.0; the env has
# 5.9.0 / 5.1.0 / 0.21.2 / 0.27.0 for verl). These are pure-Python packages, so
# they go into a separate directory that only cluster/sft.sbatch puts first
# on PYTHONPATH, instead of disabling LLaMA-Factory's version check.
#   sbatch cluster/build_sft_overlay.sbatch
set -euo pipefail
source "${SAR_CODE:-/scratch/$USER/sar/code}/cluster/common.sh"
stage_env
dst=$ROOT/envs/sft-$ENV_REV
rm -rf "$dst.partial"
# --no-deps: everything else (torch, tokenizers, huggingface-hub, accelerate)
# comes from the env and already satisfies these versions.
"$ENV_DIR/bin/uv" pip install --python "$ENV_DIR/bin/python" --target "$dst.partial" --no-deps \
  "transformers==5.5.4" "datasets==4.0.0" "peft==0.20.0" "trl==0.24.0" "fsspec[http]==2025.3.0"
rm -rf "$dst" && mv "$dst.partial" "$dst"
# Importing llamafactory.hparams runs LLaMA-Factory's own dependency check.
PYTHONPATH="$dst:$PYTHONPATH" python -P -c "
import transformers, datasets, peft, trl, llamafactory.hparams
print('transformers', transformers.__version__, 'datasets', datasets.__version__, 'peft', peft.__version__, 'trl', trl.__version__)
print('llamafactory dependency check passed')"
log "SFT overlay at $dst"
