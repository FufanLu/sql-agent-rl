#!/bin/bash
#SBATCH --job-name=sar-sft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/sar/logs/%x_%j.out
#
# LLaMA-Factory SFT, then exports the result where eval and GRPO can use it
# ($MODELS/<run name>).
#   MODEL_NAME=Qwen2.5-Coder-3B-Instruct sbatch cluster/sft.sbatch configs/sft/qwen3b_sft.yaml
set -euo pipefail
source "${SAR_CODE:-/scratch/$USER/sar/code}/cluster/common.sh"
CFG=${1:?config yaml}
MODEL_NAME=${MODEL_NAME:?}
RUN=$(basename "$CFG" .yaml)
export SAR_USE_FA=1
gpu_info
stage_env
# Package versions LLaMA-Factory is tested with (cluster/build_sft_overlay.sbatch).
[ -d "$ROOT/envs/sft-$ENV_REV" ] || { log "missing $ROOT/envs/sft-$ENV_REV: run build_sft_overlay.sbatch"; exit 1; }
export PYTHONPATH="$ROOT/envs/sft-$ENV_REV:$PYTHONPATH"
MODEL=$(stage_model "$MODEL_NAME")
OUT=$CKPT/$RUN
export WANDB_DIR=$OUT WANDB_PROJECT=sql-agent-rl
# LLaMA-Factory starts DeepSpeed only through torchrun, even on one GPU.
export FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=1
mkdir -p "$OUT"
cp "$CODE/VERSION" "$OUT/code_version.txt" 2>/dev/null || true
cd "$CODE"
llamafactory-cli train "$CFG" \
  model_name_or_path="$MODEL" dataset_dir="$DATA/processed/sft" output_dir="$OUT" \
  2>&1 | tee "$LOGS/sft_$RUN.log"
# The final save is a plain HF directory; publish it next to the base models.
rm -rf "$MODELS/$RUN"
cp -r "$OUT" "$MODELS/$RUN.partial"
rm -rf "$MODELS/$RUN.partial"/checkpoint-*
mv "$MODELS/$RUN.partial" "$MODELS/$RUN" && touch "$MODELS/$RUN/.complete"
log "SFT model at $MODELS/$RUN"
