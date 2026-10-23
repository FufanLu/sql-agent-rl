#!/bin/bash
#SBATCH --job-name=sar-export
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/sar/logs/%x_%j.out
#
# Merges a verl FSDP checkpoint (sharded actor state) into a plain Hugging
# Face model directory under $MODELS, which eval.sbatch, vLLM serving and
# quantization all read.
#   RUN=qwen3b_mt STEP=60 sbatch cluster/export_ckpt.sbatch
# -> $MODELS/qwen3b_mt-step60
set -euo pipefail
source "${SAR_CODE:-/scratch/$USER/sar/code}/cluster/common.sh"
RUN=${RUN:?} STEP=${STEP:?}
SRC=$CKPT/$RUN/global_step_$STEP/actor
DST=$MODELS/$RUN-step$STEP
[ -d "$SRC" ] || { log "no checkpoint at $SRC"; exit 1; }
stage_env
rm -rf "$DST.partial"
python -P -m verl.model_merger merge --backend fsdp --local_dir "$SRC" --target_dir "$DST.partial"
# The tokenizer and chat template travel with the actor checkpoint
# (huggingface/ subdirectory); vLLM needs them next to the weights.
[ -d "$SRC/huggingface" ] && cp -n "$SRC"/huggingface/* "$DST.partial/" 2>/dev/null || true
rm -rf "$DST" && mv "$DST.partial" "$DST" && touch "$DST/.complete"
du -sh "$DST" >&2
log "exported $DST"
