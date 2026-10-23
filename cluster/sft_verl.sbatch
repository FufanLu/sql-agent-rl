#!/bin/bash
#SBATCH --job-name=sar-sft-verl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/sar/logs/%x_%j.out
#
# SFT with verl's SFT trainer, then merges the final FSDP checkpoint into an HF
# model at $MODELS/<run>, which eval.sbatch and GRPO runs can use.
#   MODEL_NAME=Qwen2.5-Coder-3B-Instruct sbatch cluster/sft_verl.sbatch verl_sft [hydra overrides...]
# RUN_NAME names the run and the published model; SFT_TAG names a diagnostic
# run (own checkpoint dir, no published model).
set -euo pipefail
source "${SAR_CODE:-/scratch/$USER/sar/code}/cluster/common.sh"
CFG=${1:?config name in configs/sft}
shift
MODEL_NAME=${MODEL_NAME:?}
RUN=${RUN_NAME:-${SFT_TAG:-qwen3b_$CFG}}  # RUN_NAME: published model name; SFT_TAG: diagnostic run, not published
export SAR_USE_FA=1  # padding removal needs flash-attn
gpu_info
stage_env
export SAR_MODEL=$(stage_model "$MODEL_NAME")
export SAR_SFT_DATA=$DATA/processed/sft
export SAR_CKPT_DIR=$CKPT/$RUN
export WANDB_DIR=$SAR_CKPT_DIR
mkdir -p "$SAR_CKPT_DIR"
cp "$CODE/VERSION" "$SAR_CKPT_DIR/code_version_$SLURM_JOB_ID.txt" 2>/dev/null || true
cp "$SAR_SFT_DATA/sft_messages.parquet" "$SAR_CKPT_DIR/sft_data_$SLURM_JOB_ID.parquet"  # what this run trained on
cd /tmp
torchrun --standalone --nnodes=1 --nproc_per_node=1 -m verl.trainer.sft_trainer \
  --config-path "$CODE/configs/sft" --config-name "$CFG" "$@" 2>&1 | tee "$LOGS/sft_verl_$RUN.log"

if [ -n "${SFT_TAG:-}" ]; then log "diagnostic run $RUN: not publishing a model"; exit 0; fi
last=$(ls -d "$SAR_CKPT_DIR"/global_step_* | sort -V | tail -1)
log "merging $last"
rm -rf "$MODELS/$RUN.partial"
python -P -m verl.model_merger merge --backend fsdp --local_dir "$last" --target_dir "$MODELS/$RUN.partial"
[ -d "$last/huggingface" ] && cp -n "$last"/huggingface/* "$MODELS/$RUN.partial/" 2>/dev/null || true
rm -rf "$MODELS/$RUN" && mv "$MODELS/$RUN.partial" "$MODELS/$RUN" && touch "$MODELS/$RUN/.complete"
log "SFT model at $MODELS/$RUN"
