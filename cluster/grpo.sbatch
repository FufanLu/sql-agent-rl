#!/bin/bash
#SBATCH --job-name=sar-grpo
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/sar/logs/%x_%j.out
#
# One 8-hour leg of a GRPO run. A run is longer than the partition's 8-hour
# limit, so cluster/submit_grpo.sh queues several legs in a chain; each one
# resumes from the newest checkpoint in $CKPT/$RUN (trainer.resume_mode=auto),
# and a leg that finds the run finished exits right away.
#   RUN=qwen3b_mt MODEL_NAME=Qwen2.5-Coder-3B-Instruct sbatch cluster/grpo.sbatch [hydra overrides...]
set -euo pipefail
source "${SAR_CODE:-/scratch/$USER/sar/code}/cluster/common.sh"
RUN=${RUN:?set RUN to a config name in configs/grpo}
MODEL_NAME=${MODEL_NAME:?set MODEL_NAME to a directory in $MODELS}
# Runs whose config does not include _sdpa use flash-attn (padding removal).
grep -q "_sdpa" "$CODE/configs/grpo/$RUN.yaml" || export SAR_USE_FA=1
gpu_info
stage_env

export SAR_CODE=$CODE
export SAR_MODEL=$(stage_model "$MODEL_NAME")
export SAR_DB_ROOT=$(stage_dbs omnisql)/data
export SAR_DATA=$DATA/processed/sql
export SAR_CKPT_DIR=$CKPT/$RUN
[ -f "$SAR_DATA/manifest.json" ] || { log "no $SAR_DATA/manifest.json: run cluster/build_data.sbatch"; exit 1; }
mkdir -p "$SAR_CKPT_DIR"
cp "$SAR_DATA/manifest.json" "$SAR_CKPT_DIR/data_manifest.json"  # which data this run saw
cp "$CODE/VERSION" "$SAR_CKPT_DIR/code_version_$SLURM_JOB_ID.txt" 2>/dev/null || true

export WANDB_DIR=$SAR_CKPT_DIR RAY_TMPDIR=/tmp/sar-ray-$SLURM_JOB_ID
export WANDB_RUN_ID=${WANDB_RUN_ID:-$RUN} WANDB_RESUME=allow  # one W&B run across legs
cd /tmp
python -P -m verl.trainer.main_ppo \
  --config-path "$CODE/configs/grpo" --config-name "$RUN" \
  "$@" 2>&1 | tee -a "$LOGS/grpo_$RUN.log"
log "leg done"
