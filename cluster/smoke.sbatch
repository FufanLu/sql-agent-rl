#!/bin/bash
#SBATCH --job-name=sar-smoke
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/%u/sar/logs/%x_%j.out
#
# P0 stack check: verl GRPO on GSM8K with Qwen2.5-0.5B-Instruct on one GPU,
# 20 steps. Proves torch + vLLM rollout + FSDP training + Ray + weight sync
# work together on this cluster before any SQL-specific code is involved.
#   sbatch /scratch/$USER/sar/code/cluster/smoke.sbatch
# ATTN=sdpa (default until the flash-attn wheel exists) or ATTN=flash_attention_2.
set -euo pipefail
source "${SAR_CODE:-/scratch/$USER/sar/code}/cluster/common.sh"
gpu_info
stage_env
ATTN=${ATTN:-sdpa}
RMPAD=$([ "$ATTN" = flash_attention_2 ] && echo True || echo False)
MODEL=$(stage_model Qwen2.5-0.5B-Instruct)

GSM=$DATA/processed/gsm8k
[ -f "$GSM/train.parquet" ] || python -P -m sar.data.gsm8k --src "$DATA/hf/gsm8k" --out "$GSM"

RUN=smoke-gsm8k-$ATTN-$SLURM_JOB_ID
export WANDB_DIR=$CKPT/$RUN RAY_TMPDIR=/tmp/sar-ray-$SLURM_JOB_ID
mkdir -p "$WANDB_DIR"
cd /tmp
python -P -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="$GSM/train.parquet" data.val_files="$GSM/test.parquet" \
  data.train_batch_size=64 data.max_prompt_length=512 data.max_response_length=512 \
  data.val_max_samples=256 data.dataloader_num_workers=2 \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.model.use_remove_padding=$RMPAD \
  +actor_rollout_ref.model.override_config.attn_implementation=$ATTN \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n=5 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
  actor_rollout_ref.rollout.agent.num_workers=2 \
  trainer.n_gpus_per_node=1 trainer.nnodes=1 \
  trainer.logger='[console,wandb]' trainer.project_name=sql-agent-rl trainer.experiment_name="$RUN" \
  trainer.total_training_steps=20 trainer.test_freq=10 trainer.val_before_train=True \
  trainer.save_freq=-1 trainer.default_local_dir="$CKPT/$RUN" \
  2>&1 | tee "$LOGS/$RUN.log"
log "smoke done"
