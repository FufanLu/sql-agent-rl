#!/bin/bash
#SBATCH --job-name=sar-grad-probe
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/sar/logs/%x_%j.out
#
# tools/grad_probe.py on the first SFT examples under each setting the SFT
# run could have hit: bf16 with/without gradient checkpointing, and fp32,
# with the env's transformers (5.9) and with the SFT overlay's (5.5.4).
#   sbatch cluster/grad_probe.sbatch
set -euo pipefail
source "${SAR_CODE:-/scratch/$USER/sar/code}/cluster/common.sh"
gpu_info
stage_env
MODEL=$(stage_model Qwen2.5-Coder-3B-Instruct)
DATA_JSON=$DATA/processed/sft/sql_agent_sft.json
cd "$CODE"
base_pp=$PYTHONPATH
for tf in env overlay; do
  if [ "$tf" = overlay ]; then export PYTHONPATH="$ROOT/envs/sft-$ENV_REV:$base_pp"; else export PYTHONPATH="$base_pp"; fi
  for setting in "--dtype bf16 --ckpt" "--dtype bf16" "--dtype fp32"; do
    log "== transformers from $tf, $setting"
    # shellcheck disable=SC2086
    python -P tools/grad_probe.py "$MODEL" "$DATA_JSON" --n 8 $setting || log "probe failed"
  done
done
