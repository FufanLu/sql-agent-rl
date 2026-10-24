#!/bin/bash
#SBATCH --job-name=sar-gpu
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100@80g
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/sar/logs/%x_%j.out
#
# Runs a command inside the training env on one 80 GB A100 (gpu-short).
#   sbatch cluster/gpu_run.sbatch python -P tools/train_tool_tokens.py ...
set -euo pipefail
source "${SAR_CODE:-/scratch/$USER/sar/code}/cluster/common.sh"
gpu_info
stage_env
cd "$CODE"
"$@"
