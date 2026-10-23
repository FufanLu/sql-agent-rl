#!/bin/bash
#SBATCH --job-name=sar-cpu
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/sar/logs/%x_%j.out
#
# Runs a command inside the training env on a CPU node (the login node kills
# anything that imports torch).
#   sbatch cluster/cpu_run.sbatch python -P tools/check_sft_template.py /scratch/$USER/sar/models/Qwen2.5-Coder-3B-Instruct
set -euo pipefail
source "${SAR_CODE:-/scratch/$USER/sar/code}/cluster/common.sh"
stage_env
cd "$CODE"
"$@"
