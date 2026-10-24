#!/usr/bin/env bash
# Queues a GRPO run as a chain of 8-hour legs (cluster/grpo.sbatch), each
# starting after the previous one ends, however it ended.
#   cluster/submit_grpo.sh RUN MODEL_NAME LEGS [hydra overrides...]
#   cluster/submit_grpo.sh qwen3b_mt Qwen2.5-Coder-3B-Instruct 4
# Run it on the cluster login node. Every leg counts against the 8-job
# submit limit, so keep LEGS small and add more later if the run needs them.
set -euo pipefail
RUN=${1:?run} MODEL_NAME=${2:?model} LEGS=${3:?legs}
shift 3
cd "$(dirname "$0")/.."
# sbatch takes the account from $SBATCH_ACCOUNT when the cluster needs one.
dep=${AFTER:-}
for ((i = 1; i <= LEGS; i++)); do
  id=$(RUN=$RUN MODEL_NAME=$MODEL_NAME sbatch --parsable --job-name="sar-$RUN" \
    ${dep:+--dependency=afterany:$dep} cluster/grpo.sbatch "$@")
  echo "leg $i: $id"
  dep=$id
done
