#!/bin/bash
#SBATCH --job-name=sar-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/sar/logs/%x_%j.out
#
# Serves a model with vLLM (OpenAI API with hermes tool-call parsing) and runs
# the agent evaluation (sar.eval.run_eval) against it.
#   MODEL_NAME=Qwen2.5-Coder-3B-Instruct TAG=qwen3b-base sbatch cluster/eval.sbatch
#   MODEL_DIR=$CKPT/qwen3b_mt/hf/step_60 TAG=qwen3b-mt-60 sbatch cluster/eval.sbatch
# SETS (default: all six eval sets) and EVAL_ARGS (e.g. "--n 5 --temperature 0.6")
# select what to run.
set -euo pipefail
source "${SAR_CODE:-/scratch/$USER/sar/code}/cluster/common.sh"
TAG=${TAG:?set TAG, the results directory name}
SETS=${SETS:-spider_dev spider_test spider_dk spider_syn spider_realistic bird_dev}
gpu_info
stage_env
if [ -n "${MODEL_DIR:-}" ]; then MODEL=$MODEL_DIR; else MODEL=$(stage_model "${MODEL_NAME:?set MODEL_NAME or MODEL_DIR}"); fi
export SAR_DB_ROOT=$(stage_dbs omnisql)/data
SAR_DATA=$DATA/processed/sql
OUT=$ROOT/results/$TAG
mkdir -p "$OUT"
cp "$CODE/VERSION" "$OUT/code_version.txt" 2>/dev/null || true

PORT=$(python -P -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1])')
# Longest eval trajectory: ~6k-token prompt + 6 turns + tool results.
setsid vllm serve "$MODEL" --served-model-name "$TAG" --port "$PORT" --host 127.0.0.1 \
  --max-model-len 16384 --gpu-memory-utilization 0.85 --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser hermes \
  >"$OUT/vllm.log" 2>&1 &
VLLM_PID=$!
trap 'kill -TERM -- -$VLLM_PID 2>/dev/null || true; sleep 5; kill -KILL -- -$VLLM_PID 2>/dev/null || true' EXIT
for ((t = 0; t < 900; t += 5)); do
  curl -sf -o /dev/null --noproxy '*' "http://127.0.0.1:$PORT/health" && break
  kill -0 $VLLM_PID 2>/dev/null || { log "vllm exited"; tail -50 "$OUT/vllm.log" >&2; exit 1; }
  sleep 5
done
log "vllm ready after ${t}s"

files=()
if [ -n "${FILES:-}" ]; then
  for f in $FILES; do files+=("$SAR_DATA/$f"); done  # e.g. FILES=train.parquet for teacher sampling
else
  for s in $SETS; do files+=("$SAR_DATA/eval_$s.parquet"); done
fi
python -P -m sar.eval.run_eval --base-url "http://127.0.0.1:$PORT/v1" --model "$TAG" \
  --data "${files[@]}" --out "$OUT" ${EVAL_ARGS:-}
log "results in $OUT"
