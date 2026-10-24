#!/bin/bash
#SBATCH --job-name=sar-deploy
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/%u/sar/logs/%x_%j.out
#
# The trained agent served the way a product would use it, on one GPU:
#   vLLM (hermes tool parsing) <- vllm-sglang-infra gateway (tenant API key,
#   usage metering to JSONL, Prometheus metrics) <- the agent loop / MCP server.
# Checks:
#   1. the same greedy eval straight against vLLM and through the gateway
#      gives the same answers: the gateway forwards tool-call requests and
#      responses unchanged;
#   2. the gateway metered every model request to the tenant;
#   3. an MCP client (tools/mcp_demo.py) asks the agent questions through the
#      gateway.
#   MODEL_DIR=$MODELS/qwen3b_sft2_mt-step60 TAG=deploy-sft2-60 sbatch cluster/deploy_demo.sbatch
# The gateway binary comes from vllm-sglang-infra's push (GATEWAY_BIN).
set -euo pipefail
source "${SAR_CODE:-/scratch/$USER/sar/code}/cluster/common.sh"
TAG=${TAG:?set TAG, the results directory name}
SETS=${SETS:-spider_dev bird_dev}
GATEWAY_BIN=${GATEWAY_BIN:-/scratch/$USER/vsi/code/bin/linux-amd64/gateway}
gpu_info
stage_env
if [ -n "${MODEL_DIR:-}" ]; then MODEL=$MODEL_DIR; else MODEL=$(stage_model "${MODEL_NAME:?set MODEL_NAME or MODEL_DIR}"); fi
export SAR_DB_ROOT=$(stage_dbs omnisql)/data
SAR_DATA=$DATA/processed/sql
OUT=$ROOT/results/$TAG
mkdir -p "$OUT"
cp "$CODE/VERSION" "$OUT/code_version.txt" 2>/dev/null || true
free_port() { python -P -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1])'; }

# A node-local copy: replacing the file on NFS while it runs kills it with
# SIGBUS (vllm-sglang-infra failure #24).
GW=/tmp/sar-gateway-$SLURM_JOB_ID
cp "$GATEWAY_BIN" "$GW" && chmod +x "$GW"

PORT=$(free_port)
setsid vllm serve "$MODEL" --served-model-name "$TAG" --port "$PORT" --host 127.0.0.1 \
  --max-model-len 16384 --gpu-memory-utilization 0.85 --enable-prefix-caching \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --generation-config vllm \
  >"$OUT/vllm.log" 2>&1 &
VLLM_PID=$!

# One tenant with an API key; the gateway stores only its hash. Usage records
# go to a JSONL file instead of PostgreSQL. tokens_per_minute 0: no budget.
API_KEY=$(head -c 18 /dev/urandom | od -An -tx1 | tr -d ' \n')
GW_PORT=$(free_port)
cat >"$OUT/gateway.yaml" <<EOF
listen: "127.0.0.1:$GW_PORT"
backends:
  - { name: vllm-0, url: "http://127.0.0.1:$PORT" }
scheduler:
  policy: least-load
tenancy:
  tenants:
    - name: sql-agent-eval
      api_key_hashes: ["$(echo "$API_KEY" | "$GW" -hash-key)"]
      priority: interactive
      tokens_per_minute: 0
usage:
  sink: jsonl
  path: $OUT/usage.jsonl
EOF
rm -f "$OUT/usage.jsonl"
setsid "$GW" -config "$OUT/gateway.yaml" >"$OUT/gateway.log" 2>&1 &
GW_PID=$!
trap 'for p in $GW_PID $VLLM_PID; do kill -TERM -- -$p 2>/dev/null || true; done; sleep 5;
      for p in $GW_PID $VLLM_PID; do kill -KILL -- -$p 2>/dev/null || true; done; rm -f "$GW"' EXIT

for ((t = 0; t < 900; t += 5)); do
  curl -sf -o /dev/null --noproxy '*' "http://127.0.0.1:$PORT/health" && break
  kill -0 $VLLM_PID 2>/dev/null || { log "vllm exited"; tail -50 "$OUT/vllm.log" >&2; exit 1; }
  sleep 5
done
log "vllm ready after ${t}s"
# The gateway's health monitor marks the backend up and discovers its model.
for ((t = 0; t < 60; t += 2)); do
  curl -sf --noproxy '*' -H "Authorization: Bearer $API_KEY" "http://127.0.0.1:$GW_PORT/v1/models" | grep -q "\"$TAG\"" && break
  kill -0 $GW_PID 2>/dev/null || { log "gateway exited"; tail -50 "$OUT/gateway.log" >&2; exit 1; }
  sleep 2
done
log "gateway serving $TAG after ${t}s"
code=$(curl -s -m 10 -o /dev/null -w '%{http_code}' --noproxy '*' -H 'Content-Type: application/json' \
  -d "{\"model\": \"$TAG\", \"messages\": [{\"role\": \"user\", \"content\": \"hi\"}], \"max_tokens\": 1}" \
  "http://127.0.0.1:$GW_PORT/v1/chat/completions")
log "a chat request without an API key gets HTTP $code (expected 401)"

files=()
for s in $SETS; do files+=("$SAR_DATA/eval_$s.parquet"); done
log "1/3 eval straight against vLLM"
python -P -m sar.eval.run_eval --base-url "http://127.0.0.1:$PORT/v1" --model "$TAG" \
  --data "${files[@]}" --out "$OUT/direct"
log "2/3 the same eval through the gateway"
python -P -m sar.eval.run_eval --base-url "http://127.0.0.1:$GW_PORT/v1" --api-key "$API_KEY" --model "$TAG" \
  --data "${files[@]}" --out "$OUT/gateway" --save-messages
sleep 3 # the usage sink writes in the background
curl -s --noproxy '*' "http://127.0.0.1:$GW_PORT/metrics" >"$OUT/gateway_metrics.txt" || true
python -P "$CODE/tools/deploy_check.py" "$OUT/direct" "$OUT/gateway" "$OUT/usage.jsonl" | tee "$OUT/deploy_check.txt"

log "3/3 MCP client -> MCP server -> gateway -> vLLM"
SAR_LLM_BASE_URL="http://127.0.0.1:$GW_PORT/v1" SAR_LLM_MODEL=$TAG SAR_LLM_API_KEY=$API_KEY \
  python -P "$CODE/tools/mcp_demo.py" "$SAR_DATA/eval_spider_dev.parquet" "$SAR_DB_ROOT" "$OUT/mcp_demo.json" --n 5
log "results in $OUT"
