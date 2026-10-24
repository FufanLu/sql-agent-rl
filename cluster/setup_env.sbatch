#!/bin/bash
#SBATCH --job-name=sar-setup-env
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/sar/logs/%x_%j.out
#
# Builds the training environment (verl + vLLM + LLaMA-Factory + TRL) once and
# packs it for GPU jobs.
#   sbatch /scratch/$USER/sar/code/cluster/setup_env.sbatch
# Runs on the short partition because its nodes can reach PyPI; GPU nodes
# cannot. The venv is built at the node-local path GPU jobs unpack it to.
set -euo pipefail
source "${SAR_CODE:-/scratch/$USER/sar/code}/cluster/common.sh"

module load anaconda3/2024.06 >/dev/null 2>&1 || true
PYBASE=$(command -v python3)
"$PYBASE" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13), sys.version'
free_gb=$(df -Pk /tmp | awk 'NR == 2 { print int($4 / 1048576) }')
[ "$free_gb" -ge 40 ] || { log "only ${free_gb}G free in /tmp on $(hostname)"; exit 1; }

export UV_CACHE_DIR=/tmp/sar-uv-cache-$SLURM_JOB_ID UV_LINK_MODE=copy UV_NO_PROGRESS=1
trap 'rm -rf "$UV_CACHE_DIR"' EXIT
rm -rf "$ENV_DIR"
"$PYBASE" -m venv "$ENV_DIR"
"$ENV_DIR/bin/pip" install -q --upgrade pip uv
UV=("$ENV_DIR/bin/uv" pip install --python "$ENV_DIR/bin/python")

# 1. vLLM first: its wheel pins torch 2.11.0, and --torch-backend picks the
#    cu129 build of torch to match.
WHEEL=https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+${CUDA_TAG}-cp38-abi3-manylinux_2_28_x86_64.whl
log "installing vllm $VLLM_VERSION ($CUDA_TAG)"
"${UV[@]}" --torch-backend "$CUDA_TAG" "vllm @ $WHEEL"

# 2. verl without its backend extras: those pin vllm 0.24 and cu130 builds.
#    The rest of what its fsdp extra brings is listed by hand with CUDA 12
#    variants (cupy-cuda12x for ray.util.collective weight sync).
#    transformers 5.9.0 is the version verl 0.9.1 tests against.
log "installing verl $VERL_VERSION and training deps"
"${UV[@]}" --torch-backend "$CUDA_TAG" \
  "verl[verl-core,math]==$VERL_VERSION" \
  "torch==2.11.0" "transformers==5.9.0" \
  "liger-kernel>=0.8.2" "cupy-cuda12x" \
  "trl==0.27.0" "peft" "accelerate" "deepspeed" \
  "wandb" "tensorboard" "hydra-core" \
  "fastapi" "uvicorn" "httpx" "pydantic>=2" "sqlglot" "pandas" "pyarrow" \
  "lm-eval" "mcp>=2,<3"

# 3. LLaMA-Factory for SFT. --no-deps: its own pins would move torch and
#    transformers; what it needs beyond them is listed here.
log "installing LLaMA-Factory"
"${UV[@]}" --torch-backend "$CUDA_TAG" --no-deps "llamafactory"
"${UV[@]}" --torch-backend "$CUDA_TAG" "gradio" "fire" "omegaconf" "sentencepiece" "tiktoken" "protobuf" "av" "librosa" "einops" "modelscope" "tyro" || log "some LLaMA-Factory extras failed"

# 4. flash-attn, if build_flash_attn.sbatch has produced a wheel. Without it
#    training runs with sdpa and use_remove_padding=False.
fa=$(ls "$WHEELS"/flash_attn-*.whl 2>/dev/null | tail -1 || true)
if [ -n "$fa" ]; then
  log "installing $fa"
  "${UV[@]}" --no-deps "$fa"
else
  log "no flash-attn wheel in $WHEELS yet"
fi

# No GPU here, so this only proves the packages import together.
"$ENV_DIR/bin/python" -P -c "
import torch, vllm, transformers, verl, ray, trl, peft, deepspeed
print('torch', torch.__version__, 'cuda runtime', torch.version.cuda)
print('vllm', vllm.__version__, 'verl', verl.__version__, 'transformers', transformers.__version__)
print('ray', ray.__version__, 'trl', trl.__version__, 'peft', peft.__version__, 'deepspeed', deepspeed.__version__)
try:
    import flash_attn; print('flash_attn', flash_attn.__version__)
except ImportError:
    print('flash_attn missing')
"
"$ENV_DIR/bin/python" -P -c "import llamafactory; print('llamafactory', llamafactory.__version__)" || log "llamafactory import failed"
"$ENV_DIR/bin/uv" pip freeze --python "$ENV_DIR/bin/python" >"$ROOT/envs/$ENV_NAME.freeze.txt"

log "packing $ENV_TAR"
tar -C /tmp -cf "$ENV_TAR.partial" "$ENV_NAME"
mv "$ENV_TAR.partial" "$ENV_TAR"
du -sh "$ENV_TAR" >&2
log "done"
