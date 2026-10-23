# Sourced by every cluster job. Paths, versions, and staging helpers live here.
#
# Layout on the shared filesystem (/scratch is NFS: fine for big sequential
# reads, slow for the thousands of small files a Python env imports):
#   $ROOT/code     this repo (cluster/push.sh)
#   $ROOT/envs     packed Python environments (setup_env.sbatch)
#   $ROOT/wheels   wheels built here (build_flash_attn.sbatch)
#   $ROOT/models   model weights (download_model.sbatch)
#   $ROOT/data     datasets and SQLite databases (download_data.sbatch)
#   $ROOT/ckpt     training checkpoints
#   $ROOT/logs     job logs
# Jobs unpack the env and copy the weights to node-local /tmp, once per node.

ROOT=${SAR_ROOT:-/scratch/$USER/sar}
CODE=${SAR_CODE:-$ROOT/code}
MODELS=$ROOT/models
DATA=$ROOT/data
CKPT=$ROOT/ckpt
LOGS=$ROOT/logs
WHEELS=$ROOT/wheels

# The H200 nodes run driver 570 (CUDA 12.8). verl 0.9.1 locks torch 2.11 +
# vLLM 0.24 on cu130, which that driver cannot run. vLLM 0.26.0 is the newest
# release on torch 2.11.0 and has a cu129 build that runs on driver 570 through
# CUDA minor-version compatibility (verified by vllm-sglang-infra's
# check_gpu.sbatch); verl's setup.py only asks for vllm>=0.18.
VERL_VERSION=${VERL_VERSION:-0.9.1}
VLLM_VERSION=${VLLM_VERSION:-0.26.0}
CUDA_TAG=${CUDA_TAG:-cu129}
# Bump ENV_REV when the package set changes: nodes keep unpacked envs, and
# a new name is what makes them unpack the new one.
ENV_REV=${ENV_REV:-r1}
ENV_NAME=sar-verl${VERL_VERSION}-vllm${VLLM_VERSION}-${CUDA_TAG}-${ENV_REV}
ENV_DIR=/tmp/$ENV_NAME # identical path on every node, so the packed venv needs no rewriting
ENV_TAR=$ROOT/envs/$ENV_NAME.tar

mkdir -p "$ROOT/envs" "$MODELS" "$DATA" "$CKPT" "$LOGS" "$WHEELS"

# Compute nodes have no internet, and caches must not land in $HOME.
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export HF_HOME=/tmp/sar-hf-$USER
export XDG_CACHE_HOME=/tmp/sar-cache-$USER
export VLLM_CACHE_ROOT=/tmp/sar-vllm-cache-$USER
export TRITON_CACHE_DIR=/tmp/sar-triton-$USER
export PYTHONUNBUFFERED=1
# W&B cannot reach its server from a compute node: runs log offline into the
# checkpoint directory, and cluster/wandb_sync.sh uploads them from the login node.
export WANDB_MODE=${WANDB_MODE:-offline}

log() { echo "[$(date +%H:%M:%S)] $*" >&2; }

# stage_env unpacks $ENV_TAR to $ENV_DIR unless this node already has it.
# A lock serializes jobs that land on the same node; unpacking into a scratch
# directory and renaming makes a half-finished unpack impossible to pick up.
stage_env() {
  [ -f "$ENV_TAR" ] || { log "missing $ENV_TAR: run setup_env.sbatch first"; return 1; }
  (
    flock 9
    if [ ! -f "$ENV_DIR/.complete" ]; then
      local t0=$SECONDS stage=/tmp/.sar-stage-$$
      rm -rf "$ENV_DIR" "$stage" && mkdir -p "$stage"
      tar -C "$stage" -xf "$ENV_TAR"
      mv "$stage/$ENV_NAME" "$ENV_DIR" && rmdir "$stage"
      touch "$ENV_DIR/.complete"
      log "unpacked $ENV_NAME in $((SECONDS - t0))s"
    fi
  ) 9>"/tmp/.$ENV_NAME.lock"
  activate_env
}

# activate_env puts the env's tools on PATH and loads a CUDA 12.8 toolkit,
# the newest the H200 nodes' driver runs (Triton and FlashInfer JIT need it).
activate_env() {
  export PATH="$ENV_DIR/bin:$PATH" VIRTUAL_ENV="$ENV_DIR"
  # Pure-Python fixes added after the env was packed live in an overlay on
  # /scratch, so they do not force every node to unpack a new env. Fold them
  # into the env at the next rebuild (ENV_REV).
  if [ -d "$ROOT/envs/overlay-$ENV_REV" ]; then
    export PYTHONPATH="$ROOT/envs/overlay-$ENV_REV${PYTHONPATH:+:$PYTHONPATH}"
  fi
  if module load cuda/12.8.0 >/dev/null 2>&1 && command -v nvcc >/dev/null; then
    CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
    export CUDA_HOME
  fi
  export FLASHINFER_WORKSPACE_BASE=/tmp/sar-flashinfer-$USER
  export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
  # vllm-sglang-infra saw NCCL segfault while loading the node's network
  # plugin and its InfiniBand transport. Jobs here stay on one node.
  export NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN:-none}
  export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
  # The repo's package (src/sar) is importable without installing.
  export PYTHONPATH="$CODE/src${PYTHONPATH:+:$PYTHONPATH}"
}

# stage_model NAME copies $MODELS/NAME to node-local disk unless this node
# already has it, and prints the local path. Loading safetensors straight off
# NFS goes through mmap page faults and is far slower than one sequential copy.
stage_model() {
  local name=$1 src=$MODELS/$1 dst=/tmp/sar-models/$1
  [ -f "$src/.complete" ] || { log "model $name not downloaded: run download_model.sbatch"; return 1; }
  mkdir -p /tmp/sar-models
  (
    flock 9
    if [ ! -f "$dst/.complete" ]; then
      local t0=$SECONDS
      rm -rf "$dst" "$dst.partial"
      cp -r "$src" "$dst.partial" && mv "$dst.partial" "$dst"
      log "staged $name ($(du -sh "$dst" | cut -f1)) in $((SECONDS - t0))s"
    fi
  ) 9>"/tmp/sar-models/.$name.lock"
  echo "$dst"
}

# stage_dbs NAME copies $DATA/NAME (a directory of SQLite databases) to
# node-local disk. Thousands of rollouts open these files; NFS locking and
# latency would make every SQL call slow.
stage_dbs() {
  local name=$1 src=$DATA/$1 dst=/tmp/sar-data/$1
  [ -d "$src" ] || { log "data $name missing: run download_data.sbatch"; return 1; }
  mkdir -p /tmp/sar-data
  (
    flock 9
    if [ ! -f "$dst/.complete" ]; then
      local t0=$SECONDS
      rm -rf "$dst" "$dst.partial"
      # BIRD's training databases are the bulk of OmniSQL's data and nothing
      # here trains on BIRD train; leave them on /scratch.
      rsync -a --exclude 'bird/train/train_databases/' "$src/" "$dst.partial/" && mv "$dst.partial" "$dst"
      touch "$dst/.complete"
      log "staged data $name ($(du -sh "$dst" | cut -f1)) in $((SECONDS - t0))s"
    fi
  ) 9>"/tmp/sar-data/.$name.lock"
  echo "$dst"
}

# gpu_info logs the GPU, driver, and memory of this node.
gpu_info() {
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader >&2 || true
}
