#!/bin/bash
#SBATCH --job-name=sar-data
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/sar/logs/%x_%j.out
#
# Downloads the datasets and SQLite databases to $ROOT/data.
#   sbatch /scratch/$USER/sar/code/cluster/download_data.sbatch
#
#   data/hf/SkyRL-SQL-653-data   training questions (SkyRL-SQL)
#   data/hf/SkyRL-SQL-eval       Spider dev/test/DK/Syn/Realistic + BIRD dev, OmniSQL format
#   data/hf/gsm8k                verl's smoke-test task
#   data/omnisql/                OmniSQL's data.zip: the SQLite databases of
#                                Spider, BIRD and SynSQL (22 GB zipped)
set -euo pipefail
source "${SAR_CODE:-/scratch/$USER/sar/code}/cluster/common.sh"

module load anaconda3/2024.06 >/dev/null 2>&1 || true
VENV=/tmp/sar-hf-venv-$SLURM_JOB_ID
export HF_HUB_OFFLINE=0 HF_XET_CACHE=/tmp/sar-xet-$SLURM_JOB_ID HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export HF_HUB_DISABLE_PROGRESS_BARS=1
trap 'rm -rf "$VENV" "$HF_XET_CACHE"' EXIT
python3 -m venv "$VENV"
"$VENV/bin/pip" install -q "huggingface_hub[hf_xet]"

t0=$SECONDS
"$VENV/bin/python" -P - "$DATA" <<'PY'
import sys
from huggingface_hub import snapshot_download
data = sys.argv[1]
for repo in ["NovaSky-AI/SkyRL-SQL-653-data", "NovaSky-AI/SkyRL-SQL-eval", "openai/gsm8k"]:
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=f"{data}/hf/{repo.split('/')[1]}")
    print("got", repo, flush=True)
snapshot_download(repo_id="seeklhy/OmniSQL-datasets", repo_type="dataset",
                  local_dir=f"{data}/hf/OmniSQL-datasets", allow_patterns=["data.zip", "README.md"])
print("got OmniSQL-datasets", flush=True)
PY
log "downloads done in $((SECONDS - t0))s"

# Unzip next to the zip on node-local disk first: unzip writes many small
# files, which is slow on NFS, and a half-extracted tree must not look done.
if [ ! -f "$DATA/omnisql/.complete" ]; then
  t0=$SECONDS
  work=/tmp/sar-unzip-$SLURM_JOB_ID
  rm -rf "$work" && mkdir -p "$work"
  # Skipped: train_synsql.json (23 GB, 2.5M questions; SkyRL-SQL-653 is the
  # subset we train on), the db_contents_index trees (~290k small files of a
  # value-retrieval index we do not use), and the benchmarks we do not run.
  unzip -q "$DATA/hf/OmniSQL-datasets/data.zip" -d "$work" \
    -x "data/train_synsql.json" "*/db_contents_index/*" "data/temp_db_index/*" \
    "data/spider2_sqlite/*" "data/EHRSQL/*" "data/sciencebenchmark/*" "*/__MACOSX/*" "*/.git/*" "*.DS_Store"
  rm -rf "$DATA/omnisql.partial"
  mv "$work" "$DATA/omnisql.partial" 2>/dev/null || cp -r "$work" "$DATA/omnisql.partial"
  rm -rf "$work" "$DATA/omnisql"
  mv "$DATA/omnisql.partial" "$DATA/omnisql"
  touch "$DATA/omnisql/.complete"
  log "unzipped in $((SECONDS - t0))s"
fi
find "$DATA/omnisql" -maxdepth 3 -type d | head -40 >&2
du -sh "$DATA"/omnisql/*/* 2>/dev/null | head -40 >&2
