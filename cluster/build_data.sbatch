#!/bin/bash
#SBATCH --job-name=sar-build-data
#SBATCH --partition=short
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/sar/logs/%x_%j.out
#
# Builds the processed parquet files (sar.data.build) from the downloads.
# Runs every gold query once to validate the data, so it reads the databases
# from node-local disk.
#   sbatch /scratch/$USER/sar/code/cluster/build_data.sbatch
set -euo pipefail
source "${SAR_CODE:-/scratch/$USER/sar/code}/cluster/common.sh"
stage_env
OMNI=$(stage_dbs omnisql)
# build.py reads the benchmark json files and databases under data/omnisql/data;
# point it at the node-local copy through a scratch data root.
root=/tmp/sar-build-$SLURM_JOB_ID
mkdir -p "$root" && ln -sfn "$DATA/hf" "$root/hf" && ln -sfn "$OMNI" "$root/omnisql"
trap 'rm -rf "$root"' EXIT
python -P -m sar.data.build --data-root "$root" --out "$DATA/processed/sql" "$@"
cat "$DATA/processed/sql/manifest.json" >&2
