#!/usr/bin/env bash
# Long-running C1 wave-1. Only start after pilot looks healthy.
set -eo pipefail
PROJ="${PROJ:-/data/hanchengcheng/hcc_1/du/cable_harness_prim_gcnn}"
ENV_DIR="${ENV_DIR:-/data/hanchengcheng/envs/rl4scip}"
CFG="${CFG:-$PROJ/configs/rl/c1_ranking_wave1.yaml}"
WORKERS="${WORKERS:-8}"

set +u
source /data/hanchengcheng/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_DIR"
set -u

export SCIPLIB="$PROJ/artifacts/environment/phase4/scip804_prefix/lib"
export TORCH_LIB="$ENV_DIR/lib/python3.11/site-packages/torch/lib"
export LD_LIBRARY_PATH="$SCIPLIB:$TORCH_LIB:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PROJ/python${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJ"
mkdir -p results/c1_dataset/logs artifacts/datasets/c1_branch_ranking

echo "[$(date '+%F %T')] C1 wave1 collect start workers=$WORKERS"
python scripts/collect_c1_branch_ranking.py --config "$CFG" --workers "$WORKERS" --resume \
  2>&1 | tee -a results/c1_dataset/logs/wave1_collect.log

python scripts/analyze_c1_dataset.py \
  --dataset-dir artifacts/datasets/c1_branch_ranking \
  2>&1 | tee results/c1_dataset/logs/wave1_analyze.log

cp -f artifacts/datasets/c1_branch_ranking/DATASET_REPORT.md results/c1_dataset/DATASET_REPORT.md
cp -f artifacts/datasets/c1_branch_ranking/dataset_stats.json results/c1_dataset/dataset_stats.json
cp -f artifacts/datasets/c1_branch_ranking/manifest.json results/c1_dataset/manifest.json

python - <<'PY'
import json
from pathlib import Path
stats = json.loads(Path("artifacts/datasets/c1_branch_ranking/dataset_stats.json").read_text())
n = stats["n_samples"]
print(f"C1 samples={n} gate_30k={n>=30000}")
if n < 30000:
    print("GATE FAIL: expand seeds / instances / time_limit and re-run with --resume")
else:
    print("GATE PASS: may proceed to C2 residual ranker design")
PY

echo "[$(date '+%F %T')] C1 wave1 DONE"
