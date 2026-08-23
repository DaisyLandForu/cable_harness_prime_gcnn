#!/usr/bin/env bash
set -eo pipefail
PROJ="${PROJ:-/data/hanchengcheng/hcc_1/du/cable_harness_prim_gcnn}"
ENV_DIR="${ENV_DIR:-/data/hanchengcheng/envs/rl4scip}"
CFG="${CFG:-$PROJ/configs/rl/c1_ranking_pilot.yaml}"
WORKERS="${WORKERS:-4}"

set +u
source /data/hanchengcheng/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_DIR"
set -u

export SCIPLIB="$PROJ/artifacts/environment/phase4/scip804_prefix/lib"
export TORCH_LIB="$ENV_DIR/lib/python3.11/site-packages/torch/lib"
export LD_LIBRARY_PATH="$SCIPLIB:$TORCH_LIB:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PROJ/python${PYTHONPATH:+:$PYTHONPATH}"
export CFG

cd "$PROJ"
OUT_DIR=$(python - <<'PY'
import os
import yaml
from pathlib import Path
raw = yaml.safe_load(Path(os.environ["CFG"]).read_text())
print(raw["output_dir"])
PY
)
mkdir -p results/c1_dataset/logs "$OUT_DIR"

echo "[$(date '+%F %T')] C1 pilot collect start out=$OUT_DIR teacher via config"
python scripts/collect_c1_branch_ranking.py --config "$CFG" --workers "$WORKERS" --resume \
  2>&1 | tee -a results/c1_dataset/logs/pilot_collect.log

python scripts/analyze_c1_dataset.py \
  --dataset-dir "$OUT_DIR" \
  2>&1 | tee results/c1_dataset/logs/pilot_analyze.log

mkdir -p results/c1_dataset
cp -f "$OUT_DIR/DATASET_REPORT.md" results/c1_dataset/DATASET_REPORT_PILOT.md
cp -f "$OUT_DIR/dataset_stats.json" results/c1_dataset/dataset_stats_pilot.json
echo "[$(date '+%F %T')] C1 pilot DONE — require gate_label_quality=PASS before wave1"
