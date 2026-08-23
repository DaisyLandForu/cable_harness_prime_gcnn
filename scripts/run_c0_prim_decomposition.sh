#!/usr/bin/env bash
# C0.2 Prim decomposition: gcnn / z / root_z / full-prim / topology-only
set -eo pipefail
PROJ="${PROJ:-/data/hanchengcheng/hcc_1/du/cable_harness_prim_gcnn}"
ENV_DIR="${ENV_DIR:-/data/hanchengcheng/envs/rl4scip}"
CFG="${CFG:-$PROJ/configs/experiments/c0_prim_decomposition.json}"
WORKERS="${WORKERS:-4}"
EXPECTED_RUNS="${EXPECTED_RUNS:-30}"
LOG_DIR="$PROJ/results/c0_prim_decomposition/logs"
mkdir -p "$LOG_DIR"

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
echo "[$(date '+%F %T')] C0 Prim decomposition start"

python scripts/run_final_experiments.py --config "$CFG" --workers "$WORKERS" --resume \
  2>&1 | tee -a "$LOG_DIR/runner.log"

python scripts/validate_final_results.py \
  --input results/c0_prim_decomposition/raw_results.csv \
  --expected-runs "$EXPECTED_RUNS" \
  --output results/c0_prim_decomposition/validation.json \
  2>&1 | tee "$LOG_DIR/validation.log" || true

python scripts/analyze_c0_decomposition.py \
  --raw-results results/c0_prim_decomposition/raw_results.csv \
  --output-dir results/c0_prim_decomposition \
  2>&1 | tee "$LOG_DIR/decomp_analysis.log"

python scripts/analyze_c0_audit.py \
  --input-glob 'results/c0_prim_decomposition/raw/**/*.branches.csv' \
  --raw-results results/c0_prim_decomposition/raw_results.csv \
  --output-dir results/c0_audit \
  2>&1 | tee "$LOG_DIR/audit_analysis.log"

echo "[$(date '+%F %T')] C0 Prim decomposition DONE"
echo "Reports:"
echo "  results/c0_prim_decomposition/C0_PRIM_DECOMPOSITION.md"
echo "  results/c0_audit/C0_AUDIT_REPORT.md"
