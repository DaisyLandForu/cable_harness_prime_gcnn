#!/usr/bin/env bash
# Phase A: compare default / rl-gcnn / rl-gcnn-prim on a compact matrix.
set -eo pipefail
PROJ="${PROJ:-/data/hanchengcheng/hcc_1/du/cable_harness_prim_gcnn}"
ENV_DIR="${ENV_DIR:-/data/hanchengcheng/envs/rl4scip}"
CFG="${CFG:-$PROJ/configs/experiments/phaseA_prim_decode.json}"
GPUS="${GPUS:-1,2,4,5}"
WORKERS="${WORKERS:-4}"
LOG_DIR="$PROJ/results/phaseA_prim_decode/logs"
mkdir -p "$LOG_DIR"

set +u
source /data/hanchengcheng/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_DIR"
set -u

export SCIPLIB="$PROJ/artifacts/environment/phase4/scip804_prefix/lib"
export TORCH_LIB="$ENV_DIR/lib/python3.11/site-packages/torch/lib"
export LD_LIBRARY_PATH="$SCIPLIB:$TORCH_LIB:${LD_LIBRARY_PATH:-}"
export PHASE8_CUDA_DEVICES="$GPUS"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PROJ/python${PYTHONPATH:+:$PYTHONPATH}"

cd "$PROJ"
echo "[$(date '+%F %T')] Phase A prim-decode eval start"

python scripts/run_final_experiments.py --config "$CFG" --workers "$WORKERS" \
  2>&1 | tee "$LOG_DIR/runner.log"

python scripts/validate_final_results.py \
  --input results/phaseA_prim_decode/raw_results.csv \
  --expected-runs 18 \
  --output results/phaseA_prim_decode/validation.json \
  2>&1 | tee "$LOG_DIR/validation.log" || true

python scripts/analyze_final_results.py \
  --input results/phaseA_prim_decode/raw_results.csv \
  --output-dir results/phaseA_prim_decode \
  2>&1 | tee "$LOG_DIR/analysis.log" || true

python - <<'PY'
import csv, json
from pathlib import Path
from collections import defaultdict
import math

def f(x):
    try: return float(x)
    except: return float('nan')
def mean(xs):
    xs=[x for x in xs if x==x]
    return sum(xs)/len(xs) if xs else float('nan')

rows=list(csv.DictReader(open('results/phaseA_prim_decode/raw_results.csv')))
by=defaultdict(list)
for r in rows:
    by[r['method']].append(r)
report={'n':len(rows),'methods':{}}
for m, rs in sorted(by.items()):
    report['methods'][m]={
        'solved': sum(1 for r in rs if r['status']=='optimal'),
        'n': len(rs),
        'wall_mean': mean([f(r['wall_time']) for r in rs]),
        'nodes_mean': mean([f(r['nodes']) for r in rs]),
        'gap_mean': mean([f(r['gap']) for r in rs]),
    }
inst={}
for r in rows:
    key=(r['instance_id'], r['method'])
    inst.setdefault(key, []).append(r)
report['per_instance']={}
for (inst_id, method), rs in sorted(inst.items()):
    report['per_instance'][f'{inst_id}/{method}']={
        'solved': sum(1 for r in rs if r['status']=='optimal'),
        'wall_mean': mean([f(r['wall_time']) for r in rs]),
        'nodes_mean': mean([f(r['nodes']) for r in rs]),
    }
Path('results/outputs').mkdir(parents=True, exist_ok=True)
Path('results/phaseA_prim_decode/phaseA_summary.json').write_text(json.dumps(report, indent=2)+'\n')
Path('results/outputs/phaseA_prim_decode_summary.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
PY

echo "[$(date '+%F %T')] Phase A DONE"
