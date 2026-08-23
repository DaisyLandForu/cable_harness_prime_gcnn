#!/usr/bin/env bash
# Phase B: evaluate Prim-feature GCNN vs baselines.
set -eo pipefail
PROJ="${PROJ:-/data/hanchengcheng/hcc_1/du/cable_harness_prim_gcnn}"
ENV_DIR="${ENV_DIR:-/data/hanchengcheng/envs/rl4scip}"
CFG="${CFG:-$PROJ/configs/experiments/phaseB_prim_feat.json}"
GPUS="${GPUS:-1,2,4,5}"
WORKERS="${WORKERS:-4}"
EXPECTED_RUNS="${EXPECTED_RUNS:-40}"
LOG_DIR="$PROJ/results/phaseB_prim_feat/logs"
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
echo "[$(date '+%F %T')] Phase B prim-feat eval start"

python scripts/run_final_experiments.py --config "$CFG" --workers "$WORKERS" --resume \
  2>&1 | tee -a "$LOG_DIR/runner.log"

python scripts/validate_final_results.py \
  --input results/phaseB_prim_feat/raw_results.csv \
  --expected-runs "$EXPECTED_RUNS" \
  --output results/phaseB_prim_feat/validation.json \
  2>&1 | tee "$LOG_DIR/validation.log" || true

python - <<'PY'
import csv, json, math
from pathlib import Path
from collections import defaultdict

def f(x):
    try: return float(x)
    except: return float('nan')
def mean(xs):
    xs=[x for x in xs if x==x]
    return sum(xs)/len(xs) if xs else float('nan')
def geom(xs):
    xs=[x for x in xs if x==x and x>0]
    return math.exp(sum(math.log(x) for x in xs)/len(xs)) if xs else float('nan')

rows=list(csv.DictReader(open('results/phaseB_prim_feat/raw_results.csv')))
by=defaultdict(list)
for r in rows: by[r['method']].append(r)
report={'n':len(rows),'methods':{},'per_instance':{},'paired_vs_gcnn':{}}
for m, rs in sorted(by.items()):
    report['methods'][m]={
        'solved': sum(1 for r in rs if r['status']=='optimal'),
        'n': len(rs),
        'wall_mean': mean([f(r['wall_time']) for r in rs]),
        'nodes_mean': mean([f(r['nodes']) for r in rs]),
    }
pairs=defaultdict(dict)
for r in rows:
    pairs[(r['instance_id'], int(r['seed']))][r['method']]=r
for meth in sorted({r['method'] for r in rows if r['method']!='rl-gcnn' and r['method']!='default'}):
    walls=[]
    for _, m in pairs.items():
        if meth in m and 'rl-gcnn' in m and m[meth]['status']=='optimal' and m['rl-gcnn']['status']=='optimal':
            walls.append(f(m['rl-gcnn']['wall_time'])/f(m[meth]['wall_time']))
    report['paired_vs_gcnn'][meth]={'geom_wall_speedup': geom(walls), 'n': len(walls)}
inst=defaultdict(list)
for r in rows: inst[(r['instance_id'], r['method'])].append(r)
for (i,m), rs in sorted(inst.items()):
    report['per_instance'][f'{i}/{m}']={
        'solved': sum(1 for r in rs if r['status']=='optimal'),
        'wall_mean': mean([f(r['wall_time']) for r in rs]),
        'nodes_mean': mean([f(r['nodes']) for r in rs]),
    }
Path('results/outputs').mkdir(parents=True, exist_ok=True)
text=json.dumps(report, indent=2)+'\n'
Path('results/phaseB_prim_feat/phaseB_summary.json').write_text(text)
Path('results/outputs/phaseB_prim_feat_summary.json').write_text(text)
print(text)
PY

echo "[$(date '+%F %T')] Phase B DONE"
