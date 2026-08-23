#!/usr/bin/env bash
# Phase A-extend: λ scan + more instances + small-instance gating.
set -eo pipefail
PROJ="${PROJ:-/data/hanchengcheng/hcc_1/du/cable_harness_prim_gcnn}"
ENV_DIR="${ENV_DIR:-/data/hanchengcheng/envs/rl4scip}"
CFG="${CFG:-$PROJ/configs/experiments/phaseA_extend.json}"
GPUS="${GPUS:-1,2,4,5}"
WORKERS="${WORKERS:-4}"
EXPECTED_RUNS="${EXPECTED_RUNS:-60}"
LOG_DIR="$PROJ/results/phaseA_extend/logs"
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
echo "[$(date '+%F %T')] Phase A-extend eval start (jobs=$EXPECTED_RUNS)"

python scripts/run_final_experiments.py --config "$CFG" --workers "$WORKERS" --resume \
  2>&1 | tee -a "$LOG_DIR/runner.log"

python scripts/validate_final_results.py \
  --input results/phaseA_extend/raw_results.csv \
  --expected-runs "$EXPECTED_RUNS" \
  --output results/phaseA_extend/validation.json \
  2>&1 | tee "$LOG_DIR/validation.log" || true

python scripts/analyze_final_results.py \
  --input results/phaseA_extend/raw_results.csv \
  --output-dir results/phaseA_extend \
  2>&1 | tee "$LOG_DIR/analysis.log" || true

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

rows=list(csv.DictReader(open('results/phaseA_extend/raw_results.csv')))
by=defaultdict(list)
for r in rows:
    by[r['method']].append(r)
report={'n':len(rows),'methods':{},'paired_vs_gcnn':{},'paired_vs_default':{},'per_instance':{}}
for m, rs in sorted(by.items()):
    report['methods'][m]={
        'solved': sum(1 for r in rs if r['status']=='optimal'),
        'n': len(rs),
        'wall_mean': mean([f(r['wall_time']) for r in rs]),
        'nodes_mean': mean([f(r['nodes']) for r in rs]),
        'gap_mean': mean([f(r['gap']) for r in rs]),
    }

pairs=defaultdict(dict)
for r in rows:
    pairs[(r['instance_id'], int(r['seed']))][r['method']]=r

refs=['rl-gcnn-prim-l025','rl-gcnn-prim-l05','rl-gcnn-prim-l10','rl-gcnn-prim-gated']
for meth in refs:
    wall_g=[]; node_g=[]; wall_d=[]; node_d=[]
    for _, m in pairs.items():
        if meth not in m or 'rl-gcnn' not in m or 'default' not in m:
            continue
        if m[meth]['status']!='optimal' or m['rl-gcnn']['status']!='optimal':
            continue
        wall_g.append(f(m['rl-gcnn']['wall_time'])/f(m[meth]['wall_time']))
        node_g.append(max(f(m['rl-gcnn']['nodes']),1.0)/max(f(m[meth]['nodes']),1.0))
        wall_d.append(f(m['default']['wall_time'])/f(m[meth]['wall_time']))
        node_d.append(max(f(m['default']['nodes']),1.0)/max(f(m[meth]['nodes']),1.0))
    report['paired_vs_gcnn'][meth]={
        'geom_wall_speedup': geom(wall_g),
        'geom_node_ratio_gcnn_over_meth': geom(node_g),
        'n': len(wall_g),
    }
    report['paired_vs_default'][meth]={
        'geom_wall_speedup': geom(wall_d),
        'geom_node_ratio_default_over_meth': geom(node_d),
        'n': len(wall_d),
    }

inst=defaultdict(list)
for r in rows:
    inst[(r['instance_id'], r['method'])].append(r)
for (inst_id, method), rs in sorted(inst.items()):
    report['per_instance'][f'{inst_id}/{method}']={
        'solved': sum(1 for r in rs if r['status']=='optimal'),
        'wall_mean': mean([f(r['wall_time']) for r in rs]),
        'nodes_mean': mean([f(r['nodes']) for r in rs]),
    }

Path('results/outputs').mkdir(parents=True, exist_ok=True)
text=json.dumps(report, indent=2)+'\n'
Path('results/phaseA_extend/phaseA_extend_summary.json').write_text(text)
Path('results/outputs/phaseA_extend_summary.json').write_text(text)
print(text)
PY

echo "[$(date '+%F %T')] Phase A-extend DONE"
