#!/usr/bin/env bash
# C1.1 Teacher Repair: audit pilot_v2 + Ecole live diag + native SCIP SB probe.
# DO NOT start 30k wave1 from this script.
set -eo pipefail
PROJ="${PROJ:-/data/hanchengcheng/hcc_1/du/cable_harness_prim_gcnn}"
ENV_DIR="${ENV_DIR:-/data/hanchengcheng/envs/rl4scip}"

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
mkdir -p results/c1_dataset/teacher_repair/logs docs

echo "[$(date '+%F %T')] C1.1 offline+live Ecole audit"
python scripts/analyze_c1_teacher_repair.py \
  --dataset-dir artifacts/datasets/c1_branch_ranking_pilot_v2 \
  --output-dir results/c1_dataset/teacher_repair \
  2>&1 | tee results/c1_dataset/teacher_repair/logs/analyze.log

echo "[$(date '+%F %T')] build native SB probe"
make sb_native_probe CXX=/usr/bin/g++ 2>&1 | tee results/c1_dataset/teacher_repair/logs/build_probe.log

echo "[$(date '+%F %T')] native probe syn_medium_s101"
./build/sb_native_probe \
  --instance data/instances/train/syn_medium_s101.cip \
  --seed 0 --max-cands 64 --max-probes 3 --time-limit 180 --node-limit 30 \
  2>&1 | tee results/c1_dataset/teacher_repair/logs/native_syn_medium.log

echo "[$(date '+%F %T')] native probe real_06 (may be slower)"
./build/sb_native_probe \
  --instance data/instances/train/real_06.cip \
  --seed 0 --max-cands 32 --max-probes 2 --time-limit 240 --node-limit 20 \
  2>&1 | tee results/c1_dataset/teacher_repair/logs/native_real_06.log

python - <<'PY'
from pathlib import Path
import re, json
out = Path('results/c1_dataset/teacher_repair')
summary = {'native_probes': {}}
for name in ['native_syn_medium.log', 'native_real_06.log']:
    text = (out / 'logs' / name).read_text(errors='replace')
    informative = re.findall(r'native_sb_informative=(yes|no)', text)
    deltas = [int(x) for x in re.findall(r'lp_iters_delta=(-?\d+)', text)]
    sb_deltas = [int(x) for x in re.findall(r'sb_lp_iters_delta=(-?\d+)', text)]
    summary['native_probes'][name] = {
        'informative_flags': informative,
        'lp_iters_deltas': deltas,
        'sb_lp_iters_deltas': sb_deltas,
        'any_informative': any(x == 'yes' for x in informative),
        'any_positive_lp_delta': any(d > 0 for d in deltas),
        'any_positive_sb_lp_delta': any(d > 0 for d in sb_deltas),
    }
(out / 'native_probe_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
print('Reports:')
print(' ', out / 'C1_TEACHER_REPAIR_REPORT.md')
print(' ', 'docs/C1_TEACHER_REPAIR_REPORT.md')
print('DO NOT start wave1 unless sb_valid_state_ratio>=0.6 AFTER native collector is wired.')
PY

echo "[$(date '+%F %T')] C1.1 audit/probe DONE"
