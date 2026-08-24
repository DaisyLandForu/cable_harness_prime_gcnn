#!/usr/bin/env bash
# Phase B: 4 independent seeds on separate GPUs. DualPool logical batch is
# always 16 (not batch_size=64). This script is not approval for formal 4-card R1.
set -eo pipefail
PROJ="${PROJ:-/data/hanchengcheng/hcc_1/du/cable_harness_prim_gcnn}"
ENV_DIR="${ENV_DIR:-/data/hanchengcheng/envs/rl4scip}"
BASE_CFG="${BASE_CFG:-$PROJ/configs/rl/gcnn_prim_feat_pilot.yaml}"
GPUS=(${GPUS:-1 2 4 5})
SEEDS=(${SEEDS:-0 1 2 3})
OUT_ROOT="${OUT_ROOT:-$PROJ/artifacts/models/gcnn_prim_feat}"
LOG_DIR="$PROJ/logs/phaseB_train_4gpu"
mkdir -p "$LOG_DIR" "$OUT_ROOT" "$PROJ/configs/rl/generated"

set +u
source /data/hanchengcheng/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_DIR"
set -u

export SCIPLIB="$PROJ/artifacts/environment/phase4/scip804_prefix/lib"
export TORCH_LIB="$ENV_DIR/lib/python3.11/site-packages/torch/lib"
export LD_LIBRARY_PATH="$SCIPLIB:$TORCH_LIB:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$PROJ/python${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cd "$PROJ"
echo "[$(date '+%F %T')] Phase B 4-GPU train start gpus=${GPUS[*]} seeds=${SEEDS[*]}"

pids=()
for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  gpu="${GPUS[$i]}"
  cfg="$PROJ/configs/rl/generated/gcnn_prim_feat_pilot_seed${seed}.yaml"
  out="$OUT_ROOT/seed${seed}"
  python - <<PY
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path("$BASE_CFG").read_text())
if int(cfg.get("optimization", {}).get("batch_size", -1)) != 16:
    raise SystemExit("stale DualPool batch_size: must be 16")
cfg["seed"] = int("$seed")
cfg["run_name"] = f"bipartite_gcnn_prim_feat_pilot_seed{int('$seed')}"
cfg["output_dir"] = "$out"
cfg["environment"]["seed"] = int("$seed")
Path("$cfg").write_text(yaml.safe_dump(cfg, sort_keys=False))
print("wrote", "$cfg", "->", "$out")
PY
  echo "[$(date '+%F %T')] launch seed=$seed gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/train_gcnn.py --config "$cfg" \
    > "$LOG_DIR/seed${seed}.log" 2>&1 &
  pids+=($!)
done

fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[$(date '+%F %T')] seed=${SEEDS[$i]} OK"
  else
    echo "[$(date '+%F %T')] seed=${SEEDS[$i]} FAILED rc=$?"
    fail=1
  fi
done

python - <<'PY'
import json, csv
from pathlib import Path

root = Path("artifacts/models/gcnn_prim_feat")
rows = []
for seed_dir in sorted(root.glob("seed*")):
    hist = seed_dir / "training_history.csv"
    best_pt = seed_dir / "best_model_scripted.pt"
    if not best_pt.is_file():
        rows.append({"seed": seed_dir.name, "status": "missing"})
        continue
    best_val = None
    if hist.is_file():
        for r in csv.DictReader(hist.open()):
            if r.get("event") != "validation":
                continue
            raw = r.get("validation_nodes", "")
            if raw in ("", None):
                continue
            try:
                val = float(raw)
            except ValueError:
                continue
            best_val = val if best_val is None else min(best_val, val)
    rows.append({
        "seed": seed_dir.name.replace("seed", ""),
        "status": "ok",
        "best_validation_nodes": best_val,
        "model": str(best_pt),
    })

ok = [r for r in rows if r["status"] == "ok" and r["best_validation_nodes"] is not None]
ok.sort(key=lambda r: r["best_validation_nodes"])
summary = {"seeds": rows, "best_seed": ok[0]["seed"] if ok else None}
if ok:
    best_dir = root / f"seed{ok[0]['seed']}"
    # promote best to canonical paths used by phaseB eval
    for name in (
        "best_model.pt",
        "best_model_scripted.pt",
        "last_model.pt",
        "last_model_scripted.pt",
        "feature_schema.json",
        "config.yaml",
        "normalization.json",
        "parity_observation.npz",
        "training_history.csv",
    ):
        src = best_dir / name
        dst = root / name
        if src.is_file():
            dst.write_bytes(src.read_bytes())
    summary["promoted_from"] = str(best_dir)

(root / "seed_ranking.json").write_text(json.dumps(summary, indent=2) + "\n")
Path("results/outputs").mkdir(parents=True, exist_ok=True)
Path("results/outputs/phaseB_seed_ranking.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "[$(date '+%F %T')] Phase B 4-GPU train DONE fail=$fail"
exit "$fail"
