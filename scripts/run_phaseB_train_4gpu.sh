#!/usr/bin/env bash
# Independent single-GPU seeds (no DDP). DualPool logical batch is always 16
# (12 medium + 4 real_02 = 25%, or 16 medium + 0 large). All workers must load
# the same read-only normalization.json. This script is not approval to launch
# six-card formal training.
set -eo pipefail
PROJ="${PROJ:-/data/hanchengcheng/hcc_1/du/cable_harness_prim_gcnn}"
ENV_DIR="${ENV_DIR:-/data/hanchengcheng/envs/rl4scip}"
BASE_CFG="${BASE_CFG:-$PROJ/configs/rl/gcnn_prim_feat_pilot.yaml}"
NORMALIZATION_PATH="${NORMALIZATION_PATH:-$PROJ/results/probes/shared_normalization.json}"
EXPECTED_NORM_SHA="${EXPECTED_NORM_SHA:-62d8ce546167a50d23c79389609a91d88eae351a7f83d76f8f11284eaa31cc24}"
OUT_ROOT="${OUT_ROOT:-$PROJ/artifacts/models/gcnn_prim_feat}"
LOG_DIR="${LOG_DIR:-$PROJ/logs/seed_parallel_train}"

if [[ -z "${GPUS:-}" || -z "${SEEDS:-}" ]]; then
  echo "usage: GPUS='0 1 2' SEEDS='0 1 2' $0" >&2
  echo "refusing to default-launch GPUs; six-card formal training is not approved yet" >&2
  exit 2
fi
GPUS=($GPUS)
SEEDS=($SEEDS)
if [[ ${#GPUS[@]} -ne ${#SEEDS[@]} ]]; then
  echo "GPUS and SEEDS must have the same length" >&2
  exit 2
fi

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
actual_sha="$(sha256sum "$NORMALIZATION_PATH" | awk '{print $1}')"
if [[ "$actual_sha" != "$EXPECTED_NORM_SHA" ]]; then
  echo "normalization SHA mismatch: $actual_sha != $EXPECTED_NORM_SHA" >&2
  exit 2
fi
echo "[$(date '+%F %T')] seed-parallel train start gpus=${GPUS[*]} seeds=${SEEDS[*]} norm_sha=$actual_sha"

pids=()
for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  gpu="${GPUS[$i]}"
  cfg="$PROJ/configs/rl/generated/gcnn_prim_feat_pilot_seed${seed}.yaml"
  out="$OUT_ROOT/seed${seed}"
  python - <<PY
from pathlib import Path
import hashlib
import yaml
cfg = yaml.safe_load(Path("$BASE_CFG").read_text())
opt = cfg.get("optimization", {})
if int(opt.get("batch_size", -1)) != 16:
    raise SystemExit("stale DualPool batch_size: must be 16")
if int(opt.get("medium_count_limit", -1)) != 224 or int(opt.get("large_count_limit", -1)) != 32:
    raise SystemExit("DualPool limits must stay 224 medium + 32 large")
if int(opt.get("replay_capacity", -1)) != 256:
    raise SystemExit("replay_capacity must equal 224+32=256")
norm = Path("$NORMALIZATION_PATH")
sha = hashlib.sha256(norm.read_bytes()).hexdigest()
if sha != "$EXPECTED_NORM_SHA":
    raise SystemExit(f"normalization SHA mismatch: {sha}")
cfg["seed"] = int("$seed")
cfg["run_name"] = f"bipartite_gcnn_prim_feat_pilot_seed{int('$seed')}"
cfg["output_dir"] = "$out"
cfg["environment"]["seed"] = int("$seed")
cfg["normalization_path"] = str(norm)
Path("$cfg").write_text(yaml.safe_dump(cfg, sort_keys=False))
print("wrote", "$cfg", "->", "$out", "norm", sha)
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

python - <<PY
import json, csv
from pathlib import Path

root = Path("$OUT_ROOT")
expected = "$EXPECTED_NORM_SHA"
rows = []
for seed_dir in sorted(root.glob("seed*")):
    hist = seed_dir / "training_history.csv"
    best_pt = seed_dir / "best_model_scripted.pt"
    summary_path = seed_dir / "summary.json"
    sha = None
    if summary_path.is_file():
        sha = json.loads(summary_path.read_text()).get("normalization_sha256")
    if sha and sha != expected:
        raise SystemExit(f"{seed_dir} normalization SHA {sha} != {expected}")
    row = {
        "seed": seed_dir.name.replace("seed", ""),
        "status": "ok" if best_pt.is_file() else "missing",
        "model": str(best_pt) if best_pt.is_file() else "",
        "normalization_sha256": sha,
    }
    rows.append(row)

summary = {
    "seeds": rows,
    "best_seed": None,
    "normalization_sha256": expected,
    "selection_rule": "pdi_gap_composite_not_implemented",
    "note": "do not promote by minimum validation nodes; wait for frozen PDI/gap ranking",
}
(root / "seed_ranking.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "[$(date '+%F %T')] seed-parallel train DONE fail=$fail"
exit "$fail"
