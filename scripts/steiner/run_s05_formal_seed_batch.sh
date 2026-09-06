#!/usr/bin/env bash
# Foreground entry for one scheduler-managed one-V100 formal seed job.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
readonly REPO_ROOT
[[ $# -eq 1 ]] || { printf 'usage: %s SEED\n' "$0" >&2; exit 64; }
seed="$1"
case "$seed" in 101|202|303|404|505) ;; *) printf 'invalid formal seed: %s\n' "$seed" >&2; exit 64;; esac
export CUBLAS_WORKSPACE_CONFIG=:4096:8
readonly CUBLAS_WORKSPACE_CONFIG
readonly CONFIG="configs/steiner/experiments/s05_teacher_il_formal_v1.yml"
readonly PREFLIGHT="${REPO_ROOT}/results/steiner/raw/s05/gpu_preflight-s05-formal-seed-${seed}.json"

cd "$REPO_ROOT"
scripts/steiner/run_with_scip804.sh --python \
    scripts/steiner/check_s05_gpu.py --config "$CONFIG" --output "$PREFLIGHT"
exec scripts/steiner/run_with_scip804.sh --python \
    scripts/steiner/train_s05_formal_il.py --config "$CONFIG" --training-seed "$seed"
