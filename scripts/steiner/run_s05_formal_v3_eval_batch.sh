#!/usr/bin/env bash
# Foreground entry for one frozen-checkpoint formal-v3 evaluation job.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
readonly REPO_ROOT
[[ $# -eq 1 ]] || { printf 'usage: %s MODEL_SEED\n' "$0" >&2; exit 64; }
seed="$1"
case "$seed" in 101|202|303|404|505) ;; *) printf 'invalid model seed: %s\n' "$seed" >&2; exit 64;; esac
export CUBLAS_WORKSPACE_CONFIG=:4096:8
readonly CUBLAS_WORKSPACE_CONFIG

cd "$REPO_ROOT"
exec scripts/steiner/run_with_scip804.sh --python \
    scripts/steiner/evaluate_s05_formal_v3.py --model-seed "$seed"
