#!/usr/bin/env bash
# Run one disjoint S05 pilot seed shard in a scheduler-managed one-GPU job.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
readonly REPO_ROOT
readonly CONFIG="${S05_CONFIG:-configs/steiner/experiments/s05_teacher_il_pilot_v3.yml}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
readonly CUBLAS_WORKSPACE_CONFIG

[[ $# -ge 1 ]] || {
    printf 'usage: %s SEED [SEED ...]\n' "$0" >&2
    printf 'registered pilot seeds: 101 202 303\n' >&2
    exit 64
}

declare -A seen=()
training_args=()
seed_slug=""
for seed in "$@"; do
    case "$seed" in
        101|202|303) ;;
        *)
            printf 'seed %s is not a registered S05 pilot seed\n' "$seed" >&2
            exit 64
            ;;
    esac
    [[ -z "${seen[$seed]:-}" ]] || {
        printf 'duplicate seed: %s\n' "$seed" >&2
        exit 64
    }
    seen[$seed]=1
    training_args+=(--training-seed "$seed")
    seed_slug="${seed_slug:+${seed_slug}-}${seed}"
done

config_slug="$(basename -- "$CONFIG" .yml)"
readonly config_slug
readonly PREFLIGHT_PATH="${REPO_ROOT}/results/steiner/raw/s05/gpu_preflight-${config_slug}-seeds-${seed_slug}.json"
cd "$REPO_ROOT"
scripts/steiner/run_with_scip804.sh --python \
    scripts/steiner/check_s05_gpu.py --config "$CONFIG" --output "$PREFLIGHT_PATH"
scripts/steiner/run_with_scip804.sh --python \
    scripts/steiner/train_s05_il.py --config "$CONFIG" "${training_args[@]}"
