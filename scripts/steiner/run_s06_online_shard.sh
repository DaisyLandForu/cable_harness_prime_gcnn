#!/usr/bin/env bash
# Foreground entrypoint for one audited S06 custom-job shard.

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    printf 'usage: %s SHARD_INDEX [main|trace]\n' "$0" >&2
    exit 64
fi

readonly SHARD_INDEX="$1"
readonly PHASE="${2:-main}"
if [[ ! "${SHARD_INDEX}" =~ ^[0-5]$ ]]; then
    printf 'SHARD_INDEX must be one of 0,1,2,3,4,5\n' >&2
    exit 64
fi
if [[ "${PHASE}" != "main" && "${PHASE}" != "trace" ]]; then
    printf 'phase must be main or trace\n' >&2
    exit 64
fi

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
readonly RUN_ROOT="${REPO}/results/steiner/raw/s06/s06-il-online-v1"
readonly LOG_DIR="${REPO}/results/steiner/logs/s06"
readonly LOG_FILE="${LOG_DIR}/${PHASE}-shard-${SHARD_INDEX}.log"
readonly LOCK_FILE="${RUN_ROOT}/locks/${PHASE}-shard-${SHARD_INDEX}.lock"

mkdir -p "${LOG_DIR}" "${RUN_ROOT}/locks"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    printf 'S06 %s shard %s is already running\n' "${PHASE}" "${SHARD_INDEX}" >&2
    exit 75
fi

cd "${REPO}"
scripts/steiner/run_with_scip804.sh --python scripts/steiner/run_s06_online.py \
    --phase "${PHASE}" \
    --shard-index "${SHARD_INDEX}" \
    --shard-count 6 \
    --max-workers 6 \
    2>&1 | tee -a "${LOG_FILE}"
