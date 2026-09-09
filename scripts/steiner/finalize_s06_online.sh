#!/usr/bin/env bash
# Verify both six-shard waves and create the only formal S06 aggregate summary.

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
readonly RUN_ROOT="${REPO}/results/steiner/raw/s06/s06-il-online-v1"
readonly LOG_DIR="${REPO}/results/steiner/logs/s06"
readonly LOG_FILE="${LOG_DIR}/aggregate.log"

mkdir -p "${LOG_DIR}" "${RUN_ROOT}/locks"
cd "${REPO}"
scripts/steiner/run_with_scip804.sh --python scripts/steiner/run_s06_online.py \
    --aggregate-only --max-workers 6 2>&1 | tee -a "${LOG_FILE}"
