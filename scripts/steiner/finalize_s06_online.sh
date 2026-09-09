#!/usr/bin/env bash
# Verify both six-shard waves and create the only formal S06 aggregate summary.

set -euo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
readonly LOG_DIR="${REPO}/results/steiner/logs/s06"
readonly LOG_FILE="${LOG_DIR}/aggregate.log"

mkdir -p "${LOG_DIR}"
cd "${REPO}"
scripts/steiner/run_with_scip804.sh --python scripts/steiner/run_s06_online.py \
    --aggregate-only --max-workers 6 2>&1 | tee -a "${LOG_FILE}"
