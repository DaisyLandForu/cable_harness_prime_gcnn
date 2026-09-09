#!/usr/bin/env bash
# Start/resume one audited S06 shard in a detached tmux session.

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

readonly SESSION="steiner-s06-${PHASE}-shard-${SHARD_INDEX}"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly REPO="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
readonly LOG_DIR="${REPO}/results/steiner/logs/s06"
readonly LOG_FILE="${LOG_DIR}/${PHASE}-shard-${SHARD_INDEX}.tmux.log"

mkdir -p "${LOG_DIR}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    printf 'session already exists: %s\n' "${SESSION}"
    tmux attach-session -t "${SESSION}"
    exit 0
fi

tmux new-session -d -s "${SESSION}" -c "${REPO}" \
    "exec scripts/steiner/run_s06_online_shard.sh '${SHARD_INDEX}' '${PHASE}' 2>&1 | tee -a '${LOG_FILE}'"
printf 'started %s\nattach: tmux attach -t %s\nlog: tail -f %s\n' \
    "${SESSION}" "${SESSION}" "${LOG_FILE}"
