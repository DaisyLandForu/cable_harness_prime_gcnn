#!/usr/bin/env bash
# Start the S05 CUDA imitation pilot after teacher collection completes.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
readonly REPO_ROOT

session="${1:-steiner-s05-train}"
gpu="${2:-0}"
[[ "$session" =~ ^[A-Za-z0-9._-]+$ ]] || {
    printf 'invalid tmux session name: %s\n' "$session" >&2
    exit 64
}
[[ "$gpu" =~ ^[0-9]+$ ]] || {
    printf 'GPU index must be a non-negative integer\n' >&2
    exit 64
}
command -v tmux >/dev/null || {
    printf 'tmux is not installed\n' >&2
    exit 69
}
command -v nvidia-smi >/dev/null || {
    printf 'nvidia-smi is unavailable; do not start S05 training on this host\n' >&2
    exit 69
}
if tmux has-session -t "$session" 2>/dev/null; then
    printf 'S05 training session already exists: %s\n' "$session"
    printf 'Attach: tmux attach -t %s\n' "$session"
    exit 0
fi

readonly RUN_DIR="${REPO_ROOT}/results/steiner/s05/s05-teacher-il-pilot-v1"
readonly LOG_PATH="${RUN_DIR}/tmux-train.log"
mkdir -p -- "$RUN_DIR"
readonly RUN_COMMAND="set -o pipefail; cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES='${gpu}' scripts/steiner/run_with_scip804.sh --python scripts/steiner/check_s05_gpu.py && CUDA_VISIBLE_DEVICES='${gpu}' scripts/steiner/run_with_scip804.sh --python scripts/steiner/train_s05_il.py 2>&1 | tee -a '${LOG_PATH}'; code=\$?; printf 'S05_TRAIN_EXIT_CODE=%s\n' \"\$code\"; exec \${SHELL:-/bin/bash}"

tmux new-session -d -s "$session" -c "$REPO_ROOT" "$RUN_COMMAND"
printf 'Started S05 CUDA pilot: %s (physical GPU %s)\n' "$session" "$gpu"
printf 'Attach: tmux attach -t %s\n' "$session"
printf 'Log: %s\n' "$LOG_PATH"
