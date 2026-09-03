#!/usr/bin/env bash
# Start or report the resumable S03 probe in a detached tmux session.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
readonly REPO_ROOT
readonly LOCKED_CONDA_PREFIX="/home/duweiyue25/conda/envs/rl4scip"

session="${1:-steiner-s03}"
workers="${2:-6}"
[[ "$session" =~ ^[A-Za-z0-9._-]+$ ]] || {
    printf 'invalid tmux session name: %s\n' "$session" >&2
    exit 64
}
[[ "$workers" =~ ^[1-6]$ ]] || {
    printf 'workers must be in 1..6\n' >&2
    exit 64
}
command -v tmux >/dev/null || {
    printf 'tmux is not installed\n' >&2
    exit 69
}

if tmux has-session -t "$session" 2>/dev/null; then
    printf 'S03 session already exists: %s\n' "$session"
    printf 'Attach: tmux attach -t %s\n' "$session"
    exit 0
fi

readonly RUN_DIR="${REPO_ROOT}/results/steiner/raw/s03/s03-branchability-pilot-v1"
readonly LOG_PATH="${RUN_DIR}/tmux.log"
mkdir -p -- "$RUN_DIR"

readonly RUN_COMMAND="set -o pipefail; cd '${REPO_ROOT}' && CONDA_PREFIX='${LOCKED_CONDA_PREFIX}' make steiner-s03-probe && scripts/steiner/run_with_scip804.sh --python scripts/steiner/run_s03_branchability.py --max-workers '${workers}' 2>&1 | tee -a '${LOG_PATH}'; code=\$?; printf 'S03_EXIT_CODE=%s\\n' \"\$code\"; exec \${SHELL:-/bin/bash}"

tmux new-session -d -s "$session" -c "$REPO_ROOT" "$RUN_COMMAND"
printf 'Started S03 in tmux session: %s\n' "$session"
printf 'Attach: tmux attach -t %s\n' "$session"
printf 'Log: %s\n' "$LOG_PATH"
