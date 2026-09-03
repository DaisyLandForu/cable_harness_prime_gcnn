#!/usr/bin/env bash
# Start the audited, resumable CPU teacher pilot in a detached tmux session.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
readonly REPO_ROOT

session="${1:-steiner-s05-teacher}"
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
    printf 'S05 teacher session already exists: %s\n' "$session"
    printf 'Attach: tmux attach -t %s\n' "$session"
    exit 0
fi

readonly RUN_DIR="${REPO_ROOT}/results/steiner/raw/s05/s05-teacher-il-pilot-v1"
readonly LOG_PATH="${RUN_DIR}/tmux.log"
mkdir -p -- "$RUN_DIR"
readonly RUN_COMMAND="set -o pipefail; cd '${REPO_ROOT}' && scripts/steiner/run_with_scip804.sh --python scripts/steiner/collect_s05_teacher.py --max-workers '${workers}' 2>&1 | tee -a '${LOG_PATH}'; code=\$?; printf 'S05_TEACHER_EXIT_CODE=%s\n' \"\$code\"; exec \${SHELL:-/bin/bash}"

tmux new-session -d -s "$session" -c "$REPO_ROOT" "$RUN_COMMAND"
printf 'Started S05 teacher pilot: %s\n' "$session"
printf 'Attach: tmux attach -t %s\n' "$session"
printf 'Log: %s\n' "$LOG_PATH"
