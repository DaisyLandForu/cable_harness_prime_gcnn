#!/usr/bin/env bash
# Detached one-GPU launcher for one audited S05 formal training seed.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
readonly REPO_ROOT
[[ $# -eq 3 ]] || { printf 'usage: %s SESSION GPU_INDEX SEED\n' "$0" >&2; exit 64; }
session="$1"
gpu="$2"
seed="$3"
[[ "$session" =~ ^[A-Za-z0-9._-]+$ ]] || { printf 'invalid tmux session\n' >&2; exit 64; }
[[ "$gpu" =~ ^[0-9]+$ ]] || { printf 'invalid GPU index\n' >&2; exit 64; }
case "$seed" in 101|202|303|404|505) ;; *) printf 'invalid formal seed: %s\n' "$seed" >&2; exit 64;; esac
command -v tmux >/dev/null || { printf 'tmux is not installed\n' >&2; exit 69; }
command -v nvidia-smi >/dev/null || { printf 'nvidia-smi is unavailable\n' >&2; exit 69; }
if tmux has-session -t "$session" 2>/dev/null; then
    printf 'Session already exists: %s\nAttach: tmux attach -t %s\n' "$session" "$session"
    exit 0
fi

readonly RUN_DIR="${REPO_ROOT}/results/steiner/s05/s05-teacher-il-formal-v1"
readonly LOG_PATH="${RUN_DIR}/tmux-seed-${seed}.log"
mkdir -p -- "$RUN_DIR"
readonly RUN_COMMAND="set -o pipefail; cd '${REPO_ROOT}' && export CUDA_VISIBLE_DEVICES='${gpu}' CUBLAS_WORKSPACE_CONFIG=:4096:8 && scripts/steiner/run_s05_formal_seed_batch.sh '${seed}' 2>&1 | tee -a '${LOG_PATH}'; code=\$?; printf 'S05_FORMAL_SEED_${seed}_EXIT_CODE=%s\n' \"\$code\"; exec \${SHELL:-/bin/bash}"
tmux new-session -d -s "$session" -c "$REPO_ROOT" "$RUN_COMMAND"
printf 'Started: %s (physical GPU %s, seed %s)\n' "$session" "$gpu" "$seed"
printf 'Attach: tmux attach -t %s\nLog: %s\n' "$session" "$LOG_PATH"
