#!/usr/bin/env bash
# Foreground entry for one scheduler-managed S05 formal CPU teacher job.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
readonly REPO_ROOT
workers="${1:-6}"
[[ "$workers" =~ ^[1-6]$ ]] || { printf 'workers must be in 1..6\n' >&2; exit 64; }

cd "$REPO_ROOT"
exec scripts/steiner/run_with_scip804.sh --python \
    scripts/steiner/collect_s05_formal_teacher.py \
    --config configs/steiner/experiments/s05_teacher_il_formal_v1.yml \
    --max-workers "$workers"
