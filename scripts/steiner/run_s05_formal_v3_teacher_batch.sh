#!/usr/bin/env bash
# Foreground entry for the audited S05 formal-v3 CPU teacher collection.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
readonly REPO_ROOT
workers="${1:-6}"
[[ "$workers" =~ ^[1-6]$ ]] || { printf 'workers must be in 1..6\n' >&2; exit 64; }

cd "$REPO_ROOT"
exec scripts/steiner/run_with_scip804.sh --python \
    scripts/steiner/collect_s05_formal_v3_teacher.py --max-workers "$workers"
