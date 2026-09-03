#!/usr/bin/env bash
# Canonical launcher for the frozen Steiner SCIP 8.0.4 Python/CLI stack.

set -euo pipefail

readonly STACK_ID="scip804-ecole081-pyscipopt430"
readonly EXPECTED_SCIP_VERSION="8.0.4"
readonly EXPECTED_SCIP_PY_VERSION="8.04"
readonly EXPECTED_PYSCIPOPT_VERSION="4.3.0"
readonly EXPECTED_ECOLE_VERSION="0.8.1"
readonly EXPECTED_SCIP_BIN_SHA256="fa5f8b84195cdb559e7810b9add0c006657fb185730e28945c77365422d9e45d"
readonly EXPECTED_LIBSCIP_SHA256="5524e92770f25c1baa6c1469528a71fadcc25aeb0db585b0938030ec857281ee"
readonly EXPECTED_SCIP_CHILD_SHIM_SHA256="09d1e4d1b10a512738790c8727282ad1b524b075ef5606d18ee62ff8ecb1caab"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
readonly REPO_ROOT
readonly SCIP_PREFIX="${REPO_ROOT}/artifacts/environment/phase4/scip804_prefix"
readonly SCIP_BIN="${SCIP_PREFIX}/bin/scip"
readonly SCIP_LIB_DIR="${SCIP_PREFIX}/lib"
readonly LIBSCIP_SONAME="${SCIP_LIB_DIR}/libscip.so.8.0"
readonly PINNED_BIN_DIR="${SCRIPT_DIR}/pinned-bin"
readonly SCIP_CHILD_SHIM="${PINNED_BIN_DIR}/scip"
readonly LOCKED_PYTHON="/home/duweiyue25/conda/envs/rl4scip/bin/python"
readonly ELF_LOADER="/lib64/ld-linux-x86-64.so.2"

fail() {
    printf 'SCIP 8.0.4 wrapper error: %s\n' "$*" >&2
    exit 64
}

usage() {
    cat >&2 <<'EOF'
Usage:
  scripts/steiner/run_with_scip804.sh --verify-only
  scripts/steiner/run_with_scip804.sh --python PYTHON_ARGS...
  scripts/steiner/run_with_scip804.sh --scip SCIP_ARGS...

The wrapper intentionally has no arbitrary-command mode. Steiner Python and
SCIP commands must enter through one of the pinned modes above.
EOF
    exit 64
}

require_file() {
    [[ -f "$1" ]] || fail "required file is missing: $1"
}

require_sha256() {
    local path="$1"
    local expected="$2"
    local actual
    actual="$(/usr/bin/sha256sum -- "$path" | /usr/bin/awk '{print $1}')"
    [[ "$actual" == "$expected" ]] || fail \
        "checksum mismatch for ${path}: expected=${expected} actual=${actual}"
}

verify_stack() {
    local python_probe
    local cli_output
    local cli_probe

    [[ -z "${LD_PRELOAD:-}" ]] || fail "LD_PRELOAD must be empty for formal runs"
    require_file "$SCIP_BIN"
    require_file "$LIBSCIP_SONAME"
    require_file "$LOCKED_PYTHON"
    require_file "$ELF_LOADER"
    require_file "$SCIP_CHILD_SHIM"
    require_sha256 "$SCIP_BIN" "$EXPECTED_SCIP_BIN_SHA256"
    require_sha256 "$LIBSCIP_SONAME" "$EXPECTED_LIBSCIP_SHA256"
    require_sha256 "$SCIP_CHILD_SHIM" "$EXPECTED_SCIP_CHILD_SHIM_SHA256"

    # Use exactly the frozen prefix. Inherited library directories could contain
    # SCIP 9.x and are deliberately excluded.
    export LD_LIBRARY_PATH="$SCIP_LIB_DIR"
    export STEINER_SOLVER_STACK_ID="$STACK_ID"
    export STEINER_SCIP_VERSION="$EXPECTED_SCIP_VERSION"
    export SCIPOPTDIR="$SCIP_PREFIX"
    export PATH="${PINNED_BIN_DIR}:${PATH}"

    python_probe="$($LOCKED_PYTHON -c \
        'import ecole, pyscipopt; from pyscipopt import Model; print(f"{Model().version()}|{pyscipopt.__version__}|{ecole.__version__}")')" \
        || fail "the locked Python solver stack could not be imported"
    [[ "$python_probe" == "${EXPECTED_SCIP_PY_VERSION}|${EXPECTED_PYSCIPOPT_VERSION}|${EXPECTED_ECOLE_VERSION}" ]] \
        || fail "Python stack mismatch: ${python_probe}"

    cli_output="$($ELF_LOADER "$SCIP_BIN" --version)" \
        || fail "the locked SCIP CLI could not be started through the ELF loader"
    cli_probe="${cli_output%%$'\n'*}"
    [[ "$cli_probe" == "SCIP version ${EXPECTED_SCIP_VERSION} "* ]] \
        || fail "SCIP CLI mismatch: ${cli_probe}"

    printf 'verified stack=%s scip=%s pyscipopt=%s ecole=%s\n' \
        "$STACK_ID" "$EXPECTED_SCIP_VERSION" "$EXPECTED_PYSCIPOPT_VERSION" \
        "$EXPECTED_ECOLE_VERSION" >&2
}

[[ $# -ge 1 ]] || usage
mode="$1"
shift

case "$mode" in
    --verify-only)
        [[ $# -eq 0 ]] || usage
        verify_stack
        ;;
    --python)
        [[ $# -ge 1 ]] || usage
        verify_stack
        exec "$LOCKED_PYTHON" "$@"
        ;;
    --scip)
        verify_stack
        # The checked-in artifact currently lacks its executable bit. Invoking
        # its pinned ELF interpreter avoids mutating that user-owned artifact.
        exec "$ELF_LOADER" "$SCIP_BIN" "$@"
        ;;
    *)
        usage
        ;;
esac
