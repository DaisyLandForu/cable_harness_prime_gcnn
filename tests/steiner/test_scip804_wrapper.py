"""Integration checks for the canonical SCIP 8.0.4 launcher."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess

import yaml


REPO = Path(__file__).resolve().parents[2]
WRAPPER = REPO / "scripts/steiner/run_with_scip804.sh"
ENVIRONMENT = REPO / "configs/steiner/environment.lock.yml"


def run_wrapper(*arguments: str, preload: str | None = None) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = "/usr/lib:/usr/local/lib"
    if preload is None:
        environment.pop("LD_PRELOAD", None)
    else:
        environment["LD_PRELOAD"] = preload
    return subprocess.run(
        [str(WRAPPER), *arguments],
        cwd=REPO,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_wrapper_is_executable_and_constants_match_environment_lock():
    assert os.access(WRAPPER, os.X_OK)
    text = WRAPPER.read_text(encoding="utf-8")
    lock = yaml.safe_load(ENVIRONMENT.read_text(encoding="utf-8"))
    scip = lock["solver_stack"]["scip"]
    packages = lock["python_environment"]["packages"]
    expected_literals = (
        lock["decision"]["stack_id"],
        scip["version"],
        scip["executable_sha256"],
        scip["shared_library_sha256"],
        packages["pyscipopt"],
        packages["ecole"],
        lock["runtime_wrapper"]["child_scip_shim_sha256"],
    )
    for value in expected_literals:
        assert value in text
    assert "LD_LIBRARY_PATH=\"$SCIP_LIB_DIR\"" in text
    assert "--python" in text and "--scip" in text


def test_verify_only_ignores_conflicting_library_path_and_proves_locked_stack():
    result = run_wrapper("--verify-only")
    assert result.returncode == 0, result.stderr
    assert "stack=scip804-ecole081-pyscipopt430" in result.stderr
    assert "scip=8.0.4" in result.stderr
    assert "pyscipopt=4.3.0" in result.stderr
    assert "ecole=0.8.1" in result.stderr
    assert "9.2.2" not in result.stdout + result.stderr


def test_python_mode_exposes_only_the_locked_solver_stack():
    result = run_wrapper(
        "--python",
        "-c",
        (
            "import os; from pyscipopt import Model; "
            "print(Model().version()); "
            "print(os.environ['STEINER_SOLVER_STACK_ID']); "
            "print(os.environ['LD_LIBRARY_PATH'])"
        ),
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == "8.04"
    assert lines[1] == "scip804-ecole081-pyscipopt430"
    assert lines[2] == str(REPO / "artifacts/environment/phase4/scip804_prefix/lib")


def test_python_child_process_cannot_resolve_system_scip():
    result = run_wrapper(
        "--python",
        "-c",
        (
            "import shutil, subprocess; "
            "print(shutil.which('scip')); "
            "p=subprocess.run(['scip', '--version'], text=True, capture_output=True); "
            "print(p.returncode); print(p.stdout.splitlines()[0])"
        ),
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0] == str(REPO / "scripts/steiner/pinned-bin/scip")
    assert lines[1] == "0"
    assert lines[2].startswith("SCIP version 8.0.4 ")
    assert "9.2.2" not in result.stdout + result.stderr


def test_scip_mode_uses_804_despite_nonexecutable_artifact():
    result = run_wrapper("--scip", "--version")
    assert result.returncode == 0, result.stderr
    assert re.search(r"^SCIP version 8\.0\.4 ", result.stdout)
    assert "SCIP version 9.2.2" not in result.stdout


def test_wrapper_rejects_preload_and_arbitrary_command_mode():
    rejected_preload = run_wrapper("--verify-only", preload="/not/a/real/library.so")
    assert rejected_preload.returncode == 64
    assert "LD_PRELOAD must be empty" in rejected_preload.stderr

    rejected_command = run_wrapper("--", "/usr/bin/scip", "--version")
    assert rejected_command.returncode == 64
    assert "Usage:" in rejected_command.stderr
