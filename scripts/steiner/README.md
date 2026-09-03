# Steiner command-line entry points

Stage-specific CLIs live here and import only from `python/steiner_branching`.
S01 intentionally adds no solver or data command. S02 provides:

- `download_data.py`: PACE Track 1/2 odd development instances only; even/final
  selectors fail before network access.
- `generate_data.py`: deterministic config-driven synthetic `.stp` files and a
  canonical split manifest.
- `build_milp.py`: rooted MCF `.lp` and `problem_meta.json` generation.
- `check_solution.py`: P0 solve plus independent selected-edge validation.
- `lock_final_content.py`: S02 byte-only archive/member hashing for the sealed
  final selectors. It never imports a parser or solver and must not produce a
  result artifact or change `learning_runs_total`.

S03 provides:

- `run_s03_branchability.py`: strict P1 task expansion, fresh-process
  branchability/resource probes, atomic fingerprinted shards, resume, and Gate
  aggregation.
- `run_s03_tmux.sh`: detached 1 → 3 → 6 worker launcher. Re-running it with a
  new session name skips only matching completed shards.

Build the native SCIP 8.0.4 strong-branch signal probe before an S03 run:

```text
CONDA_PREFIX=/home/duweiyue25/conda/envs/rl4scip make steiner-s03-probe
scripts/steiner/run_s03_tmux.sh steiner-s03 6
```

All Steiner PySCIPOpt and SCIP commands must use the canonical launcher; do not
set `LD_LIBRARY_PATH` by hand:

```text
scripts/steiner/run_with_scip804.sh --verify-only
scripts/steiner/run_with_scip804.sh --python scripts/steiner/check_solution.py INPUT.stp
scripts/steiner/run_with_scip804.sh --scip --version
```

The launcher verifies the pinned hashes and the effective SCIP 8.0.4 /
PySCIPOpt 4.3.0 / Ecole 0.8.1 versions before execution. It deliberately
rejects arbitrary-command mode and non-empty `LD_PRELOAD` and does not inherit
the caller's library path. Its Python mode also prepends a checked-in `scip`
shim, so child processes cannot silently resolve `/usr/bin/scip` 9.2.2.
Generated data, LP files, solver output, and downloaded corpora belong under
ignored data/artifact directories.
