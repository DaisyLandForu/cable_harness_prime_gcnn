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

PySCIPOpt commands must run in the frozen environment with the SCIP 8.0.4
library directory on `LD_LIBRARY_PATH`. Generated data, LP files, solver output,
and downloaded corpora belong under ignored data/artifact directories.
