# S05 Test Report — implementation-only checkpoint

## Frozen environment

- branch: `research/steiner-migration`
- base SHA: `030199703c6e280533f1f1c7cfc8d00d7df0a6b0`
- solver stack: SCIP 8.0.4 / PySCIPOpt 4.3.0 / Ecole 0.8.1
- config canonical SHA-256:
  `34ac665eef539fa2b0f9129fb1c2e8e250ce3252289cbd2d5676dfd13e16a0fd`
- formal teacher runs: 0; learning runs: 0; GPU calls: 0; final access: 0

## Verification

```text
scripts/steiner/run_with_scip804.sh --python \
  scripts/steiner/collect_s05_teacher.py --dry-run
scripts/steiner/run_with_scip804.sh --python -m pytest -q \
  tests/steiner/test_s05_teacher_il.py
scripts/steiner/run_with_scip804.sh --python -m pytest -q tests/steiner
scripts/steiner/run_with_scip804.sh --python -m compileall -q \
  python/steiner_branching scripts/steiner tests/steiner
bash -n scripts/steiner/run_s05_teacher_tmux.sh \
  scripts/steiner/run_s05_train_tmux.sh
git diff --check
```

- S05 targeted suite: 6 passed.
- Complete Steiner suite: 84 passed, 1 expected PACE-development skip, 31.14 s.
- Dry-run: 10/10 tasks expanded; five train and five validation-IID, all five
  families, only teacher seed 1001; no artifact directory was written.
- Real frozen-SCIP test: one 48-node MCF state, all legal candidates mapped by
  probindex, every child label valid, SB calls cover candidates, SB LP iterations
  positive, and custom scores equal Ecole scores within `1e-12`.
- Shard test: write/reload preserves state and semantic hash; wrong file checksum
  and cross-split graph lineage fail closed.
- Loss/metrics/normalization tests: finite listwise gradient, exact regret
  endpoints, rank/top-k behavior, all three offline baselines, and validation
  input rejection by train-only normalization.
- Checkpoint test: CPU training step succeeds; checkpoint and normalization
  checksums validate; all state-dict tensors reload bit-exactly.
- Python and shell syntax checks: PASS. `git diff --check`: PASS except the
  intentionally verbatim, already committed external GPT audit whitespace.

## Preserved implementation failures

1. The first metric assertion compared floating Spearman output to exact 1.0;
   observed `0.9999999999999999`. The test now uses a numeric tolerance; metric
   semantics were not changed.
2. The first real test passed an unsupported keyword to Ecole 0.8.1
   `Pseudocosts`; local API inspection showed it takes no arguments, and the call
   was corrected.
3. The first C call treated success as return code 0. SCIP 8.0.4 defines
   `SCIP_OKAY` as `+1`; the bridge now checks the pinned ABI value explicitly.
   The corrected real integration test and Ecole-score parity pass.

These are implementation-test failures, not hidden/skipped experiment samples.
No Gate, data list, seed or threshold changed in response.

## Gate

The implementation tests **PASS**, but S05 scientific Gate is **NOT_RUN**. It
cannot be evaluated until S04 re-audit PASS, CPU teacher collection, GPU
training, multi-seed validation metrics and manifest reload evidence exist.
