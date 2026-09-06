# S05 Test Report — implementation-only checkpoint

## Frozen environment

- branch: `research/steiner-migration`
- base SHA: `030199703c6e280533f1f1c7cfc8d00d7df0a6b0`
- solver stack: SCIP 8.0.4 / PySCIPOpt 4.3.0 / Ecole 0.8.1
- pilot-v1 config canonical SHA-256:
  `34ac665eef539fa2b0f9129fb1c2e8e250ce3252289cbd2d5676dfd13e16a0fd`
- pilot-v2 config canonical SHA-256:
  `2146e7d67dadcef93746400a08d10441e051745075fc7a39348c5b6c80b6cacf`
- pilot-v3 config canonical SHA-256:
  `cccb611deba26726772680416f49b7404c281e3520ee813dde4b2b4a100f178f`
- formal teacher runs: 0; pilot-v2 learning attempts: 9; final access: 0

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
  scripts/steiner/run_s05_train_tmux.sh \
  scripts/steiner/run_s05_teacher_batch.sh \
  scripts/steiner/run_s05_pilot_seed_batch.sh
git diff --check
```

- S05 targeted suite after batch-sharding support: 8 passed, 3.49 s.
- Complete Steiner suite: 86 passed, 1 expected PACE-development skip, 33.54 s.
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
- Parallel-report tests: registered seed subsets get disjoint paths; duplicate,
  unknown and incomplete seed matrices fail closed; a complete two-job split
  merges to exactly nine pilot runs.
- Python and shell syntax checks: PASS. `git diff --check`: PASS.
- After teacher attempt 1 exposed legal fractional parts above 0.5, the action
  validator regression fixture now includes 0.75 and rejects only 0/1 integer
  endpoints. S04+S05 targeted tests: 18 passed in 94.17 s; complete Steiner
  suite: 86 passed, 1 expected skip in 121.35 s.
- A real previously failing community state produced 33/33 identical
  action/teacher probindex sets with fraction range 0.25--0.75. After the fix it
  traversed two consecutive strong-teacher branch states successfully.
- The S04 deterministic snapshot remained byte-identical at
  `ac2ce0c14b134245221af5140a3008f3ec6067f8867491e7cc0d0b50e2036f2c`,
  and its 8/8 Gate checks remained true.
- Pilot-v2 dry-run: 12/12 exact tasks, with only train sparse seed 100303 and
  train grid seed 100315 added. Unknown experiment IDs, task substitutions,
  reordering, path drift and unregistered seeds fail closed.
- Post-v2 implementation suite: S05 targeted 9 passed; complete Steiner suite
  87 passed, 1 expected PACE-development skip in 29.98 s. Shell/Python syntax
  and `git diff --check` pass.
- Pilot-v3 remediation suite: S05 targeted 12 passed in 6.30 s; complete Steiner
  suite 90 passed, 1 expected PACE-development skip in 32.46 s. The added real
  V100 test performs deterministic training, repeated inference and checkpoint
  reload with strict array equality. Missing/changed cuBLAS configuration fails
  closed. Python/shell syntax and `git diff --check` pass.

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

4. First authorized teacher pilot: 5/10 tasks completed and 5/10 failed because
   the local `solution_frac` validation incorrectly rejected legal values above
   0.5. Counts (65 observed, 56 valid, 1,540/1,540 mapped) and every failed task
   are retained. This was fixed before retry without weakening any S05 Gate.
5. Pilot-v2 GPU attempt: all nine training runs completed 40 epochs and wrote
   checkpoints, then failed exact reload parity because CUDA deterministic
   algorithms and cuBLAS workspace policy were not fully enabled. The failure
   reports and artifacts are retained; no threshold was relaxed.

## Gate

The remediation implementation tests **PASS**, but the latest completed S05
scientific attempt is **FAIL**. Pilot-v3 must recollect its commit-bound teacher
manifest and produce three exact-reload seed reports before aggregation. The
S04 re-audit prerequisite remains PASS.
