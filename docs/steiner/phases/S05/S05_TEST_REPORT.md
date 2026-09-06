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
- formal teacher runs: 0; pilot-v2 failed learning attempts: 9; pilot-v3
  completed learning runs: 9; final access: 0

## Formal protocol v1 static verification

- Frozen formal YAML SHA-256:
  `c843f76d69c07b1c8a8093ab6f1426656084b2de2f4e9e0d777f1af32cd4c811`.
- The pre-audit YAML parses under the frozen SCIP/Python wrapper and remains
  non-executable: `execution_authorized=false`.
- Exact expansion is 60 train, 15 validation-select and 30 validation-gate base
  graphs, balanced 12/3/6 per family; 105 unique generator seeds and 315
  graph-by-teacher tasks.
- All train/validation seeds lie inside the registered split ranges, are
  disjoint from each other and from the S05 pilot seeds, and no test/final seed
  occurs.
- State quotas sum to 640 train, 160 validation-select and 320 validation-gate;
  maximum registered observations are 5,040.
- Static config validation and `git diff --check`: PASS. The complete Steiner
  suite remains 90 passed and 1 expected PACE-development skip in 63.65 s. No
  formal collector, trainer, checkpoint or raw output was invoked or created.

## Formal implementation verification

- The audited YAML and explanation SHA-256 values are checked at runtime; a
  separate PASS activation record is mandatory and S06 remains false.
- Formal dry-run expands exactly 315 tasks: 180 train, 45 validation-select and
  90 validation-gate. Any byte change to the audited YAML fails closed.
- Selection tests require 640/160/320 states, reject graph lineage role leakage
  and retain quota shortages instead of substituting instances.
- Statistical tests use 30 graph lineages, reproduce the paired bootstrap and
  turn a negative model-minus-random direction into formal Gate FAIL.
- Targeted S05 formal+pilot suite: 15 passed. Both V100 identities and the
  deterministic CUDA preflight were verified; no training was performed by the
  preflight.
- Complete Steiner suite after formal implementation: 93 passed and 1 expected
  PACE-development skip in 36.93 s. Python compilation, shell syntax and
  `git diff --check` pass.

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
- Pilot-v3 teacher manifest SHA-256:
  `9ba08c0f30f1014396c7a3825b6bb3247fe4d6e0d2dab04c180ae26cb307d36f`;
  strict aggregate SHA-256:
  `73aa463a25d10cfd28a1e17e2396ad76eda4b945d0e2f04e7d775571bcded390`.
  All 9 runs completed, all state dicts reloaded exactly, and maximum reload
  logit error was 0.

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

The remediation implementation tests **PASS**. Pilot-v3 teacher collection,
three seed reports and strict 9-run aggregation also **PASS**, with max reload
error 0. Full S05 scientific Gate remains **NOT_EVALUATED** until the formal
state budget and five-seed protocol are frozen and executed. S06 remains
blocked; the S04 re-audit prerequisite remains PASS.
