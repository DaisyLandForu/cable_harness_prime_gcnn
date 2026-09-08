# S05 Test Report — formal-v3 audited Gate PASS

## Formal-v3 execution verification

- Teacher: 240/240 terminal (`237 completed`, `3 root_solved`), zero failures;
  3,383/3,840 valid states, 13/3,383 all-tie valid states and exact
  69,709/69,709 action mapping. Teacher Gate: **PASS**.
- Teacher resource identity: 80 visible CPUs, 128.00098 GiB RAM, six workers,
  one SCIP thread per task, frozen SCIP 8.0.4 wrapper.
- Pre-model barrier: exact 30 lineages, six per family, exact 320 selected
  states, 64 per family, every lineage represented, unique semantics and zero
  prior/cross-role/test-final lineage. All 13 checks: **PASS**.
- Selection seal was committed at `fab9b720e513da66294e95dba0511a5bfa32b9a3`
  before any model load.
- Frozen-model evaluation: seeds 101/202/303/404/505 all completed on independent
  one-V100 processes. Repeat inference error and checkpoint reload error are
  `0.0` for all five; all state dicts reload bit-exactly.
- Aggregate: all nine scientific checks **PASS**; 30-lineage bootstrap 95% CI
  `[0.0578104443, 0.1678878191]`, CV `0.0167927294`, every seed/family direction
  positive. Local Gate: **PASS**.
- Post-result targeted formal suite: **13 passed** in 19.22 s. Post-result
  complete frozen-stack suite: **103 passed, 1 expected PACE skip** in 79.14 s.
  The committed selection seal reloaded successfully against the raw selected
  manifest SHA-256.
- Result-audit closeout was documentation-only. JSON parsing, audit/result/source
  SHA cross-checks and `git diff --check` passed; no solver or training test was
  rerun because the audited result content and its 103-pass suite were unchanged.
- Teacher, selected, seal and aggregate hashes are listed in
  `S05_AUDIT_PACKET.md`; the byte-exact committed aggregate is
  `S05_FORMAL_V3_GATE_SUMMARY.json`.
- The automated pipeline watcher stopped safely on a stale Git proxy before GPU
  execution; the seal was then pushed with proxy variables removed. A separate
  aggregate watcher observed seed 505's running report and stopped before
  aggregation; aggregation was invoked once manually after all five reports
  were completed. Neither scheduling interruption changed experiment inputs or
  artifacts.
- The first standalone seal-loader verification omitted `PYTHONPATH=python` and
  failed before import with `ModuleNotFoundError`; the corrected wrapper command
  passed. This was a command-environment mistake, not an experiment or Gate
  failure.
- No model was retrained and no test/final or S06 command ran.

## Formal-v3 implementation verification

- External pre-execution verdict: **PASS**, recorded separately; audited YAML
  remains `execution_authorized=false` and is activated only by that record.
- Dry-run: exactly 240 unique tasks (80 candidates x three teacher seeds), with
  no artifact or model access.
- Production selector: arbitrary candidate/sample permutations reproduce the
  same ordered 30 lineages and 320 semantic identities; a family with fewer
  than six eligible lineages fails closed.
- Production evaluator: a failed/missing committed selection seal results in
  zero calls to the checkpoint loader. The seal loader also requires its bytes
  to exist at Git `HEAD`.
- Targeted formal suite: **13 passed** in 18.40 s.
- Complete frozen-stack suite with the registered cuBLAS environment:
  **103 passed, 1 expected PACE skip** in 46.85 s.
- Python compilation, shell syntax, SCIP 8.0.4 identity and `git diff --check`:
  **PASS**. Ruff is not installed in the locked environment and is not claimed.
- No v3 teacher task, selection, checkpoint load, GPU evaluation, test/final
  access or S06 operation occurred during these tests.

## Formal-v3 pre-execution static verification

- V3 YAML SHA-256:
  `99e75a4d4fa69f805232c637d4fc0ae750fcfccb57e9979b561f422591006242`.
- Candidate graph manifest SHA-256:
  `e65fc9a03fd683277570befe13b11f4b15ea981ee427568d56aeeccdd3847b56`.
- Protocol explanation SHA-256:
  `c61b58d23f4de6bb2709aa9c40ccab11d6db4507a6bd1c9d2e544eb68595d873`.
- V2 result-audit record SHA-256:
  `a7cd6af30e9b1e98f46efb1b40f5876bb3f08cf75a45da59b09e319de33ec48a`.
- Targeted formal tests: **10 passed**. They regenerate all 80 graphs and check
  exact hashes, 80 unique seeds/hashes, validation-IID assignment, no prior S05
  seed collision, the frozen family-scale envelope, all five local checkpoint
  manifest/model hashes, deterministic first-six eligibility selection, the
  exact 30/6/320/64 pre-model barrier, and failure with fewer than six eligible
  lineages.
- `execution_authorized=false`; no v3 collection, selection, model load, GPU
  evaluation, test/final access or S06 work occurred.
- Complete frozen-stack Steiner suite: **100 passed, 1 expected PACE skip** in
  43.64 s. The skip is unchanged and occurs because the optional odd-numbered
  PACE development-data path is not configured.

## Formal-v2 final execution verification

- Activation: external GPT `PASS`, B1 closed; frozen v2 YAML and explanation
  hashes verified.
- Reselection: all 3,431 source shard checksums verified; train 640/640,
  family counts 128 each, validation selections unchanged at 160/320.
- Post-implementation frozen-stack suite with deterministic CUDA environment:
  **97 passed, 1 expected PACE skip**.
- Formal seeds 101/202/303/404/505: **5/5 completed**, 40 epochs each,
  independent one-V100 processes, five disjoint report/checkpoint roots.
- Checkpoint verification: **5/5 bit-exact**, maximum reload inference error
  **0.0**.
- Aggregation negative path: fail-closed with
  `formal Gate must contain exactly 30 base graph lineages`; observed valid
  lineage count is 25. The exception occurred before a formal bootstrap result
  was written.
- Test/final access count: zero. No failed state, graph, seed or task was
  removed or replaced.

## Formal-v1 failure and v2 preregistration checks

- Failed manifest SHA-256 and all reported counts were read from the retained
  formal-v1 artifact; 315/315 task envelopes are terminal and failure list is
  empty.
- Static v2 checks require the original bucket targets, exactly 128 selected
  states per family/640 total, unchanged base training settings, no replacement
  instances, v1 status retained as FAIL, and no execution before GPT PASS.
- Frozen-stack complete Steiner suite: **95 passed, 1 expected PACE skip**.

The subsequent B1 ordering remediation adds a synthetic sealed-availability
fixture. It verifies literal manifest schema keys, permutation-invariant
semantic identities, exact frozen bucket composition, and failure on an
unknown key. Targeted `test_s05_formal.py`: **6 passed**; complete frozen-stack
Steiner suite after remediation: **96 passed, 1 expected PACE skip**.

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

The remediation implementation tests **PASS** and pilot-v3 remains **PASS**.
Formal-v1 and formal-v2 remain retained FAIL results. Formal-v3 completed its
fresh exact 30-lineage matrix and the local S05 Gate is **PASS**. The external
result audit returned **PASS with no blocking findings**. S05 is therefore PASS
via formal-v3; S06 implementation/handoff is authorized after the audited tag,
while test/final access remains prohibited. The S04 prerequisite remains PASS.
