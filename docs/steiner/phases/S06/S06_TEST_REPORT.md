# S06 Test Report

Status: remediated pre-execution implementation Gate PASS locally; focused
external re-audit pending; scientific Gate NOT_EVALUATED.

## Environment

- branch: `research/steiner-migration`
- base: `ed0db3ff53a73cbb71de10f35e3471a6c139f146`
- initial implementation content head:
  `9866a3743ff1c2d6ef580aedf1f91629db0c6049`
- remediation base: `bcd9e33e9a09be856fc5f61078ccb4875ea325c1`
- remediation fixed head: recorded by the focused re-audit request after this
  evidence commit
- stack: SCIP 8.0.4 / PySCIPOpt 4.3.0 / Ecole 0.8.1
- Python: 3.11.15
- local host seen during tests: 80 CPUs, 128 GiB RAM, 2 V100 32-GB GPUs

## Commands and outcomes

1. Targeted S06 suite:

   ```text
   scripts/steiner/run_with_scip804.sh --python -m pytest \
     tests/steiner/test_s06_online.py -q
   ```

   Result: `17 passed`; exit 0.

2. Complete frozen-stack Steiner suite:

   ```text
   CUBLAS_WORKSPACE_CONFIG=:4096:8 \
     scripts/steiner/run_with_scip804.sh --python -m pytest \
     tests/steiner -q -rs
   ```

   Result: `120 passed, 1 skipped`; exit 0. The only skip is the existing
   optional PACE odd development test because `PACE_ROOT` was not provided.

3. Static/runtime preflight:

   ```text
   python -m compileall -q python/steiner_branching scripts/steiner tests/steiner
   scripts/steiner/run_with_scip804.sh --verify-only
   scripts/steiner/run_with_scip804.sh --python \
     scripts/steiner/run_s06_online.py --validate-only
   git diff --check
   ```

   Result: all exit 0. Validate-only reconstructed 30 lineages, 750 main
   tasks, five diagnostics and six disjoint 125-main-task shards.

## Covered risks

- exact S05 audited tag/checkpoint/normalization and S06 config/manifest hashes;
- canonical probindex action identity and finite deterministic B0 inference;
- deterministic random action and native SCIP baseline configuration;
- real SCIP solve/objective agreement across all six methods on a toy graph;
- complete paired matrix, bootstrap, correctness and failure retention;
- exception/invalid-policy/NaN/mapping evidence keeps PAR-2=1,200 while PDI is
  unavailable and the Gate fails;
- six-way task completeness/disjointness and one-lineage-per-family balance;
- wrong-shard task injection, missing shard manifest and duplicate job lock;
- minimum CPU/RAM, zero-GPU, environment/runtime fingerprint, activation and
  audited-executable checks, including per-task runtime identity binding;
- duplicate finalizer and pre-existing aggregate output refusal;
- no aggregation before both six-shard phase barriers;
- no formal run without an external PASS activation record.

## Not executed

- No 750-task formal matrix or trace replay was executed.
- No S06 Gate result was calculated from real formal outcomes.
- Optional PACE development data was unavailable and is outside this S06
  validation-IID protocol.

## Frozen input hashes

- protocol: `e7f7e9060c25fa038a2749ca43afe6a2769c3652e93697612b8804269750f827`
- instance manifest: `b50f8048d8ab27a1fa2168e69ef179cbfc490116399180f2bb4a963bf04b292b`
- protocol explanation: `6cbd5473115524813549b628369b1dbcc11e9f849762fafe2789bad384b0fb62`
- environment lock: `f70afe548f2b640a3c1375686ad8c8ef4dced63d0229c9fa4eb36e62f6d7628e`
- validate-only runtime fingerprint:
  `fe7a032641aba58090f59cfa9d5b088cf76da23b799f2099a34e8d99b17b5687`
