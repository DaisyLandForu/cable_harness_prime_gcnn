# S06 Test Report

Status: base pre-execution implementation/protocol audit PASS; six-shard main
wave complete and sealed; execution amendment A1 tests PASS locally; amendment
audit pending; scientific Gate NOT_EVALUATED.

## Environment

- branch: `research/steiner-migration`
- base: `ed0db3ff53a73cbb71de10f35e3471a6c139f146`
- initial implementation content head:
  `9866a3743ff1c2d6ef580aedf1f91629db0c6049`
- remediation base: `bcd9e33e9a09be856fc5f61078ccb4875ea325c1`
- first remediation head: `a8fbc0986068171344387ceb68e1ee5ab5bbefbc`
- externally audited second remediation head:
  `a29ef9eeb818c1694f78d2de1b29436f1861f8c3`
- base activation/execution head:
  `a80d7459b9b1d4d5bff9743711984cd5a649ef97`
- amendment A1 content head: pending this report's substantive commit
- stack: SCIP 8.0.4 / PySCIPOpt 4.3.0 / Ecole 0.8.1
- Python: 3.11.15
- amendment test host: 48 visible CPUs, 128 GiB RAM, no CUDA device visible

## Commands and outcomes

1. Targeted S06 suite:

   ```text
   scripts/steiner/run_with_scip804.sh --python -m pytest \
     tests/steiner/test_s06_online.py -q
   ```

   Result: `21 passed`; exit 0.

2. Complete frozen-stack Steiner suite:

   ```text
   CUBLAS_WORKSPACE_CONFIG=:4096:8 \
     scripts/steiner/run_with_scip804.sh --python -m pytest \
     tests/steiner -q -rs
   ```

   Result: `123 passed, 2 skipped`; exit 0. The skips are the existing optional
   PACE odd development test (`PACE_ROOT` unavailable) and one existing S05
   real-CUDA determinism test because this host exposes no CUDA device.

3. Static/runtime preflight:

   ```text
   python -m compileall -q python/steiner_branching scripts/steiner tests/steiner
   scripts/steiner/run_with_scip804.sh --verify-only
   scripts/steiner/run_with_scip804.sh --python \
     scripts/steiner/run_s06_online.py --validate-only
   git diff --check
   ```

   Result: all exit 0. Validate-only reconstructed 30 lineages, 750 main
   tasks, five diagnostics and six disjoint 125-main-task shards, and verified
   amendment/seal hashes while reporting `amendment_execution_authorized=false`.

4. Sealed real-main barrier replay (read-only):

   ```text
   verify_main_wave_seal(...)
   _load_phase_barrier(..., phase="main", sealed_main=seal)
   ```

   Result: PASS. Exactly 761 registered files reproduced evidence-tree SHA-256
   `671cb10a78a30e9f227e6b8b86a62d3ba4437a013ba9350850bb2ef1b1fb8964`;
   six manifests passed with three recorded physical CPU models and one common
   `effective_cpu_cores=8.01` identity.

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
- committed-byte activation verification, including an uncommitted fingerprint
  mutation negative test;
- Python-owned aggregate OS lock, including two direct Python lock contenders;
- no aggregation before both six-shard phase barriers;
- no formal run without an external PASS activation record.
- scheduler host/physical CPU/affinity variation is accepted only when every
  registered effective resource and frozen runtime/code identity is equal;
- exact 761-file main evidence membership and bytes, including mutation and
  unexpected-file rejection;
- amendment main-rerun prohibition through both parent and internal task paths;
- no aggregation until both the sealed-main and complete trace barriers pass.

## Formal execution status

- All 750 main tasks and five diagnostics have terminal envelopes; their
  completion/status counts are 399 optimal, 356 timelimit, zero solver errors.
- No main task was rerun during amendment work.
- No trace replay was executed.
- No S06 Gate result was calculated from real formal outcomes.
- Optional PACE development data was unavailable and is outside this S06
  validation-IID protocol.

## Amendment A1 hashes

- execution amendment YAML:
  `b862f81893fb5dd46635dfc55366b5961268de41d74dd2f6cf89f11705690abe`;
- main-wave seal:
  `11925fc0e5c6169a0fbdb7c1770ebb0709cefe599f6c0018f07d875595a81603`;
- sealed evidence tree:
  `671cb10a78a30e9f227e6b8b86a62d3ba4437a013ba9350850bb2ef1b1fb8964`;
- distributed runner after A1:
  `f506f009e2c18a6d41c5b4d932b4901cbe0c6b79146f28d001d796398e8b5273`;
- S06 tests after A1:
  `0cb01ab3e829899e8ff88ea52d902a2acd08ec4a8f27cfbb44fbf31e57e27509`.

## Hashes at externally audited remediation head

- protocol: `e7f7e9060c25fa038a2749ca43afe6a2769c3652e93697612b8804269750f827`
- instance manifest: `b50f8048d8ab27a1fa2168e69ef179cbfc490116399180f2bb4a963bf04b292b`
- protocol explanation: `23139ea8810c608a6e816a1a4fd51d13ff3f48ca3f78c52bd4941c555e16dd48`
- environment lock: `f70afe548f2b640a3c1375686ad8c8ef4dced63d0229c9fa4eb36e62f6d7628e`
- validate-only runtime fingerprint:
  `fe7a032641aba58090f59cfa9d5b088cf76da23b799f2099a34e8d99b17b5687`
