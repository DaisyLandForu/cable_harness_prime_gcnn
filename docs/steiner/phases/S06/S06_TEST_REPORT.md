# S06 Test Report

Status: pre-execution implementation Gate PASS; scientific Gate NOT_EVALUATED.

## Environment

- branch: `research/steiner-migration`
- base: `ed0db3ff53a73cbb71de10f35e3471a6c139f146`
- implementation content head: `9866a3743ff1c2d6ef580aedf1f91629db0c6049`
- stack: SCIP 8.0.4 / PySCIPOpt 4.3.0 / Ecole 0.8.1
- Python: 3.11.15
- local host seen during tests: 80 CPUs, 128 GiB RAM, 2 V100 32-GB GPUs

## Commands and outcomes

1. Targeted S06 suite:

   ```text
   scripts/steiner/run_with_scip804.sh --python -m pytest \
     tests/steiner/test_s06_online.py -q
   ```

   Result: `10 passed`; exit 0.

2. Complete frozen-stack Steiner suite:

   ```text
   CUBLAS_WORKSPACE_CONFIG=:4096:8 \
     scripts/steiner/run_with_scip804.sh --python -m pytest \
     tests/steiner -q -rs
   ```

   Result: `113 passed, 1 skipped`; exit 0. The only skip is the existing
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
- six-way task completeness/disjointness and one-lineage-per-family balance;
- wrong-shard task injection, missing shard manifest and duplicate job lock;
- no aggregation before both six-shard phase barriers;
- no formal run without an external PASS activation record.

## Not executed

- No 750-task formal matrix or trace replay was executed.
- No S06 Gate result was calculated from real formal outcomes.
- Optional PACE development data was unavailable and is outside this S06
  validation-IID protocol.

## Frozen input hashes

- protocol: `8d0daa708c5d2ace6bbd32da867dd978f730c97216df74277208e86012db50da`
- instance manifest: `b50f8048d8ab27a1fa2168e69ef179cbfc490116399180f2bb4a963bf04b292b`
- protocol explanation: `db501869563be02fdc6e53a1074e8b6d0c46b306078ce2bcc671569faf9cc50a`
