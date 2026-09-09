# S06 Audit Packet — pre-execution implementation review

## Audit object

- branch: `research/steiner-migration`
- frozen base: `ed0db3ff53a73cbb71de10f35e3471a6c139f146`
- implementation content head: `9866a3743ff1c2d6ef580aedf1f91629db0c6049`
- substantive range:
  `ed0db3ff53a73cbb71de10f35e3471a6c139f146..9866a3743ff1c2d6ef580aedf1f91629db0c6049`
- S05 audited dependency: `steiner-s05-audited-v3` →
  `6cf7acab57525a744233ed3fdfd463f00fcd470c`
- scientific status: `NOT_EVALUATED`
- formal artifacts at content head: none

## Primary review entries

- protocol: `configs/steiner/experiments/s06_il_online_v1.yml`
- protocol explanation: `docs/steiner/phases/S06/S06_PREEXECUTION_PROTOCOL.md`
- instances: `configs/steiner/experiments/s06_online_instances_v1.json`
- online policy/metrics/Gate:
  `python/steiner_branching/evaluation/s06_online.py`
- distributed runner: `scripts/steiner/run_s06_online.py`
- custom-job entrypoint: `scripts/steiner/run_s06_online_shard.sh`
- tests: `tests/steiner/test_s06_online.py`

## Requirements-to-evidence map

| Requirement | Implementation evidence | Test evidence |
|---|---|---|
| frozen seed-202 B0 identity | config and strict checkpoint loader | protocol/hash and real solve tests |
| legal `stp_x` actions only | online B0 branchrule + S04 canonical extractor | permutation/fail-closed inherited suite and toy solve |
| fair P1 baselines | one task builder/profile for all methods | 750-task expansion and objective agreement |
| metrics/failure retention | event recorder, PDI, stats parser, solution checker | parser, aggregation and correctness-failure tests |
| paired statistics | graph-level five-seed grouping and bootstrap | complete synthetic matrix test |
| six safe custom jobs | lineage modulo partition, locks and phase manifests | disjointness, wrong/missing shard and lock tests |
| no premature execution | external activation loader and clean-input checks | missing/invalid activation tests |
| no final leakage | validation source/hash/split guards | config/manifest reconstruction tests |

## Six-shard invariants

- exactly six shard IDs 0--5;
- each shard has five lineages, one per graph family;
- each lineage carries all five solver seeds and five main methods;
- exactly 125 main tasks per shard and 750 in the union;
- no intersection between shard task-ID sets;
- trace tasks inherit the same lineage shard;
- both waves require six exact terminal manifests;
- aggregation rejects wrong-shard evidence or resource identity mismatch;
- all jobs share one persistent artifact root but write unique task paths.

## Local evidence

- targeted: `10 passed`;
- complete Steiner suite: `113 passed, 1 expected PACE skip`;
- compileall, SCIP wrapper verification, validate-only, shell syntax and
  `git diff --check`: PASS;
- no formal S06 shard exists; activation record does not yet exist.

## Hashes

- protocol YAML:
  `8d0daa708c5d2ace6bbd32da867dd978f730c97216df74277208e86012db50da`
- instance manifest:
  `b50f8048d8ab27a1fa2168e69ef179cbfc490116399180f2bb4a963bf04b292b`
- protocol explanation:
  `db501869563be02fdc6e53a1074e8b6d0c46b306078ce2bcc671569faf9cc50a`
- online implementation:
  `fa24b60483c55a135e5a5bbd126d513eaf05528042f4277e21753f0f27d64161`
- distributed runner:
  `9bfba18db60d648629b18c868b7839e06cd289c9f53c02c87d0a456c49fa458f`
- tests:
  `759156a7383dec6e7a61791309f75c98dfc4d1c8fbff1d7bf4098683da9f25e6`

## Required audit decision

Please return PASS, CONDITIONAL PASS or FAIL for implementation/protocol
execution readiness. Only PASS permits a separate activation record and the
six main custom jobs. This review is not an S06 result audit, cannot mark the
scientific Gate PASS and cannot authorize S07 or test/final access.

Suggested local conclusion: **implementation Gate PASS / formal Gate
NOT_EVALUATED**.
