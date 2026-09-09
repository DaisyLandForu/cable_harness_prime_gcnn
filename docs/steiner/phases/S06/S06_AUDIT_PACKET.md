# S06 Audit Packet — focused pre-execution remediation review

## Audit object

- branch: `research/steiner-migration`;
- frozen S06 base: `ed0db3ff53a73cbb71de10f35e3471a6c139f146`;
- original implementation content head:
  `9866a3743ff1c2d6ef580aedf1f91629db0c6049`;
- original evidence/process head:
  `bcd9e33e9a09be856fc5f61078ccb4875ea325c1`;
- remediation fixed head and range: bound by the focused GPT re-audit request
  created immediately after this evidence commit;
- S05 audited dependency: `steiner-s05-audited-v3` →
  `6cf7acab57525a744233ed3fdfd463f00fcd470c`;
- initial external verdict: `CONDITIONAL_PASS`;
- scientific status: `NOT_EVALUATED`;
- formal artifacts at remediation head: none.

## Primary review entries

- original audit record:
  `docs/steiner/audits/S06_PREEXECUTION_AUDIT_RECORD.json`;
- remediation explanation:
  `docs/steiner/phases/S06/S06_PREEXECUTION_REMEDIATION.md`;
- protocol: `configs/steiner/experiments/s06_il_online_v1.yml`;
- protocol explanation: `docs/steiner/phases/S06/S06_PREEXECUTION_PROTOCOL.md`;
- instances: `configs/steiner/experiments/s06_online_instances_v1.json`;
- online policy/metrics/Gate:
  `python/steiner_branching/evaluation/s06_online.py`;
- distributed runner: `scripts/steiner/run_s06_online.py`;
- locked finalizer: `scripts/steiner/finalize_s06_online.sh`;
- custom-job entrypoint: `scripts/steiner/run_s06_online_shard.sh`;
- tests: `tests/steiner/test_s06_online.py`.

## Blocking-finding closure map

| Finding | Remediation | Negative/runtime evidence |
|---|---|---|
| B1 failure PAR-2 semantics | terminal `failed_task_result`: stable class, unsolved, PAR-2=1,200, PDI unavailable | invalid action/mapping/NaN/solver exceptions; retained 150 PAR-2 pairs and failed Gate |
| B2 registered resource identity | validate ≥8 effective CPU, ≥96 GiB, 0 GPU, frozen environment/stack, activation-bound runtime and executable identity | insufficient CPU/RAM, visible GPU, wrong fingerprint/head; per-task and cross-shard identity checks |
| B3 aggregate immutability | non-blocking aggregate lock, launcher-only aggregation and existing summary/manifest refusal | direct/duplicate finalizer and existing-output rejection |
| B4 packet missing at fixed head | this packet is committed before the new fixed remediation head is named | focused request points only to a head that contains this file |

## Requirements-to-evidence map

| Requirement | Implementation evidence | Test evidence |
|---|---|---|
| frozen seed-202 B0 identity | config and strict checkpoint loader | protocol/hash and real solve tests |
| legal `stp_x` actions only | online B0 branchrule + S04 canonical extractor | permutation/fail-closed inherited suite and toy solve |
| fair P1 baselines | one task builder/profile for all methods | 750-task expansion and objective agreement |
| metrics/failure retention | event recorder, PDI, stats parser, solution checker, failure result schema | parser, aggregation, PAR-2/PDI and correctness-failure tests |
| paired statistics | graph-level five-seed grouping and bootstrap | complete and injected-failure synthetic matrix tests |
| six safe custom jobs | lineage modulo partition, locks and phase manifests | disjointness, wrong/missing shard and lock tests |
| registered runtime | CPU/RAM/GPU/fingerprint/head validation in parent, task and barrier | resource negatives and task/manifest mismatch tests |
| immutable aggregate | locked finalizer plus pre-write output refusal | duplicate/direct finalizer and existing-output negatives |
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
- aggregation rejects wrong-shard evidence or task/manifest runtime mismatch;
- every shard independently satisfies registered resources and all six share
  the same activation-bound runtime identity;
- all jobs share one persistent artifact root but write unique task paths;
- the finalizer is singleton and aggregate evidence cannot be overwritten.

## Local evidence

- targeted S06: `17 passed`;
- complete Steiner suite: `120 passed, 1 expected PACE skip`;
- Python compilation, SCIP wrapper verification, validate-only, shell syntax
  and `git diff --check`: PASS;
- no formal S06 shard exists; activation record does not yet exist.

## Hashes

- protocol YAML:
  `e7f7e9060c25fa038a2749ca43afe6a2769c3652e93697612b8804269750f827`;
- instance manifest:
  `b50f8048d8ab27a1fa2168e69ef179cbfc490116399180f2bb4a963bf04b292b`;
- environment lock:
  `f70afe548f2b640a3c1375686ad8c8ef4dced63d0229c9fa4eb36e62f6d7628e`;
- protocol explanation:
  `6cbd5473115524813549b628369b1dbcc11e9f849762fafe2789bad384b0fb62`;
- remediation explanation:
  `15f00d21a637edbf648c33471dce2422cbaacd32b457105e471eda7cc1badc50`;
- initial conditional-audit record:
  `142baa5fd1d6e7871e905ed5918d188bb60d9a0ecda29778685c510b8ab000d2`;
- online implementation:
  `477b8724aab0231f38a9333602ba9b85174f2cfdd807abfac2d4fb808c58428f`;
- distributed runner:
  `c2f18dd0313db2603ef08aff80130743c7f5ece80c17d94d5de10bdfa36e2437`;
- locked finalizer:
  `b0a963ca416f345ef3c44031f85114418cff38edd35506d5a63eb23e1d9a04ee`;
- tests:
  `c6c609482617e34d0e91d754c1101a3897213a0f7ad4c78900d3ae8acb9be417`.

Validate-only runtime fingerprint to bind in the post-PASS activation if the
runtime remains unchanged:
`fe7a032641aba58090f59cfa9d5b088cf76da23b799f2099a34e8d99b17b5687`.

The focused re-audit request separately binds hashes for this packet,
remediation explanation and protocol explanation after commit.

## Required audit decision

Please return PASS, CONDITIONAL PASS or FAIL and explicitly mark B1--B4
CLOSED/OPEN. Only PASS permits a separate activation record and the six main
custom jobs. This review is not an S06 result audit, cannot mark the scientific
Gate PASS and cannot authorize S07 or test/final access.

Suggested local conclusion: **B1--B4 CLOSED / implementation Gate PASS /
formal scientific Gate NOT_EVALUATED**.
