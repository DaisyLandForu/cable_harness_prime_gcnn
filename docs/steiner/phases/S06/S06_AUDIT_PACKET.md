# S06 Audit Packet — second focused pre-execution remediation review

## Audit object

- branch: `research/steiner-migration`;
- frozen S06 base: `ed0db3ff53a73cbb71de10f35e3471a6c139f146`;
- original implementation content head:
  `9866a3743ff1c2d6ef580aedf1f91629db0c6049`;
- original evidence/process head:
  `bcd9e33e9a09be856fc5f61078ccb4875ea325c1`;
- first remediation head: `a8fbc0986068171344387ceb68e1ee5ab5bbefbc`;
- first focused request head: `98a535d19924c80df47fcb3f8153d7fdf89ccf86`;
- second remediation fixed head:
  `a29ef9eeb818c1694f78d2de1b29436f1861f8c3`;
- second focused request head:
  `cb78a9ea93a5bafdf18e8a585358d195b92be200`;
- S05 audited dependency: `steiner-s05-audited-v3` →
  `6cf7acab57525a744233ed3fdfd463f00fcd470c`;
- initial external verdict: `CONDITIONAL_PASS`;
- first focused verdict: `CONDITIONAL_PASS`, with B1/B4 closed and B2/B3 open;
- second focused verdict: `PASS`, B1--B4 closed;
- scientific status: `NOT_EVALUATED`;
- formal artifacts at remediation head: none.

## Primary review entries

- original audit record:
  `docs/steiner/audits/S06_PREEXECUTION_AUDIT_RECORD.json`;
- first focused audit record:
  `docs/steiner/audits/S06_PREEXECUTION_REMEDIATION_AUDIT_RECORD.json`;
- final pre-execution PASS record:
  `docs/steiner/audits/S06_PREEXECUTION_B2_B3_AUDIT_RECORD.json`;
- first remediation explanation:
  `docs/steiner/phases/S06/S06_PREEXECUTION_REMEDIATION.md`;
- B2/B3 remediation explanation:
  `docs/steiner/phases/S06/S06_PREEXECUTION_REMEDIATION_V2.md`;
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
| B2 registered resource identity | validate resources/runtime and require activation local bytes to equal committed `HEAD` bytes | resource negatives, task/shard identity, uncommitted activation mutation rejection |
| B3 aggregate immutability | Python evidence writer owns non-blocking OS lock; existing summary/manifest refused | two direct Python contenders cannot both lock; existing-output rejection |
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
| committed activation | repo-contained path plus byte-exact `git show HEAD:<path>` check | uncommitted runtime-fingerprint mutation rejection |
| immutable aggregate | Python-owned OS lock plus pre-write output refusal | direct Python lock contention and existing-output negatives |
| no premature execution | external committed activation loader and clean-input checks | missing/uncommitted/invalid activation tests |
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
- the Python evidence writer is singleton and aggregate evidence cannot be
  overwritten, regardless of shell or direct Python entrypoint.

## Local evidence

- targeted S06: `17 passed`;
- complete Steiner suite: `120 passed, 1 expected PACE skip`;
- Python compilation, SCIP wrapper verification, validate-only, shell syntax
  and `git diff --check`: PASS;
- no formal S06 shard exists; activation record does not yet exist.

## Hashes at externally audited remediation head

- protocol YAML:
  `e7f7e9060c25fa038a2749ca43afe6a2769c3652e93697612b8804269750f827`;
- instance manifest:
  `b50f8048d8ab27a1fa2168e69ef179cbfc490116399180f2bb4a963bf04b292b`;
- environment lock:
  `f70afe548f2b640a3c1375686ad8c8ef4dced63d0229c9fa4eb36e62f6d7628e`;
- protocol explanation:
  `23139ea8810c608a6e816a1a4fd51d13ff3f48ca3f78c52bd4941c555e16dd48`;
- first remediation explanation:
  `fc4c4259c63ce29055c41ef113298cb37513692380c78d973e3770d811380fdd`;
- B2/B3 remediation explanation:
  `b922ba61f4a10efa9c01a4aa3894b1c3f4b78e51794f4d4c3f9a5f52abc789c6`;
- first focused conditional-audit source:
  `fa63f4cce005cccd78c872dad152680eaccec5ee521d6cc93ca7772b73eef611`;
- initial conditional-audit record:
  `142baa5fd1d6e7871e905ed5918d188bb60d9a0ecda29778685c510b8ab000d2`;
- first focused conditional-audit record:
  `6564c8753f2354e9b59b536ee6a2c62661ae9f69b0fe70e5763e51806202e82d`;
- online implementation:
  `d1f1a228aed066a7c32f3ce0202b3b502505e22e11b70daa6df246fda706de75`;
- distributed runner:
  `ed95e1e15ccc8dd8cde3b64b1a088a1b10a7fc6cb0dfd8dff69b1483eea9d3f7`;
- finalizer:
  `8b44c34adc312f83fdfc586c1a8ff21139f88215f43e41153cffadd442d3a954`;
- tests:
  `08e596c83b8ea76203cef16898e74d125daed1023b75030ff403bfd3c9e7aca7`.

Validate-only runtime fingerprint to bind in the post-PASS activation if the
runtime remains unchanged:
`fe7a032641aba58090f59cfa9d5b088cf76da23b799f2099a34e8d99b17b5687`.

The final focused audit PASS authorizes only a separate committed activation
and subsequent formal execution. It does not mark the scientific Gate PASS.

## Required audit decision

Recorded decision: **PASS; B1--B4 CLOSED; separate committed activation and six
main custom jobs authorized; scientific Gate NOT_EVALUATED; S07/test/final
unauthorized**.
