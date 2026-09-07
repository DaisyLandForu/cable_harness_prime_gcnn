# S05 Audit Packet — formal-v3 pre-execution preregistration

## Identity and status

- branch: `research/steiner-migration`
- base SHA: `030199703c6e280533f1f1c7cfc8d00d7df0a6b0`
- current immutable result: formal-v2 **FAIL**, accepted by external GPT audit;
  S05 remains FAIL and S06 remains blocked
- formal-v3 status: protocol/candidate identities frozen locally for GPT
  pre-execution audit; `execution_authorized=false`; no v3 run
- formal-v3 YAML:
  `configs/steiner/experiments/s05_teacher_il_formal_v3_confirmatory_gate.yml`
  (`99e75a4d4fa69f805232c637d4fc0ae750fcfccb57e9979b561f422591006242`)
- formal-v3 explanation:
  `docs/steiner/phases/S05/S05_FORMAL_V3_CONFIRMATORY_GATE.md`
  (`c61b58d23f4de6bb2709aa9c40ccab11d6db4507a6bd1c9d2e544eb68595d873`)
- formal-v3 candidate identities:
  `configs/steiner/experiments/s05_formal_v3_candidate_graphs.json`
  (`e65fc9a03fd683277570befe13b11f4b15ea981ee427568d56aeeccdd3847b56`)
- formal-v2 failure-evidence content SHA:
  `5626517c3689a1e885a69ac7f111f587864f9924`
- formal protocol v1 content SHA: `d1717a7ecb6043efd71678175a92325ac9ff4208`
- formal protocol pre-execution audit: user reports GPT PASS; activation record
  `docs/steiner/audits/S05_FORMAL_PROTOCOL_AUDIT_RECORD.json`
- formal-v1 teacher: 315/315 tasks completed, no task failures, but Gate FAIL
  because train bucket quotas reached only 527/640; no formal training started
- execution-only concurrency amendment A1: superseded before activation by the
  formal-v2 selection remediation, which incorporates five-job concurrency
- substantive range:
  `030199703c6e280533f1f1c7cfc8d00d7df0a6b0..5626517c3689a1e885a69ac7f111f587864f9924`
- S04 audit: first result CONDITIONAL PASS; remediation re-audit PASS, B1 CLOSED
- user authorization: S05 source/config/tests and scheduler-safe pilot job split
- S05 Gate: pilot-v3 pilot Gate PASS; formal-v1 teacher Gate FAIL; formal-v2
  full scientific Gate FAIL at 25/30 validation lineages; no S05
  local-gate/audited tag permitted
- teacher attempt 1: FAILED and retained; fraction-semantics remediation tested
- pilot-v1 retry: 10/10 completed, teacher-quality checks PASS, but only 53 valid
  train states; pilot-v2 capacity amendment preregistered before any GPU run
- pilot-v2: teacher capacity PASS; nine GPU runs failed exact reload parity;
  failed reports/checkpoints retained; pilot-v3 determinism remediation frozen
- pilot-v3: teacher 12/12 and training 9/9 completed; exact reload max error 0;
  aggregate primary regret improves through 64 states; formal protocol v1 is
  frozen for pre-audit but not authorized to run

This packet now records the full formal-v2 execution as a fail-closed S05
result. The review request is
`docs/steiner/audits/S05_FORMAL_V2_RESULT_GPT_AUDIT_REQUEST.md`.

GPT returned PASS only for that failure registration and allowed v3
preregistration. The separate machine record is
`docs/steiner/audits/S05_FORMAL_V2_RESULT_AUDIT_RECORD.json`. V3 freezes the
existing five model payloads, an 80-graph fresh candidate pool, a
teacher-validity-only first-six-per-family policy and an exact 30-lineage
pre-model-access barrier. The old 25 lineages cannot be reused. Independent GPT
pre-execution PASS is still required before implementation or collection.

Formal protocol v1 has now been frozen for a separate pre-execution audit at
`docs/steiner/audits/S05_FORMAL_PROTOCOL_GPT_AUDIT_REQUEST.md`. Its YAML keeps
`execution_authorized: false`; registration is not evidence that any formal
teacher or training task ran.

The user subsequently reported a GPT `PASS` authorizing formal implementation
and execution. The activation record binds that verdict to the frozen YAML and
protocol hashes, keeps the complete S05 Gate unevaluated, and does not authorize
S06. Formal code requires that record and the byte-exact audited YAML.

The scheduler was later clarified to allocate independent GPUs to independent
custom jobs rather than limiting the experiment to the current two-GPU host.
`S05_FORMAL_CONCURRENCY_AMENDMENT_A1.md` preregisters the sole scheduling
override from two to five concurrent one-V100 seed jobs. It changes no teacher,
training, validation, statistics or Gate semantics and is not active before an
external GPT PASS is separately recorded.

Formal-v1 subsequently completed with every quality/identity check passing but
`all_state_quotas_met=false`. The immutable failure record is
`S05_FORMAL_V1_TEACHER_FAILURE.md`. Formal-v2 preregisters deterministic
within-family fallback against the sealed failed manifest while preserving
640 train states and 128 per family. A1 is not activated separately; v2 folds
in five independent one-V100 jobs. No v2 selection or training is authorized
before its own GPT PASS.

The first v2 audit returned `CONDITIONAL PASS` with one blocker: the YAML named
an ordering key `canonical_instance_content_sha256` although the sealed record
schema uses `graph_sha256`. The B1 remediation changes only that literal key in
primary/fallback order, documents the no-alias rule, and adds schema,
permutation, exact-composition and unknown-key fail-closed tests. Execution
remains unauthorized pending focused re-audit.

## Review map

| Concern | Implementation | Evidence |
|---|---|---|
| registered tasks/seeds/splits | S05 YAML + `teacher_data.py` | strict config/dry-run tests |
| child-valid strong labels | `solver/strong_branching.py` | frozen SCIP + Ecole score parity |
| canonical action identity | probindex alignment + S04 bridge | permutation/bijection/real mapping tests |
| resumable data | task manifests + NPZ checksums | round-trip/corruption tests |
| no state split leakage | graph hash inherited split | cross-split negative test |
| train-only normalization | `learning/imitation.py` | validation rejection test |
| listwise objective/metrics | same | gradient/regret/rank/top-k tests |
| checkpoint reproduction | checksum manifest + strict reload | bit-exact state-dict reload |
| CUDA determinism | fixed cuBLAS env + PyTorch deterministic algorithms | real V100 repeat/reload exact test |
| interrupted long jobs | two tmux launchers | shell syntax and path review |
| scheduler parallelism | seed-selecting foreground launcher + strict aggregator | disjoint/duplicate/missing matrix tests |
| no premature formal run | required audited tag + CUDA checks | pilot artifacts say `formal_gate_evaluated=false` |

## Formal-v2 evidence

- execution/seal head: `da5b5abd8220f8bdaa1a977455d25ef9aaaa84b7`
- selection manifest SHA-256:
  `35221abeeaae507623d0175d895b5ff807d0b7e5400fce494e615fbefe8cd10e`
- five seeds: 5/5 complete, 40 epochs each, reload error 0.0
- registered/observed valid Gate lineages: 30/25
- missing: sparse 201019/201020; geometric 201024/201025/201026
- machine summary: `S05_FORMAL_V2_GATE_SUMMARY.json`
- test/final accessed: no
- Gate: **FAIL**; S06 authorized: no

## Required future evidence

Before S05 can be audited for full PASS, obtain v3 protocol pre-execution PASS,
record a separate activation, implement and test the collector/selector/barrier,
run the frozen 240-task teacher candidate budget, seal exactly 30 lineages and
320 states, then evaluate the five frozen checkpoints. Attach exact raw,
selection, report and checkpoint hashes, resource/timing evidence and final S05
Gate JSON. Failed, root-solved, invalid, all-tie and skipped entries must remain
in denominators/manifests.

## Suggested current conclusion

Pilot-v3 pilot Gate **PASS**; formal-v2 full S05 Gate **FAIL**. Formal-v3 is
preregistered but not executable. Do not create an S05 tag or start S06. The
only permitted next action is the independent v3 pre-execution audit.
