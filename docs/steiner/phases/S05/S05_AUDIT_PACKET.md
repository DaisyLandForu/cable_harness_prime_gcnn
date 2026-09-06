# S05 Audit Packet — implementation-only handoff

## Identity and status

- branch: `research/steiner-migration`
- base SHA: `030199703c6e280533f1f1c7cfc8d00d7df0a6b0`
- latest remediation content SHA: `7fa85ff7b0d37f14d4223d396a49fb96138eb8cd`
- substantive range: `030199703c6e280533f1f1c7cfc8d00d7df0a6b0..7fa85ff7b0d37f14d4223d396a49fb96138eb8cd`
- S04 audit: first result CONDITIONAL PASS; remediation re-audit PASS, B1 CLOSED
- user authorization: S05 source/config/tests and scheduler-safe pilot job split
- S05 Gate: pilot-v2 FAIL; pilot-v3 NOT_RUN; no local-gate/audited tag permitted
- teacher attempt 1: FAILED and retained; fraction-semantics remediation tested
- pilot-v1 retry: 10/10 completed, teacher-quality checks PASS, but only 53 valid
  train states; pilot-v2 capacity amendment preregistered before any GPU run
- pilot-v2: teacher capacity PASS; nine GPU runs failed exact reload parity;
  failed reports/checkpoints retained; pilot-v3 determinism remediation frozen

This packet is an engineering review entry, not a request to approve S05
scientific results. S04 re-audit remains available at
`docs/steiner/audits/S00_S04_GPT_REAUDIT_REQUEST.md`.

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
| no premature run | required audited tag + CUDA checks | dry-run only; counters remain zero |

## Required future evidence

Before S05 can be audited for PASS, attach the completed teacher manifest,
training report for all pilot seeds and learning-curve sizes, exact raw/checkpoint
hashes (not bytes), GPU preflight, resource/timing summary, validation-vs-random
effect, reload parity, final S05 Gate JSON and phase docs. Failed, root-solved,
invalid, all-tie and skipped entries must remain in denominators/manifests.

## Suggested current conclusion

Remediation implementation **PASS**, latest scientific attempt **FAIL**. Do not
create an S05 Gate tag or start S06. Only the registered v3 rerun is authorized.
