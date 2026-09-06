# S05 Audit Packet — deterministic pilot handoff

## Identity and status

- branch: `research/steiner-migration`
- base SHA: `030199703c6e280533f1f1c7cfc8d00d7df0a6b0`
- latest remediation content SHA: `7fa85ff7b0d37f14d4223d396a49fb96138eb8cd`
- substantive range: `030199703c6e280533f1f1c7cfc8d00d7df0a6b0..7fa85ff7b0d37f14d4223d396a49fb96138eb8cd`
- S04 audit: first result CONDITIONAL PASS; remediation re-audit PASS, B1 CLOSED
- user authorization: S05 source/config/tests and scheduler-safe pilot job split
- S05 Gate: pilot-v3 pilot Gate PASS; full scientific Gate NOT_EVALUATED;
  no S05 local-gate/audited tag permitted
- teacher attempt 1: FAILED and retained; fraction-semantics remediation tested
- pilot-v1 retry: 10/10 completed, teacher-quality checks PASS, but only 53 valid
  train states; pilot-v2 capacity amendment preregistered before any GPU run
- pilot-v2: teacher capacity PASS; nine GPU runs failed exact reload parity;
  failed reports/checkpoints retained; pilot-v3 determinism remediation frozen
- pilot-v3: teacher 12/12 and training 9/9 completed; exact reload max error 0;
  aggregate primary regret improves through 64 states; formal protocol v1 is
  frozen for pre-audit but not authorized to run

This packet supports the pilot PASS only, not a request to approve the full S05
scientific Gate. S04 re-audit remains available at
`docs/steiner/audits/S00_S04_GPT_REAUDIT_REQUEST.md`.

Formal protocol v1 has now been frozen for a separate pre-execution audit at
`docs/steiner/audits/S05_FORMAL_PROTOCOL_GPT_AUDIT_REQUEST.md`. Its YAML keeps
`execution_authorized: false`; registration is not evidence that any formal
teacher or training task ran.

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

## Required future evidence

Before S05 can be audited for full PASS, obtain protocol pre-audit PASS, run and
attach the frozen formal teacher budget/config, five-seed formal report,
preregistered validation significance,
exact raw/checkpoint hashes (not bytes), resource/timing evidence and final S05
Gate JSON. Failed, root-solved, invalid, all-tie and skipped entries must remain
in denominators/manifests.

## Suggested current conclusion

Remediation implementation **PASS** and pilot-v3 pilot Gate **PASS**. Full S05
scientific Gate is **NOT_EVALUATED**. Do not create an S05 Gate tag or start S06;
the next authorized activity is formal-protocol preregistration, not training.
