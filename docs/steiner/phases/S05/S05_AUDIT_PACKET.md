# S05 Audit Packet — implementation-only handoff

## Identity and status

- branch: `research/steiner-migration`
- base SHA: `030199703c6e280533f1f1c7cfc8d00d7df0a6b0`
- content SHA: filled by the metadata commit after implementation commit
- substantive range: base through the S05 implementation content commit
- S04 audit: first result CONDITIONAL PASS; remediation re-audit PENDING
- user waiver: source/config/tests/tmux only
- S05 Gate: NOT_RUN; no local-gate or audited tag is permitted

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
| interrupted long jobs | two tmux launchers | shell syntax and path review |
| no premature run | required audited tag + CUDA checks | dry-run only; counters remain zero |

## Required future evidence

Before S05 can be audited for PASS, attach the completed teacher manifest,
training report for all pilot seeds and learning-curve sizes, exact raw/checkpoint
hashes (not bytes), GPU preflight, resource/timing summary, validation-vs-random
effect, reload parity, final S05 Gate JSON and phase docs. Failed, root-solved,
invalid, all-tie and skipped entries must remain in denominators/manifests.

## Suggested current conclusion

Implementation-only **PASS**, scientific stage **NOT_RUN**. Do not create an S05
Gate tag, start S06, or treat the absence of a run as a negative experiment.
