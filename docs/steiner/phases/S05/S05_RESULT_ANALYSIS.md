# S05 Result Analysis — pilot-v2 failed; deterministic rerun pending

This checkpoint answers only whether the planned S05 pipeline is ready to run;
it does not answer whether imitation learning works.

## What is established

- Pilot-v1 expands deterministically to ten tasks and the separately
  preregistered pilot-v2 to twelve, without touching test/final ranges.
- A real frozen SCIP branch state supplies candidate-aligned, finite strong
  scores plus explicit child validity/cost evidence.
- Feature/label shards survive checksum-verified reload and preserve complete
  probindex-to-edge identity.
- Train-only normalization, listwise optimization, ranking metrics, baseline
  diagnostics and checkpoint reload are executable on synthetic test fixtures.
- The S04 audited tag prerequisite is satisfied. Pilot-v2 data/training are
  retained as failed evidence; pilot-v3 is pending, so no learned claim can be
  produced.

## Pilot-v1 teacher result

The corrected pilot-v1 teacher collection completed all 10 tasks in 627 seconds.
It produced 117/160 valid states (73.125%), no all-tie valid state, exact
3,086/3,086 action mapping and zero split leakage. These teacher-quality checks
pass their registered thresholds.

Only 53 valid states belonged to train; validation had 64. Validation data cannot
be moved into train, and the 64-state curve cannot be lowered silently. Thus v1
is capacity-insufficient for the complete learning curve despite sound teacher
quality. No GPU/model run was attempted.

## What is not established

- No training seed has yet produced a Gate-eligible completed report; v2
  histories/checkpoints are failure diagnostics rather than a learning claim.
- The implementation test's one real state is ABI/correctness evidence, not a
  dataset-quality estimate.
- Pseudocost is only an offline diagnostic. It is not the full relpscost solver
  behavior and must not be reported as such.
- Nothing here supports progression to S06 or any claim that learned branching
  beats random, most-infeasible, pseudocost or relpscost.

## Pilot-v2 GPU result

The v2 data-capacity amendment succeeded, but its GPU run is not an eligible
learned result. Seeds 101/202/303 each completed the 16/32/64-state, 40-epoch
matrix and wrote checkpoints. Every run then failed the unchanged exact reload
Gate due to non-deterministic CUDA logit differences. Checksums, state dicts and
normalization were intact, and repeated inference of the same in-memory model
showed the same noise. These metrics are diagnostic only and are not promoted
to completed results.

## Teacher attempt 1 (failed, retained)

The first CPU run at Git head `93984e5` completed 5/10 tasks. It retained 65/160
expected states, of which 56 were valid, with 0 all-tie valid states and exact
1,540/1,540 action mapping. Five tasks failed closed on an implementation check,
so the attempt is not eligible for teacher Gate evaluation.

The failure was not an identity or solver-resource failure. Ecole returned legal
LP candidates with `solution_frac=0.75`; the implementation incorrectly required
that feature to be at most 0.5. A remediated run must use one new Git fingerprint
and cannot combine the old attempt's successful shards with new-code shards.

## Current decision

Teacher-quality evidence: **PASS**. Latest complete GPU attempt: **FAIL**.
Complete S05 scientific Gate: **FAIL / STOP before S06**. Pilot-v3 is the only
authorized rerun and keeps the exact zero-error Gate; it must satisfy
`S05_PILOT_V3_DETERMINISM_REMEDIATION.md` before aggregation.
