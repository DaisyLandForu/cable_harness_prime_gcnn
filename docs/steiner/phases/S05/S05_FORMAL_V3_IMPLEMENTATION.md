# S05 formal-v3 execution implementation

## Scope

This implementation activates only the externally audited formal-v3 protocol.
It does not retrain or alter the five formal-v2 models. Formal-v1 and v2 remain
immutable failures, S06 remains blocked, and test/final data remain prohibited.

## Enforced execution order

1. The collector verifies the activation record, all frozen protocol/candidate
   hashes, SCIP 8.0.4 identity, S04 audited tag, committed executable identity,
   24 CPU/128 GiB availability and the exact 240-task expansion.
2. Six workers collect all 80 candidates x three teacher seeds. Every terminal,
   failed, invalid and all-tie result remains in the manifest denominator.
3. Only a teacher PASS permits deterministic selection. The selector verifies
   every shard checksum, takes the first six teacher-eligible lineages per
   family, selects 64 states per family by lineage round-robin, and checks the
   exact 30-lineage/320-state matrix with zero prior-role leakage.
4. The selected-manifest hash is written into a tracked selection seal. The
   evaluator refuses all checkpoint loading until that seal is committed
   byte-for-byte at `HEAD`.
5. Each of five independent one-GPU jobs loads one frozen checkpoint, evaluates
   the same sealed 320 states, and requires repeated-inference and checkpoint-
   reload maximum error to equal exactly 0.0.
6. Aggregation runs once over all five reports, bootstraps exactly 30 graph
   lineages and evaluates the frozen all-seed, all-family, CI and CV Gates.

## Failure behavior

All evidence writers refuse overwrite. A teacher, selection, barrier,
checkpoint, deterministic-inference or scientific-Gate failure is retained and
stops before S06. No retry-to-success, candidate replacement, model selection,
threshold reduction or test/final access is implemented.
