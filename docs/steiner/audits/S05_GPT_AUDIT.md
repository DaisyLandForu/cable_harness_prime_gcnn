# S05 GPT Result Audit — PASS

## Audit identity

- audit date: 2026-09-08 UTC
- reviewer: external GPT supplied by the user; exact model identifier not supplied
- branch: `research/steiner-migration`
- immutable result content head:
  `6cf7acab57525a744233ed3fdfd463f00fcd470c`
- result-audit request metadata head:
  `4c48352dc652925d8b4ff4d42b63ba89e675770a`
- result Gate summary SHA-256:
  `4aa035ad7538219da72fa4a8028223cf340181d5d1495539a051acf7601f742d`
- supplied audit text SHA-256:
  `e3be1a0b0232aa33de1b0cc152210271de7e68cd23dcb527a26bfe131eac3b87`
- machine-readable record:
  `docs/steiner/audits/S05_FORMAL_V3_RESULT_AUDIT_RECORD.json`

## Verdict

**PASS. Blocking findings: NONE.**

The reviewer accepts formal-v3's confirmatory scientific Gate and authorizes
S05 completion as **PASS via formal-v3**. The historical outcomes remain
unchanged: formal-v1 is FAIL, formal-v2 is FAIL and formal-v3 is PASS.

After this separate audit record is committed, an S05 audited tag may be
created and S06 implementation/handoff may begin. This verdict does not
authorize early access to any frozen test/final data.

## Accepted evidence

- Activation, implementation, committed selection seal and evaluation occurred
  in the required order.
- The pre-model barrier selected exactly 30 fresh lineages, six per family, and
  320 semantic-unique states before loading any checkpoint.
- Teacher collection completed 240/240 terminal tasks with zero failures,
  3,383/3,840 valid states and 69,709/69,709 mapped candidates.
- All five frozen model checkpoints completed the identical evaluation without
  retraining; repeat and reload error were exactly 0.0.
- The registered lineage-level bootstrap produced mean effect 0.1126971393 and
  95% CI `[0.0578104443, 0.1678878191]`; seed-regret CV was 0.0167927294.
  All five seed effects and all five family aggregate directions were positive.
- Eight negative graph effects were retained. Supplemental family-wise tests
  were correctly marked non-Gate and were not claimed as individually
  significant.
- The supported claim is limited to offline strong-branch regret improvement
  on the registered teacher-evaluable small/medium synthetic envelope. Online
  solve time, node count, relpscost superiority, large-scale generalization and
  final-test performance remain unestablished.

The reviewer independently checked the Git-visible protocol, ordering,
barriers, aggregation and committed summary. The 240 SCIP tasks, raw shard
replay, five V100 evaluations and bootstrap replay remain hash-bound runtime
evidence under the research contract rather than independently rerun evidence.

## Non-blocking findings

1. Mark stale `S05_RESULT_ANALYSIS.md` wording as a pre-formal-v3 historical
   assessment.
2. Treat `all_five_training_seeds_complete` as a historical field name: in v3
   it means all five frozen models were evaluated, not retrained. Do not rename
   the field inside the already hash-bound Gate summary.
3. Use the fixed Git result commit and committed evidence as the immutable
   authority; uploaded audit-packet copies are navigation aids only.

These findings do not change the verdict, Gate inputs, metrics or claim scope.
