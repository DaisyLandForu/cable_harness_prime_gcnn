# S05 pilot-v2 preregistration amendment

Date: 2026-09-06 UTC

## Reason fixed before v2 execution

Pilot-v1 completed all 10 teacher tasks at Git head
`c26a32690813f9867578a8cc784a09a5efda2849`. Its teacher-quality diagnostics
passed, but it produced only 53 valid train states, fewer than the already
registered 64-state learning-curve checkpoint. No GPU training or model metric
was run or inspected before this amendment.

The v1 manifest is retained at
`results/steiner/raw/s05/s05-teacher-il-pilot-v1/manifest.json`, SHA-256
`90ed3cad41ec1cc88891b01f18c85ce713d18aa5f00974da207d07af23a6c430`.
It is evidence of a completed but capacity-insufficient pilot and will not be
mixed into the v2 manifest.

## Frozen capacity extension

Pilot-v2 contains the original ten tasks plus exactly two train tasks:

| family | bucket | nodes/terminals | generator seed | S03 relpscost decisions |
|---|---|---:|---:|---:|
| sparse Erdős--Rényi | medium-mid | 96/19 | 100303 | 3006 |
| grid with holes | medium-mid | 96/19 | 100315 | 2227 |

Both choices come from the already completed S03 train matrix, not from S05
model performance or final/test data. Their committed S03 shard SHA-256 values
are respectively
`71d1d061a8a36df4546082f513673dbd7da17f38bdaf180454b7c8f138fd5e50`
and
`c6353154b9ae44ee000ba752dee06b6ab3d81cbe631a2bd7f84a5c938109303a`.

- config: `configs/steiner/experiments/s05_teacher_il_pilot_v2.yml`
- config file SHA-256:
  `a385c02dc09d13bf6daf94b5dc83d82f5438639c9a107b3aed10ccafebe6b0c4`
- canonical config SHA-256:
  `2146e7d67dadcef93746400a08d10441e051745075fc7a39348c5b6c80b6cacf`
- content commit: `e9722cf84a51433af06c98318a401b51ce2f15c2`
- task count: 12; maximum expected states: 192
- maximum train states: 112; maximum validation states: 80

## Unchanged contract

Teacher seed 1001, training seeds 101/202/303, learning-curve counts 16/32/64,
SCIP 8.0.4/P1 controls, 16-state task cap, 19/5/1 schema, split policy, strong
iteration budget and every Gate threshold remain unchanged. No validation state
may enter normalization/training and final/test remains sealed.

The v2 collection is accepted for GPU handoff only if all 12 tasks complete,
the existing teacher Gate diagnostics pass, checksums and split lineage pass,
and at least 64 valid train states exist. If it still produces fewer than 64,
S05 stops for a new audit decision; no further instance addition or lower curve
size is automatic.
