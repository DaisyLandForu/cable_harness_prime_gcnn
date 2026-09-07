# S05 Plan — Strong-branch teacher and B0 imitation learning

Status: S04 audit prerequisite PASS; pilot-v3 PASS; formal-v1/v2 FAIL retained;
formal-v3 local Gate PASS and external result audit pending; S06 blocked

## Frozen start

- branch: `research/steiner-migration`
- base / S04 remediation phase head:
  `030199703c6e280533f1f1c7cfc8d00d7df0a6b0`
- S04 remediation content head:
  `4ab54ffa2b80f06ac8a9ecfe662a04df7899b072`
- solver stack: `scip804-ecole081-pyscipopt430`
- representation/model: `milp_bipartite_v1` / `b0_milp_gcnn_v1`
- protocol: P1; one SCIP thread per worker
- split policy: instance/base-graph lineage only; no state-level split
- teacher seeds: pilot `[1001]`; formal remains frozen `[1001,1002,1003]`
- training seeds: pilot `[101,202,303]`; formal remains frozen
  `[101,202,303,404,505]`
- final-test access: prohibited; learning runs at start: 0

The first S04 audit returned CONDITIONAL PASS. The probindex remediation was
subsequently re-audited as PASS with B1 closed, and
`steiner-s04-audited-v2` anchors the S04 phase head. S05 collection/training is
therefore audit-eligible, but it still requires resource/CUDA preflight and does
not authorize final-test access or an S05 Gate claim before results exist.

The starting worktree also contains unrelated artifact/build/aviation/user
changes. They remain outside this stage and must not be staged.

## Goal

Prepare a fail-closed, resumable pipeline that can later collect strong-branch
labels for legal Steiner edge actions and train the standard B0 listwise
imitation baseline without changing the S04 state/action contract.

## Implementation scope

1. strict S05 pilot configuration using only registered train/validation seed
   ranges and S03's branchable family/bucket guidance;
2. frozen SCIP 8.0.4 strong-branch extraction with per-child validity,
   infeasibility, LP-error, cost and candidate probindex/edge identity;
3. one checksum-addressed state shard per sampled node plus an atomic run
   manifest; cached shards are reused only after full fingerprint validation;
4. shard loading that rejects corruption, split-lineage overlap, identity drift,
   NaN labels and state-level reassignment;
5. train-only feature normalization and a listwise cross-entropy objective over
   valid candidate lists;
6. top-k, tie-aware Spearman rank correlation and normalized strong-branch
   regret, with random/most-infeasible/pseudocost offline diagnostics;
7. multi-seed training, checkpoint/normalization checksums and a reload manifest
   sufficient to reproduce logits;
8. detached tmux launchers for interactive hosts plus foreground launchers for
   scheduler-managed CPU/GPU jobs;
9. disjoint pilot-seed reports and fail-closed aggregation, so two one-GPU jobs
   can run concurrently without overwriting checkpoints or reports.

SCIP's Ecole `Pseudocosts` signal is recorded as the honest offline cheap-score
diagnostic; it is not mislabeled as the full reliability-pseudocost branching
rule. The real `relpscost` solver comparison remains an online S06 evaluation.

## Preregistered pilot shape

Pilot-v1 covered all five synthetic families, with one S03-recommended
medium-mid graph per family in each of train and validation-IID. It follows a
teacher trajectory, collects at most 16 states per task, and retains root-solved,
missing, invalid-child, all-tie, timeout and failed tasks in the manifest.

Pilot-v1 completed with teacher-quality diagnostics passing but only 53 valid
train states. Before any GPU/model run, the approved pilot-v2 capacity amendment
added train seeds 100303 (sparse) and 100315 (grid), chosen from S03's existing
branchability evidence. V1 and v2 use separate experiment IDs, paths and hashes.
See `S05_PILOT_V2_AMENDMENT.md`.

Pilot-v2 subsequently collected 85 valid train and 64 valid validation states,
but all nine GPU curve/seed runs failed the exact checkpoint-logit reload check
because CUDA determinism was incomplete. Pilot-v3 keeps the exact Gate and all
experimental choices, adds the complete cuBLAS/PyTorch determinism contract,
and writes to independent paths. See
`S05_PILOT_V3_DETERMINISM_REMEDIATION.md`.

The learning curve remains nested train-state prefixes `[16, 32, 64]` and
training seeds `[101,202,303]`. These are engineering/pilot runs, not formal
results. A later formal state count may be frozen only after the pilot analysis;
it cannot be chosen with final-test evidence.

## Verification before any long run

- strict config rejects unknown fields, unregistered seeds and final-test ranges;
- candidate probindices exactly equal Ecole `action_set` and map 100% to edge IDs;
- permutation/missing/duplicate/out-of-range identities fail closed;
- invalid child flags and failed/skipped tasks remain serializable and reloadable;
- shard byte checksum and semantic state checksum are both verified on resume;
- train/validation graph lineages are disjoint;
- normalization sees train shards only;
- listwise loss/metrics handle ties and reject invalid/non-finite labels;
- checkpoint reload reproduces logits exactly on a fixed state;
- real CUDA repeated inference and checkpoint reload are bit exact under the
  registered cuBLAS/PyTorch determinism contract;
- CPU-only unit/integration tests pass before any GPU command is handed off.

## Future Gate and stop conditions

S05 is not eligible for PASS in this implementation-only turn. Once authorized,
the formal Gate will require: teacher valid-state fraction at least 60%, all-tie
valid-state fraction at most 40%, 100% action mapping, validation normalized SB
regret better than random for every reported training seed with stable aggregate
behavior, zero split leakage, and a checksum-verified reload manifest.

Any identity ambiguity, unexpected solver stack, corrupt/missing shard, NaN,
invalid action, leakage, or failed seed remains recorded and stops Gate
aggregation. The pipeline must not add weak pseudocost labels to disguise an
invalid strong teacher.

Pilot-v3 now satisfies the pilot checks for all three seeds and all curve sizes.
Because the aggregate primary metric still improves at the largest tested size,
the pilot does not establish saturation. Formal protocol v1 now preregisters
640 train, 160 checkpoint-selection validation and 320 held-out Gate validation
states; 105 base graphs / 315 teacher tasks; formal training seeds
`[101,202,303,404,505]`; and a base-graph paired bootstrap Gate. See
`S05_FORMAL_PROTOCOL.md` and
`configs/steiner/experiments/s05_teacher_il_formal_v1.yml`.

The protocol remains deliberately non-executable (`execution_authorized:
false`) until external GPT pre-audit returns PASS. No formal data or model is
created by this registration, and S06 remains blocked until the full S05 Gate
passes.

## External effects

This turn may commit/push source, small configs, tests and process documents only
after CPU verification. Raw shards, solver logs, checkpoints, normalization
artifacts and tmux logs stay under ignored directories and are not committed.
No tag, PR, merge, rebase, amend, force push or formal experiment is authorized.

## Execution addendum

The subsequently audited formal-v3 confirmatory protocol completed without
model retraining. Its fresh teacher and exact 30-lineage validation barrier
passed, all five frozen checkpoints reloaded exactly, and the local scientific
Gate passed. See `S05_FORMAL_V3_GATE_SUMMARY.json` and
`S05_RESULT_ANALYSIS.md`. This addendum records the outcome without changing the
historical preregistration text above. External result audit remains mandatory
before an S05 audited tag or S06 handoff.
