# S06 Plan — IL online solve evaluation and RL entry decision

Status: ACTIVE / IMPLEMENTATION AUDIT IN PROGRESS / FORMAL RUN NOT AUTHORIZED

## Frozen start

- branch: `research/steiner-migration`
- base SHA: `ed0db3ff53a73cbb71de10f35e3471a6c139f146`
- S05 audited result head:
  `6cf7acab57525a744233ed3fdfd463f00fcd470c`
- required tag: `steiner-s05-audited-v3`, peeled target
  `6cf7acab57525a744233ed3fdfd463f00fcd470c`
- solver stack: `scip804-ecole081-pyscipopt430`
- formulation/schema/model: `rooted_mcf_v1` / `milp_bipartite_v1` /
  `b0_milp_gcnn_v1`
- representative IL checkpoint: formal seed 202, selected before formal-v3 Gate
- protocol family: P1 `controlled-branching-v1`; one SCIP thread per solve
- formal solver seeds: `[0, 1, 2, 3, 4]`
- statistics: paired instance/solver-seed observations, instance-level 10,000
  replicate bootstrap, seed `20260902`
- final-test access: prohibited; S12 remains the first authorized learned
  final-test stage
- observed resources at start: 80 CPUs, 128 GiB RAM, no swap, two Tesla
  V100-SXM2-32GB GPUs; formal execution is frozen as six independent CPU
  custom jobs, each requesting 8 CPUs and 96 GiB RAM with six workers

The starting worktree contains unrelated user-owned build outputs, frozen SCIP
prefix material, aviation scripts/results and untracked documents. They are
outside S06 and must not be staged, rewritten or deleted.

## Goal

Determine whether the S05 B0 imitation checkpoint's offline ranking gain
translates into a real branch-and-bound signal under the controlled P1 solve
protocol. The stage must measure end-to-end solver outcomes and learned-policy
overhead, retain every failure, and make a fail-closed decision on whether S07
RL work is scientifically justified.

## Non-goals

- no RL environment, replay, reward, DQN or BBMDP implementation;
- no model retraining, checkpoint reselection from S06 outcomes or B1 network;
- no P2 production-performance, SCIP-Jack/P3/P4 or C++ deployment claim;
- no test-IID, OOD-test, PACE-even, DIMACS final or other sealed final access;
- no aviation regression repair and no changes to legacy defaults;
- no Gate reduction, dropped instance/seed, hidden fallback or post-result
  instance substitution.

## Implementation and protocol work

1. Audit the repository's actual PySCIPOpt/Ecole online branching interfaces,
   S05 checkpoint loader and canonical probindex-to-edge identity. Do not assume
   an online B0 branchrule already exists.
2. Freeze a machine-readable S06 protocol before learned solve outcomes. It
   must bind the validation lineage set, methods, solver seeds, P1 limits,
   resource identity, representative seed-202 checkpoint bytes, metrics,
   pairing/statistics and failure rules.
3. Implement a fail-closed online B0 policy that scores exactly SCIP's current
   legal fractional binary `stp_x_*` candidates and branches on the selected
   SCIP variable. Model/feature/identity errors are run failures, not silent
   fallbacks.
4. Implement the required controlled baselines: B0-IL, random candidate,
   most-infeasible, relpscost and SCIP default. Run strong branching only on a
   preregistered affordable subset. Every method must use the same instance,
   solver seed, P1 limits and single-thread setting.
5. Record solved status, wall time, PAR-2, PDI, final gap, nodes, LP iterations,
   time to first incumbent, root gap, branch decisions, feature extraction,
   inference and total callback overhead, invalid action, mapping failure,
   fallback, NaN, timeout, node-limit, OOM and solver error.
6. Save compact paired trace evidence for preregistered diagnostic cases or
   failures: early branch choices, depth, dual bound and subtree size. Do not
   commit raw per-node logs.
7. Aggregate only a complete registered matrix. Use solved rate, PAR-2 and PDI
   in the frozen order; report instance-level bootstrap intervals, paired
   wins/losses and greater-than-2x catastrophic slowdown without deleting bad
   outcomes.
8. Produce the required phase documents and audit packet. Formal execution and
   any later S07 handoff require the applicable audit/Gate sequence.
9. Partition the complete matrix into six deterministic lineage shards using
   `instance_manifest_index modulo 6`. Each shard contains one lineage per
   family and all methods/seeds for each lineage. Main and trace waves use
   separate six-shard barriers before the only permitted final aggregation.

## Tests required before formal execution

- exact S05 tag/checkpoint/model/normalization/config SHA binding;
- config rejects unknown methods, seeds, lineages, limits and final selectors;
- action set equals canonical probindex reconstruction and edge mapping;
- permutation, duplicate, missing, continuous-variable and invalid-name cases
  fail closed;
- deterministic random and most-infeasible action choices on fixed candidates;
- learned logits finite, repeatable and aligned with the selected SCIP variable;
- real frozen-SCIP toy solves exercise every branchrule and preserve objective;
- no-candidate returns the registered legal SCIP control result;
- feature/inference/callback timers are monotone and internally consistent;
- timeout/node-limit/OOM/invalid-policy outcomes receive the registered PAR-2
  treatment and remain in the manifest;
- complete-matrix, pairing, bootstrap and catastrophic-slowdown aggregation;
- six-way partition disjointness/completeness, indivisible lineage assignment,
  duplicate-job lock and main/trace barrier failure tests;
- final-test and cross-role lineage guards;
- complete Steiner regression suite and unchanged S04/S05 identity/reload
  contracts.

## Gate and stop conditions

S06 can pass only if the frozen B0-IL policy is stable against the registered
random/most-infeasible weak baselines on real P1 solves and every correctness
check passes. Formal learned runs require zero invalid action, NaN, mapping
failure and unexpected fallback. Solved/unsolved outcomes, PAR-2/PDI, overhead
and catastrophic slowdowns must all remain visible.

If offline ranking is good but solve performance is worse, S06 stops and must
produce an explicit paired objective-mismatch analysis before any proposal to
use RL to repair the mismatch. If B0-IL does not beat either weak baseline with
stable evidence, S06 fails and S07 does not start.

The exact comparison statistic and minimum stability/effect requirements must
be frozen in the machine-readable S06 protocol before the formal matrix. They
cannot be chosen after observing pilot or formal learned-policy outcomes.

## External effects and Git policy

- Small config, source, tests, compact summaries and process documents may be
  committed on `research/steiner-migration`.
- Raw solve logs, per-node traces, generated instances, checkpoints, build
  outputs and large reports remain in ignored artifact directories.
- No stage branch, merge, rebase, amend, force-push or published-tag movement.
- Push is allowed only after a local Gate PASS and a fast-forward check, per the
  standing migration authorization.
- No S06 audited tag and no S07 implementation before an external S06 audit
  PASS.
