# S06 Result Analysis

Status: formal main wave terminal and sealed; trace NOT_RUN; scientific Gate
NOT_EVALUATED. S07 remains prohibited.

S06 asks whether the S05 B0 imitation model's offline ranking improvement
becomes a real branch-and-bound improvement. The registered six-job main wave
has now produced all 750 Gate-relevant task envelopes plus five fullstrong
diagnostics. At the solver-status level there are 399 optimal and 356 timelimit
outcomes, with zero solver-error envelopes.

Those counts are not a learned-policy result. They do not say whether B0-IL is
better or worse than random/mostinf, because the fail-closed barrier stopped
before paired effects, confidence intervals, the trace-trigger set or the
scientific Gate were calculated.

## Barrier finding

All six jobs used the same registered effective resources and frozen runtime,
but the scheduler placed them on three Xeon host models with affinity masks of
24, 48 and 320 CPUs. The original barrier incorrectly demanded that physical
CPU model and host affinity be identical across independent jobs. Each cgroup
actually provided the same 8.01 effective CPU quota and the same memory, GPU,
software, activation and code identities.

This did not break the registered paired design: every graph lineage's five
methods and five solver seeds stayed together on one shard. It did expose an
implementation overconstraint that must be audited before the evidence can be
used downstream.

## Evidence-preserving response

The existing 755 task envelopes and six main manifests are immutable under the
761-file tree root:

```text
671cb10a78a30e9f227e6b8b86a62d3ba4437a013ba9350850bb2ef1b1fb8964
```

Amendment A1 does not retry hard instances or change the solver/model/Gate. It
only defines the correct cross-shard identity: effective registered resources
and frozen runtime/code must match, while hostname, physical CPU model and host
affinity remain recorded but need not be equal. The implementation refuses to
rerun main and verifies every sealed byte before deriving trace tasks.

## What remains unknown

Until amendment A1 receives an external PASS, a separate activation is
committed, all six diagnostic trace manifests terminate, and the singleton
aggregator runs, there is no valid S06 solved-rate, PAR-2, PDI, node, overhead,
confidence-interval or Gate conclusion.

The eventual result may support only a P1 controlled-profile online branching
claim inside the registered synthetic small/medium envelope. It cannot by
itself establish production SCIP speedup, large-scale/OOD generalization,
SCIP-Jack superiority, RL benefit or final-test performance.

Current Gate: **NOT_EVALUATED**.
