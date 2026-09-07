# S05 formal-v3 confirmatory Gate protocol

Status: **FROZEN FOR GPT PRE-EXECUTION AUDIT; NOT AUTHORIZED TO RUN**

## Why v3 exists

Formal-v2 trained all five registered models successfully, but its 320 Gate
states came from only 25 of the required 30 graph lineages. The aggregator
correctly stopped before formal statistics. GPT accepted that FAIL record and
allowed a new preregistered confirmatory evaluation.

V3 repairs only the validation-denominator problem. It does **not** change or
repeat model training. The five v2 seeds, checkpoints, 640 training states,
optimizer, 40 epochs, checkpoint selection, model architecture and
normalization remain frozen byte-for-byte. The v2 25-lineage numbers remain
diagnostic history and cannot enter the v3 Gate.

The v1 protocol's accidental prose reference to 20 bootstrap graphs is not
carried forward. V3 has one unambiguous denominator everywhere: exactly 30
fresh base-graph lineages, six per family.

## Risk-reduction design

Registering exactly 30 fresh graphs would repeat the failure mode: one graph
with zero valid teacher state would invalidate the whole matrix. V3 therefore
freezes 80 candidates, 16 per family, before any teacher or model access. Their
identities and generated graph hashes are committed in
`configs/steiner/experiments/s05_formal_v3_candidate_graphs.json`.

Every candidate receives the same three teacher seeds and a maximum of 16
states per task, for 240 tasks and 3,840 possible state slots. A lineage is
eligible only if all collection tasks are terminal, no task failed, every
observed action mapped, and the lineage has at least 12 semantic-unique valid
states across its three trajectories. Candidate selection then takes the first
six eligible lineages per family by the already frozen candidate rank and graph
hash. It may not inspect model logits, regret, teacher score magnitude or the
v2 family outcome.

The threshold of 12 is a preflight safeguard, not a relaxed quality Gate. Six
eligible lineages guarantee at least 72 valid state identities before selecting
the unchanged 64-state family quota. The final state selection round-robins
through all six lineages, so every selected lineage contributes at least one
state. Fewer than six eligible lineages or fewer than 64 unique states in any
family makes v3 FAIL before a checkpoint is loaded.

The registered scale envelope is deliberately limited to sizes that the prior
teacher evidence can evaluate reliably:

- sparse Erdős--Rényi: small and medium, interleaved by frozen rank;
- random geometric: small;
- grid with holes, community block and bridge bottleneck: medium.

This choice uses teacher-validity evidence, not model performance. In
particular, random-geometric small graphs produced the adverse v2 diagnostic
and remain fully in scope, so v3 does not avoid that negative signal. Any claim
is limited to these teacher-evaluable registered envelopes; v3 does not prove
large-scale or every family-by-scale generalization.

## Mandatory barrier before model access

After all 240 teacher tasks finish, the selector must produce and checksum-seal
one new manifest. Before any model or GPU evaluation, a fail-closed preflight
must prove all of the following:

1. exactly 30 selected graph hashes, exactly six per family;
2. all 30 are fresh and disjoint from train, validation-select, pilot, every
   formal-v1/v2 Gate lineage, test and final;
3. every selected lineage contributes at least one of exactly 320 selected
   states, exactly 64 per family;
4. every candidate/task/shard and both candidate/selected manifests pass their
   checksums, with no semantic duplicate or role/split leakage;
5. all five checkpoint manifests and payload hashes equal the frozen values;
6. no checkpoint has been loaded while candidate eligibility or state
   selection is being decided.

Any failed check stops v3 without GPU evaluation. There is no unregistered
replacement, repeated solving until success, threshold reduction or post-result
protocol patch.

## Confirmatory evaluation and Gate

Only after the barrier passes are the five existing checkpoints evaluated on
the same sealed 320 states. Each seed remains an independent one-GPU process;
no DDP or shared mutable state is allowed. The random baseline, checkpoint
reload error 0.0, graph-level paired effects, 10,000-replicate bootstrap seed,
all-seed direction, all-family direction and regret CV threshold 0.15 remain
the v1/v2 definitions.

The bootstrap unit is always one of the 30 selected base graph lineages. State
rows are never treated as independent bootstrap samples. A PASS requires every
seed and every family to improve over random, the primary 95% CI lower bound to
be above zero, exact checkpoint reload, and every identity/quality check.

Formal-v3 failure is retained as failure and stops before S06. Test/final
remain prohibited regardless of the result.

## Expected resource cost

The only expensive repeat is fresh teacher labeling: 240 CPU tasks rather than
the previous 315. On the observed formal-v1 workload, 315 tasks represented
25.55 serial task-hours, or about 4.26 ideal hours with six workers. V3 avoids
large instances, so the practical estimate is about **2--4 hours** on 24 CPU
cores and 128 GiB RAM with six one-thread workers. The registered 600-second
per-task limit gives a conservative all-timeout ceiling near 6.7 worker-hours
plus overhead.

There is no 40-epoch retraining cost. After sealing, five checkpoint-only
evaluation jobs may run in parallel; their runtime should be much shorter than
training. These estimates are operational guidance, not Gate conditions.

## Authorization boundary

This document, its YAML and the candidate manifest are preregistration only.
`execution_authorized` remains false. The allowed order is:

```text
freeze protocol and identities
-> independent GPT pre-execution PASS
-> separate activation record
-> implementation and tests
-> 240-task teacher collection
-> deterministic selection and seal
-> pre-model-access Gate
-> five frozen-checkpoint evaluations
-> one formal aggregation
```

No v3 teacher collection or model evaluation may start before the independent
PASS is recorded. V3 preregistration does not turn S05 into PASS and does not
authorize S06.
