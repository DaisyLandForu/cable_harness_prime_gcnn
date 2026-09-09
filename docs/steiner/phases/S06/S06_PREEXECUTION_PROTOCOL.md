# S06 pre-execution protocol — IL online solve evaluation v1

Status: second narrow remediation after focused CONDITIONAL PASS; B2/B3
re-audit required; formal execution is not yet authorized.

## Scientific question

S05 established an offline strong-branch regret improvement for the frozen B0
imitation model. S06 asks a different question: does the preregistered seed-202
checkpoint improve real branch-and-bound outcomes under the controlled P1
profile? S06 does not retrain or reselect a model and does not access any
test/final split.

The formal comparison uses 30 validation-IID graph lineages already selected
by S05 formal-v3 before model access. Every lineage is solved with solver seeds
0--4 and the same five main methods: B0-IL, deterministic random candidate,
most-infeasible, relpscost and SCIP default. This gives 750 Gate-relevant
tasks. Full strong branching is diagnostic only on the first registered
lineage of each family, giving five additional tasks.

## Frozen solve semantics

- SCIP 8.0.4 / Ecole 0.8.1 / PySCIPOpt 4.3.0 only;
- rooted MCF v1 and the canonical probindex identity established in S04;
- P1: 600 seconds, 200,000 nodes, 8,192 MB, one SCIP thread;
- presolve and separation rounds disabled, heuristics off, restarts disabled;
- B0 inference is CPU-only, one Torch thread and deterministic;
- illegal/missing/misaligned actions, NaN logits and hidden fallback are errors;
- timeout, node limit, memory limit and solver error remain in the denominator;
- every policy/runtime exception is a terminal failed result with a stable
  failure class, `solved=false` and the frozen PAR-2 penalty of 1,200 seconds;
- PDI is explicitly unavailable for such a failed task rather than fabricated,
  so the correctness/PDI Gate fails while the PAR-2 pair remains present;
- no registered graph, solver seed or failed task may be dropped or replaced.

The primary comparison order is solved rate, PAR-2, then primal-dual integral.
Effects are paired within graph/solver-seed. The 10,000-replicate bootstrap
samples the 30 base graph lineages and keeps all five solver seeds together.
S06 must beat both frozen weak baselines, random and mostinf, under every
registered Gate condition. Trace replays are diagnostic and cannot rescue a
failed Gate.

## Six independent custom-job shards

The formal execution is exactly six independent CPU custom jobs. Assignment is
defined solely by the committed instance-manifest position:

```text
shard_index = instance_manifest_index modulo 6
```

Because the manifest has six consecutive lineages per family, each shard gets
exactly five lineages: one from every family. A lineage is indivisible. Its
five solver seeds and all five methods therefore remain on the same shard.
Every shard has exactly 125 main tasks; shard 0 additionally has the five
fullstrong diagnostic tasks. Trace tasks inherit the originating lineage's
shard.

Each job requests at least 8 effective CPU cores, at least 96 GiB RAM and no
visible GPU, runs six SCIP workers and uses the same frozen runtime,
repository executable content, activation record and shared persistent
artifact root. Before any task, the runner checks cgroup/affinity CPU capacity,
cgroup/host memory, CUDA visibility, the frozen environment lock, SCIP stack,
activation checksum, audited executable head and an activation-bound runtime
fingerprint. The exact runtime identity is written into every task envelope and
its shard manifest. Each shard has a distinct lock, log and manifest. The final
aggregator refuses to run unless all six main manifests and all six trace
manifests have the exact expected task identities, terminal shard files and
identical registered runtime identities. This prevents overlapping jobs,
missing pairs and accidental aggregation across incompatible environments.

Formal execution has two waves:

1. six `main` jobs produce the 750 main and five fullstrong terminal shards;
2. after the six-main barrier, six `trace` jobs deterministically reconstruct
   the trigger set and run their own disjoint trace subsets (including an
   explicit empty manifest when a shard has no trigger);
3. a single Python aggregation process acquires and holds a non-blocking OS
   file lock while it verifies both barriers and creates the Gate summary. It
   refuses to overwrite either an existing formal summary or an existing run
   manifest. The shell finalizer is only a convenience entrypoint; calling the
   Python entrypoint directly cannot bypass lock ownership.

Custom jobs run in the foreground because the platform scheduler owns process
lifetime. The tmux wrapper starts only one shard and is reserved for an
interactive host; it must never be used to start all 36 workers on the current
128-GiB machine.

## Pre-model and pre-execution barriers

The committed YAML keeps `execution_authorized: false`. Formal tasks can load
the seed-202 checkpoint only after a separate activation record states that an
external audit returned PASS for the exact content head and both frozen input
hashes. That activation must also bind the runtime fingerprint printed by
validate-only. The activation path must resolve inside the repository, exist in
Git at `HEAD`, and have local bytes exactly equal to `git show HEAD:<path>`.
The loader also rejects executable changes after the audited head and
uncommitted changes under the protected source/config/script/test paths.

The activation authorizes S06 formal execution only. It does not authorize
model retraining, B1/RL implementation, S07, or test/final access.

## Frozen identities

- protocol YAML SHA-256:
  `e7f7e9060c25fa038a2749ca43afe6a2769c3652e93697612b8804269750f827`
- instance manifest SHA-256:
  `b50f8048d8ab27a1fa2168e69ef179cbfc490116399180f2bb4a963bf04b292b`
- frozen environment lock SHA-256:
  `f70afe548f2b640a3c1375686ad8c8ef4dced63d0229c9fa4eb36e62f6d7628e`
- seed-202 manifest SHA-256:
  `b7f271c36243499e1510a5db19156afb3f607c13abf059c4f8349d49d0efb48c`
- seed-202 model SHA-256:
  `1b78d2b35a86829eb6afca801b86955a9e006ff678f224cb79fc400d1b42884f`
- seed-202 normalization SHA-256:
  `b0dcdca751525d9144cdaa4009bb99f542820addfe47f1bd90c64b4bee96a5b6`

## Claim boundary

A PASS can establish an online branching signal only within the registered
teacher-evaluable validation-IID synthetic envelope and P1 controlled profile.
It does not establish production SCIP performance, large-scale generalization,
SCIP-Jack superiority, RL benefit or final-test performance.
