# S05 Formal Protocol v1 — preregistration for GPT review

Status: **FROZEN FOR PRE-AUDIT; NOT AUTHORIZED TO RUN**

Machine-readable contract:
`configs/steiner/experiments/s05_teacher_il_formal_v1.yml`.
Frozen YAML SHA-256:
`c843f76d69c07b1c8a8093ab6f1426656084b2de2f4e9e0d777f1af32cd4c811`.

## Purpose and boundary

This protocol turns the completed S05 pilot into one formal teacher/IL
experiment. It freezes every choice that could otherwise be changed after
seeing results: graph lineages, teacher and training seeds, state quotas,
checkpoint selection, statistical unit, Gate thresholds, hardware class and
failure handling.

It does not authorize collection or training. Implementation and execution may
start only if the external protocol audit returns `PASS`. S06, test-IID,
test-OOD and final-test access remain prohibited regardless of this pre-audit.
The YAML itself stays immutable after review; a later committed GPT-PASS record
must reference its exact SHA-256 and supplies execution authorization without
silently editing the protocol.

## Why 640 train states

The eligible pilot used nested train prefixes 16/32/64. Aggregate normalized
strong-branch regret improved at every increase and was best at 64, so the
pilot showed useful signal but not saturation. Formal v1 therefore uses 640
train states: ten times the largest pilot point and exactly 128 states from
each of the five synthetic graph families. This is a bounded first formal
checkpoint, not a claim that 640 or 20,000 states is universally optimal.

The formal run is deliberately not a continuation of pilot checkpoints. Every
formal seed trains from a fresh deterministic initialization on the newly
selected 640-state dataset.

## Frozen collection matrix

- 60 train base graphs, 15 `validation_select` base graphs and 30
  `validation_gate` base graphs;
- every graph is run with teacher seeds 1001/1002/1003, producing exactly 315
  registered tasks and at most 5,040 observed states;
- each task uses the frozen P1/SCIP 8.0.4 controls and collects at most 16
  trajectory states;
- train generator seeds are 101000--101059; validation-IID seeds are
  201000--201044. They are disjoint from the pilot and remain inside their
  registered split ranges;
- only S03-supported graph/size buckets are used: sparse small/medium/large,
  geometric small/medium, grid medium/large, community medium, and bridge
  medium/large.

All tasks must reach a terminal manifest status. A root-solved or zero-state
task remains an honest completed task; an implementation/solver failure is a
Gate failure. There are no replacement seeds.

## State selection and leakage prevention

State selection occurs only after the 315-task manifest is complete. Duplicate
semantic states do not count twice. Valid states are selected against quotas
frozen in the YAML by cycling over every `(base graph lineage, teacher seed)`
trajectory at each state index, so one early high-yield trajectory cannot
silently dominate a bucket.

The final immutable datasets contain:

| role | base graphs | selected states | permitted use |
|---|---:|---:|---|
| train | 60 | 640 | normalization and optimization |
| validation_select | 15 | 160 | epoch/checkpoint selection only |
| validation_gate | 30 | 320 | one formal offline Gate evaluation |

The two validation roles share the registered `validation_iid` seed range but
are disjoint at base-graph lineage level. No state, terminal-set derivative,
weight derivative or duplicate content may cross roles. Normalization uses
train only. Test and final selectors are never opened.

## Frozen five-seed training

Training seeds are exactly 101, 202, 303, 404 and 505. Each seed runs 40 epochs
from scratch with the existing B0 architecture, loss, optimizer and
deterministic CUDA contract. Its checkpoint is the epoch with minimum
`validation_select` normalized regret; exact ties choose the earliest epoch.

All formal jobs use a Tesla V100-SXM2-32GB, one process per GPU. At most two
seed jobs run concurrently, so a two-V100 allocation executes two independent
jobs rather than data-parallelizing one model. Seed 202 is the predesignated
representative handoff checkpoint; `validation_gate` results cannot change
that choice.

## Statistical analysis and Gate

The primary paired effect is
`random normalized regret - model normalized regret`, so positive is better.
Within each of the 30 Gate base-graph lineages, states and teacher trajectories
are averaged first; the five training-seed effects are then averaged within
the lineage. A 95% paired percentile bootstrap resamples the 20 base graphs
10,000 times with seed 20260902. The primary lower confidence bound must exceed
zero. Percentile endpoints use the linear 0.025/0.975 quantiles. Random scores
are generated independently per semantic state from a SHA-256-derived seed and
applied to canonical probindex-ordered candidates, so state file order cannot
change the baseline.

In addition:

- all five training seeds must individually have lower mean Gate regret than
  the fixed random baseline;
- every graph family must have positive aggregate mean improvement;
- the coefficient of variation of the five seed-level mean regrets must be at
  most 0.15, using sample standard deviation (`ddof=1`) divided by their mean.
  This is preregistered as a conservative stability ceiling, about three times
  the pilot-64 observed value (~0.05). A zero mean passes only if all five
  regrets are exactly zero;
- most-infeasible and SCIP pseudocost ranking are reported as offline
  diagnostics, not substituted teacher labels or primary Gate references;
- Wilcoxon/Holm results and effect sizes are supplemental and cannot override
  the primary paired bootstrap result.

Teacher Gate remains unchanged: at least 60% valid states, at most 40% all-tie
among valid states, and 100% action mapping. Every frozen state quota must be
met. All five seed jobs, checksum reloads, exact state-dict reload and maximum
absolute logit reload error 0.0 are mandatory.

The 60% numerator is valid observed states and its denominator is all 5,040
registered state slots. The tie denominator is valid observed states before
semantic deduplication; the mapping denominator is every observed candidate
row. Deduplication affects quota eligibility, not these quality denominators.

## Stop rules

Any missing quota, task failure, seed failure, NaN, identity ambiguity,
mapping error, leakage, unexpected SCIP/CUDA stack, checksum mismatch,
nonzero reload error or failed statistical condition makes S05 `FAIL`. Failed
and skipped evidence is retained. The response is to diagnose and register a
new audited protocol version—not lower a Gate, add replacement instances,
substitute pseudocost labels, read final data or begin S06.

## Resource and execution envelope

- teacher collection: 24 CPU cores, 128 GiB RAM, no GPU, 6 single-threaded
  workers; worst-case task cap is 600 seconds;
- training: five independent one-V100 jobs, 8 CPU cores and 32 GiB RAM each;
  at most two jobs concurrently on the available two-V100 host;
- rough planning estimate, not a Gate: teacher 3--9 hours depending on hard
  task timeouts; each seed approximately 1--2 hours, or roughly 3--5 hours for
  five seeds with two GPUs after teacher completion.

Raw shards, checkpoints and per-state logs remain outside Git. Git records only
the protocol, code, tests, small aggregate reports and content hashes.
