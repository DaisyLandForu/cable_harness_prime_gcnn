# S06 main-wave barrier failure and execution amendment A1

Status: main evidence sealed; amendment preregistered; external review required
before trace or aggregation.

## What completed

The six registered custom jobs produced all 755 terminal envelopes:

- 750 Gate-relevant main tasks;
- five non-Gate fullstrong diagnostics;
- six completed main shard manifests;
- 399 solver outcomes marked `optimal` and 356 marked `timelimit`;
- zero task envelopes marked `solver_error`.

These are completion/status counts only. No learned-versus-baseline effect,
trace-trigger set or scientific Gate was computed before this amendment.

The 755 task envelopes and six main manifests are sealed by
`S06_MAIN_WAVE_V1_SEAL.json`. Its 761-file evidence-tree SHA-256 is:

```text
671cb10a78a30e9f227e6b8b86a62d3ba4437a013ba9350850bb2ef1b1fb8964
```

No main task may be rerun, replaced or edited under amendment A1. Downstream
execution first recomputes the complete tree and fails if any registered file
or byte changed.

## Why the original barrier stopped

Every job received the same registered effective request and runtime:

- CPU quota/effective CPU: `8.01` cores;
- memory limit: `103080263680` bytes (at least the registered 96 GiB);
- visible GPUs: zero;
- Python 3.11.15 and frozen SCIP 8.0.4 stack;
- identical environment lock, container-runtime fingerprint, activation,
  audited executable head and execution Git head.

The scheduler nevertheless placed the six jobs on three physical CPU models.
Their host affinity masks exposed 24, 48 or 320 physical/logical CPUs even
though the cgroup quota restricted every job to 8.01 effective cores. The
original implementation incorrectly required `cpu_model` and
`cpu_affinity_count` to be equal across independent jobs, although neither
field was a registered resource request or scientific protocol variable. The
barrier therefore stopped before trace generation.

This is an orchestration identity bug, not evidence that a solver task failed.
Within every graph lineage, all five methods and all five solver seeds stayed
on the same shard and used the exact same recorded runtime identity. The
registered paired comparison unit is therefore unchanged.

## Minimal amendment

`s06_execution_amendment_a1.yml` changes only the cross-shard compatibility
tuple:

- effective quota/RAM/GPU and frozen software/activation/code identities must
  still be equal across all six shards;
- hostname, physical CPU model and host affinity count remain recorded, but
  are no longer cross-shard equality keys;
- every task envelope must still byte-match the runtime identity in its own
  shard manifest;
- the sealed main evidence is reused byte-exactly and cannot be rerun;
- trace must use a new, separately audited and committed amendment activation.

No checkpoint, graph, solver seed, method, time/node limit, metric, pairing,
bootstrap rule, Gate threshold or claim boundary changes.

## Fail-closed order

```text
seal existing 761 files
  -> externally audit amendment A1
  -> commit separate PASS activation
  -> verify seal and normalized main barrier
  -> derive diagnostic trace trigger
  -> run six trace shards
  -> require complete trace barrier
  -> aggregate once and evaluate the unchanged S06 Gate
```

Until the external review returns PASS, trace, aggregation, S07 and test/final
access remain unauthorized.
