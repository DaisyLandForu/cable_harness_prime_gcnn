# S05 formal v2 — teacher quota feasibility remediation

Status: **FROZEN FOR GPT PRE-EXECUTION AUDIT; NOT AUTHORIZED**

## What failed

Formal-v1 completed all 315 registered teacher tasks without a solver failure.
It observed 3,431 states, of which 3,207 were fully valid, and mapped all
88,549 candidate rows. Valid fraction, all-tie fraction, mapping, terminal-task
and leakage checks passed. The formal-v1 teacher Gate nevertheless remains
`FAIL` because its exact per-bucket train quotas selected only 527 of the
required 640 states.

The shortages were 48 sparse-large states (`0/48`), 61 geometric-medium states
(`3/64`) and four bridge-medium states (`60/64`). Sparse-large and
geometric-medium jobs commonly consumed the 600-second task budget while some
candidate child LPs did not become valid within the fixed 10,000-iteration
strong-branch limit. This is a data-yield/protocol-feasibility failure, not a
task crash, mapping failure or split leak.

Formal-v1 and its manifest SHA-256 remain immutable evidence of that failure.
Running the same deterministic task/seed matrix again would reproduce the same
feasibility problem and is not a valid repair.

## Preregistered repair

V2 keeps 640 train states and exactly 128 states per graph family. It first
selects as many states as available under every original bucket target. If a
family is short of 128, it fills the deficit only from unused valid states of
the same family using the frozen ordering:

```text
state_index, bucket_order, graph_sha256, teacher_seed, semantic_sha256
```

No teacher score, model result or validation result participates in fallback.
Applied to the sealed v1 manifest, the deterministic train composition is:

| family | selected bucket counts | total |
|---|---|---:|
| sparse Erdős–Rényi | small 56, medium 72, large 0 | 128 |
| random geometric | small 125, medium 3 | 128 |
| grid with holes | medium 64, large 64 | 128 |
| community block | medium 128 | 128 |
| bridge bottleneck | medium 60, large 68 | 128 |

This does not claim balanced scale coverage. In particular, S05 training cannot
support a claim about sparse-large or broadly about geometric-medium behavior.
Later scale generalization must still be evaluated only in a separately frozen
stage; test/final remains unopened.

## Evidence reuse and unchanged rules

V2 reuses the completed teacher evidence instead of rerunning it. Reselection
must verify the exact failed-manifest SHA-256 and every referenced NPZ SHA-256,
retain every v1 task/shard, and write a separate v2 selection manifest. No v1
file is overwritten and no replacement graph or seed is introduced.

Validation-select and validation-gate membership and counts remain unchanged.
The model, 40 epochs, optimizer, deterministic CUDA contract, five training
seeds, checkpoint rule, 10,000-replicate lineage bootstrap, seed/family/CV
Gates, exact reload 0.0, representative seed 202 and all failure rules remain
unchanged. Full S05 and S06 remain unauthorized.

The previously pending concurrency A1 is superseded before activation and
incorporated here: after v2 audit PASS, the five registered seeds may run as
five independent one-V100 scheduler jobs. This changes wall-clock overlap only;
there is no DDP or shared model/optimizer state.

## Required authorization

No v2 reselection or formal training may start until an external GPT audit
returns `PASS` for the exact v2 YAML/Markdown hashes and that result is recorded
in a separate commit. If the audit rejects within-family fallback as an
unacceptable weakening of bucket coverage, S05 stays failed and the next
revision must instead change the teacher formulation/budget and collect new
evidence; the existing Gate will not be silently lowered.
