# S05 formal execution amendment A1 — five independent seed jobs

Status: **FROZEN FOR GPT PRE-EXECUTION AUDIT; NOT YET AUTHORIZED**

Machine-readable amendment:
`configs/steiner/experiments/s05_teacher_il_formal_v1_concurrency_a1.yml`.

## Single permitted change

The audited S05 formal-v1 protocol assumed one host with two V100 GPUs and
limited global training concurrency to two jobs. The actual scheduler can
allocate a separate GPU to every custom job. A1 therefore changes only
`training.max_concurrent_training_jobs` from `2` to `5`.

This is a wall-clock scheduling change, not multi-GPU training. Each of seeds
101, 202, 303, 404 and 505 remains a fresh, independent, single-process run on
one Tesla V100-SXM2-32GB with 8 CPU cores and 32 GiB RAM. DDP, shared model or
optimizer state, changed batch semantics, seed replacement and result-based
reruns remain prohibited. Each seed keeps its own report and checkpoint path.

## Everything else remains frozen

The base protocol YAML and explanation stay byte-exact at their audited
SHA-256 values. A1 does not change teacher tasks or data, five training seeds,
640 selected train states, 40 epochs, optimizer settings, deterministic CUDA
contract, checkpoint selection, validation roles, bootstrap, Gate thresholds,
failure retention, representative seed 202, or the prohibition on test/final
access and S06.

The 315-task teacher collection was already started under the audited base
protocol at formal run head `13a8a83761f660c96cb238471de2e2c0a2d54a8b`.
Teacher execution is unaffected by A1. Seeds 101 and 202 may run concurrently
under the already-authorized base limit after the teacher Gate passes. Seeds
303, 404 and 505 must not raise global concurrency above two until A1 receives
an external GPT `PASS` and that PASS is recorded in a separate committed
activation record.

## Run identity and scheduler requirements

All five jobs must consume the same completed teacher manifest and execute the
same formal training implementation/run head. An amendment metadata commit is
not permission to mix code revisions inside the five-seed experiment. Every
scheduler job must expose exactly one V100, run the foreground batch launcher,
perform the existing CUDA deterministic preflight, and write only its own seed
artifact paths.

After A1 activation, all five jobs may overlap. If 101 or 202 has already
started or completed under the base authorization, those runs remain the
registered seed runs and must not be repeated merely to make their start times
match the other three.

## Gate boundary

A1 cannot turn S05 into PASS. Full S05 still requires all five seed reports,
the frozen aggregate statistics, exact checkpoint reload, every existing Gate
condition and a later result audit. A failed or missing seed is retained and
causes the existing stop policy; it is never replaced by an extra run.
