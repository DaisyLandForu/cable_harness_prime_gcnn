# S06 Changelog

Status: pre-execution implementation complete; formal result not run.

## Added

- Frozen a 30-lineage validation-IID online instance manifest derived byte-for-byte
  from the S05 formal-v3 pre-model selection.
- Frozen the P1 online comparison protocol: 750 Gate-relevant tasks, five
  fullstrong diagnostics, seed-202 B0 checkpoint and instance-level paired
  statistics.
- Added a fail-closed online branchrule for B0-IL and deterministic random,
  native mostinf/relpscost/default/fullstrong execution, PDI/event metrics,
  solution validation and diagnostic trace replay.
- Added deterministic six-way lineage sharding for custom jobs. Every shard
  contains one graph per family and all 25 method/solver-seed tasks for each
  graph.
- Added separate main/trace barriers, shard manifests, resource identity
  matching, duplicate-shard locks and a single final aggregator.
- Added foreground custom-job, one-shard tmux and finalization launchers.
- Added unit, negative, aggregation and real frozen-SCIP integration tests.

## Protocol/API changes

- Formal execution is no longer a single-host six-worker run. It is exactly six
  independent jobs with six workers each, assigned by
  `instance_manifest_index modulo 6`.
- Each job requests 8 CPU cores, 96 GiB RAM and zero GPUs. Total maximum solve
  concurrency is 36; per-SCIP threads and every scientific parameter are
  unchanged.
- Main and trace phases are separate. Final aggregation requires six terminal
  manifests from each phase, including explicit empty trace manifests.

## Unchanged boundaries

- No S05 checkpoint, training data or normalization changed.
- No model retraining/reselection, B1 or RL implementation was performed.
- No test/final selector was read or solved.
- No formal S06 solve result exists at this content head.
- Unrelated aviation, build and local SCIP-prefix worktree changes were not
  staged or modified.

## Remaining

- external pre-execution audit and separate PASS activation;
- six main custom jobs, six conditional trace jobs and final aggregation;
- scientific Gate decision, result documents, result audit and audited tag.
