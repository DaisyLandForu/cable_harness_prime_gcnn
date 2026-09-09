# S06 Changelog

Status: external pre-execution audit PASS; separate committed activation is the
remaining prerequisite for formal execution; formal result not run.

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
- Added fail-closed exception evidence: stable failure classes, retained PAR-2
  penalty and explicitly unavailable PDI.
- Added registered-resource and runtime validation before tasks, with the exact
  runtime identity persisted in every task and shard manifest.
- Added an aggregate-level lock and immutable-output refusal for the formal
  summary and run manifest.
- Recorded the external CONDITIONAL PASS without overwriting its four findings.
- Required the activation file itself to be committed byte-exact at `HEAD`;
  uncommitted activation edits are rejected before execution.
- Moved aggregate OS-lock ownership into the Python process that writes formal
  evidence, eliminating the spoofable shell environment-variable guard.
- Recorded the first focused re-audit: B1/B4 closed, B2/B3 remained open at
  `a8fbc0986068171344387ceb68e1ee5ab5bbefbc`.
- Recorded the second focused re-audit PASS at the fixed B2/B3 remediation head
  `a29ef9eeb818c1694f78d2de1b29436f1861f8c3`; all B1--B4 are closed.

## CONDITIONAL PASS remediation

- B1: failed policy/runtime tasks now remain in all applicable PAR-2 pairs with
  `par2_seconds=1200`; unavailable PDI is not invented and therefore prevents a
  Gate PASS.
- B2: six jobs must each satisfy at least 8 effective CPUs, at least 96 GiB,
  zero visible GPUs, the frozen SCIP/environment identity and the same
  activation-bound runtime fingerprint and executable content.
- B3: only the locked finalizer may aggregate, and existing aggregate outputs
  are immutable.
- B4: this updated audit packet is part of the new fixed remediation object.

## Focused B2/B3 closure

- B2: `load_s06_activation()` now reads the local bytes, requires the resolved
  path to remain inside the repository, obtains `HEAD:<relative path>` through
  Git and requires byte equality before parsing/accepting the record.
- B3: `_aggregate()` itself opens `aggregate.lock` and obtains
  `LOCK_EX|LOCK_NB`; the lock remains held across all validation and both
  atomic writes. A forged environment variable has no effect.

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

- separate committed PASS activation;
- six main custom jobs, six conditional trace jobs and final aggregation;
- scientific Gate decision, result documents, result audit and audited tag.
