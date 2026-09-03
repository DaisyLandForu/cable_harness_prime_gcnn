# S05 Changelog — implementation scaffold

Status: source complete for CPU verification; formal experiment NOT_RUN

## Prepared components

- Added a strict pilot config covering five train and five validation-IID
  medium-mid synthetic instances. Pilot teacher seed is 1001, pilot training
  seeds are 101/202/303, and the nested learning curve is 16/32/64 states.
- Added a frozen SCIP 8.0.4 C bridge around `SCIPgetVarStrongbranchFrac()`.
  Every legal candidate records down/up bound validity, infeasibility, LP error,
  score, probindex, node/depth, elapsed time, LP iterations and SB calls.
- Candidate labels are aligned to Ecole actions only by a complete probindex
  bijection. The real integration test also matches custom scores against Ecole
  `StrongBranchingScores` to `1e-12`.
- Pseudocosts are extracted before teacher calls, preventing strong-branch
  updates from leaking into the offline cheap-score baseline.
- Added immutable teacher samples and atomic compressed NPZ shards with both
  file and semantic checksums. Resume requires matching task, config, Git commit
  and every referenced shard checksum.
- Added train-only normalization, listwise cross-entropy, top-1/top-3,
  tie-aware Spearman correlation and normalized SB regret. Offline diagnostics
  include random, most-infeasible and honest pseudocost (not mislabeled as full
  SCIP relpscost).
- Added multi-seed CUDA training, nested learning curves, failure/skip retention,
  checksum-verified checkpoint/normalization manifests and exact reload parity.
- Added separate CPU-teacher and CUDA-training tmux launchers. Raw shards, logs,
  reports and checkpoints are ignored by Git.
- Added a non-training CUDA allocation/preflight script for the replacement GPU
  host; it records visible devices, memory, CUDA/PyTorch versions and a finite
  tensor smoke checksum under the ignored raw artifact root.
- Added foreground batch launchers for the platform's non-interactive jobs.
  Pilot training can now select disjoint registered seeds, writes one report per
  seed shard, and uses disjoint checkpoint directories.
- Added a strict pilot report aggregator. It refuses missing/duplicate or
  unregistered seeds, incomplete 3-by-3 run matrices, failed runs, and mismatched
  Git/config/teacher/audited-tag identities. Formal seeds remain blocked until
  the pilot learning curve fixes the formal data budget.

## Deliberate blockers and non-goals

- S04 re-audit subsequently passed and `steiner-s04-audited-v2` now anchors the
  S04 phase head. Collection remains NOT_RUN pending the scheduled job/resource
  preflight.
- No formal teacher shard, learned update, GPU call, checkpoint or validation
  result was produced. `formal_gate_evaluated` remains false by construction.
- No final-test selector/data was read, and no S03 task, 19/5/1 schema, Gate,
  aviation source or legacy failure was changed.
- Exact online relpscost comparison remains S06. S05 stores pre-teacher SCIP
  pseudocost scores only and names them accordingly.

## Teacher attempt 1 remediation

- The first authorized CPU pilot at Git head `93984e5` retained 65 observed
  states but ended failed: five tasks completed and five reported
  `SCIP action ... is not fractional`.
- Diagnosis showed exact action/teacher probindex agreement (33/33 on the
  reproduced state) and a legal Ecole `solution_frac` range of 0.25--0.75.
  `solution_frac` is the fractional part in `(0, 1)`, while the validator had
  incorrectly treated it as nearest-integer distance capped at 0.5.
- The validator now accepts the documented fractional-part interval and still
  rejects integer endpoints. Mapping, schema, thresholds, seeds and teacher
  scores were not changed. Attempt 1 remains archived as failed evidence and
  is not mixed with the remediated run.
