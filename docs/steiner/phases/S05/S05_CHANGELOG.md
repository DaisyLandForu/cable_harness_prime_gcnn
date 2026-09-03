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
