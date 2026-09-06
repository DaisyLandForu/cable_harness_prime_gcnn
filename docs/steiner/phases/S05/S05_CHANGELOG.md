# S05 Changelog — implementation scaffold

Status: pilot-v2 GPU attempt FAILED and retained; pilot-v3 pilot Gate PASS;
formal protocol v1 frozen for pre-audit; formal experiment NOT_RUN

## Formal protocol v1 preregistration

- Froze a non-executable formal configuration with 60 train, 15
  validation-select and 30 validation-gate base graphs. Three teacher seeds per
  graph produce 315 registered tasks with no replacement on failure.
- Froze 640 balanced train states, 160 lineage-disjoint checkpoint-selection
  states and 320 lineage-disjoint Gate states. Deterministic quota selection
  round-robins over base graph lineages.
- Froze fresh training seeds 101/202/303/404/505, one process per V100 and at
  most two concurrent jobs. Seed 202 is the a-priori representative handoff.
- Froze a 10,000-replicate paired bootstrap over 30 base-graph lineages,
  all-seed/all-family improvement requirements and seed-regret CV <= 0.15.
- Kept the 60% valid, 40% tie, 100% mapping and exact reload 0.0 Gates. Test,
  final, replacement instances, Gate relaxation and pseudocost label
  substitution remain prohibited.
- Added `S05_FORMAL_PROTOCOL_GPT_AUDIT_REQUEST.md`. No formal collection,
  training, tag or S06 work was performed.

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

- S04 re-audit passed and `steiner-s04-audited-v2` anchors the S04 phase head.
- No formal teacher shard or formal training was produced.
  `formal_gate_evaluated` remains false by construction; all completed GPU work
  is explicitly classified as pilot.
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

## Pilot-v1 capacity result and pilot-v2 preregistration

- The remediated pilot-v1 run completed 10/10 tasks in 627 seconds: 126 observed,
  117 valid, 0 all-tie, 3,086/3,086 mapped and zero split leakage.
- Its split counts were train 58 observed / 53 valid and validation 68 observed /
  64 valid. Therefore the registered 64-train-state curve could not run.
- Before any GPU/model run, the user approved a separate pilot-v2 with two
  S03-proven train tasks: sparse seed 100303 and grid seed 100315. It preserves
  all seeds, limits, curve sizes and Gates, and writes to independent v2 raw,
  report and checkpoint roots.

## Pilot-v2 GPU failure and pilot-v3 remediation

- Pilot-v2 teacher completed 12/12 with 85 valid train and 64 valid validation
  states. All three seeds completed every 40-epoch curve run, but exact reload
  parity failed on CUDA logit noise between `9.5367431640625e-07` and
  `1.52587890625e-05`.
- State dicts and normalization reloaded bit exactly; even repeated inference
  without reload reproduced the noise. V2 reports/checkpoints remain failed.
- Pilot-v3 freezes `CUBLAS_WORKSPACE_CONFIG=:4096:8` before process start and
  enables PyTorch deterministic algorithms. Preflight/report/checkpoint
  manifests record the controls and mismatches fail closed.
- The zero-error reload Gate, tasks, data, seeds, model, optimizer, curve,
  epochs and metrics were not changed.
- V3 recollected the exact registered 12-task teacher set at run head `7dd05e3`,
  then completed seeds 101/202/303 and all 9 curve runs with zero reload error.
- Strict aggregation passed. Mean regret improved 0.596116 -> 0.496956 ->
  0.447868 as states increased 16 -> 32 -> 64; random was 0.685977.
- Added `S05_PILOT_GATE_SUMMARY.json`. It explicitly classifies the result as a
  pilot PASS and leaves full S05 significance/formal Gate NOT_EVALUATED.
