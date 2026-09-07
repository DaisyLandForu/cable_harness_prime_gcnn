# S05 Changelog — formal-v3 activated implementation

Status: formal-v1/v2 FAIL retained; formal-v3 pre-execution audit PASS and
implementation verified; teacher/Gate not yet run; S06 blocked

## Formal-v3 activation and executable pipeline

- Recorded the user-supplied GPT `PASS` in a separate activation record without
  changing the audited YAML.
- Added byte-exact activation/protocol/candidate loaders and exact 80-graph,
  240-task expansion.
- Added the six-worker collector with retained failures, envelope-derived
  mapping counts and teacher-only lineage eligibility.
- Added deterministic first-six-per-family selection, exact 30-lineage/320-
  state checks, prior-role isolation, checkpoint checksum preflight and a
  selection seal that must be committed before model access.
- Added five independent frozen-checkpoint evaluators. They do no training and
  require CUDA repeat/reload error 0.0 before producing a completed report.
- Added one-shot 30-lineage aggregation and CPU/GPU batch/tmux launchers.
- Added production permutation, shortage and barrier-order regression tests.
  No teacher task, checkpoint load, test/final access or S06 action occurred at
  this implementation checkpoint.

## Formal-v3 preregistration

- Recorded GPT's acceptance of the formal-v2 `FAIL / STOP before S06`; its PASS
  does not mean S05 passed and does not authorize v3 execution.
- Froze all five completed v2 checkpoint manifest and model hashes. V3 changes
  no model, training data, loss, hyperparameter, epoch, training seed or
  checkpoint-selection result and performs no retraining.
- Froze 80 fresh validation-IID candidate graph identities, 16 per family,
  with no seed or generated graph-hash collision against local prior S05
  evidence. The pool contains no test/final seed.
- Froze a teacher-validity-only rule: all 240 tasks must terminate without
  failure; a lineage needs at least 12 semantic-unique valid states; the first
  six eligible lineages per family are selected by precommitted rank/hash.
  Model outputs, regret and teacher score magnitude cannot influence selection.
- Froze a pre-model-access barrier requiring exactly 30 fresh lineages, six per
  family, at least one selected state per lineage, exactly 64 states per family
  and 320 total, complete checksums, and zero role/prior/test/final leakage.
- Fixed the v3 denominator unambiguously at 30 lineages everywhere; the frozen
  v1 prose typo saying 20 remains historical and is not edited or inherited.
- Limited claims to the registered teacher-evaluable scale envelope and kept
  random-geometric small graphs in scope despite their adverse v2 diagnostic.
- Added static hash, checkpoint, graph regeneration, split/freshness,
  deterministic selection and fail-closed barrier tests. No teacher, model,
  GPU, test/final or S06 action was run.

## Formal-v2 execution result

- Recorded the external re-audit `PASS / B1=CLOSED` in a separate activation
  record without changing the frozen v2 YAML.
- Verified all 3,431 sealed formal-v1 shard checksums and deterministically
  selected 640 train states, exactly 128 per family. The actual selection
  manifest SHA-256 is
  `35221abeeaae507623d0175d895b5ff807d0b7e5400fce494e615fbefe8cd10e`.
- Trained seeds 101/202/303/404/505 independently for 40 epochs on one V100
  each. All five completed and all checkpoint reload errors equal 0.0.
- Formal aggregation stopped before bootstrap because the frozen Gate matrix
  contains valid states from 25 rather than all 30 registered base graphs.
  Five registered validation graphs produced zero valid teacher state.
- Recorded S05 Gate **FAIL** without lowering the denominator, substituting
  graphs, imputing effects, rerunning teacher to success, or accessing
  test/final. S06 remains unauthorized.

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

## Formal implementation after protocol audit PASS

- Recorded the user-supplied GPT PASS in a machine-readable activation record
  bound to the immutable protocol content head and both file hashes.
- Added a byte-exact formal loader and exact 105-graph/315-task expansion.
- Added six-worker collection, terminal-status retention, teacher Gate
  aggregation and deterministic 640/160/320 role selection with lineage checks.
- Added fresh, one-V100-per-seed training with validation-select checkpointing,
  held-out validation-gate evaluation, exact full-Gate reload comparison and
  overwrite refusal.
- Added five-seed aggregation using 30 graph-lineage bootstrap units, the frozen
  all-seed/all-family/CV Gates and supplemental Wilcoxon/Holm diagnostics.
- Added batch and tmux launchers plus a scheduler job guide. No DDP or training
  semantic was introduced.

## Formal execution concurrency amendment A1

- Preserved the byte-exact audited v1 YAML and registered a separate,
  non-executable amendment after learning that the scheduler can allocate five
  independent V100 jobs rather than sharing one two-GPU host.
- The only permitted override is
  `training.max_concurrent_training_jobs: 2 -> 5`. Every seed, data item,
  hyperparameter, deterministic setting, validation role, statistic, Gate and
  failure rule remains unchanged.
- The formal teacher run is unaffected. The base limit of two remains binding
  until an external GPT PASS is recorded for A1; S05 full Gate and S06 remain
  NOT_EVALUATED/unauthorized.

## Formal-v1 teacher FAIL and v2 preregistration

- Retained the byte-exact failed v1 manifest: 315/315 tasks, 3,207/5,040 valid
  state slots and 88,549/88,549 mapped candidates passed quality checks, while
  exact train bucket quotas selected only 527/640 states.
- Recorded shortages of 48 sparse-large, 61 geometric-medium and four
  bridge-medium states. No task failed and no seed-101/202 training started.
- Preregistered v2 deterministic same-family fallback over the sealed manifest.
  Total train budget remains 640 and every family remains exactly 128; model,
  optimizer, validation, statistics, reload and quality Gates are unchanged.
- A1 is superseded before activation and its five-independent-job scheduling is
  incorporated into v2. Reselection and training remain blocked on v2 GPT PASS.

## Formal-v2 audit B1 remediation

- Recorded the first v2 verdict as CONDITIONAL PASS: methodology accepted, but
  canonical ordering named a field absent from the sealed manifest schema.
- Replaced only `canonical_instance_content_sha256` with literal
  `graph_sha256` in primary/fallback ordering and documented that aliases are
  forbidden.
- Added static schema, arbitrary-permutation, exact frozen bucket composition
  and missing-key fail-closed tests. No reselection, training or Gate change was
  performed.

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
