# S03 Plan — Branchability and MCF Resource Gate

Status: executed; local Gate PASS; GPT audit NOT_RUN
Stage branch: `research/steiner-migration`
Stage base SHA: `91c30a48e6a06019d16d8b7529fe2d35bfa708fa`
Preregistration date: 2026-09-03 UTC

## Objective

Measure whether the S02 rooted-undirected MCF formulation produces enough legal
Steiner edge-variable branching decisions for learning-to-branch, whether small
strong-branching samples contain a useful ranking signal, and whether the MCF
model is safe at the planned six-worker rollout concurrency.

S03 does not train a model, tune against final-test data, implement S04/S05, or
repair the unrelated legacy aviation regressions.

## Frozen inputs

- Only synthetic `train` seeds from `100000..199999` may be used.
- The exact instance matrix, baseline list, limits, strong-branch sampling rule,
  and tolerances live in
  `configs/steiner/experiments/s03_branchability_pilot_v1.yml`. Its canonical
  SHA-256 must be frozen before starting the formal pilot.
- Required graph families are sparse/configuration, geometric, grid/holes,
  community/block, and bridge/bottleneck/multi-corridor.
- Solver stack is SCIP 8.0.4 + PySCIPOpt 4.3.0 + Ecole 0.8.1, entered only via
  `scripts/steiner/run_with_scip804.sh`.
- Solver protocol is P1: 600 s, 200,000 nodes, 8,192 MB, one SCIP thread,
  presolve rounds 0, separation rounds/root rounds 0, heuristics off, restarts 0,
  estimate node selector, solver seed 0.
- Baselines are SCIP default branching, `relpscost`, and `mostinf`. The
  `relpscost` run is the primary branchability inventory; all profiles contribute
  to mapping and resource checks.

## Measurements

For every formal task retain success, timeout, memory limit, root solved, no
legal edge action, solver error, and skipped/resumed outcomes. Record model
variables/constraints, continuous flow variables, build time, root LP bound and
gap where available, fractional edge count, observed legal edge decisions,
nodes, LP iterations, solve time, peak process RSS, and action-mapping counts.

A high-priority observer branch rule records LP branching candidates but returns
`DIDNOTRUN`, so the selected SCIP baseline remains responsible for each action.
Legal actions are original binary `stp_x_*` variables with a complete canonical
edge mapping. Branch states are de-duplicated by SCIP node number.

Strong branching is sampled only on the preregistered `relpscost` tasks and only
for the first preregistered number of eligible states. A state is valid when it
has at least two legal candidates, every sampled candidate returns finite valid
up/down information without an LP error, and at least two finite candidate
scores exist. An all-tie state has score range at most the frozen absolute plus
relative tolerance. No failed state or candidate may be discarded.

## Execution and restart policy

Each task runs in a fresh process and writes one atomic JSON shard below the
ignored `results/steiner/raw/s03/` tree. A shard is reused only when its task and
configuration fingerprints match. Concurrency ramps through one, three, and six
workers before the formal matrix. Raw solver logs and per-state observations are
not committed; the small aggregate Gate summary is committed.

The tmux launcher is only an operational wrapper around the same immutable
configuration. Interrupted runs resume from valid shards.

## Frozen Gate

S03 is PASS only when all conditions hold:

1. At least 60% of primary-profile instances have at least five legal decisions.
2. Among nontrivial primary-profile instances, median legal decisions is at
   least 10.
3. At least 60% of sampled strong-branch states are valid.
4. No more than 40% of valid strong-branch states are all-tie.
5. Action mapping is exactly 100% over every observed legal candidate.
6. Fresh-worker peak RSS p95 is at most 8,192 MB.

The MCF-to-SCF trigger is also reported when any of these holds: more than
1,000,000 continuous flow variables, build-time p95 above 60 s, single-worker
RSS p95 above 8,192 MB, or projected six-worker RSS (`6 * p95`) above 49,152 MB.
A trigger is not hidden by deleting cases; it becomes an explicit S03 risk and
blocks declaring the current MCF range ready for S04.

If any Gate item fails, work stops in S03. Any revised range requires a new,
versioned preregistration and must preserve the failed results.

## Required outputs

- `S03_CHANGELOG.md`
- `S03_TEST_REPORT.md`
- `S03_RESULT_ANALYSIS.md`
- `S03_AUDIT_PACKET.md`
- `S03_COMMANDS.md`
- machine-readable aggregate Gate summary

Only after a local PASS may S03 changes be committed and pushed to the single
long-lived branch `research/steiner-migration`. No merge, rebase, amend, or
force-push is permitted.

## Execution outcome

The formal config was frozen before execution at SHA-256
`cab4d8d96b02f427b8fedba6698cb9ee68b7e8a27059f8eaf319a0ede96ac1f1`.
All 90 formal tasks and 10 ramp tasks produced valid shards. Every frozen Gate
check passed and no MCF-to-SCF resource trigger fired. Exact values are in
`S03_GATE_SUMMARY.json`; interpretation and limitations are in
`S03_RESULT_ANALYSIS.md`. S04 was not started.
