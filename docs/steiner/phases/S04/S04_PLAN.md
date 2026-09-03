# S04 Plan — Standard MILP Bipartite B0

Status: local Gate PASS; combined GPT audit pending by explicit user waiver

## Frozen identity

- branch: `research/steiner-migration`
- base SHA: `931c7ae05c299c54bbdf59ecd458b64c7ca42282`
- prior S03 content/phase: `495d699cceefd243d4ab4c510be051f9df94833a` /
  `bb6079b7844dcc42fed4976c812795c842d6411b`
- solver stack: `scip804-ecole081-pyscipopt430`
- data boundary: synthetic train only; no validation or sealed final-test access
- deterministic model seed: 404; solver seed: 0
- frozen config canonical SHA-256:
  `056ce49bce41c731138a83b3befbc97e006585311bcf8f5298532c2d86f830dc`

The initial worktree contains pre-existing aviation scripts, build binaries,
environment artifacts, audit binaries, figures, and unrelated experiment files.
They are outside S04 and must remain unstaged and unmodified by this stage.

## Goal

Build the aviation-independent B0 contract: an immutable 19/5/1 MILP bipartite
state, legal fractional binary `stp_x_*` actions with explicit edge mapping, the
exact one-round candidate closure, and a small Gasse-style bipartite GCNN.

The final B0 uses sum aggregation, matching the standard Gasse scatter-add
baseline; an earlier local mean-aggregation prototype was discarded before the
frozen snapshot and is not a Gate result.

## Non-goals

- no strong-branch teacher collection or imitation learning (S05)
- no normalization learned from validation/test data
- no B1/global/aviation/Prim/DSU features
- no original-graph GNN, RL, checkpoint, TorchScript, or C++ deployment
- no GPU requirement and no changes to the legacy aviation stack

## Planned implementation

- `solver/bipartite_observation.py`: versioned schema, finite immutable state,
  Ecole observation copying, legal-action filtering and transformed-name mapping
- `solver/graph_state.py`: candidate exact closure with explicit local/global maps
- `models/milp_gcnn.py`: one variable→constraint→variable sum-aggregation B0
- `configs/steiner/models/b0_milp_gcnn_v1.yml`: strict architecture/snapshot Gate
- `scripts/steiner/run_s04_b0_snapshot.py`: deterministic CPU snapshot and timing
- `tests/steiner/test_s04_bipartite.py`: unit/property/integration/Gate coverage
- `docs/steiner/phases/S04/**` and `docs/steiner/STATUS.md`: evidence and handoff
- `RESEARCH_CONTRACT.md` and `S03_S04_RESOURCE_PREFLIGHT.md`: correct stale
  inventory/resource tense against already committed S03 evidence; no Gate,
  split, seed, metric or scientific rule changes

## Verification matrix

1. schema widths are exactly variable 19, constraint 5, edge 1;
2. arrays are finite, copied and immutable; NaN/Inf and malformed shapes fail;
3. empty legal action sets return an explicit empty state, never a fake action;
4. only current fractional binary `stp_x_*` candidates survive filtering;
5. original/transformed names and parallel edges map one-to-one to edge IDs;
6. exact closure retains candidates, their incident constraints, all variables on
   those constraints, and stable global/local index maps;
7. full/closure candidate logits agree within `1e-5` with identical argmax;
8. a real SCIP 8.0.4 MCF branch state satisfies the same mapping/schema checks;
9. fixed model/state hashes, parameter count, CPU inference p50/p95 are recorded;
10. the complete Steiner regression suite remains green apart from documented
    pre-existing skips.

## Gate and stop conditions

S04 PASS requires all eligible deterministic snapshots to have finite features
and logits, 100% legal-action-to-edge mapping, full/closure maximum absolute
logit error at most `1e-5`, and 100% argmax agreement. Parameter count and CPU
inference timing must be present. Any mapping ambiguity, non-finite value,
integration failure, or parity failure stops in S04; the Gate is not weakened.

## External effects

Small committed JSON summaries/snapshots are allowed. Build products, raw solver
states and logs remain ignored. If and only if the local Gate passes, S04 commits
may be fast-forward pushed to `research/steiner-migration`. No PR, merge, rebase,
amend, force-push, tag push, S05 work, or learned final-test run is authorized.

After local S04 completion, S00--S04 are submitted together through
`S04_AUDIT_PACKET.md`. The waiver permits S04 implementation only; S05 remains
blocked until the combined GPT audit passes.
