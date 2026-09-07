# S05 Result Analysis — formal-v2 FAIL and formal-v3 preregistration

## Formal-v3 scope

GPT accepted the formal-v2 failure record and allowed a new protocol to be
preregistered, but did not turn S05 into PASS or authorize execution. V3 is a
minimal confirmatory-validation remediation: all five successfully trained v2
models are frozen and no model is retrained. The only expensive repeated work
is teacher labeling on completely new validation graphs.

To avoid repeating the 25/30 lineage failure after GPU work, v3 registers 80
fresh candidates (16 per family) and deterministically selects the first six
teacher-evaluable lineages per family. Eligibility requires at least 12 unique
valid states and cannot use model outputs or regret. Before loading a model, a
new hard barrier requires exactly 30 lineages, six per family and 320 states,
with every lineage represented and no prior/test/final overlap.

This improves execution reliability without weakening the scientific Gate.
The cost is a narrower conclusion: any successful result applies only to the
registered teacher-evaluable small/medium scale envelope, not to every scale.
Random-geometric small graphs remain in the new pool, so the adverse v2 signal
is tested rather than designed away. V3 remains NOT RUN pending independent
GPT pre-execution PASS.

## Formal-v2 outcome

All five registered seeds completed 40 epochs and reloaded exactly. Their
state-level validation-gate regrets were 0.583436, 0.678485, 0.673100,
0.651227 and 0.619285; the fixed random state-level regret was 0.747095.
This is encouraging model evidence, but it is not sufficient for the frozen
scientific Gate.

The 320 selected Gate states cover only 25 of 30 registered base-graph
lineages. The absent lineages are sparse seeds 201019/201020 and geometric
seeds 201024/201025/201026. Their 45 observed teacher states are all invalid
because at least one strong-branch child validity condition fails; some also
have a strong-call shortfall. Consequently no paired effect exists for those
five registered bootstrap units.

The aggregator correctly stopped rather than silently resampling 25 units.
For diagnosis only, the available-25 bootstrap CI is
`[0.056274, 0.191707]` and seed-regret CV is 0.068635, while the available
random-geometric family effect is negative at -0.041805. These are explicitly
non-Gate statistics because the preregistered 30-lineage matrix does not exist.

Formal-v1 remains FAIL for train bucket feasibility. Formal-v2 is also **FAIL**
for validation lineage completeness. Neither failure invalidates the retained
teacher or model artifacts, but neither may be rewritten as a PASS. A future
formal-v3 requires a new audited protocol; S06 and test/final remain prohibited.

## Formal-v1 teacher result

The first formal-v1 collection completed all 315 tasks but its teacher Gate is
**FAIL**. It passed task completion, 63.630952% valid-state yield, 0.436545%
all-tie rate, 100% mapping and zero role/split leakage. It failed only exact
bucket quota feasibility: train selected 527/640, while both validation roles
met 160/160 and 320/320.

The shortfalls are sparse-large 0/48, geometric-medium 3/64 and bridge-medium
60/64. This is not repaired by repeating identical deterministic tasks or by
lowering 640. Formal-v1 remains failed and no formal model seed ran. A separate
formal-v2 protocol proposes sealed-manifest reuse plus deterministic same-family
fallback, retains 128 states per family, and requires GPT PASS before use.

This checkpoint shows that the pilot model learns teacher-ranking signal; it
does not yet establish formal statistical significance or online SCIP benefit.

## What is established

- Pilot-v1 expands deterministically to ten tasks and the separately
  preregistered pilot-v2 to twelve, without touching test/final ranges.
- A real frozen SCIP branch state supplies candidate-aligned, finite strong
  scores plus explicit child validity/cost evidence.
- Feature/label shards survive checksum-verified reload and preserve complete
  probindex-to-edge identity.
- Train-only normalization, listwise optimization, ranking metrics, baseline
  diagnostics and checkpoint reload are executable on synthetic test fixtures.
- The S04 audited tag prerequisite is satisfied. Pilot-v2 data/training are
  retained as failed evidence; pilot-v3 provides the eligible pilot result.

## Pilot-v1 teacher result

The corrected pilot-v1 teacher collection completed all 10 tasks in 627 seconds.
It produced 117/160 valid states (73.125%), no all-tie valid state, exact
3,086/3,086 action mapping and zero split leakage. These teacher-quality checks
pass their registered thresholds.

Only 53 valid states belonged to train; validation had 64. Validation data cannot
be moved into train, and the 64-state curve cannot be lowered silently. Thus v1
is capacity-insufficient for the complete learning curve despite sound teacher
quality. No GPU/model run was attempted.

## What is not established

- Formal five-seed stability and preregistered statistical significance have
  not been evaluated.
- The implementation test's one real state is ABI/correctness evidence, not a
  dataset-quality estimate.
- Pseudocost is only an offline diagnostic. It is not the full relpscost solver
  behavior and must not be reported as such.
- Nothing here supports progression to S06 or a claim about online solve time,
  nodes, production relpscost, or final-test performance.

## Pilot-v2 GPU result

The v2 data-capacity amendment succeeded, but its GPU run is not an eligible
learned result. Seeds 101/202/303 each completed the 16/32/64-state, 40-epoch
matrix and wrote checkpoints. Every run then failed the unchanged exact reload
Gate due to non-deterministic CUDA logit differences. Checksums, state dicts and
normalization were intact, and repeated inference of the same in-memory model
showed the same noise. These metrics are diagnostic only and are not promoted
to completed results.

## Teacher attempt 1 (failed, retained)

The first CPU run at Git head `93984e5` completed 5/10 tasks. It retained 65/160
expected states, of which 56 were valid, with 0 all-tie valid states and exact
1,540/1,540 action mapping. Five tasks failed closed on an implementation check,
so the attempt is not eligible for teacher Gate evaluation.

The failure was not an identity or solver-resource failure. Ecole returned legal
LP candidates with `solution_frac=0.75`; the implementation incorrectly required
that feature to be at most 0.5. A remediated run must use one new Git fingerprint
and cannot combine the old attempt's successful shards with new-code shards.

## Pilot-v3 learning curve

Pilot-v3 completed the strict three-seed aggregate. Mean normalized SB regret
fell from 0.596116 at 16 states to 0.496956 at 32 and 0.447868 at 64, compared
with random 0.685977. This is a 13.10%, 27.55% and 34.71% mean relative
improvement. Across-seed sample standard deviation also fell from 0.072927 to
0.027604 and 0.022333.

All nine seed/curve runs beat random and reload exactly. Two of three seeds have
their best pilot regret at 64; seed 303 is marginally better at 32 (0.469350 vs
0.472148). Aggregate regret favors 64, but top-3 and rank correlation do not
improve monotonically from 32 to 64. The curve therefore supports expanding to
a preregistered formal collection; it does not demonstrate saturation or
justify selecting a final policy from this pilot alone.

Teacher-quality checks: **PASS**. Deterministic pilot Gate: **PASS**. Formal-v1
teacher quota Gate: **FAIL**. Formal-v2 complete S05 scientific Gate: **FAIL /
STOP before S06** because only 25/30 registered validation lineages contain a
valid paired effect. No final-test data was accessed.
