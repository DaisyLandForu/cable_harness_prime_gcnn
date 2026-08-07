# Phase 5 Candidate MLP-DQN Training Report

## 1. Stage conclusion

Phase 5 completed the Candidate MLP-DQN minimum learning loop on real aircraft
cable-harness instances. The implementation includes masked candidate scoring,
Double DQN, a target network, 3-step returns, replay, epsilon-greedy exploration,
gradient clipping, validation checkpoints, TorchScript export, and structured
training/evaluation artifacts.

The corrected pilot has a measurable learning signal: RL solved all three validation
seeds, while random solved two. With the 200-node limit counted as 200 nodes, mean
nodes fell from 130.3 for random to 109.3 for RL, a 16.1% reduction. Mean SCIP solving
time fell only 1.7%, from 30.33 to 29.83 seconds. One seed regressed from 100 to 146
nodes, so the result is promising but not yet uniformly robust and is not evidence of
improvement over SCIP-default.

## 2. Files added or changed

| Path | Purpose |
|---|---|
| `python/rl_branching/candidate_features.py` | Candidate extraction, aviation category one-hot, running normalization |
| `python/rl_branching/candidate_model.py` | Shared Candidate MLP and TorchScript export |
| `python/rl_branching/dqn.py` | Double DQN learner, target updates, clipping, deterministic tie-breaking |
| `python/rl_branching/replay.py` | Replay buffer and 3-step transition accumulator |
| `python/rl_branching/training_config.py` | Typed YAML configuration and validation |
| `python/rl_branching/trainer.py` | Training, validation, checkpointing, evaluation, summary reconstruction |
| `python/rl_branching/observation.py` | Deterministic BBMDP global state features |
| `scripts/train_candidate_mlp.py` | Training entry point |
| `scripts/validate_mlp_artifacts.py` | Eager/TorchScript full-Q parity and finite-loss checks |
| `scripts/reevaluate_candidate_mlp.py` | Evaluation-only entry point with independent policy seeds |
| `scripts/evaluate_candidate_policy.py` | Isolated policy and action-trace reproducibility probe |
| `configs/rl/smoke.yaml` | 500-step closure test |
| `configs/rl/pilot.yaml` | 5,000-step real-instance pilot |
| `configs/rl/full_mlp.yaml` | 20,000-step future full configuration; not run in this stage |
| `tests/python/test_candidate_mlp.py` | Feature, mask, n-step, DQN, tie-break, export, and config tests |
| `requirements/phase5.txt` | PyTorch 2.5.1 CUDA 12.1 pin |

No phase-5 change was made to the MILP variables, constraints, objective, or
`code/scip_tree.cpp`. C++ model integration belongs to phase 6.

## 3. Data isolation

| Role | Real instance | Use |
|---|---|---|
| train | `data/instances/train/real_06.cip` | Environment interaction and replay |
| validation | `data/instances/validation/real_08.cip` | Checkpoint selection and pilot comparison |
| test | untouched | Reserved for later held-out experiments |
| transfer | untouched | Reserved for later transfer experiments |

States from an instance are never randomly split between train and validation. The
current pilot intentionally uses only real instances from the reorganized
`code/data` source; synthetic instances are not used for the reported result.

## 4. State, action, and model

Each legal action is one of SCIP's current fractional LP branching candidates. The
network applies the same MLP weights to every candidate and returns one scalar Q
value per candidate. Selection is an argmax restricted to that action set.

The final schema has 39 inputs per candidate:

- 19 Ecole variable features, including objective, type, bounds, reduced cost, LP
  solution, fractionality, age, incumbent, and basis indicators;
- 14 deterministic global tree features, including depth, node counts, open/leaves,
  LP iterations, bounds, gap, and incumbent count;
- 6 aviation variable categories: `m`, `z`, `y`, `absf`, `f`, and `other`.

`solving_time` was initially included as a fifteenth global feature. Independent
processes then produced identical candidate sets and LP features but different model
actions because root presolve wall time varied with host load. It was removed and the
model was retrained from scratch. The schema was bumped to version 2, and wall-clock
features are explicitly listed as excluded in `feature_schema.json`.

The final MLP is `39 -> 128 -> 128 -> 1`, with 21,761 trainable parameters. It is too
small for useful multi-GPU data parallelism; SCIP transition generation is the
dominant cost.

## 5. Learning configuration

The pilot uses Double DQN, a hard-updated target network every 250 gradient steps,
3-step returns, gamma 1, Smooth L1 loss, Adam at `3e-4`, batch size 32, replay capacity
10,000, eight updates per environment step, and gradient clipping at 10. Epsilon
decays from 1.0 to 0.05 over 4,000 gradient steps.

SCIP uses DFS, one thread, restarts disabled, cuts beyond the root disabled, a 60
second limit, and a 200-node limit. Truncated episodes do not bootstrap. All solver,
NumPy, Python, and PyTorch random sources receive explicit seeds.

## 6. Commands executed

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n rl4scip env PYTHONPATH=python \
  python scripts/train_candidate_mlp.py --config configs/rl/smoke.yaml

CUDA_VISIBLE_DEVICES=0 conda run -n rl4scip env PYTHONPATH=python \
  python scripts/train_candidate_mlp.py --config configs/rl/pilot.yaml

conda run -n rl4scip env PYTHONPATH=python \
  python scripts/validate_mlp_artifacts.py \
  --artifact-dir artifacts/models/mlp \
  --instance data/instances/validation/real_08.cip --seed 100

CUDA_VISIBLE_DEVICES=0 conda run -n rl4scip env PYTHONPATH=python \
  python scripts/reevaluate_candidate_mlp.py --artifact-dir artifacts/models/mlp
```

## 7. Smoke result

The corrected smoke completed 500 gradient steps and two episodes. Its RL policy
solved validation seed 100 in 46 nodes; random and untrained both reached the 50-node
limit. This is only a closure test. The exported and eager models agreed exactly on
all 151 candidate Q values (`max_abs_error = 0`).

## 8. Pilot training result

The corrected pilot completed 5,000 gradient steps, 20 episodes, and built a replay
buffer of 735 transitions. The estimated training duration through final validation
was 1,278 seconds. Peak process RSS was 1,543.7 MiB and peak allocated GPU memory was
64.42 MiB on a Tesla V100-SXM2-32GB.

All 100 logged update losses were finite, from 0.04598 to 12.12238. Validation was
always optimal and evolved as follows:

| Gradient step | Validation nodes |
|---:|---:|
| 1,192 | 78 |
| 2,160 | 85 |
| 3,056 | 68 |
| 4,128 | 63 |
| 5,000 | 66 |

The 4,128-step checkpoint is `best_model.pt`; the 5,000-step model is retained as
`last_model.pt`. The non-monotonic curve indicates mild overfitting after the best
checkpoint, so checkpoint selection is necessary.

## 9. Pilot comparison

All methods use the same SCIP seed set `{100, 101, 102}`. Each method also receives an
independent deterministic policy seed, preventing one policy's episode length from
changing another policy's RNG stream.

| Seed | Random status/nodes | Untrained status/nodes | RL status/nodes |
|---:|---|---|---|
| 100 | optimal / 91 | optimal / 107 | optimal / 63 |
| 101 | optimal / 100 | optimal / 104 | optimal / 146 |
| 102 | nodelimit / 200 | nodelimit / 200 | optimal / 119 |

| Method | Solved | Mean nodes | Mean solving time |
|---|---:|---:|---:|
| random | 2/3 | 130.33 | 30.33 s |
| untrained | 2/3 | 137.00 | 33.90 s |
| RL best | 3/3 | 109.33 | 29.83 s |

Relative to random, RL reduces the capped arithmetic mean node count by 16.1% and
mean solving time by 1.7%. It also solves the seed that random leaves at the node
limit. Seed 101 is a clear counterexample, with RL using 46% more nodes than random.
Therefore aggregate validation is better than random, but the direction is not
consistent for every seed and the evidence remains a small pilot.

## 10. Correctness and reproducibility

- Eager PyTorch and TorchScript produced all 151 Q values with maximum absolute error
  zero and selected the same candidate.
- A 20-decision isolated episode was repeated in separate processes with identical
  candidate/global hashes, actions, node IDs, Q values, and action-trace SHA-256
  `11b3ec0325739d95e8ad327e9e01854c2b089933df47638ab6ef71bef9e7541a`.
- Every environment action is validated against the current immutable action set.
- NaN/Inf model outputs and losses are rejected.
- Stable Q ties are resolved by transformed variable name and action index.
- Timeout and node-limit episodes have explicit truncation semantics and zero
  terminal bootstrap under this configuration.

## 11. Artifacts

The corrected pilot artifacts are in `artifacts/models/mlp/`:

- `best_model.pt`, `last_model.pt`;
- `best_model_scripted.pt`, `last_model_scripted.pt`;
- `config.yaml`, `feature_schema.json`, `normalization.json`;
- `training_history.csv`, `evaluation.csv`, `summary.json`;
- `parity_observation.npz`, `parity.json`.

The pre-fix 40-dimensional model is preserved only for diagnosis under
`artifacts/models/mlp/pre_repro_fix/` and must not be used for integration.

## 12. Acceptance and next-stage gate

| Criterion | Result |
|---|---|
| Smoke loop and checkpoint | Pass |
| Double DQN, target network, 3-step return, replay, action mask | Pass |
| No NaN/Inf | Pass |
| Save/reload and eager/export parity | Pass |
| Fixed observation/action trajectory reproducible | Pass after removing wall-clock input |
| Validation better than random in aggregate | Pass: solved rate 3/3 vs 2/3; capped mean nodes -16.1% |
| Every seed improves over random | Not pass: seed 101 regresses |
| Evidence of improvement over SCIP-default | Not evaluated in phase 5 |

The minimum MLP learning loop is complete and suitable for phase-6 integration work,
provided phase 6 is treated as correctness and inference-overhead validation rather
than a deployment claim. Full GCNN work should remain gated on broader multi-instance,
multi-seed evidence. The most valuable immediate improvement is parallel CPU actor
collection across more real training scenarios, not multi-GPU training of this small
MLP.
