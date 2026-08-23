# Acyclic Cable Harness

SCIP-based MILP optimization for acyclic aircraft cable-harness routing. The current development line adds reproducible branching baselines before introducing RL branching.

**方法与实验评估底稿**（可直接交给外部模型评审）：[`docs/implementation_and_experiments.md`](docs/implementation_and_experiments.md)。

## Current Environment

- Conda environment: `rl4scip`
- SCIP: 8.0.4, SoPlex 6.0.4
- Compiler: GCC 11.4 with C++17
- Primary source: `code/scip_tree.cpp`
- Real instances: `code/data/edges-{1..9}.csv` and `code/data/pairs-{1..9}.csv`
- Synthetic instances: `code/data/synthesis/`

## Build

```bash
conda run -n rl4scip make
```

Override the local SCIP location when necessary:

```bash
conda run -n rl4scip make SCIP_ROOT=/path/to/scipoptsuite-8.0.4
```

## Run One Instance

```bash
build/scip_tree \
  --instance-id 9 \
  --branching relpscost \
  --seed 0 \
  --time-limit 60 \
  --node-limit -1 \
  --threads 1 \
  --output-json results/example.json
```

Available baseline methods are `default`, `relpscost`, `random`, `mostinf` (or `most-infeasible`), and `strong`. Phase 3 also provides candidate-safe `custom-random` and `custom-mostinf` rules. The legacy positional interface remains available as `build/scip_tree [copy_num] [div_part]`.

Export without solving:

```bash
build/scip_tree --instance-id 9 --export-milp data/instance_9.cip --build-only
```

## Reproduce Baselines

```bash
conda run -n rl4scip python scripts/run_baselines.py \
  --binary build/scip_tree \
  --instances 1,2,3,4,5,6,7,8,9 \
  --methods default,relpscost,random,mostinf \
  --strong-instances 9 \
  --seeds 0 \
  --time-limit 30 \
  --threads 1

conda run -n rl4scip python scripts/validate_baselines.py
```

See `docs/rl_branching_audit.md` and `docs/baseline_report.md` for assumptions, results, and limitations. RL training and inference commands will be added in their corresponding approved stages.

## Build The Phase-2 Dataset

The dataset keeps real scenarios grouped and uses deterministic synthetic instances to cover matched small, medium, and large scales across train, validation, and test.

```bash
conda run -n rl4scip python scripts/build_dataset.py \
  --config configs/dataset/phase2.json \
  --binary build/scip_tree \
  --instances-dir data/instances \
  --generated-dir data/generated \
  --results-dir results/dataset \
  --manifest data/instances/manifest.csv \
  --resume
```

Validate hashes, split isolation, deterministic regeneration, and SCIP readability:

```bash
conda run -n rl4scip python scripts/validate_dataset.py \
  --config configs/dataset/phase2.json \
  --manifest data/instances/manifest.csv \
  --scip-binary /home/duweiyue25/SCIP/scipoptsuite-8.0.4/build/bin/scip
```

The split policy and dataset limitations are documented in `docs/dataset_report.md`.

## Validate The Custom Branchrule

Run a real instance with a structured, optional branch-decision log:

```bash
build/scip_tree \
  --instance-id 9 \
  --branching custom-random \
  --seed 0 \
  --time-limit 60 \
  --threads 1 \
  --output-json results/custom_branching/raw/real_09_custom-random_seed0.json \
  --branch-log results/custom_branching/raw/real_09_custom-random_seed0_branches.csv
```

Build and run the strategy, fallback, reproducibility, and memory-accounting unit test:

```bash
conda run -n rl4scip make test-custom-branching
```

The implementation and real-instance acceptance evidence are documented in `docs/custom_branchrule_report.md`.

## Phase-4 BBMDP Environment

The Python environment uses PySCIPOpt 4.3.0 and a source build of Ecole 0.8.1, both linked to SCIP 8.0.4. Common dependencies are pinned in `requirements/phase4.txt`; SCIP-dependent build details are recorded in `docs/bbmdp_environment_report.md`.

Run the transition contract tests:

```bash
conda run -n rl4scip env PYTHONPATH=python \
  pytest -q tests/python/test_bbmdp_environment.py
```

Run a controlled DFS episode on a real exported instance:

```bash
conda run -n rl4scip env PYTHONPATH=python \
  python scripts/run_bbmdp_smoke.py \
  --config configs/rl/environment_smoke.yaml \
  --instance data/instances/train/real_06.cip \
  --policy random \
  --output results/phase4/transitions/real_06_random_seed0.json
```

## Phase-5 Candidate MLP-DQN

Phase 5 trains a shared Candidate MLP on the current fractional LP candidates. The
39-dimensional input contains 19 Ecole variable features, 14 deterministic global
tree features, and a 6-way aircraft-harness variable category. Wall-clock solving
time is deliberately excluded from the model state.

Install the pinned PyTorch runtime in the existing environment:

```bash
conda run -n rl4scip python -m pip install -r requirements/phase5.txt
```

Run the smoke or pilot configuration on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n rl4scip env PYTHONPATH=python \
  python scripts/train_candidate_mlp.py --config configs/rl/smoke.yaml

CUDA_VISIBLE_DEVICES=0 conda run -n rl4scip env PYTHONPATH=python \
  python scripts/train_candidate_mlp.py --config configs/rl/pilot.yaml
```

Validate checkpoint reload, finite losses, and eager/TorchScript Q-value parity:

```bash
conda run -n rl4scip env PYTHONPATH=python \
  python scripts/validate_mlp_artifacts.py \
  --artifact-dir artifacts/models/mlp \
  --instance data/instances/validation/real_08.cip \
  --seed 100
```

Re-run the fixed-seed random/untrained/RL comparison without retraining:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n rl4scip env PYTHONPATH=python \
  python scripts/reevaluate_candidate_mlp.py \
  --artifact-dir artifacts/models/mlp
```

Run the phase 4 and phase 5 Python tests together:

```bash
conda run -n rl4scip env PYTHONPATH=python \
  pytest -q tests/python/test_bbmdp_environment.py tests/python/test_candidate_mlp.py
```

The pilot configuration uses `real_06` for training and `real_08` for validation;
test and transfer instances remain untouched. Artifacts are under
`artifacts/models/mlp/`, and the results and limitations are documented in
`docs/training_report.md`.

## Phase-6 C++ RL-MLP Inference

The production solver loads the 39-feature Candidate MLP TorchScript model once and
scores only the current fractional LP candidates. CPU is the recommended device for
this small model; CUDA is supported but had substantially higher per-callback
overhead on the V100 pilot.

Build with Make in `rl4scip`:

```bash
conda run -n rl4scip make
conda run -n rl4scip make model-runner-parity
```

The equivalent CMake build links the packaged LibTorch runtime directly and does not
require `nvcc`:

```bash
conda run -n rl4scip cmake -S . -B build/cmake-release \
  -DSCIP_ROOT=/home/duweiyue25/SCIP/scipoptsuite-8.0.4 \
  -DTORCH_ROOT=/home/duweiyue25/conda/envs/rl4scip/lib/python3.11/site-packages/torch
conda run -n rl4scip cmake --build build/cmake-release -j2
```

Run the C++ RL rule on a real instance:

```bash
conda run -n rl4scip ./build/scip_tree \
  --instance-id 9 \
  --branching rl-mlp \
  --seed 0 \
  --time-limit 60 \
  --threads 1 \
  --rl-model artifacts/models/mlp/best_model_scripted.pt \
  --rl-device cpu \
  --rl-fallback relpscost \
  --rl-max-depth -1 \
  --rl-min-candidates 1 \
  --rl-log results/phase6/example_branches.csv \
  --output-json results/phase6/example.json
```

`--rl-max-depth` and `--rl-min-candidates` provide shallow/hybrid deployment gates.
Use `--rl-fallback default` to retain SCIP's normal plugin order after a fallback, or
`relpscost` to make reliability pseudocost the first built-in rule after RL. Missing
models, invalid outputs, extraction errors, or gate failures return control to SCIP
without starting Python or re-reading the model.

Validate complete Python/C++ Q vectors and the integration artifacts:

```bash
conda run -n rl4scip env PYTHONPATH=python \
  python scripts/validate_cpp_mlp_parity.py \
  --artifact-dir artifacts/models/mlp \
  --binary build/model_runner_parity \
  --output-dir results/phase6/parity \
  --device cpu

conda run -n rl4scip python scripts/validate_rl_mlp_integration.py \
  --default-json results/phase6/real09_default_final.json \
  --rl-json results/phase6/real09_rl_mlp_cpu_final.json \
  --rl-log results/phase6/real09_rl_mlp_cpu_final_branches.csv \
  --repeat-json results/phase6/real09_rl_mlp_cpu_repeat.json \
  --repeat-log results/phase6/real09_rl_mlp_cpu_repeat_branches.csv \
  --model-fallback-json results/phase6/real09_rl_invalid_model_fallback.json \
  --gate-fallback-json results/phase6/real09_rl_gate_fallback.json \
  --output results/phase6/integration_validation.json
```

See `docs/integration_report.md` for architecture, measured inference overhead,
fallback evidence, and the phase-6 acceptance decision.

## Phase-7 Bipartite GCNN-DQN

Phase 7 adds a full variable-constraint bipartite observation, two rounds of
`index_add_` message passing, prioritized replay, scalar Double DQN, and an optional
18-bin HL-Gauss head. It uses base PyTorch operations and does not require PyTorch
Geometric. Training uses one GPU; SCIP state generation is the dominant cost.

Train the scalar smoke/pilot or the optional HL-Gauss smoke:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n rl4scip env \
  PYTHONPATH=python CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python scripts/train_gcnn.py --config configs/rl/gcnn_smoke.yaml

CUDA_VISIBLE_DEVICES=0 conda run -n rl4scip env \
  PYTHONPATH=python CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python scripts/train_gcnn.py --config configs/rl/gcnn_pilot.yaml

CUDA_VISIBLE_DEVICES=0 conda run -n rl4scip env \
  PYTHONPATH=python CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python scripts/train_gcnn.py --config configs/rl/gcnn_hlgauss_smoke.yaml
```

Build the C++ graph runner and validate complete Python/C++ Q vectors:

```bash
conda run -n rl4scip make -j2
conda run -n rl4scip make gcnn-model-runner-parity

conda run -n rl4scip env PYTHONPATH=python \
  python scripts/validate_cpp_gcnn_parity.py \
  --artifact-dir artifacts/models/gcnn \
  --binary build/gcnn_model_runner_parity \
  --output-dir results/phase7/parity_cpp \
  --device cpu --tolerance 1e-5
```

Run the native C++ GCNN branching rule on a real held-out instance:

```bash
conda run -n rl4scip ./build/scip_tree \
  --instance-id 9 \
  --branching rl-gcnn \
  --seed 0 \
  --time-limit 60 \
  --threads 1 \
  --rl-model artifacts/models/gcnn/best_model_scripted.pt \
  --rl-device cuda \
  --rl-fallback relpscost \
  --rl-max-depth -1 \
  --rl-min-candidates 1 \
  --rl-log results/phase7/e2e/real09_rl_gcnn_cuda_branches.csv \
  --output-json results/phase7/e2e/real09_rl_gcnn_cuda.json
```

Use `--rl-max-depth 5`, `10`, `20`, or `50` for shallow GCNN followed by the
configured SCIP fallback. The phase-7 architecture, pilot evidence, parity results,
and current inference-overhead limitation are documented in
`docs/gcnn_report.md`.

## Phase-8 Controlled and Production Experiments

Phase 8 evaluates the same branching methods under two protocols. The
`controlled-bbmdp` protocol uses DFS, disables restarts, and limits separation to
the root. The `production-scip` protocol preserves the project's original SCIP
settings and changes only branching-variable selection. Both protocols use one
SCIP thread and fixed seeds.

Install the plotting dependency and rebuild the solver:

```bash
conda run -n rl4scip python -m pip install -r requirements/phase8.txt
conda run -n rl4scip make -j2
```

Run or resume the 430-run real-instance experiment. The runner writes each run
immediately, so the same command safely resumes after interruption:

```bash
conda run -n rl4scip python scripts/run_final_experiments.py \
  --config configs/experiments/phase8_full.json --resume --workers 2
```

Validate objectives, protocol settings, return codes, and every logged RL action,
then create aggregate tables and figures:

```bash
conda run -n rl4scip python scripts/validate_final_results.py \
  --input results/final/raw_results.csv --expected-runs 430 \
  --output results/final/validation.json

conda run -n rl4scip python scripts/analyze_final_results.py \
  --input results/final/raw_results.csv --output-dir results/final \
  --bootstrap-samples 2000
```

Train the four equal-budget GCNN ablation checkpoints:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n rl4scip env PYTHONPATH=python \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 python scripts/train_gcnn.py \
  --config configs/rl/ablation_nstep1.yaml
CUDA_VISIBLE_DEVICES=0 conda run -n rl4scip env PYTHONPATH=python \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 python scripts/train_gcnn.py \
  --config configs/rl/ablation_hlgauss.yaml
CUDA_VISIBLE_DEVICES=0 conda run -n rl4scip env PYTHONPATH=python \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 python scripts/train_gcnn.py \
  --config configs/rl/ablation_no_categories.yaml
CUDA_VISIBLE_DEVICES=0 conda run -n rl4scip env PYTHONPATH=python \
  CUBLAS_WORKSPACE_CONFIG=:4096:8 python scripts/train_gcnn.py \
  --config configs/rl/ablation_no_global.yaml
```

Model and shallow-hybrid evaluations use the same resumable runner:

```bash
conda run -n rl4scip python scripts/run_final_experiments.py \
  --config configs/experiments/phase8_model_ablations.json --resume --workers 1

conda run -n rl4scip python scripts/run_final_experiments.py \
  --config configs/experiments/phase8_depth_ablations.json --resume --workers 1

conda run -n rl4scip python scripts/analyze_phase8_ablations.py \
  --main results/final/raw_results.csv \
  --models results/final/ablations/models/raw_results.csv \
  --depth results/final/ablations/depth/raw_results.csv \
  --output-dir results/final
```

Raw per-run JSON, SCIP logs, and branch-decision CSV files are retained below
`results/final/raw/`. Aggregate results and all reproducible plots are written to
`results/final/`. The conclusions and deployment decision are documented in
`docs/FINAL_RL_BRANCHING_REPORT.md`.
