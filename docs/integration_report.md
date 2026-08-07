# Phase 6 Candidate MLP C++ Integration Report

## 1. Stage conclusion

Phase 6 is complete. The phase-5 Candidate MLP is integrated into the existing SCIP
8.0.4 solve path as a native C++ branching rule using TorchScript and LibTorch. The
MILP variables, constraints, objective, feasibility checks, and solve sequence were
not changed.

On real instance `real_09`, C++ RL-MLP selected 12 legal candidates, solved the model
to the same objective as SCIP-default, and produced an identical decision sequence
in a separate repeat run. Model-load and candidate-gate failures both fell back to
SCIP and solved normally. Python/C++ complete Q vectors agree exactly on CPU and
CUDA for the fixed 151-candidate observation.

This is an integration and correctness result, not evidence that RL beats SCIP. On
the tested instance, RL used 13 nodes versus 9 for default. Broader performance
claims remain reserved for the phase-8 multi-instance, multi-seed experiment.

## 2. Architecture

The deployment path is:

1. `scip_tree` creates SCIP and registers all default plugins.
2. For `--branching rl-mlp`, it registers `RlMlpBranchrule` at priority 1,000,000.
3. `ModelRunner` loads the TorchScript module once during rule construction, selects
   CPU or CUDA, switches to eval/inference mode, and performs warm-up forwards.
4. Each LP callback obtains SCIP's current priority fractional LP candidates.
5. C++ extracts the same 19 variable, 14 global-tree, and 6 aviation-category
   features used by the phase-5 schema.
6. LibTorch returns one finite scalar Q value per candidate; stable argmax is applied
   only inside that candidate vector.
7. The selected variable is checked against the immutable callback candidate set and
   branched with `SCIPbranchVarVal`.
8. Any extraction, model, dimension, NaN/Inf, legality, depth, or candidate-count
   failure returns `SCIP_DIDNOTRUN`, allowing a built-in SCIP rule to continue.

No Python process is started by the solver and no disk access occurs inside the
branch callback. `--rl-fallback relpscost` explicitly places reliability pseudocost
immediately below RL; `--rl-fallback default` leaves SCIP's built-in ordering intact.

## 3. Files added or changed

| Path | Purpose |
|---|---|
| `src/rl/scip_feature_extractor.hpp/.cpp` | Candidate-safe 39-feature extraction and variable categories |
| `src/rl/model_runner.hpp/.cpp` | One-time TorchScript loading, warm-up, CPU/CUDA inference, finite checks |
| `src/rl/rl_mlp_branchrule.hpp/.cpp` | SCIP ObjBranchrule, action masking, fallback, structured branch log |
| `src/rl/rl_branchrule.hpp` | Shared inference/fallback statistics |
| `code/scip_tree.cpp` | RL CLI, plugin registration, output metrics, fallback priority |
| `tests/model_runner_parity.cpp` | Standalone C++ full-Q fixture runner |
| `tests/test_custom_branchrule.cpp` | Aviation category and existing branchrule unit tests |
| `scripts/validate_cpp_mlp_parity.py` | Python/C++ Q and argmax parity validation |
| `scripts/validate_rl_mlp_integration.py` | End-to-end correctness and fallback acceptance checks |
| `Makefile`, `CMakeLists.txt` | SCIP plus packaged LibTorch build and rpath configuration |
| `README.md` | Phase-6 build, run, parity, and validation commands |

## 4. Environment and model

| Component | Value |
|---|---|
| SCIP | 8.0.4 |
| Compiler | GCC 11.4, C++17 |
| Python environment | `rl4scip` |
| PyTorch/LibTorch | 2.5.1+cu121 |
| C++ ABI | `_GLIBCXX_USE_CXX11_ABI=0` |
| GPU used for CUDA test | Tesla V100-PCIE-32GB |
| TorchScript model | `artifacts/models/mlp/best_model_scripted.pt` |
| Model size | 98 KiB |
| Model SHA-256 | `9b2c0b17a4b055d18ca3615ce61dfbdc6437127e1bf83958c8d4e9a0e8470a20` |

The server has the CUDA runtime needed by PyTorch but no `nvcc`. CMake therefore
links the PyTorch package's existing shared libraries directly. This is sufficient
because the project compiles no custom CUDA source.

## 5. Build and test commands

```bash
conda run -n rl4scip make
conda run -n rl4scip make model-runner-parity
conda run -n rl4scip make test-custom-branching

conda run -n rl4scip cmake -S . -B build/cmake-phase6d \
  -DSCIP_ROOT=/home/duweiyue25/SCIP/scipoptsuite-8.0.4 \
  -DTORCH_ROOT=/home/duweiyue25/conda/envs/rl4scip/lib/python3.11/site-packages/torch
conda run -n rl4scip cmake --build build/cmake-phase6d -j2
```

Both Make and CMake completed. The custom branchrule test passed from both build
trees. Compiler warnings are inherited unused-parameter warnings from SCIP/LibTorch
headers and pre-existing unused parameters in `scip_tree.cpp`.

## 6. Python/C++ parity

The fixed phase-5 observation contains 151 candidates. Python and C++ both emit the
complete Q vector before comparing it.

| Device | Max absolute error | Python argmax | C++ argmax | Result |
|---|---:|---:|---:|---|
| CPU | 0.0 | 17 | 17 | Pass |
| CUDA | 0.0 | 17 | 17 | Pass |
| CPU, CMake binary | 0.0 | 17 | 17 | Pass |

Evidence is stored in `results/phase6/parity/` and
`results/phase6/parity_cmake/`.

## 7. Real-instance integration results

All rows use `real_09`, SCIP seed 0, one thread, and a 60-second limit.

| Run | Status | Objective | Nodes | RL decisions | Fallbacks | Inference total |
|---|---|---:|---:|---:|---:|---:|
| SCIP default | optimal | 0.0022776 | 9 | 0 | 0 | 0 |
| RL-MLP CPU | optimal | 0.0022776 | 13 | 12 | 0 | 0.005899 s |
| RL-MLP CPU repeat | optimal | 0.0022776 | 13 | 12 | 0 | 0.037757 s |
| RL-MLP CUDA | optimal | 0.0022776 | 13 | 12 | 0 | 0.692156 s |
| Missing model, relpscost fallback | optimal | 0.0022776 | 9 | 0 | 10 | 0 |
| Candidate gate, relpscost fallback | optimal | 0.0022776 | 9 | 0 | 10 | 0 |
| Candidate gate, default fallback | optimal | 0.0022776 | 9 | 0 | 10 | 0 |

Every run reported a feasible solution and passed the project's cycle check. The
CPU RL run recorded 12 legality checks and zero illegal actions. Its mean callback
inference was 0.492 ms and maximum was 1.309 ms after one-time warm-up. The separate
repeat selected the same variable name and candidate index at every decision.

CUDA is functionally correct but inappropriate for this 21,761-parameter MLP and
small dynamic candidate batches: device transfer and synchronization dominate. CPU
inference was about 117 times lower in accumulated callback time in this test.

Wall-clock measurements varied materially under shared-server load and include
model startup, so these single runs are not used for speedup claims.

## 8. Structured outputs

The single-run JSON now includes:

- `branch_decisions`;
- `rl_inference_total`, `rl_inference_mean`, `rl_inference_max`;
- `fallback_count`;
- model, device, fallback, depth, and minimum-candidate configuration.

Optional branch CSV logs contain node, depth, candidate count, selected candidate and
variable indices, variable name, LP value, fractionality, Q value, inference and
selection time, legality, SCIP result, and fallback reason. Logging is disabled when
`--rl-log` is omitted.

Reproducibility evidence is stored under `results/phase6/`: `build_make.log`,
`build_cmake.log`, `test_cpp.log`, `test_python.log`, parity Q vectors and summaries,
solver stdout logs, run JSON files, branch CSV files, and
`integration_validation.json`.

## 9. Automated acceptance

`results/phase6/integration_validation.json` records all checks as passed:

- all default, RL, repeat, and fallback runs are optimal and feasible;
- RL and fallback objectives match default within tolerance;
- every RL action is legal;
- the repeat decision sequence is identical;
- missing-model and candidate-gate fallback paths are exercised.

The original `default` mode does not instantiate or load `ModelRunner`; RL callback
and inference counters remain zero. The model is loaded only for `rl-mlp`.

## 10. Limitations and phase-7 gate

The phase-6 correctness gate is passed. CPU should be the default MLP deployment
device, and shallow RL plus reliability-pseudocost fallback is available through
`--rl-max-depth` for later experiments.

The tested phase-5 model did not reduce nodes relative to SCIP-default on `real_09`.
This does not invalidate integration, but it means there is still no deployment
claim. Phase 7 may implement the bipartite GCNN as approved, while model usefulness
must ultimately be judged by held-out and transfer experiments in phase 8. GCNN
exportability and sparse-message-passing inference cost should be measured before
committing to GPU deployment.
