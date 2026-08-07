# 阶段 7：Bipartite GCNN-BBMDP 实现与验证报告

## 1. 阶段结论

阶段 7 已完成最小但完整的 Bipartite GCNN-BBMDP 闭环：SCIP 状态图提取、
两轮二部图消息传递、Candidate Q 评分、Double DQN、3-step return、PER、软更新
target network、标量 Q、可选 HL-Gauss、TorchScript 导出，以及 C++ `rl-gcnn`
分支规则均已实现并真实运行。

工程正确性验收通过。未参与训练的真实 `real_09` 上，C++ GCNN 的 8 次动作全部
合法，得到与 SCIP-default 相同的最优目标 `0.0022776`；独立重复运行的完整分支
序列一致；缺失模型和深度门限均能回退到 `relpscost`。

学习效果只形成了 pilot 级信号。受控 DFS 验证中，标量 GCNN 相对 random 将 3 个
seed 的截断平均节点数从 97 降到 81，降幅 16.5%，平均 SCIP solving time 从
34.30 秒降到 32.80 秒，降幅 4.4%。这不是相对 SCIP-default 的提升证据，也不是
最终部署结论。生产 C++ 短实例中的动态图推理占 solving time 的 36% 至 60%，明显
未达到最终目标的 5%；阶段 8 必须重点评估浅层混合策略和推理优化。

## 2. 数据边界

| 用途 | 真实实例 | 说明 |
|---|---|---|
| 训练 | `real_06` | 仅用于环境交互和 replay |
| 验证 | `real_08` | checkpoint 选择及 random/untrained/RL pilot |
| C++ 集成测试 | `real_09` | 未进入训练和 checkpoint 选择 |
| transfer | 未使用 | 保留给阶段 8 |

本阶段未使用 `code/data/synthesis` 中的人工合成数据，也未把同一 MILP 的 B&B 状态
拆到不同 split。

## 3. 图状态与动作

一个状态包含完整的当前变量-约束二部图：

- 变量节点：19 维 Ecole/SCIP 特征和 6 维航空布线变量类别；
- 约束节点：14 维行特征和 6 维航空布线约束类别；
- 边：原始系数、按行 L2 归一化系数和符号，共 3 维；
- 全局树状态：depth、已处理/总/开放节点、叶节点、LP 迭代、primal/dual bound、
  gap 和 incumbent 数量，共 14 维；
- 动作集合：SCIP 当前 fractional LP branching candidates 的 transformed-variable
  index mask。

约束的 lhs/rhs 展开顺序和边符号与当前 Ecole 0.8.1 `NodeBipartite` 实现一致。
`real_06` 探针得到 28,089 个展开行、8,372 个变量和 93,555 条边；Python 重建的
行/列索引完全一致，归一化边最大误差约 `2.324e-8`。图和 action set 在 transition
中复制为不可变数组，不保存跨步失效的 SCIP pointer。

航空布线变量类别为 `m/z/y/absf/f/other`；约束类别为
`flow/absolute/topology/selection/imbalance/other`。完整维度和顺序保存在每个模型
目录的 `feature_schema.json` 中。

## 4. 网络与学习算法

`BipartiteGCNNQNetwork` 使用基础 PyTorch 张量操作实现，没有引入 PyTorch
Geometric：

1. 分别编码 variable、constraint、edge 和 global 特征；
2. variable-to-constraint 消息传递并用 `index_add_` 做 mean aggregation；
3. constraint-to-variable 消息传递；
4. 只收集 action mask 指定的候选变量 embedding；
5. 将候选 embedding 与全局 embedding 拼接，输出每个候选的 Q。

学习器支持 Double DQN、3-step return、gamma 1、prioritized replay、importance
sampling、soft target update、epsilon-greedy、可选 Boltzmann、Adam 和 gradient
clipping。truncated episode 默认不 bootstrap。

标量版本用 Smooth L1 loss。HL-Gauss 版本输出 18 bins，对负 Q 使用
`z=log2(-Q)`，再以 `Q=-2^z` 恢复期望值，`z_min=-1`、`z_max=12`、
`sigma=0.75` 均由 YAML 配置。

## 5. 新增或修改文件

| 路径 | 作用 |
|---|---|
| `python/rl_branching/observation.py` | 扩展约束与边特征，并保持阶段 4/5 接口兼容 |
| `python/rl_branching/graph_features.py` | 图状态、类别、action mask 和归一化 |
| `python/rl_branching/gcnn_model.py` | 标量/HL-Gauss 二部图 GCNN 与 TorchScript 导出 |
| `python/rl_branching/graph_replay.py` | PER 与 importance-sampling 权重 |
| `python/rl_branching/gcnn_dqn.py` | Graph Double DQN 学习器 |
| `python/rl_branching/gcnn_config.py` | GCNN YAML 配置校验 |
| `python/rl_branching/gcnn_trainer.py` | 真实 Ecole 训练、验证、checkpoint 和评测 |
| `scripts/train_gcnn.py` | 训练入口 |
| `scripts/validate_gcnn_artifacts.py` | loss、reload、eager/TorchScript parity |
| `scripts/validate_cpp_gcnn_parity.py` | Python/C++ 完整 Q 向量 parity |
| `scripts/validate_rl_gcnn_integration.py` | C++ 端到端自动验收 |
| `configs/rl/gcnn_*.yaml` | smoke、pilot、HL-Gauss 配置 |
| `src/rl/scip_graph_feature_extractor.*` | C++ 当前 LP 二部图提取 |
| `src/rl/gcnn_model_runner.*` | 一次加载的 LibTorch 图模型 runner |
| `src/rl/rl_gcnn_branchrule.*` | SCIP `rl-gcnn` 分支插件、日志和 fallback |
| `tests/python/test_gcnn.py` | 图、PER、标量/HL loss 和导出测试 |
| `tests/gcnn_model_runner_parity.cpp` | 固定图观测的 C++ Q runner |
| `code/scip_tree.cpp` | 增加 `--branching rl-gcnn` 注册；MILP 未改 |
| `Makefile`, `CMakeLists.txt` | GCNN C++ 目标及 LibTorch 链接 |

## 6. 执行命令

标量 smoke 和 pilot：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n rl4scip env \
  PYTHONPATH=python CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python scripts/train_gcnn.py --config configs/rl/gcnn_smoke.yaml

CUDA_VISIBLE_DEVICES=0 conda run -n rl4scip env \
  PYTHONPATH=python CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python scripts/train_gcnn.py --config configs/rl/gcnn_pilot.yaml
```

HL-Gauss 闭环：

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n rl4scip env \
  PYTHONPATH=python CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  python scripts/train_gcnn.py --config configs/rl/gcnn_hlgauss_smoke.yaml
```

构建与测试：

```bash
conda run -n rl4scip make -j2
conda run -n rl4scip make test-custom-branching gcnn-model-runner-parity
conda run -n rl4scip env PYTHONPATH=python pytest -q \
  tests/python/test_bbmdp_environment.py \
  tests/python/test_candidate_mlp.py tests/python/test_gcnn.py
```

C++ 真实实例求解：

```bash
conda run -n rl4scip ./build/scip_tree \
  --instance-id 9 --branching rl-gcnn --seed 0 \
  --time-limit 60 --threads 1 \
  --rl-model artifacts/models/gcnn/best_model_scripted.pt \
  --rl-device cuda --rl-fallback relpscost \
  --rl-max-depth -1 --rl-min-candidates 1 \
  --rl-log results/phase7/e2e/real09_rl_gcnn_cuda_branches.csv \
  --output-json results/phase7/e2e/real09_rl_gcnn_cuda.json
```

## 7. 标量 GCNN 训练结果

Smoke 完成 100 个梯度步，loss 全部有限，范围 `0.205` 至 `2.561`；checkpoint、
schema、normalization 和 TorchScript 均成功保存。Smoke 的 30-node 上限太小，所有
策略均触顶，因此只证明闭环。

Pilot 完成 1,000 个梯度步和 3 个 episode，replay size 为 64，训练 wall time 为
595.3 秒。100 个结构化更新日志中的 loss 全部有限，范围 `0.001005` 至
`15.3623`。验证节点数依次为：

| 梯度步 | 状态 | 节点数 |
|---:|---|---:|
| 308 | optimal | 79 |
| 636 | nodelimit | 100 |
| 1,000 | optimal | 76 |

最后一个 checkpoint 是当前 best。进程 RSS 峰值约 1,920.7 MiB，PyTorch allocated
GPU memory 峰值约 67.0 MiB；单张 V100 足够，训练瓶颈仍是 SCIP 环境交互。

固定验证实例 `real_08`、node limit 100 的 pilot 对比如下：

| Seed | Random | Untrained | RL |
|---:|---|---|---|
| 100 | optimal / 91 | nodelimit / 100 | optimal / 76 |
| 101 | optimal / 100 | nodelimit / 100 | nodelimit / 100 |
| 102 | nodelimit / 100 | nodelimit / 100 | optimal / 67 |

Random 和 RL 都求解 2/3。按 node limit 计入截断，平均节点数分别为 97 和 81，
RL 减少 16.5%；平均 solving time 分别为 34.30 秒和 32.80 秒，RL 减少 4.4%。
RL 三次求解的模型前向累计 3.142 秒，占其 solving time 总和约 3.19%。seed 101
仍然触顶，因此目前只能判断 aggregate 优于 random，不能判断多 seed 方向完全一致。

## 8. HL-Gauss 闭环结果

HL-Gauss smoke 完成 40 个梯度步，8 个采样日志的 loss 全部有限，从 `2.8420`
下降到 `1.9122`。best/last checkpoint 和 18-bin TorchScript 均可重新加载。20-node
评估中 random、untrained 和 RL 都触顶，不提供效果结论。

CUDA eager/TorchScript 的 argmax 均为位置 3，最大绝对 Q 误差为 `3.052e-5`。
由于 `-2^z` 会放大浮点归约差异，HL-Gauss 明确使用 `1e-4` parity tolerance；
标量 GCNN 仍使用 `1e-5`。

## 9. Python/C++ parity

标量 best model 的固定真实图观测有 154 个候选。完整 Q 向量结果为：

| Runner | 设备 | 最大绝对误差 | Python/C++ argmax | 结果 |
|---|---|---:|---|---|
| Make | CPU | 0 | 24 / 24 | Pass |
| Make | CUDA | `5.722e-6` | 24 / 24 | Pass |
| CMake | CPU | 0 | 24 / 24 | Pass |

模型在 solver 初始化时加载一次；callback 内只构造张量、执行 inference mode 前向、
检查有限值并在候选向量内做稳定 argmax。没有启动 Python 子进程，也没有在每次
callback 读取模型文件。

## 10. C++ 端到端结果

`real_09`、seed 0、one thread、60 秒限制：

| 模式 | 状态 | 目标 | 节点 | RL 决策 | Fallback | 推理累计 |
|---|---|---:|---:|---:|---:|---:|
| SCIP-default | optimal | 0.0022776 | 9 | 0 | 0 | 0 |
| RL-GCNN CUDA | optimal | 0.0022776 | 8 | 8 | 0 | 1.758 s |
| RL-GCNN CUDA repeat | optimal | 0.0022776 | 8 | 8 | 0 | 2.541 s |
| RL-GCNN CPU | optimal | 0.0022776 | 8 | 8 | 0 | 1.000 s |
| missing model | optimal | 0.0022776 | 9 | 0 | 10 | 0 |
| depth 0 hybrid | optimal | 0.0022776 | 12 | 1 | 10 | 0.076 s |

两次 CUDA 运行的 8 个 node/depth/candidate/variable 决策完全一致，所有运行均通过
原项目 cycle check。自动验收文件
`results/phase7/e2e/integration_validation.json` 的所有检查为 true。

一次运行中 8 节点少于 default 的 9 节点，但样本和 seed 都只有一个，不能据此
声称优于 default。更重要的是，生产路径动态图推理在该短实例中占 solving time
36%（CPU）以及 48%/60%（CUDA）。CUDA 第 4 次以后通常为 5 至 25 ms，但前 3 次
有 0.2 至 1.3 秒的动态大图冷启动；CPU 单次较稳定但约 0.125 秒。该问题必须进入
阶段 8 的 inference overhead 和浅层混合消融。

## 11. 测试与日志

- Python：阶段 4/5/7 共 17 项测试全部通过，137.61 秒；
- Make：完整 `scip_tree` 构建通过；
- CMake：`scip_tree`、MLP/GCNN parity runner 和 custom test 全部构建通过；
- C++ custom branchrule test：通过；
- `git diff --check`：阶段收尾执行并记录；
- 编译警告来自 SCIP 头文件和原 `scip_tree.cpp` 的既有 unused parameters。

主要产物位于：

- `artifacts/models/gcnn/`：标量 pilot；
- `artifacts/models/gcnn/smoke/`：标量 smoke；
- `artifacts/models/gcnn_hlgauss/smoke/`：HL-Gauss smoke；
- `results/phase7/parity_cpp/`、`results/phase7/parity_cmake/`：完整 Q parity；
- `results/phase7/e2e/`：default、RL、repeat、fallback JSON/CSV；
- `results/phase7/*.log`：训练、构建和测试日志。

标量部署模型 SHA-256 为
`3ffac237bf189f4807f9d3d6334d3624cc8fd5c133e1b3cddcf093e424a2ab33`。

## 12. 验收与阶段 8 入口

| 条件 | 结果 |
|---|---|
| 完整变量-约束图与候选 mask | Pass |
| 两轮消息传递且无需 PyG | Pass |
| scalar Q、Double DQN、3-step、PER、soft target | Pass |
| 可选 18-bin HL-Gauss | Pass，效果未比较 |
| TorchScript/LibTorch 稳定导出 | Pass |
| Python/C++ 完整 Q 和 argmax parity | Pass |
| C++ action 100% 合法、目标正确、fallback 可用 | Pass |
| validation aggregate 优于 random | Pilot pass：nodes -16.5% |
| 相对 SCIP-default 的可靠收益 | 尚未评估 |
| C++ 推理占比低于 5% | Fail on short `real_09` |

因此阶段 7 的工程目标完成，可以进入阶段 8。下一阶段应优先做统一的多实例、
多 seed controlled/production 对比，并先测 `D in {5,10,20,50,unlimited}` 的浅层
GCNN + 深层 relpscost；若 GCNN 节点收益无法覆盖当前动态图提取和推理开销，MLP
仍是更现实的部署候选。
