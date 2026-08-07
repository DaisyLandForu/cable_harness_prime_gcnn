# 阶段 4：BBMDP 训练环境报告

## 1. 阶段目标与结论

本阶段在真实航空布线 CIP 实例上建立了 BBMDP 风格的 Python branching 环境。环境能够逐次返回变量-约束二部图、全局 B&B 特征和当前合法动作集合，接受一个候选变量动作，再返回节点增量奖励和下一状态。

阶段 4 验收通过：43 条真实 B&B transition 全部满足动作合法性、维度、奖励、终止 bootstrap 和变量索引映射检查；固定 seed 的重复 episode 在动作、奖励、节点数和 LP iterations 上完全一致。原 C++ default 求解器在安装 Python 依赖前后的稳定指标也完全一致。

本阶段没有安装 PyTorch、没有训练 DQN，也没有得到任何 RL 性能结论。模型训练必须等待阶段 5 批准。

## 2. 环境与隔离策略

继续使用用户指定的 `rl4scip` conda 环境，没有创建第二套运行环境。为避免 Python 扩展构建修改原 C++ 工具链，SCIP 8.0.4 被安装到项目内的独立前缀：

```text
artifacts/environment/phase4/scip804_prefix
```

PySCIPOpt 和 Ecole 链接该前缀中的 `libscip.so.8.0`；原项目仍使用 `/home/duweiyue25/SCIP/scipoptsuite-8.0.4` 和原 `build/scip_tree`。安装前后保存了：

- `conda_explicit_before.txt` / `conda_explicit_after.txt`；
- `native_sha256_before.txt` / `native_sha256_after.txt`；
- `pip_freeze_after.txt`。

两份 native SHA-256 清单无差异，说明原 `scip_tree` 和原 SCIP 动态库未被替换。阶段 4 实际版本为：

| 组件 | 版本 |
|---|---|
| Python | 3.11.15 |
| SCIP / SoPlex | 8.0.4 / 6.0.4 |
| PySCIPOpt | 4.3.0，源码构建 |
| Ecole | 0.8.1，源码构建 |
| NumPy / SciPy | 1.26.4 / 1.13.1 |
| pandas / PyYAML | 2.2.3 / 6.0.2 |
| pytest | 8.3.5 |

Python 通用依赖固定在 `requirements/phase4.txt`。PySCIPOpt/Ecole 没有写成普通 wheel 依赖，因为它们必须与当前 SCIP ABI 对齐。

## 3. 兼容性与 BBMDP 复用审计

审计的源码版本为：

| 项目 | 版本/commit | 结论 |
|---|---|---|
| PySCIPOpt | v4.3.0 / `a5936a7617904cf41c3f176b4516acbf6bf3791f` | 与 SCIP 8 系列 API 对齐，已从源码构建 |
| Ecole | v0.8.1 / `c6a6d872b773192608b30bdd3b99481a8b6f55ba` | BSD-3-Clause；已针对 Python 3.11 构建并通过 doctor |
| `abfariah/bbmdp` | `67679978c72b484a65efa7d4d78e8fee3e41b479` | README/安装说明接近当前栈，但仓库根目录没有 LICENSE/COPYING/NOTICE |

BBMDP 仓库列出的关键环境是 SCIP 8.0.3、PySCIPOpt 4.3.0、Ecole、PyTorch 2.6 和 PyG 2.6.1。本项目是 SCIP 8.0.4，因此没有降级项目，也没有直接复制 BBMDP 的 `Branching`、`RetroNodeBipartite`、agent、learner 或 replay buffer 代码。当前实现只复用有明确 BSD-3-Clause 许可证的 Ecole 接口，并独立实现环境契约。若以后要复用 BBMDP 源码，需先取得或确认其许可证。

Ecole 0.8.1 固定的 pybind11 2.9.1 无法在 Python 3.11 下编译，错误来自 CPython 3.11 不再暴露完整 `PyFrameObject`。构建时仅对临时 Ecole 源码将 pybind11 更新为 2.10.4；补丁保存在 `artifacts/environment/phase4/ecole-0.8.1-python311.patch`。未修改 Ecole、SCIP 或 PySCIPOpt 的算法代码。

验证命令：

```bash
conda run -n rl4scip python -m ecole.doctor
```

doctor 报告 Ecole 0.8.1 的编译时和运行时 SCIP 均为 8.0.4，完整日志位于 `results/phase4/logs/ecole_doctor.log`。

## 4. 环境定义

### 4.1 状态

`CopiedNodeBipartite` 基于 Ecole 0.8.1 `NodeBipartite`，每一步复制并冻结：

- constraint features：5 维；
- variable features：19 维；
- sparse edge indices 和 1 维 edge coefficient；
- 15 维全局 B&B 特征；
- transformed variable name；
- 当前 action set。

全局特征包括 depth、processed/total/open nodes、可行/不可行叶节点、LP iterations、solving time、primal/dual bound、gap、incumbent 数量，并为可能无穷的 bound/gap 配套 finite indicator。数组中的 NaN/Inf 在边界处被清理。完整顺序记录在 `configs/rl/bbmdp_feature_schema.json`。

环境返回的状态不保存 `SCIP*`、`SCIP_VAR*` 或节点指针。变量名、索引、特征和搜索树 node/parent ID 均复制为 Python 值，因此 replay buffer 不会持有失效 SCIP pointer。

### 4.2 动作

动作是 Ecole 当前 `action_set` 中的整数，即 transformed problem `variable_features` 的行索引。`step()` 在调用 SCIP 前检查：

1. 动作属于当前 action set；
2. 索引在 variable feature 行范围内；
3. 索引对应的变量名与当前模型中的 transformed variable 一致。

非法动作直接抛出 `ValueError`，不会提交给 SCIP。这个 action schema 与阶段 3 的 C++ candidate-safe 原则一致；阶段 6 仍需做固定 observation 的 Python/C++ parity test。

### 4.3 奖励

默认奖励严格按任务定义：

```text
r_t = -(N_{t+1} - N_t)
```

其中 `N` 是 SCIP 已创建的总节点数。也支持 `constant_minus_one`。`gamma` 被配置类固定为 1，其他值会被拒绝。

节点限制恰好触发时，最后一步可能得到 `-0`：SCIP 在创建下一个节点前停止，此时 `N_{t+1}=N_t`。这是奖励公式的直接结果，不是日志缺失。

### 4.4 终止与截断

- `optimal`、`infeasible`、`unbounded`、`inforunbd`：terminal；
- `timelimit`、`nodelimit`、`memlimit`、`userinterrupt`、SCIP/Ecole 异常：truncation；
- terminal bootstrap 恒为 0；
- 默认 `bootstrap_on_truncation=false`，所以 timeout/node-limit transition 也使用 0 bootstrap；
- SCIP/Ecole 异常被记录为 `scip_error`，不会伪装成正常 terminal。

## 5. Controlled BBMDP 参数

`BBMDPConfig.scip_parameters()` 设置：

| 目标 | SCIP 参数 |
|---|---|
| DFS node selection | `nodeselection/dfs/stdpriority=1000000`、`memsavepriority=1000000` |
| 关闭 separation | `separating/maxrounds=0` |
| 关闭 restart | `estimation/restarts/restartpolicy=n`、`limits/restarts=0`、`presolving/maxrestarts=0` |
| 单线程 | `parallel/minnthreads=1`、`parallel/maxnthreads=1`、`lp/threads=1` |
| 固定随机性 | `randomseedshift`、`permutationseed`、`lpseed` 使用同一 seed |
| episode 资源限制 | `limits/time`、可选 `limits/nodes` |

这里为控制变量关闭了包括 root 在内的 separation，比“仅关闭 root 后 cuts”更严格。阶段 8 的 `controlled_bbmdp` 协议可以增加 root cuts 作为独立配置，但不得与当前日志混为同一实验协议。

## 6. 修改文件

| 文件 | 作用 |
|---|---|
| `python/rl_branching/config.py` | 配置校验、奖励模式和受控 SCIP 参数 |
| `python/rl_branching/observation.py` | 复制式二部图与全局特征提取、schema 校验 |
| `python/rl_branching/environment.py` | reset/step、动作校验、奖励、终止/截断和搜索树快照 |
| `python/rl_branching/__init__.py` | 稳定的阶段 5 导入接口 |
| `scripts/run_bbmdp_smoke.py` | real CIP episode、random/mostinf 策略和 JSON 转移日志 |
| `scripts/validate_bbmdp_transitions.py` | 离线验证 action、shape、reward、bootstrap 和复现性 |
| `tests/python/test_bbmdp_environment.py` | 配置及真实 SCIP transition 单元测试 |
| `configs/rl/environment_smoke.yaml` | node-limit 20 的默认 smoke 配置 |
| `configs/rl/environment_constant_smoke.yaml` | constant reward、node-limit 3 配置 |
| `configs/rl/bbmdp_feature_schema.json` | Python/C++ 共用特征契约起点 |
| `requirements/phase4.txt` | 通用 Python 依赖版本 |
| `README.md` | 阶段 4 运行和测试命令 |

没有修改阶段 3 的 C++ branchrule，也没有修改 MILP 变量、约束、目标函数或解后处理。

## 7. 执行与验证命令

单元测试：

```bash
conda run -n rl4scip env PYTHONPATH=python \
  pytest -q tests/python/test_bbmdp_environment.py
```

真实 episode：

```bash
conda run -n rl4scip env PYTHONPATH=python \
  python scripts/run_bbmdp_smoke.py \
  --config configs/rl/environment_smoke.yaml \
  --instance data/instances/train/real_06.cip \
  --policy random \
  --output results/phase4/transitions/real_06_random_seed0.json
```

离线验证与重复运行比较：

```bash
conda run -n rl4scip python scripts/validate_bbmdp_transitions.py \
  results/phase4/transitions/real_06_random_seed0.json \
  results/phase4/transitions/real_06_mostinf_seed0.json \
  results/phase4/transitions/real_06_constant_seed0.json \
  results/phase4/transitions/real_09_root_terminal_seed0.json \
  --repeat-pair \
  results/phase4/transitions/real_06_random_seed0.json \
  results/phase4/transitions/real_06_random_seed0_repeat.json
```

C++ 回归：

```bash
conda run -n rl4scip make
conda run -n rl4scip make test-custom-branching
conda run -n rl4scip ./build/scip_tree \
  --instance-id 9 --branching default --seed 0 \
  --time-limit 60 --threads 1 \
  --output-json results/phase4/cpp_smoke_after.json
```

## 8. 真实实例结果

所有 episode 使用阶段 2 从真实 `code/data` 导出的 CIP，没有使用 `code/data/synthesis/`。`real_06` 在 controlled profile 下会产生分支；`real_09` 在该 profile 下根节点即求得最优，因此专门用于 zero-transition terminal 测试。

| 实例/策略 | reward | status | transitions | nodes | LP iterations | total reward |
|---|---|---|---:|---:|---:|---:|
| real_06 / random | node increment | nodelimit | 20 | 20 | 9,058 | -19 |
| real_06 / random repeat | node increment | nodelimit | 20 | 20 | 9,058 | -19 |
| real_06 / mostinf | node increment | nodelimit | 20 | 20 | 9,159 | -19 |
| real_06 / random | constant -1 | nodelimit | 3 | 3 | 8,944 | -3 |
| real_09 / random | node increment | optimal | 0 | 1 | 2,924 | 0 |

验证器报告 `validated 4 episodes and 43 transitions`。random seed 0 的两次运行具有完全相同的 20 个 action、逐步 reward、observation shape、节点数和 LP iterations；solving time 只作观测指标，不要求逐位一致。

pytest 结果为 `5 passed`，覆盖：受控参数、gamma 限制、真实 transition 契约、非法 action、candidate-name 映射、奖励方程、下一状态维度、terminal bootstrap、环境关闭后的无 pointer 生命周期，以及 0.001 秒 time limit 的显式 timeout 处理。

## 9. C++ 无回归结果

安装前后均运行真实实例 9、default branching、seed 0、单线程、60 秒限制：

| 指标 | 安装前 | 安装后 |
|---|---:|---:|
| status | optimal | optimal |
| objective | 0.0022776 | 0.0022776 |
| 业务目标 | 0.0025445 | 0.0025445 |
| nodes | 9 | 9 |
| LP iterations | 3,596 | 3,596 |

`make test-custom-branching` 同样通过，SCIP BMS 未报告内存泄漏。环境增加的 Python 包不会在 C++ 程序启动或 branching callback 中加载；因此阶段 4 对原求解路径没有运行时开销。

## 10. 风险与阶段 5 边界

1. Ecole 0.8.1 已可用，但项目停留在较旧的 SCIP 8 ABI；后续不要单独升级 PySCIPOpt、Ecole 或 SCIP。
2. Ecole 的 action index 是 transformed variable row；最终 C++ 特征提取器必须通过保存 observation 做逐值 parity，不能假设阶段 3 的 candidate index 与之相同。
3. `NodeBipartite` 的 19/5 维基础特征来自 Ecole，航空布线变量类别 one-hot 和更丰富 pseudocost/history 特征尚未加入；这是阶段 5 Candidate MLP 的明确工作。
4. 当前 smoke episode 只验证环境闭环，不构成训练数据规模，也不支持 RL 优于 random/default 的结论。
5. `real_09` 在 controlled profile 根节点终止，因此训练实例必须按“能产生足够 branching transitions”筛选，同时仍按原 MILP 实例划分 train/validation/test。
6. PyTorch、replay buffer、Double DQN、3-step return 和 checkpoint 尚未实现。阶段 5 应先安装与当前 CUDA/驱动匹配的 PyTorch，再做 500--2000 steps smoke，而不是直接引入 GCNN/PyG。

阶段 5 可以直接复用本阶段的 `BBMDPBranchingEnv`、不可变 observation、action mask 和 transition JSON 契约。在用户批准阶段 5 前，本项目不会开始模型训练。
