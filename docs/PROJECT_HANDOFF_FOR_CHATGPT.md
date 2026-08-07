# 航空布线 RL Branching 项目工作交接文档

> 用途：作为阶段 0-8 的事实底稿，交给 ChatGPT 进一步整理组会汇报、PPT 大纲、讲稿和答疑材料。
>
> 数据截止：2026-08-07；仓库提交：`c9e779a6df5a1a1c3aa1a005a410559340230962`（`add rl`）。

## 1. 仓库与复现基线

- 远程仓库：<https://github.com/DaisyLandForu/cable_harness_rl.git>
- 主分支：`main`
- 本文核对的提交：`c9e779a6df5a1a1c3aa1a005a410559340230962`
- 工作目录：`/home/duweiyue25/acyclic_cable_harness`
- Conda 环境：`rl4scip`
- 核心 C++ 入口：`code/scip_tree.cpp`
- RL Python 包：`python/rl_branching/`
- 真实原始数据：`code/data/edges-{1..9}.csv`、`code/data/pairs-{1..9}.csv`
- 人工合成数据：`code/data/synthesis/`

本项目遵循两条边界：没有改变航空布线 MILP 的变量、约束和目标函数；不指定 RL 参数时仍走原项目的 default SCIP 求解路径。最终正式训练和阶段 8 评测以新的 `code/data` 真实数据为主，合成数据只用于阶段 2 的数据生成能力和规模覆盖验证。

## 2. 先读结论

项目已完成从可复现 baseline、真实实例划分、BBMDP 环境、Candidate MLP-DQN、Bipartite GCNN-DQN，到 C++ SCIP 原生推理和双协议实验的闭环。工程正确性通过，但当前 RL branching **尚不满足生产部署判据**。

阶段 8 在 430 次主实验中得到：

| production-scip 方法 | solved rate | shifted gmean wall time | mean PAR-2 | 相对 default 成对 speedup（95% CI） |
|---|---:|---:|---:|---:|
| default | 25.7% | 21.329 s | 47.388 s | 1.000 |
| random | 31.4% | 20.615 s | 45.139 s | 1.077（0.980, 1.196） |
| mostinf | 34.3% | 20.444 s | 44.078 s | 1.109（1.024, 1.221） |
| RL-MLP | 28.6% | 21.301 s | 46.167 s | 1.021（0.964, 1.089） |
| RL-GCNN | 37.1% | 23.984 s | 44.387 s | 0.966（0.855, 1.095） |

核心判断：

1. MLP 的 wall time 只比 default 改善约 0.13%，置信区间跨 1，远低于预设的 5% 判据。
2. GCNN solved rate 较高，但 wall time 比 default 恶化约 12.4%。
3. jointly solved 样本中，MLP 和 GCNN 分别比 default 多约 6.8% 和 19.9% 节点。
4. RL 没有明显优于 random；在受控 DFS 协议中甚至显著更慢。
5. 2,454 次主实验 RL action 全部合法，无崩溃或意外 fallback，说明问题主要在策略质量和成本，而非接入正确性。
6. 当前工程候选应优先保留 MLP；它成本低、部署简单，但还不应替换 SCIP-default。

## 3. 事实口径与阅读优先级

为避免把不同阶段数据混为一谈，汇报时按以下优先级引用：

1. 最终结论以 `docs/FINAL_RL_BRANCHING_REPORT.md` 和 `results/final/` 为准。
2. 阶段 5、阶段 7 的 pilot 只证明训练和接入闭环，也可用于解释模型演化，不能替代正式实验。
3. 阶段 1 baseline 用于证明原求解器兼容、参数生效和早期性能画像；阶段 8 baseline 才用于和最终 RL 做统一协议对比。
4. 最终 MLP/GCNN 模型使用 14 维全局特征，不含 wall-clock solving time。阶段 4 的历史配置 `configs/rl/bbmdp_feature_schema.json` 曾记录 15 维，不能作为部署模型的最终 schema；应以 `artifacts/models/*/feature_schema.json` 为准。
5. 最大实例 `real_04` 在 30 秒内没有进入 B&B，不能表述为“RL branching 在最大实例失败”，只能表述为“当前实验预算下无法评估 branching 效果”。

## 4. 原航空布线 MILP

### 4.1 业务目标

模型在候选航空布线网络上选择分层拓扑和线缆路由，使结构保持无环并满足连接、流平衡和层级关系。SCIP objective 主要是边长与聚合线缆权重对应的路由代价；业务总成本还加上固定的叶端接入成本。

### 4.2 变量类别

| 变量 | 类型 | 含义 |
|---|---|---|
| `m_k_p` | binary | 线缆对/需求 `k` 分配到层 `p` |
| `z_i_j_p` | binary | 层 `p` 中选择有向拓扑边 `(i,j)`，是主要整数分支对象之一 |
| `f_i_j_k` | continuous | 需求 `k` 在有向边 `(i,j)` 上的流量，范围约为 `[-1,1]` |
| `absf_i_j_k` | continuous | `f` 的绝对值线性化变量 |
| `y_i_p` | continuous | MTZ 风格拓扑次序变量，用于表达无环性 |

### 4.3 约束类别

- `fforbid`：禁止不允许的流。
- `abs1/abs2`：流绝对值线性化。
- `flow_balance`：节点流平衡。
- `flow_symmetry`：双向边流关系。
- `onlym`：需求只能选择相应层。
- `imbalance`：端点或层分配关系。
- `zlower`：拓扑边与流/分配之间的联动下界。
- `topo_seq1/topo_seq2`：MTZ 风格无环次序。
- `only_father`：每个节点的父边选择约束。

阶段 0 还记录了两个模型层风险，但本项目没有擅自修改：边权用整数存储时可能截断小数，以及 `zlower` 的方向性值得后续独立验证。这些属于原模型审计问题，不应与 RL branching 效果混淆。

## 5. 阶段 0-8 工作脉络

| 阶段 | 目标 | 主要完成内容 | 主要证据 |
|---|---|---|---|
| 0 | 完整审计 | 识别 SCIP 版本、构建链路、MILP、默认 branching/node selector、真实数据和风险 | `docs/rl_branching_audit.md` |
| 1 | 可复现 baseline | 新增统一 CLI、结构化 JSON、五种 baseline、golden/default 回归和 validator | `docs/baseline_report.md` |
| 2 | 数据集 | 真实实例分组、合成生成器、CIP/metadata/manifest、split 隔离与 hash 验证 | `docs/dataset_report.md` |
| 3 | 自定义分支插件 | 实现 candidate-safe custom-random/custom-mostinf 和结构化 branch log | `docs/custom_branchrule_report.md` |
| 4 | BBMDP 环境 | 源码构建兼容 SCIP 8.0.4 的 PySCIPOpt/Ecole，定义状态、动作、奖励和 transition 测试 | `docs/bbmdp_environment_report.md` |
| 5 | Candidate MLP-DQN | 完成训练、验证、checkpoint、TorchScript 导出和 random/untrained/RL pilot | `docs/training_report.md` |
| 6 | C++ MLP 接入 | LibTorch 单次加载、候选集内推理、fallback、Python/C++ parity、端到端求解 | `docs/integration_report.md` |
| 7 | Bipartite GCNN-DQN | 二部图消息传递、PER/Double DQN、scalar/HL-Gauss、TorchScript 和 C++ 接入 | `docs/gcnn_report.md` |
| 8 | 正式实验 | controlled/production 双协议、430 次主实验、150 次消融、统计、绘图和最终报告 | `docs/FINAL_RL_BRANCHING_REPORT.md` |

## 6. Baseline 设计

### 6.1 方法

- `default`：不提升其他规则优先级，保持 SCIP 原有插件排序；当前实际最高生效规则为 reliability pseudocost。
- `relpscost`：显式选择 SCIP reliability pseudocost。
- `random`：SCIP 内置随机分支。
- `mostinf`：SCIP most-infeasible 分支。
- `strong`：strong branching，仅在小实例 `real_09` 上执行。
- 阶段 3 另有 `custom-random` 和 `custom-mostinf`，用于验证自定义 callback 接口，不作为主要 baseline。

除 default 外，指定 baseline 的 branching priority 提升到 `1,000,000`，确保目标规则实际生效。自定义 RL 规则也使用高优先级；fallback relpscost 的优先级低一档，只有 RL 返回 `DIDNOTRUN` 时接管。

### 6.2 统一控制变量

CLI 支持：

```text
--branching <method>
--seed <int>
--time-limit <seconds>
--node-limit <int>
--threads 1
--output-json <path>
--export-milp <path>
--build-only
```

单次 JSON 包含状态、目标值、primal/dual bound、gap、wall/presolve/solve time、节点数、LP iterations、primal-dual integral、首个可行解时间、变量/整数变量/约束数、实际 branching 和 node selector、可行性检查等字段。

### 6.3 阶段 1 结果

阶段 1 共 37 次运行：9 个真实实例，`default/relpscost/random/mostinf` 各一次 seed 0，时限 30 秒、单线程；`strong` 仅在 `real_09` 上运行。

| 方法 | optimal / 运行数 | 中位 wall time | 中位 nodes |
|---|---:|---:|---:|
| default | 3/9 | 30.68 s | 2 |
| relpscost | 3/9 | 30.68 s | 2 |
| random | 3/9 | 30.59 s | 4 |
| mostinf | 3/9 | 30.63 s | 12 |

`real_09` smoke 中 default 约 4.09 秒/9 nodes，random 约 2.21 秒/4 nodes，mostinf 约 2.19 秒/12 nodes，strong 约 10.88 秒/18 nodes。这个单实例结果只证明策略生效，不能据此宣称 random 普遍优于 default。

## 7. 数据集设置

### 7.1 最终真实实例划分

切分单位是完整原始 MILP 实例，绝不把同一实例的 B&B 状态拆到不同集合。

| split | 实例 | 规模 | variables | integer variables | constraints | 阶段 2 default baseline |
|---|---|---|---:|---:|---:|---|
| train | `real_06` / TESTA10 | medium | 20,682 | 2,708 | 48,140 | optimal，27.35 s，29 nodes |
| train | `real_07` / TESTA02 | medium | 38,100 | 3,036 | 93,965 | timeout |
| validation | `real_08` / TESTA03 | medium | 27,094 | 2,880 | 65,069 | optimal，28.83 s，12 nodes |
| test | `real_09` / TESTJK01 | small | 10,746 | 2,668 | 22,075 | optimal，4.09 s，9 nodes |
| transfer | `real_01` / F0001 | medium | 52,708 | 1,700 | 135,556 | timeout |
| transfer | `real_02` | large | 83,878 | 1,924 | 214,744 | timeout |
| transfer | `real_03` | medium | 41,390 | 3,792 | 100,803 | timeout |
| transfer | `real_04` | large/max | 326,502 | 5,168 | 863,691 | timeout，未进入 B&B |
| transfer | `real_05` | medium | 39,408 | 3,056 | 97,427 | timeout |

最终 MLP/GCNN 训练实际只使用 `real_06`，验证使用 `real_08`，`real_09` 和所有 transfer 实例不参与训练。`real_07` 虽在 manifest 的 train split 中，但正式报告中的 pilot 模型没有用它训练。这个细节对解释泛化不足非常重要。

### 7.2 合成数据

阶段 2 还创建了 15 个 deterministic synthetic 实例，用于验证生成、导出和 matched-scale split：train 9 个、validation 3 个、test 3 个，覆盖 small/medium/large。加上 9 个真实实例，manifest 共 24 个实例，均有 CIP 和 JSON metadata。

合成 small/medium/large 的典型规模分别约为 `522/160/1023`、`3626/512/8363`、`20377/1292/49675`（变量/整数变量/约束）。这些实例没有用于阶段 8 正式结论，汇报时不要将它们计入最终训练样本数量。

## 8. 软件和硬件环境

### 8.1 软件栈

| 组件 | 版本/设置 |
|---|---|
| OS 编译环境 | GCC 11.4，C++17 |
| SCIP | 8.0.4 |
| SoPlex | 6.0.4 |
| Python | 3.11.15 |
| PySCIPOpt | 4.3.0，从源码链接 SCIP 8.0.4 |
| Ecole | 0.8.1，从源码构建；为 Python 3.11 使用 pybind11 2.10.4 兼容补丁 |
| PyTorch | 2.5.1+cu121 |
| 数值栈 | NumPy 1.26.4、SciPy 1.13.1、Pandas 2.2.3 |
| C++ ABI | `_GLIBCXX_USE_CXX11_ABI=0`，与 PyTorch 包保持一致 |

项目没有降级原 SCIP，而是在 `artifacts/environment/phase4/scip804_prefix` 中建立与 SCIP 8.0.4 一致的 Python 训练依赖。C++ 原生 SCIP 与 Python/Ecole 使用同一 SCIP 主版本，减少训练与部署语义差异。

### 8.2 资源使用

历史节点可见 4 张 V100-SXM2 32GB、约 80 个逻辑 CPU 和 256 GiB 内存，但实际 MLP/GCNN 训练只需要单卡。训练的瓶颈是 SCIP 环境逐步推进和图特征提取，GPU 前向/反向计算很小：MLP pilot GPU 分配约 64 MiB，GCNN pilot 约 67 MiB。多卡数据并行不能有效加速单个环境轨迹；更合理的扩展方式是多 CPU worker 并行收集 episode，再用单 GPU learner 更新。

当前交接时的会话不一定分配 GPU，因此复现前应先检查 `nvidia-smi` 和 `torch.cuda.is_available()`。阶段 8 的历史结果已经包含真实 V100 CUDA 运行，不应因当前会话无卡而改写历史事实。

## 9. BBMDP 训练环境

### 9.1 MDP 定义

- 状态：当前 MILP 变量-约束二部图、候选变量局部特征，以及 incumbent、bounds、gap、depth、open nodes 等全局 B&B 特征。
- 动作：SCIP 当前 fractional LP branching candidates 中选择一个变量。
- 奖励：默认 `r_t = -(N_{t+1}-N_t)`，也支持 constant `-1`。
- 终止：optimal、open nodes empty、SCIP error 等；time/node limit 按 truncation 处理。
- 折扣：`gamma = 1`。
- timeout bootstrap：明确置零，不从无效终态继续 bootstrap。

动作使用 Ecole 的变量行索引表示，并通过 action set 做 mask。环境不会跨 transition 保存失效 SCIP pointer。

### 9.2 受控搜索协议

- DFS node selection。
- single thread、fixed seed。
- restart disabled。
- `separating/maxrounds = 0`，即非 root 分离关闭；root 轮次保留 SCIP 对应设置。
- 用于让训练更接近 BBMDP 中的树搜索决策过程。

Transition 测试覆盖：action 合法性、维度、奖励与节点增量、terminal bootstrap、candidate/index 映射、指针生命周期和 timeout 语义。阶段 4 共 43 个 transition 检查通过，并验证同 seed episode 可复现。

## 10. Candidate MLP-DQN

### 10.1 输入和网络

每个候选独立共享同一套网络，输入共 39 维：

- 19 维 Ecole variable features。
- 14 维确定性全局树特征，已删除 `solving_time`。
- 6 维航空变量类别 one-hot：`m/z/y/absf/f/other`。

网络结构：

```text
candidate feature (39)
  -> Linear(39, 128) -> ReLU
  -> Linear(128, 128) -> ReLU
  -> Linear(128, 1)
  -> masked argmax over current candidates
```

总参数量 21,761。特征 normalization 作为模型 buffer 保存；排序 tie 使用稳定的变量名/索引规则，保证复现。

### 10.2 学习设置

- Double DQN。
- 3-step return，`gamma=1`。
- hard target update every 250 gradient steps。
- replay buffer 10,000，batch size 32，minimum replay 32。
- 每个 environment step 最多 8 次 learner update。
- Smooth L1 loss、Adam `3e-4`、gradient clipping 10。
- epsilon 从 1.0 线性降到 0.05，约 4,000 steps。
- controlled DFS；单线程；restart off；non-root cuts off；每 episode 60 秒/200 nodes。

### 10.3 Pilot 结果

pilot 共 5,000 gradient steps、20 episodes、replay 最终约 735 transitions，训练约 1,278 秒。validation 最好 checkpoint 在 step 4,128，节点数从早期的 78/85 改善到 63，随后为 66，说明存在学习信号但样本有限。

固定 seeds 100/101/102 的 pilot 比较：random 2/3 solved、平均 capped nodes 130.33；untrained 2/3、137；RL 3/3、109.33。RL 相对 random 节点约少 16.1%，时间约少 1.7%，但 seed 101 从 random 100 nodes 退化到 RL 146 nodes。该结果只支持进入 C++ 集成，不能视为最终泛化结论。

## 11. Bipartite GCNN-DQN

### 11.1 图特征

- variable node：19 维 Ecole variable + 6 维航空变量类别，共 25 维。
- constraint node：14 维约束特征 + 6 维航空约束类别，共 20 维。
- edge：原系数、归一化系数、符号，共 3 维。
- global：14 维。

约束类别 one-hot 为 `flow/absolute/topology/selection/imbalance/other`。约束数值特征包括 normalized bias、objective cosine similarity、tightness、dual、age、lhs/rhs、activity、slack、equality/side indicators 等。

### 11.2 网络结构

```text
variable encoder:   25 -> 128 -> 64
constraint encoder: 20 -> 128 -> 64
edge encoder:        3 -> 128 -> 64
global encoder:     14 -> 128 -> 64

variable-to-constraint message:
  MLP([variable_embedding, edge_embedding]) -> 64
  mean aggregate by constraint using index_add
  row update MLP([row_embedding, aggregated_message]) -> 64

constraint-to-variable message:
  MLP([row_embedding, edge_embedding]) -> 64
  mean aggregate by variable using index_add
  variable update MLP([variable_embedding, aggregated_message]) -> 64

candidate Q head:
  [updated_variable_embedding, global_embedding] (128)
  -> 128 -> output bins
```

实现刻意不依赖 PyTorch Geometric，而使用基础 `index_add`，便于 TorchScript 和 C++ 部署。scalar Q 版本输出 1 维；HL-Gauss 输出 18 bins，`z in [-1,12]`、`sigma=0.75`，从 log2(-Q) 分布还原期望值。

### 11.3 学习设置与 pilot

- Double DQN、3-step、`gamma=1`。
- prioritized replay：`alpha=0.6`，`beta=0.4 -> 1.0`。
- replay 64、batch 2、minimum replay 8。
- 每环境步 4 次更新，Adam `3e-4`，clip 10。
- soft target update `tau=0.01`。
- epsilon `1.0 -> 0.05`，约 800 steps。

pilot 为 1,000 steps、3 episodes，约 595 秒。随机策略与 GCNN-RL 的平均 capped nodes 为 97 vs 81（约少 16.5%），solve time 34.30 vs 32.80 秒（约少 4.4%），两者都是 2/3 solved。正式阶段 8 没有维持这一优势。

## 12. 模型接入 SCIP 后的完整流程

### 12.1 初始化

1. 命令行解析 `--branching rl-mlp|rl-gcnn`、模型路径、device、fallback、最大深度、最少候选数和日志路径。
2. SCIP 创建问题并保留原变量、约束、目标、presolve、heuristics、cuts 和 node selector 设置。
3. 注册 RL `ObjBranchrule`，优先级高于 fallback。
4. solver 初始化时只加载一次 TorchScript 模型，切换 `eval`，执行 no-grad/warmup；不启动 Python 子进程。

### 12.2 每次 branching callback

```text
SCIP 到达需要分支的 LP 节点
  -> SCIPgetLPBranchCands 获取 fractional LP candidates
  -> 只取 npriolpcands 对应的当前合法候选
  -> 检查 depth/min-candidates gate
  -> 从当前 SCIP 状态提取变量/约束/边/全局特征
  -> 按 artifact normalization 做归一化
  -> TorchScript forward（no-grad）
  -> 检查输出维度和 NaN/Inf
  -> 仅在 candidate mask 内 argmax
  -> 再次验证选中变量属于当前 candidate set
  -> SCIPbranchVarVal 执行分支
  -> 写 branch CSV 和推理耗时统计
```

### 12.3 Fallback

以下情况返回 `DIDNOTRUN`，让 `relpscost` 或 default 插件接管：模型加载/推理异常、特征维度不匹配、NaN/Inf、无合法候选、超过 RL 最大深度、候选数低于阈值。模型不会在 callback 中重复读磁盘。

### 12.4 Parity 和端到端证据

- MLP 固定 observation 上 151 个候选的 Python/C++ CPU/CUDA Q values 和 argmax 一致。
- GCNN Python/C++ 最大 Q 误差约 `5.7e-6`，argmax 一致。
- `real_09` 端到端：default 9 nodes；MLP CPU 13 nodes，12 次 RL action，总 inference 约 0.0059 秒；GCNN 单次 pilot 8 nodes，但 CUDA 初始化/图推理开销明显。
- missing model 和 depth/min-candidate gate 均验证能 fallback 并完成求解。

## 13. 阶段 8 正式实验设计

### 13.1 两套协议

**controlled-bbmdp**：DFS、restart off、non-root cuts off、单线程、固定 seed，目标是尽量对齐训练环境。

**production-scip**：保留原 `scip_tree.cpp` 的 estimate node selector 和生产求解设置，只替换 branching variable selection；实验仍统一设为单线程，保证方法间可比。

### 13.2 实例、方法和预算

- 实例：validation `real_08`，test `real_09`，transfer `real_01` 至 `real_05`。
- seeds：0、1、2、3、4。
- methods：default、relpscost、random、mostinf、RL-MLP、RL-GCNN；strong 仅 `real_09`。
- 每次 time limit 30 秒、threads 1、两个外层 worker。
- 每套协议 215 次，两套共 430 次。
- 另有模型消融 50 次、深度消融 100 次；阶段 8 正式 C++ 运行总计 580 次。

### 13.3 指标和统计

原始字段包括 status、objective、bounds、gap、wall/presolve/solve time、nodes、LP iterations、PDI、first solution、branch decisions、RL inference total/mean/max、fallback、模型规模等。

聚合包括 solved rate、shifted geometric mean time/nodes、median、mean final gap、paired speedup、wins、average rank、bootstrap 95% CI 和 timeout PAR-2。图包括 training curve、cactus、performance profile、wall-time speedup scatter、node/time reduction、inference overhead、ID/transfer 和 size comparison。

## 14. 正式实验结果解释

### 14.1 controlled-bbmdp

| 方法 | solved / 35 | shifted gmean wall | mean PAR-2 | 相对 default speedup |
|---|---:|---:|---:|---:|
| default | 8 | 21.327 s | 48.276 s | 1.000 |
| random | 13 | 20.539 s | 43.333 s | 1.143（1.039, 1.278） |
| mostinf | 12 | 20.545 s | 44.174 s | 结果见 CSV |
| RL-MLP | 8 | 21.841 s | 48.333 s | 0.978（0.941, 1.014） |
| RL-GCNN | 9 | 24.407 s | 47.809 s | 0.906（0.812, 0.998） |

相对 random，MLP/GCNN time speedup 分别为 0.856 和 0.793，均更慢；但 jointly solved 子集的节点约少 4%。这说明奖励可能学到了一点树规模信号，却没有转化为 wall-clock 优势。

### 14.2 production-scip 分 split

- test `real_09`：default、MLP、GCNN 均 5/5 solved；MLP 平均约 2.06 秒，与 default 接近；GCNN 约 4.06 秒，推理占 solving time 约 25.4%。
- validation `real_08`：default 4/5、MLP 4/5、GCNN 5/5；GCNN 推理占约 5.69%。
- medium transfer：default 0/15、MLP 1/15、GCNN 3/15。GCNN 有 solved 数的正面信号，但节点和 wall time 没有稳定改善。
- large transfer：所有方法均未在时限内求解。

生产协议中，所有运行摊薄后的 inference 占 solving time：MLP 约 0.035%，GCNN 约 3.14%；只看实际发生 RL branching 的活跃运行，分别约 0.10% 和 9.39%。GCNN 超过了预设的 5% 成本目标。

### 14.3 最大实例

`real_04` 有 326,502 个变量和 863,691 个约束。所有方法五个 seed 都在 presolve 阶段达到时限，平均 presolve 约 27.4 秒，节点数和 branch decision 都为 0，也没有可比较的 bounds。GCNN 即使 callback 没执行，模型/CUDA 初始化仍使平均 wall time 从 default 约 34.72 秒升到 36.68 秒。

后续若要真正评估最大实例，应把模型构建、presolve 和 B&B 预算拆开，或导出共同的 presolved CIP 后再比较 branching。

## 15. 消融实验

### 15.1 模型消融

所有变体使用 `real_06`、单训练 seed、1,000 gradient steps，并在 `real_08/real_09` 各五个 seed 上 C++ 评测。

| 变体 | solved | mean PAR-2 | shifted gmean wall |
|---|---:|---:|---:|
| scalar 3-step | 10/10 | 14.391 s | 10.165 s |
| scalar 1-step | 10/10 | 14.266 s | 10.079 s |
| HL-Gauss 3-step | 8/10 | 20.634 s | 10.551 s |
| no aviation categories | 10/10 | 14.289 s | 10.085 s |
| no global features | 10/10 | 14.254 s | 10.076 s |

在当前预算下，3-step 没有优于 1-step，HL-Gauss 退化；去掉航空类别或全局特征只有约 1% 变化。由于每个变体只有一个训练 seed，这些小差异不具备统计说服力，不能据此永久删除特征。

### 15.2 浅层 RL + 深层 relpscost

| max RL depth | solved / 20 | mean PAR-2 | inference 占比 | depth fallback |
|---|---:|---:|---:|---:|
| 5 | 8 | 40.307 s | 2.56% | 22 |
| 10 | 8 | 40.242 s | 2.61% | 17 |
| 20 | 10 | 37.107 s | 2.71% | 11 |
| 50 | 10 | 37.119 s | 2.81% | 9 |
| unlimited | 11 | 35.557 s | 2.84% | 0 |

浅层限制没有改善总体结果：D=5/10 损失 solved rate，D=20/50 也不如 unlimited。将 wall time 减去 inference total 的模拟仍未改变总体排序，说明 GCNN 的问题不仅是 forward，还包括动态图提取、初始化和策略引起的搜索路径变化。

## 16. 正确性与失败诊断

### 16.1 已通过

- 430 个主实验任务键完整唯一，进程返回码全部为 0。
- 132 次 optimal、298 次 time limit。
- jointly solved 目标值在 `1e-8` 相对容差内一致。
- 所有 optimal solution 通过原项目可行性检查。
- 2,454 次主实验 action、679 次模型消融 action、327 次深度消融 action 全部合法。
- 无崩溃、非法 action、NaN/Inf 或意外 fallback。
- Python 单元测试 18/18 通过；最终 Make/C++ runner 构建通过。
- checkpoint 可重载，loss 有限，eager/TorchScript/C++ argmax 一致。

### 16.2 未达标原因

1. 真实模型训练实际只有 `real_06`，训练分布过窄。
2. GCNN 1,000-step pilot 和单训练 seed 对高维图策略明显不足。
3. SCIP relpscost 自带 pseudocost/strong-branching 历史强化，RL 特征没有完全复现其优势。
4. 节点增量 reward 与真实 wall-clock、LP 难度、模型推理成本不完全一致。
5. DFS 训练与 production estimate node selector 存在状态分布偏移。
6. GCNN 每次 callback 重建动态图，活跃运行推理占比高。
7. 多个 transfer 实例在 presolve/root 阶段耗尽预算，branching 可影响空间小。
8. 现有航空类别和全局特征尚未在多训练 seed 下显示稳定贡献。

## 17. 推荐后续路线

优先级建议：

1. 增加真实同分布训练实例，而不是依赖随意扰动目标的合成数据。
2. 用 SCIP relpscost/strong branching 生成 expert action，先做 imitation learning warm start。
3. 在 production node selector 参数分布上 fine-tune Candidate MLP。
4. 补充可靠 pseudocost、LP solve cost、branch history 等特征，并把 inference/wall-time penalty 纳入目标。
5. 对最大实例单独设计 presolved-state 协议，让实验真正进入 B&B。
6. 暂缓扩大 GCNN，先完成静态图缓存和增量动态特征更新；否则更多 GPU 不能解决主要瓶颈。

## 18. 关键复现命令

```bash
# 构建
conda run -n rl4scip make

# 阶段 1 baseline
conda run -n rl4scip python scripts/run_baselines.py \
  --binary build/scip_tree \
  --instances 1,2,3,4,5,6,7,8,9 \
  --methods default,relpscost,random,mostinf \
  --strong-instances 9 --seeds 0 --time-limit 30 --threads 1
conda run -n rl4scip python scripts/validate_baselines.py

# 数据集验证
conda run -n rl4scip python scripts/validate_dataset.py \
  --config configs/dataset/phase2.json \
  --manifest data/instances/manifest.csv \
  --scip-binary /home/duweiyue25/SCIP/scipoptsuite-8.0.4/build/bin/scip

# MLP / GCNN 训练
CUDA_VISIBLE_DEVICES=0 conda run -n rl4scip env PYTHONPATH=python \
  python scripts/train_candidate_mlp.py --config configs/rl/pilot.yaml
CUDA_VISIBLE_DEVICES=0 conda run -n rl4scip env PYTHONPATH=python \
  python scripts/train_gcnn.py --config configs/rl/gcnn_pilot.yaml

# 单次 C++ RL 求解示例
conda run -n rl4scip ./build/scip_tree \
  --instance-id 9 --branching rl-mlp \
  --rl-model artifacts/models/mlp/best_model_scripted.pt \
  --rl-device cpu --rl-fallback relpscost \
  --seed 0 --time-limit 30 --threads 1 \
  --output-json results/example_rl_mlp.json

# 阶段 8 分析和完整性验证（原始运行不必重复才能制作汇报）
conda run -n rl4scip env PYTHONPATH=python \
  python scripts/analyze_final_results.py \
  --input results/final/raw_results.csv --output-dir results/final
conda run -n rl4scip env PYTHONPATH=python \
  python scripts/validate_final_results.py \
  --input results/final/raw_results.csv \
  --expected-runs 430 \
  --output results/final/validation.json
```

完整参数以 `README.md` 和相应 config 为准。正式实验已经完成并保存，不建议为了组会临时重跑 580 次任务。

## 19. 产物索引

- 审计：`docs/rl_branching_audit.md`
- baseline：`docs/baseline_report.md`、`results/baseline/`
- 数据集：`docs/dataset_report.md`、`data/instances/manifest.csv`
- 自定义规则：`docs/custom_branchrule_report.md`
- BBMDP 环境：`docs/bbmdp_environment_report.md`
- MLP 训练：`docs/training_report.md`、`artifacts/models/mlp/`
- C++ MLP 接入：`docs/integration_report.md`
- GCNN：`docs/gcnn_report.md`、`artifacts/models/gcnn/`
- 正式结果：`docs/FINAL_RL_BRANCHING_REPORT.md`
- 430 次原始主实验：`results/final/raw_results.csv`
- 汇总和比较：`results/final/summary.csv`、`paired_comparisons.csv`、`rl_vs_random.csv`
- 消融：`results/final/ablation_summary.csv`、`ablation_contrasts.csv`
- 图：`results/final/figures/`
- 完整性验证：`results/final/validation.json`

## 20. 建议的组会汇报结构

1. 问题与动机：为什么 branching variable selection 可能影响航空布线 MILP。
2. 原模型和改造边界：MILP 不变，只替换分支决策。
3. 工程路线：阶段 0-8 一页总览。
4. Baseline 与双协议：受控 BBMDP vs production SCIP。
5. 数据集：真实实例划分、规模、无泄漏、训练数据不足。
6. BBMDP：状态、动作、奖励、终止和合法性约束。
7. MLP：39 维候选特征、DQN 设置、pilot 信号。
8. GCNN：二部图、两轮消息传递、为何部署成本更高。
9. SCIP 接入：callback 完整流程、candidate mask、fallback、parity。
10. 正式结果：production 主表和 controlled 对照。
11. transfer/最大实例/消融：哪些有信号，哪些不能下结论。
12. 结论与下一步：工程闭环成功，当前部署价值不足，优先 imitation + 更多真实数据 + MLP fine-tune。

## 21. 可直接交给 ChatGPT 的提示词

```text
你是一名熟悉 MILP、SCIP Branch-and-Bound、强化学习和科研汇报的研究助手。
请严格基于我提供的《航空布线 RL Branching 项目工作交接文档》整理今晚组会材料，
不要虚构实验、不要把 pilot 当作正式结果，也不要声称 RL 已经超过 SCIP-default。

请输出：
1. 一份 12-15 页的中文 PPT 结构，每页含标题、3-6 个要点、建议使用的项目图表；
2. 一份 12-15 分钟中文讲稿，说明 baseline 设计、数据集 split、训练环境、
   BBMDP 定义、MLP/GCNN 完整结构、接入 SCIP 后的执行流程、双协议实验和结果；
3. 对正式结果做科研口径分析，区分统计显著性、工程正确性和部署价值；
4. 列出老师最可能追问的 15 个问题及简洁回答；
5. 明确项目的正面贡献、失败原因、局限和下一步优先级；
6. 对所有数字注明来自阶段 8 正式实验还是阶段 5/7 pilot。

必须保留以下关键事实：
- MILP 变量、约束和目标没有改变；未指定 RL 时 default 行为不变；
- 最终正式训练使用 real_06，validation real_08，test real_09，transfer real_01-05；
- 主实验为 2 个协议、430 次运行，另有 150 次消融；
- MLP wall time 相对 default 只改善约 0.13%，GCNN 恶化约 12.4%；
- RL 没有明显优于 random；所有 RL action 均合法；
- 最大 real_04 在时限内没有进入 B&B，因此无法评估 branching；
- 当前不建议部署，后续优先更多真实数据、expert imitation 和 production MLP fine-tune。
```

## 22. 最终十问速答

1. RL 是否减少节点？没有稳定减少；生产 jointly solved 中反而增加。
2. RL 是否减少 wall time？MLP 基本持平，GCNN 明显变慢。
3. 推理开销？活跃生产运行约 MLP 0.10%、GCNN 9.39%。
4. 未见实例是否有效？可正确求解，但没有稳定性能泛化。
5. 能否迁移最大实例？本协议无法判断，因为未进入 B&B。
6. MLP 和 GCNN 哪个更合适？当前 MLP 更适合继续迭代。
7. DFS 训练能否迁移默认 node selector？功能可迁移，性能尚不可靠。
8. 是否有实际部署价值？当前没有达到预设门槛。
9. 主要瓶颈？真实训练数据少、reward 错位、SCIP 内置历史信息强、GCNN 动态图成本和 presolve 占比。
10. 下一步最值得做什么？更多真实训练实例 + relpscost/strong imitation warm start + production MLP fine-tune。
