# DSU-Prime-GCNN 核心改造与 real_04 实验实施方案

> 目标：保留“树结构应从已有邻域继续生长”的 Prim 核心思想，将其改造成单一、可训练、可部署的 RL-GCNN 方法，并尽快完成 real_04 的可信实验。
>
> 基线仓库：cable_harness_prime_gcnn，审查提交 26298486e341b32b115fb2f6717bdb1e0f54697e。
>
> 固定环境：SCIP 8.0.4、SoPlex 6.0.4、PySCIPOpt 4.3.0、Ecole 0.8.1、Python 3.11、PyTorch/LibTorch 2.5.1+cu121。本轮不迁移 SCIP 9.2.2。

---

## 1. 最终方法

最终只保留一个学习策略：**DSU-Prime-GCNN**。

~~~text
原始线束 MILP
  → SCIP 预处理并求解当前节点 LP
  → 取得合法 fractional branching candidates
  → 构造变量—约束二部图
  → 用并查集提取当前树的连通分量与邻域关系
  → GCNN 为每个合法候选输出 Q 值
  → 在候选集合内选择 Q 最大的变量
  → SCIP 执行分支、剪枝并继续精确求解
~~~

Prime-GCNN 不直接生成最终线束树，也不替代 SCIP 的 B&B。它只替换 branching variable selection：决定当前优先对哪个小数变量分支。MILP 的变量、约束、目标、可行性、上下界、剪枝和最优性证明全部保留。

最终主线删除：

- Candidate MLP 和 rl-mlp；
- controlled-bbmdp / production-scip 双协议；
- Q + λ·PrimScore 固定解码偏置；
- none/z/root_z/prim/topology 多种 bias mode；
- Prim feature、λ、depth gate 等兼容开关；
- C0/C1/SB teacher 的生产入口。

SCIP 内置的 project-default 和 random 只作为实验基线，不属于第二套学习模型。

---

## 2. MILP 与 RL 的边界

原始变量保持不变：

| 变量 | 类型 | 含义 |
|---|---|---|
| m_k_p | binary | 需求 k 分配到第 p 层 |
| z_i_j_p | binary | 第 p 层选择有向边 (i,j) |
| f_i_j_k | continuous | 需求流 |
| absf_i_j_k | continuous | 流绝对值线性化 |
| y_i_p | continuous | MTZ 风格拓扑顺序 |

SCIP 在需要分支时调用 SCIPgetLPBranchCands，模型只能从返回的合法小数候选集合中选择动作。没有可处理的 LP candidates 时允许 SCIP_DIDNOTRUN；模型加载、schema、特征、输出或候选映射错误时，实验必须标记 invalid_policy_run，不能静默回退到 relpscost 后继续记录为 Prime-GCNN。

---

## 3. DSU-Prim 结构特征

### 3.1 连通分量

DSU = Disjoint-Set Union，并查集，用于维护每个层 p 上当前已经确定的无向连通分量。

对每个层 p：

1. 遍历本层 z_i_j_p；
2. 只有 SCIPvarGetLbLocal(z) > 0.5，即当前 B&B 节点已确定 z=1，才合并端点 i、j；
3. SCIPvarGetLPSol(z) 不用于合并，它继续作为 Ecole 的连续 LP 特征；
4. 连通性按无向边处理，变量的方向仍由变量名和原始约束表达。

Python/Ecole 端必须从 PySCIPOpt 读取真实 local lower bound，不能用 solution value 和 is-at-lower-bound 近似。getVars(transformed=True) 只在变量数量变化时调用，移植旧仓库已经验证的缓存修复。

### 3.2 六维特征

对候选边 z_i_j_p 生成：

| 特征 | 定义 |
|---|---|
| prim_frontier | 一个端点位于已生长分量，另一个端点尚未进入 |
| prim_merge | 两端属于两个不同的已生长分量 |
| prim_cycle | 两端属于同一个已生长分量 |
| prim_unseen | 两端均未进入任何已生长分量 |
| prim_src_component_ratio | 起点分量规模 / 本层已生长节点数；未生长为 0 |
| prim_dst_component_ratio | 终点分量规模 / 本层已生长节点数；未生长为 0 |

前四个状态互斥。对 m/y/f/absf/other，六维全部置零，由原始变量特征和变量类别处理。

根节点还没有固定 z=1 的边时，大部分 z 变量为 prim_unseen=1；GCNN 先选择初始分支。进入 z=1 子节点后，frontier、merge、cycle 信息出现，形成“从已有邻域继续生长”的 Prim 路径。

prim_cycle 只作为风险信号，不做 hard mask。近似结构判断不能代替数学约束。

---

## 4. 相邻关系与图输入

模型同时使用两种相邻关系：

1. **约束邻接**：两条 z 边变量共同出现在同一 MILP 约束中时，通过“变量 → 约束 → 变量”二跳传播交换信息；
2. **拓扑邻接**：DSU 根据边是否共享端点、是否属于同一连通分量，显式产生 frontier/merge/cycle 特征。

因此不再额外维护边—边图。MILP 耦合由变量—约束二部图表达，原拓扑关系由 DSU 特征表达。

| 对象 | 数值特征 | 类别特征 | 编码器实际输入 |
|---|---:|---:|---:|
| 变量节点 | Ecole 19 + DSU-Prim 6 = 25 | 变量类别 6 | 31 |
| 约束节点 | 扩展行特征 14 | 约束类别 6 | 20 |
| 边 | 系数、归一化系数、符号 = 3 | 0 | 3 |
| 全局状态 | depth、nodes、bounds、gap、LP iterations、incumbent 等 14 | 0 | 14 |

变量类别固定为 m/z/y/absf/f/other；约束类别固定为 flow/absolute/topology/selection/imbalance/other。

25 和 31 不冲突：模型文件接收的 variable_features tensor 是 25 维，进入变量编码器前再拼接 6 维类别 one-hot，编码器输入为 31 维。

---

## 5. Prime-GCNN 模型

部署时只有一个 Prime-GCNN。训练时 online network 和 target network 结构相同，target 只是稳定 DQN 目标的延迟副本，不是第二种策略。

### 5.1 编码器

| 模块 | 网络 | 输出 |
|---|---|---:|
| 变量编码器 | Linear(31,128) → ReLU → Linear(128,64) → ReLU | 64 |
| 约束编码器 | Linear(20,128) → ReLU → Linear(128,64) → ReLU | 64 |
| 边编码器 | Linear(3,128) → ReLU → Linear(128,64) → ReLU | 64 |
| 全局编码器 | Linear(14,128) → ReLU → Linear(128,64) → ReLU | 64 |

所有数值特征按训练期冻结的 mean/std 归一化，类别 one-hot 不归一化。

### 5.2 一轮消息传播

初始编码：

\[
h_v=\mathrm{MLP}_v([x_v,c_v]),\qquad
h_r=\mathrm{MLP}_r([x_r,c_r]),\qquad
h_e=\mathrm{MLP}_e(x_e)
\]

变量向约束发送消息并更新约束：

\[
m_r=\operatorname{Mean}_{v\in N(r)}
\mathrm{MLP}_{v\rightarrow r}([h_v,h_e])
\]

\[
h'_r=\mathrm{MLP}^{update}_r([h_r,m_r])
\]

约束向变量发送消息并更新变量：

\[
m_v=\operatorname{Mean}_{r\in N(v)}
\mathrm{MLP}_{r\rightarrow v}([h'_r,h_e])
\]

\[
h'_v=\mathrm{MLP}^{update}_v([h_v,m_v])
\]

只保留一轮 variable → row → variable。显式 DSU 特征已经提供连通分量信息，本轮不增加 GNN 层数，避免放大 real_04 的推理成本。

### 5.3 Q Head

只对 SCIP 当前候选变量打分：

~~~text
候选变量表示 64 + 全局表示 64
→ Linear(128,128)
→ ReLU
→ Linear(128,1)
~~~

\[
Q(s,a)=\mathrm{MLP}_Q([h'_a,h_g])
\]

奖励表示负求解代价，因此 Q 越大代表预计后续代价越低：

\[
a^*=\arg\max_{a\in A(s)}Q(s,a)
\]

推理端不叠加 λ 或手工 PrimScore。

---

## 6. DQN 训练

每个 episode：

~~~text
加载训练 CIP
  → SCIP 运行到 branching decision
  → 提取图状态 s_t 和候选集合 A(s_t)
  → ε-greedy 选择候选 a_t
  → SCIP 执行分支并运行到下一决策点
  → 得到 r_t、s_{t+1}、A(s_{t+1})
  → 写入 replay buffer
  → 采样 batch 更新 online network
~~~

经验项为：

\[
(s_t,a_t,r_t,s_{t+1},A(s_{t+1}),terminated,truncated)
\]

### 6.1 奖励

\[
r_t=-\Delta N_t-10^{-4}\Delta LP_t
\]

- ΔN：两次决策间增加的 B&B 节点数；
- ΔLP：两次决策间增加的 LP iterations。

time/node limit 是人工截断，需要 bootstrap；optimal、infeasible、无开放节点才是真正终止。

### 6.2 固定超参数

| 参数 | 值 |
|---|---:|
| Double DQN | 开启 |
| n-step | 3 |
| gamma | 1.0 |
| loss | Smooth L1 |
| batch size | 64 |
| replay capacity | 2048 |
| min replay size | 128 |
| learning rate | 3e-4 |
| updates per env step | 8 |
| target tau | 0.01 |
| PER alpha | 0.6 |
| PER beta | 0.4 → 1.0 |
| gradient clip | 10.0 |
| epsilon | 1.0 → 0.05 |
| loss mode | scalar，不启用 HL-Gauss |

---

## 7. 两阶段训练数据方案

real_06+real_07 只够验证流程。最终采用“多个中等实例预训练 → 加入 real_02 大图截断轨迹”的折中方案。

### 7.1 数据划分

~~~text
阶段 A 训练：real_01 / real_03 / real_05 / real_06 / real_07
阶段 B 训练：上述五个实例 + real_02 截断轨迹
验证：       real_08
额外诊断：   real_09
最终测试：   real_04
~~~

real_04 不参与 normalization 调整、训练、checkpoint 选择、奖励选择和特征调参。

### 7.2 阶段 A：多中等实例预训练

~~~text
gradient steps: 0–2399，共 2400 steps
instances: real_01 / 03 / 05 / 06 / 07
instance sampling: 每个实例 20%
episode time limit: 60s
episode node limit: 100
~~~

按实例均衡采样，防止某个容易产生大量 branching states 的实例支配 replay buffer。

### 7.3 阶段 B：加入 real_02 截断轨迹

~~~text
gradient steps: 2400–2999，共 600 steps
sampling:
  real_02: 30%
  real_01 / 03 / 05 / 06 / 07: 合计 70%，每个 14%

real_02 episode time limit: 300s
real_02 episode node limit: 100
中等实例 time limit: 60s
中等实例 node limit: 100
~~~

real_02 不要求完整求解。它用于让模型见到大图上的候选规模、LP 特征、图稀疏性和 Prime 分量分布。阶段 B 继续混合中等实例，避免只适应大图。

### 7.4 Normalization

normalization 不能只看中等实例后永久冻结。训练开始前收集至少 12 个 warmup states：

~~~text
real_01 / 02 / 03 / 05 / 06 / 07
每个实例至少 2 个状态
~~~

之后冻结 mean/std。四个 seed 使用同一 schema，但各自保存 normalization 和模型，不相互覆盖。

### 7.5 验证和选模

每 250 gradient steps 在 real_08 固定 seeds 100/101/102 上验证。best checkpoint 的排序为：

~~~text
solved count 更高
→ mean primal-dual integral 更低
→ mean final gap 更低
→ PAR-2 更低
→ mean LP iterations 更低
→ mean nodes 更低
~~~

禁止再按最小节点数单独选 best。

---

## 8. 四卡训练与推理

四卡训练不是把一个模型拆到四张卡，而是四个独立 seed：

~~~text
GPU 0 → training seed 0
GPU 1 → training seed 1
GPU 2 → training seed 2
GPU 3 → training seed 3
~~~

每个进程拥有独立的 SCIP/Ecole environment、replay buffer、online/target network、输出目录和 manifest：

~~~text
artifacts/models/prime_gcnn_dsu/seed0/
artifacts/models/prime_gcnn_dsu/seed1/
artifacts/models/prime_gcnn_dsu/seed2/
artifacts/models/prime_gcnn_dsu/seed3/
~~~

现有 Phase-B seed0 的记录是 3000 steps、71 episodes、5871.37s，约 1.63 小时。扩大实例并加入 real_02 后会更慢；四卡并行避免四个 seed 串行，但不会令单个 seed 加速四倍。

最终只根据 real_08 指标选择一个部署模型，禁止根据 real_04 选 seed。

单次 SCIP 推理仍为一个进程、一个模型、一个 GPU。多 GPU 只能并行独立实验。real_04 是大实例；同一服务器四路并行会争抢 CPU cache、内存带宽和 RAM，因此正式 wall/PDI 对比默认 workers=1。只有任务拥有隔离 CPU/内存或位于不同机器时，才允许四路正式并行。

---

## 9. Python/C++ 一致性

唯一 schema 固定为：

~~~text
variable numeric = 25
variable categories = 6
row numeric = 14
row categories = 6
edge = 3
global = 14
~~~

删除运行时 19/25 维选择。模型目录必须包含 manifest.json，至少记录：

~~~json
{
  \"schema_version\": 2,
  \"method\": \"dsu-prime-gcnn\",
  \"variable_feature_dim\": 25,
  \"model_sha256\": \"...\",
  \"schema_sha256\": \"...\",
  \"normalization_sha256\": \"...\",
  \"scip_version\": \"8.0.4\",
  \"ecole_version\": \"0.8.1\",
  \"torch_version\": \"2.5.1+cu121\",
  \"training_commit\": \"...\"
}
~~~

C++ 启动时校验 manifest、文件 SHA 和维度；第一处合法 branching callback 再 lazy-load TorchScript 和 warmup。

必须在同一个 SCIP branching state 上比较：

1. transformed variable order；
2. candidate names/indices；
3. local lower bounds；
4. 六维 DSU-Prim 特征；
5. 完整 25 维变量矩阵；
6. row/edge/global tensors；
7. candidate Q 向量和 argmax。

验收门槛：

~~~text
feature max_abs_error <= 1e-6
Q max_abs_error <= 1e-5
candidate order identical
argmax identical
~~~

只把同一 NPZ 分别送给 Python/C++ forward，不能替代端到端特征 parity。

---

## 10. 最小代码改造

本轮先修改影响方法和实验正确性的主链，不先清理全部历史文件：

| 文件 | 修改 |
|---|---|
| python/rl_branching/prim_bias.py | 替换为 DSU 六维特征 |
| python/rl_branching/observation.py | 缓存 transformed vars；读取真实 local bounds |
| python/rl_branching/graph_features.py | 固定唯一 25 维 schema |
| python/rl_branching/gcnn_config.py | 增加两阶段课程和实例采样 |
| python/rl_branching/gcnn_trainer.py | 两阶段 sampler、hybrid reward、截断 bootstrap、状态感知选模 |
| src/rl/prim_bias.cpp/.hpp | 实现相同 DSU |
| src/rl/scip_graph_feature_extractor.cpp/.hpp | 固定追加六维特征 |
| src/rl/gcnn_model_runner.cpp/.hpp | manifest 校验和 lazy load |
| src/rl/rl_gcnn_branchrule.cpp/.hpp | 删除多模式；错误 fail-fast |
| code/scip_tree.cpp | 正式入口只保留单协议和 rl-gcnn |
| scripts/run_final_experiments.py | 单协议、hash-aware resume、结果完整性检查 |

MLP、C0/C1 和历史脚本先从构建目标和正式配置中排除，real_04 实验结束后再迁移到历史 tag。

---

## 11. real_04 最快实验方案

### 11.1 固定实例和版本

~~~text
data/instances/transfer/real_04.cip
SHA256 = 8aa29d655d77df8b6b169dbcd25801dd770af9aab405d2f91d40c4524a2be711
~~~

本轮继续使用 SCIP 8.0.4。迁移 SCIP 9.2.2 会同时改变 Ecole ABI、SCIP 默认行为、模型分布和 baseline，不能与当前 real_04 目标混合。

### 11.2 R0：模型与部署门禁

先在 real_08 运行 C++ 集成测试：

~~~text
time limit: 300s
seed: 100
threads: 1
device: cuda
~~~

通过条件：

~~~text
model loaded = true
variable feature dim = 25
rl_actions > 0（若进入 B&B）
unexpected fallback = 0
selected_is_candidate = true
all Q finite
Python/C++ parity passed
~~~

### 11.3 R1：real_04 600 秒 smoke

~~~text
method: dsu-prime-gcnn
seed: 0
time limit: 600s
node limit: -1
threads: 1
workers: 1
device: cuda
branch log: enabled
~~~

通过条件：

~~~text
status ∈ {optimal, time_limit}
rl_actions > 0
unexpected fallback = 0
model/schema hashes match
selected_is_candidate = true
无 OOM、NaN、异常
~~~

计算：

\[
overhead\_ratio=
\frac{T_{graph\_extract}+T_{inference}+T_{model\_load}}{T_{solve}}
\]

若 overhead_ratio > 10% 或完整图 OOM，再实现候选精确二跳子图：

~~~text
候选变量
  → 候选相邻的全部约束行
  → 这些约束连接的全部变量
~~~

当前网络只有一轮消息传播；保留完整行邻居和相同 mean degree 时，候选 Q 应保持不变。优化前必须验证 full-graph/subgraph 的 Q max_abs_error <= 1e-5 且 argmax 一致。

### 11.4 R2：最快三 seed

Prime-GCNN 先运行：

~~~text
instance: real_04
method: dsu-prime-gcnn
seeds: 0,1,2
time limit: 3600s
node limit: -1
threads: 1
workers: 1
device: cuda
~~~

旧仓库已有 production-scip、SCIP 8.0.4、threads=1、seeds 0/1/2 的 project-default/random 一小时结果。只有以下内容全部一致才允许复用：

- real_04.cip SHA；
- SCIP/SoPlex 版本；
- solver 参数；
- 数据构建路径；
- 硬件和资源隔离口径；
- baseline 非 RL 路径代码哈希。

任意一项不同，重新运行：

~~~text
project-default × seeds 0,1,2
random          × seeds 0,1,2
dsu-prime-gcnn  × seeds 0,1,2
= 9 runs × 3600s
~~~

同一服务器正式比较默认串行；多台相同且隔离的机器可按 seed 并行。

### 11.5 R3：补齐五 seed

三 seed 通过后补 seeds 3、4：

~~~text
project-default / random / dsu-prime-gcnn
× seeds 0,1,2,3,4
× real_04
= 15 runs
~~~

单实例五 seed 只能说明 real_04 上的稳定性，不能外推为所有线束实例的普遍提升。

---

## 12. real_04 指标与门禁

| 优先级 | 指标 | 解释 |
|---|---|---|
| 主指标 | final gap @ 3600s | 固定预算下最终解与界的距离 |
| 主指标 | primal-dual integral | 整个求解过程的上下界质量，越低越好 |
| 主指标 | first solution time | 首个可行解出现时间 |
| 主指标 | solved/optimal | 是否在预算内完成 |
| 次指标 | primal/dual bound | 判断改善来自可行解还是下界 |
| 次指标 | LP iterations | 节点内部计算量 |
| 次指标 | nodes | 搜索树规模；不能单独作为结论 |
| 开销 | extract/inference/load | 模型实际额外成本 |

最低有效性门槛：

~~~text
所有运行 rl_actions > 0
unexpected fallback = 0
model/schema/hash 一致
无非法 candidate
无 NaN/OOM/crash
固定预算下 PDI 或 final gap 相对 project-default 稳定改善
GCNN 总开销 <= solving time 的 10%
~~~

已有 real_04 一小时结果证明“节点更少”可能伴随更差 final gap，因此禁止以节点数作为唯一胜负标准。

---

## 13. 配置和运行命令

建议新增：

~~~text
configs/rl/prime_gcnn_dsu_seed0.yaml
configs/rl/prime_gcnn_dsu_seed1.yaml
configs/rl/prime_gcnn_dsu_seed2.yaml
configs/rl/prime_gcnn_dsu_seed3.yaml
configs/experiments/real04_prime_gcnn_smoke.json
configs/experiments/real04_prime_gcnn_3seed.json
configs/experiments/real04_prime_gcnn_5seed.json
~~~

四卡训练：

~~~bash
cd /data/hanchengcheng/hcc_1/du/cable_harness_prim_gcnn
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_prime_gcnn_dsu_train_4gpu.sh
~~~

real_04 smoke：

~~~bash
PYTHONPATH=python python scripts/run_final_experiments.py \
  --config configs/experiments/real04_prime_gcnn_smoke.json \
  --workers 1
~~~

三 seed：

~~~bash
PYTHONPATH=python python scripts/run_final_experiments.py \
  --config configs/experiments/real04_prime_gcnn_3seed.json \
  --workers 1 \
  --resume
~~~

resume 只有在 config/model/schema/instance SHA 全部匹配且上次状态完整时才允许跳过；失败 JSON、部分输出或不同模型结果必须重跑。

---

## 14. 最快执行顺序

~~~text
P0  修 DSU 特征、25维契约、getVars 崩溃、单协议和 fail-fast
  ↓
P1  Python/C++ 端到端特征与 Q parity
  ↓
P2  4 GPU 并行训练 seeds 0–3
     阶段A：01/03/05/06/07，2400 steps
     阶段B：加入02截断轨迹，600 steps
  ↓
P3  仅按 real_08 选择部署 seed
  ↓
P4  real_04 seed0，600s smoke
  ↓
P5  real_04 seeds0–2，3600s
  ↓
P6  分析 gap/PDI/首解/开销，通过后补 seeds3–4
  ↓
P7  实验完成后再清理 MLP、双协议、C0/C1 和历史产物
~~~

真正的门禁不是训练 loss 是否下降，而是：部署端确实使用同一份 25 维 DSU-Prime schema，每个动作都来自 SCIP 合法候选，没有静默 fallback，并且 real_04 的 fixed-budget gap/PDI 改善能够覆盖 GCNN 开销。
