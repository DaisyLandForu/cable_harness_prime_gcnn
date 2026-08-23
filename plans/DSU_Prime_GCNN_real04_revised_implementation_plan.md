# DSU-Prime-GCNN 修订版完整方案与 real_04 最快实验 Runbook

> 文档性质：最终设计、实施清单和实验口径。本文中的新增文件名与命令是目标接口；在相应代码完成并通过门禁前，不应把它们当成仓库已经具备的功能。
>
> 方法目标：保留 Prime-GCNN 的核心思想——因为目标结构是一棵树，所以让 branching policy 像 Prim 一样优先判断当前树邻域中的边——但只保留一个可训练、可部署的 RL-GCNN 主线。
>
> 第一目标：尽快得到 real_04 上口径可信的一小时结果。任何不直接影响该目标的多协议、MLP、固定偏置、teacher 和大规模重构均后置。
>
> 固定软件栈：SCIP 8.0.4、SoPlex 6.0.4、PySCIPOpt 4.3.0、Ecole 0.8.1、Python 3.11、PyTorch/LibTorch 2.5.1+cu121。本轮不升级 SCIP。
>
> 审查基准：cable_harness_prime_gcnn 提交 26298486e341b32b115fb2f6717bdb1e0f54697e；cable_harness_rl 仅作为旧训练器、缓存修复和历史实验口径的迁移来源。

---

## 0. 最终决策

此前方案中的六项主要风险均成立，其中前三项会直接导致“模型尚未训完或训完无法部署”。修订后的硬决策如下。

| 问题 | 最终决策 | 原因 |
|---|---|---|
| Replay 爆内存 | 训练前完成候选精确二跳子图；Replay 同时按条数和实际字节淘汰；real_02 独立小池 | 2048 条完整 GraphState 在 real_02 上不可接受 |
| real_04 callback 太慢 | 训练前运行 real_04 首分支图探针；正式提取默认使用精确二跳子图 | 不能等训练完成后才发现推理不可用 |
| 训练/生产协议不一致 | 只保留 project-production-v1；训练、验证、baseline、real_04 共用 | 协议迁移失败比网络容量更可能导致 real_04 退化 |
| 奖励与 gap/PDI 不完全一致 | 第一版仍用节点加 LP 混合奖励；checkpoint 与正式结论改按 solved、PDI、gap、PAR-2 | 先修正选模与记录，避免立即引入高方差长时程奖励 |
| 3000 gradient steps 太薄 | 改为 800 次真实环境决策、每步 4 次更新，约 3200 次梯度更新 | 用环境经验数表达预算，避免“很多更新、很少新轨迹” |
| baseline 复用风险 | 默认重跑 real_04 的 6 个 baseline；旧结果仅作历史参考 | 参数、二进制、硬件隔离任一不一致都会破坏比较 |
| 训练集加入多个真实实例 | 接受；报告表述改为“多真实实例训练、未见 real_04 目标评测” | 目标是 real_04 效果，不再宣称纯零样本泛化 |

最终方法名固定为 **DSU-Prime-GCNN**。部署时只有一个学习模型。SCIP 的 project-default 与 random 只是实验对照，不是第二、第三个学习模型。

---

## 1. 方法边界：模型做什么，不做什么

原始线束 MILP、SCIP 预处理、LP、cuts、heuristics、B&B、可行性检查和最优性证明全部保留。模型只在 SCIP 请求 branching 时，从 SCIP 当前返回的合法 fractional LP candidates 中选择一个变量。

~~~mermaid
flowchart TD
    A["线束 MILP"] --> B["SCIP 求解当前节点 LP"]
    B --> C["合法 fractional candidates"]
    C --> D["候选精确二跳图 + DSU-Prim 特征"]
    D --> E["单个 Prime-GCNN 输出候选 Q"]
    E --> F["选择 argmax Q 的合法变量"]
    F --> G["SCIP 分支、剪枝并继续精确求解"]
    G --> B
~~~

模型不直接生成整棵树，也不把 Prim 当作固定贪心求解器。Prim 思想进入状态特征，由 GCNN 学习在不同 LP 状态下何时选择 frontier、merge、unseen 或避开 cycle。

### 1.1 本轮删除或退出主线

- Candidate MLP 与 rl-mlp；
- controlled-bbmdp 与 production-scip 双协议；
- Q 加 lambda 乘 PrimScore 的固定解码偏置；
- none、z、root_z、prim、topology 等 bias mode；
- Prim lambda、depth gate、feature on/off 等兼容开关；
- C0、C1、strong-branching teacher 的训练和生产入口；
- 运行时 19 维或 25 维特征切换；
- 模型错误后静默回退 relpscost 并继续把结果记为 RL。

历史代码暂不必物理删除，但必须从默认构建、默认配置和正式实验入口中隔离。real_04 结果完成后再迁移到历史 tag。

### 1.2 合法动作与错误语义

- 动作集合唯一来源是 SCIPgetLPBranchCands；
- 模型只能返回该集合中的变量；
- 当前没有 LP branching candidate 时，branchrule 可以正常返回 SCIP_DIDNOTRUN；
- 模型、manifest、schema、特征、候选映射、CUDA 或 Q 值异常必须令本次运行失败并标记 invalid_policy_run；
- 正式 DSU-Prime-GCNN 运行要求 unexpected_fallback 等于 0。

---

## 2. Prime 结构如何进入状态

### 2.1 每层维护并查集

对每个层 p 建立一个 DSU。遍历本层 z_i_j_p 变量，只有当前 B&B 节点的局部下界满足

\[
\operatorname{lb}_{local}(z_{i,j,p}) > 0.5
\]

才合并端点 i 与 j。这代表该边已经在当前节点被确定为 1。

以下信息不能用于 DSU 合并：

- LP solution 大于 0.5；
- 变量当前接近上界；
- 由浮点阈值猜测出的“可能选中”；
- incumbent 中出现但当前节点尚未固定的边。

LP solution 仍作为普通连续特征进入 GCNN，但不代表已经成为当前树的一部分。Python/Ecole 与 C++ 都必须读取真实 local lower bound。

### 2.2 六维 DSU-Prim 特征

对 z_i_j_p 计算：

| 特征 | 定义 |
|---|---|
| prim_frontier | 恰好一个端点已进入某个已生长分量 |
| prim_merge | 两端已生长，且属于两个不同分量 |
| prim_cycle | 两端已生长，且属于同一分量 |
| prim_unseen | 两端均未进入任何已生长分量 |
| prim_src_component_ratio | 起点分量大小除以本层已生长节点数；未生长为 0 |
| prim_dst_component_ratio | 终点分量大小除以本层已生长节点数；未生长为 0 |

前四个二值状态互斥。对 m、y、f、absf 和 other 变量，六维均为 0。根节点没有固定为 1 的 z 边时，大部分 z 候选是 prim_unseen；选出第一条边并进入相应子节点后，frontier、merge、cycle 才逐步出现。

prim_cycle 是风险信号，不是 hard mask。是否能选某条边仍由 MILP 约束与 SCIP 决定，近似拓扑判断不能替代数学可行性。

### 2.3 是否使用边之间的相邻关系

使用，而且有两条信息通路：

1. **约束相邻关系**：两条边变量若共同出现在某个 MILP 约束中，会通过变量到约束再到变量的消息传播相互影响；
2. **原拓扑相邻关系**：DSU 根据端点、连通分量和当前已固定树边产生 frontier、merge、cycle 等显式特征。

因此无需再维护一套独立边—边图。二部图表达代数耦合，DSU 特征表达树生长语义。

---

## 3. 候选精确二跳子图：从补丁提升为 P0 必选项

### 3.1 子图闭包

当前模型只做一轮 variable → row → variable 消息传播。对当前所有候选 C，构造：

\[
R_C=\bigcup_{v\in C}N(v)
\]

\[
V_C=C\cup\bigcup_{r\in R_C}N(r)
\]

子图保留：

- 所有合法候选变量；
- 与任一候选相邻的全部约束行；
- 上述每一行连接的全部变量；
- 这些行与变量之间的全部原始二部图边；
- 与完整图相同的行度数、mean 聚合、归一化和全局特征；
- 原候选顺序到子图局部索引的确定性映射。

不能随机截断行邻居，不能只保留每行 top-k 变量，也不能先对完整图做未知归一化再在子图中换一套度数。

### 3.2 为什么候选 Q 可以保持不变

一轮传播中，候选 v 的更新只依赖：

1. 与 v 相邻的行；
2. 这些行的更新表示；
3. 每个相关行上的全部变量初始表示；
4. 原始边特征、完整行 mean 分母和相同全局状态。

候选精确二跳闭包完整保留了这四项，所以在实现正确、网络确实只有一轮传播时，候选 Q 应与完整图相同，仅允许浮点舍入误差。

若未来增加第二轮消息传播，此等价性不再自动成立，必须扩大闭包或重新证明。第一版禁止增加 GNN 层数。

### 3.3 候选并集仍过大时

如果 real_04 候选并集二跳图仍接近完整图，则按确定顺序把候选分块，例如每块 64 个候选：

~~~text
候选块
  → 该块候选相邻的全部行
  → 这些行的全部变量
  → 仅输出该块候选 Q
  → 拼接全部候选 Q 后统一 argmax
~~~

候选分块是精确计算，只增加重复编码，不改变网络。块大小由图探针决定，不按实验结果调优。

---

## 4. real_04 首分支图探针

该探针必须在正式训练前完成。它是工程可行性检查，不读取策略优劣标签、不参与超参数选择，因此不改变 real_04 作为最终目标实例的地位。

### 4.1 探针行为

在 project-production-v1 下运行 real_04，抵达第一个合法 branching state 后：

1. 记录候选数、完整变量数、行数和非零边数；
2. 仅按数量和 tensor dtype 先估算完整图内存；
3. 若完整图预计超过 8 GiB 或主机可用内存的 50%，跳过完整 tensor 物化，标记 full_materialization_skipped；
4. 构造候选精确二跳图并记录变量、行、边、构造时间、估算字节和 RSS；
5. 若二跳并集仍过大，再测试 64 候选分块的 p50、p95 和最大规模；
6. 写出 JSON 后主动中断，不继续求解。

至少输出：

~~~text
candidate_count
full_variable_count / full_row_count / full_edge_count
full_estimated_bytes / full_extract_seconds / full_peak_rss
twohop_variable_count / twohop_row_count / twohop_edge_count
twohop_actual_bytes / twohop_extract_seconds / twohop_peak_rss
chunk_count / chunk_p50_bytes / chunk_p95_bytes / chunk_max_bytes
instance_sha256 / profile_sha256 / binary_sha256
~~~

### 4.2 训练前门禁

满足以下条件后才启动四卡训练：

- 二跳提取能完成且无 OOM；
- 候选索引均可映射；
- 若完整图可物化，full 与 two-hop 的 Q max_abs_error 不大于 1e-5，argmax 一致；
- 若完整图不可物化，在较小 real_08 状态完成相同等价性测试；
- 预计单状态 Replay 大小低于 512 MiB，或已启用候选分块并降低到阈值以内；
- 单次提取与推理的保守估计不会明显超过 real_04 一小时预算的 10%。

这里的 10% 是前置估算；最终仍以 600 秒 smoke 的累计开销为准。

---

## 5. 图输入与模型结构

### 5.1 唯一输入契约

| 对象 | 数值特征 | 类别特征 | 编码器输入 |
|---|---:|---:|---:|
| 变量节点 | Ecole 19 + DSU-Prim 6 = 25 | m/z/y/absf/f/other 共 6 | 31 |
| 约束节点 | 扩展行特征 14 | flow/absolute/topology/selection/imbalance/other 共 6 | 20 |
| 二部图边 | 系数、归一化系数、符号共 3 | 无 | 3 |
| 全局状态 | depth、nodes、bounds、gap、LP iterations、incumbent 等 14 | 无 | 14 |

25 是模型文件收到的 variable numeric tensor 维度；31 是拼接 6 维类别 one-hot 后变量编码器的输入维度。运行时不再允许 19/25 二选一。

所有连续特征使用训练 warmup 后冻结的 mean/std。类别 one-hot 不归一化。缺失或无穷的 bound/gap 必须按 schema 中固定规则裁剪并增加既有 missing 标记，Python/C++ 规则完全一致。

### 5.2 编码器

| 模块 | 网络 | 输出维度 |
|---|---|---:|
| Variable encoder | Linear(31,128) → ReLU → Linear(128,64) → ReLU | 64 |
| Row encoder | Linear(20,128) → ReLU → Linear(128,64) → ReLU | 64 |
| Edge encoder | Linear(3,128) → ReLU → Linear(128,64) → ReLU | 64 |
| Global encoder | Linear(14,128) → ReLU → Linear(128,64) → ReLU | 64 |

### 5.3 一轮消息传播

\[
h_v=\operatorname{Enc}_v([x_v,c_v]),\quad
h_r=\operatorname{Enc}_r([x_r,c_r]),\quad
h_e=\operatorname{Enc}_e(x_e)
\]

\[
m_r=\operatorname{Mean}_{v\in N(r)}
\operatorname{MLP}_{v\rightarrow r}([h_v,h_e])
\]

\[
h'_r=\operatorname{MLP}^{update}_r([h_r,m_r])
\]

\[
m_v=\operatorname{Mean}_{r\in N(v)}
\operatorname{MLP}_{r\rightarrow v}([h'_r,h_e])
\]

\[
h'_v=\operatorname{MLP}^{update}_v([h_v,m_v])
\]

Mean 必须使用完整保留行邻居后的真实度数。空邻域按 schema 置零，不允许 Python 与 C++ 使用不同默认值。

### 5.4 候选 Q head

只为 SCIP 当前合法候选计算：

~~~text
候选变量表示 64 + 全局表示 64
  → Linear(128,128)
  → ReLU
  → Linear(128,1)
~~~

\[
Q(s,a)=\operatorname{MLP}_Q([h'_a,h_g]),\qquad
a^*=\arg\max_{a\in A(s)}Q(s,a)
\]

Q 值后不再叠加手工 PrimScore。部署时只有这一份 TorchScript 模型。

---

## 6. 训练 MDP 与 Double DQN

### 6.1 一次环境交互

~~~text
SCIP 运行到 branching decision
  → 提取候选精确二跳状态 s_t
  → epsilon-greedy 选择合法候选 a_t
  → SCIP 执行分支并继续到下一决策
  → 计算 r_t 和 next state
  → 写入受内存约束的 prioritized replay
  → 执行 4 次梯度更新
~~~

transition 逻辑字段为：

\[
(s_t,a_t,r_t,s_{t+1},A(s_{t+1}),terminated,truncated)
\]

### 6.2 奖励

新增 RewardMode.HYBRID_NODE_LP：

\[
r_t=-\Delta N_t-10^{-4}\Delta LP_t
\]

- ΔN 是相邻两次决策之间新增的 B&B nodes；
- ΔLP 是相邻两次决策之间新增的 LP iterations；
- 所有差分先检查非负与计数器重置；
- 奖励、ΔN、ΔLP 分别写入训练日志。

该奖励仍不等于 gap/PDI，这是明确接受的 MVP 风险。第一版不把 PDI 直接塞入逐步奖励，因为长时程、incumbent 缺失和时间尺度会显著增加方差。补偿措施是用 PDI/gap 选 checkpoint 和做最终验收。

### 6.3 终止与截断

- optimal、infeasible、unbounded 或搜索树真正结束：terminated；
- time limit、node limit、训练器主动课程截断：truncated；
- truncated 且 Ecole 提供有效 next observation：保留 bootstrap；
- truncated 且 Ecole 返回 None：丢弃不完整 n-step 尾部，不伪造 terminal Q=0；
- 任何 SCIP restart 不能被误记为 episode terminal。

### 6.4 固定训练超参数

| 参数 | 值 |
|---|---:|
| Double DQN | 开启 |
| n-step | 3 |
| gamma | 1.0 |
| loss | Smooth L1 |
| logical batch size | 16 |
| min replay size | 64 |
| learning rate | 3e-4 |
| updates per environment decision | 4 |
| target soft-update tau | 0.01 |
| PER alpha | 0.6 |
| PER beta | 0.4 → 1.0，按 3200 次更新线性变化 |
| gradient clip | 10.0 |
| epsilon | 1.0 → 0.05，按 640 次环境决策衰减 |
| loss mode | scalar；不启用 HL-Gauss |

logical batch 16 不代表一次把 16 个大图全部送进 GPU。训练器按当前空闲显存做 micro-batch packing，默认单次 packed tensor 不超过可用显存的 25%，再用梯度累积形成 logical batch 16。OOM 时本轮 run 失败并记录，不在背后动态改模型或丢边。

### 6.5 Double DQN 更新

训练时存在 online network 与 target network。二者结构完全相同，target 只是 online 参数的延迟副本，不是第二个策略；导出时只导出选中的 online checkpoint。

对 n-step transition，online network 在 next candidates 中选动作，target network 估值：

\[
a'=\arg\max_{a\in A(s_{t+n})}Q_{online}(s_{t+n},a)
\]

\[
y_t=R_t^{(n)}
+\mathbf{1}_{bootstrap}\gamma^n
Q_{target}(s_{t+n},a')
\]

\[
L=\frac{1}{B}\sum_i w_i\,
\operatorname{SmoothL1}(Q_{online}(s_i,a_i)-y_i)
\]

bootstrap 指示量在真实 terminal 时为 0，在拥有有效 next state 的人工 truncation 时为 1。PER priority 使用绝对 TD error 加固定 epsilon 更新；importance weight 使用当前 beta，并在 logical batch 内归一化。

---

## 7. Replay 的强制内存设计

### 7.1 双池与双上限

每个训练进程拥有独立 Replay，放在 CPU 内存：

| 池 | 最大 transition 数 | 字节预算 | logical batch 配额 |
|---|---:|---:|---:|
| medium：real_01/03/05/06/07 | 224 | 3 GiB | 12 |
| large：real_02 | 32 | 1 GiB | 4 |
| 合计 | 256 | 4 GiB/worker | 16 |

当 real_02 池不足 4 条时，用 medium 补齐；阶段 A 尚无 real_02 时全部从 medium 取样。阶段 B 不允许大实例轨迹把中等实例完全挤出。

### 7.2 实际字节计数

每个 GraphState 计算：

\[
\text{state\_bytes}=\sum_{\text{tensor}} \text{storage\_nbytes}
\]

transition 的预算必须覆盖 state、next_state、候选索引、标量和容器开销。若 MVP 尚未实现相邻 transition 的 tensor storage 去重，则按重复存储后的实际占用计数，不能只按理论唯一 state 估算。

日志至少记录：

~~~text
replay_count
replay_bytes
state_bytes_p50 / p95 / max
medium_bytes / real02_bytes
evictions_by_count / evictions_by_bytes
host_rss / cuda_allocated / cuda_reserved
~~~

同时达到以下规则：

- 超过条数上限或字节预算时按各池自己的 FIFO/PER 生命周期淘汰；
- 单个 state 超过 512 MiB 立即 fail-fast，并回到候选分块设计；
- feature tensors 使用 float32；
- 图索引在确认小于 2 的 31 次方后以 int32 存 Replay，送入需要 int64 的算子前再转换；
- 禁止通过随机删边或固定 edge cap 掩盖内存问题；
- state/next_state 引用去重可作为后续优化，不是第一版上线前的必需条件。

### 7.3 四进程主机预算

四卡并行意味着 Replay 上限合计约 16 GiB，此外还有四个 SCIP 进程、CIP 模型、Python 对象和 page cache。开训前必须确认主机可用 RAM 留有至少 30% 安全余量；达不到时把每 worker 预算等比例下调，而不是交换到磁盘继续训练。

---

## 8. 训练与生产使用同一个 SCIP 协议

### 8.1 唯一协议 project-production-v1

这是当前项目 production-scip 的单线程、可复现版本，不等于 stock SCIP default。

| 项目 | project-production-v1 |
|---|---|
| SCIP plugins | SCIPincludeDefaultPlugins |
| node selector | 保留 SCIP 8.0.4 当前项目实际生效的 estimate；启动后断言并记录 |
| cuts/separation | SCIP 默认开启；不设置 separating/maxrounds=0 |
| restart | SCIP 默认开启；不设置任何禁用 restart 参数 |
| presolve | 默认开启 |
| heuristics | 默认开启；RENS freq 50、priority 100000；ALNS freq 50、priority 90000 |
| branching preference | branching/preferbinary=true |
| parallel/minnthreads | 1 |
| parallel/maxnthreads | 1 |
| lp/threads | 1 |
| gap limit | 0，除非实验配置明确另设 |
| random seeds | randomseedshift、permutationseed、lpseed 均取 run seed |

明确删除训练端这些旧覆盖：

~~~text
nodeselection/dfs/stdpriority = 1000000
nodeselection/dfs/memsavepriority = 1000000
separating/maxrounds = 0
estimation/restarts/restartpolicy = n
limits/restarts = 0
presolving/maxrestarts = 0
~~~

### 8.2 单一参数来源

新增：

~~~text
configs/scip/project-production-v1.set
~~~

- C++ 在创建 SCIP 并注册插件后通过 SCIPreadParams 读取；
- Python 训练器解析同一 set 文件并转换为 Ecole scip_params；
- time limit、node limit 与三个 seed 参数作为 run-specific overrides，应用在 profile 之后；
- 每次运行写出 base profile SHA、最终完整参数 dump SHA、SCIP 版本、二进制 SHA；
- 启动后记录实际 active node selector 与 branchrule priority；
- 任一未知参数或设置失败都直接失败。

训练、real_08 验证、real_04 project-default、random 和 DSU-Prime-GCNN 使用相同 base profile。三种 real_04 方法唯一允许的搜索差别是 branching rule。

### 8.3 三种正式方法

| 报告名称 | 当前 CLI 可映射名称 | 实际分支规则 |
|---|---|---|
| project-default | default | 当前项目 relpscost 路径 |
| random | custom-random | 在合法 LP candidates 中按 run seed 随机 |
| DSU-Prime-GCNN | rl-gcnn | 单个已选 TorchScript 模型 |

若后续重命名 CLI，也必须保留 manifest 与结果 JSON 中稳定的方法名。

---

## 9. 数据划分与 800 次环境决策课程

### 9.1 数据角色

~~~text
训练阶段 A：real_01 / real_03 / real_05 / real_06 / real_07
训练阶段 B：上述五个实例 + real_02 截断轨迹
验证：       real_08
额外诊断：   real_09，仅在 real_08 排名不稳定时使用
最终测试：   real_04
~~~

real_04 不参与 normalization、训练、checkpoint 选择、奖励选择、块大小效果调参或 seed 选择。首分支探针只用于确定计算可行性。

报告中的科学表述固定为：

> 在 real_01/02/03/05/06/07 上进行多真实实例训练，在未参与训练和选模的 real_04 上进行目标评测。

不得再写“只用两个数据集训练”或“向所有未见真实实例零样本迁移”。

### 9.2 Normalization warmup

训练更新开始前收集至少 12 个状态：

~~~text
real_01 / 02 / 03 / 05 / 06 / 07
每个实例至少 2 个合法 branching state
~~~

计算连续特征 mean/std 后立即冻结。四个训练 seed 使用同一份冻结 normalization，确保模型之间可比较；该文件只由独立 warmup seed 生成，不按 real_08 或 real_04 表现修改。

### 9.3 阶段 A：600 次环境决策

| 实例 | 决策配额 | episode time limit | episode node limit |
|---|---:|---:|---:|
| real_01 | 120 | 60 s | 100 |
| real_03 | 120 | 60 s | 100 |
| real_05 | 120 | 60 s | 100 |
| real_06 | 120 | 60 s | 100 |
| real_07 | 120 | 60 s | 100 |
| 合计 | 600 |  |  |

每个 seed 独立打乱实例轮转顺序，但最终配额必须相等。一个 episode 没有产生 branching decision 时记录 no_branch_episode 并重新抽取，不计入 600。

### 9.4 阶段 B：200 次环境决策

| 实例 | 决策配额 | episode time limit | episode node limit |
|---|---:|---:|---:|
| real_02 | 60 | 300 s | 100 |
| real_01 | 28 | 60 s | 100 |
| real_03 | 28 | 60 s | 100 |
| real_05 | 28 | 60 s | 100 |
| real_06 | 28 | 60 s | 100 |
| real_07 | 28 | 60 s | 100 |
| 合计 | 200 |  |  |

real_02 只提供截断轨迹，不要求求解完成。若它在多次 episode 中都不能产生合法 decision，应先检查协议、候选提取和 time limit，而不是用更多梯度更新补足。

### 9.5 总训练量

~~~text
环境决策：600 + 200 = 800
每次决策更新：4
预计梯度更新：约 3200
epsilon 衰减：前 640 次环境决策
~~~

环境决策才是主进度单位。gradient step、episode、SCIP node 和 wall time同时记录，但不能替代真实交互数。

---

## 10. 四卡并行与模型选择

### 10.1 四个独立 seed

~~~text
GPU 0 → training seed 0
GPU 1 → training seed 1
GPU 2 → training seed 2
GPU 3 → training seed 3
~~~

每个进程独立拥有 SCIP/Ecole environment、online/target network、Replay、optimizer 和日志。四卡不是把一个小模型做 tensor parallel，也不会让一个 SCIP 实例快四倍。

模型输出：

~~~text
artifacts/models/prime_gcnn_dsu/seed0/
artifacts/models/prime_gcnn_dsu/seed1/
artifacts/models/prime_gcnn_dsu/seed2/
artifacts/models/prime_gcnn_dsu/seed3/
~~~

加入 real_02 且改为 800 次真实决策后，单 seed wall time 很可能超过旧 seed0 的约 1.63 小时；无法在探针前给出可靠完成时间。四卡的意义是把四个 seed 同时跑完。

### 10.2 不阻塞训练的 checkpoint 策略

每 100 次环境决策保存一次 checkpoint，但训练过程中不插入长时间 real_08 求解。四个 seed 全部结束后分两轮验证：

1. 快筛：所有 checkpoint 在 real_08 seed 100 上跑 120 秒；
2. 复筛：快筛排名前 2 的 checkpoint 在 real_08 seeds 100/101/102 上跑 300 秒。

这样既避免每 250 gradient steps 阻塞训练，又保留独立验证选模。最终只选择一个 seed 的一个 checkpoint 部署到 real_04。

### 10.3 checkpoint 排序

若 Python 路径可以可靠记录 PDI：

~~~text
solved count 更高
→ mean PDI 更低
→ mean final gap 更低
→ mean PAR-2 更低
→ mean LP iterations 更低
→ mean nodes 更低
~~~

若 PySCIPOpt/Ecole 不能稳定给出 PDI：

~~~text
solved count 更高
→ mean final gap 更低
→ mean PAR-2 更低
→ mean LP iterations 更低
→ mean nodes 更低
~~~

C++ 正式 real_04 仍必须记录 PDI。禁止按 training loss 或最少 nodes 单独选择部署模型。

---

## 11. 评价器必须新增的指标

当前 evaluate_gcnn_episode 不足以支持正式选模。每个 episode 至少记录：

~~~text
status
solving_time
node_count
lp_iterations
primal_bound
dual_bound
final_gap
primal_dual_integral（接口可得时）
first_solution_time
terminated / truncated / truncation_reason
rl_actions
unexpected_fallback
graph_extract_seconds
inference_seconds
model_load_seconds
~~~

gap 的无 incumbent、无有限 bound 和已最优情形必须使用一套固定规则，写入 schema 文档。PAR-2 使用该轮 time limit 的两倍作为未解惩罚。

---

## 12. Python/C++ 特征、模型与子图一致性

### 12.1 端到端 parity

在同一个 SCIP branching state 上比较：

1. transformed variable order；
2. candidate names、原索引与局部索引；
3. local lower bounds；
4. 六维 DSU-Prim 特征；
5. 完整 25 维变量矩阵；
6. row、edge、global tensors；
7. full graph 与 candidate two-hop graph 的 candidate Q；
8. Python 与 C++ 的 candidate Q 和 argmax。

验收：

~~~text
feature max_abs_error <= 1e-6
Q max_abs_error <= 1e-5
candidate order identical
argmax identical
~~~

只把同一个 NPZ 分别送入 Python/C++ forward，只验证了网络执行，不足以验证两端从 SCIP 状态开始的特征提取。

### 12.2 getVars 与局部下界

移植旧仓库已经验证的 transformed variable 缓存修复：

- 模型变换后缓存稳定的变量顺序和 name 到 index 映射；
- 只有变量数量或 transformed stage 改变时才刷新；
- 不在每个 callback 盲目调用 getVars(transformed=True)；
- local bound 从真实 SCIP transformed var 读取；
- cache miss、重复变量名或候选不在缓存中均 fail-fast。

### 12.3 模型 manifest

每个模型目录至少包含：

~~~json
{
  "schema_version": 2,
  "method": "dsu-prime-gcnn",
  "variable_numeric_dim": 25,
  "variable_category_dim": 6,
  "row_numeric_dim": 14,
  "row_category_dim": 6,
  "edge_dim": 3,
  "global_dim": 14,
  "message_passing_rounds": 1,
  "model_sha256": "...",
  "schema_sha256": "...",
  "normalization_sha256": "...",
  "scip_profile_sha256": "...",
  "scip_version": "8.0.4",
  "ecole_version": "0.8.1",
  "torch_version": "2.5.1+cu121",
  "training_commit": "..."
}
~~~

C++ 启动时校验 manifest 和全部 SHA；第一次进入合法 callback 时 lazy-load TorchScript 并 warm up。加载时间单独记账。

---

## 13. 代码改造清单

以下是达到 real_04 结果所需的最小主链。

当前 RewardMode 只有 negative_node_increment 与 constant_minus_one，evaluate_gcnn_episode 也没有完整记录 gap/PDI。因此 hybrid reward、截断 bootstrap 和选模指标都必须在训练器与评价器中真实实现，不能只改 YAML。

| 文件或目标文件 | 必做修改 |
|---|---|
| python/rl_branching/prim_bias.py | 单一 DSU 六维特征；删除多 bias mode |
| python/rl_branching/observation.py | transformed vars 稳定缓存；真实 local bounds |
| python/rl_branching/graph_features.py | 固定 25 维 schema；实现候选精确二跳闭包 |
| python/rl_branching/gcnn_config.py | 800 environment decisions、两阶段配额、单协议配置 |
| python/rl_branching/gcnn_trainer.py | HYBRID_NODE_LP、截断 bootstrap、四次更新、micro-batch、选模指标 |
| python/rl_branching/graph_replay.py | 新增双池、条数和字节双上限、实际 tensor byte 日志 |
| python/rl_branching/config.py | 删除 DFS/cuts/restart 旧覆盖；读取 project-production-v1 |
| python/rl_branching/environment.py | 正确区分 terminal、truncation、restart 和 None observation |
| src/rl/prim_bias.cpp/.hpp | 与 Python 完全相同的 DSU 逻辑 |
| src/rl/scip_graph_feature_extractor.cpp/.hpp | 固定 25 维；精确二跳图；候选分块 |
| src/rl/gcnn_model_runner.cpp/.hpp | manifest/SHA 校验、lazy load、finite Q 检查 |
| src/rl/rl_gcnn_branchrule.cpp/.hpp | 单一模式、合法候选映射、错误 fail-fast、开销计时 |
| code/scip_tree.cpp | 只保留 project-production-v1 正式入口；完整参数/指标输出 |
| tools/graph_probe.cpp | real_04 首分支全图估算、二跳图与分块探针 |
| scripts/run_prime_gcnn_dsu_train_4gpu.sh | 四个独立 seed 的 GPU/输出目录绑定 |
| scripts/run_final_experiments.py | 单协议、workers=1、hash-aware resume、结果完整性检查 |
| configs/scip/project-production-v1.set | 唯一 SCIP base profile |

### 13.1 单元与集成测试

至少新增：

- DSU 在 unseen、frontier、merge、cycle 四种状态的手工小图测试；
- 只有 local lb 大于 0.5 才 union 的回归测试；
- 非 z 变量六维全零；
- full 与 two-hop Q parity；
- candidate chunk 与 union-two-hop Q parity；
- Replay 条数淘汰、字节淘汰、real_02 配额与单状态 512 MiB fail-fast；
- truncated with next state bootstrap；
- truncated without next state 丢弃 n-step tail；
- Python/C++ 同状态特征 parity；
- manifest 维度、SHA、版本不匹配时非零退出；
- RL 输出不在候选集、Q 包含 NaN/Inf 时非零退出；
- 三种方法最终参数 dump 除 branching rule 外一致。

---

## 14. real_04 最快可信实验路径

固定实例：

~~~text
data/instances/transfer/real_04.cip
SHA256 = 8aa29d655d77df8b6b169dbcd25801dd770af9aab405d2f91d40c4524a2be711
~~~

### R0：训练前图规模探针

~~~bash
cd /data/hanchengcheng/hcc_1/du/cable_harness_prim_gcnn
make -j8

./build/graph_probe \
  --instance data/instances/transfer/real_04.cip \
  --scip-profile configs/scip/project-production-v1.set \
  --output results/probes/real04_first_branch.json \
  --seed 0 \
  --threads 1
~~~

命令是目标接口。graph_probe 完成实现、编译和门禁前不要直接执行。

### R1：四卡训练

~~~bash
cd /data/hanchengcheng/hcc_1/du/cable_harness_prim_gcnn
CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash scripts/run_prime_gcnn_dsu_train_4gpu.sh
~~~

每个进程日志必须同时显示 environment_decisions、gradient_updates、Replay bytes、实例配额、epsilon、host RSS 和 CUDA memory。

### R2：real_08 选一个模型

先运行全部 checkpoint 的 seed100、120 秒快筛，再对前 2 名运行 seeds100/101/102、300 秒复筛。选出一个 model SHA，并冻结：

~~~text
selected_model.json
model.pt
normalization.json
schema.json
manifest.json
~~~

此后不能根据 real_04 结果在四个 seed 间重新挑模型。

### R3：real_04 600 秒 smoke

~~~text
method: DSU-Prime-GCNN
seed: 0
time limit: 600 s
node limit: -1
threads: 1
workers: 1
device: cuda
branch log: enabled
~~~

通过条件：

~~~text
status in {optimal, time_limit}
rl_actions > 0
unexpected_fallback = 0
model/schema/profile/instance SHA 全部匹配
selected_is_candidate = true
无 OOM、NaN、异常退出
累计 GCNN overhead <= solving time 的 10%
~~~

\[
\text{overhead ratio}=
\frac{T_{extract}+T_{inference}+T_{load}}{T_{solve}}
\]

同时报告每次 callback 的 extract 与 inference p50、p95、max，避免平均值掩盖长尾。

### R4：seed0 三方法 quick-look

用完全正式的一小时配置依次运行：

~~~text
project-default × seed0 × 3600 s
random          × seed0 × 3600 s
DSU-Prime-GCNN  × seed0 × 3600 s
~~~

三次约 3 小时，均写入正式结果目录，后续可直接作为 formal seed0，不重复跑。

若出现以下任一项，暂停剩余 6 次正式运行并先修实现：

- RL 没有接管动作或有 unexpected fallback；
- overhead 超过 10%；
- 参数 dump 不一致；
- 非法 candidate、OOM、NaN、crash；
- DSU-Prime-GCNN 的 gap 与 PDI 均明显劣于两个 baseline，且差异不是单次求解未记录完整造成。

### R5：补 seeds 1 和 2

quick-look 通过后追加：

~~~text
project-default / random / DSU-Prime-GCNN
× seeds 1,2
× 3600 s
= 6 个新 run
~~~

与 seed0 合计为 9 个正式 run，单机串行总预算约 9 小时。旧仓库结果默认不复用；只有实例 SHA、完整参数 dump SHA、SCIP/SoPlex、二进制、硬件和资源隔离全部一致时，才可把旧结果明确标记为复用。

### R6：可选补 seeds 3 和 4

三 seed 结果通过最低有效性门槛后，再补三方法的 seeds3/4，形成 15 个 run。该步骤用于稳定性，不阻塞尽快看到 real_04 初步结果。

### 14.1 正式并行规则

- 四张 GPU 用于四个训练 seed 并行；
- real_04 同一服务器默认 workers=1、顺序执行；
- 只有独立机器或隔离 CPU、RAM 和存储资源时，才按 seed 并行正式 wall/PDI 对比；
- 随意四路并发会改变内存带宽、CPU cache 和求解 wall time，不可用于公平结论。

### 14.2 从代码完成到看到结果的最短时间线

| 阶段 | 预计 wall time | 能看到什么 |
|---|---:|---|
| real_04 首分支探针 | 取决于 root LP；设置 3600 s 安全上限 | 部署图规模与内存是否可行 |
| 四卡训练 | 预计数小时，必须以实测为准 | 四个 seed 的训练产物 |
| real_08 两轮选模 | 四卡并行时约几十分钟到一小时级 | 唯一冻结模型 |
| real_04 smoke | 10 分钟 | 模型是否真正接管、是否 OOM、开销是否合格 |
| seed0 三方法 quick-look | 约 3 小时串行 | 第一份可比较的一小时结果 |
| 补 seeds1/2 | 约 6 小时串行 | 三 seed 正式结论 |

最快的真实反馈点是 600 秒 smoke；最快的性能对照点是其后的 seed0 三方法 3 小时 quick-look。quick-look 使用正式配置并直接计入 9-run 结果，所以不会浪费三小时。

---

## 15. real_04 指标与结论口径

| 优先级 | 指标 | 用法 |
|---|---|---|
| 主指标 | final gap at 3600 s | 固定预算下最终上下界距离 |
| 主指标 | primal-dual integral | 整个一小时内界质量，越低越好 |
| 主指标 | first solution time | 是否更快得到可行解 |
| 主指标 | solved/optimal | 是否在预算内完成 |
| 支撑指标 | primal bound / dual bound | 区分改善来自可行解还是下界 |
| 支撑指标 | PAR-2 | 汇总未解实例的时间惩罚 |
| 次指标 | LP iterations | 节点内部工作量 |
| 次指标 | nodes | 搜索树大小，禁止单独定胜负 |
| 系统指标 | extract/inference/load | GCNN 部署成本 |

最低有效性门槛：

~~~text
每个 RL run 的 rl_actions > 0
unexpected_fallback = 0
没有非法 candidate、OOM、NaN 或 crash
方法间 instance/profile/version/hardware 口径一致
三 seed 上 PDI 或 final gap 相对 project-default 有稳定改善
GCNN 总开销 <= solving time 的 10%
~~~

已有历史现象表明“节点更少”可以同时伴随“最终 gap 更差”。因此不能用 nodes 代替方法效果，也不能用 training loss 下降证明 real_04 有效。

---

## 16. 风险复核与发生后的第一动作

| 风险 | 当前评级 | 已纳入的前置处理 | 若仍发生，第一动作 |
|---|---|---|---|
| Replay 内存 | 极高 | 二跳子图、4 GiB 字节上限、双池、512 MiB 单状态门禁 | 减小候选 chunk 和 Replay 字节预算；不删边 |
| real_04 callback 开销 | 极高 | 训练前首分支探针、默认二跳、必要时候选分块 | 优化 tensor 构造/缓存；不先加 GNN 层 |
| 训练/生产分布偏移 | 高 | project-production-v1 全链统一 | 先核参数 dump、node selector、restart/cuts，再怀疑模型 |
| 奖励与 PDI/gap 偏移 | 中高 | hybrid reward；按 PDI/gap 选模 | 第二轮才研究 bound-improvement reward |
| 800 决策仍偏薄 | 中高 | 多实例配额、real_02 60 次决策、4 updates/decision | real_04 先跑；若结构有效再扩到 1600 决策 |
| baseline 口径错误 | 高 | 默认重跑 6 次、SHA-aware resume | 删除不可比复用标记并重跑 |
| 多真实实例表述过度 | 中 | 固定报告用语 | 不声称零样本普适性 |
| 模型结构不够深 | 低于前三项 | DSU 已显式提供结构；先用一轮保证精确二跳 | 仅在协议和部署均通过后做层数消融 |

对 real_04 结果不佳时，排查顺序固定为：

~~~text
接管与候选是否合法
→ Python/C++ 与 full/two-hop parity
→ SCIP 协议和完整参数是否一致
→ callback 开销是否吞掉预算
→ reward/选模是否偏离 gap/PDI
→ 训练决策量是否不足
→ 最后才考虑增加 GNN 层或更复杂模型
~~~

---

## 17. 推荐配置文件

建议新增：

~~~text
configs/scip/project-production-v1.set
configs/rl/prime_gcnn_dsu_seed0.yaml
configs/rl/prime_gcnn_dsu_seed1.yaml
configs/rl/prime_gcnn_dsu_seed2.yaml
configs/rl/prime_gcnn_dsu_seed3.yaml
configs/experiments/real04_graph_probe.json
configs/experiments/real04_prime_gcnn_smoke.json
configs/experiments/real04_prime_gcnn_3seed.json
configs/experiments/real04_prime_gcnn_5seed.json
~~~

训练 YAML 的核心字段应收敛为：

~~~yaml
method: dsu-prime-gcnn
scip_profile: configs/scip/project-production-v1.set
message_passing_rounds: 1
variable_numeric_dim: 25

training:
  total_environment_decisions: 800
  stage_a_decisions: 600
  stage_b_decisions: 200
  updates_per_environment_decision: 4
  logical_batch_size: 16
  min_replay_size: 64

replay:
  total_capacity: 256
  total_memory_gib: 4
  medium_capacity: 224
  real02_capacity: 32
  medium_batch: 12
  real02_batch: 4
  max_single_state_mib: 512

exploration:
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay_environment_decisions: 640

reward:
  mode: hybrid_node_lp
  lp_iteration_weight: 0.0001
  bootstrap_on_truncation: true
~~~

这些配置不再包含 protocol、bias_mode、lambda、Candidate MLP 或 feature dimension compatibility 开关。

---

## 18. 最快执行顺序与验收清单

~~~mermaid
flowchart TD
    P0["P0 统一 SCIP profile"] --> P1["P1 二跳提取、Replay、DSU"]
    P1 --> P2["P2 real_04 首分支探针"]
    P2 --> P3["P3 Python/C++ 与子图 parity"]
    P3 --> P4["P4 四卡训练 800 decisions"]
    P4 --> P5["P5 real_08 选单一模型"]
    P5 --> P6["P6 real_04 smoke 与 seed0 quick-look"]
    P6 --> P7["P7 补 seeds 1/2；通过后补 3/4"]
~~~

### 开训前

- [ ] project-production-v1 在 Python/C++ 的完整参数 dump SHA 一致；
- [ ] estimate node selector、cuts、restart、heuristics 与 threads=1 已实际生效；
- [ ] DSU 只使用 local lb 固定边；
- [ ] 25 维 schema 唯一；
- [ ] candidate exact two-hop 与 candidate chunk 已实现；
- [ ] real_04 首分支探针完成；
- [ ] Replay 双池、字节淘汰和 512 MiB 门禁通过；
- [ ] full/two-hop/chunk Q parity 通过；
- [ ] Python/C++ 端到端 parity 通过。

### 开 real_04 前

- [ ] 四个 seed 均完成 800 environment decisions；
- [ ] 每个实例配额满足；
- [ ] real_08 已冻结唯一 selected model SHA；
- [ ] manifest、normalization、schema、profile SHA 完整；
- [ ] C++ real_08 集成测试 rl_actions 大于 0、fallback 为 0；
- [ ] baseline 决定为重跑或已通过全部复用条件。

### 结果有效前

- [ ] smoke overhead 不超过 10%；
- [ ] seed0 三方法结果完整并可复用；
- [ ] 三 seed 共 9 个 run 的配置、SHA、硬件口径一致；
- [ ] 报告同时给出 gap、PDI、first solution、bounds、LP、nodes 和开销；
- [ ] 结论不只依赖 nodes；
- [ ] 报告明确是多真实实例训练、real_04 目标评测。

---

## 19. 一句话实现定义

这版方案是在统一的 project-production-v1 SCIP 搜索环境中，用 DSU 从当前节点已固定为 1 的 z 边提取 Prim 式树生长状态，在候选精确二跳变量—约束图上由单轮 GCNN 为所有合法 branching candidates 估计 Q，并以受字节限制的 Double DQN 从 real_01/03/05/06/07 与 real_02 截断轨迹学习；最终只用 real_08 选一个模型，再以单线程、无静默回退的一小时 project-default/random/DSU-Prime-GCNN 对照评测 real_04。
