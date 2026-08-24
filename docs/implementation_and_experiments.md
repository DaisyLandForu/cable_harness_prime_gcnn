# 航空线束 MILP 分支强化学习：`cable_harness_prim_gcnn` 技术方案与实验事实

> 用途：作为可交给外部模型评估的自包含事实底稿。  
> 数据截止：2026-08-25；工作区：`cable_harness_prim_gcnn/`。  
> 请严格基于本文评估，不要虚构未做的实验，不要把 pilot / 单 seed 当成多 seed 正式结论，不要用 nodes 单独定胜负。

相关分阶段原稿：`docs/FINAL_RL_BRANCHING_REPORT.md`、`docs/PROJECT_HANDOFF_FOR_CHATGPT.md`、`docs/gcnn_report.md`、`results/outputs/phaseA_*.md`、`results/outputs/phaseB_*.md`、`results/c0_prim_decomposition/C0_PRIM_DECOMPOSITION.md`、`88/`。仓库总览见根目录 [`EXPERIMENT_REPORT.md`](../../EXPERIMENT_REPORT.md)。

---

## 0. 一句话结论（先读）

工程闭环已经打通：SCIP 8.0.4 + Ecole BBMDP + Candidate MLP / 二部图 GCNN-DQN + C++ TorchScript 部署。
**旧 30s 双协议上，策略质量尚未达到可部署标准。** 当前最强的 *learned* 30s 变体仍是冻结旧 GCNN + 解码偏置 `score = Q + 0.5 · PrimScore`（阶段 A），跨实例符号会翻转，不能替换 SCIP-default。

**2026-08-25 更新（DSU-Prime-GCNN，`real_04`，seed0，7200s）**：最大实例已经进入 B&B，可以评 branching。相对同期重跑的 `project-default` / `random`，正式权重的 DSU-Prime-GCNN **PDI 最低、首解最快**，推理开销 0.82%、构图+选择总开销 5.4%，均低于 10% 门禁，且 `fallback=0`。但它在 7200s 内 **未证明最优**（final gap 0.017%），而 default / random 分别在 6689s / 6141s 证到 `optimal`。这是 **单 seed、三路并行** 的结果，不能外推为稳定部署结论，也不能按 `real_04` 回头选模。详见 §5.10。

---

## 1. 问题设定

### 1.1 业务问题

航空电缆线束在候选网络上选择分层拓扑与路由，要求无环、满足连接/流平衡/层级。SCIP 目标主要是边长与聚合线缆权重对应的路由代价。

### 1.2 学习问题

**不改 MILP 的变量、约束、目标。** 只替换 SCIP 分支定界中的 **branching variable selection**。

- 状态：当前 LP 节点的变量–约束二部图 + 全局 B&B 特征（可选 Prim 邻域特征）
- 动作：当前 fractional LP branching candidates 中选一个变量
- 奖励（已实现）：默认 \(r_t = -(N_{t+1}-N_t)\)，即负节点增量；也支持 constant `-1`
- 折扣：\(\gamma=1\)
- 终止：optimal / 无开节点 / SCIP error；time/node limit 按 truncation，且 **timeout 不 bootstrap**

未指定 RL 时，仍走原项目 `production-scip/default`（实际最高生效规则是 SCIP `relpscost`）。

### 1.3 变量与约束（原模型，未改）

| 变量 | 类型 | 含义 |
|---|---|---|
| `m_k_p` | binary | 需求 k 分配到层 p |
| `z_i_j_p` | binary | 层 p 上选择有向拓扑边 (i,j)，主要整数分支对象 |
| `f_i_j_k` | continuous | 需求流 |
| `absf_i_j_k` | continuous | \|f\| 线性化 |
| `y_i_p` | continuous | MTZ 风格拓扑次序 |

约束族：`fforbid`, `abs1/abs2`, `flow_balance`, `flow_symmetry`, `onlym`, `imbalance`, `zlower`, `topo_seq1/2`, `only_father`。

业务先验：每层拓扑应近似**树状生长**。阶段 A/B 把 Prim 风格“割边扩展”注入分支。

---

## 2. 数据划分（硬边界）

切分单位是完整 MILP 实例，不把同一实例的 B&B 状态拆到不同集合。合成数据只用于阶段 2 能力验证，**不进入正式训练/评测结论**。

| split | 实例 | 规模（var / int / cons） | 阶段 2 default |
|---|---|---|---|
| train | `real_06` | 20,682 / 2,708 / 48,140 | optimal, 27.35s, 29 nodes |
| train | `real_07` | 38,100 / 3,036 / 93,965 | timeout |
| validation | `real_08` | 27,094 / 2,880 / 65,069 | optimal, 28.83s, 12 nodes |
| test | `real_09` | 10,746 / 2,668 / 22,075 | optimal, 4.09s, 9 nodes |
| transfer | `real_01`–`real_05` | 最大 `real_04`：326,502 / 5,168 / 863,691 | 多数 timeout；`real_04` 30s 内未进 B&B |

**关键事实**：阶段 5/7/8 正式 GCNN 实际只训 `real_06`、验 `real_08`。阶段 B 扩到 `real_06+real_07`，仍只有 2 个训练实例。

**评估污染**：`real_09` 和部分 transfer 已用于 λ 扫描，后续只能标 exploratory，不能当 untouched test 调参。

---

## 3. 完整实现路线

```
Phase 0  审计 SCIP/MILP/默认 branching
Phase 1  可复现 baseline（default/relpscost/random/mostinf/strong）
Phase 2  真实实例划分 + CIP 导出
Phase 3  自定义 branchrule 接口
Phase 4  Ecole BBMDP（状态/动作/奖励/截断）
Phase 5  Candidate MLP-DQN（39 维）
Phase 6  C++ LibTorch 接入 rl-mlp
Phase 7  二部图 GCNN-DQN + C++ rl-gcnn
Phase 8  双协议正式实验 430 + 消融 150

之后是本仓库特有的 Prim 线：
Phase A      冻结 GCNN，解码偏置 Q + λ·PrimScore
Phase A-ext  λ∈{0.25,0.5,1.0} + 门控 + 更多实例
Phase B      6 维 Prim 特征拼进变量节点，在线 DQN 再训
Phase C0     Prim 拆解（z / root-z / full-prim / topology-only）
Phase C1     SB ranking 采数（pilot 失败）
Phase C1.1   SB teacher repair（当前允许推进，尚未通过门禁）
Phase DSU    DSU-Prime-GCNN：候选精确二跳 + Stage A/B 正式训练 + `real_04` 7200s 评测（§5.10）
```

硬禁令（已写入方法契约）：禁止 hard-mask `prim_both_in`；禁止以最小节点数为唯一训练/选模目标；禁止用 test/transfer 调参；禁止在 C1.1 未通过时扩 30k 采数。

---

## 4. 核心方法

### 4.1 训练环境（BBMDP）

- SCIP 8.0.4 + PySCIPOpt 4.3.0 + Ecole 0.8.1（同源 SCIP）
- 受控训练协议：DFS、单线程、关 restart、非 root cuts 关
- 动作 = Ecole 变量行索引，经 action set mask
- 不跨 transition 保存失效 SCIP pointer
- 生产部署：C++ `ObjBranchrule`，优先级 `1,000,000`，几乎全树接管；失败回退 `relpscost`

训练协议与生产协议不一致：生产保留原 `estimate` node selector 和默认 cuts/heuristics，只换分支变量选择。这是后续迁移失败的重要原因之一。

### 4.2 Candidate MLP（对照，非 Prim 主线）

输入 39 维：Ecole 变量 19 + 全局树 14（**不含 wall-clock**）+ 航空类别 one-hot 6（`m/z/y/absf/f/other`）。

```
Linear(39,128)-ReLU-Linear(128,128)-ReLU-Linear(128,1)
→ candidate mask 内稳定 argmax
```

参数量 21,761。Double DQN，3-step，γ=1，hard target 每 250 步，replay 10k，Adam 3e-4。

### 4.3 二部图 GCNN（主模型）

不依赖 PyTorch Geometric，用 `index_add_` 做 mean aggregation，便于 TorchScript/C++。

**图特征**

| 对象 | 维度 | 内容 |
|---|---|---|
| 变量节点 | 25（B 阶段可到 31） | Ecole 19 + 航空变量类别 6（+ Prim 6） |
| 约束节点 | 20 | 行特征 14 + 约束类别 6（flow/absolute/topology/selection/imbalance/other） |
| 边 | 3 | 原系数、行 L2 归一化系数、符号 |
| 全局 | 14 | depth、开节点、bounds、gap、incumbent 等 |

`real_06` 探针约：8,372 变量、28,089 展开行、93,555 边。每次 callback **重建动态图**。

**网络**

```
var 25→128→64    row 20→128→64    edge 3→128→64    global 14→128→64
var→row message + mean agg + row update
row→var message + mean agg + var update
Q head: [updated_var, global] (128) → 128 → 1（或 18-bin HL-Gauss）
只对当前 candidate 打分
```

注意：实现是 **1 轮**（各方向一次）mean message passing，不是深层 GNN。已知风险：难捕获长程连通。

**学习器**

- Double DQN + 3-step + γ=1
- PER：α=0.6，β=0.4→1.0
- soft target τ=0.01
- ε: 1.0→0.05
- 标量 Smooth L1；可选 HL-Gauss：18 bins，\(z=\log_2(-Q)\)，\(Q=-2^z\)，\(z\in[-1,12]\)，σ=0.75
- truncated episode 默认不 bootstrap

阶段 7 pilot：1000 gradient steps、3 episodes、只训 `real_06`。阶段 B：3000 steps、batch=64、4 seed、`real_06+real_07`。

### 4.4 Prim 结构先验（本仓库核心增量）

由当前 LP/界构造每层生长集 \(S_p\)：`z` 满足 `lb>0.5` 或 `LP>0.5` 时，把两端点加入 \(S_p\)。

**PrimScore（阶段 A 解码）**

| 候选 | 分数 |
|---|---|
| z 割边（恰一端 ∈ S） | +1.0 |
| z 两端 ∉ S | +0.25 |
| z 两端 ∈ S | −0.5 |
| S 空时的 z | +0.5 |
| m / y 落在 S 上 | +0.3 / +0.15 |
| 其他 | 0 |

部署：

\[
\text{score}(a) = Q(a) + \lambda\cdot\text{PrimScore}(a)
\]

默认试 \(\lambda=0.5\)。门控：`--rl-prim-min-depth`、`--rl-prim-require-grown`。

**阶段 B 特征（6 维二进制，拼到变量特征后）**

`prim_is_cut, prim_both_in, prim_both_out, prim_grown_empty, prim_m_on_grown, prim_y_on_grown`

**C0 拆解模式**（同一 λ，换 bias 定义）

- `none`：纯 GCNN
- `z`：全深度凡是 z 都 +1（变量族先验）
- `root_z`：仅 depth=0 的 z +1
- `prim`：完整 PrimScore（含空集 +0.5）
- `topology`：PrimScore 但空集不加 +0.5（只保留连通/割边）

**已知几何粗糙处**

- \(S\) = 活跃边端点并集，**不是**从根出发的连通分量
- `both_in` ≠ 真正成环
- 0.5 硬阈值使 0.49/0.51 类别翻转
- 固定 λ 相对 Q 尺度不可迁移（审计：λ·|bias|/Q_std 均值约 0.47，约 26% 决策 Prim 量级超过 Q_std）
- 早期 Python grown 主要看 LP，C++ 看 `lb || lp`（C0 已对齐并加测试）

### 4.5 C++ 部署流程

```
SCIP 需要分支
  → SCIPgetLPBranchCands
  → depth / min-candidates gate
  → 提特征 + artifact normalization
  → TorchScript no-grad forward（模型只加载一次）
  → 检查维数/NaN
  → candidate mask 内稳定 argmax
  → 可选 Q + λ·PrimScore
  → SCIPbranchVarVal
  → 失败则 DIDNOTRUN，交给 relpscost
```

Python/C++ Q 向量 parity：MLP 一致；GCNN 最大误差约 `5.7e-6`，argmax 一致。阶段 8 主实验 2,454 次 RL action 全部合法，无崩溃或意外 fallback。

### 4.6 后续目标形态（C2–C6，未实现）

团队已明确不再堆“更长 GCNN-DQN + Prim mask”：

\[
\text{Score}(a)=S_{\text{SCIP}}(a)+\Delta_\theta(s,a)
\quad\text{仅当 }g_\theta(s)>\tau\text{ 才覆盖}
\]

SCIP `relpscost` 做主专家与安全 fallback；ML 只做浅层残差；最后才 multi-objective RL。C1.1 必须先修 SB teacher。

---

## 5. 已完成实验与结果

下面按时间/阶段列。**正式部署结论以阶段 8 为准；Prim 线是后续探索。**

### 5.1 阶段 1 Baseline（早期，30s，seed0）

9 真实实例 × default/relpscost/random/mostinf。多数 timeout。只证明 CLI/策略生效，不能当最终对比。

### 5.2 阶段 5 MLP pilot（不能当正式结论）

5000 gradient steps。`real_08` 固定 seed：random 截断节点 130.3，RL 109.3（约 -16.1% 节点，时间约 -1.7%）。seed 101 会退化。

### 5.3 阶段 7 GCNN pilot（不能当正式结论）

1000 steps，3 episodes。`real_08` node-limit 100：random 97 vs RL 81 节点（-16.5%），solving time 34.30 vs 32.80s（-4.4%）。
`real_09` C++：GCNN 8 nodes vs default 9，但推理占短实例 solving time 36–60%。阶段 8 没有维持这一优势。

### 5.4 阶段 8 正式实验（主结论）

- 协议：`controlled-bbmdp`（对齐训练）+ `production-scip`（对齐生产）
- 实例：`real_08/09` + transfer `real_01–05`
- 方法：default / relpscost / random / mostinf / RL-MLP / RL-GCNN（strong 仅 `real_09`）
- 5 seeds，30s，单线程
- 主实验 430 次 + 模型消融 50 + 深度消融 100 = 580
- 132 optimal / 298 time limit；全部 RL action 合法

**production-scip（35 instance-seed）**

| 方法 | solved | shifted gmean wall | mean PAR-2 | vs default paired speedup (95% CI) |
|---|---:|---:|---:|---|
| default | 25.7% | 21.329 s | 47.388 s | 1.000 |
| random | 31.4% | 20.615 s | 45.139 s | 1.077 (0.980, 1.196) |
| **mostinf** | **34.3%** | **20.444 s** | **44.078 s** | **1.109 (1.024, 1.221)** |
| RL-MLP | 28.6% | 21.301 s | 46.167 s | 1.021 (0.964, 1.089) |
| RL-GCNN | 37.1% | 23.984 s | 44.387 s | 0.966 (0.855, 1.095) |

要点：

1. MLP 相对 default 只快约 0.13%，CI 跨 1，远低于预设 5% 门槛。
2. GCNN solved 更高，但 wall 慢约 12.4%。
3. jointly solved 中，MLP/GCNN 分别比 default **多**约 6.8% / 19.9% 节点。
4. RL **没有明显优于 random**。controlled 下 vs random 的 time speedup：MLP 0.856、GCNN 0.793。
5. 活跃生产运行推理占比：MLP 0.10%，GCNN 9.39%（超过 5% 成本目标）。
6. `real_04` 全部卡在 presolve（约 27.4s），节点=0，**无法评估 branching**。GCNN 仅初始化就把 wall 从 34.72s 抬到 36.68s。
7. medium transfer：default 0/15，MLP 1/15，GCNN 3/15。有 solved 信号，但节点和墙钟不稳定。

**controlled-bbmdp**

| 方法 | solved/35 | shifted gmean wall | vs default |
|---|---:|---:|---|
| default | 8 | 21.327 s | 1.000 |
| random | 13 | 20.539 s | 1.143 |
| RL-MLP | 8 | 21.841 s | 0.978 |
| RL-GCNN | 9 | 24.407 s | 0.906 |

受控 jointly solved 中 RL 约少 4% 节点，但没有变成墙钟优势。

**模型消融**（各变体单训练 seed、1000 steps，评 `real_08/09`）

3-step 不优于 1-step；HL-Gauss 退化（8/10 vs 10/10）；去掉航空类别或全局特征差异约 1%，单 seed 无统计力。

**浅层混合（GCNN 只管前 D 层，其后 relpscost）**

D=5/10 损失 solved；D=20/50 也不如 unlimited。减去推理时间的模拟仍不改排序 → 问题不只是 forward，还有动态图提取、初始化和策略路径。

### 5.5 阶段 A：Prim 解码偏置（18 runs，全 optimal）

冻结旧 `gcnn/best_model_scripted.pt`，λ=0.5。实例 `real_09/08/01` × 2 seeds。

| method | wall_mean | nodes_mean |
|---|---:|---:|
| default | 19.14 s | 33.2 |
| rl-gcnn | 37.55 s | 46.0 |
| **rl-gcnn-prim λ=0.5** | **16.96 s** | **19.2** |

- vs 纯 GCNN：wall 几何约 **1.28× 更快**，节点约 1.58× 更少。
- vs default：整体几何约 0.77×（略慢），主要被小实例拖累。
- **`real_01` 最强**：GCNN 132/65 节点 → prim **14/12**；墙钟 115s/58s → **26s/23s**，且优于 default。
- `real_09` seed0：prim 可变差（8→35 节点）。

### 5.6 阶段 A 扩展：λ 扫描 + 门控（60 runs，全 optimal）

实例 +`real_03/05`。方法：default / gcnn / λ∈{0.25,0.5,1.0} / gated(λ=0.5, depth≥1, require S≠∅)。

| method | wall_mean | nodes_mean | vs gcnn wall |
|---|---:|---:|---:|
| default | **21.3 s** | 29.0 | — |
| rl-gcnn | 36.3 s | 35.7 | 1.00 |
| λ=0.25 | 40.1 s | 36.9 | 0.99× |
| **λ=0.5** | **31.0 s** | **27.9** | **1.05×** |
| λ=1.0 | 56.0 s | 56.6 | 1.03× |
| gated | 38.2 s | 38.9 | 0.92× |

硬实例子集 `{01,05,08}` 上 λ=0.5 墙钟几何 31.8s vs gcnn 41.3s。

跨实例翻转：

- `real_01`：λ≥0.5 极好；λ=0.25 不足；门控关掉 depth=0 后收益大减。
- `real_05` seed0：λ=0.5 → 147 nodes / 135s；λ=1.0 → **463 / 387s**（灾难）。
- 门控未修好小实例，还削弱 `real_01` → **不建议默认开**。
- 相对 default 仍未全面超越。

### 5.7 阶段 B：Prim 特征再训练（40 runs，全 optimal）

4×A100，3000 steps，选 seed0（val nodes 60）。评测：`real_09/08/01/05` × 5 方法 × 2 seeds。

| method | wall_mean | nodes_mean | vs 旧 gcnn wall |
|---|---:|---:|---:|
| **default** | **22.4 s** | 36.0 | — |
| 旧 rl-gcnn | 41.0 s | 44.4 | 1.00 |
| **A decode λ=0.5** | **33.5 s** | **34.6** | **1.08×** |
| B prim-feat | 56.9 s | 62.2 | 0.85× 更慢 |
| B feat+decode | 66.5 s | 73.2 | 0.71× 更慢 |

局部：`real_08@seed0` B 特征 69→26 节点。但 `real_01` 上 B 远弱于 A（246/34 vs 14/12）。
**结论：工程打通，pilot 再训未超过固定 decode；短期实用仍是旧 GCNN + λ=0.5。**

训练预算很薄：2 个实例、3000 step、`updates_per_env_step=8`、ε 按 gradient 衰减 → 有效交互少；normalizer warmup=8 过激进。

### 5.8 阶段 C0：Prim 拆解（30 runs exploratory，2 seeds，全 optimal）

焦点实例 `real_01/05/08`。自动 CLAIM：`TOPOLOGY_CONNECTIVITY`。

shifted-geomean wall vs gcnn：

| instance | gcnn | z-bias | root-z | full-prim | topology-only |
|---|---:|---:|---:|---:|---:|
| real_01 | 82.2s | 0.97× | 0.99× | **3.34×** | **3.36×** |
| real_05 | 45.8s | **2.26×** | 1.41× | 0.66× | 0.67× |
| real_08 | 21.6s | 1.61× | **1.68×** | 1.11× | 1.06× |

裁决：

- `real_01` 收益 **不是** empty-S 的 root z+0.5（root-z≈gcnn）
- 也 **不是** 全深度 z-family prior（z-bias≈gcnn）
- **是** 非空 S 上的 cut-edge / topology（topology≈full-prim；`real_01` 上 selected_bias 均值=1.0，几乎总在选割边 z）
- **不能全局部署**：同一套 topology 在 `real_05` 明显伤 wall
- `real_05` 的 Q top1–top2 margin 约 0.03，argmax 不稳定
- 因此禁止据此做 Stage C hard-mask `both_in`；必须做实例自适应 / confidence gate

C0 仅 2 seeds，终局判断前应 5 seeds 复验。

### 5.9 阶段 C1 ranking 采数（未通过）

pilot_v2：132 states，0 crash，但：

- 113/132 ≈ 85.6% 是 `pseudocost_fallback`
- 仅 19/132 ≈ 14.4% 标成 `sb`
- top1–top2 margin 大量≈0，hard top1 无意义
- `gate_label_quality=PASS` 只证明 pipeline 能跑，**不是** SB expert 质量通过

根因（已验证）：Ecole `StrongBranchingScores` 在该家族上 **没有真正做 SB LP**（二次 extract 的 NLPIterations delta=0），分数几乎全是地板 `1e-12`。PySCIPOpt 无 `startStrongbranch` 绑定，native 路径必须走 SCIP C++。

C1.1 teacher repair 是当前允许推进的阶段，门禁：SB valid-state ≥60% 才允许扩到 ≥20k 高质量 SB。**尚未宣布通过。** 其后的 cheap ranker / confidence gate 仍未做。DSU-Prime-GCNN 与 `real_04` 7200s 评测见 §5.10，不再走 C1→C6 那条线。

### 5.10 DSU-Prime-GCNN：`real_04` seed0 7200s 结论

这是当前 DSU 主线的第一份可比较 held-out 结果。口径按修订方案：主指标是固定预算下的 **final gap、PDI、首解时间、是否 optimal**；nodes 只作支撑，禁止单独定胜负。`real_04` 未参与 normalization、训练或选模。

#### 设定

| 项 | 值 |
|---|---|
| 实例 | `data/instances/transfer/real_04.cip`（326,502 vars / 5,168 int / 863,691 cons） |
| 协议 | `project-production-v1`，threads=1，`estimate` node selector |
| 方法 | `project-default`（relpscost）/ `random`（custom-random）/ **DSU-Prime-GCNN** |
| 时限 | 7200s，node_limit=-1 |
| seed | 0（未补 1/2） |
| 模型 | `artifacts/models/gcnn_formal_s0to3/seed0/best_model_scripted.pt`（Stage A 2400×16+0 + Stage B 600×12+4；在 `real_08` 上按 PDI/gap/PAR-2 选出，step=1040） |
| 配置 | `configs/experiments/real04_formal_seed0.json` |
| 产物 | `results/12_real04_formal_seed0_three_method/` |
| 并行 | workers=3，三路同时跑；相对排序有效，绝对墙钟可能慢于串行隔离跑 |

对照：`results/11_seed0_three_method_quicklook/` 是同一实例、同一协议、3600s、**R1 近乎未训权重**（约 2 次更新），不能与本表混称“正式训练结论”。

#### 7200s 主表（seed0）

| 方法 | 状态 | gap | PDI ↓ | 首解 (s) ↓ | 节点 | LP iter | 墙钟 (s) | 接管 | fallback | 非法 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **DSU-Prime-GCNN** | time_limit | 0.017% | **9.18e4** | **852** | 945 | 464k | 7204 | 1994 | 0 | 0 |
| project-default | **optimal** | **0** | 1.79e5 | 1789 | 2556 | 224k | 6689 | 0 | 0 | 0 |
| random | **optimal** | **0** | 3.02e5 | 2944 | 1992 | 395k | 6141 | 1947 | 0 | 0 |

相对 default：PDI 低约 **49%**，首解快约 **2.1×**。相对 random：PDI 低约 **70%**，首解快约 **3.5×**。三方法协议哈希一致（`scip_profile_sha256=91bdea83…`，`effective_search_params_sha256=3413ed82…`）。GCNN 推理 `rl_inference_total=59.4s`（占 solving time **0.82%**）；含构图的选择总耗时 `custom_selection_time_total=386s`（**5.4%**）。均低于 10% 开销门禁。

#### 和 3600s R1 quick-look 的差别（同一 `real_04` / seed0）

| 方法 | 3600s（R1 权重）gap / PDI / 首解 | 7200s（正式权重）gap / PDI / 首解 |
|---|---|---|
| project-default | 0.019% / 1.76e5 / 1758s | **optimal 0** / 1.79e5 / 1789s |
| DSU-Prime-GCNN | 3.03% / 1.97e5 / 1925s | 0.017% / **9.18e4** / **852s** |
| random | 3.87% / 2.95e5 / 2918s | **optimal 0** / 3.02e5 / 2944s |

正式 Stage A/B 权重把 GCNN 从“劣于 default、略好于 random”改成“PDI 与首解最好、gap 接近但未证优”。default 的 3600s 可行解与 7200s 证优用的是同一 primal 量级（约 3.07254 → 3.07196）。random 在 3600s 仍有 3.87% gap，多给的一小时里证到了最优——说明 **3600s 不够给 random/GCNN 下“证优”结论**，7200s 才分开了“找可行解”和“收紧证明”。

#### 可以下的结论

1. **工程门禁通过**：GCNN 真正接管 1994 次，非法候选 0，静默 fallback 0，无 OOM/NaN/crash，开销 <10%。旧 30s 协议上“`real_04` 未进 B&B、无法评 branching”这条，在 7200s 生产协议下已经作废。
2. **PDI / 首解：GCNN 明显更好。** 它更早给出可行解（852s vs 1789/2944），所以整段预算的 primal-dual 积分最低。这正是方案指定的主指标之一。
3. **证优 / 最终 gap：GCNN 尚未赢。** default 与 random 在时限内把 gap 收到 0；GCNN 打满 7200s 仍剩 0.017%。不能写成“已经可以替换 SCIP-default”。
4. **不能用节点数宣称胜利。** GCNN 945 节点 vs default 2556，但 default 已经最优；历史已有“节点更少、gap 更差”的反例，本表 gap 方向虽未恶化到那个程度，仍禁止用 nodes 单独定胜负。

#### 还不能下的结论

- 不能外推到 seed 1/2 或其它线束实例；方案要求三 seed 上 PDI 或 gap 相对 default **稳定**改善，目前只有 seed0。
- 不能根据本表在四个训练 seed 间重挑模型（`best_seed` 保持为 null；本评测固定用 seed0 的 `real_08` checkpoint）。
- `real_08` 选模信号弱：验证实例常在 1 个节点就最优，PDI 记成 inf，实际靠 PAR-2 打破平手。seed0 选出的是 step 1040，不是 3000 步 last 权重。
- workers=3 有 CPU 争用；若要严格 wall-time 论文数字，应 workers=1 重跑。

复现：

```bash
python scripts/run_final_experiments.py --config configs/experiments/real04_formal_seed0.json
```

---

## 6. 失败原因（30s / Prim 线，评估时应逐条审视）

下列 12 条主要针对阶段 8 与 Prim-A/B/C1，**不要直接套到 §5.10 的 DSU 7200s 单次结果上**。DSU 线已改：多真实实例训练、hybrid 奖励、生产协议统一、`real_04` 已进 B&B；仍缺的是多 seed 稳定性与证优。

1. **训练分布过窄**：正式 GCNN 实际 1 个实例；B 也只有 2 个。
2. **训练预算过小**：GCNN 1000–3000 gradient steps、单/四训练 seed，对高维图策略不够。
3. **目标错位**：reward / 选模看节点数，评测看 wall/gap；已有“节点更少但 gap 更差”的反例。
4. **截断漏洞**：`bootstrap_on_truncation=false` + time/node limit，可能奖励“节点少但每个节点很贵”。
5. **对手太强**：SCIP `relpscost` 自带 pseudocost / strong-branching 历史；RL 特征没有复现这些。
6. **协议偏移**：DFS 训练 vs 生产 `estimate` node selector。
7. **GCNN 成本**：每次 callback 重建动态大图；活跃运行推理约 9.4%；短实例 CUDA 冷启动可占主导。
8. **全树接管**：高优先级 + `max_depth=-1`，模型失败时才看到 relpscost，没有置信度门控。
9. **Prim 过粗且不可自适应**：固定 λ，跨实例可正可负。
10. **大实例曾不可评**：旧 30s 协议下 `real_04` 耗尽在 presolve。7200s 生产协议下已进入 B&B（§5.10）；当前缺口改成“单 seed、未证优”。
11. **教师信号失败**：C1 没采到真正的 SB ranking 标签。
12. **1 轮 mean MP**：图模型容量可能不足以表达连通扩张（C0 却显示连通先验有效），特征和网络可能不匹配。

---

## 7. 当前“最好可用”策略排序

| 优先级 | 策略 | 依据 |
|---|---|---|
| 1 `real_04` 研究候选 | **DSU-Prime-GCNN（正式 seed0 权重）** | 7200s 上 PDI/首解最好，开销合格；但未证优，仅 seed0 |
| 2 `real_04` 求解器基线 | **project-default / random** | 7200s 内均证到 optimal；default 的 PDI 仍差于 GCNN |
| 3 30s 中小实例研究折中 | 旧 rl-gcnn + Prim decode λ=0.5 | A / A-ext / B 中相对纯 GCNN 最稳；`real_01` 极强、跨实例会翻号 |
| 4 30s 求解器基线 | default / random / mostinf | 阶段 8 上 mostinf 墙钟最好 |
| 禁止当默认 | λ=1.0、当前 gated、hard-mask both_in | 灾难或未验证 |
| 未完成 | C1.1+ 残差排序 / gate；`real_04` seeds 1/2 | 缺高质量 teacher；缺多 seed 稳定性 |

预设最终验收（尚未达到）：vs default wall shifted-geomean ≥1.10×；vs random ≥1.05×；medium/hard paired win >60%；无不可控 >2× 变慢；推理+提取 < 总 wall 约 5–10%。

---

## 8. 请评估的问题（不要改写事实，只判断）

请从科研和方法论角度回答：

1. **总体路线是否合理？** 在“不改 MILP、只学 branching”的约束下，BBMDP + Double DQN + GCNN + C++ 部署这条链，哪些步骤必要，哪些过早？
2. **阶段 8 能否支持“RL 无效”？** 还是只能支持“当前 30s 预算/数据/奖励下无效”？最弱的统计环节曾是 30s、`real_04` 未进 B&B、训练实例=1。DSU 7200s seed0 之后，这些问题哪些已经关闭、哪些变成“单 seed 未证优”？
3. **Prim-A 是否构成有效结构先验？** C0 支持 topology connectivity，但跨实例符号翻转。固定 `Q+λ·Prim` 是否本质上不可部署？更合理的是可学习 α(s)、confidence gate，还是根本不该碰 hand-crafted Prim？
4. **阶段 B 失败说明什么？** 是“特征无用”，还是“在线 DQN + 2 实例 + 3000 step 学不会先验”？若重做，应 imitation / listwise ranking / 更长 RL / 改图编码器，哪条更合理？
5. **奖励设计是否致命？** 负节点增量 + 按节点选模 + 按墙钟验收，是否保证学偏？应改成什么（wall、PDI、SB-regret、推理代价）？
6. **C1.1→C2 残差排序是否对症？** `Score=S_SCIP+Δ_θ`、SB teacher、listwise、浅层覆盖，能否同时解决：数据窄、目标错位、relpscost 过强、Prim 不可自适应、GCNN 太贵？有没有更简单的对照（例如只学何时信任 Prim / 只在 z-cut 上加权）？
7. **GCNN 是否选错模型族？** 阶段 8 已显示 MLP 更便宜、GCNN 更慢；C0 又显示需要连通信息。静态 root encoder + cheap candidate MLP + DSU 连通特征，是否比动态二部图 GCNN 更匹配问题和部署约束？
8. **实验设计缺陷清单**：协议污染、2-seed C0、单训练 seed 消融、test 参与 λ 扫描、30s 切掉 branching 空间。哪些必须先修，才能再谈“方法无效”？
9. **若只能再做 1–2 个实验**，最有信息量的是什么？请给出可证伪假设、实例/协议、成功/失败标准。（当前最缺的是 `real_04` seeds 1/2 是否重复 PDI 优势。）
10. **对组会/论文的诚实表述**：哪些是工程贡献，哪些是阴性科学结果，哪些还不能下结论。

评估时请区分三层：**(A) 工程正确性**（30s 线与 DSU 7200s 均通过）、**(B) 科学信号**（Prim-A / C0 topology 有条件成立；DSU 在 `real_04` seed0 上 PDI/首解有信号）、**(C) 部署价值**（30s 线没有；DSU 7200s 单 seed 也不能替换 default，因为未证优）。
