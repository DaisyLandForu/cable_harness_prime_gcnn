# 阶段 A 汇报：Prim 解码偏置（Decode Bias）

**工作区**：`/data/hanchengcheng/hcc_1/du/cable_harness_prim_gcnn`  
**日期**：2026-08-08  
**状态**：已完成实现 + 编译 + 冒烟 + 紧凑矩阵评测；**等待你批准后再进入阶段 B**。

---

## 1. 本阶段做了什么

在**不改模型权重**的前提下，为 `rl-gcnn` 增加 Prim 风格解码偏置：

\[
\text{score} = Q + \lambda \cdot \text{PrimScore}
\]

- `λ=0`：基线 `rl-gcnn`
- `λ=0.5`：方法名 `rl-gcnn-prim`（本阶段默认试验值）

### PrimScore 约定（已实现）

从当前 LP/界推断已生长节点集 \(S_p\)（`z` 变量 `lb>0.5` 或 `LP>0.5`）：

| 候选变量 | PrimScore |
|---|---|
| `z` 割边（恰一端在 \(S\)） | +1.0 |
| `z` 两端都不在 \(S\) | +0.25 |
| `z` 两端都在 \(S\) | −0.5 |
| \(S\) 为空时的 `z` | +0.5 |
| `m` 落在 \(S\) 上 | +0.3 |
| `y` 落在 \(S\) 上 | +0.15 |

### 代码落点

| 位置 | 内容 |
|---|---|
| `python/rl_branching/prim_bias.py` | Python PrimScore / bias |
| `python/rl_branching/gcnn_dqn.py` | `stable_graph_argmax(..., lambda_prim=...)` |
| `src/rl/prim_bias.{hpp,cpp}` | C++ 同逻辑 |
| `src/rl/rl_gcnn_branchrule.cpp` | 推理时 `Q + λ·PrimScore` |
| `code/scip_tree.cpp` | CLI `--rl-prim-lambda`，JSON 字段 `rl_prim_lambda` |
| `configs/experiments/phaseA_prim_decode.json` | 评测配置 |
| `scripts/run_phaseA_prim_decode.sh` | 一键跑矩阵 |
| `tests/python/test_prim_bias.py` | 单元测试 |

---

## 2. 验证结果

### 2.1 工程验证

| 检查项 | 结果 |
|---|---|
| `make` 编译 `build/scip_tree`（SCIP prefix + Torch） | 通过 |
| CLI `--rl-prim-lambda` | 可见 |
| 冒烟：`real_09` + `λ=0.5`，60s | optimal，35 nodes，`rl_prim_lambda=0.5` 写入 JSON |
| `pytest tests/python/test_prim_bias.py` | **3 passed** |
| 矩阵 validation | **18/18 optimal**，`rl_prim_lambda` 字段与方法一致 |

### 2.2 评测矩阵

- 实例：`real_09` / `real_08` / `real_01`
- 方法：`default` / `rl-gcnn(λ=0)` / `rl-gcnn-prim(λ=0.5)`
- seeds：`0,1` → **18 jobs**，全部最优解
- 原始结果：`results/phaseA_prim_decode/`
- 汇总 JSON：`results/phaseA_prim_decode/phaseA_summary.json`

### 2.3 方法级均值（6 runs / method）

| method | solved | wall_mean (s) | nodes_mean |
|---|---:|---:|---:|
| default | 6/6 | 19.14 | 33.2 |
| rl-gcnn | 6/6 | 37.55 | 46.0 |
| **rl-gcnn-prim** | 6/6 | **16.96** | **19.2** |

成对几何平均（prim 相对对照）：

- wall：**prim vs rl-gcnn ≈ 1.28× 更快**
- wall：prim vs default ≈ 0.77×（整体略慢于 default，主要由小实例拖累）
- nodes：gcnn/prim ≈ 1.58（prim 节点更少）

### 2.4 分实例（关键信号）

**`real_01`（中等/迁移，本阶段最强正向信号）**

| seed | default nodes/wall | rl-gcnn | **rl-gcnn-prim** |
|---|---|---|---|
| 0 | 154 / 62.1s | 132 / 115.2s | **14 / 26.1s** |
| 1 | 22 / 28.2s | 65 / 58.2s | **12 / 23.0s** |

相对纯 GCNN：节点约 **7–10× 更少**，墙钟约 **2–4× 更快**，且优于 default。

**`real_08`**：prim 略好于纯 GCNN（nodes 69→52 @seed0），仍慢于 default。

**`real_09`（很小）**：prim 在 seed0 变差（nodes 8→35），seed1 三者均为 1 节点；小实例上偏置噪声更明显。

---

## 3. 结论（阶段 A）

1. **工程上可行**：C++/Python 偏置通路打通，编译与 18/18 矩阵无失败。
2. **科学上有条件成立**：在较难实例 `real_01` 上，Prim 解码偏置相对冻结 GCNN **显著减节点、降墙钟**；对已几乎 trivial 的小实例可能有害或中性。
3. **相对 SCIP default**：整体几何平均尚未全面超越；收益主要体现在“纯 GCNN 变差”的场景被纠偏。
4. **阶段 A 目标已达成**：证明“不改权重、只改解码”可产生可测差异，且方向与树状生长先验一致。

---

## 4. 建议的下一步（阶段 B，需你批准）

**阶段 B（显式邻域特征 + 再训练）** 建议在批准后推进：

1. 把 Prim/邻域结构做成 **GCNN 输入特征**（不只是 decode 后处理）。
2. 在现有 GCNN 上做浅层微调或模仿学习，让模型内生偏好割边扩展。
3. 扩大评测：更多 `real_*` + seeds，并扫 `λ∈{0.25,0.5,1.0}` 作为对照（A 的消融可并行保留）。

**可选并行小实验（仍属 A 收尾，若不批准 B）**：只扫 `λ` 与小实例门控（例如 depth>k 或 |S|>0 才开偏置），成本低。

---

## 5. 请你批准

请确认是否：

- **[A-OK]** 阶段 A 通过，进入阶段 B；或  
- **[A-extend]** 先扩展 A（λ 扫描 / 更多实例 / 小实例门控）；或  
- **[A-adjust]** 调整 PrimScore / λ 后再评一次。

在收到明确批准前，**不会启动阶段 B**。
