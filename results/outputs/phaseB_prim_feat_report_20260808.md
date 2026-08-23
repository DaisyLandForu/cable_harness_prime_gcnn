# 阶段 B 汇报：Prim 邻域特征 + GCNN 再训练

**工作区**：`/data/hanchengcheng/hcc_1/du/cable_harness_prim_gcnn`  
**日期**：2026-08-08  
**状态**：训练与评测均完成；**等待你批准后再进入阶段 C 或调整 B**。

---

## 1. 本阶段做了什么

在变量特征后追加 6 维 Prim 邻域标志（ECOLE 19 → **25**），并用在线 GCNN-DQN 再训练：

| 特征 | 语义 |
|---|---|
| `prim_is_cut` | z 割边 |
| `prim_both_in` | z 两端已在 S |
| `prim_both_out` | z 两端都在 S 外 |
| `prim_grown_empty` | S 为空时的 z |
| `prim_m_on_grown` / `prim_y_on_grown` | m/y 落在已生长节点 |

训练：4×A100（GPU 1/2/4/5），seed 0–3，`batch=64`，`steps=3000`，数据 `real_06+real_07`。  
最优 seed：**seed0**（val nodes **60**）→ `artifacts/models/gcnn_prim_feat/best_model_scripted.pt`

部署开关：`--rl-prim-features 1`（旧 19 维模型仍可用 `0`）。

---

## 2. 训练结果

| seed | steps | best val nodes | 墙钟 |
|---|---:|---:|---:|
| **0（选用）** | 3000 | **60** | 97.9 min |
| 1 | 3000 | 75 | 96.8 min |
| 2 | 3000 | 63 | 95.8 min |
| 3 | 2760（早停） | 71 | 89.5 min |

---

## 3. 评测结果（40/40 optimal）

矩阵：`real_09/08/01/05` × {default, 旧gcnn, A解码λ=0.5, B特征, B特征+解码} × seeds 0–1

| method | solved | wall_mean | nodes_mean |
|---|---:|---:|---:|
| **default** | 8/8 | **22.4s** | 36.0 |
| rl-gcnn（旧） | 8/8 | 41.0s | 44.4 |
| **rl-gcnn-prim-decode（A）** | 8/8 | **33.5s** | **34.6** |
| rl-gcnn-prim-feat（B） | 8/8 | 56.9s | 62.2 |
| rl-gcnn-prim-feat-decode | 8/8 | 66.5s | 73.2 |

相对旧 GCNN 的成对几何 wall 加速（>1 更好）：

| method | vs gcnn |
|---|---:|
| A decode λ=0.5 | **1.08×** |
| B feat | 0.85×（更慢） |
| B feat + decode | 0.71×（更慢） |

### 分实例要点

- **`real_08@seed0`**：B 特征有局部收益（nodes 69→**26**，wall 39→21s）
- **`real_01`**：A 解码仍最强（14/12 nodes）；B 特征变差（246/34）
- **`real_05`**：B 与 A 都有方差，整体未稳定赢 default
- **`real_09`**：小实例上 decode 仍可能有害；纯特征接近旧 GCNN

---

## 4. 结论

1. **工程完成**：25 维特征 Python/C++ 对齐，4 seed 训练与 40-run 评测通过。
2. **科学结论偏负**：当前 pilot 级再训练的 Prim 特征 GCNN **尚未超过** 阶段 A 的固定 decode 偏置；整体也未超过 SCIP default。
3. **解读**：特征通路有效（`real_08` 有信号），但在线 DQN + 仅 2 个训练实例 + 3000 step **不够让模型内生稳定利用 Prim 信号**；硬叠 λ decode 反而更差。
4. **实用建议（短期）**：部署仍优先 **A：旧 GCNN + λ=0.5**；B 模型暂作研究基线。

---

## 5. 建议的下一步（请选）

按原计划：

1. **进阶段 C（推荐若继续结构先验）**  
   浅层 action mask：禁止/降权明显坏动作（如 `both_in` 的 z），比再堆 λ 更干净。

2. **B 加码再训**  
   加 `syn_*` / 更难实例、更长 step，或模仿学习（SB）暖启动后再 RL；成本更高。

3. **停结构线，固化 A**  
   以 λ=0.5 decode 为默认，转向别的优化（切平面/启发式/大实例 `real_04`）。

在收到明确批准前，**不会启动阶段 C**。
