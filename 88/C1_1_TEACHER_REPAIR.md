# Phase C1.1 — Strong-Branching Teacher Repair

## 决策（冻结）

**暂停** `run_c1_ranking_wave1.sh`（30k 正式采集）。

pilot_v2 的 `gate_label_quality=PASS` 只证明 **pipeline_integrity**：

| 事实 | 含义 |
|------|------|
| 132 states，0 crash | 流程能跑 |
| 113/132 ≈ 85.6% `pseudocost_fallback` | **没有**高质量 SB expert |
| 19/132 ≈ 14.4% 标成 `sb` | 仍可能是非退化偶然，需复核 |
| top1–top2 margin 大量≈0 | hard top1 标签无意义 |

正式目标改为：

> **≥20,000 high-quality Strong-Branching labeled states**（可另收 weak/aux，须单独统计）

## 已知根因（已验证）

1. **Candidate semantics 正确**：Ecole `action_set` 与 `getLPBranchCands` 名称集合 154/154 对齐（`pseudo_candidates=False`）。
2. **Ecole `StrongBranchingScores` 未真正做 SB LP**：同一节点上 NodeBipartite-only 与 SB observation 的 `NLPIterations` 相同；二次 `extract` 的 LP iter **delta=0**。
3. 返回值几乎全是 product-score 地板 **`1e-12`**（非 NaN）→ 状态应记为 `strong_degenerate_floor` / `strong_invalid_score`，不是“没候选”。
4. 目标函数放大到 `1e9` 后仍全是 `1e-12` → **不是单纯 obj scale 问题**。
5. 当前 PySCIPOpt **无** `startStrongbranch` / `getVarStrongbranch*` 绑定 → native 路径走 **SCIP C++**。

## 门禁（C1.1 → 才允许 C1.2 wave）

| Gate | 要求 |
|------|------|
| pipeline valid | ≥99% |
| SB valid-state ratio | **≥60%（目标≥70%）** |
| SB finite candidate coverage | ≥95% |
| unexplained fallback | <5% |
| duplicate state_id | 0 |
| train contamination | 0 |
| margin 统计 | 必须输出（不要求全无 tie） |

## C2 预告（本阶段不实现训练）

- 主模型：**Root/Static GNN once + cheap Candidate MLP + topology + SCIP fallback**
- Loss：pairwise / listwise / soft-label（非 hard top1 CE）
- Pseudocost = **feature / weak_teacher**，不是主 expert
- Topology：`α_θ(s)·P(a|s)` 可正可负，禁止固定 `Q+0.5·Prim`

## 路线位置

```
C0 ✅ → C1.1（现在）→ C1.2 (≥20k SB) → C2 cheap ranker → C3 DSU → C4 gate → 评测 → C5 RL → C6 real_04
```
