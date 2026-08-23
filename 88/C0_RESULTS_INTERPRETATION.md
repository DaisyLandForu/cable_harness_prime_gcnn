# C0 结果解读（2026-08-08 exploratory, 2 seeds）

30/30 optimal。自动 CLAIM=`TOPOLOGY_CONNECTIVITY` **对 real_01 成立**，但对全集必须加限制条件。

## 一句话

> **real_01 的 Phase-A 收益不是 empty-S 的 root z+0.5，而是非空 S 上的 cut-edge / topology 打分；同一套 Prim/topology 在 real_05 上会明显伤 wall-time。**

## 关键数字（shifted-geomean wall vs gcnn）

| instance | gcnn | z-bias | root-z | full-prim | topology-only |
|----------|------|--------|--------|-----------|---------------|
| real_01 | 82.2s (1.00×) | 0.97× | 0.99× | **3.34×** | **3.36×** |
| real_05 | 45.8s (1.00×) | **2.26×** | 1.41× | 0.66× | 0.67× |
| real_08 | 21.6s (1.00×) | 1.61× | **1.68×** | 1.11× | 1.06× |

## 对原假设的裁决

| 假设 | 裁决 |
|------|------|
| real_01 收益 ≈ root-z / empty-S z prior | **否定**（root-z≈gcnn） |
| real_01 收益 ≈ 全深度 z-family prior | **否定**（z-bias≈gcnn） |
| real_01 收益 ≈ Prim connectivity（去掉 empty-S prior 后仍在） | **支持**（topology≈full-prim） |
| Prim 可全局部署 | **否定**（real_05 seed0：prim/topo ~135s vs z-bias ~17s） |

## Audit 要点

- 1263 次 branching decisions（validator 里 `rl_branch_decisions: 0` 是字段汇总 bug，以 `branch_decisions` / CSV 为准）。
- λ·\|bias\| / Q_std 均值 ~0.47；约 26% 决策 Prim 量级超过 Q_std → 固定 λ=0.5 仍有尺度问题。
- real_01 上 full-prim/topology 的 `selected_bias` 均值 = **1.0**（始终在选 cut-edge），且几乎全选 z。
- real_05 Q top1–top2 margin 很小（~0.03），argmax 不稳定，与灾难 seed 一致。

## 对后续路线的含义

1. **不要** Stage C hard-mask `both_in`。
2. **不要**把 empty-S +0.5 当成 real_01 的主故事；topology encoder / component-aware features 仍值得做。
3. **必须** confidence gate：同一先验在不同实例可正可负。
4. C0 exploratory 仅 2 seeds；终局判断前建议 5 seeds 复验（尤其 real_05）。
5. C0 gate：诊断目标已达成 → 可讨论进入 C1 数据重建，但仍禁止用 test/transfer 调参。
