# Aggressive V2 Method：SCIP-Guided Structural Residual Branching

## 一句话

**SCIP relpscost 为主专家 + 结构残差排序 + 置信度门控 + 浅层接管 +（最后）多目标 RL。**

不是继续堆：GCNN-DQN + Prim mask + 更长训练。

## 阶段总览

| Phase | 内容 | Gate |
|-------|------|------|
| **C0** | instrumentation + Prim 拆解 + Python/C++ parity | ✅ 基本完成 |
| **C1.1** | **Strong-Branching Teacher Repair**（修标签，500–1000 probe） | SB valid-state **≥60%** 才允许扩数据 |
| **C1.2** | **≥20k high-quality SB** ranking states（+可选 weak PC，分开统计） | 非“30k total states” |
| **C2** | Static/root encoder + **cheap candidate MLP** + listwise/pairwise | ML SB-regret < SCIP SB-regret |
| **C3** | component-aware topology（DSU；α(s) 可正可负） | 只涨 train 不涨 val → 停扩 |
| **C4** | confidence gate + shallow override + relpscost fallback | 不能压灾难慢速 → 不可部署 |
| **C5** | multi-objective / tree-aware RL fine-tune | 仅 C2–C4 稳定且接近/超过 SCIP 后 |
| **C6** | real_04 专项（PDI/gap@budget / 可选 primal） | 不用 nodes 当核心 |

**禁止**：在 C1.1 未通过时跑 `run_c1_ranking_wave1.sh`。  
**禁止**：把 pseudocost fallback 当主 expert 扩到 30k。  
详见 [`C1_1_TEACHER_REPAIR.md`](C1_1_TEACHER_REPAIR.md)。

## C0 交付物

```
results/c0_audit/
  decision_logs.csv
  instance_summary.csv
  q_prim_scale_analysis.csv
  C0_AUDIT_REPORT.md

results/c0_prim_decomposition/
  raw_results.csv
  C0_PRIM_DECOMPOSITION.md

tests/python/test_prim_parity.py  (+ optional C++ snapshot tool)
```

## 最终形态（目标）

```
Score(a) = S_SCIP(a) + Δ_θ(s,a)
override only if g_θ(s) > τ
shallow depths preferred for ML
```

然后：`Imitation → Multi-objective RL`。

## 验收（最终）

- vs default wall shifted-geomean ≥ 1.10×
- vs random ≥ 1.05×
- medium/hard paired win > 60%
- 无不可控 >2× catastrophic slowdown
- gap 不系统性劣化
- inference+extract < 总 wall 约 5–10%
- real_04 看 fixed-budget gap/PDI
- test/transfer 绝不用于调参
