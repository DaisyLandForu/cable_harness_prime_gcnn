# Go / No-Go

| Checkpoint | 条件 | 失败动作 |
|------------|------|----------|
| C0 | 审计 + Prim 拆解 + parity | ✅ 已完成 |
| **C1.1** | SB valid-state ≥60%；finite cand ≥95%；unexplained fallback <5% | **禁止 wave1**；继续修 collector |
| **C1.2** | ≥**20k high-quality SB** states（weak PC 另计） | 未达标不得进 C2 最终训练 |
| C2 | ML SB-regret < SCIP SB-regret；listwise/pairwise | 先修数据/标签，不进 RL |
| C3 | topology 改善 validation | 只涨 train → 停扩模型 |
| C4 | gate 降低 catastrophic slowdown | 不得称 deployable |
| 最终 | 见 AGGRESSIVE_V2_METHOD 验收表 | 报告如实失败，不改协议粉饰 |

## 硬禁令

1. 不以最小节点数为唯一训练/选模目标  
2. 不 hard-mask `prim_both_in`  
3. 不用 test/transfer 调参  
4. 不删除 A/B/P0/P1 frozen baseline  
5. 结果不符假设时如实报告  
6. **不把 `gate_label_quality=PASS`（pipeline）当成 SB expert 质量通过**  
7. **不在 C1.1 未通过时执行 30k wave1**  
8. Pseudocost 只作 weak/aux feature，不作主 expert ground truth
