# 88 — Aggressive V2：SCIP-Guided Structural Residual Branching

本目录是**下一阶段方法契约与操作手册**。仓库工作区：`cable_harness_prim_gcnn`。

## 核心判断（必须记住）

1. **Prim-A 是当前最好的 learned branching variant，但不是最好的 solver policy。**
2. 效果不好的主因不是“步数不够 / Prim 特征不够多”，而是：目标错位、数据不足、全树接管、Prim 过粗、train/serve 不一致、验证指标错位。
3. **禁止**直接做 Stage C = hard-mask `prim_both_in`。
4. **禁止**继续以最小节点数作为唯一训练/选模目标。
5. **禁止**用 test/transfer 调 λ / gate / 模型。
6. SCIP `relpscost` 必须成为主专家与安全 fallback，而不是模型失败时才出现。

## 文档索引

| 文件 | 内容 |
|------|------|
| [`AGGRESSIVE_V2_METHOD.md`](AGGRESSIVE_V2_METHOD.md) | 完整方法路线 C0–C6 |
| [`NOTES_AND_PITFALLS.md`](NOTES_AND_PITFALLS.md) | 当前代码级风险与注意点 |
| [`C0_RUNBOOK.md`](C0_RUNBOOK.md) | C0 命令（已完成） |
| [`C0_RESULTS_INTERPRETATION.md`](C0_RESULTS_INTERPRETATION.md) | C0 结果解读 |
| [`C1_RUNBOOK.md`](C1_RUNBOOK.md) | C1 旧采数说明（wave1 已暂停） |
| [`C1_1_TEACHER_REPAIR.md`](C1_1_TEACHER_REPAIR.md) | C1.1 方法说明 |
| [`C1_1_RUNBOOK.md`](C1_1_RUNBOOK.md) | **当前：C1.1 执行命令** |
| [`PHASE_C0_STATUS.md`](PHASE_C0_STATUS.md) | C0 脚本落地状态 |
| [`GO_NO_GO.md`](GO_NO_GO.md) | 阶段门禁 |

## 当前只允许执行

**Phase C1.1 — Strong-Branching Teacher Repair**  
（审计 + native SB probe；500–1000 修复后重采）。  
**不要**跑 `run_c1_ranking_wave1.sh`。
