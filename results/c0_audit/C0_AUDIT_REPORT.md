# C0 Audit Report

## Scope

- Decision rows: **1263**
- Source glob: `results/c0_prim_decomposition/raw/**/*.branches.csv`

Joined raw results: `results/c0_prim_decomposition/raw_results.csv` (30 rows).

## Must-answer questions

- λ·|Prim| / Q_std 平均约为 **0.471**；Prim 量级超过 Q_std 的决策占比约 **26.1%**。
- real_01 决策 depth 分布（前几档计数）: `{0: 10, 1: 10, 2: 10, 3: 10, 4: 10}`。
- real_01 选中变量族: z=100.0%, m=0.0%, y=0.0%。
- real_05 决策数=445, depth_mean=45.42, depth_max=109。
- real_05 Q top1-top2 margin mean=0.0304 (极小 margin 可能对应不稳定 argmax)。

## Artifacts

- `results/c0_audit/decision_logs.csv`
- `results/c0_audit/instance_summary.csv`
- `results/c0_audit/q_prim_scale_analysis.csv`

## Next

- 对照 `C0_PRIM_DECOMPOSITION.md` 的 claim。
- 若 Prim 经常支配 Q（ratio>1），固定 λ=0.5 不可迁移——进入 residual/normalized score 设计。
- 灾难实例按 depth × family 切片后再看。
