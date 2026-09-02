# ADR 0004：配对、多 seed、封存测试的分层评测

- 状态：Accepted
- 日期：2026-09-02
- 决策阶段：S00

## 背景

求解时间有 censored runs、重尾和 solver seed 变化；generic formulation 与 SCIP-Jack 的模型、预处理和 branching entity 也不同。单次均值或只看 solved runs 会产生误导。

## 决策

使用 `protocols_v1.yml` 的 P0--P4、固定 limits 和 seed。比较以 instance × solver seed 配对，formal 使用 5 solver seeds；学习模型逐个报告 5 training seeds。主指标顺序为 solved rate、PAR-2、PDI，诊断指标和失败计数不得省略。

不确定性用 10,000 次、instance-level、paired percentile bootstrap，95% CI，seed `20260902`。Wilcoxon/Holm 是补充。checkpoint 只用 validation lexicographic rule。superiority 主张要求更高优先级指标不恶化，并在首个有差异的主指标上给出 effect size 和 95% CI；不能用较低优先级 metric 覆盖 solved-rate 恶化。

final-test selector 由 `final_test_v1.yml` 的 SHA-256 冻结，S12 前不运行 learned policy。追加 final runs 标 post-hoc。P3 只比较完整 solver 指标；不比较 generic MCF/SCF 和 SCIP-Jack nodes。P4 hard subset 只按冻结 native SCIP-Jack 的 branchability 选取。

## 安全阈值

- correctness、invalid action、NaN、mapping failure 和 unexpected fallback 是先决 Gate，不能由统计收益抵消。
- 超过 2 倍 catastrophic slowdown 的比例必须报告；learned policy 相对 reference 增加超过 5 percentage points 即触发风险审查，95% 上界增加超过 10 points 时不得作稳定收益主张。
- 部署 callback overhead 不得超过 wall time 的 10%，否则必须报告为未通过部署预算。

## 后果

统计功效不足时结论是证据不足，不把 CI 跨零写成成功。S00 Gate PASS 只说明研究设计被冻结，不说明任何 learned baseline 有效。
