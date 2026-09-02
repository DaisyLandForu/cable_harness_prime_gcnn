# ADR 0003：Strong-branch IL 初始化，RL 只验证整树增量

- 状态：Accepted
- 日期：2026-09-02
- 决策阶段：S00

## 背景

branching 动作延迟影响整棵树，rollout 昂贵且候选集合变化。从零 DQN 容易把算力花在无效探索；仓库已有名为 BBMDP 的代码也不能仅凭名称视为论文语义复现。

## 决策

主要学习路线是：合法 `x_e` strong-branch teacher → B0 listwise/ranking imitation → frozen best-validation IL 初始化 RL。from-scratch RL 使用相同架构和预算，只作为初始化消融。

S07 逐轨迹核对 state、action、transition、reward、return、natural terminal 和 truncation。在通过之前名称固定为 `branching_dqn_v1`；通过后才可称 `BBMDP-style`。Double DQN、target network、n-step 和 PER 均为显式开关，每项需要单元测试；它们不是因为旧仓库已有就自动启用。

validation solve 的 solved rate/PAR-2/PDI lexicographic rule 选择 checkpoint。TD loss 只作诊断。所有 training seed、失败 rollout、truncation、NaN、invalid/mapping/fallback 均计入报告。

## 停止条件

- teacher valid/tie Gate 失败：修 collector/定义，不用低质量 pseudocost 掩盖。
- IL 不优于 random/most-infeasible：不进入长期 RL。
- 语义审计无误但 RL 不优于 IL：接受 RL 负结果并停止扩规模。
- 多 seed 不稳定：不得只保留最好 seed。

## 未采用方案

- 大规模 pure DQN from scratch 作为主线。
- 以训练 loss 或 final test 选择 reward/checkpoint。
- 在 representation、reward 和数据同时变化时归因 RL 增益。
