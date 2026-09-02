# ADR 0001：SPG 使用 rooted MCF 作为基础 formulation

- 状态：Accepted
- 日期：2026-09-02
- 决策阶段：S00

## 背景

仓库现有 MILP 是航空布线模型，变量命名、拓扑副本和动作语义不能视为经典 SPG。研究首先需要可由小图穷举、known optimum 和独立 solution checker 交叉验证的 formulation。

## 决策

采用 `rooted_mcf_v1`：规范化后最小 terminal 为 root；每个无向 `edge_id` 对应二元 `stp_x_e########`；每个非 root terminal 和有向 arc 对应连续 flow。目标、流平衡和 (f^t_{ij}+f^t_{ji}\le x_e) 以研究契约为准。学习动作只允许 SCIP 当前合法 fractional binary `x_e`。

平行边按唯一 `edge_id` 区分。metadata 必须保存 original/canonical vertex ID、edge endpoints、variable name、root、terminals、source/content hash 和 formulation version。transformed variable 映射使用显式 metadata，不截取任意字符串猜测。

只有超过研究契约冻结的 flow count/build time/RSS 阈值，才登记并实现 SCF；不得静默替换。正权、连通性或支持 section 不满足时显式拒绝。

## 后果

- 优点：动作干净，连接性直观，便于 correctness；适合少 terminal 的 PACE Track 1。
- 代价：变量/约束随 \(|E||T|\) 增长，必须在 S03 做资源审计。
- S02 前没有任何“Steiner MILP 已实现”的主张；本 ADR 只冻结目标语义。

## 未采用方案

- 直接迁移航空 MCF：问题和变量语义不同。
- 一开始使用 SCF：规模更小但 LP 行为不同，会提前混入 formulation 因子。
- cut-based 或 SCIP-Jack 原生模型：适合作为专业 solver 路线，不适合作为首个自建 correctness 锚点。
