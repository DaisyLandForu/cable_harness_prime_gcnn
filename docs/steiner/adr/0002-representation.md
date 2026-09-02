# ADR 0002：标准 MILP 二部图 B0 是表示基线

- 状态：Accepted
- 日期：2026-09-02
- 决策阶段：S00

## 背景

当前 `BipartiteGCNNQNetwork` 不是纯通用 B0：它拼接航空类别、Prim/DSU 六维、扩展 row/edge 和 global state。Steiner `stp_x_e*` 不满足旧 `z/m/y` 正则，直接复用会产生错误语义或零特征。

## 决策

S04 的 `milp_bipartite_v1` 固定为 19 维 Ecole variable、5 维 Ecole constraint、1 维 normalized coefficient edge，一轮 variable→constraint→variable message passing。模型只为合法 `x_e` candidates 输出 logits；normalization 仅来自 train。

候选 exact two-hop closure 只有在一轮传播且 full/closure logits 最大误差不超过 `1e-5`、argmax 100% 一致后使用。第二轮传播必须扩大闭包、使用 full graph 或把采样登记为新实验因子。

原始 Steiner graph encoder 和 late dual-view fusion 是 S08 的独立方法变量。component/DSU 是 S09 的可选 soft feature，不 hard-mask action。任何增强使用新 schema ID，不能悄悄改变 B0。

## 后果

- B0 能直接回答通用 MILP signal 是否存在，并与文献基线对齐。
- 旧航空代码保持原行为，新 Steiner 栈不通过隐藏 problem-type 分支复用它。
- typed edge/arc/vertex candidates 延至 S10，在那之前不部署 `x_e`-only C++ 接口。

## 未采用方案

- 把旧 25 维变量输入直接当 Steiner baseline。
- 先实现 dual-view/cross-attention 再建立 B0。
- 把原始图 GNN 当作 LP/MILP state 的替代品。
