# S04 Result Analysis

## S04 回答的问题

S04 不判断模型“聪不聪明”，只判断将来训练/部署会依赖的工程合同是否成立：

1. 标准 19/5/1 MILP 二部图能否从真实 SCIP 8.0.4 状态稳定取得；
2. SCIP 合法候选是否都能唯一回到原始 Steiner edge；
3. 为一次 message-passing 构造的候选闭包是否与 full graph 给出相同 logits；
4. 未训练 B0 是否 finite、可确定复现，并有基本 CPU 成本记录。

## 数据、协议与模型

- 数据：frozen synthetic generator 的 `sparse_erdos_renyi`，48 nodes、5 terminals、
  generator seed 100300。该 seed 属于 train split。
- SCIP：P1、solver seed 0、单线程；从真实 MCF B&B 轨迹保留前三个 branch states。
- 模型：未训练 B0，model seed 404，19/5/1，64 embedding、64 hidden、一轮 sum
  aggregation，共 68,161 parameters。
- action progression：每步选择当前候选中最小 `edge_id`，只为固定 snapshot
  trajectory，不代表 learned policy 或 baseline。
- validation/final test：均未访问；无训练和 checkpoint selection。

## 关键结果

| 指标 | 结果 | Gate |
|---|---:|---:|
| real branch states | 3 | ≥3 |
| legal candidates mapped | 31/31（100%） | 100% |
| full/closure max absolute logit error | 0.0 | ≤1e-5 |
| full/closure argmax agreement | 3/3（100%） | 100% |
| finite features/logits | 全部 finite | 必须 |
| parameters | 68,161 | 必须记录 |
| CPU timings | p50/p95 已记录 | 必须记录 |

每个 full state 是 817 constraints、981 variables、4,789 coefficient edges。
closure 分别缩到 44/99/132、40/90/120、40/90/120。full forward p95 约
13.27--13.35 ms，closure p95 约 1.54--1.58 ms；在这一台主机、这一张图和一个
未训练模型上，closure 大约快 8.6 倍。计时是环境描述，不是跨方法性能排名。

Ecole 在尚无 incumbent 时会给两列 incumbent features 产生 NaN。S04 的 v1
contract 仅将这两列 NaN 变为 zero sentinel，并继续拒绝所有其他 NaN/Inf。
因此 Gate 的 finite 条件是一个显式、可测试的输入约定，而不是悄悄清洗任意坏值。

## Gate 判断

`S04_GATE_SUMMARY.json` 的 7 个 checks 全为 true，本地 S04 Gate **PASS**：

- exact action mapping 100%；
- feature/logit finite；
- exact-closure logits 在容差内且 argmax 全一致；
- snapshot 数、parameter count、CPU timing 齐全。

这足以把 B0 state/model contract 交给审计，但不能自动进入 S05。按用户 waiver，
必须先完成 S00--S04 联合 GPT 只读审计并获得 PASS。

## 不能从结果推出什么

- 未训练 logits 没有 teacher label，不能证明 branching quality 或优于 SCIP 规则。
- 3 states/1 synthetic graph 只用于 deterministic engineering parity，不代表图族、
  规模或 OOD 泛化。
- CPU microbenchmark 不能预测整棵树 wall time，更不能与 S03 跨主机 timing 比较。
- S04 没有验证 strong-branch teacher、loss、训练稳定性或 GPU；这些属于 S05。
- S04 只覆盖 SPG edge actions，不能外推 vertex/arc/Steiner-family typed actions。
- SteinLib/DIMACS 再分发许可和旧航空 4 个 regression 失败均未由本阶段解决。

## Remediation v2 结论

首次 GPT 审计的 blocking concern 是“结果很可能正确，但 correctness 依赖未验证
的 list-order 假设”。v2 不再使用 list position：三个 frozen states 的所有
2,943 个 variable rows 均显式完成 `row == probindex` 双射后才允许继续动作映射。

| remediation 指标 | 结果 | Gate |
|---|---:|---:|
| probindex identity states | 3/3 | 全部 |
| probindex identity rows | 2,943/2,943 | 100% |
| legal actions mapped | 31/31 | 100% |
| full/closure max error | 0.0 | ≤1e-5 |
| full/closure argmax | 3/3 | 100% |
| snapshot content change | 无 | 必须可复现 |

因此 S04 remediation 本地 Gate 为 **PASS（8/8）**。这关闭了代码层面的 identity
缺口，但 GPT 尚未复审，所以审计状态仍不是最终 PASS。用户只授权随后准备 S05
代码和 tmux 命令；正式 teacher collection、训练与 S05 Gate 仍必须等待复审与
GPU 环境验收。
