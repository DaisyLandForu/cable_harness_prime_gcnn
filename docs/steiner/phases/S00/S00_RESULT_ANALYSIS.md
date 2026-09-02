# S00 Result Analysis

## 实验问题

S00 不回答 learned branching 是否有效。它回答的是：在运行任何 Steiner learned policy 前，问题、动作、数据边界、协议、seed、baseline、指标、统计和禁止比较项是否已形成可审计契约。

## 数据和 split

- 没有下载或读取 benchmark instance。
- Synthetic split 以 base-graph lineage 分组，使用五段互斥 seed ranges。
- PACE Track 1/2 odd 为 development，even 为 final。
- SteinLib development families：B、C、I080、I160、1R、ALUE；final families：D、E、I320、2R、DIW。
- DIMACS 11 official SPG bundle 为 final。
- final selector 展开 106 entries，SHA-256 为 `8c0324c1a82485c2187825977fe2807e31512a6435e2f58f6a1d17babbfbddd1`。
- final learned runs：0；result artifact：空。

## Baseline、seed 和资源限制

baseline、5 个 formal solver seeds、5 个 training seeds、3 个 teacher seeds、bootstrap seed，以及 P0--P4 limits 均已写入 `protocols_v1.yml`。S00 没有实际 baseline run，因此没有 time/nodes/PDI 样本或置信区间。

## 关键核实结果

1. 当前仓库不存在 Steiner 独立栈；主方案列出的接口确属目标接口。
2. 旧 GCNN/observation 是航空耦合输入，不可当作 B0 已实现。
3. 冻结目标环境可识别为 SCIP 8.0.4 / SoPlex 6.0.4 / Ecole 0.8.1 / PySCIPOpt 4.3.0 / PyTorch 2.5.1，但当前 shell 需要修复 executable mode/loader path 才能用于后续正式运行。
4. 默认系统 SCIP 是 9.2.2，必须由 run manifest 拒绝，不能与 8.0.4 数据混合。
5. 当前 GPU 不可见；S00 不把历史 4×V100 日志当作当前硬件事实。

## 异常和失败

- prefix direct execute 与 bare Python imports 失败，已作为显式前置条件保留，没有通过修改用户 artifacts 掩盖。
- GPU probe unavailable，后续 CUDA 训练必须重新检查。
- 没有删样本、降 Gate 或用 final test 调参。

## Gate S00 分析

| Gate 条件 | 证据 | 结论 |
|---|---|---|
| SPG/MCF/root/`x_e` action 无歧义 | contract §1；ADR 0001；protocol action contract | 满足 |
| P0--P4 limits 和 seeds 固定 | `protocols_v1.yml` + unit test | 满足 |
| split/final 封存规则固定 | split/final manifests + unit test | 满足 |
| final list 有 hash 且未运行 | 106 entries；SHA；learning runs 0 | 满足 |
| baseline/metrics/statistics 固定 | contract §6；ADR 0004；protocol config | 满足 |
| 禁止比较和主张边界固定 | contract §7；protocol prohibited list | 满足 |
| 环境事实完整且不伪装缺失 | environment lock；失败探测记录 | 满足 |
| 四 ADR/状态/过程文档完整 | `docs/steiner/**` | 满足 |
| 自动化测试和 Git 卫生 | 8/8 contract tests；staged diff check exit 0；仅 S00 allowed paths | 满足 |

**本地 Gate S00：PASS。** 环境问题不改变 S00 的“记录与冻结”正确性，但它们是后续执行 solver 的真实风险，必须在首次 solver integration 前关闭。S01 未开始。

## 不能推出的结论

- 不能推出 MCF builder、parser 或 solution checker 正确；它们尚未实现。
- 不能推出所选实例具有 branchability。
- 不能推出 B0、IL、RL、dual-view 或 component 有任何收益。
- 不能推出 SCIP 8.0.4 当前已可直接运行正式矩阵。
- 不能从 SPG 契约推出 RPCSTP/PCSTP 或 Steiner-family 泛化。
