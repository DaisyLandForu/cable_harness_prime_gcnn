# Phase C0 实现状态（脚本已就绪，等待你跑实验）

## 已认可的路线

**SCIP-Guided Structural Residual Branching**（C0→C6）。  
**禁止**立刻做 Stage C `both_in` hard-mask。  
当前只推进 **Phase C0**。

## 已落地代码

| 项 | 状态 |
| --- | --- |
| `--rl-bias-mode none\|z\|root_z\|prim\|topology` | ✅ C++/CLI/runner |
| C0 branching CSV instrumentation | ✅ `rl_gcnn_branchrule.cpp` |
| Python grown-set `lb\|\|lp` + bias modes | ✅ `prim_bias.py` |
| unit/parity tests | ✅ 9 passed |
| `configs/experiments/c0_prim_decomposition.json` | ✅ |
| `scripts/run_c0_prim_decomposition.sh` | ✅ |
| `scripts/analyze_c0_*.py` | ✅ |
| 计划/注意点文档 | ✅ `88/*.md` |

## 你需要执行的命令

见 `C0_RUNBOOK.md`（推荐整段复制）。预计探索矩阵约 30 runs（3 instances × 5 methods × 2 seeds），medium 实例 time_limit 300–600s。

## Gate

跑完后阅读：

1. `results/c0_prim_decomposition/C0_PRIM_DECOMPOSITION.md` 的 **CLAIM**
2. `results/c0_audit/C0_AUDIT_REPORT.md`

若 CLAIM = `ROOT_Z_FAMILY_PRIOR`，必须如实写入后续方法设计，不得继续假装“Prim connectivity 已验证”。

**通过 C0 前不要启动 C1 大规模采数。**
