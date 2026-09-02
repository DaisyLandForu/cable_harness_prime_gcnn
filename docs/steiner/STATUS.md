# Steiner RL Branching 迁移状态

更新时间：2026-09-02 UTC

## 当前状态

- 当前阶段：S00 研究契约与环境冻结
- 阶段状态：LOCAL_GATE_PASS（等待 GPT 只读审计；S01 仍未开始）
- base SHA：`88ade1ac614fb12f882a10ba9b5d35b15c7b4d01`
- 工作 branch：`research/steiner-s00-contract`
- 下一阶段：S01 未开始，只有 S00 本地 Gate 和 GPT 审计均通过后才可进入
- final test：`steiner-spg-final-test-v1` 已封存；learning runs = 0

## 阶段登记表

| 阶段 | 目标 | 本地 Gate | GPT 审计 | branch | commit |
|---|---|---|---|---|---|
| S00 | 研究契约与环境冻结 | PASS | NOT_RUN | `research/steiner-s00-contract` | pending content commit |
| S01 | 独立研究栈骨架 | NOT_STARTED | NOT_RUN | — | — |
| S02 | 数据解析与 MCF correctness | NOT_STARTED | NOT_RUN | — | — |
| S03 | Branchability 与资源审计 | NOT_STARTED | NOT_RUN | — | — |
| S04 | B0 二部图与动作映射 | NOT_STARTED | NOT_RUN | — | — |
| S05 | Strong-branch teacher 与 IL | NOT_STARTED | NOT_RUN | — | — |
| S06 | IL solve evaluation | NOT_STARTED | NOT_RUN | — | — |
| S07 | BBMDP 语义与 RL | NOT_STARTED | NOT_RUN | — | — |
| S08 | Dual-view | NOT_STARTED | NOT_RUN | — | — |
| S09 | Component 消融（可选） | NOT_STARTED | NOT_RUN | — | — |
| S10 | Steiner-family typed policy | NOT_STARTED | NOT_RUN | — | — |
| S11 | 胜出模型部署与 parity | NOT_STARTED | NOT_RUN | — | — |
| S12 | 冻结 benchmark | NOT_STARTED | NOT_RUN | — | — |
| S13 | 发布与论文证据包 | NOT_STARTED | NOT_RUN | — | — |

## 冻结入口

- 研究契约：`docs/steiner/RESEARCH_CONTRACT.md`
- 协议/seed/指标：`configs/steiner/experiments/protocols_v1.yml`
- split 规则：`configs/steiner/splits/split_policy_v1.yml`
- final-test selector：`configs/steiner/splits/final_test_v1.yml`
- 环境事实与决策：`configs/steiner/environment.lock.yml`
- S00 审计入口：`docs/steiner/phases/S00/S00_AUDIT_PACKET.md`（阶段收尾时生成）

## 已知阻塞与风险

1. 当前默认 `/usr/bin/scip` 是 9.2.2，不属于冻结栈。
2. SCIP 8.0.4 prefix 的 `scip`/`soplex` 当前 mode 为 `0644`；Ecole/PySCIPOpt bare import 缺少 `libscip.so.8.0` 搜索路径。S00 只记录，不修改用户环境产物。
3. 当前 shell 无 `nvidia-smi`，PyTorch 报告 CUDA unavailable；训练前必须重新探测。
4. final suites 的下载 archive/per-file content hashes 将由 S02 在不运行 final learned policy 的前提下补充；selector membership 已在 S00 冻结。
