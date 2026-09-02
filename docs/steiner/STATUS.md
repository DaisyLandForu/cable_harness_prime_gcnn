# Steiner RL Branching 迁移状态

更新时间：2026-09-02 UTC

## 当前状态

- 当前阶段：S02 数据解析与 MCF correctness
- 阶段状态：LOCAL_GATE_PASS（内容提交完成；S03 未开始）
- base SHA：`35a90ec5e52e2fad8301e3441ff6b286c7701d04`
- 工作 branch：`research/steiner-s02-formulation`
- S00 远端：`origin/research/steiner-s00-contract` 已核实指向 `a0bf0e3c1a702e1c85384f864defc86abbda29a5`
- 治理偏差：S00 GPT 审计仍为 NOT_RUN；用户明确要求继续 S01--S02，未把缺失审计改写为 PASS
- S01：本地 Gate PASS，phase head `35a90ec5e52e2fad8301e3441ff6b286c7701d04`
- 下一阶段：本次请求止于 S02；未开始 S03
- final test：selector 106 entries；content lock 338 members；只做 byte hash；
  learning runs = 0

## 阶段登记表

| 阶段 | 目标 | 本地 Gate | GPT 审计 | branch | commit |
|---|---|---|---|---|---|
| S00 | 研究契约与环境冻结 | PASS | NOT_RUN | `research/steiner-s00-contract` | `8b90375b6617a1ddcba34b872dbdbc11411cc042` |
| S01 | 独立研究栈骨架 | PASS | NOT_RUN | `research/steiner-s01-scaffold` | `35a90ec5e52e2fad8301e3441ff6b286c7701d04` |
| S02 | 数据解析与 MCF correctness | PASS | NOT_RUN | `research/steiner-s02-formulation` | `19c7f46b91a1d05c46dbdeeba00bf863b37a7f5a` |
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
- S00 审计入口：`docs/steiner/phases/S00/S00_AUDIT_PACKET.md`
- S01 审计入口：`docs/steiner/phases/S01/S01_AUDIT_PACKET.md`
- S02 审计入口：`docs/steiner/phases/S02/S02_AUDIT_PACKET.md`

## 已知阻塞与风险

1. 当前默认 `/usr/bin/scip` 是 9.2.2，不属于冻结栈。
2. SCIP 8.0.4 prefix 的 `scip`/`soplex` 当前 mode 为 `0644`；Ecole/PySCIPOpt bare import 缺少 `libscip.so.8.0` 搜索路径。S00 只记录，不修改用户环境产物。
3. 当前 shell 无 `nvidia-smi`，PyTorch 报告 CUDA unavailable；训练前必须重新探测。
4. final archive/member content hashes 已由 S02 byte-only 锁定；SteinLib/DIMACS
   来源未提供可确认的显式 redistribution license，raw data 不得随仓库发布。
5. 旧航空 regression 当前有 4 个既有失败（2 个 Prim 语义断言、2 个 build
   可执行权限）；S01/S02 未修改旧源码，详见 S01/S02 test report。
6. PACE odd correctness 样例在 1 node 求解；尚不能证明 branchability，必须由
   S03 预注册 Gate 判断。
