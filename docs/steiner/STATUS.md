# Steiner RL Branching 迁移状态

更新时间：2026-09-03 UTC

## 当前状态

- 当前阶段：S02 数据解析与 MCF correctness；尚未执行 S03 pilot
- 阶段状态：LOCAL_GATE_PASS；S03/S04 资源前检及运行防护已完成
- S02 base SHA：`35a90ec5e52e2fad8301e3441ff6b286c7701d04`
- 唯一活动 branch：`research/steiner-migration`
- 长期分支起点：S02 phase head
  `25be2e18c4020bed4cb8563618687b148d1f405f`
- 治理 content commit：`9b0bd862178b03c388714599c13db21fc5e59dee`
- 远端长期分支：已存在；2026-09-03 只读核实
  `origin/research/steiner-migration` 指向
  `1dfb73e0abed9e6825fc2fc0f7720085ca2ed4ff`
- checkpoint tags：三个 `steiner-s00/s01/s02-local-gate-v1` 已在本地创建，
  且远端均已存在；GPT audit 状态均为 NOT_RUN
- 治理策略：master plan v1.3；阶段审计使用不可变 SHA/range/tag，不再创建
  远端阶段分支
- 研究契约：v1.2；新增 SCIP wrapper、资源放量和数据发布边界，预注册 Gate
  数值未改变，修订待 GPT 审计
- S00 远端：`origin/research/steiner-s00-contract` 已核实指向 `a0bf0e3c1a702e1c85384f864defc86abbda29a5`
- 旧 S00/S01/S02 远端分支：保留为只读历史指针，不删除、不续写、不改写
- 治理偏差：S00 GPT 审计仍为 NOT_RUN；用户明确要求继续 S01--S02，未把缺失审计改写为 PASS
- S01：本地 Gate PASS，phase head `35a90ec5e52e2fad8301e3441ff6b286c7701d04`
- 下一阶段：S03；只能从 1 worker 开始资源/branchability pilot，未开始正式运行
- final test：selector 106 entries；content lock 338 members；只做 byte hash；
  learning runs = 0

## 阶段登记表

所有阶段均由 `research/steiner-migration` 累计承载；表中 SHA/tag 是审计身份。

| 阶段 | 目标 | 本地 Gate | GPT 审计 | content head | phase head / local tag |
|---|---|---|---|---|---|
| S00 | 研究契约与环境冻结 | PASS | NOT_RUN | `8b90375b6617a1ddcba34b872dbdbc11411cc042` | `a0bf0e3c1a702e1c85384f864defc86abbda29a5` / `steiner-s00-local-gate-v1` |
| S01 | 独立研究栈骨架 | PASS | NOT_RUN | `05b42791226347d31647547c344ef46c9dc4e87d` | `35a90ec5e52e2fad8301e3441ff6b286c7701d04` / `steiner-s01-local-gate-v1` |
| S02 | 数据解析与 MCF correctness | PASS | NOT_RUN | `19c7f46b91a1d05c46dbdeeba00bf863b37a7f5a` | `25be2e18c4020bed4cb8563618687b148d1f405f` / `steiner-s02-local-gate-v1` |
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
- Git 治理：`configs/steiner/git_governance_v1.yml`
- 单分支决策：`docs/steiner/adr/0005-single-branch-git-governance.md`
- SCIP 8.0.4 入口：`scripts/steiner/run_with_scip804.sh`
- S03/S04 资源前检：`configs/steiner/resource_preflight_20260903.yml`
- 公共数据来源政策：`configs/steiner/data_provenance_v1.yml`
- 旧航空独立排期：`docs/steiner/AVIATION_REGRESSION_BACKLOG.md`
- S00 审计入口：`docs/steiner/phases/S00/S00_AUDIT_PACKET.md`
- S01 审计入口：`docs/steiner/phases/S01/S01_AUDIT_PACKET.md`
- S02 审计入口：`docs/steiner/phases/S02/S02_AUDIT_PACKET.md`

## 已知阻塞与风险

1. 当前默认 `/usr/bin/scip` 是 9.2.2，不属于冻结栈；所有 Steiner 命令必须经
   `run_with_scip804.sh`。wrapper 已实测 SCIP 8.0.4/PySCIPOpt 4.3.0/Ecole 0.8.1，
   prefix 的 0644 artifact 未被修改。
2. cgroup 可见 48 个 logical CPU，但实际 quota 约 8.01 cores；RAM 上限 65,537
   MiB 且无 swap。S03 可条件启动，但必须按 1 → 3 → 6 worker 放量，不能直接
   以 6×8 GiB 顶到内存边界。
3. 当前无 GPU；S03/S04 不需要 GPU。首次 CUDA teacher/IL 训练前必须重新申请
   并探测 GPU。
4. final archive/member content hashes 已由 S02 byte-only 锁定；SteinLib/DIMACS
   raw data 继续只从官方源下载并验 checksum，未确认再分发许可前不得公开打包。
5. 旧航空 regression 当前有 4 个既有失败，已登记为独立 backlog；不得在 S03
   commit range 中混改。
6. PACE odd correctness 样例在 1 node 求解；尚不能证明 branchability，必须由
   S03 预注册 Gate 判断。
7. S00--S02 GPT audit 均为 NOT_RUN；local-gate tag 不能当作 audited tag。
8. master plan v1.3 / research contract v1.2 的治理与 S03 前置修订尚未做 GPT 审计。

远端 branch/tag 已由用户完成首次发布；后续只允许在本地 Gate PASS 后对长期
分支做 fast-forward `git push`，不得使用 `--force`。
