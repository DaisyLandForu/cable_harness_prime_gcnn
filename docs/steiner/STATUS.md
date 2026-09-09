# Steiner RL Branching 迁移状态

更新时间：2026-09-09 UTC

## 当前状态

- 当前阶段：S05 formal-v3 confirmatory scientific Gate 和外部结果审计均
  **PASS**，S05 可按 formal-v3 完成验收。Formal-v1 的 527/640 quota failure 和
  formal-v2 的 25/30 lineage failure 均原样保留。
- formal-v1 仍因 train bucket quota 527/640 为 FAIL；formal-v2 的 640-state
  same-family selection 经 GPT `PASS / B1=CLOSED` 后执行。未重跑 teacher、未换
  graph/seed、未降低 Gate、未访问 test/final。
- 阶段状态：S04 remediation **GPT PASS**，B1 CLOSED；formal-v1/v2 FAIL retained；
  S05 formal-v3 **RESULT AUDIT PASS / BLOCKING FINDINGS NONE**。S06 online
  runner、冻结协议和六分片 custom-job scaffold 已实现并通过本地测试，正在准备
  pre-execution GPT 审计；formal solve 仍为 NOT_RUN，test/final 仍禁止访问。
- S04 base SHA：`931c7ae05c299c54bbdf59ecd458b64c7ca42282`。
- S04 content SHA：`d7a78a33151822f3a8a57fdc0224ede333583646`。
- S04 remediation v2 content SHA：
  `4ab54ffa2b80f06ac8a9ecfe662a04df7899b072`。
- S03 content SHA：`495d699cceefd243d4ab4c510be051f9df94833a`；phase SHA
  由 metadata commit 的 annotated local tag target 与最终 handoff 固定。
- 唯一活动 branch：`research/steiner-migration`。
- 正式实验：90/90 formal、10/10 ramp；44 optimal、46 timelimit；
  config SHA-256
  `cab4d8d96b02f427b8fedba6698cb9ee68b7e8a27059f8eaf319a0ede96ac1f1`。
- Gate：branchable fraction 70%；nontrivial median 127；strong valid 85%；
  all-tie 11.76%；mapping 100%；RSS p95 1,505.08 MB；全部 PASS。
- MCF/SCF：max flows 288,204、build p95 21.006 s、six-worker RSS projection
  9,030.49 MB；所有 SCF trigger 为 false。
- S04 B0：19/5/1、68,161 parameters；3 个真实 SCIP branch states、31/31
  candidates 映射；2,943/2,943 variable rows 通过 canonical probindex identity；
  full/closure 最大 logit 误差 0、argmax 3/3 一致；remediation Gate 8/8 PASS。
- S05 scaffold：12 个 preregistered pilot tasks、strong child-validity bridge、
  checksum shards、listwise IL/metrics/checkpoint reload、CPU/GPU foreground/tmux
  launchers和严格 seed-shard 汇总；87 passed、1 expected skip。training/GPU runs
  pilot-v2 teacher 为 12 completed、158 observed、149 valid、3,939/3,939
  mapped、train valid 85。v2 GPU exact-reload Gate FAIL；v3 tests 为 90 passed、
  1 expected skip；v3 mean regret 为 0.596116/0.496956/0.447868，全部 9 个
  runs 优于 random 0.685977 且 reload error 0。formal-v2 完整 Gate 为 FAIL。
- S05 formal-v3：240/240 teacher tasks terminal、3,383/3,840 valid、
  69,709/69,709 mapped；pre-model barrier 30 lineages/320 states PASS；五个冻结
  checkpoints 全部 reload error 0；primary effect 0.112697、95% CI
  `[0.057810, 0.167888]`、CV 0.016793，local Gate PASS。
- 资源：当前 S05 host 可见 128 GiB RAM 和 2 张 Tesla V100-SXM2 32GB；
  pilot-v2 已完成三次单卡 seed 作业。所有 SCIP solver workers 仍为单线程。
- final test：selector 106 entries、content lock 338 members；S03 未读取/求解，
  learning runs = 0。
- 下一步：固定 S06 implementation content head 并完成 pre-execution GPT 审计；
  只有 PASS 和独立 activation record 后，才可提交六个 main shard custom jobs。
  冻结的 test/final 不因 S05 PASS 自动解封。

## 阶段登记表

所有阶段均由 `research/steiner-migration` 累计承载；SHA/tag 是不可变审计身份。

| 阶段 | 目标 | 本地 Gate | GPT 审计 | content head | phase head / local tag |
|---|---|---|---|---|---|
| S00 | 研究契约与环境冻结 | PASS | NOT_RUN | `8b90375b6617a1ddcba34b872dbdbc11411cc042` | `a0bf0e3c1a702e1c85384f864defc86abbda29a5` / `steiner-s00-local-gate-v1` |
| S01 | 独立研究栈骨架 | PASS | NOT_RUN | `05b42791226347d31647547c344ef46c9dc4e87d` | `35a90ec5e52e2fad8301e3441ff6b286c7701d04` / `steiner-s01-local-gate-v1` |
| S02 | 数据解析与 MCF correctness | PASS | NOT_RUN | `19c7f46b91a1d05c46dbdeeba00bf863b37a7f5a` | `25be2e18c4020bed4cb8563618687b148d1f405f` / `steiner-s02-local-gate-v1` |
| S03 | Branchability 与资源审计 | PASS | NOT_RUN（waiver 至 S04 联合审计） | `495d699cceefd243d4ab4c510be051f9df94833a` | `bb6079b7844dcc42fed4976c812795c842d6411b` / `steiner-s03-local-gate-v1` |
| S04 | B0 二部图与动作映射 | PASS（v2 remediation） | PASS；B1 CLOSED | `4ab54ffa2b80f06ac8a9ecfe662a04df7899b072` | `030199703c6e280533f1f1c7cfc8d00d7df0a6b0` / `steiner-s04-audited-v2` |
| S05 | Strong-branch teacher 与 IL | formal-v3 PASS；v1/v2 FAIL retained | PASS；blocking findings NONE | `6cf7acab57525a744233ed3fdfd463f00fcd470c` | `6cf7acab57525a744233ed3fdfd463f00fcd470c` / `steiner-s05-audited-v3` |
| S06 | IL solve evaluation | IMPLEMENTATION_READY / FORMAL_NOT_RUN | NOT_RUN | — | — |
| S07 | BBMDP 语义与 RL | NOT_STARTED | NOT_RUN | — | — |
| S08 | Dual-view | NOT_STARTED | NOT_RUN | — | — |
| S09 | Component 消融（可选） | NOT_STARTED | NOT_RUN | — | — |
| S10 | Steiner-family typed policy | NOT_STARTED | NOT_RUN | — | — |
| S11 | 胜出模型部署与 parity | NOT_STARTED | NOT_RUN | — | — |
| S12 | 冻结 benchmark | NOT_STARTED | NOT_RUN | — | — |
| S13 | 发布与论文证据包 | NOT_STARTED | NOT_RUN | — | — |

## 冻结入口

- 研究契约：`docs/steiner/RESEARCH_CONTRACT.md`
- 主方案：`plans/STEINER_RL_BRANCHING_MIGRATION_MASTER_PLAN.md`
- 协议/seed/指标：`configs/steiner/experiments/protocols_v1.yml`
- S03 正式配置：`configs/steiner/experiments/s03_branchability_pilot_v1.yml`
- S04 B0 配置：`configs/steiner/models/b0_milp_gcnn_v1.yml`
- S05 active pilot 配置：`configs/steiner/experiments/s05_teacher_il_pilot_v3.yml`
- S05 formal pre-audit 配置：
  `configs/steiner/experiments/s05_teacher_il_formal_v1.yml`
- S05 formal concurrency amendment A1：
  `configs/steiner/experiments/s05_teacher_il_formal_v1_concurrency_a1.yml`
- S05 formal-v2 selection remediation：
  `configs/steiner/experiments/s05_teacher_il_formal_v2_selection_remediation.yml`
- S05 formal-v3 confirmatory Gate（已由独立 activation 授权 teacher execution）：
  `configs/steiner/experiments/s05_teacher_il_formal_v3_confirmatory_gate.yml`
- S05 formal-v3 frozen candidates：
  `configs/steiner/experiments/s05_formal_v3_candidate_graphs.json`
- split：`configs/steiner/splits/split_policy_v1.yml`
- final seal：`configs/steiner/splits/final_test_v1.yml`
- SCIP 8.0.4 入口：`scripts/steiner/run_with_scip804.sh`
- initial/resume resource：`configs/steiner/resource_preflight_20260903.yml`、
  `configs/steiner/resource_preflight_s03_resume_20260903.yml`
- Git 治理：`configs/steiner/git_governance_v1.yml`
- 公共数据政策：`configs/steiner/data_provenance_v1.yml`
- 旧航空 backlog：`docs/steiner/AVIATION_REGRESSION_BACKLOG.md`
- 当前 S05 审计入口：`docs/steiner/phases/S05/S05_AUDIT_PACKET.md`
- S05 最终 GPT 审计：`docs/steiner/audits/S05_GPT_AUDIT.md`

## 已知风险与边界

1. 默认系统 SCIP 可能是 9.x；Steiner 命令必须经过 8.0.4 wrapper。
2. 46/90 P1 tasks timelimit；P1 是 branchability 控制协议，不能宣传为 production
   performance。
3. 两个 large geometric 在 root LP timelimit 且没有分支，small/large community
   和 small bridge 也有低分支桶；后续只按 S03 建议范围采 teacher。
4. strong sample 仅预期 20 states、实际 17 valid；S05 仍必须独立验证 teacher
   quality 和 learning curve。
5. 正式 shards 跨 Gold 6148 和 Silver 4214 两个 CPU host；不得做 wall-time
   baseline 排名。固定资源 Gate 有很大安全余量。
6. 当前两张 V100 的 CUDA preflight 均通过；pilot-v2 暴露的是确定性配置缺口，
   不是显存或 CUDA 可见性问题。V3 每个作业仍必须独立执行 preflight。
7. SteinLib/DIMACS 未确认再分发许可；继续只提交官方 source/checksum，不提交 raw。
8. 旧航空 4 个既有失败未在 S03/S04 混改；首次 S00--S04 GPT audit 的
   CONDITIONAL PASS 已通过 S04 remediation 复审升级为 PASS。
9. S04 只在一个 synthetic-train 图的 3 个真实分支状态上验证工程 parity；它
   不能证明未训练模型有 branching 质量，也不能外推生产求解速度。
10. S05 formal-v1 的 sparse-large 与 geometric-medium strong labels 在冻结预算下
    严重不足；v2 即使通过也只能证明 family-balanced IL，不证明这些桶的规模泛化。
