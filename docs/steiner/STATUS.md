# Steiner RL Branching 迁移状态

更新时间：2026-09-03 UTC

## 当前状态

- 当前阶段：S04 canonical probindex identity remediation 已完成。
- 阶段状态：S04 remediation **LOCAL_GATE_PASS**；首次 GPT 联合审计为
  **CONDITIONAL PASS**，复审 PENDING。用户于 2026-09-03 明确授权在本地 Gate
  PASS 并推送后先准备 S05 源码与脚本；这不是审计 PASS，不授权正式 teacher、
  训练、validation/final 访问或 S05 Gate 判定。
- S04 base SHA：`931c7ae05c299c54bbdf59ecd458b64c7ca42282`。
- S04 content SHA：`d7a78a33151822f3a8a57fdc0224ede333583646`。
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
- 资源：正式运行和换机恢复环境都是 24.01-core cgroup/128 GiB RAM；恢复环境
  无 GPU。S03 CPU-only，未申请或使用训练资源。
- final test：selector 106 entries、content lock 338 members；S03 未读取/求解，
  learning runs = 0。
- 下一步：推送 remediation，准备 S05 implementation scaffold；随后从
  `docs/steiner/phases/S04/S04_AUDIT_PACKET.md` 做 GPT 复审。复审未 PASS 前不跑
  正式 teacher collection 或训练。

## 阶段登记表

所有阶段均由 `research/steiner-migration` 累计承载；SHA/tag 是不可变审计身份。

| 阶段 | 目标 | 本地 Gate | GPT 审计 | content head | phase head / local tag |
|---|---|---|---|---|---|
| S00 | 研究契约与环境冻结 | PASS | NOT_RUN | `8b90375b6617a1ddcba34b872dbdbc11411cc042` | `a0bf0e3c1a702e1c85384f864defc86abbda29a5` / `steiner-s00-local-gate-v1` |
| S01 | 独立研究栈骨架 | PASS | NOT_RUN | `05b42791226347d31647547c344ef46c9dc4e87d` | `35a90ec5e52e2fad8301e3441ff6b286c7701d04` / `steiner-s01-local-gate-v1` |
| S02 | 数据解析与 MCF correctness | PASS | NOT_RUN | `19c7f46b91a1d05c46dbdeeba00bf863b37a7f5a` | `25be2e18c4020bed4cb8563618687b148d1f405f` / `steiner-s02-local-gate-v1` |
| S03 | Branchability 与资源审计 | PASS | NOT_RUN（waiver 至 S04 联合审计） | `495d699cceefd243d4ab4c510be051f9df94833a` | `bb6079b7844dcc42fed4976c812795c842d6411b` / `steiner-s03-local-gate-v1` |
| S04 | B0 二部图与动作映射 | PASS（v2 remediation） | CONDITIONAL PASS；复审 PENDING | remediation content commit | remediation metadata / `steiner-s04-local-gate-v2` |
| S05 | Strong-branch teacher 与 IL | IMPLEMENTATION_ONLY（waiver） | NOT_RUN | — | — |
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
- 主方案：`plans/STEINER_RL_BRANCHING_MIGRATION_MASTER_PLAN.md`
- 协议/seed/指标：`configs/steiner/experiments/protocols_v1.yml`
- S03 正式配置：`configs/steiner/experiments/s03_branchability_pilot_v1.yml`
- S04 B0 配置：`configs/steiner/models/b0_milp_gcnn_v1.yml`
- split：`configs/steiner/splits/split_policy_v1.yml`
- final seal：`configs/steiner/splits/final_test_v1.yml`
- SCIP 8.0.4 入口：`scripts/steiner/run_with_scip804.sh`
- initial/resume resource：`configs/steiner/resource_preflight_20260903.yml`、
  `configs/steiner/resource_preflight_s03_resume_20260903.yml`
- Git 治理：`configs/steiner/git_governance_v1.yml`
- 公共数据政策：`configs/steiner/data_provenance_v1.yml`
- 旧航空 backlog：`docs/steiner/AVIATION_REGRESSION_BACKLOG.md`
- 当前联合审计入口：`docs/steiner/phases/S04/S04_AUDIT_PACKET.md`

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
6. 恢复服务器没有 GPU；S04 未使用 GPU，S05 首次训练前必须重新验收 CUDA。
7. SteinLib/DIMACS 未确认再分发许可；继续只提交官方 source/checksum，不提交 raw。
8. 旧航空 4 个既有失败未在 S03/S04 混改；首次 S00--S04 GPT audit 为
   CONDITIONAL PASS，S04 remediation 尚待复审。
9. S04 只在一个 synthetic-train 图的 3 个真实分支状态上验证工程 parity；它
   不能证明未训练模型有 branching 质量，也不能外推生产求解速度。
