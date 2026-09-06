# S05 Formal Protocol v1 — GPT pre-execution audit request

请只读审计 S05 formal protocol，不修改仓库，也不要把 pilot PASS 误判为完整
S05 PASS。

## 固定审计对象

- branch：`research/steiner-migration`
- protocol base：`5cd494a631c1f06c20659a7229937ac5b59a8fb9`
- protocol content head：`d1717a7ecb6043efd71678175a92325ac9ff4208`
- substantive range：
  `5cd494a631c1f06c20659a7229937ac5b59a8fb9..d1717a7ecb6043efd71678175a92325ac9ff4208`
- protocol YAML：
  `configs/steiner/experiments/s05_teacher_il_formal_v1.yml`
- protocol explanation：
  `docs/steiner/phases/S05/S05_FORMAL_PROTOCOL.md`
- protocol YAML SHA-256：
  `c843f76d69c07b1c8a8093ab6f1426656084b2de2f4e9e0d777f1af32cd4c811`
- protocol explanation SHA-256：
  `6b121bb215ad9ad5444ce2c8328dc7049b3abdfd0a7bf191fec888441ec572c9`
- S05 current evidence entry：`docs/steiner/phases/S05/S05_AUDIT_PACKET.md`
- pilot machine summary：
  `docs/steiner/phases/S05/S05_PILOT_GATE_SUMMARY.json`
- master plan：`plans/STEINER_RL_BRANCHING_MIGRATION_MASTER_PLAN.md`
- research contract：`docs/steiner/RESEARCH_CONTRACT.md`

## 审计目的

这不是结果复审，而是 formal 实验运行前的预注册审计。当前
`execution_authorized: false`，没有运行 formal teacher、formal training 或访问
test/final。只有本次返回 `PASS` 后，才允许补实现 formal loader/launchers、测试，
再按原协议运行；任何实现差异都必须重新登记。
PASS 后通过单独提交的审计结果文件引用 protocol SHA-256 来激活，不回写或覆盖
本次接受审计的 YAML。

## 请重点判断

1. pilot learning curve 是否足以支持把 formal train budget 预注册为 640 states，
   且五个 family 各 128，而不把它夸大成饱和或领域定律；
2. 105 个 base graphs、315 个 teacher tasks、固定 seed/bucket 和 quota 是否覆盖
   S03 建议范围，并明确禁止失败后换实例；
3. `train`、`validation_select`、`validation_gate` 是否真正按 base-graph lineage
   隔离，避免同一 validation 既选 checkpoint 又做显著性 Gate；
4. 五个 formal seeds `[101,202,303,404,505]` 是否必须全部从头训练、全部保留，
   双 V100 仅做独立 seed 并发而不会改变单 seed 语义；
5. 以 30 个 `validation_gate` base graphs 为 paired bootstrap unit、10,000 次、
   seed 20260902，以及 all-seed/all-family direction 和 CV <= 0.15 Gate 是否
   定义清楚、没有把相关 state 当独立样本；
6. teacher 60%/40%/100% Gate、quota、exact reload 0.0、失败保留和禁止
   test/final 是否维持既有研究契约，没有偷偷降低阈值；
7. seed 202 预先指定为后续 handoff checkpoint，是否足以阻止根据 Gate 结果
   cherry-pick training seed。

## 已有但不应越界解释的证据

- pilot-v3 teacher：12/12 tasks，149/158 valid，3,939/3,939 mapped；
- pilot-v3 training：3 seeds x 3 curve sizes 全部完成，reload error 0；
- mean regret：16/32/64 states = 0.596116/0.496956/0.447868；random =
  0.685977；
- pilot Gate PASS；完整 S05 Gate 仍是 `NOT_EVALUATED`；
- 没有 online solve-time/node 结论，也没有 S06/final-test 结论。

请返回 `PASS`、`CONDITIONAL PASS` 或 `FAIL`，逐项列出 blocking findings。
只有 `PASS` 才允许 formal implementation/execution；即便 PASS，也不能在完整
S05 Gate 通过前开始 S06 或创建 S05 audited tag。
