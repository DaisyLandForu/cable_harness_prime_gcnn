# S04 Audit Packet — S00--S04 联合审计入口

## Remediation v2 当前结论

- 首次审计请求：`docs/steiner/audits/S00_S04_GPT_AUDIT_REQUEST.md`，SHA-256
  `004473911535ad994191f452d0acdd6c41ab1c05eb3dc2893b4c9ba7e3d93d8d`
- GPT 返回原文：`docs/steiner/audits/S00_S04_GPT_AUDIT.md`，SHA-256
  `abfc6f20f97092857edf8dbf606695800da6d483c27418c1e4232fed3edd277f`
- 首次结论：**CONDITIONAL PASS**；blocking finding B1 是 Ecole row identity
  依赖 PySCIPOpt list order。
- 修复：`solver/scip_identity.py` 以 frozen SCIP 8.0.4 probindex 为 canonical row
  identity，每次 extraction 验证完整双射；非冻结 stack/prefix/checksum 均拒绝。
- v2 local Gate：**PASS（8/8）**；3/3 states、2,943/2,943 rows、31/31
  actions、max error 0、argmax 3/3，snapshot 未变。
- v2 content head：由紧随其后的 metadata commit 写入；planned local tag：
  `steiner-s04-local-gate-v2`。v1 tag 保留，不改写历史。
- GPT re-audit：**PENDING**；本地 PASS 不能写成最终审计 PASS。

## 不可变审计对象

- branch：`research/steiner-migration`
- S04 base SHA：`931c7ae05c299c54bbdf59ecd458b64c7ca42282`
- S04 content head SHA：`d7a78a33151822f3a8a57fdc0224ede333583646`
- S04 phase head SHA：本 metadata commit 的 annotated local tag target；由最终
  handoff 和 `steiner-s04-local-gate-v1^{}` 固定
- S04 substantive range：
  `931c7ae05c299c54bbdf59ecd458b64c7ca42282..d7a78a33151822f3a8a57fdc0224ede333583646`
- planned local tag：`steiner-s04-local-gate-v1`
- remote：只允许 local Gate PASS 后 fast-forward push 同名长期分支
- PR：未创建；不 merge/rebase/amend/force-push，不 push local-gate tag
- GPT audit：首次联合审计 **CONDITIONAL PASS**，remediation re-audit PENDING。
  用户后续 waiver 只允许 S05 implementation scaffold；正式数据/训练/Gate 仍阻塞。

## 联合审计历史锚点

| 阶段 | base | content | local-gate tag target |
|---|---|---|---|
| S00 | `88ade1ac614fb12f882a10ba9b5d35b15c7b4d01` | `8b90375b6617a1ddcba34b872dbdbc11411cc042` | `a0bf0e3c1a702e1c85384f864defc86abbda29a5` |
| S01 | `a0bf0e3c1a702e1c85384f864defc86abbda29a5` | `05b42791226347d31647547c344ef46c9dc4e87d` | `35a90ec5e52e2fad8301e3441ff6b286c7701d04` |
| S02 | `35a90ec5e52e2fad8301e3441ff6b286c7701d04` | `19c7f46b91a1d05c46dbdeeba00bf863b37a7f5a` | `25be2e18c4020bed4cb8563618687b148d1f405f` |
| S03 | `91c30a48e6a06019d16d8b7529fe2d35bfa708fa` | `495d699cceefd243d4ab4c510be051f9df94833a` | `bb6079b7844dcc42fed4976c812795c842d6411b` |

阶段审计包分别位于 `docs/steiner/phases/S00` 至 `S04`。S00--S02 的旧阶段分支
名称是历史记录；ADR 0005 后所有成果已经累计到当前长期分支，不能按旧分支名
重新切分或 merge。

## S04 需求映射

| 需求 | 实现/配置 | 测试/证据 |
|---|---|---|
| versioned 19/5/1 | `solver/bipartite_observation.py` | exact names/widths、shape/finite tests |
| immutable graph state | 同上 | copy/read-only/malformed/empty tests |
| fractional binary edge actions | observation + naming | transformed/parallel/non-binary/duplicate tests；31/31 real mapping |
| candidate exact closure | `solver/graph_state.py` | stable maps；full/closure logit parity |
| aviation-independent B0 | `models/milp_gcnn.py` | no aviation symbols；68,161 params；strict config |
| real SCIP integration | snapshot runner + P1 | 3 real states；wrapper-only test twice |
| determinism/performance | forward/Gate JSON | byte-identical snapshot；CPU p50/p95 |
| no final/training leakage | train seed/config/runner | no optimizer/loss/checkpoint/GPU/final selector |

## S04 变更边界

```text
configs/steiner/models/b0_milp_gcnn_v1.yml
python/steiner_branching/milp/naming.py
python/steiner_branching/models/{__init__,milp_gcnn}.py
python/steiner_branching/solver/{__init__,branchability,bipartite_observation,graph_state}.py
scripts/steiner/{README.md,run_s04_b0_snapshot.py}
tests/steiner/{test_s00_contract.py,test_s04_bipartite.py}
docs/steiner/{RESEARCH_CONTRACT,S03_S04_RESOURCE_PREFLIGHT,STATUS}.md
docs/steiner/phases/S04/**
plans/STEINER_RL_BRANCHING_MIGRATION_MASTER_PLAN.md
```

既有 `artifacts/**`、`build/**`、`results/audit/**`、航空 scripts、用户 figures、
未跟踪 experiment config 和 raw S03 data 明确排除在 staged diff 外。

## 一键复现

```text
scripts/steiner/run_with_scip804.sh --verify-only
scripts/steiner/run_with_scip804.sh --python -m pytest -q tests/steiner
scripts/steiner/run_with_scip804.sh --python \
  scripts/steiner/run_s04_b0_snapshot.py \
  --snapshot-output /tmp/s04-snapshot.json \
  --summary-output /tmp/s04-summary.json
cmp docs/steiner/phases/S04/S04_FORWARD_SNAPSHOT.json /tmp/s04-snapshot.json
```

## Hash 与 Gate 证据

- config canonical：`056ce49bce41c731138a83b3befbc97e006585311bcf8f5298532c2d86f830dc`
- config file：`9360f5893103adcf3b12baa3f8b2d1d5e0549791c7da68060e43542257ed1fc2`
- schema：`f47a74b08a3ab88f07733f17b8932a2bf6878ed271321938c121845f44d037fc`
- model state：`42b63f31ccbb0b5416db53d38e9ca0072daedbfba3d84d30094a9d443ac3795c`
- snapshot canonical：`d7ed96707bb625b64c4251840b348e78cae518d1ead1018c80e540ac04b3d536`
- snapshot file：`ac2ce0c14b134245221af5140a3008f3ec6067f8867491e7cc0d0b50e2036f2c`
- Gate summary v1 file：`f687460a8e6bbae9b0159cc1a851bff97cc3814ef9f800e848dbc6d01d08c633`
- Gate summary remediation v2 file：
  `ea94f25e3c42fed2b464f9c914af521ed8e8d1a6537505bbb2a164b65f7b937b`
- tests：78 passed、1 expected skipped；real snapshot byte-identical
- Gate：probindex identity 2,943/2,943、31/31 mapping、max error 0、argmax
  3/3、finite、parameter/timing recorded；local **PASS（8/8）**

## 已知风险与联合审计重点

1. `copy_node_bipartite` 对 Ecole 无 incumbent 的 features 13/14 使用 zero
   sentinel。请核对该选择是否需要 missingness bit/new schema，或对 S05 B0 可接受。
2. v2 已移除 transformed PySCIPOpt list-order 依赖。请复审 canonical probindex
   bridge、fail-closed tests 与 `probindex_identity_complete` 是否充分关闭 B1。
3. 请独立检查 one-round exact closure 的 receptive field：candidate→incident rows
   →rows 上全部 variables，是否对本 sum-message implementation 严格充分。
4. S04 evidence 只有 1 train graph/3 states，适合 parity Gate，不支持模型质量结论。
5. 研究契约 v1.3 只校正历史时态和已提交的资源事实；请确认没有隐含 Gate 变更。
6. S03 有 46/90 timelimit、两个 geometric root-LP timeout 和跨 CPU host timing；
   请按 S03 审计包检查主张边界。
7. SteinLib/DIMACS raw 再分发许可未确认；旧航空 4 个失败仍是独立 backlog。
8. staged boundary 必须不含既有 artifact/build/航空/用户文件或任何 secret。

## 建议审计结论

Codex 建议 S04 remediation 本地 **PASS**，请围绕 B1 做 GPT 复审。复审结论
仍为 PENDING；只有 GPT 给出最终 PASS 并提交审计记录后，才能创建 audited tag、
采集 S05 正式 teacher 或训练。用户对 S05 的临时 waiver 仅覆盖源码/脚本准备。
