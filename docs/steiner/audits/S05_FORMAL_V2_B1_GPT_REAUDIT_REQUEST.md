# S05 formal-v2 B1 — GPT focused re-audit request

请只读复审 formal-v2 ordering-key remediation，不修改仓库。本次只需判断首次
`CONDITIONAL PASS` 的唯一 blocker B1 是否关闭。

## 固定对象

- branch：`research/steiner-migration`
- first v2 content head：`6404c047e49b505a2384bd30aae16de8b621f09e`
- recorded conditional-audit commit：
  `0a6b5ffc06ab49f97e38680ac493bcd7f1577f19`
- B1 remediation content head：
  `05ffd02a7d491dcbc75dc4700c6fc0e5bd4a925b`
- focused remediation range：
  `0a6b5ffc06ab49f97e38680ac493bcd7f1577f19..05ffd02a7d491dcbc75dc4700c6fc0e5bd4a925b`
- complete post-first-review range：
  `6404c047e49b505a2384bd30aae16de8b621f09e..05ffd02a7d491dcbc75dc4700c6fc0e5bd4a925b`
- conditional audit record：
  `docs/steiner/audits/S05_FORMAL_V2_SELECTION_REMEDIATION_AUDIT_RECORD.json`
- remediation explanation：
  `docs/steiner/phases/S05/S05_FORMAL_V2_B1_ORDERING_REMEDIATION.md`

## 新冻结 hashes

- v2 YAML：
  `f101c038610d391b76c589fc81d08298160bac4883ed169f71953c7159137cbf`
- v2 explanation：
  `626568f03e0c622d05616af009b8507cdb832135953c52f06181de908633fff8`
- B1 remediation explanation：
  `0746b0d08cfe10e98980d6764636db9e0259fec1a5d4b5d0a1d7f695ca32289e`

## B1 修复

权威 YAML 现在逐字使用 sealed manifest/`TeacherSample` 的真实字段：

```yaml
primary_order:
  [state_index, graph_sha256, teacher_seed, semantic_sha256]
fallback_order:
  [state_index, bucket_order, graph_sha256, teacher_seed, semantic_sha256]
```

说明文件同步声明 `graph_sha256` 是唯一合法 literal lineage field，不允许 alias。
`bucket_order` 只由 manifest 的 `bucket_id` 和 YAML 冻结顺序派生。

新增测试验证：

1. primary/fallback 中每个 literal key 均存在于 manifest record schema；
2. `bucket_order` 必须能由 `bucket_id` 冻结映射；
3. 模拟 sealed availability 的 candidate pool 任意 permutation 后，选中的 640 个
   `semantic_sha256` 身份及顺序完全一致；
4. bucket composition 精确为 `56/72/0、125/3、64/64、128、60/68`；
5. 不存在的 ordering field 立即 `KeyError`，fail closed。

## 未改变边界

formal-v1 仍为 FAIL；没有 reselect、teacher rerun 或 formal training。640 states、
5×128、五 seeds、40 epochs、optimizer、CUDA determinism、validation roles、
statistics、quality/learning Gates、exact reload 0、failure retention、五个独立
one-V100 job semantics 和 claim restrictions 均未改变。`execution_authorized` 仍为
false，S06/test/final 仍禁止。

验证：targeted formal tests `6 passed`；完整 frozen-stack Steiner suite
`96 passed, 1 expected PACE skip`。

请返回 `PASS`、`CONDITIONAL PASS` 或 `FAIL`，并明确写出 `B1 = CLOSED/OPEN`。
只有 PASS 才允许登记 activation、实现/执行 fail-closed reselector、冻结实际
640-state manifest SHA 并开始五 seed 训练；不等于完整 S05 PASS 或 S06 授权。
