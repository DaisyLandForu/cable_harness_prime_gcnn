# S05 formal-v2 result — GPT failure audit request

请只读审计 S05 formal-v2 的执行结果和 fail-closed 判断，不修改仓库，不把
25-lineage 描述性统计当成预注册 30-lineage Gate。

## 固定执行对象

- branch：`research/steiner-migration`
- execution/seal head：`da5b5abd8220f8bdaa1a977455d25ef9aaaa84b7`
- failure-evidence content head：`5626517c3689a1e885a69ac7f111f587864f9924`
- result evidence range：
  `da5b5abd8220f8bdaa1a977455d25ef9aaaa84b7..5626517c3689a1e885a69ac7f111f587864f9924`
- v2 selection manifest SHA-256：
  `35221abeeaae507623d0175d895b5ff807d0b7e5400fce494e615fbefe8cd10e`
- v2 Gate summary：
  `docs/steiner/phases/S05/S05_FORMAL_V2_GATE_SUMMARY.json`
- v2 selection seal：
  `docs/steiner/phases/S05/S05_FORMAL_V2_SELECTION_SEAL.json`
- base protocol：`docs/steiner/phases/S05/S05_FORMAL_PROTOCOL.md`
- v2 revision：`docs/steiner/phases/S05/S05_FORMAL_V2_SELECTION_REMEDIATION.md`

## 已完成且没有失败的部分

- 5/5 formal seeds 从头训练 40 epochs；
- 每个 seed 均评估冻结的 160 select states 和 320 Gate states；
- 5/5 checkpoint/state-dict reload bit exact，最大推理误差严格为 0；
- 所有 seed 在可用 lineage 上的 mean regret 均低于冻结 random baseline；
- 没有访问 test/final，没有替换 seed、graph 或失败状态。

## fail-closed 事件

聚合器在运行任何正式 bootstrap 结论前抛出：

```text
ValueError: formal Gate must contain exactly 30 base graph lineages
```

冻结的 320 个 `validation_gate` states 只覆盖 25/30 个注册 base graphs。缺失的
5 个图在 formal-v1 teacher evidence 中虽然产生了共 45 个 observed states，但
valid states 都是 0：

- random geometric：seeds 201024、201025、201026；
- sparse Erdős–Rényi：seeds 201019、201020。

它们全部存在 strong-child validity failure，部分同时存在 strong-call shortfall。
因此不能为这些 lineage 计算真实 model-vs-random paired effect。

## 当前本地判断

`S05 Gate = FAIL`，S06 不授权。不能擅自把 bootstrap unit 从 30 改成 25、给缺失
lineage 人工填零、换图、重跑到成功，或用 test/final 调参。

25-lineage 描述性统计仅用于诊断：bootstrap 95% CI 为
`[0.05627423, 0.19170749]`，seed regret CV 为 `0.0686353`，但
random-geometric family mean effect 为 `-0.0418052`。这些数字不能覆盖正式
denominator failure，也不能宣称完整 S05 PASS。

另请注意冻结说明文件存在一个内部文字矛盾：先写“30 个 Gate lineages”，随后又写
bootstrap resamples “20 base graphs”。审计请求、105-graph instance matrix、30 个
`validation_gate` 注册图和 executable aggregator 均明确要求 30，因此本次执行按
30 fail-closed，没有在看到结果后选择更有利的 20 或 25。请将该矛盾纳入判断。

## 请判断

1. 当前 `FAIL / STOP before S06` 是否正确；
2. 这是完整 S05 终止，还是允许事前冻结 formal-v3 remediation；
3. 若允许 v3，是否必须重新定义 missing-lineage policy，并使用新的独立 validation
   base graphs/teacher collection，而不能继续重用已经看过表现的 25-lineage Gate；
4. 是否还存在比 lineage denominator 更早或独立的 blocking finding。

请返回 `PASS`（认可本次 FAIL 登记）、`CONDITIONAL PASS` 或 `FAIL`，并明确
S05 是否仍为 FAIL、S06 是否保持禁止，以及任何 v3 的最小合法边界。
