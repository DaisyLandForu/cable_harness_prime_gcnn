# S01 Result Analysis

## 问题与结果

S01 验证“能否建立不依赖航空命名和 solver ABI 的独立研究骨架”。结果为：package/六子包可 clean import；严格配置、typed contracts、seed、artifact path 和 logging 行为均由测试覆盖；没有导入旧 `rl_branching` 或 solver bindings。

## Gate

| 条件 | 证据 | 结论 |
|---|---|---|
| 新包可导入 | clean import + pytest | 满足 |
| 最小配置/schema tests | strict config tests | 满足 |
| unknown field 拒绝 | `StrictConfigError` test | 满足 |
| seed/artifact/logging 可复现 | scaffold tests | 满足 |
| S00 contract 保持 | 8 个 S00 tests 同批通过 | 满足 |
| 旧航空默认行为未改 | S01 diff 不含旧 Python/C++ | 满足 |
| 无生成/大产物 | staged path/size check，无 unexpected/大文件 | 满足 |

旧 regression 的 4 个失败未阻塞 S01 的“无行为修改”Gate，因为相关源码/二进制在阶段开始前已处于该状态，且 S01 diff 为零；这些失败仍是仓库风险，不能写成全量 regression PASS。

最终 staged diff check 退出码 0，允许路径检查通过。**本地 Gate S01：PASS**。S01 不提供任何 parser/MILP correctness 或模型有效性证据。

## 不能推出的结论

- 不能推出任何数据格式已支持或 MCF 正确。
- 不能推出 SCIP 8 prefix 已恢复直接可执行。
- 不能推出 GPU、GCNN、IL 或 RL 可用。
- 不能把 typed dataclass 存在等同于后续跨变体 typed policy 已实现。
