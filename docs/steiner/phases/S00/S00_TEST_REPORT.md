# S00 Test Report

## 环境

- 日期：2026-09-02 UTC
- branch：`research/steiner-s00-contract`
- base：`88ade1ac614fb12f882a10ba9b5d35b15c7b4d01`
- Python：`/home/duweiyue25/conda/envs/rl4scip/bin/python` 3.11.15
- PyYAML：6.0.2
- 测试不加载 solver，不接触 final data。

## 已执行测试

命令：

```text
/home/duweiyue25/conda/envs/rl4scip/bin/python -m unittest -v tests/steiner/test_s00_contract.py
```

退出码：0。

结果：8 passed，0 failed，0 skipped。

同时执行 discovery 入口：

```text
/home/duweiyue25/conda/envs/rl4scip/bin/python -m unittest discover -s tests/steiner -p 'test_*.py' -v
```

退出码：0；结果同为 8 passed，0 failed，0 skipped。

| 测试 | 结果 | 覆盖风险 |
|---|---|---|
| required documents | PASS | 契约/ADR/状态/计划遗漏 |
| protocol IDs/limits/seeds | PASS | profile 或 seed 含糊、事后更换预算 |
| baselines/metrics/statistics | PASS | baseline 缺失、选指标/选 checkpoint 泄漏 |
| action/branchability Gates | PASS | 动作扩大、Gate 降低 |
| synthetic seed disjointness | PASS | generator split 泄漏 |
| public dev/final separation | PASS | PACE/SteinLib dev-test 混用 |
| final manifest seal/hash | PASS | final membership 漂移、提前运行 |
| environment selected/conflicting stacks | PASS | SCIP 8/9 混跑、缺失前置条件 |

## 失败和 skipped 探测

这些不是 contract unit-test 失败，但属于 S00 环境审计结果：

- 默认 Miniforge Python 3.12：PyYAML、Ecole、PySCIPOpt unavailable；因此不是冻结运行环境。
- `rl4scip` bare Ecole/PySCIPOpt import：FAIL，`libscip.so.8.0` 不在动态库搜索路径。
- 加 `artifacts/environment/phase4/scip804_prefix/lib` 后 Ecole/PySCIPOpt import：PASS。
- SCIP 8.0.4 prefix binaries direct execution：FAIL（mode `0644`, permission denied）；通过 ELF dynamic loader 的只读 version probe：PASS。
- 当前 `nvidia-smi`：unavailable；PyTorch CUDA probe：`available=false`, device count 0。
- 系统 `/usr/bin/scip --version`：PASS，但版本 9.2.2；按契约禁止用于正式结果。

初次 `git diff --cached --check`：FAIL，发现 8 个新文件有多余 EOF 空行；已删除这些空行。最终 `git diff --cached --check`：PASS，退出码 0、无输出。

## 未执行测试/实验

- final test learned runs：SKIPPED BY DESIGN，封存到 S12；计数保持 0。
- Steiner parser/MILP/solver/model tests：SKIPPED，S00 不存在这些实现，属于 S02/S04 后续 Gate。
- 旧航空全量 regression：SKIPPED，S00 未修改旧源码；当前工作区还有大量与本阶段无关的用户二进制/脚本改动，运行结果不能归因于 S00。
- SCIP solve smoke：SKIPPED，S00 只冻结契约，且冻结 prefix 当前需先修复可执行/loader 前置条件。

## 测试产物

没有生成原始日志、数据或 checkpoint。测试源码与 S00 配置的 SHA-256 在 `S00_AUDIT_PACKET.md` 记录；提交后以 Git blob/commit 提供不可变索引。
