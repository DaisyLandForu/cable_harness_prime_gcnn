# 阶段 3：无模型自定义 Branchrule 接入报告

## 1. 阶段目标与结论

本阶段只验证 SCIP branching 插件闭环，不引入 RL 模型或新推理依赖。航空布线 MILP 的变量、约束、目标函数和解后处理保持不变。

已完成：

- `--branching custom-random`；
- `--branching custom-mostinf`；
- SCIP 8.0.4 LP branching candidate 提取；
- 只从最高 branching priority 的 fractional LP candidates 中选择；
- `SCIPbranchVarVal` 分支；
- 无候选、提取/选择异常和非法动作时返回 `SCIP_DIDNOTRUN`；
- 可关闭的逐分支 CSV 日志；
- JSON 汇总指标；
- 策略、seed、索引边界、fallback 和 SCIP 内存记账单元测试；
- 真实实例端到端求解、default golden 回归和复现性测试。

阶段 3 验收通过。这里证明的是插件接入正确，不证明 RL 或 custom 策略优于 SCIP default。

## 2. 修改文件

| 文件 | 作用 |
|---|---|
| `src/rl/scip_feature_extractor.hpp/.cpp` | 用 SCIP 8.0.4 API 提取可行动作集合及 LP 值、fractionality、变量索引 |
| `src/rl/rl_branchrule.hpp/.cpp` | `ObjBranchrule`、random/most-infeasible 策略、合法性检查、fallback、CSV 日志和计时 |
| `code/scip_tree.cpp` | 增加 CLI、按需注册插件、优先级设置和 JSON 指标 |
| `tests/test_custom_branchrule.cpp` | 空 action set、most-infeasible、固定 seed、action mask、pseudo fallback、内存记账测试 |
| `scripts/validate_custom_branching.py` | 校验 objective、可行性、插件名、调用计数和逐行动作合法性 |
| `Makefile`、`CMakeLists.txt` | 编译新源文件和测试目标 |
| `README.md` | 阶段 3 使用和测试命令 |

未增加 `model_runner`，因为本阶段没有模型；阶段 6 再按实际部署后端实现。

## 3. SCIP 8.0.4 接口决策

实际头文件 `scip/scip_branch.h` 说明 branching rule 应从 `SCIPgetLPBranchCands` 返回数组的前 `npriolpcands` 个变量中选择。本实现因此将这部分定义为 action set，并保存：

- SCIP 变量指针，只在当前 callback 内使用；
- LP solution value；
- fractionality；
- candidate index；
- transformed problem variable index。

选中后再次检查指针确实属于当前 action set，然后调用 `SCIPbranchVarVal`。插件名称分别为 `rlcustomrandom` 和 `rlcustommostinf`，实际优先级为 `1000000`；默认 `relpscost` 优先级保持 `10000`。

插件只在指定 custom 方法时注册。默认、baseline 和 legacy 路径不会创建插件对象，也不会执行候选提取或日志代码。

## 4. Branch 日志

`--branch-log <path>` 可选。未指定时不打开文件，也不向 stdout 输出逐分支信息。CSV 字段为：

```text
event_index,node_id,depth,candidate_count,selected_candidate_index,
selected_variable_index,selected_variable_name,lp_value,fractionality,
selection_time_seconds,selected_is_candidate,result,fallback_reason
```

JSON 新增 custom 调用、决策、候选数、合法性、fallback 和选择耗时指标。原阶段 1 字段保持兼容。

## 5. 构建与测试命令

```bash
conda run -n rl4scip make
conda run -n rl4scip make test-custom-branching
```

真实数据 smoke test：

```bash
conda run -n rl4scip ./build/scip_tree \
  --instance-id 9 --branching custom-random --seed 0 \
  --time-limit 60 --threads 1 \
  --output-json results/custom_branching/raw/real_09_custom-random_seed0.json \
  --branch-log results/custom_branching/raw/real_09_custom-random_seed0_branches.csv

conda run -n rl4scip ./build/scip_tree \
  --instance-id 9 --branching custom-mostinf --seed 0 \
  --time-limit 60 --threads 1 \
  --output-json results/custom_branching/raw/real_09_custom-mostinf_seed0.json \
  --branch-log results/custom_branching/raw/real_09_custom-mostinf_seed0_branches.csv
```

产物验证：

```bash
conda run -n rl4scip python scripts/validate_custom_branching.py \
  --reference results/custom_branching/raw/real_09_default_seed0.json \
  results/custom_branching/raw/real_09_custom-random_seed0.json \
  results/custom_branching/raw/real_09_custom-mostinf_seed0.json
```

## 6. 真实实例结果

数据来自 `code/data/edges-9.csv` 和 `code/data/pairs-9.csv`，不是 `code/data/synthesis/`。三种方法均为 seed 0、单线程、60 秒限制。

| 方法 | status | objective | nodes | LP iterations | 规则调用/决策 | 非法动作 | fallback | 选择均值 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| default | optimal | 0.0022776 | 9 | 3,596 | 10 / 0 | 0 | 0 | 0 |
| custom-random | optimal | 0.0022776 | 7 | 3,663 | 7 / 7 | 0 | 0 | 3.48 us |
| custom-mostinf | optimal | 0.0022776 | 12 | 3,583 | 11 / 11 | 0 | 0 | 3.42 us |

两种 custom 运行均通过 `SCIPcheckSol`。共检查 18 次真实 callback 动作，全部属于当时 action set。`custom-mostinf` 的节点数和阶段 1 内置 `mostinf` 的 12 节点一致，也从侧面验证策略语义。

这张表只有一个实例和一个 seed，时间与节点差异不能解释为统计性能改善。

## 7. Default 回归与开销

使用阶段 0/1 的原始 golden 配置重跑真实实例 1：default、seed 0，不设置时限，不强制线程数。

| 指标 | 阶段 1 golden | 阶段 3 |
|---|---:|---:|
| status | optimal | optimal |
| objective | 2.93411 | 2.93411 |
| 业务目标 | 3.28268 | 3.28268 |
| nodes | 154 | 154 |
| LP iterations | 20,745 | 20,745 |
| relpscost calls | 158 | 158 |
| solving time | 143.8796 s | 143.7353 s |
| custom calls | 不适用 | 0 |

所有稳定指标完全一致；计时差为正常波动。default 模式没有 callback、候选提取或结构化分支日志开销。

## 8. 复现、fallback 与内存

- `custom-random` seed 0 重跑后，忽略计时列的 7 行 branch trace 完全一致，nodes 和 LP iterations 也一致。
- 空候选选择返回 `-1`，callback 将其映射为 `SCIP_DIDNOTRUN`。
- external/pseudo candidate 模式沿用 `ObjBranchrule` 的 `SCIP_DIDNOTRUN`，单元测试显式覆盖 pseudo fallback。
- 候选提取错误/异常、选择异常、非法动作和 branch API 错误均记录原因并返回 `SCIP_DIDNOTRUN`，让较低优先级 SCIP 规则继续执行。
- `SCIPincludeObjBranchrule(..., TRUE)` 将插件对象所有权交给 SCIP，随 `SCIPfree` 释放。
- 单元测试调用 `BMScheckEmptyMemory()`，未报告泄漏。服务器未安装 Valgrind，因此没有 Valgrind 报告。

## 9. 验收结果

| 条件 | 结果 |
|---|---|
| custom-random 完整求解真实小实例 | 通过 |
| 所选变量始终属于当前 candidate set | 通过，18/18 合法 |
| objective 与 default 一致且解可行 | 通过 |
| 无明显内存泄漏 | 通过 SCIP BMS 检查；无 Valgrind |
| custom 关闭时无额外求解路径开销 | 通过，golden 稳定指标完全一致、custom calls=0 |
| 优先级和 fallback 符合预期 | 通过，custom=1,000,000；default=10,000；DIDNOTRUN 单测通过 |

## 10. 阶段 4 接口准备

阶段 4 可以复用 `extractLpBranchCandidates` 作为 action 映射的唯一入口，避免 Python 环境与最终 C++ 推理对候选顺序产生两套定义。下一阶段仍需单独决定训练环境如何控制节点选择、restart、cuts 和 episode 生命周期；本阶段没有预先绑定 Ecole、PySCIPOpt 或 BBMDP 依赖。

原始结果位于 `results/custom_branching/raw/`，汇总位于 `results/custom_branching/summary.csv`。
