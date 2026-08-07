# 阶段 1：可复现 Branching Baseline 报告

日期：2026-08-03  
环境：Conda `rl4scip`，SCIP 8.0.4 / SoPlex 6.0.4，GCC 11.4  
数据范围：`code/data/` 下 9 组真实航空布线实例

## 1. 本阶段目标与边界

本阶段在不改变 MILP 变量、约束、目标函数和后处理定义的前提下，建立统一的运行参数、结果记录、模型导出和 branching baseline。没有增加 RL branchrule、训练环境或模型依赖，也没有进入阶段 2。

核心验收目标：

- 原始 default 模式不回归；
- 所有真实实例均可被明确读取和建模；
- default、reliability pseudocost、random、most-infeasible、strong 可切换；
- seed、时间、节点和线程设置可复现；
- 每次运行保存独立 SCIP 日志和结构化 JSON；
- 最优解 objective 跨策略一致且通过 SCIP 可行性检查。

## 2. 修改文件

| 文件 | 修改内容 |
|---|---|
| `code/scip_tree.cpp` | 向后兼容 CLI、真实数据 schema 校验、SCIP 参数控制、branchrule priority 切换、MILP 导出、JSON 指标、解可行性检查、SCIP 资源释放 |
| `Makefile` | 当前服务器的主构建入口，显式链接 SCIP 8.0.4 |
| `CMakeLists.txt` | 可选 CMake 入口；当前 `rl4scip` 未安装 CMake，因此未作为主入口 |
| `scripts/run_baselines.py` | 串行执行统一实验，保存 raw JSON/log，生成 `runs.csv` 和 `summary.csv` |
| `scripts/validate_baselines.py` | 字段完整性、可行性、objective 一致性和 default/relpscost 等价检查 |
| `README.md` | 当前环境、构建、单次运行、导出和 baseline 复现命令 |

没有修改 `project/scip_tree.cpp`、`code/scip_heur.cpp` 或任何真实数据文件。

## 3. 真实数据输入

确认并支持三类真实 `pairs` 布局：

- 紧凑布局：端点为第 4/5 列、权重为第 6 列；实例 1～4 使用该语义，其中实例 1 保留 11 列附加字段。
- 扩展布局：端点为第 4/6 列、权重为第 10 列；实例 5～9 使用该布局。
- 读取器根据关键字段是否为空识别布局，不再仅按总列数猜测。

节点分类采用真实数据所需规则：`M` 或 `N` 开头且剩余字符全为数字时才视为中心节点；`N301A-X01` 等连接器仍是叶节点。输入端点不存在、叶节点没有中心连接、schema 不支持或模型集合为空时，程序明确报错并退出，不再静默跳过。

9 个实例全部通过 `--build-only`：

| 实例 | 原始变量 | 整数变量 | 原始约束 |
|---:|---:|---:|---:|
| 1 | 52,708 | 1,700 | 135,556 |
| 2 | 83,878 | 1,924 | 214,744 |
| 3 | 41,390 | 3,792 | 100,803 |
| 4 | 326,502 | 5,168 | 863,691 |
| 5 | 39,408 | 3,056 | 97,427 |
| 6 | 20,682 | 2,708 | 48,140 |
| 7 | 38,100 | 3,036 | 93,965 |
| 8 | 27,094 | 2,880 | 65,069 |
| 9 | 10,746 | 2,668 | 22,075 |

## 4. CLI 与兼容性

新增参数：

```text
--branching default|relpscost|random|mostinf|most-infeasible|strong
--seed <int>
--time-limit <seconds>
--node-limit <int>
--threads <int>
--output-json <path>
--export-milp <path>
--build-only
--instance-id <1-9>
--edges <path>
--pairs <path>
--copy-num <int>
--div-part <int>
```

旧接口 `scip_tree [copy_num] [div_part]` 保留。无显式 `--threads` 时继续使用原程序的 `parallel/minnthreads=4`、`parallel/maxnthreads=16`、`lp/threads=4`；受控实验显式使用 `--threads 1`。

default 不修改任何 branchrule priority。当前 SCIP 默认最高优先级规则仍是 `relpscost`。其他方法只把目标插件 priority 提高到 `1000000`：

| CLI 方法 | SCIP 插件 |
|---|---|
| `default` | SCIP 默认，当前为 `relpscost` |
| `relpscost` | `relpscost` |
| `random` | `random` |
| `mostinf` | `mostinf` |
| `strong` | `fullstrong` |

node selector 始终为当前项目默认的 `estimate`；presolve、cuts、restart 和 heuristics 设置在方法间保持一致。

## 5. 构建和执行命令

构建：

```bash
conda run -n rl4scip make
```

导出 smoke 实例 CIP：

```bash
build/scip_tree \
  --instance-id 9 --branching default --seed 0 --threads 1 \
  --time-limit 60 --node-limit -1 \
  --export-milp results/baseline/smoke_instance_9.cip \
  --build-only \
  --output-json results/baseline/raw/build/instance_9_export.json
```

全量 baseline：

```bash
conda run -n rl4scip python scripts/run_baselines.py \
  --binary build/scip_tree \
  --instances 1,2,3,4,5,6,7,8,9 \
  --methods default,relpscost,random,mostinf \
  --strong-instances 9 \
  --seeds 0 \
  --time-limit 30 \
  --node-limit -1 \
  --threads 1 \
  --output-dir results/baseline
```

校验：

```bash
conda run -n rl4scip python scripts/validate_baselines.py \
  --raw-dir results/baseline/raw --expected-runs 37
```

输出：`Baseline validation passed: 37 runs`。

## 6. 单次 JSON 字段

JSON 包含：instance id、输入路径、method、seed、SCIP version、status、SCIP objective、业务后处理 objective、primal/dual bound、gap、应用 wall time、presolve/solve time、presolve 后 solve time、nodes、LP iterations、primal-dual integral、first solution time、原始变量/整数变量/约束数、实际目标 branchrule、branchrule callback 次数、node selector、线程与 limits、是否存在解、SCIP 完整可行性检查结果。

无 incumbent 或无有限界时使用 JSON `null`，不会用零伪装缺失值。PDI 和首解时间来自当前 SCIP 8.0.4 的统计结构；升级 SCIP 时需重新核对字段兼容性。

## 7. Smoke Test

真实最小实例 9 在 seed 0、单线程下全部求优：

| 方法 | objective | nodes | wall time (s) | 目标 rule calls | 可行 |
|---|---:|---:|---:|---:|---|
| default | 0.0022776 | 9 | 4.09 | relpscost: 10 | 是 |
| relpscost | 0.0022776 | 9 | 4.10 | relpscost: 10 | 是 |
| random | 0.0022776 | 4 | 2.21 | random: 4 | 是 |
| mostinf | 0.0022776 | 12 | 2.19 | mostinf: 11 | 是 |
| strong | 0.0022776 | 18 | 10.88 | fullstrong: 19 | 是 |

本结果证明接口和策略切换生效，不代表 random 优于 SCIP default；单实例、单 seed 不具备性能统计意义。

## 8. 全量 30 秒 Baseline

共有 37 次运行：9 实例 x 4 常规方法，加实例 9 的 strong。全部进程正常退出，无崩溃；30 秒时限是阶段 1 的统一探测预算。

| 方法 | runs | optimal | timeout | solved rate | median wall (s) | median nodes |
|---|---:|---:|---:|---:|---:|---:|
| default | 9 | 3 | 6 | 33.3% | 30.68 | 2 |
| relpscost | 9 | 3 | 6 | 33.3% | 30.57 | 2 |
| random | 9 | 3 | 6 | 33.3% | 30.59 | 4 |
| mostinf | 9 | 3 | 6 | 33.3% | 30.63 | 12 |
| strong | 1 | 1 | 0 | 100% | 10.88 | 18 |

求优分布：

- 实例 5：仅 random 在 30 秒内求优；mostinf 得到 gap `8.70e-6` 的可行解。
- 实例 6：四种常规方法均求优，objective 均为 `0.0093259`。
- 实例 8：default、relpscost、mostinf 求优，objective 均为 `0.0096121`；random 有可行解但 gap 为 `0.1266`。
- 实例 9：五种方法均求优，objective 均为 `0.0022776`。
- 实例 1：均超时；mostinf 找到 objective `2.93418992`、gap `2.72e-5` 的可行解。
- 实例 2、3、7：30 秒内没有 incumbent。
- 实例 4：约 27 秒用于 presolve，时限到达时尚未进入 B&B，四种 branching 方法均为 0 nodes。

因此这组结果是工程 baseline 和难度画像，不足以宣称某种方法更优。尤其实例 4 的 30 秒结果完全不包含 branching 信息。

## 9. Default 回归与复现性

原设置 golden 回归使用实例 1、default、seed 0，不设置 time limit，也不强制单线程：

| 指标 | 阶段 0 修改前 | 阶段 1 修改后 |
|---|---:|---:|
| status | optimal | optimal |
| SCIP objective | 2.93411 | 2.93411 |
| 业务总目标 | 3.28268 | 3.28268 |
| nodes | 154 | 154 |
| LP iterations | 20,745 | 20,745 |
| strong-branch probes（SCIP 日志） | 1,574 | 1,574 |
| solving time | 144.49 s | 143.88 s |

objective、树规模和 LP 迭代完全一致，时间差属于正常墙钟波动，default 无行为回归。

实例 9 的 default seed 0 和 random seed 0 各重复一次，objective、nodes、LP iterations 和目标 rule calls 均一致。random seed 1 得到相同 objective，但在根节点由其他 SCIP 机制完成，未调用 branching rule；这说明 seed 已改变随机求解轨迹，同时不能把未发生分支的运行当作 random branching 决策样本。

## 10. 正确性结论

- 37 次正式 baseline 进程均正常退出。
- 所有 optimal incumbent 均通过 `SCIPcheckSol` 的 bounds、integrality、LP row 和约束检查。
- 共同求优实例的 objective 在 tolerance `1e-8 * max(1, |obj|)` 内完全一致。
- default 和显式 relpscost 在共同求优时 status、objective、nodes、LP iterations 一致。
- 时限样本可能因墙钟截止点出现少量 LP iteration 差异，不将其误判为不复现。
- smoke 中四个目标 SCIP branchrule 都有非零 callback 计数，证明切换生效。
- `SCIPfree` 已恢复到成功路径，变量和约束的应用侧引用被释放；日志未报告内存错误。尚未运行 Valgrind，因此不能把这一点表述为完整泄漏证明。

## 11. 风险与阶段 2 输入

1. 当前 edge weight 仍沿用原模型的整数存储，真实 CSV 小数会被截断。这是既有模型行为，本阶段为避免改变 objective 未修正；需要由业务侧单独确认是否为设计意图。
2. 实例 4 的 presolve 本身已接近 30 秒，后续比较 branching 必须给予更长预算，或在 controlled protocol 中统一处理 presolved 模型。
3. 单 seed baseline 用于工程验收，不用于最终统计结论；阶段 8 仍需至少 5 seeds。
4. PDI 与首解时间读取使用当前 SCIP 8.0.4 内部统计字段，未来版本升级要增加兼容层。
5. CMake 在当前环境不可用；Makefile 已实际验证。没有为阶段 1 安装新依赖。

## 12. 验收状态

| 条件 | 状态 |
|---|---|
| 原 default 模式与修改前一致 | 通过 |
| JSON 字段完整 | 通过 |
| 相同 seed 可复现 | 通过：确定性搜索字段一致 |
| 不同 branching 方法确实生效 | 通过：priority 与 callback 计数验证 |
| objective 和解可行性无异常 | 通过 |
| 所有真实规模均运行 | 通过：实例 1～9，统一 30 秒预算 |
| strong 仅用于小规模 | 通过：实例 9 |

阶段 1 完成后应暂停，等待用户批准是否进入阶段 2。
