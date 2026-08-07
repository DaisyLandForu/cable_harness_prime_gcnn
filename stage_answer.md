**阶段 0 已完成**

完整审计报告已生成：[rl_branching_audit.md](/home/duweiyue25/acyclic_cable_harness/docs/rl_branching_audit.md)

本阶段没有修改 `scip_tree.cpp`、数学模型或其他核心 C++ 代码，也没有安装新依赖或提交 Git。

**核心结论**

- 当前可用主线是 `code/scip_tree.cpp`。
- SCIP 版本：`8.0.4`，SoPlex `6.0.4`。
- 编译器：GCC `11.4`；`rl4scip` 中 Python `3.11.15`。
- 服务器有 4 张 Tesla V100 32GB，但训练栈尚未安装。
- 当前真正生效的 branching rule 是 `relpscost`。
- 当前 node selector 是 `estimate`。
- 原程序开启 SCIP 默认 presolve、cuts、heuristics，并配置了 4～16 线程，不满足后续受控实验的单线程条件。
- 项目没有现成 CMake/Makefile，也没有统一实验参数和结构化结果输出。
- `project/scip_tree.cpp` 依赖缺失的 `xlslib.h`，目前不能直接编译。
- 数据文件存在多种不兼容的列格式，部分数据会越界读取或被静默跳过。

**真实运行结果**

使用 SCIP 8.0.4 显式头文件和动态库编译成功。主要命令为：

```bash
conda run -n rl4scip g++ -std=c++17 -O2 code/scip_tree.cpp \
  -I/home/duweiyue25/SCIP/scipoptsuite-8.0.4/scip/src \
  -I/home/duweiyue25/SCIP/scipoptsuite-8.0.4/build/scip \
  -L/home/duweiyue25/SCIP/scipoptsuite-8.0.4/build/lib \
  -Wl,-rpath,/home/duweiyue25/SCIP/scipoptsuite-8.0.4/build/lib \
  -lscip -o results/audit/scip_tree_code_probe
```

默认实例真实求解结果：

- 状态：optimal
- SCIP objective：`2.93411`
- 航空布线后处理总目标：`3.28268`
- 求解时间：`144.49 s`
- B&B 节点：`154`
- LP iterations：`20,745`
- 变量：`52,708`
- 整数变量：`1,700`
- 约束：`135,556`
- strong-branch probes：`1,574`

日志位于：

- [默认运行日志](/home/duweiyue25/acyclic_cable_harness/results/audit/run_original_default.log)
- [Smoke test 日志](/home/duweiyue25/acyclic_cable_harness/results/audit/run_original_smoke.log)
- [SCIP 插件信息](/home/duweiyue25/acyclic_cable_harness/results/audit/scip_plugins.log)
- [数据清单](/home/duweiyue25/acyclic_cable_harness/results/audit/data_inventory.csv)

**架构建议**

保留项目当前 SCIP 8.0.4，不降级。训练侧优先尝试与 SCIP 8 对应的 PySCIPOpt，并对 Ecole 做最小兼容性验证；不稳定时使用 PySCIPOpt/C++ 自建 BBMDP 环境。推理侧建议先完成 Candidate MLP，并优先采用 ONNX Runtime CPU，GCNN 放在闭环验证之后。

BBMDP 官方仓库没有发现明确许可证，因此可以研究接口和论文思想，但不应直接复制其代码。

**需要特别确认**

审计运行期间，`code/data/` 被外部操作重新整理：原来的顶层 `edges.csv`、`pairs.csv` 等文件被移动或删除，出现了新的 `1～9` 数据及 `synthesis/` 目录。该变化不是本阶段命令产生的，我没有回滚它。当前无参数运行可能因默认数据路径不存在而失败。

阶段 1 建议以已验证的 `code/scip_tree.cpp` 和上述运行结果作为 golden reference，先建立统一 CLI、单线程控制、JSON 输出和 baseline 框架。

我现在停在阶段 0，等待你的手动批准。请同时确认阶段 1 应以当前新的 `code/data` 目录布局为准，还是以 Git 中原来的数据布局为准。

**阶段 1 已完成，已暂停**

阶段报告：[baseline_report.md](/home/duweiyue25/acyclic_cable_harness/docs/baseline_report.md)

主要交付：

- [scip_tree.cpp](/home/duweiyue25/acyclic_cable_harness/code/scip_tree.cpp)：统一 CLI、五种 baseline、JSON、MILP 导出、输入校验和资源释放。
- [run_baselines.py](/home/duweiyue25/acyclic_cable_harness/scripts/run_baselines.py)：批量实验与 CSV 汇总。
- [validate_baselines.py](/home/duweiyue25/acyclic_cable_harness/scripts/validate_baselines.py)：objective、可行性和字段校验。
- [summary.csv](/home/duweiyue25/acyclic_cable_harness/results/baseline/summary.csv)
- [runs.csv](/home/duweiyue25/acyclic_cable_harness/results/baseline/runs.csv)
- [README.md](/home/duweiyue25/acyclic_cable_harness/README.md)

**验证结果**

- 9 个真实实例全部成功建模。
- 共完成 37 次 baseline：9 个实例乘 4 种常规方法，另对实例 9 运行 strong。
- 所有进程正常退出，无崩溃。
- 共同求优实例的 objective 完全一致。
- 所有 optimal 解均通过 SCIP 完整可行性检查。
- Smoke test 中 `relpscost`、`random`、`mostinf`、`fullstrong` 均有真实 callback 调用，策略切换生效。
- 相同 seed 的 objective、节点数、LP 迭代和 branchrule 调用数可复现。
- 最终验证输出：`Baseline validation passed: 37 runs`。

**Default 回归**

修改前后实例 1 的结果完全一致：

| 指标 | 修改前 | 修改后 |
|---|---:|---:|
| SCIP objective | 2.93411 | 2.93411 |
| 业务总目标 | 3.28268 | 3.28268 |
| B&B nodes | 154 | 154 |
| LP iterations | 20,745 | 20,745 |
| strong-branch probes | 1,574 | 1,574 |

因此原 default 行为没有回归。

**30 秒实验概况**

- default、relpscost、random、mostinf 都是 `3/9` 个实例求优，但求优实例集合不完全相同。
- 实例 6 四种方法全部求优。
- 实例 8 除 random 外均求优。
- 实例 9 五种方法全部求优。
- 实例 4 约 27 秒耗在 presolve，尚未进入 B&B，不能用它评价 branching。
- 这些结果是工程 baseline，不足以声称某种策略性能更好。

构建入口已实际验证：

```bash
conda run -n rl4scip make
```

当前环境没有 CMake，因此 Makefile 是正式入口，没有为此安装新依赖。真实数据和 MILP 数学表达式均未修改，也没有提交代码。

阶段 1 到此结束，等待你批准进入阶段 2：构建并划分航空布线实例数据集。

**阶段 2 已从断点恢复并完成，现已暂停**

完整报告：[dataset_report.md](/home/duweiyue25/acyclic_cable_harness/docs/dataset_report.md)  
实例清单：[manifest.csv](/home/duweiyue25/acyclic_cable_harness/data/instances/manifest.csv)  
划分配置：[phase2.json](/home/duweiyue25/acyclic_cable_harness/configs/dataset/phase2.json)

**数据集构成**

| Split | 实例数 | 内容 |
|---|---:|---|
| train | 11 | 2 个真实场景，9 个合成实例 |
| validation | 4 | 1 个真实场景，3 个合成实例 |
| test | 4 | 1 个真实场景，3 个合成实例 |
| transfer | 5 | F0001 和全部 TESTA01 真实实例 |
| 合计 | 24 | 9 个真实，15 个合成 |

train、validation、test 均覆盖 small、medium、large，合成 seed 分别为：

- train：`101/102/103`
- validation：`201`
- test：`301`

真实场景没有跨 split。相关的四个 TESTA01 变体整体保留在 transfer，避免场景泄漏。

**最大实例**

`real_04` 仅位于 transfer：

- 变量：326,502
- 整数变量：5,168
- 约束：863,691
- CIP：约 92.7 MB
- 30 秒 baseline：仍处于 presolve，B&B nodes 为 0

它没有进入训练集。

**合成器**

新增了业务语义一致的 [instance_generator.py](/home/duweiyue25/acyclic_cable_harness/scripts/instance_generator.py)：

- `N`：主干中心节点
- `E`：入口中心节点
- `L`：实际线缆叶端点
- 每根线缆对应唯一入口节点对
- 变化真实拓扑、线缆数量、约束密度和物理边长
- 不通过无意义的 objective 随机扰动扩充数据

旧 `synthesis/dataext.py` 增加了 deterministic seed，并修复了两个中心节点时可能永久循环的问题；旧合成文件不进入正式 manifest。

**真实运行与验收**

- 24 个 CIP、24 个 metadata JSON、30 个合成源 CSV 全部生成。
- CIP 总大小约 198 MB。
- 24 个实例中 15 个 optimal、9 个 time limit。
- 15 个新合成实例中 12 个在 30 秒内求优。
- 全部 15 个合成实例重新生成后，edge/pair SHA-256 完全一致。
- 所有源 CSV 和 CIP 哈希与 metadata 一致。
- 24 个 CIP 均由独立 SCIP 8.0.4 CLI 成功重新读取。
- `--resume` 再次执行时跳过全部完整实例，manifest 前后 SHA-256 完全一致。

验证日志：[validation.log](/home/duweiyue25/acyclic_cable_harness/results/dataset/validation.log)

```text
Dataset validation passed: 24 instances
Split counts: train=11, validation=4, test=4, transfer=5
```

断点续跑入口已固化在 [build_dataset.py](/home/duweiyue25/acyclic_cable_harness/scripts/build_dataset.py)。本阶段没有修改真实 CSV，也没有改变 MILP 变量、约束或目标函数。

阶段 2 到此结束，等待你批准进入阶段 3：实现无模型版自定义 SCIP branching rule。

**阶段 3 已完成**

已接入两种无模型自定义分支规则：

- `custom-random`
- `custom-mostinf`

核心实现位于 [rl_branchrule.cpp](/home/duweiyue25/acyclic_cable_harness/src/rl/rl_branchrule.cpp) 和 [scip_feature_extractor.cpp](/home/duweiyue25/acyclic_cable_harness/src/rl/scip_feature_extractor.cpp)。插件只从 SCIP 当前最高优先级 fractional LP candidates 中选择，异常时返回 `SCIP_DIDNOTRUN`，交由内置规则 fallback。

真实实例 `real_09` 结果：

| 方法 | Objective | Nodes | 合法动作 | Fallback |
|---|---:|---:|---:|---:|
| default | 0.0022776 | 9 | 不适用 | 0 |
| custom-random | 0.0022776 | 7 | 7/7 | 0 |
| custom-mostinf | 0.0022776 | 12 | 11/11 | 0 |

两种自定义方法均得到可行最优解，累计 18 次动作全部合法。平均候选选择耗时约 `3.4 微秒`。这些是单实例接入测试，不能据此声称性能优于 SCIP。

默认模式 golden 回归也完全一致：

- Objective：`2.93411`
- 业务目标：`3.28268`
- Nodes：`154`
- LP iterations：`20,745`
- relpscost calls：`158`
- custom calls：`0`

已通过：

- `conda run -n rl4scip make`
- `conda run -n rl4scip make test-custom-branching`
- 固定 seed 分支轨迹复现
- 动作合法性与结果校验
- fallback 单元测试
- SCIP `BMScheckEmptyMemory`
- `git diff --check`

完整说明见 [custom_branchrule_report.md](/home/duweiyue25/acyclic_cable_harness/docs/custom_branchrule_report.md)，实验汇总见 [summary.csv](/home/duweiyue25/acyclic_cable_harness/results/custom_branching/summary.csv)。服务器没有安装 Valgrind，因此内存检查目前使用 SCIP 自带 BMS 机制。

没有提交或推送任何代码。阶段 4 尚未开始，等待你手动批准。

**阶段 4 已完成**

已在现有 `rl4scip` 环境中完成 BBMDP 训练环境建设，未影响原 C++ 求解链路：

- 源码构建 PySCIPOpt 4.3.0 和 Ecole 0.8.1，均链接 SCIP 8.0.4。
- SCIP Python 动态库使用项目内独立前缀，原 `scip_tree` 和 SCIP 库安装前后哈希一致。
- 实现二部图状态、15 维全局树特征、候选动作 mask、节点增量奖励、终止/截断处理。
- 所有 observation 均为副本，不在 replay 数据中保存失效 SCIP 指针。
- 使用真实实例 `real_06` 和 `real_09`，未使用人工合成数据。
- 43 条真实 transition 全部通过合法性、维度、奖励和索引映射验证。
- 固定 seed 重跑的动作、奖励、节点数和 LP iterations 完全一致。
- Python 测试：`5 passed`。
- Ecole doctor：编译时和运行时 SCIP 均为 8.0.4。
- C++ default 回归：目标值 `0.0022776`、9 个节点、3596 次 LP 迭代，与安装前完全一致。
- C++ 自定义 branchrule 测试通过，只有 SCIP 头文件自身的 unused-parameter 警告。
- 没有提交或推送代码。

完整说明、修改文件、运行命令、实验结果和风险分析见 [bbmdp_environment_report.md](/home/duweiyue25/acyclic_cable_harness/docs/bbmdp_environment_report.md:1)。环境入口位于 [environment.py](/home/duweiyue25/acyclic_cable_harness/python/rl_branching/environment.py)，特征契约位于 [bbmdp_feature_schema.json](/home/duweiyue25/acyclic_cable_harness/configs/rl/bbmdp_feature_schema.json)。

本阶段没有安装 PyTorch、训练 DQN 或声称 RL 有性能提升。阶段 4 到此停止，等待你批准进入阶段 5。

**阶段 5 完成**

Candidate MLP-DQN 最小训练闭环已经完成，报告见 [training_report.md](/home/duweiyue25/acyclic_cable_harness/docs/training_report.md)，复现命令已补充到 [README.md](/home/duweiyue25/acyclic_cable_harness/README.md)。

**关键修复**

初版模型错误地使用了墙钟 `solving_time` 特征，导致同一 SCIP 状态因机器负载不同而选择不同变量。现已：

- 删除该特征并从头重训；
- 输入维度从 40 调整为 39；
- feature schema 升级到版本 2；
- 使用变量名和 action index 稳定处理 Q 值并列；
- 两次独立 20-node 运行的所有决策完全一致，轨迹 SHA-256 为 `11b3ec...e7541a`。

**训练结果**

使用真实实例 `real_06` 训练、`real_08` 验证，test 和 transfer 未使用：

- 5,000 个梯度步；
- 20 个训练 episode；
- replay buffer 735 条；
- 100 个 loss 全部有限；
- 最佳 checkpoint 位于 4,128 步；
- validation 节点曲线：`78 -> 85 -> 68 -> 63 -> 66`；
- TorchScript 与 eager Python 的 151 个 Q 值最大误差为 `0`；
- 峰值 GPU 显存约 64.4 MiB，峰值 RSS 约 1.54 GiB。

| 方法 | 求优率 | 平均节点 | 平均求解时间 |
|---|---:|---:|---:|
| random | 2/3 | 130.33 | 30.33 s |
| untrained | 2/3 | 137.00 | 33.90 s |
| RL | 3/3 | 109.33 | 29.83 s |

RL 相对 random 的截断平均节点减少 **16.1%**，平均时间改善约 **1.7%**。不过 seed 101 上 RL 为 146 节点、random 为 100 节点，因此结果并非每个 seed 同向，暂不能宣称策略已经稳定泛化，更不能宣称超过 SCIP-default。

**验证结果**

- Python 环境及 DQN 测试：`12 passed`
- C++ custom branchrule 回归：通过
- 真实实例 9 default 回归：optimal，objective `0.0022776`，解可行
- `pip check`：无依赖冲突
- SCIP 二进制和 `libscip` 哈希：与阶段 4 完全一致
- 未修改阶段 5 范围外的 MILP 定义，也未提交代码

模型位于 [artifacts/models/mlp](/home/duweiyue25/acyclic_cable_harness/artifacts/models/mlp)，最终评测在 [evaluation.csv](/home/duweiyue25/acyclic_cable_harness/artifacts/models/mlp/evaluation.csv)，复现证据在 [reproducibility_check.log](/home/duweiyue25/acyclic_cable_harness/results/phase5/reproducibility_check.log)。

阶段 5 的工程闭环通过，学习效果为“聚合优于 random，但跨 seed 稳定性仍不足”。阶段 6 可以进行 C++ 接入和推理开销验证；完整 GCNN 仍建议等待更广泛的多实例、多 seed 证据。等待你批准进入阶段 6。

**阶段 6 已完成**

已将阶段 5 的 Candidate MLP 通过 TorchScript/LibTorch 原生接入 SCIP：

- 模型只在求解器初始化时加载一次；
- 回调内仅进行特征提取和 C++ 前向推理；
- 只从当前 fractional LP candidates 中选择；
- 支持 CPU/CUDA、深度门控、候选数门控；
- 模型缺失、输出异常或动作异常时自动 fallback；
- 不启动 Python 子进程；
- 未修改原 MILP 的变量、约束和目标函数。

**真实结果**

在真实实例 `real_09`、seed 0、单线程条件下：

| 方法 | 状态 | 目标值 | 节点 | RL 决策 |
|---|---|---:|---:|---:|
| SCIP default | optimal | 0.0022776 | 9 | 0 |
| RL-MLP CPU | optimal | 0.0022776 | 13 | 12 |

12 次 RL 动作全部合法，重复运行的变量选择序列完全一致。

CPU 推理开销：

- 累计：5.899 ms
- 平均：0.492 ms/次
- 最大：1.309 ms

CUDA 数值完全正确，但累计推理约 0.692 秒，比 CPU 慢约 117 倍。这个 MLP 很小，部署时应优先使用 CPU。

缺失模型、候选数门控、`relpscost` fallback 和 `default` fallback 均已实际测试，全部得到相同最优解。

需要明确的是：该实例上 RL 使用 13 个节点，default 使用 9 个节点，因此阶段 6 证明的是**接入正确且开销可控**，不能声称 RL 优于 SCIP。

**验证结果**

- Python/C++ CPU Q 值最大误差：`0`
- Python/C++ CUDA Q 值最大误差：`0`
- argmax：完全一致
- C++ 分支测试：通过
- Python 阶段 4/5 回归：`12 passed`
- Make 与 CMake：均构建成功
- 最终自动验收：7 项全部通过

完整报告见 [integration_report.md](/home/duweiyue25/acyclic_cable_harness/docs/integration_report.md)，使用方法已补充到 [README.md](/home/duweiyue25/acyclic_cable_harness/README.md)。日志和结构化结果位于 [results/phase6](/home/duweiyue25/acyclic_cable_harness/results/phase6)。

本阶段没有提交或推送 Git，也没有改动你整理的数据。阶段 6 验收条件已经满足，现等待你批准进入阶段 7。


**阶段 7 完成**

已从断点继续并完成 Bipartite GCNN-BBMDP：

- 实现完整变量-约束二部图、两轮消息传递、PER、Double DQN、3-step return。
- 支持 scalar Q 和可选 18-bin HL-Gauss。
- 新增 C++ `--branching rl-gcnn`，支持 CPU/CUDA、深度门限和 `relpscost` fallback。
- Python/C++ 完整 Q parity：CPU 误差 0，CUDA 最大误差 `5.72e-6`，argmax 一致。
- `real_09` 上 8 次 RL 动作全部合法，零 fallback，最优目标与 default 相同：`0.0022776`。
- 缺失模型和深度门限 fallback 均已真实验证。
- Python 共 17 项测试全部通过；Make 和 CMake 构建均通过。

受控 pilot 中，GCNN 相对 random：

- 平均节点：97 → 81，减少 **16.5%**
- 平均 solving time：34.30s → 32.80s，减少 **4.4%**
- Python 环境推理占比约 **3.19%**

但生产 C++ 短实例上的动态图推理占比仍为 **36%–60%**，尚不满足最终 5% 判据，因此不能宣称优于 SCIP-default。阶段 8 应重点测试浅层 GCNN + 深层 relpscost。

完整报告见 [gcnn_report.md](/home/duweiyue25/acyclic_cable_harness/docs/gcnn_report.md)，复现命令已补充到 [README.md](/home/duweiyue25/acyclic_cable_harness/README.md)。模型位于 `artifacts/models/gcnn/`，验收结果位于 `results/phase7/`。

未提交或推送任何代码。按约定停在阶段 7，等待你批准进入阶段 8。