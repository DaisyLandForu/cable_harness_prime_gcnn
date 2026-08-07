你现在需要在当前工作空间的航空布线优化项目中，基于 BBMDP（A Markov Decision Process for Variable Selection in Branch & Bound）的思想，为 SCIP 的 Branch-and-Bound 增加强化学习分支变量选择策略。

项目核心文件预计包含 scip_tree.cpp，但你必须以当前工作空间中的实际文件为准，不得假设目录、函数名、SCIP 版本或编译方式。

==================================================
一、最终目标
==================================================

保留现有航空布线 MILP 的变量、约束、目标函数和整体 SCIP 求解流程，仅扩展 B&B 中的 branching variable selection，使项目支持：

1. 原始 SCIP 默认分支策略；
2. reliability pseudocost；
3. random branching；
4. strong branching（用于小实例实验）；
5. RL branching。

最终完成：

- 航空布线实例生成或导出；
- baseline 实验；
- BBMDP 风格 RL 环境；
- DQN 模型训练；
- 模型验证；
- RL 分支规则接入 scip_tree.cpp；
- 完整对比实验；
- 最终可行性报告。

不能只完成代码接口，必须实际编译、运行、训练并输出实验结果。不得在没有真实日志的情况下声称 RL 有提升。

==================================================
二、总体执行原则
==================================================

1. 分阶段实施，不能直接大规模重写 scip_tree.cpp。
2. 每个阶段必须：
   - 说明修改文件；
   - 给出执行命令；
   - 完成编译或测试；
   - 保存日志；
   - 输出阶段报告；
   - 满足验收条件后再进入下一阶段。
3. 保持原功能向后兼容：
   - 不指定 RL 参数时，程序行为必须与原项目一致。
4. 不要删除原代码，不要进行无关重构。
5. 所有随机过程必须可通过 seed 复现。
6. 训练集、验证集、测试集必须按原始 MILP 实例划分，不能把同一实例的 B&B 状态随机拆到不同集合。
7. 所有模型输出只能在 SCIP 当前 fractional LP branching candidates 中选择。
8. 模型异常时必须 fallback 到 SCIP 原分支策略。
9. 优先完成最小闭环，再实现完整 GCNN。
10. 对所有不确定的 SCIP API、参数和版本兼容性，先检查当前安装的 SCIP 头文件和官方示例，不要凭记忆编写。

==================================================
三、阶段 0：完整审计，暂不修改核心代码
==================================================

首先递归读取当前工作空间内所有相关文件，包括但不限于：

- scip_tree.cpp；
- CMakeLists.txt / Makefile；
- SCIP 相关头文件和链接配置；
- 数据集；
- 实例生成代码；
- 运行脚本；
- README；
- 已有实验结果；
- 目标函数和约束实现；
- 变量命名和变量类别。

执行并记录：

- git status；
- 项目目录结构；
- SCIP 版本；
- 编译器版本；
- Python 版本；
- CUDA/GPU 信息；
- 当前项目编译命令；
- 当前运行命令；
- 当前 SCIP 参数；
- 当前真正生效的 branching rule；
- 当前 node selection rule；
- 是否使用 presolve、cuts、restart、heuristics；
- 是否单线程；
- 当前有哪些航空布线数据集及其规模。

重点分析 scip_tree.cpp：

1. SCIP 对象在哪里创建和释放；
2. 默认插件在哪里注册；
3. 变量、约束和目标函数在哪里构建；
4. SCIPsolve 在哪里调用；
5. 是否已经有树搜索、节点、变量日志；
6. 是否方便导出 CIP/MPS；
7. 如何添加自定义 branching rule；
8. 原始实例如何参数化；
9. 当前最大实例为何求解困难；
10. 各类变量和约束在航空布线业务中的含义。

输出：

docs/rl_branching_audit.md

该文档必须包含：

- 项目结构；
- 完整构建链路；
- SCIP 版本；
- 当前 branching/node selection；
- 数据集清单；
- 变量和约束类别；
- 计划修改的具体文件；
- 依赖方案；
- 风险列表；
- Python/Ecole 与当前 SCIP 版本是否兼容；
- 推荐的训练和 C++ 推理架构。

如果当前 SCIP 版本与 BBMDP 官方实现使用的版本不同，不要直接降级项目。需要提出以下选择的比较：

A. 使用项目当前 SCIP 版本自行构建训练环境；
B. 在独立容器中使用 BBMDP 官方环境做 proof-of-concept；
C. 使用与当前 SCIP 版本对应的 PySCIPOpt；
D. 使用 Ecole，但明确版本差异。

==================================================
四、阶段 1：建立可复现 baseline
==================================================

在修改 branching rule 前，先让原项目支持统一的命令行参数或配置文件：

--branching default
--seed <int>
--time-limit <seconds>
--node-limit <int>
--threads 1
--output-json <path>
--export-milp <path>
--build-only

如果项目已有参数系统，则扩展现有系统，不要重复实现。

输出的单次运行 JSON/CSV 至少包含：

- instance_id；
- method；
- seed；
- SCIP version；
- solve status；
- objective；
- primal bound；
- dual bound；
- final gap；
- wall-clock time；
- presolve time；
- solving time after presolve；
- number of nodes；
- LP iterations；
- primal-dual integral；
- first feasible solution time；
- number of variables；
- number of integer variables；
- number of constraints。

建立 baseline 方法：

- default；
- relpscost；
- random；
- most-infeasible；
- strong branching（仅小规模）。

所有 baseline 必须使用相同：

- 实例；
- time limit；
- thread 数；
- seed 集合；
- presolve；
- cuts；
- restart；
- heuristics；
- node selection。

先在至少一个小实例上运行 smoke test，再在所有已有规模上运行 baseline。

输出：

results/baseline/raw/
results/baseline/summary.csv
docs/baseline_report.md

验收条件：

- 原始 default 模式结果与修改前一致；
- 输出字段完整；
- 相同 seed 可复现；
- 不同 branching 方法确实生效；
- objective 和解可行性无异常。

==================================================
五、阶段 2：构建航空布线实例数据集
==================================================

检查项目是否已有多个可变参数实例。

如果已有实例生成器：

- 为其增加 deterministic seed；
- 批量生成 small / medium / large；
- 保存生成参数和 metadata。

如果只有少量固定实例：

- 分析哪些输入参数可以变化；
- 在不改变问题定义的前提下构造一组同类实例；
- 不要通过随意扰动目标函数制造无意义数据。

优先变化：

- 线缆数量；
- 图节点数和边数；
- 候选路径数量；
- 约束密度；
- 网络拓扑；
- 障碍或容量配置；
- bundle 结构；
- 其他当前项目真实存在的参数。

每个实例导出：

data/instances/<split>/<instance_id>.cip
data/instances/<split>/<instance_id>.mps（可选）
data/instances/<split>/<instance_id>.json

split 包括：

- train；
- validation；
- test；
- transfer。

要求：

- train/validation/test 为同规模但不同原始场景或 seed；
- transfer 为更大规模或结构分布变化实例；
- 当前最大数据集不能进入训练集；
- 不允许数据泄漏。

输出：

data/instances/manifest.csv
docs/dataset_report.md

manifest 记录：

- split；
- instance_id；
- seed；
- 规模类别；
- 业务参数；
- 变量数；
- 整数变量数；
- 约束数；
- baseline 求解时间；
- baseline 节点数。

==================================================
六、阶段 3：实现自定义 branching rule 的无模型版本
==================================================

先只验证 SCIP 插件接入，不训练模型。

根据当前 SCIP 版本，使用 C API branching rule 或 scip::ObjBranchrule。

建议新增或等价实现：

src/rl/rl_branchrule.hpp
src/rl/rl_branchrule.cpp
src/rl/scip_feature_extractor.hpp
src/rl/scip_feature_extractor.cpp
src/rl/model_runner.hpp
src/rl/model_runner.cpp

实际路径适配当前项目。

branching callback 中必须：

1. 使用当前版本正确的 API 获取 LP branching candidates；
2. 读取候选变量及其 LP solution；
3. 选择一个合法变量；
4. 调用 SCIPbranchVar 或 SCIPbranchVarVal；
5. 返回正确 SCIP_RESULT；
6. 没有候选或发生异常时返回 DIDNOTRUN。

先实现：

--branching custom-random
--branching custom-mostinf

并记录每次分支：

- node id；
- depth；
- candidate count；
- selected variable name/index；
- LP value；
- fractionality；
- selection time；
- fallback reason。

不要在正式大实验中逐分支打印到 stdout，使用可关闭的结构化日志。

验收条件：

- 自定义 random 可以完整求解小实例；
- 所选变量始终属于当前 candidate set；
- 最终 objective 与 default 一致；
- 无内存泄漏；
- RL 关闭时不引入额外时间开销；
- 插件优先级和 fallback 行为符合预期。

==================================================
七、阶段 4：实现 BBMDP 训练环境
==================================================

优先考虑从 C++ 导出的 CIP/MPS 文件建立 Python 训练环境。

先评估能否复用 abfariah/bbmdp 官方实现中的：

- Branching environment；
- NodeBipartite observation；
- RetroNodeBipartite；
- SearchTree；
- RewardAgent；
- TreeDQNAgent；
- TreeDQNLearner；
- DQNPolicy；
- replay buffer；
- HL-Gauss loss。

不得直接复制后假设可用，需要检查许可证、依赖版本和当前 SCIP 兼容性。

环境定义：

状态：
- 当前 MILP 变量—约束二部图；
- 当前 incumbent 和全局 B&B 特征。

动作：
- 当前 fractional LP branching candidates。

奖励优先使用：
r_t = -(N_{t+1} - N_t)

同时允许配置：
- constant -1；
- negative node increment。

终止：
- optimal；
- open nodes empty；
- time limit；
- node limit；
- SCIP error。

BBMDP-faithful 训练配置：

- DFS node selection；
- restart disabled；
- cuts beyond root disabled；
- single thread；
- fixed seed；
- gamma = 1。

实现 transition 单元测试：

- action 必须属于 action_set；
- next observation 维度正确；
- reward 与节点增加一致；
- terminal bootstrap 为零；
- variable index 与 candidate index 映射正确；
- 不保留失效的 SCIP pointer；
- timeout episode 有明确处理方式。

==================================================
八、阶段 5：先训练 Candidate MLP-DQN
==================================================

不要立即部署完整 GCNN。

实现一个共享权重 Candidate MLP：

输入：
- 每个候选变量的特征；
- 当前全局 B&B 特征；
- 航空布线变量类别 one-hot。

输出：
- 每个候选变量一个 scalar Q value。

必须支持 action mask。

训练使用：

- Double DQN；
- target network；
- 3-step return；
- gamma = 1；
- replay buffer；
- epsilon-greedy；
- gradient clipping；
- Adam；
- validation early stopping。

超参数通过 YAML/JSON 配置，不要硬编码。

至少实现三个配置：

configs/rl/smoke.yaml
configs/rl/pilot.yaml
configs/rl/full_mlp.yaml

训练日志记录：

- episode；
- gradient step；
- loss；
- TD error；
- epsilon；
- reward；
- episode nodes；
- episode solving time；
- validation nodes；
- validation time；
- replay size；
- Q-value mean/std；
- selected candidate rank；
- GPU/CPU memory。

Smoke test：

- 少量小实例；
- 500～2000 steps；
- 只验证闭环和 checkpoint。

Pilot：

- 小中型实例；
- 5k～20k steps；
- 比较 untrained、random、RL。

模型保存：

artifacts/models/mlp/
- best_model.pt 或 .onnx；
- last_model；
- config；
- feature_schema.json；
- normalization.json；
- training_history.csv。

feature_schema 必须明确每一维特征，保证 Python 和 C++ 一致。

验收条件：

- loss 无 NaN；
- 模型能够保存和重新加载；
- 相同 observation 的 Python 和导出模型结果一致；
- validation 至少明显优于 random；
- 如果不优于 random，先诊断，不要直接进入完整 GCNN。

==================================================
九、阶段 6：将 MLP 模型接入 C++ SCIP
==================================================

选择 TorchScript、LibTorch 或 ONNX Runtime，依据当前环境最容易稳定部署的方式。

要求：

- 模型在 solver 初始化时加载一次；
- branching callback 内只做特征提取和前向推理；
- 使用 eval/no-grad；
- 模型输出做 NaN/Inf 检查；
- 只在候选集合中 argmax；
- 记录推理时间；
- 模型失败时 fallback；
- 不启动外部 Python 进程；
- 不在每次回调中读取磁盘。

新增参数：

--branching rl-mlp
--rl-model <path>
--rl-device cpu|cuda
--rl-fallback relpscost|default
--rl-max-depth <int>
--rl-min-candidates <int>
--rl-log <path>

实现 Python/C++ parity test：

对固定保存的 observation：

- Python 输出完整 Q values；
- C++ 输出完整 Q values；
- 最大误差在设定 tolerance 内；
- argmax candidate 完全一致。

然后在小实例上完成端到端求解。

验收条件：

- objective 与 default 一致；
- 所有分支动作合法；
- fallback 正常；
- 推理时间已记录；
- 多次运行无崩溃；
- C++ 与 Python 输出一致。

==================================================
十、阶段 7：实现完整 Bipartite GCNN-BBMDP
==================================================

只有 MLP 闭环成功后才实施。

状态图：

- variable nodes；
- constraint nodes；
- variable-constraint edges。

变量特征至少包括：

- variable type；
- objective coefficient；
- LP solution；
- fractionality；
- local/global bounds；
- reduced cost；
- pseudocost；
- branch priority/history；
- 当前 depth、gap、bounds、open nodes 等全局特征；
- 航空布线变量类别。

约束特征至少包括：

- lhs/rhs；
- activity；
- slack；
- dual；
- equality indicator；
- 航空布线约束类别。

边特征：

- coefficient；
- normalized coefficient；
- sign。

网络：

- variable embedding；
- constraint embedding；
- variable-to-constraint convolution；
- constraint-to-variable convolution；
- per-variable Q head；
- candidate mask。

先实现 scalar Q regression 版本，再实现可选 HL-Gauss 版本。

HL-Gauss 版本：

- 输出 18 bins；
- 支持从 scalar target 生成 histogram；
- 支持从 distribution 还原期望 Q；
- 对大范围负 Q 值做 log2(-Q) 变换；
- 配置 zmin、zmax、sigma。

训练：

- Double DQN；
- 3-step return；
- PER；
- soft target update；
- epsilon + 可选 Boltzmann exploration。

模型导出前必须验证所用 PyTorch Geometric 操作是否可被 TorchScript/ONNX 支持。

如果无法稳定导出：

1. 将消息传递改写为基础 torch index_add/scatter；
2. 或在 C++ 中手动实现两轮 sparse message passing；
3. 不允许使用每次分支调用 Python 子进程作为最终方案。

==================================================
十一、阶段 8：完整实验
==================================================

执行两套实验协议。

A. controlled_bbmdp：

- DFS；
- no restart；
- cuts only at root；
- one thread；
- fixed solver parameters。

B. production_scip：

- 保留原 scip_tree.cpp 求解设置；
- 只替换 branching variable selection。

方法：

- default；
- relpscost；
- random；
- mostinf；
- strong（小实例）；
- rl-mlp；
- rl-gcnn；
- 可选 rl-gcnn-hlgauss。

数据：

- validation；
- held-out test；
- transfer；
- 当前最大实例。

每个方法每个实例至少运行 5 个 seed，资源不足时先执行 pilot，但最终报告必须明确实验规模。

所有实验写入统一 CSV：

results/final/raw_results.csv

字段至少包括：

instance_id, split, size, method, seed, status,
objective, primal_bound, dual_bound, gap,
wall_time, presolve_time, solve_time,
nodes, lp_iterations, primal_dual_integral,
first_solution_time, branch_decisions,
rl_inference_total, rl_inference_mean,
rl_inference_max, fallback_count,
n_vars, n_int_vars, n_constraints

聚合：

- solved rate；
- shifted geometric mean time；
- geometric/shifted geometric mean nodes；
- median；
- mean final gap；
- paired speedup；
- wins；
- average rank；
- bootstrap 95% CI；
- timeout 的 PAR-2。

绘图：

- training curve；
- cactus plot；
- performance profile；
- per-instance wall-time speedup scatter；
- node reduction vs time reduction；
- inference overhead；
- ID vs transfer comparison；
- 不同规模上的性能变化。

所有绘图脚本可复现，保存到：

results/final/figures/

==================================================
十二、消融实验
==================================================

至少完成：

1. MLP vs GCNN；
2. one-step vs three-step TD；
3. scalar MSE vs HL-Gauss；
4. DFS vs 项目默认 node selector；
5. 不含航空布线变量类别 vs 包含变量类别；
6. 无全局树特征 vs 有全局树特征；
7. 不同最大 RL 深度；
8. 纯 RL vs 浅层 RL + 深层 relpscost；
9. 推理开销关闭/模拟分析。

浅层混合策略：

- depth <= D 使用 RL；
- depth > D 使用 relpscost。

测试 D ∈ {5, 10, 20, 50, unlimited}。

这可能在保留 RL 高层决策价值的同时降低推理开销和分布外风险。

==================================================
十三、正确性和失败诊断
==================================================

对所有 jointly solved 实例检查：

abs(obj_rl - obj_default) <= tolerance * max(1, abs(obj_default))

检查：

- status；
- solution feasibility；
- variable bounds；
- constraint violation；
- branching candidate legality；
- memory；
- fallback；
- model output；
- feature normalization；
- Python/C++ parity。

如果 RL 没有超过 SCIP-default，不得只写“模型效果不好”，必须分析：

- 是否明显优于 random；
- 是否只减少 nodes 但增加 time；
- 推理占比；
- 候选变量数量；
- 训练数据量；
- train/test 分布差异；
- 小/中/大实例性能；
- DFS 与生产 node selector 差异；
- pseudocost 特征是否正确；
- 航空布线变量类别是否有用；
- 是否过拟合；
- 是否因 SCIP 内置 branching 的额外强化机制导致节点比较不公平；
- 是否需要 IL warm start；
- 是否适合只在浅层使用 RL。

==================================================
十四、成功判据
==================================================

硬性正确性要求：

- 原 default 模式无回归；
- objective 无错误；
- action 100% 合法；
- fallback 可用；
- 无崩溃和明显内存泄漏。

学习效果要求：

- RL 明显优于 random；
- validation 曲线有改善；
- 多个 seed 结果方向一致。

航空布线可行性建议判据：

- held-out test solved rate 不下降；
- wall-clock shifted geometric mean 比 default 改善至少 5%；
- 或节点数改善至少 10%，且 wall-clock 不明显恶化；
- RL 推理时间占总求解时间低于 5%；
- transfer 不出现大面积灾难性退化；
- 最大实例至少在 time/gap/nodes 中有一项得到实际改善。

如果没有达到，也必须形成可信的负面结论，说明当前 RL branching 对航空布线项目暂不可行的原因和下一步建议。

==================================================
十五、最终交付物
==================================================

代码：

- 可切换 branching rule 的 scip_tree；
- 自定义 RL branchrule；
- 特征提取；
- Python RL 环境；
- MLP-DQN；
- GCNN-DQN；
- 训练和评测脚本；
- 模型导出和 C++ 推理；
- 单元测试和集成测试。

文档：

docs/rl_branching_audit.md
docs/baseline_report.md
docs/dataset_report.md
docs/training_report.md
docs/integration_report.md
docs/FINAL_RL_BRANCHING_REPORT.md

数据与结果：

data/instances/manifest.csv
results/baseline/
results/final/raw_results.csv
results/final/summary.csv
results/final/figures/
artifacts/models/

README 必须补充：

- 环境安装；
- SCIP/Python/CUDA 版本；
- 编译命令；
- 数据生成命令；
- baseline 命令；
- 训练命令；
- 模型导出命令；
- RL-SCIP 求解命令；
- 完整实验复现命令。

最终报告必须明确回答：

1. RL 是否减少 B&B 节点数？
2. RL 是否减少真实 wall-clock time？
3. 推理开销是多少？
4. 是否在未见实例上有效？
5. 是否能迁移到最大航空布线实例？
6. MLP 和 GCNN 哪个更合适？
7. DFS 训练是否能迁移到项目默认 node selector？
8. 与 SCIP-default 相比是否具有实际部署价值？
9. 如果没有，主要瓶颈是什么？
10. 下一步最值得进行的改进是什么？

现在从阶段 0 开始。先完整读取工作空间、运行原项目并生成 docs/rl_branching_audit.md。在审计完成前，不要直接重写 scip_tree.cpp 或安装大规模新依赖。