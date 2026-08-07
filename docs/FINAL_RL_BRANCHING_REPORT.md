# 航空布线 SCIP 强化学习分支最终可行性报告

## 1. 结论摘要

本项目已经完成从航空布线真实 MILP、可复现 baseline、BBMDP 环境、Candidate
MLP-DQN、Bipartite GCNN-DQN，到 C++ SCIP 原生推理和双协议实验的完整闭环。
工程正确性通过，但当前模型**不满足生产部署判据**。

在 `production-scip` 的 35 个实例-seed 配对上：

| 方法 | solved rate | shifted gmean wall time | mean PAR-2 | 相对 default 成对 speedup (95% CI) |
|---|---:|---:|---:|---:|
| default | 25.7% | 21.329 s | 47.388 s | 1.000 |
| random | 31.4% | 20.615 s | 45.139 s | 1.077 (0.980, 1.196) |
| mostinf | 34.3% | 20.444 s | 44.078 s | 1.109 (1.024, 1.221) |
| RL-MLP | 28.6% | 21.301 s | 46.167 s | 1.021 (0.964, 1.089) |
| RL-GCNN | 37.1% | 23.984 s | 44.387 s | 0.966 (0.855, 1.095) |

MLP 的 shifted gmean wall time 仅比 default 好约 0.13%，远低于 5% 判据，置信区间
跨 1。GCNN solved rate 更高，但 wall time 比 default 恶化约 12.4%。在 jointly solved
样本中，MLP 和 GCNN 分别比 default 多用约 6.8% 和 19.9% 节点。

RL 也没有达到“明显优于 random”的学习要求。`controlled-bbmdp` 下 MLP/GCNN 相对
random 的成对 time speedup 分别为 0.856 和 0.793；`production-scip` 下分别为
0.948 和 0.896，后两者置信区间跨 1。受控 DFS 的 jointly solved 子集中，RL 约少
4% 节点，但该信号没有转化为时间优势或稳定迁移。

## 2. 实验范围

- 数据只使用新的 `code/data` 导出的真实实例，不使用 synthesis 训练或正式评测；
- 训练：`real_06`；验证：`real_08`；测试：`real_09`；
- transfer：`real_01` 至 `real_05`，最大 `real_04` 不进入训练；
- 两种协议、7 种方法、5 个 seed，主实验共 430 次；
- 模型消融 50 次，深度消融 100 次，阶段 8 正式 C++ 运行共 580 次；
- 每次 SCIP 单线程、30 秒限制、固定 seed，原始 JSON、SCIP log 和 branch CSV 全保留；
- 主实验中 132 次 optimal、298 次 time limit；2,454 次 RL branching action 全部合法；
- 模型消融 679 次、深度消融 327 次动作也全部合法。

`controlled-bbmdp` 使用 DFS、禁用 restart、只允许 root cuts；`production-scip` 保留
原项目 estimate node selector 和生产参数，只替换 branching variable selection。

## 3. 正确性

硬性正确性要求全部通过：

- 430 个主实验任务键完整且唯一，进程返回码全部为 0；
- 所有 optimal 解通过原项目可行性检查；
- jointly solved 实例的目标值在 `1e-8` 相对容差内一致；
- 所有 RL 选择都属于当前 fractional LP candidate set；
- 主实验和消融均没有非法 action、崩溃或意外 fallback；
- 浅层实验中的 fallback 均由深度门限触发，行为符合设计；
- 四个消融 checkpoint 均可重载，loss 有限，TorchScript argmax 与 eager 一致；
- 最终 Make 构建、custom branchrule、MLP/GCNN runner 构建通过；
- Python 测试 18/18 通过。

没有方法改动航空布线 MILP 的变量、约束或目标。未指定 RL 参数时仍进入原有
`production-scip/default` 路径。

## 4. MLP 与 GCNN

在生产协议中，GCNN 相对 MLP 的成对 time speedup 为 0.946，即 GCNN 平均更慢；
jointly solved 时 GCNN 也约多用 12.3% 节点。MLP 的总推理占 solving time 约
0.035%，GCNN 为 3.14%。但这个总体数字被大量“没有发生分支”的 transfer 运行
稀释：只看实际发生 RL branching 的生产运行，MLP 为 0.10%，GCNN 为 9.39%。

分布上差异更明显：

- `real_09` test：两者均 5/5 solved；MLP 平均约 2.06 秒，GCNN 约 4.06 秒，GCNN
  推理占 solving time 约 25.4%；
- `real_08` validation：default 4/5、MLP 4/5、GCNN 5/5；GCNN 推理占约 5.69%；
- medium transfer：default 0/15、MLP 1/15、GCNN 3/15；GCNN 存在搜索质量信号，
  但其 wall time 和节点数没有形成稳定改进。

因此，若必须在当前两个模型中选择工程部署候选，MLP 更合适：推理成本小、实现
简单、故障面较窄。但当前 MLP 的策略质量仍不足以替换 SCIP-default。

## 5. 模型消融

所有模型使用相同 `real_06`、seed 0、1000 gradient steps 和网络预算；C++ 评测使用
`real_08/real_09` 各 5 seed。

| 模型 | solved | mean PAR-2 | shifted gmean wall | 推理占比 |
|---|---:|---:|---:|---:|
| scalar 3-step | 10/10 | 14.391 s | 10.165 s | 7.68% |
| scalar 1-step | 10/10 | 14.266 s | 10.079 s | 7.56% |
| HL-Gauss 3-step | 8/10 | 20.634 s | 10.551 s | 7.75% |
| no aviation categories | 10/10 | 14.289 s | 10.085 s | 7.59% |
| no global features | 10/10 | 14.254 s | 10.076 s | 7.52% |

结论：当前训练预算下 3-step 没有优于 1-step；HL-Gauss 明显退化；航空类别与全局
树特征的差异约 1%，没有可靠贡献证据。由于每个变体只有一个固定训练 seed，这些
小差异应视为“不显著”，不能解释成应永久删除对应特征。

## 6. 深度混合策略

production 下使用 `real_08/09/04/05`、每个 D 五 seed：

| 最大 RL depth | solved | mean PAR-2 | 推理占比 | fallback |
|---|---:|---:|---:|---:|
| 5 | 8/20 | 40.307 s | 2.56% | 22 |
| 10 | 8/20 | 40.242 s | 2.61% | 17 |
| 20 | 10/20 | 37.107 s | 2.71% | 11 |
| 50 | 10/20 | 37.119 s | 2.81% | 9 |
| unlimited | 11/20 | 35.557 s | 2.84% | 0 |

D=5/10 只小幅降低推理，却损失 solved rate；D=20/50 也没有优于 unlimited。当前
模型的推理主要集中在浅层少数大图状态，单纯限制深度不能解决冷启动和特征提取
成本，也会过早丢失仍有价值的 RL action。

将实测 wall time 减去 `rl_inference_total` 的“零推理成本”模拟仍未改变总体排序，
说明 GCNN 问题不只是前向推理：策略造成的搜索路径、动态图提取与模型加载也重要。

## 7. 最大实例与 transfer

最大真实实例 `real_04` 有 326,502 个变量和 863,691 个约束。所有方法的五个 seed
均在预处理阶段达到时限，平均 presolve 约 27.4 秒，SCIP 节点数和 branch decision
均为 0，也没有 primal/dual bound。因此任何 branching rule 都没有机会执行，无法
满足“最大实例在 time/gap/nodes 至少一项改进”的目标。GCNN 即使没有 callback，
仍因 TorchScript/CUDA 初始化使平均 wall time 从 default 的 34.72 秒增加到 36.68 秒。

这一结果不是“RL 在最大实例搜索失败”，而是当前 30 秒协议无法进入 B&B。后续若
专门研究最大实例，应分别设置 build/presolve 和 B&B 预算，或预先导出 presolved
CIP，再从相同 presolved state 比较 branching。

中型 transfer 上 GCNN 将 solved 数从 default 的 0/15 提高到 3/15，其中
`real_01` 为 2/5、`real_05` 为 1/5。这是最值得保留的正面信号，但样本少、节点更多、
总时间未改善，尚不足以支持生产价值。

## 8. DFS 到生产节点选择的迁移

受控 DFS 下 GCNN solved rate 为 25.7%，生产 estimate 下为 37.1%；其相对 default
的成对 time speedup由 0.906 变为 0.966，但两者都未超过 1。说明训练策略在生产
node selector 下没有灾难性失效，并出现更多 solved transfer runs；但搜索状态分布
明显改变，受控环境中的小幅节点信号没有稳定转化为生产 wall-clock 收益。

当前不能认为 DFS 训练已经可靠迁移。更合适的训练方式是混合 node selector 课程，
或直接在 production 参数分布上 fine-tune，并将 wall-clock/推理成本纳入奖励或选择。

## 9. 未达到判据的原因

1. 真实训练实例只有 `real_06`，模型容易学习实例特定的候选排序；
2. 1000-step GCNN pilot 和单训练 seed 对高维图策略仍偏小；
3. SCIP-default 的 relpscost 已利用 pseudocost/strong-branching 信息，学习策略没有
   获得同等稳定的历史估计；
4. GCNN 每个 callback 构造大规模动态图，活跃分支运行中的推理占比约 9.4%；
5. 小实例的 CUDA 冷启动可占主要 solving time；
6. 很多 transfer 实例在 presolve/root 阶段耗尽预算，branching 的可影响空间很小；
7. DFS 训练与 production estimate node selection 存在状态分布偏移；
8. 当前 reward 主要优化节点增量，不能直接约束 wall-clock、LP 难度和推理成本；
9. 航空类别和全局特征在单训练 seed 下未显示稳定增益，特征利用不足。

## 10. 最终十问

1. **RL 是否减少 B&B 节点？** 没有稳定减少。production jointly solved 中 MLP/GCNN
   分别比 default 多约 6.8%/19.9%；受控 random 对比中仅有约 4% 的小幅减少。
2. **RL 是否减少 wall-clock？** MLP 约改善 0.13%，不显著；GCNN 恶化约 12.4%。
3. **推理开销？** production 总体 MLP 0.035%、GCNN 3.14%；只看活跃分支运行分别
   约 0.10% 和 9.39%，GCNN 未达到 5%。
4. **未见实例有效吗？** `real_09` 正确且稳定 solved，但 GCNN 约慢一倍；MLP 与
   default 时间接近。不能称为性能泛化。
5. **能迁移到最大实例吗？** 本协议下不能判断；`real_04` 没有进入 B&B。
6. **MLP 和 GCNN 哪个更合适？** 当前工程部署应选 MLP，但仍不应替换 default。
7. **DFS 训练能迁移到默认 node selector 吗？** 功能上能，性能上尚不可靠。
8. **相对 SCIP-default 有部署价值吗？** 当前没有达到 solved rate、5% 时间或 10%
   节点改进的组合判据。
9. **主要瓶颈？** 数据量、reward 与 wall time 不一致、SCIP 内置历史信息优势、GCNN
   动态图开销，以及大实例根本没有进入 branching。
10. **下一步最值得做什么？** 先构建更多真实同分布训练实例，并用 SCIP
    relpscost/strong branching 做 imitation warm start；随后在 production node
    selector 下 fine-tune 一个低成本 Candidate MLP，将 pseudocost、LP solve cost 和
    inference penalty 纳入目标。GCNN 优化应暂缓，除非先实现静态图缓存和增量特征。

## 11. 产物与复现

主结果：

- `results/final/raw_results.csv`：430 次统一结果；
- `results/final/summary.csv`、`paired_comparisons.csv`、`rl_vs_random.csv`；
- `results/final/instance_summary.csv`；
- `results/final/validation.json`；
- `results/final/figures/`：training curve、cactus、performance profile、speedup、
  node/time、inference、ID/transfer、scale 和四张消融图。

消融：

- `results/final/ablations/models/`：50 次模型消融及 validation；
- `results/final/ablations/depth/`：100 次深度消融及 validation；
- `results/final/ablation_summary.csv`、`ablation_contrasts.csv`；
- `artifacts/models/ablations/`：四个 best/last checkpoint、TorchScript、schema、
  normalization 和 training history。

完整命令已写入 `README.md`。所有工作均在 `rl4scip` 环境、本地工作区完成，没有
提交或推送 GitHub。
