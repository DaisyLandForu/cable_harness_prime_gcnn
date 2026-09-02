# Steiner RL Branching 研究契约 v1

状态：S00 冻结候选；只有修改本文件及对应 ADR/config、记录理由、提升版本并重新审计，才能改变本契约。

## 1. 研究边界

基础问题是连通无向正权图上的经典 Steiner Tree Problem in Graphs（SPG）。输入为 (G=(V,E))、严格正边权 (c_e) 和 terminals (T\subseteq V)，目标是求连接全部 terminals 的最小权重子图。基础阶段不把 directed、node-weighted、prize-collecting 或航空布线实例称为同一个问题。

S00--S09 的首选 formulation 是 rooted multi-commodity flow `rooted_mcf_v1`：

- 规范化后编号最小的 terminal 是 root；不得按验证或测试结果换 root。
- 每条无向边有唯一、可支持平行边的 `edge_id` 和一个二元 (x_e)。
- 对每个非 root terminal 建两个方向上的连续 commodity flow (f^t_{ij}\in[0,1])。
- 目标为 \(\min\sum_e c_e x_e\)，每个 commodity 满足 root 流出 1、目标 terminal 流入 1、其他点净流量 0。
- 每条无向边和 commodity 满足 (f^t_{ij}+f^t_{ji}\le x_e)。
- 学习策略只从 SCIP 当前合法、fractional、binary 的 `stp_x_e*` LP branch candidates 中选择；连续流变量不属于动作。
- SCIP 保留精确求解、可行性和 bound 责任；学习模型不生成或修补解。

SCF 不是默认实现。当 S03 的 MCF pilot 出现任一条件时才触发受控 SCF 决策：每实例连续流变量超过 1,000,000、模型建立时间 p95 超过 60 秒、单 worker RSS p95 超过 8,192 MB，或 6 workers 的预计总 RSS 超过 49,152 MB。触发不允许静默替换 formulation；必须登记新的 formulation ID，并把 formulation 作为实验因子。

## 2. 表示和模型契约

`B0` 是首个学习基线：Ecole/NodeBipartite 风格变量 19 维、约束 5 维、归一化系数边 1 维，一轮 variable-to-constraint-to-variable message passing，只给合法 `x_e` candidates 打分。航空变量/约束 one-hot、旧 Prim/DSU 六维、14 维扩展 row、3 维扩展 edge 和 14 维 global state 均不进入 B0。

任何增强都使用新 schema/version 并作为单独实验因子。原始图 GNN 与 late dual-view fusion 是后续消融，不是 B0 完成条件。S10 前不假定所有 Steiner-family 动作都能映射到边；变体必须使用 typed candidate 和 topology-valid mask。

当前仓库核实结果：尚无 `python/steiner_branching`；现有 `BipartiteGCNNQNetwork` 输入由 19 维 Ecole 变量特征、6 维航空 Prim 特征、6 类航空变量、14 维扩展 row、6 类航空约束、3 维 edge 和 14 维 global state 组成。现有 `prim_bias.py` 只解析 `z/m/y` 航空命名。因此旧栈不是已存在的 Steiner 接口，S01/S04 必须独立建立并重新测试。

## 3. 学习路线

1. strong branching teacher 只标注合法 `x_e` candidates；无效 child、tie 和采样失败全部保留在 manifest。
2. `B0` listwise/ranking imitation 是主要 learned baseline，也是 RL 初始化来源。
3. RL 主实验从 validation 选定且冻结的 IL checkpoint 初始化；from-scratch 使用相同网络、数据机会和计算预算，只作消融。
4. 在 S07 完成逐轨迹 state/action/transition/reward/terminal/truncation 语义审计前，方法只称 `branching_dqn_v1`，不得称 BBMDP 复现。
5. timeout/node limit/memory limit 是 truncation 或失败状态，不得伪装为自然 terminal。
6. replay 同时受 transition 数和实际字节预算限制；checkpoint 只由 validation solve 指标选择。
7. 多个 training seed 的失败、NaN、mapping failure 和 skipped 结果不得删除。

若 IL 不超过 random/most-infeasible 弱基线，不进入长期 RL。若语义审计后的 RL 不超过 IL，接受负结果并停止扩大 RL，不靠更多 GPU、网络层或只报告最好 seed 改写结论。

## 4. 数据、split 和封存

机器可读规则在 `configs/steiner/splits/split_policy_v1.yml`。最小约束如下：

- 按 `base_graph_lineage` 划分，同一 base graph 的 terminal、weight、relabeling 和 formulation 衍生物不得跨 split。
- B&B states 继承 instance split，禁止 state-level 随机切分。
- normalization 只使用 train。
- synthetic train/validation/test/OOD 使用互斥 seed 区间；图族覆盖要求写在 split 配置中。
- PACE Track 1/2 odd 是 public development，even 是 sealed final test。
- SteinLib family 整体分配；不得把同一 family 的困难样本事后挪走。
- DIMACS 11 official SPG competition bundle 是 sealed final test。
- 内容重复以 canonical instance SHA-256 去重，最严格 split 优先。

封存入口为 `configs/steiner/splits/final_test_v1.yml`，包含 106 个 canonical suite/member entries，selector hash 为 `8c0324c1a82485c2187825977fe2807e31512a6435e2f58f6a1d17babbfbddd1`。S00 时 `learning_runs_total=0`。S02 下载前后必须增加 archive/per-file content hashes，但不得改变 selector membership。第一轮 learned final-test 只能在 S12 代码、checkpoint、normalization 和 profile 全部冻结后运行；额外运行均标记 post-hoc，不能继续调参后称为原 final test。

## 5. 协议、资源和 seed

唯一机器可读入口为 `configs/steiner/experiments/protocols_v1.yml`：

- P0 `correctness-v1`：60 s、10,000 nodes、4,096 MB。
- P1 `controlled-branching-v1`：600 s、200,000 nodes、8,192 MB；关闭 presolve rounds、separation rounds、heuristics 和 restarts，固定 estimate node selector。
- P2 `generic-scip-v1`：600 s、200,000 nodes、8,192 MB；恢复 generic SCIP 默认搜索组件。
- P3 `scip-jack-external-v1`：1,800 s、无 node cap、16,384 MB；保留专业求解器自身组件。
- P4 `scip-jack-branching-hard-v1`：1,800 s、500,000 nodes、16,384 MB；hard subset 只按冻结的 native baseline branchability 选择。

所有 solver 单线程。pilot solver seed 为 `[0]`，formal solver seeds 为 `[0,1,2,3,4]`，formal training seeds 为 `[101,202,303,404,505]`，teacher collection seeds 为 `[1001,1002,1003]`，bootstrap seed 为 `20260902`。正式结果必须保留完整 instance × solver-seed 组合，并逐个报告所有 training seeds。

冻结环境选择 SCIP 8.0.4 / SoPlex 6.0.4 / Ecole 0.8.1 / PySCIPOpt 4.3.0 / PyTorch 2.5.1+cu121。实际探测和当前运行阻塞记录在 `configs/steiner/environment.lock.yml`。默认 `/usr/bin/scip` 9.2.2 不允许混入正式矩阵；升级必须用新 stack/profile ID 做兼容性实验。

## 6. Baseline 和指标

必须保留：SCIP default、relpscost、random candidate、most-infeasible、预算子集 strong branching、B0 imitation、IL-initialized RL、from-scratch RL。增强模型和 SCIP-Jack internal policy 只有对应阶段 Gate 通过后加入。learned policy 的主 solver reference 是 relpscost，SCIP default 作为 production reference；strong branching 不是全量低成本 baseline。

主指标按以下顺序解释，不得从中挑一个最好看的：

1. solved rate（越高越好）；
2. PAR-2（越低越好，未解/timeout/node-limit/OOM/solver-error/invalid-policy 均为两倍 time limit）；
3. primal-dual integral（越低越好）。

wall time shifted geometric mean 使用 1 s shift。必须同时报告 final gap、nodes、LP iterations、time to first incumbent、root gap、branch decisions、feature/inference/callback overhead、invalid/fallback/timeout/OOM 和超过 2 倍 catastrophic slowdown 的比例。只报告 solved instances 的均值无效。

比较按 instance/solver seed 配对；95% paired percentile bootstrap 以 instance 为 resampling unit，10,000 replicates，seed `20260902`。Wilcoxon + Holm 只作补充，不能代替 effect size。checkpoint 在 validation 上依次最大化 solved rate、最小化 PAR-2、最小化 PDI；training loss 和 final test 都不能选最终模型。

## 7. Gate 和主张纪律

S03 branchability 预注册阈值由 `protocols_v1.yml` 固定：至少 60% 实例有不少于 5 次合法 decision，非平凡实例 median 不少于 10，strong-branch valid states 不少于 60%，all-tie states 不超过 40%，action mapping 为 100%，worker RSS p95 不超过 8,192 MB。只允许在 learned results 前因测量定义 bug 修改一次，并完整记录理由。

任何正式 learned run 要求 invalid action、NaN、mapping failure、unexpected fallback 全为 0。最终部署 callback overhead 上限冻结为总 wall time 的 10%；若超过，不得只报去除开销的 solver time。

禁止：

- 比较 generic MCF/SCF 与 SCIP-Jack 的 nodes 并推断 branching 优劣；
- 把 P3 完整 solver 差异归因于 branching；
- 混淆 P1 机制结果与 P2/P3 实际性能；
- 用 final test 调 generator、网络、reward、阈值、checkpoint 或样本清单；
- 删除失败实例或 seed；
- 从 SPG-only 证据声称 Steiner-family 泛化；
- 在 typed action 尚未验证时把 edge Q 直接用于 vertex/arc branching。

工程 Gate 只说明阶段产物可信，不等于 learned policy 有效。每阶段 FAIL 时停止，不进入下一阶段；外部 GPT 审计通过后才允许合并到研究主线。
