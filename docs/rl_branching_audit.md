# SCIP 强化学习分支阶段 0 审计报告

审计日期：2026-08-03（UTC）  
工作区：`/home/duweiyue25/acyclic_cable_harness`  
Git 提交：`7a675433c813412f817af50c7b168a527af147c3`（`main`）  
执行环境：所有 Python 和编译探测均通过 Conda 环境 `rl4scip` 执行。

## 1. 审计结论

当前仓库包含航空布线 MILP 源码和多组 CSV 数据，但不包含构建系统、依赖声明、运行脚本或有效的使用文档。`rl4scip` 环境本身没有 SCIP 包，但用户目录 `/home/duweiyue25/SCIP/scipoptsuite-8.0.4/` 中存在完整的 SCIP 8.0.4 优化构建。该构建没有加入 `PATH`、include path 或 library path，因此仅使用环境默认路径的首次编译真实失败；补齐显式路径后，`code/scip_tree.cpp` 已成功编译并完成 smoke 与默认配置求解。

默认路径编译探测真实失败于：

```text
fatal error: scip/scip.h: No such file or directory
```

显式使用 SCIP 8.0.4 后，原始默认配置得到：

```text
SCIP Status        : problem is solved [optimal solution found]
Solving Time (sec) : 144.49
Solving Nodes      : 154
Primal Bound       : 2.93411
Dual Bound         : 2.93411
Gap                : 0.00 %
Total best res     : 3.28268
Cycle check        : No cycle
```

关键日志：

- `results/audit/compile_code_scip_tree.log`
- `results/audit/compile_project_scip_tree.log`
- `results/audit/compile_code_scip_tree_8.0.4.log`
- `results/audit/compile_project_scip_tree_8.0.4.log`
- `results/audit/run_original_smoke.log`
- `results/audit/run_original_default.log`
- `results/audit/scip_plugins.log`
- `results/audit/environment.log`
- `results/audit/dependency_probe.log`

SCIP 8.0.4 插件表确认当前默认 branching rule 是 `relpscost`，默认 node selector 是 `estimate`。本阶段只完成一组 default golden run，不把它冒充多 seed baseline，也没有任何 RL 结果。

## 2. 初始 Git 状态

审计开始时仓库已经存在用户修改。阶段 0 未恢复、覆盖或提交这些内容：

```text
 D dataset/edges.csv
 D dataset/pairs-3-40.csv
 D dataset/pairs-4-246.csv
 D dataset/pairs.csv
?? dataset/edges-1.csv
?? dataset/edges-5.csv ... dataset/edges-9.csv
?? dataset/network_data.rar
?? dataset/pairs-1.csv
?? dataset/pairs-3.csv ... dataset/pairs-9.csv
?? related_paper/
```

`code/`、`project/` 和论文源码在审计开始时没有已跟踪文件改动。本阶段没有执行 Git commit 或 push。

审计运行期间，`code/data/` 在 07:21 UTC 出现了非本阶段命令产生的并发变化：原有 `edges.csv`、`pairs.csv`、缩放数据和若干辅助文件被移入未跟踪的 `code/data/synthesis/`，目录顶层改为 1 至 9 号数据布局。阶段 0 没有恢复或整理这些文件。默认求解在变化前已完成 CSV 读取和模型构造，导出的 LP 哈希及求解日志仍可复核；但当前目录状态下再次无参数运行会因 `code/data/edges.csv` 不再存在而失败。进入阶段 1 前需要确认以新的 1 至 9 号布局还是 Git HEAD 布局为准。

## 3. 项目结构

| 路径 | 作用 | 审计判断 |
|---|---|---|
| `code/scip_tree.cpp` | 原始分层无环布线 MILP | 可作为最早 baseline 源码；默认读取 1 号 11 列数据 |
| `project/scip_tree.cpp` | 后续业务版本，增加大小写归一化和 XLS 输出 | Git 历史更新较新，但默认 4 号输入与解析列不匹配，且依赖未安装的 `xlslib` |
| `code/scip_heur.cpp` | 带最短路预处理、固定部分决策的启发式 MILP 版本 | 不是纯 branching baseline，不应替代 `scip_tree` 主线 |
| `code/data/` | 1 至 4 号数据、合成缩放数据、先验解 | 当前主要可见数据源 |
| `dataset/` | 1 至 9 号场景及压缩包 | 5 至 9 号为未跟踪新增数据；格式不完全统一 |
| `dataset/dataext.py` | 合成图生成器 | 无 deterministic seed，固定参数写在脚本中 |
| `paper/` | 项目论文及数学模型说明 | 对变量和业务目标的解释与源码大体对应 |
| `related_paper/` | BBMDP 等相关论文 | 当前未跟踪 |
| `README.md` | 项目说明 | 只有项目标题，没有构建或运行方法 |

完整目录快照保存于 `results/audit/directory_structure.log`。

## 4. 构建与运行链路

### 4.1 仓库中实际存在的构建方式

仓库没有以下任一文件：`CMakeLists.txt`、`Makefile`、构建 shell 脚本、`environment.yml`、`requirements.txt` 或 `pyproject.toml`。因此不存在可直接复现的“当前项目编译命令”。

根据源码依赖，阶段 0 重建并执行了以下可用编译命令：

```bash
conda run -n rl4scip g++ -std=c++17 -O2 code/scip_tree.cpp \
  -I/home/duweiyue25/SCIP/scipoptsuite-8.0.4/scip/src \
  -I/home/duweiyue25/SCIP/scipoptsuite-8.0.4/build/scip \
  -L/home/duweiyue25/SCIP/scipoptsuite-8.0.4/build/lib \
  -Wl,-rpath,/home/duweiyue25/SCIP/scipoptsuite-8.0.4/build/lib \
  -lscip -o results/audit/scip_tree_code_probe
```

该命令成功。SCIP CLI 报告版本 8.0.4、SoPlex 6.0.4、GCC 11.4.0、optimized/thread-safe shared build。另有孤立的 `/home/duweiyue25/SCIP/lib/libscip.so.9.0.0.0`，但没有在同一路径发现对应头文件和 CLI，本项目不应混用它。

`project/scip_tree.cpp` 在显式解决 SCIP 路径后继续失败于 `xlslib.h: No such file or directory`。因此当前可运行主线是 `code/scip_tree.cpp`；较新的 `project/` 版本还需要处理 XLS 依赖和输入 schema，不能直接作为已验证 baseline。

### 4.2 源码定义的运行方式

源码只有位置参数，没有统一 CLI：

```bash
# 预期从 code/ 目录运行，否则相对 data/ 路径失效
./scip_tree [copy_num] [div_part]
```

- `copy_num` 默认 4，表示可选布线层/副本数。
- `div_part` 默认 1，只保留 `unique_center_pairs / div_part` 个中心点对。
- `code/scip_heur.cpp` 的第二个参数含义不同，是 MIP gap。
- 没有 instance、seed、time limit、node limit、threads、output JSON、build-only 等参数。

`code/scip_tree.cpp` 会在求解前写 `./save/model_scip_tree_data4+cp<copy_num>.lp`，但仓库没有 `code/save/`。阶段 0 在临时工作目录创建了 `data` 软链接和 `save/`，没有修改源码目录。默认运行实际导出了 9.8 MiB LP，其中精确记录 52,708 个变量、1,700 个 binary 和 135,556 条约束。`project/scip_tree.cpp` 注释掉了该导出，却会向不存在的 `./result/` 输出 XLS。

### 4.3 原项目实测

| 运行 | 参数 | SCIP solve time | 节点 | LP iterations | SCIP objective | 业务总目标 | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| smoke | `1 20` | 0.11 s | 1 | 626 | 0.02028 | 0.36885 | optimal, no cycle |
| 原始默认 | 无参数，即 `4 1` | 144.49 s | 154 | 20,745 | 2.93411 | 3.28268 | optimal, no cycle |

默认运行 presolve 用时 2.16 秒，将 52,708 个原始变量和 135,556 条原始约束降至 10,161 个变量和 38,841 条约束。根节点处理约 29 秒，最终进行了 1,574 次 strong-branch probes，峰值日志内存约 562 MiB。当前程序没有单独统计 wall time、first feasible time 或 primal-dual integral，阶段 1 需补齐。

### 4.4 建议的阶段 1 构建链路

新增顶层 CMake，而不是继续依赖手写编译命令：

```text
CMake -> find_package(SCIP CONFIG REQUIRED)
      -> scip_tree executable
      -> optional XLS output dependency
      -> later: rl_branchrule and model runtime libraries
```

应先确定一个 canonical 主入口。建议以更新的 `project/scip_tree.cpp` 为业务主线，同时先修复其输入 schema 检查和可选 XLS 依赖；使用 `code/scip_tree.cpp` 的 1 号场景作为修改前行为参考。两个文件不能长期独立演进，否则 baseline 与 RL 可能运行在不同模型上。

## 5. 环境清单

| 项目 | 实测值 |
|---|---|
| OS | Ubuntu 22.04.4 LTS, Linux 5.15 |
| CPU | 2 x Intel Xeon Gold 6148，共 80 逻辑 CPU |
| 内存 | 256 GiB，无 swap |
| Python | 3.11.15，来自 `rl4scip` |
| C++ 编译器 | GCC/G++ 11.4.0 |
| GPU | 4 x Tesla V100-SXM2 32GB |
| NVIDIA 驱动 | 580.159.03 |
| 驱动报告 CUDA | 13.0 |
| CUDA toolkit / `nvcc` | 未安装 |
| SCIP | 8.0.4，用户目录外部构建，未安装进 `rl4scip` |
| PySCIPOpt / Ecole | 未安装 |
| PyTorch / PyG / ONNX Runtime | 未安装 |

`rl4scip` 当前 pip 包只有 `packaging`、`pip`、`setuptools` 和 `wheel`，Python 训练栈基本为空。C++ 编译和运行命令仍由 `conda run -n rl4scip` 执行，只是显式链接用户目录中的 SCIP 8.0.4。GPU 可用于后续训练，但 V100 与具体 PyTorch CUDA wheel 的兼容性必须在安装时实测，不能仅依据驱动显示的 CUDA 13.0 选择包。

## 6. `scip_tree.cpp` 求解流程

### 6.1 SCIP 生命周期

`SolveMIPProblem` 中依次执行：

1. `SCIPcreate(&scip)` 创建求解器。
2. `SCIPincludeDefaultPlugins(scip)` 注册默认插件。
3. `SCIPcreateProbBasic` 创建 `MIPProblem`。
4. 创建变量、目标和线性约束。
5. `SCIPsolve(scip)` 启动求解。
6. 读取 status、primal bound 和最优解变量。

非最优分支调用了 `SCIPfree(&scip)`，但成功路径的 `SCIPfree` 被注释，变量和约束也没有 `SCIPreleaseVar` / `SCIPreleaseCons`。因此当前成功求解路径存在明确的资源释放缺口。自定义 branchrule 前应先用最小改动建立 RAII 或统一清理路径，否则重复训练 episode 会放大泄漏。

### 6.2 变量类别及业务含义

| 变量前缀 | 类型 | 索引 | 含义 |
|---|---|---|---|
| `m_k_p` | binary | 中心点对 `k`、层 `p` | 将一组等价线缆需求分配到一个布线层 |
| `z_i_j_p` | binary | 有向中心边、层 | 该层是否选择该方向的拓扑边，是主要 B&B 整数决策 |
| `f_i_j_k` | continuous [-1,1] | 有向边、商品/中心点对 | 第 `k` 类线缆在边上的流量方向和大小 |
| `absf_i_j_k` | continuous [0,1] | 无向边、商品 | 线性化 `abs(f)`，用于路径成本 |
| `y_i_p` | continuous [0,1] | 中心节点、层 | MTZ 风格拓扑顺序，辅助消除环 |
| `x_i_j_k` | 仅声明/注释 | 边、商品 | `scip_tree` 中未实际创建；`scip_heur` 中存在 |

航空布线节点由 CSV 中设备接口、中心/中继节点和入口节点组成。源码把以 `N`/`M` 开头的中心节点以及部分 `E` 节点纳入中心图，叶节点通过唯一接入边映射到中心图。`project/` 版本进一步要求 `N`/`M` 后全部为数字，避免把业务端子名误识别为中心节点。

### 6.3 目标函数

SCIP 内部最小化：

```text
sum_k sum_(i,j) edge_length(i,j) * aggregated_pair_weight(k) * absf(i,j,k)
```

求解后 `ret_compose` 再把叶节点到中心节点的固定接入段成本加回，输出 `Total best res`。因此阶段 1 的 JSON 必须同时区分：

- `scip_objective`：SCIP 实际优化的中心图目标；
- `business_total_objective`：补回固定接入段后的业务总成本。

否则 baseline 的 objective 容易混淆。

### 6.4 约束类别

| 约束名 | 作用 |
|---|---|
| `fforbid` | 禁止某线缆对使用与端点无关的入口边 |
| `abs1`, `abs2` | 约束 `absf >= f` 和 `absf >= -f` |
| `flow_balance` | 每个中心点对的单位流守恒 |
| `flow_symmetry` | 约束 `f_ij + f_ji = 0` |
| `onlym` | 每个中心点对恰好分配到一个层 |
| `imbalance` | 使前一层的分配数量不小于后一层，减少层对称性 |
| `zlower` | 把某层中的流使用与拓扑边 `z` 关联 |
| `topo_seq1/2` | 基于 `y` 的有向无环顺序约束 |
| `only_father` | 每个节点每层最多一个入边，形成森林式拓扑 |

容量约束和显式 single-direction 约束在 `scip_tree` 中被注释。双向 `z` 通常会被两条拓扑顺序约束共同排除，但需要通过导出模型与可行解检查验证，不能仅凭注释认定正确。

发现一个需要阶段 1 先做回归测试的建模风险：`zlower` 只使用无向边固定方向对应的 `f(i,j,k)`；当实际流以负值表示反方向时，该关联可能变弱。阶段 0 不改数学模型，但应把该问题纳入可行性与 objective 对照测试，避免把模型问题误判为 branching 策略效果。

## 7. 当前 SCIP 搜索配置

下表严格区分源码设置与运行实测：

| 项目 | 源码可确认 | 运行实测 |
|---|---|---|
| 默认插件 | 调用 `SCIPincludeDefaultPlugins` | SCIP 8.0.4 默认插件已注册 |
| branching rule | 未显式提升具体 rule，只设置 `branching/preferbinary=true` | `relpscost` priority 10000，为最高优先级；默认运行有 1,574 次 strong-branch probes |
| node selector | 未设置，注释中曾考虑 `hybridestim` | `estimate` std priority 200000，为普通内存模式最高优先级 |
| presolve | 未关闭 | 默认启用；默认运行 8 rounds、2.16 s |
| cuts/separation | `scip_tree` 未覆盖默认值 | 默认启用；根节点日志产生 37 cuts |
| restart | 未设置 | `estimation/restarts/restartpolicy=e`；本次仅 154 nodes，未达到默认 1000-node restart 门槛 |
| heuristics | 默认启发式，并把 RENS/ALNS 的 freq 和 priority 改高 | 启用；`zirounding` 在 139 s 找到首个日志可见可行解 |
| threads | `parallel/minnthreads=4`、`parallel/maxnthreads=16`、`lp/threads=4` | 参数设置成功，明确不是受控单线程配置 |
| solve API | `SCIPsolve`，并非注释掉的 `SCIPsolveConcurrent` | 标准 `SCIPsolve` 完成 optimal solve |

上述 branching/node selection 结论来自实际使用的 SCIP 8.0.4 插件表与优先级，而不是依据其他版本推测。完整插件表见 `results/audit/scip_plugins.log`，默认参数见 `results/audit/scip_8.0.4_default.set`。阶段 1 切换 baseline rule 时应通过 priority 修改并在每次输出中记录最终优先级，避免只接受 CLI 字符串却没有真正接管分支。

`code/scip_heur.cpp` 另行把 separation 和 presolve 调得更激进，并启用 `heuristics/fastprimal`。它不应与 `scip_tree` baseline 混跑。该文件还以整数 API 设置 `separating/minefficacy`，疑似参数类型错误且未检查返回码，是独立风险。

## 8. 数据集审计

### 8.1 实际场景规模

以下模型规模按 `copy_num=4` 和源码循环静态估算，约束数尚未包含 `fforbid`：

| 场景 | 原始边 | 图节点 | 独立中心点对 K | 中心边 | 估算变量 | 估算整数变量 | 估算约束 |
|---|---:|---:|---:|---:|---:|---:|---:|
| code-1 | 564 | 556 | 105 | 160 | 52,708 | 1,700 | 135,556 |
| code-2 | 244 | 218 | 181 | 150 | 83,878 | 1,924 | 214,744 |
| code-3 | 488 | 437 | 26 | 461 | 41,390 | 3,792 | 99,919 |
| code-4 | 641 | 546 | 194 | 549 | 326,502 | 5,168 | 840,023 |
| dataset-5 | 409 | 347 | 32 | 366 | 39,408 | 3,056 | 95,891 |
| dataset-6 | 339 | 295 | 17 | 330 | 20,682 | 2,708 | 47,936 |
| dataset-7 | 400 | 339 | 31 | 364 | 38,100 | 3,036 | 92,539 |
| dataset-8 | 376 | 322 | 22 | 349 | 27,094 | 2,880 | 64,233 |
| dataset-9 | 340 | 297 | 7 | 330 | 10,746 | 2,668 | 21,977 |

4 号场景的困难主要来自 `center_edges * K * copy_num` 相关变量和约束增长，而不是简单的 CSV 行数。它应进入最终 transfer/最大实例评估，不能进入训练集。

### 8.2 合成缩放数据

`code/data/` 还有 0.03x 至 4.0x 的合成文件，最大一组约 2,000 节点、2,040 边、200 对需求。但这些文件存在多种历史 schema 和节点命名规则，并非都能被当前 C++ 读取：

- 当前生成器写 6 列 pair，C++ 要求至少 7 列，结果会静默得到 `K=0`。
- 2、3、4 号 pair 是 7 列，C++ 却读取第 10 列权重，存在越界访问。
- 1 号 `code/data/pairs.csv` 是当前唯一与固定索引一致的 11 列默认输入。
- `dataset/pairs-1.csv` 的 11 列含义又与 `code/data/pairs.csv` 不同。
- 5 至 9 号是 10 列，和 `project/` 的新数据布局更接近。

完整行数和 schema 清单见 `results/audit/data_inventory.csv`。

### 8.3 生成器与划分风险

`dataset/dataext.py` / `code/data/dataext.py`：

- 使用模块级 `random`，没有设置或记录 seed；
- 参数和输出文件名硬编码；
- 通过中心链保证连通，再随机加中心边和叶节点；
- 随机生成边长和 pair weight；
- 没有 metadata、split、实例 ID 或哈希；
- 当前只生成 `scale_factors = [0.02]`。

阶段 2 可以复用其拓扑构造思想，但必须先统一 schema、增加显式 seed 和 metadata，并按原始实例划分 train/validation/test/transfer。现有文件不能直接随机拆 B&B 状态。

`dataset/network_data.rar` 因服务器没有 `file`、`unrar` 或 `7z` 工具，本阶段没有解包或修改，内容仍未知。

## 9. 日志、MILP 导出和插件接入可行性

- 当前没有节点、候选变量、branch decision 或统计 JSON 日志。
- `code/scip_tree.cpp` 已有 `SCIPwriteOrigProblem(..., "lp", FALSE)`，说明可在模型构造后导出；阶段 1 应扩展为 CLI 控制的 CIP，MPS 可选。
- `SCIPcreate`、默认插件注册和 `SCIPsolve` 都集中在 `SolveMIPProblem`，适合在默认插件后、建模前注册自定义 branchrule。
- 已直接检查本地 SCIP 8.0.4 头文件：它提供 `scip::ObjBranchrule`、`SCIPincludeObjBranchrule`、`SCIPgetLPBranchCands`、`SCIPbranchVar` 和 `SCIPbranchVarVal`。本地 `examples/Binpacking/src/branch_ryanfoster.c` 还给出了 C API 的 candidate 获取和插件注册范例。阶段 3 必须以这些本地签名为准。
- branchrule 应只接管 fractional LP candidates；异常或无候选时返回 `SCIP_DIDNOTRUN`，让 SCIP 的其他分支规则继续工作。

建议不要在现有 1,100 行文件中直接堆入模型推理代码。阶段 3 再引入：

```text
src/rl/rl_branchrule.hpp/.cpp
src/rl/scip_feature_extractor.hpp/.cpp
src/rl/model_runner.hpp/.cpp
```

`scip_tree.cpp` 只负责解析参数、注册插件和汇总统计。

## 10. BBMDP、SCIP、PySCIPOpt 与 Ecole 兼容性

### 10.1 官方 BBMDP 实现事实

官方仓库：<https://github.com/abfariah/bbmdp>

- 论文实验使用 SCIP 8.0.3。
- `INSTALL.md` 指定 PySCIPOpt 4.3.0。
- 训练环境基于 Ecole Branching、NodeBipartite 和 DFS。
- 代码已有 SearchTree、RewardAgent、TreeDQNAgent、TreeDQNLearner、PER、3-step、GCNN 和 18-bin HL-Gauss 的实现参考。
- 默认实验关闭 restart，非根节点 cuts 为 0，time limit 为 3600 秒。
- GCNN 使用 5 维约束、1 维边和 43 维变量特征，其中 24 维为额外树特征。

不能直接复制该仓库代码：截至审计时仓库根目录没有 `LICENSE`，GitHub License API 返回 404，源码头也没有许可证声明；`INSTALL.md` 提到的 `requirements.txt` 实际不在仓库中。应先取得许可证确认，或只依据论文和公开 API 独立实现。

### 10.2 当前维护版本事实

- 本项目当前实际 C++ 求解器为 SCIP 8.0.4，API version 104；BBMDP 使用 8.0.3，属于同一 8.0 minor line 的相邻补丁版本。
- SCIP 当前官方 release 为 10.0.2（2026-04-02）：<https://github.com/scipopt/scip/releases/tag/v10.0.2>
- PySCIPOpt 官方兼容表给出 SCIP 8.0 对应 PySCIPOpt 4.x；BBMDP 明确使用 4.3.0：<https://pyscipopt.readthedocs.io/en/stable/build.html>
- Ecole stable 文档为 0.8.x；当前 conda-forge 0.8.2 只构建 SCIP 9 和 SCIP 10 变体，因此要与本项目 SCIP 8.0.4 同栈，需要从源码构建兼容版本并做测试。
- Ecole 项目明确处于非活跃维护状态，只处理关键问题：<https://github.com/ds4dm/ecole>

项目当前 8.0.4 与 BBMDP 8.0.3 不同，但没有理由降级项目。所有最终 C++、PySCIPOpt 和训练环境应优先共享 8.0.4；8.0.3 只用于独立论文复现对照。

### 10.3 四种方案比较

| 方案 | 优点 | 风险 | 本项目建议 |
|---|---|---|---|
| A. 用项目当前 SCIP 自建环境 | C++ 与训练同为 8.0.4，搜索分布最一致 | 需要自己维护 environment callback 和树状态 | 推荐主线，也是最终 fallback 方案 |
| B. 独立容器跑 BBMDP 官方栈 | 精确贴近论文 SCIP 8.0.3，可快速验证算法概念 | 与最终 8.0.4 仍有补丁差异；容器工具尚未审计 | 可做 POC，不作为最终部署栈 |
| C. 使用对应 PySCIPOpt | 可直接读取 C++ 导出的 CIP/MPS，版本映射清楚 | 需从源码或兼容包安装 PySCIPOpt 4.x，并验证 Python 3.11 支持 | 推荐 8.0.4 + PySCIPOpt 4.3.x，先做最小 API 测试 |
| D. 使用 Ecole | Branching 环境和 NodeBipartite 可大幅缩短训练环境实现 | 当前 conda 包不再提供 SCIP 8 变体，Ecole 又非活跃维护 | 尝试从源码链接 8.0.4；失败则回到 A/C |

推荐顺序：保留已验证的 SCIP 8.0.4 C++ 栈；在 `rl4scip` 中安装对应 PySCIPOpt 4.x，并尝试将 Ecole 源码链接到同一 8.0.4。若 Ecole 对 Python 3.11 或本地 SCIP 构建不稳定，则通过 PySCIPOpt/C++ event handler 自建最小 BBMDP 环境。独立 8.0.3 容器仅用于 proof-of-concept，不替换项目求解器。

## 11. 推荐训练与 C++ 推理架构

```text
CSV scenario
  -> C++ build-only model construction
  -> versioned CIP + metadata + feature schema
  -> Python training environment (same SCIP major)
  -> Candidate MLP Double-DQN first
  -> validation and fixed-observation parity corpus
  -> ONNX export
  -> C++ branchrule + ONNX Runtime CPU inference
  -> fallback to relpscost/default
```

选择理由：

- CIP 保留 SCIP 模型语义和变量名，适合训练环境复用；MPS 只作为通用交换备份。
- MLP 先闭环，候选数变化通过共享 candidate encoder 和 mask 处理。
- 训练可使用 V100；branch callback 优先 CPU 推理，避免小张量 GPU 同步开销。
- ONNX Runtime 比在 C++ 中引入完整 LibTorch 更轻；最终决定仍以阶段 5 的导出兼容测试为准。
- GCNN 后续应把 PyG MessagePassing 改写为基础 `index_add/scatter`，再尝试 ONNX；禁止 callback 启动 Python 子进程。
- Python 与 C++ 共用 `feature_schema.json`、normalization、变量类别映射和固定 observation parity fixtures。

## 12. 计划修改文件

阶段 1 的最小范围：

- 新增顶层 `CMakeLists.txt`。
- 在 canonical `scip_tree.cpp` 中增加统一 CLI、seed、limits、method、JSON、build-only 和 export 参数。
- 增加轻量运行统计与 JSON 输出模块，避免继续膨胀主文件。
- 增加 `scripts/` 下可复现构建和 baseline 脚本。
- 新增 `results/baseline/` 和 `docs/baseline_report.md`。

阶段 3 以后才增加 `src/rl/` branchrule、feature extractor 和 model runner。阶段 4 以后才增加 Python RL 包、配置、训练脚本和模型产物。阶段 0 未修改任何核心源码。

## 13. 风险清单

| 等级 | 风险 | 处理建议 |
|---|---|---|
| 高 | SCIP 8.0.4 在环境外且依赖手写 include/library path | 阶段 1 用 CMake 固化路径和 runtime RPATH，不混用孤立 SCIP 9 库 |
| 高 | `project/` 版本缺少 `xlslib`，当前不可编译 | 将 XLS 设为可选依赖，先保留 `code/` 的已验证 baseline |
| 高 | 无构建系统，命令和依赖不可复现 | 阶段 1 建立 CMake 和锁定环境清单 |
| 高 | 审计期间 `code/data/` 被外部流程重排，默认输入路径现已缺失 | 保留现状，阶段 1 开始前由用户确认 canonical 数据布局 |
| 高 | `project/scip_tree.cpp` 默认 7 列输入却读取第 10 列 | 阶段 1 加 schema 校验与显式实例参数 |
| 高 | 成功求解不释放 SCIP，重复 episode 会泄漏 | 在训练前建立统一清理路径并做 Valgrind/ASAN smoke test |
| 高 | 原程序强制多线程，随机性和 baseline 不受控 | 阶段 1 强制实验模式单线程并记录 seed/参数 |
| 高 | `zlower` 对负方向流的关联可能不足 | 用导出模型、可行性和 objective 对照测试，不在 branching 改动中暗改模型 |
| 高 | BBMDP 官方代码没有明确许可证 | 不直接复制，先确认许可或独立实现 |
| 中 | BBMDP 8.0.3 与项目 8.0.4 存在补丁版本差异 | 优先在 8.0.4 训练和部署；8.0.3 只做 POC |
| 中 | Ecole 非活跃维护，树 API 可能随 SCIP 变化 | 先做 transition contract tests，保留 PySCIPOpt/C++ 备选 |
| 中 | 数据 schema、节点分类和命名规则不统一 | 阶段 1/2 统一 parser、metadata 和验证器 |
| 中 | 生成器无 seed，现有缩放数据来源不可复现 | 阶段 2 显式 RNG，并保存参数及文件哈希 |
| 中 | SCIP 目标与业务总目标不同 | 输出两个字段并在正确性检查中分别比较 |
| 中 | 强化学习可能只降 nodes、不降 wall time | 全程记录推理开销，优先 MLP 和浅层混合策略 |

## 14. 阶段 0 验收状态

| 验收项 | 状态 |
|---|---|
| 递归读取项目结构、核心源码、数据和论文模型 | 完成 |
| 记录 git status、编译器、Python、GPU | 完成 |
| 确认 SCIP 版本 | 完成：SCIP 8.0.4 / SoPlex 6.0.4 |
| 编译原项目 | 完成：`code/scip_tree.cpp` 成功；`project/` 缺 `xlslib` 的失败日志已保存 |
| 运行原项目 | 完成：smoke 与无参数默认配置均 optimal |
| 确认真实 branching/node selector | 完成：`relpscost` / `estimate` |
| 数据规模和 schema 清单 | 完成 |
| BBMDP/PySCIPOpt/Ecole 兼容方案 | 完成 |
| 推荐训练和 C++ 推理架构 | 完成 |
| 修改核心源码 | 未进行，符合阶段 0 要求 |

阶段 1 的进入条件是：用户批准依赖方案和 canonical 入口。推荐保留 SCIP 8.0.4，以 `code/scip_tree.cpp` 的默认结果 `2.93411 / 154 nodes / 144.49 s` 作为第一份 golden reference，再解决 `project/` 的 XLS 和 schema 问题。未得到批准前不安装依赖、不修改 `scip_tree.cpp`。
