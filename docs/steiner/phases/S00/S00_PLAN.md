# S00 研究契约与环境冻结计划

## 阶段目标

在任何 Steiner 数据生成、MILP 实现、采样或学习实验之前，冻结会影响研究结论的定义：SPG/MCF/action 契约、P0--P4 协议、资源限制、数据划分、final-test 封存、seed、baseline、指标、统计口径和环境事实。建立阶段状态与四份 ADR，并用自动化测试检查机器可读配置和封存清单的一致性。

## 非目标

- 不创建 `python/steiner_branching` 包，不实现 parser、generator、MCF、observation、模型、teacher 或 RL。
- 不下载或运行 PACE、SteinLib、DIMACS final test。
- 不训练、选择或导出 checkpoint。
- 不修改航空布线代码、构建产物、既有实验脚本或结果。
- 不开始 S01。

## 输入、依赖和假设

- 阶段开始时 UTC 日期：2026-09-02。
- 起始 branch：`main`；阶段工作 branch：`research/steiner-s00-contract`。
- base SHA：`88ade1ac614fb12f882a10ba9b5d35b15c7b4d01`。
- 主方案：`plans/STEINER_RL_BRANCHING_MIGRATION_MASTER_PLAN.md`（工作开始时为用户提供的未跟踪文件）。
- `docs/steiner/STATUS.md` 与 S00 计划在阶段开始时不存在；没有上一阶段审计。
- 主方案中的目标接口必须以当前仓库和环境探测结果核实，不假定已实现。

## 初始 Git 状态

开始时执行：

```text
git branch --show-current
git rev-parse HEAD
git status --short --branch
```

初始状态为 `main...origin/main`，除主方案外存在用户已有改动。保留清单如下；S00 不编辑、不恢复、不暂存这些无关路径：

```text
M  artifacts/environment/phase4/scip804_prefix/bin/{fscip,gcg,scip,soplex}
T  artifacts/environment/phase4/scip804_prefix/lib/{libgcg.so,libgcg.so.3.5,libscip.so,libscip.so.8.0,libsoplexshared.so,libsoplexshared.so.6.0}
M  build/**
M  results/audit/{scip_tree_code_probe,scip_tree_phase1}
M  scripts/analyze_c0_audit.py
M  scripts/analyze_c0_decomposition.py
M  scripts/analyze_c1_dataset.py
M  scripts/analyze_c1_teacher_repair.py
M  scripts/collect_c1_branch_ranking.py
M  scripts/run_c0_prim_decomposition.sh
M  scripts/run_c1_1_teacher_repair.sh
M  scripts/run_c1_ranking_pilot.sh
M  scripts/run_c1_ranking_wave1.sh
M  scripts/run_phaseA_extend.sh
M  scripts/run_phaseB_prim_feat.sh
M  scripts/run_phaseB_train_4gpu.sh
?? configs/experiments/real04_formal_seed12.json
?? docs/figures/
?? docs/implementation_and_experiments_prim.md
?? plans/STEINER_RL_BRANCHING_MIGRATION_MASTER_PLAN.md
```

其中主方案是本阶段直接输入，预计作为研究入口纳入阶段提交；其余均保持原样。状态中的 `build/**` 包含多个已修改编译产物，绝不进入提交。

## 计划新增或修改的文件

- `plans/STEINER_RL_BRANCHING_MIGRATION_MASTER_PLAN.md`：纳入阶段研究入口，不改写其内容。
- `docs/steiner/STATUS.md`
- `docs/steiner/RESEARCH_CONTRACT.md`
- `docs/steiner/adr/0001-formulation.md`
- `docs/steiner/adr/0002-representation.md`
- `docs/steiner/adr/0003-learning.md`
- `docs/steiner/adr/0004-evaluation.md`
- `configs/steiner/environment.lock.yml`
- `configs/steiner/experiments/protocols_v1.yml`
- `configs/steiner/splits/split_policy_v1.yml`
- `configs/steiner/splits/final_test_v1.yml`
- `tests/steiner/test_s00_contract.py`
- `docs/steiner/phases/S00/S00_{CHANGELOG,TEST_REPORT,RESULT_ANALYSIS,AUDIT_PACKET}.md`
- `docs/steiner/phases/S00/S00_COMMANDS.txt`

## 测试矩阵

| 检查 | 目的 | 通过条件 |
|---|---|---|
| YAML 解析 | 配置可机器读取 | 所有 S00 YAML 可由 `yaml.safe_load` 解析 |
| 契约必填项 | 防止协议、seed、baseline、指标遗漏 | 测试列出的键和值全部存在且唯一 |
| split 隔离 | 防止 final test 泄漏 | train/dev/test 来源与 seed 范围互斥；final entries 不允许开发用途 |
| final-test hash | 冻结封存清单 | canonical entries SHA-256 与配置声明一致 |
| final-test guard | 证明本阶段未运行学习模型 | 每个 final suite 为 `sealed`、`learning_runs: 0`、无结果路径 |
| 环境探测 | 区分事实、缺失与目标版本 | 每个依赖记录 probe command/status；不得把 unavailable 写成已安装 |
| 文档一致性 | Gate 与 ADR 不互相矛盾 | 关键 ID、profile、主指标和禁用比较项交叉一致 |
| Git 卫生 | 不混入用户改动和生成产物 | staged diff 仅含本计划列出的 S00 路径 |

## Gate S00

仅当以下条件全部满足才判定 PASS：

1. SPG、MCF、root 和仅 `x_e` action 定义无歧义。
2. P0--P4 的用途、资源上限、solver/training seeds 已预注册。
3. 数据按 instance/base-graph 分组；开发与 final test 来源边界明确。
4. final-test 清单存在 canonical SHA-256，状态为 sealed，学习模型运行计数为 0。
5. 主 baseline、主/诊断指标、配对统计和 checkpoint 选择规则已冻结。
6. 禁止比较和主张边界明确；不使用 final test 调参。
7. SCIP/SoPlex/PySCIPOpt/Ecole/PyTorch/编译器/CPU/GPU 的实际探测结果已记录，缺失依赖显式标为缺失。
8. 四份 ADR、研究契约、状态页、环境锁和 S00 过程文档完整。
9. S00 自动化测试通过，且提交候选不含 build、大数据、checkpoint、原始逐状态日志或无关用户改动。

任一项失败则 Gate FAIL，停止，不提交/push，也不开始 S01。

## 允许的外部副作用

- 本地创建并切换到 `research/steiner-s00-contract`。
- 只读探测本机工具、Python 包、硬件和 Git remote。
- Gate PASS 后，创建本地提交并 push 到同名远端 branch。
- 不创建 PR、不 merge、不 force-push、不下载 benchmark 数据、不执行 final test。
