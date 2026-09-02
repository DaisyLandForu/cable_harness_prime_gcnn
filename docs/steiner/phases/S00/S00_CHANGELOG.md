# S00 Changelog

## 完成内容

- 纳入用户提供的 Steiner RL branching 迁移主方案，作为后续阶段唯一主入口。
- 建立 `docs/steiner/STATUS.md` 和 S00 阶段登记。
- 新增 `RESEARCH_CONTRACT.md`，冻结 SPG/rooted MCF/root/action、学习路线、split、profile、seed、baseline、指标、统计和禁止比较项。
- 新增四份 ADR：formulation、representation、learning、evaluation。
- 新增机器可读 `protocols_v1.yml`、`split_policy_v1.yml`、`final_test_v1.yml` 和 `environment.lock.yml`。
- 新增 S00 契约自动化测试，检查配置必填项、split 隔离、Gate 阈值、环境版本和 final-test hash/封存状态。
- 建立 S00 计划、测试报告、结果分析、审计包和命令记录。

## 文件级变化

| 文件 | 变化 |
|---|---|
| `plans/STEINER_RL_BRANCHING_MIGRATION_MASTER_PLAN.md` | 用户提供的未跟踪主方案纳入阶段提交；S00 未改写内容 |
| `docs/steiner/STATUS.md` | 新增阶段状态、入口和已知风险 |
| `docs/steiner/RESEARCH_CONTRACT.md` | 新增研究总契约 |
| `docs/steiner/adr/0001-formulation.md` | 冻结 rooted MCF 与 SCF 触发条件 |
| `docs/steiner/adr/0002-representation.md` | 冻结 19/5/1 B0 和增强表示边界 |
| `docs/steiner/adr/0003-learning.md` | 冻结 IL-init RL 主线和停止条件 |
| `docs/steiner/adr/0004-evaluation.md` | 冻结配对统计、final test 和主张规则 |
| `configs/steiner/environment.lock.yml` | 记录实际与目标 solver/Python/hardware 状态 |
| `configs/steiner/experiments/protocols_v1.yml` | 冻结 P0--P4、limits、seeds、baselines、metrics 和 S03 Gate |
| `configs/steiner/splits/split_policy_v1.yml` | 冻结 base-graph grouping、synthetic seed ranges 和公开数据分配 |
| `configs/steiner/splits/final_test_v1.yml` | 封存 106 个 canonical final entries 及 selector SHA-256 |
| `tests/steiner/test_s00_contract.py` | 新增 8 个 S00 contract tests |
| `docs/steiner/phases/S00/*` | 新增阶段过程和审计文档 |

## 接口、schema 和配置变化

- 配置 schema 均为 version 1；S00 没有新增 Python package/API。
- protocol set ID：`steiner-protocols-v1`。
- formulation ID：`rooted_mcf_v1`。
- split policy ID：`steiner-split-policy-v1`。
- final manifest ID：`steiner-spg-final-test-v1`。
- environment stack ID：`scip804-ecole081-pyscipopt430`。
- final selector SHA-256：`8c0324c1a82485c2187825977fe2807e31512a6435e2f58f6a1d17babbfbddd1`。

## 数据迁移

无。未下载、复制、解析或求解 PACE/SteinLib/DIMACS 数据；未创建 checkpoint、replay、build 或逐状态日志。S02 只允许在保持 selector membership 不变的前提下补 archive/per-file content hashes。

## 与主方案的偏差

- 主方案 S00 明列 P0--P3；本阶段同时冻结了主方案第 6 节已定义的 P4，避免 S10/S12 事后选择 hard-subset 口径。没有运行 P4。
- 除主方案明列的 environment lock 外，增加机器可读 protocol/split/final manifests 和契约测试，以使 Gate 可自动核验。
- 冻结 SCIP 8.0.4/Ecole 0.8 路线，但当前 prefix 可执行权限与动态库搜索路径损坏是既有用户状态；S00 只记录为后续前置条件，没有修改环境产物。

## 明确未完成

- S01 package/scaffold 未开始。
- parser、generator、MCF/SCF、solver profile 实现、observation、模型、teacher、IL/RL 均未开始。
- final dataset content SHA、license artifact 和本地下载命令由 S02 完成。
- SCIP-Jack 具体版本在接入阶段固定；当前只冻结其协议角色和比较限制。
- GPU 在当前 shell 不可见；需要在任何 CUDA training profile 前重新探测。
