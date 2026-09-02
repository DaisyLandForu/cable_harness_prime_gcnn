# S01 独立研究栈骨架计划

## 阶段目标

建立可独立导入、无航空命名依赖的 `steiner_branching` Python 骨架；提供严格配置、核心 typed dataclass、统一 logging/seed/artifact 规则，并以测试证明未知字段拒绝、schema invariant 和路径安全。S01 PASS 后才允许开始 S02。

## 非目标

- 不实现 `.gr/.stp` parser、canonicalization、generator、split、MCF 或 solution checker；这些属于 S02。
- 不实现 Ecole observation、action mapping、GCNN、teacher、IL 或 RL。
- 不修改 `python/rl_branching`、C++、旧配置或航空默认行为。
- 不下载或运行任何 benchmark/final test。

## 输入、基线与偏差

- 起始 branch：`research/steiner-s00-contract`。
- base SHA：`a0bf0e3c1a702e1c85384f864defc86abbda29a5`。
- 工作 branch：`research/steiner-s01-scaffold`。
- S00 本地 Gate：PASS；远端 `origin/research/steiner-s00-contract` 已存在并指向 base SHA。
- S00 GPT audit：NOT_RUN。用户明确要求继续 S01--S02，因此本阶段在不降低本地 Gate 的前提下继续，并把缺失外部审计保留为治理风险。
- 初始工作区仍有 S00 已记录的用户 `artifacts/**`、`build/**`、`results/audit/**`、旧 `scripts/**` 改动及三个无关未跟踪入口；全部保留且不暂存。

## 计划文件

- `python/steiner_branching/__init__.py`
- `python/steiner_branching/{config,contracts,runtime}.py`
- `python/steiner_branching/{data,milp,solver,models,learning,evaluation}/__init__.py`
- `configs/steiner/scaffold_smoke.yml`
- `scripts/steiner/README.md`
- `tests/steiner/{conftest,test_scaffold}.py`
- `docs/steiner/STATUS.md`
- `docs/steiner/phases/S01/S01_{CHANGELOG,TEST_REPORT,RESULT_ANALYSIS,AUDIT_PACKET}.md`
- `docs/steiner/phases/S01/S01_COMMANDS.txt`

## 设计约束

- package import 不得导入 Ecole、PySCIPOpt 或旧 `rl_branching`。
- 配置未知字段、缺失字段、错误 schema version 必须显式失败。
- `SteinerGraph`、`ProblemMetadata`、`GraphSchema`、`RunManifest` 使用 immutable dataclass，并在构造时验证基础 invariant。
- run ID/stage/artifact kind 只接受安全字符，所有 artifact path 必须位于配置 root 下。
- seed helper 覆盖 Python/NumPy；Torch 已安装时同步 seed，但不要求 CUDA。
- logging 使用稳定、单一格式，重复配置不得叠加 handler。

## 测试矩阵

| 测试 | 通过条件 |
|---|---|
| import | `steiner_branching` 及六个子包可导入，不触发 solver 依赖 |
| strict config | 最小 YAML 可加载；unknown/missing/schema mismatch 均失败 |
| dataclass invariants | 合法对象 round-trip；重复 ID、非法 hash/root/schema 均失败 |
| seed | 同 seed 的 Python/NumPy 序列一致；不需要 GPU |
| artifact path | 合法布局稳定；`..`、绝对 run ID、非法 kind 被拒绝 |
| logging | 格式稳定，重复配置只有一个 managed handler |
| regression boundary | commit diff 不含 `python/rl_branching/**` 或 C++ 文件 |
| S00 regression | S00 contract tests 继续通过 |

## Gate S01

以下全部满足才 PASS：新包/子包可导入；最小配置和 schema 测试通过；未知字段拒绝；seed/artifact/logging 规则可复现；S00 tests 通过；stage diff 无旧航空源码、build、data、checkpoint 或用户改动。FAIL 时停止，不开始 S02。

## 外部副作用

- 允许创建本地 branch/commit 和测试临时目录。
- 不下载数据、不调用 solver、不运行 final test。
- 本请求没有新增 merge/force-push 授权；远端写入状态在阶段报告中明确说明。
