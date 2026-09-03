# 旧航空 regression 独立排期

Backlog ID：`AVIATION-REGRESSION-001`

状态：`DEFERRED_SEPARATE_WORKSTREAM`

已知基线：59 passed、4 failed、0 skipped（S01 记录）

## 隔离决定

这 4 个失败不在 Steiner S03 中修复，也不允许与 S03 的 branchability、资源
采集、SCF 决策或审计提交混在一起。它们属于旧航空维护工作，不属于 Steiner
迁移阶段；单一长期分支规则只约束 Steiner 迁移，因此该维护工作应在 S03 审计
checkpoint 之后，从合适的仓库主线创建独立 worktree/维护分支处理，例如
`maintenance/aviation-regression-baseline`。本文件只排期，不创建该分支。

S03 的提交范围如果触及 `python/rl_branching/**`、`src/rl/**`、`tests/python/**`、
旧 `scripts/run_*`、`build/**` 或航空配置，除非只是审计中明确允许的只读引用，
应视为阶段边界失败并停止。

## 四个待办

| 类别 | 失败 | 独立处理方式 |
|---|---|---|
| Prim 语义 | `test_parse_z_and_grown_sets` | 对照旧契约确认 `lower_bounds` 缺省时应返回空还是推导集合；先复现再决定修源码或测试 |
| Prim 语义 | `test_prim_variable_features` | 对照六维 component ratio 的生产语义和历史结果；不得只改期望值求绿 |
| build/权限 | `test_dsu_sixdim_scip_fixture_matches_cpp_extractor` | 在干净构建中生成 `build/graph_probe`，验证构建规则产生可执行文件；不依赖当前脏 build |
| build/权限 | `test_scip_tree_help_exposes_remapped_seed_triple` | 在干净构建中生成 `build/scip_tree`，验证 mode 和 help 输出；不直接提交 build 二进制 |

## 建议排期与验收

1. S03 完成并形成不可变 phase checkpoint 后再领取该 backlog，避免两个问题域的
   commit range 重叠。
2. 在独立干净 worktree 复跑 `tests/python`，保存 59/4 基线和环境信息。
3. 先处理两个构建可执行问题，再分别审计两个 Prim 语义断言。
4. 每个语义修复先增加/明确复现测试；若需要改变旧行为，单独记录兼容性影响。
5. 验收目标是干净构建下 `tests/python` 全部通过；不得把失败改成 skip，也不得
   把当前 `build/**` 或环境 artifact 提交 Git。
6. 完成后单独评审和合并；不要把维护提交 cherry-pick 进某个 S03 Gate commit。
