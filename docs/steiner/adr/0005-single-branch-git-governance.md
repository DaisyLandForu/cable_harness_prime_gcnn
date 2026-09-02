# ADR 0005：迁移使用单一长期分支和不可变审计 checkpoint

- 状态：Accepted
- 日期：2026-09-02
- 决策阶段：S02 后、S03 前的治理迁移

## 背景

S00--S02 按旧主方案分别推送阶段分支，但它们并非三条独立开发线：S01 从
S00 phase head 创建，S02 又从 S01 phase head 创建，提交历史完全线性。阶段
分支有利于展示增量，却增加了“哪个分支才是完整迁移代码”的理解成本。

分支名本身还是可移动引用，不能提供比完整 commit SHA 更强的审计保证。阶段
审计真正需要冻结的是 base、实质 content head、metadata phase head、测试证据
和精确 diff range。

## 决策

唯一活动迁移分支改为 `research/steiner-migration`，初始点为累计 S02 phase
head `25be2e18c4020bed4cb8563618687b148d1f405f`。旧 S00/S01/S02 分支保留为
只读历史指针，不删除、不移动、不继续开发。

未来阶段直接在长期分支上线性追加提交，只有本地 Gate PASS 才允许
fast-forward push。审计以完整 SHA/range 为准：

- local Gate checkpoint：`steiner-sXX-local-gate-vN`；
- GPT PASS checkpoint：`steiner-sXX-audited-vN`；
- tag 必须 annotated、只增版本、不得移动；
- GPT FAIL 的修复以追加 commit 保留历史，不 rebase/amend/force-push；
- 阶段之间不 merge，S13 前不把长期分支合并到仓库目标主线。

当前 S00--S02 的 GPT audit 均为 NOT_RUN，因此现阶段只能创建 local-gate tags，
不能创建 audited tags。

## 后果

- 最新长期分支始终包含全部迁移代码，用户只需跟踪一个活动分支。
- 阶段 review 仍可通过 `base..content_head` 精确查看，不依赖阶段分支。
- 单一分支要求严格禁止改写已发布历史，否则多个阶段的审计 range 会同时失效。
- 若某阶段 Gate FAIL，下一阶段不得开始；remediation 留在同一线性历史中。
- 旧阶段文档中的 branch 字段保留为当时事实，不回写伪造历史；当前治理状态以
  master plan v1.2、`STATUS.md` 和机器可读 governance config 为准。

## 未采用方案

- 删除旧阶段分支：没有必要，且会损失已有远端审计导航入口。
- 把旧分支重新 merge：它们已是祖先关系，会制造无意义 merge commit。
- 继续按阶段创建活动分支：增加操作和理解成本，审计完整性仍主要依赖 SHA。
