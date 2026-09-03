# Steiner RL Branching S00--S04 联合 GPT 审计说明

请把本文件全文作为审计提示词使用。此次任务是独立、只读验收，不是继续开发。

## 你的角色与审计目标

你是本项目的独立技术与科研审计者。请对 Steiner RL branching 迁移总方案以及
S00、S01、S02、S03、S04 五个已完成阶段做联合只读审计。

不要只复述阶段报告，也不要默认 Codex 给出的本地 PASS 正确。你需要实际检查
commit diff、源码、配置、测试、机器可读结果和各阶段 Gate，判断这些证据是否
足以允许进入 S05 strong-branch teacher/模仿学习阶段。

审计期间不要修改代码、不要提交、不要 push、不要创建 tag/PR、不要开始 S05，
也不要运行任何训练或 final test。若证据不足，明确判为证据不足或 FAIL，不要用
猜测补齐。

## 仓库与不可变审计身份

- GitHub 仓库：
  `https://github.com/DaisyLandForu/cable_harness_prime_gcnn`
- 长期分支：`research/steiner-migration`
- 本次审计固定 phase head：
  `123e4f3daaef1b8d15f2cc8f02a06f6edd6887fc`
- 固定 commit 页面：
  `https://github.com/DaisyLandForu/cable_harness_prime_gcnn/tree/123e4f3daaef1b8d15f2cc8f02a06f6edd6887fc`
- 整体迁移检查范围：
  `88ade1ac614fb12f882a10ba9b5d35b15c7b4d01..123e4f3daaef1b8d15f2cc8f02a06f6edd6887fc`
- S04 content head：
  `d7a78a33151822f3a8a57fdc0224ede333583646`
- S04 local-gate tag：`steiner-s04-local-gate-v1`，仅为本地 checkpoint，未推送；
  远端审计请以完整 SHA 为准。
- PR：未创建。PR 不是本次审计身份，也不需要为了审计创建 PR。

请先 checkout 或固定浏览 phase head SHA，不要以可能继续移动的 branch tip 代替
不可变 SHA。若无法访问仓库或某项文件，明确写 `NOT_VERIFIED`，不要假装检查过。

## 必读顺序

先完整阅读以下三个总入口：

1. `plans/STEINER_RL_BRANCHING_MIGRATION_MASTER_PLAN.md`
2. `docs/steiner/RESEARCH_CONTRACT.md`
3. `docs/steiner/STATUS.md`

然后从联合入口开始：

4. `docs/steiner/phases/S04/S04_AUDIT_PACKET.md`

再按 S00 → S04 顺序阅读每个阶段的过程材料。每阶段不能只读 audit packet；至少
同时检查 PLAN、CHANGELOG、TEST_REPORT、RESULT_ANALYSIS、COMMANDS 和对应源码/
配置/diff：

| 阶段 | 目的 | 审计入口 | 其他必读材料 |
|---|---|---|---|
| S00 | 冻结研究契约、问题、split、seed、Gate、final-test 边界 | `docs/steiner/phases/S00/S00_AUDIT_PACKET.md` | 同目录 `S00_PLAN.md`、`S00_CHANGELOG.md`、`S00_TEST_REPORT.md`、`S00_RESULT_ANALYSIS.md`、`S00_COMMANDS.txt` |
| S01 | 建立独立的 `steiner_branching` 研究栈骨架 | `docs/steiner/phases/S01/S01_AUDIT_PACKET.md` | 同目录 `S01_PLAN.md`、`S01_CHANGELOG.md`、`S01_TEST_REPORT.md`、`S01_RESULT_ANALYSIS.md`、`S01_COMMANDS.txt` |
| S02 | 数据解析、canonical ID、rooted MCF 和 correctness checker | `docs/steiner/phases/S02/S02_AUDIT_PACKET.md` | 同目录 `S02_PLAN.md`、`S02_CHANGELOG.md`、`S02_TEST_REPORT.md`、`S02_RESULT_ANALYSIS.md`、`S02_COMMANDS.txt` |
| S03 | 验证 branchability、strong-branch signal、映射和 CPU/RAM 范围 | `docs/steiner/phases/S03/S03_AUDIT_PACKET.md` | 同目录 `S03_PLAN.md`、`S03_CHANGELOG.md`、`S03_TEST_REPORT.md`、`S03_RESULT_ANALYSIS.md`、`S03_COMMANDS.md`、`S03_GATE_SUMMARY.json` |
| S04 | 建立 19/5/1 B0、动作映射、候选闭包和 deterministic forward | `docs/steiner/phases/S04/S04_AUDIT_PACKET.md` | 同目录 `S04_PLAN.md`、`S04_CHANGELOG.md`、`S04_TEST_REPORT.md`、`S04_RESULT_ANALYSIS.md`、`S04_COMMANDS.txt`、`S04_GATE_SUMMARY.json`、`S04_FORWARD_SNAPSHOT.json` |

## 各阶段不可变锚点

| 阶段 | base SHA | content head SHA | phase head SHA |
|---|---|---|---|
| S00 | `88ade1ac614fb12f882a10ba9b5d35b15c7b4d01` | `8b90375b6617a1ddcba34b872dbdbc11411cc042` | `a0bf0e3c1a702e1c85384f864defc86abbda29a5` |
| S01 | `a0bf0e3c1a702e1c85384f864defc86abbda29a5` | `05b42791226347d31647547c344ef46c9dc4e87d` | `35a90ec5e52e2fad8301e3441ff6b286c7701d04` |
| S02 | `35a90ec5e52e2fad8301e3441ff6b286c7701d04` | `19c7f46b91a1d05c46dbdeeba00bf863b37a7f5a` | `25be2e18c4020bed4cb8563618687b148d1f405f` |
| S03 | `91c30a48e6a06019d16d8b7529fe2d35bfa708fa` | `495d699cceefd243d4ab4c510be051f9df94833a` | `bb6079b7844dcc42fed4976c812795c842d6411b` |
| S04 | `931c7ae05c299c54bbdf59ecd458b64c7ca42282` | `d7a78a33151822f3a8a57fdc0224ede333583646` | `123e4f3daaef1b8d15f2cc8f02a06f6edd6887fc` |

S00--S02 报告里的旧阶段分支名是采用单一长期分支前的历史记录。当前 Git 治理以
`docs/steiner/adr/0005-single-branch-git-governance.md`、master plan 和
`configs/steiner/git_governance_v1.yml` 为准。请检查迁移历史的连续性，但不要要求
重新拆分或 merge 阶段分支。

## 必须检查的实现与配置

除阶段报告外，至少检查：

- `configs/steiner/environment.lock.yml`
- `configs/steiner/experiments/protocols_v1.yml`
- `configs/steiner/experiments/s03_branchability_pilot_v1.yml`
- `configs/steiner/splits/split_policy_v1.yml`
- `configs/steiner/splits/final_test_v1.yml`
- `configs/steiner/splits/final_test_content_v1.json`
- `configs/steiner/data_provenance_v1.yml`
- `configs/steiner/models/b0_milp_gcnn_v1.yml`
- `python/steiner_branching/**`
- `scripts/steiner/**`
- `tests/steiner/**`

不要泛读无关航空实验文档或把旧航空模型当作 Steiner B0。旧航空的 4 个既有失败
登记在 `docs/steiner/AVIATION_REGRESSION_BACKLOG.md`，需要判断其是否确实与本次
迁移隔离，而不是要求在本审计中顺便修复。

## 分阶段核验问题

### S00：研究与实验契约

1. SPG、rooted MCF、合法动作、SCF trigger、baseline 和指标是否定义无歧义？
2. instance-level split、synthetic seed 区间和 train-only normalization 是否能阻止
   state-level leakage？
3. PACE even、SteinLib/DIMACS final selectors 是否真正封存，是否存在通过 CLI 或
   import 绕过的路径？
4. 失败、timeout、OOM、invalid action 和所有 training seeds 是否被要求保留？
5. v1.3 对历史 inventory/资源文字的修订是否确实没有改变 Gate、split 或指标？

### S01：独立研究栈

1. `python/steiner_branching` 是否真正独立于航空 Prim/DSU、航空命名和旧全局状态？
2. config/schema/artifact/runtime API 是否 strict、versioned、deterministic、fail closed？
3. 是否存在无意修改旧航空默认行为或通过旧模块暗中复用错误语义？

### S02：数据与 MCF correctness

1. PACE/SteinLib parser 是否严格拒绝不支持的格式，而非静默降级？
2. canonicalization、parallel-edge ID、hash 和 variable naming 是否稳定且无碰撞？
3. rooted MCF 的目标、flow conservation、capacity linking 和 root 选择是否数学正确？
4. solution checker 是否独立于 MILP builder，toy/穷举/PACE optimum 证据是否充分？
5. 官方源 + checksum 政策是否避免未经许可提交 SteinLib/DIMACS raw bytes？

### S03：branchability 与资源

1. 90 个 formal tasks 和 10 个 ramp tasks 是否来自预注册矩阵，所有 timeout/skip/
   missing strong states 是否仍在分母中？
2. 70% branchable fraction、nontrivial median 127、strong valid 85%、all-tie 11.76%、
   mapping 100% 的定义和聚合代码是否与冻结 Gate 一致？
3. observer `DIDNOTRUN`、priority candidate slice、transformed-name normalization 和
   native strong-branch probe 是否保持 SCIP baseline/action 语义？
4. 2,415,538/2,415,538 candidate mappings 是否能支持“当前 SPG edge action 映射
   完整”，并且没有把连续 flow/auxiliary variables 当作动作？
5. p95 RSS、6-worker projection、build time 和 flow counts 是否足以支持当前 MCF
   范围？跨 Gold 6148/Silver 4214 结果是否仅用于固定 Gate而没有做 wall-time 排名？
6. 46/90 timelimit、零分支 buckets 和两个 geometric root-LP timeout 是否限制了
   后续 teacher 数据范围和科研主张？

### S04：B0、动作映射与候选闭包

1. `milp_bipartite_v1` 是否严格为 variable/constraint/edge 19/5/1，且不含航空
   one-hot、Prim/DSU、扩展 row/edge 或 global state？
2. SCIP action set 是否只接受当前 fractional binary `stp_x_*`，transformed names
   和 parallel edges 是否能唯一、100% 回映射到 metadata edge ID？
3. Ecole 无 incumbent 时 features 13/14 的 NaN→zero sentinel 是否语义明确、仅限
   这两列、对其他 NaN/Inf fail closed；进入 S05 前是否需要 missingness bit/new schema？
4. PySCIPOpt transformed variable 顺序和 Ecole NodeBipartite columns 的对齐假设是否
   可靠，现有测试能否发现错位？
5. exact closure 是否精确包含一轮 variable→constraint→variable 的完整 receptive
   field，并保留稳定 local/global map 与 edge order？
6. full/closure max error 0、argmax 3/3、31/31 mapping、68,161 parameters 和 CPU
   timing 是否可从代码/JSON/hash 复核？
7. 1 个 train graph/3 states 是否只被用于工程 parity，而没有被夸大成 learned
   branching quality 或生产性能证据？

## 跨阶段必须检查

1. 对每个阶段分别检查其 `base..content` substantive diff，以及 content..phase
   metadata diff；再检查整体迁移范围，确认没有遗漏中间治理/runtime commits。
2. 检查是否混入 `build/**`、raw corpora、checkpoints、逐状态日志、用户航空脚本、
   secret/PAT 或其他阶段外文件。
3. 检查 Gate 是否被事后降低，是否删过失败样本、timeout、skip 或失败 seed。
4. 检查 S00--S04 是否读取/求解 sealed final test，或使用 validation/final test 调参。
5. 区分三种结论：correctness、branchability、learned-policy quality。S00--S04 尚无
   teacher/IL/RL 训练，不能声称 learned policy 有效。
6. 检查 SCIP 8.0.4 wrapper 是否真正阻止系统 SCIP 9.2.2 混入。
7. 检查机器可读 config、报告数字、hash、命令和 STATUS 是否相互一致。
8. 判断已知问题是否阻塞 S05：Ecole sentinel/order alignment、S03 teacher 范围、
   raw 数据许可、旧航空失败、首次 CUDA 前资源验收。

## 建议复现命令

若审计环境具备仓库和冻结依赖，执行：

```text
git checkout --detach 123e4f3daaef1b8d15f2cc8f02a06f6edd6887fc
git diff --check 88ade1ac614fb12f882a10ba9b5d35b15c7b4d01..123e4f3daaef1b8d15f2cc8f02a06f6edd6887fc
scripts/steiner/run_with_scip804.sh --verify-only
scripts/steiner/run_with_scip804.sh --python -m pytest -q tests/steiner

s04_verify_dir=$(mktemp -d /tmp/steiner-s04-audit.XXXXXX)
scripts/steiner/run_with_scip804.sh --python \
  scripts/steiner/run_s04_b0_snapshot.py \
  --snapshot-output "$s04_verify_dir/S04_FORWARD_SNAPSHOT.json" \
  --summary-output "$s04_verify_dir/S04_GATE_SUMMARY.json"
cmp docs/steiner/phases/S04/S04_FORWARD_SNAPSHOT.json \
  "$s04_verify_dir/S04_FORWARD_SNAPSHOT.json"
```

预期完整 Steiner suite 为 75 passed、1 expected skipped；skip 是未提供
`STEINER_PACE_DEV_ROOT` 的 PACE odd development integration。不要为了消除 skip
临时访问 sealed data。若你的环境无法运行冻结的 SCIP/Ecole stack，请把 runtime
复现标为 `NOT_VERIFIED`，仍继续完成静态源码、配置、diff 和证据审计。

S03 raw per-task shards 按政策位于 ignored
`results/steiner/raw/s03/s03-branchability-pilot-v1/`，不在 Git。如果你只能访问
远端仓库，应检查 committed aggregator、tests、`S03_GATE_SUMMARY.json` 和 raw index
说明，并明确指出哪些 aggregate 无法从远端 raw shards 独立重算；不要假称看过 raw。

## 强制输出格式

请返回一份完整 Markdown 审计报告，建议保存为：
`docs/steiner/audits/S00_S04_GPT_AUDIT.md`。

报告必须按以下结构：

1. **审计结论**：只给出 `PASS`、`CONDITIONAL PASS` 或 `FAIL` 之一，并用一段话说明
   是否允许进入 S05。
2. **审计身份与可复现范围**：列出实际检查的 commit SHA/range、可访问与不可访问
   的证据、实际执行的命令。
3. **逐阶段 Gate 核验**：S00、S01、S02、S03、S04 分别给出
   `证据充分 / 证据不足 / 未通过`，每项附具体 `文件:行号` 或 commit/diff 证据。
4. **跨阶段一致性**：split/final-test、SCIP stack、Git 边界、失败保留、hash、主张
   边界是否一致。
5. **Blocking issues**：按严重程度列出；每项说明影响、证据、最小修复和必须补的
   测试。没有则明确写“无”。
6. **Non-blocking issues**：同样给出证据和建议关闭阶段。
7. **复现实验检查**：区分静态验证、实际运行通过、因环境未运行三种状态。
8. **当前可以成立的结论**与**当前不能成立的结论**。
9. **复审清单**：若不是 PASS，给出最小 remediation checklist；若 PASS，列出进入
   S05 前仍必须执行的操作，例如 CUDA preflight、teacher pilot 和审计记录提交。

判定规则：任何 formulation correctness、action mapping、split/final-test leakage、
Gate 口径、失败样本删除、无法复现的关键 hash 或 secret 泄漏问题都应视为 blocking。
仅文案、格式或不影响 S05 科学有效性的改进才可作为 non-blocking。不要因为本地报告
写着 PASS 就自动给 PASS，也不要把“没有 raw 数据再分发许可”误判为算法错误；应
判断当前只发布 URL/checksum、不发布 raw 的政策是否合规且足以复现。
