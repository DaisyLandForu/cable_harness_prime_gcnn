# S05 formal-v3 confirmatory Gate — GPT pre-execution audit request

请只读审计 formal-v3 协议，不修改仓库。本次是新 confirmatory validation 的
运行前预审；不要把 formal-v2 失败审计的 PASS 误判成 S05 PASS，也不要授权 S06。

## 固定对象

- branch：`research/steiner-migration`
- v3 preregistration base：`55833c54d02f3e5bcbc4bd2e19d95f6f3f019293`
- v3 protocol content head：`8d7accd2a948935174d3113f8787e58dae7936b3`
- substantive range：
  `55833c54d02f3e5bcbc4bd2e19d95f6f3f019293..8d7accd2a948935174d3113f8787e58dae7936b3`
- v3 YAML：
  `configs/steiner/experiments/s05_teacher_il_formal_v3_confirmatory_gate.yml`
- v3 explanation：
  `docs/steiner/phases/S05/S05_FORMAL_V3_CONFIRMATORY_GATE.md`
- frozen candidate identities：
  `configs/steiner/experiments/s05_formal_v3_candidate_graphs.json`
- formal-v2 result audit record：
  `docs/steiner/audits/S05_FORMAL_V2_RESULT_AUDIT_RECORD.json`

## 固定 hashes

- v3 YAML：
  `99e75a4d4fa69f805232c637d4fc0ae750fcfccb57e9979b561f422591006242`
- v3 explanation：
  `c61b58d23f4de6bb2709aa9c40ccab11d6db4507a6bd1c9d2e544eb68595d873`
- candidate identity manifest：
  `e65fc9a03fd683277570befe13b11f4b15ea981ee427568d56aeeccdd3847b56`
- v2 result audit record：
  `a7cd6af30e9b1e98f46efb1b40f5876bb3f08cf75a45da59b09e319de33ec48a`

## 不可改变的历史

- formal-v1：`FAIL`，train bucket quota 527/640；
- formal-v2：五个 seed 全部训练完成且 reload error 0，但 Gate 只有 25/30
  lineages，完整 S05 Gate `FAIL`；
- v2 的 25-lineage CI/CV 和 random-geometric 负 family effect 只作 diagnostic；
- GPT 对 v2 结果的 PASS 只认可 `FAIL / STOP before S06` 的登记，并只允许
  preregister v3，未授权 v3 execution。

## v3 的唯一修复范围

V3 不重新训练。它 byte-identify 并冻结 v2 的五个 seed 101/202/303/404/505
checkpoint manifest 和 model payload，只建立一套全新的 confirmatory
validation Gate。

为了避免再次在 GPU 之后才发现 0-valid lineage，协议事前冻结 80 个候选图，五个
family 各 16。每张图固定三个 teacher seeds、每任务最多 16 states，共 240 tasks。
候选图的 family/scale/seed/graph SHA-256 已逐条提交。

lineage eligibility 只能使用 teacher validity：所有任务 terminal、零 task failure、
100% action mapping，并且三条 trajectory 合计至少 12 个 semantic-unique valid
states。每个 family 按冻结的 `(candidate_rank, graph_sha256)` 取最早六个 eligible
lineages。严禁读取 checkpoint、model logits、regret、teacher score magnitude 或
v2 model outcome 来决定图。任一 family 不足六个直接 FAIL，不准加入未注册候选或
retry-until-success。

最终状态按 lineage round-robin 选 64/family、320 total。在加载任何 checkpoint
前，必须 seal selected manifest 并硬检查：

```text
unique_gate_lineages == 30
family_lineage_counts == 6/6/6/6/6
every selected lineage >= 1 selected state
selected state counts == 64/64/64/64/64 == 320
zero prior-S05 / old-v2-Gate / cross-role / test-final lineage
all candidate/task/shard/selected-manifest checksums valid
```

任一失败都 `stop_without_loading_any_checkpoint`。只有该 barrier PASS 后，才允许
五个 frozen checkpoints 分别评估同一 sealed 320-state set。

## 请重点判断

1. v3 是否保持了最小合法边界：只修 fresh Gate denominator，不改模型或训练；
2. 80→30 的预注册 candidate-pool 设计，以及“至少 12 valid + 固定顺序”的选择，
   是否属于允许的 teacher-validity-only deterministic selection，而非按模型挑图；
3. scale envelope 是否足够诚实：结论仅限注册的 teacher-evaluable small/medium
   envelope，不声称 every family×scale；random-geometric small（已有 adverse
   diagnostic）仍在范围内；
4. candidate identity manifest 是否确实固定 80 个 unique validation-IID graphs，
   且与 prior S05 roles/pilots、test/final 的隔离规则明确；
5. pre-model-access barrier 是否能在任何 GPU/model evaluation 前拦截 v2 的
   25/30 类问题；
6. teacher 60%/40%/100%、320 states、5 seeds、exact reload 0、all-seed/
   all-family direction、30-lineage paired bootstrap、10,000 replicates、CV<=0.15
   和 test/final prohibition 是否没有降低；
7. 新 v3 YAML/Markdown/tests 是否只使用 30 作为 bootstrap denominator，已经不再
   继承 frozen v1 的 20/30 文档矛盾；
8. `execution_authorized: false` 是否保证本次 PASS 后仍须用独立 activation record，
   而不能直接修改已审 YAML。

## 本地静态证据

- 80/80 graphs 可确定性重新生成且 hashes 精确匹配；80 seeds/hashes 全唯一；
- 与 formal-v1 和全部 S05 pilot 注册 seeds/generated graph hashes 零碰撞；
- 五个本地 checkpoint manifest/model payload hashes 全匹配；
- targeted formal tests：10 passed；
- complete frozen-stack Steiner suite：100 passed、1 个既有 expected PACE skip；
- compileall、SCIP 8.0.4 wrapper identity、`git diff --check`：PASS；
- 没有 v3 teacher、selection、checkpoint load、GPU evaluation 或 test/final access。

请返回 `PASS`、`CONDITIONAL PASS` 或 `FAIL`，逐项列出 blocking findings。只有
`PASS` 才允许登记独立 activation，并进入 implementation/tests/teacher collection；
即使 PASS，也必须在 30-lineage pre-model barrier 通过前禁止 checkpoint/GPU，且
在完整 v3 scientific Gate PASS 前继续禁止 S05 tag、S06 和 test/final。
