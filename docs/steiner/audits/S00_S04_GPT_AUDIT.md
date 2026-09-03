# Steiner RL Branching S00--S04 联合 GPT 审计报告

**审计对象**：`DaisyLandForu/cable_harness_prime_gcnn` / Steiner RL branching migration  
**固定 phase head**：`123e4f3daaef1b8d15f2cc8f02a06f6edd6887fc`  
**整体迁移范围**：`88ade1ac614fb12f882a10ba9b5d35b15c7b4d01..123e4f3daaef1b8d15f2cc8f02a06f6edd6887fc`  
**审计性质**：独立、只读、S05 前置验收  
**审计日期**：2026-09-03

---

## 1. 审计结论

# CONDITIONAL PASS

**当前不允许进入 S05 strong-branch teacher / imitation learning。** S00--S03 的研究契约、独立研究栈、rooted MCF correctness 与 branchability/resource Gate 在固定提交上的静态证据总体充分；S04 的 19/5/1 B0、NaN sentinel、候选闭包和 deterministic forward 也基本成立。但 S04 存在一个直接触及 **action identity** 的 blocking evidence gap：`SteinerNodeBipartite.extract()` 将 `model.as_pyscipopt().getVars(transformed=True)` 的**列表位置**直接当作 Ecole `NodeBipartite.variable_features` 的行顺序来绑定变量名，而 Ecole 对 variable row 的公开语义是按 SCIP `probindex` 索引，PySCIPOpt `getVars()` 的公开接口并未提供“返回顺序必与 probindex 一致”的契约。现有 31/31 mapping 和 3-state integration test 可以证明当前样本下映射成功，却无法排除“变量名列表在若干 `stp_x_*` 之间发生一致排列错位、仍通过 binary/fractionality/name 格式校验”的情况。进入 teacher 阶段后，这会使 strong-branch label 绑定到错误 edge ID，因此按本次审计规则属于 blocking action-mapping 问题。

放行条件是：将 Ecole row index ↔ SCIP transformed variable `probindex` ↔ variable name 的对应关系做成**显式、可验证、fail-closed 的契约**，补充能主动发现排列错位的 frozen-stack integration test，并重新生成/核对 S04 snapshot、Gate summary 和完整 Steiner suite。完成这些最小修复后，不需要推翻 S00--S03，也不需要重做 S04 模型设计。

---

## 2. 审计身份与可复现范围

### 2.1 固定 Git 身份

本审计只按以下不可变对象判断，不使用可能继续移动的 `research/steiner-migration` branch tip：

| 范围 | SHA |
|---|---|
| migration base | `88ade1ac614fb12f882a10ba9b5d35b15c7b4d01` |
| fixed phase head | `123e4f3daaef1b8d15f2cc8f02a06f6edd6887fc` |
| overall range | `88ade1ac614fb12f882a10ba9b5d35b15c7b4d01..123e4f3daaef1b8d15f2cc8f02a06f6edd6887fc` |
| S00 content / phase | `8b90375b6617a1ddcba34b872dbdbc11411cc042` / `a0bf0e3c1a702e1c85384f864defc86abbda29a5` |
| S01 content / phase | `05b42791226347d31647547c344ef46c9dc4e87d` / `35a90ec5e52e2fad8301e3441ff6b286c7701d04` |
| S02 content / phase | `19c7f46b91a1d05c46dbdeeba00bf863b37a7f5a` / `25be2e18c4020bed4cb8563618687b148d1f405f` |
| S03 content / phase | `495d699cceefd243d4ab4c510be051f9df94833a` / `bb6079b7844dcc42fed4976c812795c842d6411b` |
| S04 content / phase | `d7a78a33151822f3a8a57fdc0224ede333583646` / `123e4f3daaef1b8d15f2cc8f02a06f6edd6887fc` |

S02 phase head 到 S03 base、S03 phase head 到 S04 base 之间存在 ADR / Git governance / frozen runtime hardening 等中间提交。它们在整体 migration range 内，未按“阶段分支”被遗漏；最终 `run_with_scip804.sh`、Git governance config 和 ADR 均纳入本审计。

### 2.2 已实际检查的证据

静态检查覆盖：

- `plans/STEINER_RL_BRANCHING_MIGRATION_MASTER_PLAN.md`
- `docs/steiner/RESEARCH_CONTRACT.md`
- `docs/steiner/STATUS.md`
- S00--S04 的 `AUDIT_PACKET / PLAN / CHANGELOG / TEST_REPORT / RESULT_ANALYSIS / COMMANDS`
- S03/S04 machine-readable `*_GATE_SUMMARY.json`
- S04 `S04_FORWARD_SNAPSHOT.json`
- `configs/steiner/**` 中本审计指定的 environment/protocol/split/final/provenance/B0/pilot 配置
- `python/steiner_branching/**` 中 parser、contracts、runtime、canonicalization、rooted MCF、checker、S03 aggregation、S04 observation/closure/model 相关实现
- `scripts/steiner/**` 中 final-data lock/download、S03/S04 runner、SCIP 8.0.4 wrapper
- `tests/steiner/**` 中 S00--S04 对应测试
- 各阶段 `base..content` 的提交 patch/file scope，以及 `content..phase` 的 audit/report metadata patch
- 固定 migration range 的提交连续性和阶段间治理/runtime commits

### 2.3 不可访问或未独立重算的证据

- S03 ignored raw per-task shards：`results/steiner/raw/s03/s03-branchability-pilot-v1/`，远端仓库不包含，故 shard-level aggregate **NOT_VERIFIED**。
- SteinLib/DIMACS raw bytes：按政策不提交；本审计没有访问 sealed final bytes，符合 final-test 边界。
- 本地冻结 SCIP/Ecole runtime：当前审计执行容器无法解析 `github.com`，不能建立本地 checkout，因此完整 runtime rerun **NOT_VERIFIED**。
- S04 committed snapshot / summary：完成了源码、配置、机器可读结果之间的静态一致性检查，但没有在本环境重新生成字节并 `cmp`，故 byte-for-byte regeneration **NOT_VERIFIED**。
- 全仓库本地 secret scanner / `git grep`：因无法 checkout 未执行。固定提交 patch/file scope 中未发现 PAT/secret 证据，但“完整本地 secret scan”本身为 **NOT_VERIFIED**。

以上 NOT_VERIFIED 是审计环境限制，不等于项目证据天然不可复现；其中 S03 raw shards 的不可远端重算是项目明确的数据保留策略。

### 2.4 实际执行的命令

实际尝试：

```text
git clone https://github.com/DaisyLandForu/cable_harness_prime_gcnn.git /tmp/steiner-audit
```

结果：失败，执行环境 DNS 无法解析 `github.com`。因此以下建议复现命令**没有在本审计环境伪装成已运行**：

```text
git checkout --detach 123e4f3daaef1b8d15f2cc8f02a06f6edd6887fc
git diff --check 88ade1ac614fb12f882a10ba9b5d35b15c7b4d01..123e4f3daaef1b8d15f2cc8f02a06f6edd6887fc
scripts/steiner/run_with_scip804.sh --verify-only
scripts/steiner/run_with_scip804.sh --python -m pytest -q tests/steiner
scripts/steiner/run_with_scip804.sh --python scripts/steiner/run_s04_b0_snapshot.py ...
cmp docs/steiner/phases/S04/S04_FORWARD_SNAPSHOT.json <regenerated snapshot>
```

---

## 3. 逐阶段 Gate 核验

## S00 — 证据充分

S00 的研究与实验契约足以作为后续阶段冻结边界。

1. **SPG / rooted MCF / action / SCF / baseline / metrics 定义无实质歧义。** `docs/steiner/RESEARCH_CONTRACT.md` 与 `configs/steiner/experiments/protocols_v1.yml` 将动作限制为当前 SCIP LP 中 fractional binary `stp_x_*` edge variables，SCF trigger、baseline、失败计分和训练/验证/final 边界均单独冻结。不存在把 flow variable 或 auxiliary variable 纳入 action contract 的文字入口。

2. **instance-level leakage 防护成立。** `configs/steiner/splits/split_policy_v1.yml` 明确以 `base_graph_lineage` 做互斥 split，不允许按 B&B state 拆分；synthetic train/validation/final seed 区间互斥；normalization 只允许从 train instances 拟合。该设计阻止同一基础图的不同状态跨 train/validation/final 泄漏。

3. **final selectors 有双层封存。** `configs/steiner/splits/final_test_v1.yml` 冻结 final selector/首个可读取阶段，`configs/steiner/splits/final_test_content_v1.json` 冻结内容锁；`scripts/steiner/lock_final_content.py` 以字节/hash 为边界，不依赖 parser/solver；final-data download/lock 路径有阶段检查。静态检查未发现通过普通实验 CLI 绕过 first-allowed-stage 的显式入口。

4. **失败不允许从分母中删除。** `protocols_v1.yml` 要求 failed/timeout/OOM/skipped/invalid action 与全部 training seeds 留痕；checkpoint 只能由 validation 选择，final test 不参与调参。

5. **v1.3 修订未改变科研 Gate。** S04 范围内对 master/research inventory 的修订是历史描述与资源事实校正，阶段材料明确保持 split、seed、Gate、metrics 不变；对应 patch 未显示 lowering Gate 的内容。

**审计判断**：S00 可继续沿用，无需因历史阶段分支名变化重做。

---

## S01 — 证据充分

S01 已建立与旧航空 Prim/DSU 语义隔离的 Steiner research stack。

1. `python/steiner_branching/` 是独立 package；S01 substantive diff 主要新增 `contracts/config/runtime/artifacts` 等 Steiner 组件，没有通过 import 旧航空 Prim/DSU 状态来实现 Steiner API。
2. `python/steiner_branching/config.py` 对 schema/stage/unknown fields 采用 strict validation；`contracts.py` 使用冻结 dataclass、严格 SHA/canonical identifiers、版本化 schema/run manifest；`runtime.py` 对 artifact path 与 deterministic seed 做约束。
3. S01 content diff 未修改旧航空求解主链的默认行为；旧航空回归失败保留在 `docs/steiner/AVIATION_REGRESSION_BACKLOG.md`，没有被伪装成 Steiner regression。
4. S01 报告记录的 legacy failures 与 Steiner 新测试被分开解释，未把“旧测试本来失败”改写为整体绿灯。

**审计判断**：独立性、strict/versioned/fail-closed 基础足以支持 S02--S05，不要求先修旧航空 4 个已登记失败。

---

## S02 — 证据充分

S02 的 parser、canonicalization、rooted MCF 和 independent checker 在静态代码层面成立。

1. **Parser fail closed。** `python/steiner_branching/data/steinlib.py` 对 section/record type/计数/EOF 做严格检查；unsupported directed/rooted 扩展不是静默忽略。PACE parser 复用该严格路径而不是另写宽松降级器。
2. **Canonical IDs 稳定。** canonicalization 对节点和无向 edge tuple 做确定性排序后编号，parallel edges 保留独立 edge ID；变量名使用固定宽度 edge/commodity/arc ID，transformed SCIP prefix normalization 有独立 helper。完全相同端点+成本的重复平行边在数学上不可区分，但不会发生 ID collision。
3. **rooted MCF 数学形式正确。** `python/steiner_branching/milp/mcf.py`：每条原图 edge 建 binary `x_e` 并以 edge cost 计目标；每个非 root terminal 建 commodity；每条无向 edge 建双向 flow arcs；root 流守恒为 +1、target 为 -1、其他为 0；两方向 flow 通过 `f_uv^k + f_vu^k <= x_e` 与 edge selection 联动；目标最小化。root 由 canonical terminals 确定性选择。
4. **Checker 与 builder 的核心判断独立。** `python/steiner_branching/milp/checker.py` 的 selected-edge feasibility/objective 根据 canonical graph 独立计算，不调用 MCF constraints 来“自证正确”；toy/brute-force/PACE known optimum 证据形成不同检查路径。
5. **数据许可策略合理。** `configs/steiner/data_provenance_v1.yml` 要求官方 URL/checksum、本地 raw cache ignored；未获许可时不 redistribue SteinLib/DIMACS raw bytes，只发布 selector/hash/script。这是复现与许可的合理折中，不构成算法 blocker。

**审计判断**：没有发现 formulation correctness blocker。

---

## S03 — 证据充分（raw shard-level 独立重算 NOT_VERIFIED）

S03 的冻结 Gate 定义、聚合代码和 committed summary 互相一致；raw shards 不在 Git，故本审计不能逐 task 从原始 shard 独立重新求和。

1. **任务矩阵与分母被冻结。** `configs/steiner/experiments/s03_branchability_pilot_v1.yml` 的 formal matrix 为 5 families × 3 buckets × 2 instances × 3 baselines = **90 formal tasks**；ramp 为 1/3/6 workers 对应总计 **10 ramp tasks**。`expand_tasks()` 验证 split 必须是 train，避免借用 validation/final。
2. **branchable fraction 没有按“成功任务”缩分母。** S03 aggregator 以预注册 30 个 instance 单元为 denominator 计算 primary branchability；formal task missing 会触发 Gate，而不是从 denominator 删除。committed summary 为 branchable fraction **0.70**、nontrivial median **127**。
3. **Strong signal 分母保留 missing states。** preregistered strong subset × `max_states=2` 得到 expected 20 states；summary 记录 17 valid、3 missing，valid fraction **17/20 = 85%**；all-tie **2/17 = 11.76%**，没有用 17 代替 expected 20 计算 valid rate。
4. **Observer 保持 SCIP baseline/action 语义。** `CandidateObserver.observe()` 读取 `getLPBranchCands()` 的 priority candidate slice；observer callback 返回 `DIDNOTRUN`，用于观察而非执行自定义 branch；transformed name normalization 后只认可 registered `x` edge names。
5. **2,415,538 / 2,415,538 mapping 只证明 S03 当前 SPG edge candidates 的名称映射完整。** S03 candidate source 是 SCIP branching candidate variables，不依赖 S04 Ecole variable-row/name positional join，因此 S04 的 order-alignment blocker 不反向否定这个 S03 数字。
6. **资源 Gate 与范围主张相符。** committed summary 记录 p95 RSS、6-worker projection、build time、flow counts 等；其作用是判断当前 MCF pilot 范围是否可承受。Gold/Silver 6148/4214 等跨实例结果没有被当作正式 wall-time ranking。
7. **46/90 timelimit、零分支 bucket 和 geometric root-LP timeout 被保留。** 这些结果限制后续 teacher 数据范围与论文主张，但本身不是删除失败样本。S05 应只在预注册、可产生稳定 candidate/strong signal 的 teacher pilot 范围开始，不能把 S03 解释为“所有 Steiner 输入都适合强分支模仿”。

**NOT_VERIFIED 边界**：远端缺失 `results/steiner/raw/s03/...`，所以 **90 个 formal task 的逐 shard 数字、2,415,538 candidate mapping 的原始逐状态累加以及 RSS 原始采样无法由本审计独立重算**。静态 aggregator + committed machine summary 的逻辑未发现 denominator manipulation，但这不同于“看过 raw”。

---

## S04 — 证据不足

S04 的 B0/closure/parity 设计基本正确，但动作身份存在一个进入 S05 前必须关闭的证据缺口。

### 已核验成立

1. **`milp_bipartite_v1` 是严格 19/5/1。** `python/steiner_branching/solver/bipartite_observation.py` 定义 19 variable features、5 constraint features、1 edge feature；`configs/steiner/models/b0_milp_gcnn_v1.yml` 固定 `19/5/1`、embedding/hidden=64、1 round、sum aggregation。测试显式禁止 aviation/Prim/DSU schema 名称。
2. **NaN sentinel 设计是窄范围 fail closed。** `bipartite_observation.py` 将 `UNDEFINED_INCUMBENT_FEATURE_INDICES = (13, 14)`；`copy_node_bipartite()` 仅允许这两列的 NaN，并将其置 0；其他 NaN/Inf 直接抛错。该语义与 Ecole “无 incumbent 时 incumbent statistics 未定义”一致。S05 前不强制增加 missingness bit；若新增 bit，应升级 schema，而不能静默把 19 改成 21。
3. **合法 action filter 的目标语义正确。** `with_legal_edge_actions()` 要求 Ecole `action_set` index 有界、唯一，对应行必须 binary 且 fractional，随后变量名必须精确解析为 metadata 中的 `stp_x_*` edge ID；continuous flow/auxiliary variables 不应成为 action。
4. **exact closure 与一轮 GCNN receptive field 匹配。** `python/steiner_branching/solver/graph_state.py` 从 candidate variables 取所有 incident constraints，再取这些 constraints 上全部 variables，正好覆盖一轮 variable→constraint→variable；stable `np.flatnonzero` 与原 edge mask 保持 global/local index 和 edge order。对当前 1-round model，candidate logits 不依赖闭包外节点。
5. **68,161 参数可从源码独立推导。** 64-dim、两层 MLP 且带 bias：variable encoder 5,440 + constraint encoder 4,544 + edge encoder 4,288 + 四个 message/update MLP 49,664 + output 4,225 = **68,161**，与 `S04_GATE_SUMMARY.json` 一致。
6. **工程 parity 机器记录内部一致。** committed summary/snapshot 声称 full/closure max error 0、argmax 3/3、31/31 mapping、deterministic snapshot；测试也要求两次 snapshot 字节一致、mapping_rate=1、parity error≤1e-5、argmax=1。由于本环境未运行 frozen stack，这些 runtime 数值属于“committed evidence verified for consistency”，不是本审计重新测得。
7. **证据范围没有被夸大。** S04 报告将 1 train graph / 3 states 明确定位为 engineering parity；没有把它描述成 learned branching quality、speedup 或 production performance。

### 阻塞缺口：Ecole row ↔ PySCIPOpt variable-name 顺序未被证明

关键实现：

- `python/steiner_branching/solver/bipartite_observation.py:278-284`（文件实际约 298 行）：`raw = super().extract(...)` 后调用 `model.as_pyscipopt().getVars(transformed=True)`，再按返回列表顺序生成 `names`，最后 `copy_node_bipartite(raw, names)`。
- `python/steiner_branching/solver/bipartite_observation.py:226-250`：`with_legal_edge_actions()` 用 Ecole `action_set` 的整数 index 同时索引 `variable_features[index]` 与 `variable_names[index]`。

Ecole `NodeBipartite` 的公开语义明确指出 variable rows 按 SCIP variable `probindex` 索引，并可直接用 branching `action_set` 索引；而 PySCIPOpt `Model.getVars(transformed=True)` 的公开文档只承诺“取得 transformed problem variables”，没有审计可依赖的“列表顺序等于 SCIP probindex 顺序”契约。

因此当前链路实际依赖：

```text
Ecole variable row index
= SCIP transformed variable probindex
= PySCIPOpt getVars(transformed=True) list position
```

前一个等式是 Ecole 的接口语义；**后一个等式在当前代码和测试中没有被显式证明或强制**。

现有 integration test `tests/steiner/test_s04_bipartite.py` 只检查 real snapshot 能产生 `mapping_rate == 1.0`、3 states、parity/argmax 成功。若 `getVars()` 列表中的多个 `stp_x_*` 变量彼此排列错位，`features[index]` 仍可能是 binary/fractional，`names[index]` 仍可能解析成另一个合法 edge ID，31/31 mapping 仍可全部通过，却把 strong-branch label 绑定到错误 edge。

这不是“理论上可能有 bug”式泛泛担忧，而是 teacher/IL 阶段 action label identity 的必要不变量未被测试覆盖。按本审计判定规则，属于 blocking action-mapping evidence gap。

---

## 4. 跨阶段一致性

### 4.1 split / final-test

**一致。** S00 冻结 `base_graph_lineage` instance split、互斥 synthetic seed ranges、train-only normalization；S03 pilot config 只允许 train；S04 snapshot 使用 train graph。未发现 S00--S04 读取 sealed PACE-even / SteinLib-DIMACS final content 用于训练、Gate 调参或模型选择的证据。final test 首次允许阶段仍晚于当前阶段。

### 4.2 SCIP stack

**静态上成立，runtime NOT_VERIFIED。** `scripts/steiner/run_with_scip804.sh:531-590` 在 formal mode 前：

- 拒绝继承 `LD_PRELOAD`；
- 校验 pinned SCIP binary、`libscip` 和 child shim SHA256；
- 将 `LD_LIBRARY_PATH` 重置为 frozen SCIP lib dir，避免继承系统 SCIP 9.x library path；
- 固定 `SCIPOPTDIR`；
- probe `Model().version() / pyscipopt.__version__ / ecole.__version__`；
- 通过 ELF loader probe SCIP CLI version；
- wrapper 只有 `--verify-only / --python / --scip`，无 arbitrary-command formal mode。

`configs/steiner/environment.lock.yml` 同时登记系统 SCIP 9.2.2 为不允许进入 formal Steiner stack。静态实现确实针对“9.2.2 混入”设计了 fail-closed 防线；本审计未实际执行 `--verify-only`。

### 4.3 Git 边界

**总体一致。** 各 substantive content patch 主要落在 `configs/steiner`、`docs/steiner`、`python/steiner_branching`、`scripts/steiner`、`tests/steiner` 和 S03 native probe/Makefile target；`content..phase` 主要为 audit/report/status metadata。S02→S03 与 S03→S04 之间的 Git governance/runtime hardening commits 包含在整体 migration range，不能因 base SHA 不连续就误判为漏 merge。

在已检查 patch/file scope 中未发现把 `build/**`、模型 checkpoints、S03 per-state raw shards 或 SteinLib/DIMACS raw corpora 纳入提交，也未发现修改用户旧航空主链默认行为的证据。完整本地 `git diff --check` 与 secret scan 因环境限制为 NOT_VERIFIED。

### 4.4 失败保留与 Gate 口径

**一致。** S00 contract 要求 failure/timeout/OOM/skip/invalid action/training seed 留痕；S03 committed summary 仍保留 46/90 timelimit、missing strong states 和 zero-branch buckets；聚合代码用 preregistered denominator 而不是只统计成功 task。没有发现为了过 Gate 删除失败样本或事后降低阈值的实质 diff。

### 4.5 hash / machine-readable result

**静态内部一致，关键 runtime hash 重算 NOT_VERIFIED。** config、audit packet、Gate summary、snapshot 对 solver stack/model/schema/Gate 数字的引用没有发现相互冲突；S04 参数量可独立从源码重算为 68,161。由于本环境不能运行 frozen stack，snapshot byte hash 和 runtime output hash 未独立重生成。

### 4.6 科研主张边界

**一致。** 当前证据链只支持：

```text
S02 correctness
→ S03 branchability / resource feasibility
→ S04 untrained B0 engineering parity
```

还不存在：

```text
strong-branch teacher quality
→ imitation quality
→ RL improvement
→ SCIP speedup/generalization
```

S00--S04 没有 teacher/IL/RL 训练，因此任何“learned policy 已有效”“优于 SCIP default”“可泛化到 Steiner 或所有 graph MILP”的表述当前都不成立。

---

## 5. Blocking issues

### B1 — HIGH：Ecole variable row 与 PySCIPOpt transformed variable name 的 positional alignment 未被契约化

**影响**：S05 strong-branch teacher 的 label 需要精确对应 SPG edge action。若 Ecole row `i` 与 `getVars()[i]` 不是同一个 SCIP variable，则 edge ID、teacher score、candidate closure 与训练 label 会发生静默错配；这直接破坏 imitation dataset 的科学有效性。

**证据**：

- `python/steiner_branching/solver/bipartite_observation.py:278-284`：按 `getVars(transformed=True)` 返回位置生成 `variable_names`。
- `python/steiner_branching/solver/bipartite_observation.py:226-250`：用同一个 Ecole action index 同时索引 feature row 和 positional `variable_names`。
- Ecole 0.8.1 `NodeBipartite` 文档：variable features 按 SCIP `probindex` 排列/action-set 可用于索引。
- PySCIPOpt `getVars()` 文档：没有提供“返回列表顺序 == probindex 顺序”的公开保证。
- `tests/steiner/test_s04_bipartite.py` 的真实 stack test 只断言 `mapping_rate==1.0`，没有逐 variable 验证 `Ecole row index == SCIP probindex == exact transformed variable name`，也没有 permutation-negative test。

**最小修复**：不要依赖 `getVars()` list position。建立一个由 SCIP `probindex` 显式索引的 `probindex -> transformed variable name` 映射，再按 Ecole row index 构造 `variable_names`；若任一 probindex 缺失、重复、越界或名称/metadata 不一致，立即 fail closed。若当前 PySCIPOpt 版本没有稳定暴露 probindex，应在冻结 SCIP 8.0.4 stack 中增加一个最小 native/helper bridge，而不是用猜测的排序规则替代。

**必须补的测试**：

1. frozen SCIP/Ecole integration：对每个 transformed variable 验证 `Ecole row i` 对应的 SCIP `probindex == i`，并验证该 probindex 对应 exact variable name；至少覆盖 `x`、flow/continuous、parallel edge 情况。
2. permutation-negative test：人工打乱 `getVars()` 返回顺序或构造等价 mock，确认新实现不会静默得到 100% mapping，而是仍通过 probindex 纠正或直接 fail。
3. 对 3-state S04 snapshot 重跑 31/31 mapping、full/closure parity、argmax，并重新提交 machine-readable hashes。
4. 完整 `tests/steiner` 必须回到冻结预期：`75 passed, 1 expected skipped`（无 `STEINER_PACE_DEV_ROOT` 时）。

**关闭阶段**：S04 remediation / S05 entry gate 之前。**在 B1 关闭前不得生成 teacher dataset。**

---

## 6. Non-blocking issues

### N1 — S04 incumbent NaN→0 没有 missingness bit

**证据**：`bipartite_observation.py` 只允许 features 13/14 的 NaN，并只对这两列做 zero sentinel，其他 nonfinite fail closed。

**判断**：当前不阻塞 S05。19/5/1 schema 已明确把“无 incumbent”编码为 0 sentinel；增加 missingness bit 会改变 frozen feature schema，不能作为无版本变化的小修。S05 teacher pilot 应额外记录“无 incumbent state 占比”，若后续实验显示 sentinel 与真实 0 分布混淆，再以 `milp_bipartite_v2` 做受控 ablation。

**建议关闭阶段**：S05/S06 feature ablation 决策点，而不是 S04 hotfix。

### N2 — S03 raw shards 不能从公共远端独立重算 aggregate

**证据**：raw per-task shards 按 policy ignored，仅保留 aggregator、tests、raw index 说明与 committed `S03_GATE_SUMMARY.json`。

**判断**：不构成算法错误，也不应要求把可能受许可/体积约束的 raw 全部提交。但对最终论文级审计，建议保留一个内部 immutable bundle manifest：每个 shard 的 relative path、SHA256、task key、bytes，允许有权限的审计环境从内部归档重算 committed aggregate。

**建议关闭阶段**：S05 teacher 数据生成前建立统一 raw-manifest 规范，S12 final audit 时强制使用。

### N3 — 本次外部审计没有实际运行 frozen SCIP/Ecole stack

这是审计环境限制，而不是仓库缺陷。进入 S05 前仍应由能够 checkout 固定 SHA、访问冻结依赖的环境执行第 7 节 runtime checklist，并把日志作为新的 audit evidence 保存。

**建议关闭阶段**：B1 修复后的 S04 re-audit。

### N4 — 旧航空 4 个已登记 regression failures

`docs/steiner/AVIATION_REGRESSION_BACKLOG.md` 已将其与 Steiner migration 隔离；当前 Steiner package/config/action/schema 不依赖旧航空 Prim/DSU 语义，S00--S04 substantive diff 也没有要求借修 Steiner 顺手改变旧航空默认行为。

**判断**：不阻塞 S05。只有未来共享 runtime/config 发生重新耦合时才需要升级为 blocker。

---

## 7. 复现实验检查

| 检查项 | 状态 | 审计说明 |
|---|---|---|
| fixed SHA / phase SHA / overall range 静态核对 | **STATIC VERIFIED** | 使用固定 GitHub tree/commit/patch，不使用 branch tip |
| S00--S04 指定文档/配置/源码/测试静态检查 | **STATIC VERIFIED** | 已检查核心科研契约与实现路径 |
| S00--S04 `base..content` substantive scope | **STATIC VERIFIED** | 阶段 patch/file scope 与目的基本一致 |
| `content..phase` metadata scope | **STATIC VERIFIED** | audit/status/commands 为主；未发现借 metadata 改科研算法 |
| intermediate governance/runtime commits | **STATIC VERIFIED** | 位于整体 range 内；最终治理/runtime 文件已检查 |
| S02 rooted MCF 数学正确性 | **STATIC VERIFIED** | 目标、flow conservation、capacity linking、root 逻辑成立 |
| S03 aggregate code denominator 逻辑 | **STATIC VERIFIED** | formal denominator/missing strong/failure retention 未见缩分母 |
| S03 raw shard-level 重算 | **NOT_VERIFIED** | raw ignored，不在远端 |
| S04 68,161 参数量 | **STATIC + INDEPENDENTLY RECALCULATED** | 由源码层尺寸重算一致 |
| S04 closure receptive-field 充分性 | **STATIC VERIFIED** | 与 1-round variable→constraint→variable 完全匹配 |
| S04 Ecole↔SCIP variable identity | **INSUFFICIENT EVIDENCE** | positional join 没有 probindex contract/test |
| `git diff --check` | **NOT_VERIFIED** | 本环境无法本地 checkout |
| `run_with_scip804.sh --verify-only` | **NOT_VERIFIED** | 本环境无冻结 runtime checkout |
| full Steiner pytest `75 passed, 1 expected skipped` | **NOT_VERIFIED** | committed report 可读，但本审计未重新运行 |
| S04 snapshot byte-for-byte regeneration / `cmp` | **NOT_VERIFIED** | 未重新运行 frozen stack |
| sealed final test 未被本审计访问 | **VERIFIED BY AUDIT PROCEDURE** | 审计没有下载/读取 sealed bytes |
| exhaustive local secret/PAT scan | **NOT_VERIFIED** | 已检查 patch/file scope未见 secret，但未能本地全仓 grep/scanner |

---

## 8. 当前可以成立的结论 与 当前不能成立的结论

### 当前可以成立的结论

1. Steiner migration 已建立与旧航空 Prim/DSU 研究语义隔离的独立栈，并冻结了 problem/split/seed/Gate/final-test contract。
2. 当前 rooted MCF formulation 和独立 solution checker 的静态实现没有发现 correctness blocker；parser/canonicalization/variable naming 采用 strict、deterministic 路径。
3. S03 committed evidence 支持“在**冻结 pilot 矩阵**上存在足够的 branchability 和 strong-branch signal，可继续设计 teacher pilot”，同时 46/90 timelimit 等失败限制了适用范围。
4. S04 的 `milp_bipartite_v1` 是标准 19/5/1 未训练 B0；一轮 exact closure 与 GCNN receptive field 匹配；68,161 参数量可独立复核；3-state snapshot 只构成 engineering parity evidence。
5. SCIP 8.0.4 wrapper 的静态实现对系统 SCIP 9.2.2 混入采取了 checksum、library-path、version-probe 与 restricted-mode 防护。

### 当前不能成立的结论

1. **不能声称 S04 action mapping 已被充分证明。** 当前 31/31 只能证明样本运行时 mapping function 接受了这些 action，不能证明 Ecole row 与 positional `getVars()` name 是同一 SCIP variable。
2. **不能开始 S05 teacher/IL 数据生产。** 在 action identity 关闭前，teacher label 可能被静默绑定到错误 edge。
3. 不能声称 learned policy 有效，因为 S00--S04 没有 teacher imitation 或 RL training。
4. 不能声称优于 SCIP default、获得 wall-clock speedup、节点数下降或跨分布泛化。
5. 不能把 S03 0.70 branchability / 85% strong-valid 外推为“所有 Steiner 图”或“所有 graph MILP”的普遍性质。
6. 不能把 S04 1 graph / 3 states 的 parity 当成 model quality、production performance 或科研最终实验。
7. 本审计不能声称亲自复现了 `75 passed, 1 skipped`、S04 byte-identical snapshot 或 S03 raw aggregate；这些均按上文标为 NOT_VERIFIED。

---

## 9. 复审清单

由于本次为 **CONDITIONAL PASS**，进入 S05 前只要求最小 remediation，不建议借机扩展研究范围。

- [ ] **关闭 B1**：以 SCIP `probindex` 为显式主键构建 Ecole row ↔ transformed variable name 映射，删除对 `getVars()` list position 的隐式依赖。
- [ ] 新增 frozen-stack **probindex identity integration test**，逐 row 验证 exact variable identity。
- [ ] 新增 **permutation-negative test**，确保变量列表乱序不会静默得到错误 edge mapping。
- [ ] 在 `123e4f3...` 后的最小修复 commit 上重新执行 `scripts/steiner/run_with_scip804.sh --verify-only`。
- [ ] 重新执行完整 Steiner suite，要求 `75 passed, 1 expected skipped`；不得为了消除 PACE odd dev skip 访问 sealed data。
- [ ] 重新生成 S04 `S04_FORWARD_SNAPSHOT.json` / `S04_GATE_SUMMARY.json`，执行 byte comparison，并更新 hash/audit packet；31/31 mapping、full/closure parity、argmax 必须继续通过。
- [ ] 对 remediation diff 做 `git diff --check` 和 secret/raw/build/checkpoint scan；不要将 S03 raw shards、raw corpora、checkpoint、逐状态日志提交到 Git。
- [ ] 在 S05 plan 中显式继承 S03 teacher 范围限制：保留 timeout/missing strong states，禁止只采“容易成功”的 states 改变 denominator；teacher pilot 先小规模验证 label coverage/tie rate/collection cost。
- [ ] 首次 CUDA/训练前按 master plan 完成资源 preflight，但不得把 CUDA preflight 与 action-identity 修复混为同一个 Gate。
- [ ] 将本次审计报告与 B1 remediation/re-audit 结果提交为审计记录；**只有 B1 复审通过后才开始 S05 teacher collection**。

---

## 最终审计决定

**CONDITIONAL PASS — S00、S01、S02、S03 可保留；S04 需要一个局部但科研上关键的 action-identity 修复与复审。当前 phase head `123e4f3daaef1b8d15f2cc8f02a06f6edd6887fc` 不应直接进入 S05。**
