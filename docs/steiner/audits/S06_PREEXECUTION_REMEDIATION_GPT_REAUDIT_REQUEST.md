# S06 pre-execution B1--B4 remediation — GPT focused re-audit request

请只读复审 S06 的窄范围 pre-execution remediation，不修改仓库。本次只需判断
首次 `CONDITIONAL PASS` 的 B1--B4 是否关闭。仓库仍没有 S06 formal result；不要
判定 scientific Gate PASS，不要授权 S07 或 test/final。

## 固定对象

- branch：`research/steiner-migration`
- S06 frozen base：`ed0db3ff53a73cbb71de10f35e3471a6c139f146`
- original implementation content head：
  `9866a3743ff1c2d6ef580aedf1f91629db0c6049`
- original evidence/process head：
  `bcd9e33e9a09be856fc5f61078ccb4875ea325c1`
- remediation content head：
  `a8fbc0986068171344387ceb68e1ee5ab5bbefbc`
- focused remediation range：
  `bcd9e33e9a09be856fc5f61078ccb4875ea325c1..a8fbc0986068171344387ceb68e1ee5ab5bbefbc`
- complete post-implementation evidence/remediation range：
  `9866a3743ff1c2d6ef580aedf1f91629db0c6049..a8fbc0986068171344387ceb68e1ee5ab5bbefbc`
- initial audit record：
  `docs/steiner/audits/S06_PREEXECUTION_AUDIT_RECORD.json`
- remediation explanation：
  `docs/steiner/phases/S06/S06_PREEXECUTION_REMEDIATION.md`
- audit entry：`docs/steiner/phases/S06/S06_AUDIT_PACKET.md`

## 固定 hashes

- protocol YAML：
  `e7f7e9060c25fa038a2749ca43afe6a2769c3652e93697612b8804269750f827`
- instance manifest（未改变）：
  `b50f8048d8ab27a1fa2168e69ef179cbfc490116399180f2bb4a963bf04b292b`
- environment lock：
  `f70afe548f2b640a3c1375686ad8c8ef4dced63d0229c9fa4eb36e62f6d7628e`
- protocol explanation：
  `6cbd5473115524813549b628369b1dbcc11e9f849762fafe2789bad384b0fb62`
- remediation explanation：
  `15f00d21a637edbf648c33471dce2422cbaacd32b457105e471eda7cc1badc50`
- audit packet：
  `58914f3e4fd62c036794577cfc19e5d414e9266edbbd6729542bbe5ccd0a03cc`
- initial conditional-audit record：
  `142baa5fd1d6e7871e905ed5918d188bb60d9a0ecda29778685c510b8ab000d2`
- online implementation：
  `477b8724aab0231f38a9333602ba9b85174f2cfdd807abfac2d4fb808c58428f`
- distributed runner：
  `c2f18dd0313db2603ef08aff80130743c7f5ece80c17d94d5de10bdfa36e2437`
- locked finalizer：
  `b0a963ca416f345ef3c44031f85114418cff38edd35506d5a63eb23e1d9a04ee`
- tests：
  `c6c609482617e34d0e91d754c1101a3897213a0f7ad4c78900d3ae8acb9be417`

## B1 — failure PAR-2 semantics

Exception/invalid-policy task 现在写完整 terminal result：

```text
solved = false
failure_class = invalid_action | mapping_failure | nan_score |
                solver_or_runtime_error
par2_seconds = 1200
primal_dual_integral = null
pdi_available = false
```

Aggregator 不再因为 `execution_status=solver_error` 就删除该 task；只要配对 task
envelope 存在，它仍进入 PAR-2 pair。PDI 不伪造，PDI completeness/correctness/
solver-error Gate 均会 FAIL。负向测试覆盖 invalid action、mapping、NaN 和 generic
solver error；单一注入失败仍得到 150 个 PAR-2 pairs、149 个 PDI pairs、Gate FAIL。

## B2 — registered runtime identity

每个 shard 在任何 task 前显式验证：

```text
effective CPU >= 8
memory >= 96 GiB
visible GPU == 0
SCIP stack == scip804-ecole081-pyscipopt430
environment lock SHA exact
runtime fingerprint == activation record
audited executable content == activation head
activation record checksum present
```

Runtime fingerprint 覆盖 environment lock、Python executable bytes、Python/numpy/
PyYAML/Torch/Ecole/PySCIPOpt versions 和 solver stack。父进程把同一个完整 runtime
identity 写入 shard manifest 并传给每个隔离 task；task 再验证并写入 envelope。
Barrier 同时要求 task identity 与其 manifest byte-equivalent，以及六个 jobs 的
registered compatibility tuple 相同。hostname 不要求相同。

Validate-only 在当前 frozen runtime 产生 fingerprint：
`fe7a032641aba58090f59cfa9d5b088cf76da23b799f2099a34e8d99b17b5687`。
只有本次复审 PASS 后，才可将它写入独立 activation；若 custom job runtime 不同，
会在 task 前 FAIL，不会产生可混用的 formal evidence。

## B3 — aggregate immutability

`finalize_s06_online.sh` 现在先获得 non-blocking `aggregate.lock`，并设置只有该
launcher 才能提供的 aggregate guard。Python aggregate 在读取 barrier/写结果前
拒绝既有 `S06_GATE_SUMMARY.json` 或 run `manifest.json`。两个 finalizer、直接绕过
launcher、或在已有证据上重跑都会 fail closed；测试覆盖 lock 和 existing-output。

## B4 — evidence packet

`S06_AUDIT_PACKET.md` 已存在于固定 remediation head `a8fbc098...`，记录原始
CONDITIONAL PASS、B1--B4 closure map、测试和 hashes。首次审计请求/记录保留原
head 和原 verdict，没有覆盖历史。

## 未改变的科学边界

seed-202 checkpoint、30 lineages、5 solver seeds、5 main methods、P1 600s/
200k nodes/8192MB/1 thread、750-task matrix、fullstrong diagnostic subset、六分片
assignment、trace policy、30-lineage bootstrap、random+mostinf 双 Gate、阈值和
claim boundary 均未改变。没有 formal solve、checkpoint reselection/retraining、
S07 implementation 或 test/final access。YAML 仍为 `execution_authorized: false`，
activation 文件不存在。

## 本地验证

- targeted S06：`17 passed`；
- complete frozen-stack Steiner suite：`120 passed, 1 expected PACE skip`；
- compileall、shell syntax、SCIP 8.0.4 wrapper、validate-only、`git diff --check`：
  PASS；
- validate-only：30 lineages、750 main、5 fullstrong、6×125 main tasks；
- S06 formal result：`NOT_RUN`；scientific Gate：`NOT_EVALUATED`。

请返回 `PASS`、`CONDITIONAL PASS` 或 `FAIL`，并明确写出 B1、B2、B3、B4 各自
`CLOSED/OPEN`。只有 `PASS` 才允许登记独立 activation 并提交六个 main custom
jobs；即使 PASS，也不等于 S06 scientific PASS，不允许创建 S06 audited tag、进入
S07 或访问 test/final。
