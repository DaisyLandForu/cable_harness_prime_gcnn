# S06 pre-execution B2/B3 — GPT second focused re-audit request

请只读复审 S06 的第二次窄范围 remediation，不修改仓库。本次只需判断上一轮
`CONDITIONAL PASS` 中仍为 OPEN 的 B2、B3 是否关闭。B1、B4 已由上一轮明确
CLOSED；除非发现回归，不需重新展开。仓库仍没有 S06 formal result，不要判定
scientific Gate PASS，不要授权 S07 或 test/final。

## 固定对象

- branch：`research/steiner-migration`
- original implementation head：
  `9866a3743ff1c2d6ef580aedf1f91629db0c6049`
- first remediation head：
  `a8fbc0986068171344387ceb68e1ee5ab5bbefbc`
- first focused audit-request head：
  `98a535d19924c80df47fcb3f8153d7fdf89ccf86`
- B2/B3 remediation content head：
  `a29ef9eeb818c1694f78d2de1b29436f1861f8c3`
- focused substantive range：
  `98a535d19924c80df47fcb3f8153d7fdf89ccf86..a29ef9eeb818c1694f78d2de1b29436f1861f8c3`
- first focused audit record：
  `docs/steiner/audits/S06_PREEXECUTION_REMEDIATION_AUDIT_RECORD.json`
- B2/B3 remediation explanation：
  `docs/steiner/phases/S06/S06_PREEXECUTION_REMEDIATION_V2.md`
- audit entry：`docs/steiner/phases/S06/S06_AUDIT_PACKET.md`

## 固定 hashes

- protocol YAML（本轮未修改）：
  `e7f7e9060c25fa038a2749ca43afe6a2769c3652e93697612b8804269750f827`
- instance manifest（未修改）：
  `b50f8048d8ab27a1fa2168e69ef179cbfc490116399180f2bb4a963bf04b292b`
- protocol explanation：
  `23139ea8810c608a6e816a1a4fd51d13ff3f48ca3f78c52bd4941c555e16dd48`
- first-remediation historical explanation：
  `fc4c4259c63ce29055c41ef113298cb37513692380c78d973e3770d811380fdd`
- B2/B3 remediation explanation：
  `b922ba61f4a10efa9c01a4aa3894b1c3f4b78e51794f4d4c3f9a5f52abc789c6`
- audit packet：
  `8b0937e88b86d89390f6c5f640e6456387fd470765a879301176242258031f52`
- first focused CONDITIONAL PASS record：
  `6564c8753f2354e9b59b536ee6a2c62661ae9f69b0fe70e5763e51806202e82d`
- activation loader / online implementation：
  `d1f1a228aed066a7c32f3ce0202b3b502505e22e11b70daa6df246fda706de75`
- distributed runner / Python lock owner：
  `ed95e1e15ccc8dd8cde3b64b1a088a1b10a7fc6cb0dfd8dff69b1483eea9d3f7`
- shell finalizer：
  `8b44c34adc312f83fdfc586c1a8ff21139f88215f43e41153cffadd442d3a954`
- tests：
  `08e596c83b8ea76203cef16898e74d125daed1023b75030ff403bfd3c9e7aca7`

## B2 修复：activation committed-byte identity

`load_s06_activation()` 现在按以下顺序 fail closed：

1. resolve activation path 并读取 local bytes；
2. 要求 resolved path 位于 repository 内；
3. 执行 `git show HEAD:<repository-relative-path>`，失败则拒绝；
4. 要求 committed bytes 与 local bytes 完全相等；
5. 之后才解析 JSON 并核验 PASS、protocol/instance/S05/head/runtime 字段；
6. 继续执行 audited-head ancestry 和 protected executable clean checks。

因此不能再通过未提交地修改 `container_runtime_fingerprint` 来适配另一套环境。
运行时记录的 activation SHA 只可能来自已经通过 HEAD byte-equivalence 的文件。

新增负向测试模拟：Git 始终返回原 committed activation bytes，本地把 runtime
fingerprint 从 `a...a` 改成 `b...b`；loader 必须以
`activation differs from its committed bytes at HEAD` 失败。

## B3 修复：evidence-writing Python owns OS lock

Shell-owned lock 和 `S06_AGGREGATE_LOCK_HELD=1` guard 已删除。现在 `_aggregate()`
本身：

```text
open <formal-run>/locks/aggregate.lock
flock(LOCK_EX | LOCK_NB)
hold descriptor
  -> existing-output refusal
  -> main/trace barriers
  -> summary atomic write
  -> run-manifest atomic write
unlock only after the complete critical section
```

不论从 shell finalizer 还是直接 Python 入口进入，实际 evidence writer 都必须获得
同一 kernel lock。伪造旧环境变量没有作用。负向测试在持锁期间显式设置旧变量为
`1`，再直接调用 `_aggregate()`；第二个 Python aggregator 仍因 lock occupied 失败。
既有 summary/run-manifest overwrite refusal 保持不变。

## 未改变边界

本轮没有修改 protocol YAML。checkpoint、30 instances、5 solver seeds、methods、
P1、750-task matrix、trace、bootstrap、Gate thresholds 和 claim boundary 均未改变。
B1 的 failure PAR-2 schema 没有回退，B4 audit packet 仍存在。activation 文件仍不
存在，没有 formal task、checkpoint retraining/reselection、S07 work 或 test/final
access。

## 本地验证

- targeted S06：`17 passed`；
- complete frozen-stack Steiner suite：`120 passed, 1 expected PACE skip`；
- compileall、shell syntax、SCIP 8.0.4 wrapper、validate-only、`git diff --check`：
  PASS；
- validate-only runtime fingerprint：
  `fe7a032641aba58090f59cfa9d5b088cf76da23b799f2099a34e8d99b17b5687`；
- S06 formal result：`NOT_RUN`；scientific Gate：`NOT_EVALUATED`。

请返回 `PASS`、`CONDITIONAL PASS` 或 `FAIL`，并明确写出：

```text
B2 = CLOSED/OPEN
B3 = CLOSED/OPEN
```

只有 `PASS` 且 B2/B3 均 CLOSED，才允许登记并提交独立 activation，之后提交六个
main custom jobs。即使 PASS，也不等于 S06 scientific PASS，不允许创建 S06
audited tag、进入 S07 或访问 test/final。
