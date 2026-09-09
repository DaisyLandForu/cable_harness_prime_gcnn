# S06 IL online evaluation v1 — GPT pre-execution audit request

请只读审计 S06 protocol、online runner 和六分片执行实现，不修改仓库。本次是
正式 online solve 前的实现/协议预审；没有 S06 正式结果，不要判定 scientific
Gate PASS，也不要授权 S07 或访问 test/final。

## 固定对象

- branch：`research/steiner-migration`
- frozen base：`ed0db3ff53a73cbb71de10f35e3471a6c139f146`
- implementation content head：`9866a3743ff1c2d6ef580aedf1f91629db0c6049`
- substantive range：
  `ed0db3ff53a73cbb71de10f35e3471a6c139f146..9866a3743ff1c2d6ef580aedf1f91629db0c6049`
- protocol：`configs/steiner/experiments/s06_il_online_v1.yml`
- protocol explanation：
  `docs/steiner/phases/S06/S06_PREEXECUTION_PROTOCOL.md`
- instance identities：
  `configs/steiner/experiments/s06_online_instances_v1.json`
- audit entry：`docs/steiner/phases/S06/S06_AUDIT_PACKET.md`

## 固定 hashes

- protocol YAML：
  `8d0daa708c5d2ace6bbd32da867dd978f730c97216df74277208e86012db50da`
- instance manifest：
  `b50f8048d8ab27a1fa2168e69ef179cbfc490116399180f2bb4a963bf04b292b`
- protocol explanation：
  `db501869563be02fdc6e53a1074e8b6d0c46b306078ce2bcc671569faf9cc50a`
- online implementation：
  `fa24b60483c55a135e5a5bbd126d513eaf05528042f4277e21753f0f27d64161`
- distributed runner：
  `9bfba18db60d648629b18c868b7839e06cd289c9f53c02c87d0a456c49fa458f`
- tests：
  `759156a7383dec6e7a61791309f75c98dfc4d1c8fbff1d7bf4098683da9f25e6`

## 请重点核验

1. seed-202 checkpoint 是否在 S06 outcome 前已固定，且 model/config/
   normalization bytes fail closed；
2. B0 是否只对 SCIP 当前合法 fractional binary `stp_x_*` 候选评分，并沿用 S04
   canonical probindex identity；mapping/NaN/非法 action 是否作为失败而非 fallback；
3. random、mostinf、relpscost、SCIP default 和 fullstrong 是否使用同一 P1
   formulation/seed/limit/thread 条件，且 fullstrong 明确不参与 Gate；
4. solved/PAR-2/PDI/first incumbent/root LP/nodes/overhead/correctness 是否定义充分，
   timeout、limit、OOM 和 solver error 是否完整保留；
5. 统计是否以 30 个 base-graph lineage bootstrap，五个 solver seeds 在 lineage
   内保持一起，且 Gate 必须同时超过 random 和 mostinf；
6. 六分片是否真正按完整 lineage 分割：每 shard 五图、每 family 一图、每图
   25 个 main tasks、union=750、intersection=0；
7. main/trace 六分片 barrier、wrong-shard 检查、资源身份匹配、duplicate lock 和
   finalizer 是否足以防止多 custom job 覆盖、漏跑或错误聚合；
8. `execution_authorized: false`、独立 activation、protected-input clean check 和
   test/final/S07 prohibition 是否阻止提前执行和阶段越界。

## 本地证据

- targeted S06：10 passed；
- complete frozen-stack Steiner suite：113 passed、1 个既有 expected PACE skip；
- compileall、SCIP 8.0.4 wrapper identity、validate-only、shell syntax、
  `git diff --check`：PASS；
- validate-only：30 lineages、750 main、5 fullstrong、6×125 main tasks；
- formal shards：0；activation record：不存在；scientific Gate：NOT_EVALUATED。

请返回 `PASS`、`CONDITIONAL PASS` 或 `FAIL`，逐项列出 blocking findings。只有
PASS 才允许登记独立 activation 并提交六个 main custom jobs；即使 PASS，也不等于
S06 scientific PASS，不允许创建 S06 audited tag、开始 S07 或访问 test/final。
