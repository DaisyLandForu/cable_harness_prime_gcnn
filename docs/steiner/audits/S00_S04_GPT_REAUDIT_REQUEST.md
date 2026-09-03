# S00--S04 GPT Re-audit Request — S04 probindex remediation v2

请只读复审 Steiner RL branching 迁移 S04 remediation，不修改仓库。

## 固定审计对象

- branch：`research/steiner-migration`
- v1 phase head：`123e4f3daaef1b8d15f2cc8f02a06f6edd6887fc`
- remediation content head：`4ab54ffa2b80f06ac8a9ecfe662a04df7899b072`
- remediation substantive range：
  `123e4f3daaef1b8d15f2cc8f02a06f6edd6887fc..4ab54ffa2b80f06ac8a9ecfe662a04df7899b072`
- 首次审计原文：`docs/steiner/audits/S00_S04_GPT_AUDIT.md`
- 入口：`docs/steiner/phases/S04/S04_AUDIT_PACKET.md`

## 本次只需确认的 blocking finding

首次审计给出 CONDITIONAL PASS，B1 指出 Ecole variable row 与 PySCIPOpt
`getVars(transformed=True)` list position 的对应关系没有公开契约。请核对 v2 是否
真正实现并验证以下唯一 canonical identity：

```text
Ecole variable row i
  -> SCIP probindex i
  -> transformed SCIP_VAR
  -> transformed name
  -> original stp_x name
  -> edge_id
```

重点检查：

1. `python/steiner_branching/solver/scip_identity.py` 是否用 probindex 重排，而非
   仅对 `getVars()` list position 作事后名称恢复；
2. `0..n_vars-1` 是否在每次 extraction 时形成完整双射；count/missing、duplicate、
   out-of-range、非法名称是否 fail closed；
3. C bridge 是否只加载 repository-frozen SCIP 8.0.4 绝对路径，核对 wrapper
   identity、prefix 和 checksum，且不会回退到系统 `libscip.so`；
4. permutation/negative/parallel-edge/real frozen integration tests 是否足以防止
   list-order regression；
5. `S04_GATE_SUMMARY.json` 是否明确包含
   `probindex_identity_complete: true`。

## 已有本地证据

- complete Steiner suite：78 passed、1 expected PACE skip；
- frozen states：3/3；canonical identity rows：2,943/2,943；
- legal action mapping：31/31；
- full/closure max error：0；argmax agreement：3/3；
- deterministic snapshot file SHA-256 未变化：
  `ac2ce0c14b134245221af5140a3008f3ec6067f8867491e7cc0d0b50e2036f2c`；
- remediation Gate summary SHA-256：
  `ea94f25e3c42fed2b464f9c914af521ed8e8d1a6537505bbb2a164b65f7b937b`；
- Gate：8/8 true，local PASS。

## 未改变的边界

19/5/1 schema、missingness policy、split/profile/seed、`1e-5` Gate、S03 90 tasks
和旧航空 backlog 均未改变；没有采 teacher、训练、checkpoint、validation/final
访问。用户只 waiver 了 S05 implementation scaffold，不代表本次审计 PASS。

请给出 `PASS`、`CONDITIONAL PASS` 或 `FAIL`，并明确判断上述 B1 是否关闭。只有
PASS 才允许创建 audited tag，并放行 S05 正式 teacher collection/训练。
