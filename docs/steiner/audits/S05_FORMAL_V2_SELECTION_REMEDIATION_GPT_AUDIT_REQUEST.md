# S05 formal-v2 selection remediation — GPT pre-execution audit request

请只读审计 formal-v1 teacher FAIL 后的 v2 修订，不修改仓库。当前不授权
reselection、formal training、S06 或 test/final 访问。

## 固定审计对象

- branch：`research/steiner-migration`
- audited formal-v1 protocol content：
  `d1717a7ecb6043efd71678175a92325ac9ff4208`
- formal-v1 implementation/run head：
  `13a8a83761f660c96cb238471de2e2c0a2d54a8b`
- earlier concurrency-A1 registration head：
  `d858aa31b7d583c945892c3c284ea486bc7e5799`
- v2 remediation content head：
  `6404c047e49b505a2384bd30aae16de8b621f09e`
- substantive range：
  `d858aa31b7d583c945892c3c284ea486bc7e5799..6404c047e49b505a2384bd30aae16de8b621f09e`
- v2 YAML：
  `configs/steiner/experiments/s05_teacher_il_formal_v2_selection_remediation.yml`
- v2 explanation：
  `docs/steiner/phases/S05/S05_FORMAL_V2_SELECTION_REMEDIATION.md`
- immutable v1 failure record：
  `docs/steiner/phases/S05/S05_FORMAL_V1_TEACHER_FAILURE.md`
- v2 YAML SHA-256：
  `91e157a8d7853133e3c4fe5865e863763874e41622febc87b94a39e54f6b7913`
- v2 explanation SHA-256：
  `97bdc9b83b0f5eaa89951230b601258b92038297c1c5b94e8050f5e49a3cfb1a`
- v1 failure record SHA-256：
  `e909efa4afa85ee6359d6f8a4f621ab04bf3b703d6d2bb08f9be84f878041c46`
- retained raw manifest SHA-256（raw bytes 不入 Git）：
  `2bb3b4875d173571df5e4fb7e9c1e1d3f4615708308e891f4e4ccdbaac4c149c`

## 已发生且不得改写的 v1 结果

- 315/315 tasks completed；0 task failures；
- 3,431 observed、3,207 valid；valid/expected = 63.630952%；
- all-tie/valid = 0.436545%；mapping = 88,549/88,549；leakage = 0；
- validation-select = 160/160；validation-gate = 320/320；
- train = 527/640，因此 `all_state_quotas_met=false`，teacher Gate **FAIL**；
- shortages：sparse-large 0/48、geometric-medium 3/64、bridge-medium 60/64；
- watcher 正确未启动 seed 101/202；没有 formal model result 可供事后挑选。

## v2 请求的修复

不重复执行相同的 deterministic teacher tasks，也不删除 v1 失败。V2 对 sealed
manifest 做 checksum-verified reselection：先按原 bucket targets 取满可用状态，
某 family 不足 128 时，只从同 family 未使用的 valid/unique states 中按固定顺序
补足。仍然是五个 family 各 128、总计 640；validation membership/count 不变。

固定 fallback ordering：

```text
state_index, bucket_order, graph_sha256, teacher_seed, semantic_sha256
```

它在 sealed manifest 上唯一产生：

- sparse：small 56、medium 72、large 0；
- geometric：small 125、medium 3；
- grid：medium 64、large 64；
- community：medium 128；
- bridge：medium 60、large 68。

此前 A1 尚未激活，现由 v2 supersede；v2 同时冻结最多五个独立 one-V100 seed
jobs。每个 seed 的数据、初始化、optimizer、RNG、report/checkpoint 均独立，无 DDP。

## 请重点判断

1. v1 是否被诚实保留为 FAIL，是否不存在删样本、替换任务、伪造 teacher PASS；
2. 原样重跑相同 task/seed 无法修复确定性 quota shortage，因此复用 checksum-sealed
   teacher evidence 是否合理；
3. same-family fallback 是否可以作为透明的新协议，而不是偷偷把 640 或 5×128
   降低；若你认为它仍构成不可接受的 post-hoc bucket Gate relaxation，请判 FAIL；
4. 固定 ordering、manifest/shard checksum、role/semantic isolation 是否足以让
   reselection 唯一且 fail-closed；
5. v2 是否明确限制结论：不能据此宣称 sparse-large 或 geometric-medium scale
   coverage/generalization；
6. 640 states、五 seeds、40 epochs、optimizer、CUDA determinism、validation、
   statistics、quality/learning Gates、exact reload 0、seed-202 handoff、失败保留、
   test/final prohibition 是否保持不变；
7. 五个独立 V100 scheduler jobs 是否只改变 wall-clock overlap；
8. `execution_authorized: false` 是否确保只有本次 PASS 被单独登记后才可实现并执行
   reselection/training。

## 本地验证

- v2 static + S05 CUDA/reload targeted suite：17 passed；
- complete frozen-stack Steiner suite：95 passed、1 expected PACE skip；
- raw manifest SHA 与机器可读 v2 failure evidence 一致；
- formal training/test/final：NOT_RUN。

请返回 `PASS`、`CONDITIONAL PASS` 或 `FAIL`，逐项列出 blocking findings。
只有 `PASS` 才允许实现 v2 fail-closed reselector/loader/launchers，再运行五个 formal
seeds；PASS 不等于完整 S05 PASS，也不授权 S06。
