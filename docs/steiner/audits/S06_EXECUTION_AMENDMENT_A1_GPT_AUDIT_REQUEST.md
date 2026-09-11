# S06 execution amendment A1 — GPT focused pre-trace audit request

请只读审计 S06 main-wave evidence seal 和 runtime-compatibility amendment A1，
不要修改仓库、不要计算 model-vs-baseline effect，也不要把 main task 完成误判为
S06 scientific PASS。

## 固定对象

- branch：`research/steiner-migration`
- original activation/execution head：
  `a80d7459b9b1d4d5bff9743711984cd5a649ef97`
- amendment substantive content head：
  `54bba290f9c9683ae2409e7c43e414cf9d64cc0c`
- substantive range：
  `a80d7459b9b1d4d5bff9743711984cd5a649ef97..54bba290f9c9683ae2409e7c43e414cf9d64cc0c`
- base protocol：
  `configs/steiner/experiments/s06_il_online_v1.yml`
- amendment YAML：
  `configs/steiner/experiments/s06_execution_amendment_a1.yml`
- main-wave seal：
  `docs/steiner/phases/S06/S06_MAIN_WAVE_V1_SEAL.json`
- failure/remediation explanation：
  `docs/steiner/phases/S06/S06_MAIN_WAVE_BARRIER_FAILURE.md`
- audit packet：`docs/steiner/phases/S06/S06_AUDIT_PACKET.md`

## 固定 hashes

- base protocol：
  `e7f7e9060c25fa038a2749ca43afe6a2769c3652e93697612b8804269750f827`
- instance manifest：
  `b50f8048d8ab27a1fa2168e69ef179cbfc490116399180f2bb4a963bf04b292b`
- base activation：
  `2e8d30c47bf746762b8cdf2b9c605c16e45ab57c68d73d7d11a0f8e87b8a21fd`
- amendment YAML：
  `b862f81893fb5dd46635dfc55366b5961268de41d74dd2f6cf89f11705690abe`
- main-wave seal：
  `11925fc0e5c6169a0fbdb7c1770ebb0709cefe599f6c0018f07d875595a81603`
- sealed 761-file evidence tree：
  `671cb10a78a30e9f227e6b8b86a62d3ba4437a013ba9350850bb2ef1b1fb8964`
- failure/remediation explanation：
  `d12255b3214c431d0304bbcda56c01c4bceaf1a6c61a8562f3ee489992150d3d`
- audit packet：
  `fb615f8e94ee5ea1aad2f743528e1c841fb144f4a2da263900f1562848b3a547`
- distributed runner：
  `f506f009e2c18a6d41c5b4d932b4901cbe0c6b79146f28d001d796398e8b5273`
- tests：
  `0cb01ab3e829899e8ff88ea52d902a2acd08ec4a8f27cfbb44fbf31e57e27509`
- test report：
  `eb922a70f1606a157e2a6d08c2832ed35b47fbd30ca68fdeebd68ac4e9f2f46c`

## 事实边界

原 pre-execution 审计已 PASS，并由独立 base activation 授权 main。用户提交的六个
custom jobs 已生成：

- 750/750 Gate main task envelopes；
- 5/5 non-Gate fullstrong diagnostic envelopes；
- 6/6 completed main manifests；
- execution status 755 completed；solver status 399 optimal、356 timelimit；
- zero solver-error envelope。

这些只是 terminal/status counts。原 barrier 在读取/聚合 paired method effects、
生成 trace trigger 和创建 Gate summary 之前停止；没有 S06 scientific result。

停止原因是六个 scheduler jobs 落到三种 Xeon host，`cpu_model` 和
`cpu_affinity_count` 不相等。实际注册资源/运行身份全部相同：effective CPU 8.01、
memory 103080263680 bytes、0 GPU、Python/SCIP/environment/runtime/base activation/
audited executable/execution Git head 均一致。每个 graph lineage 的 5 methods ×
5 solver seeds 仍在同一 shard。

## A1 的唯一变化

跨 shard 必须相等的 canonical keys 现在是：

```text
cpu_quota_cores
effective_cpu_cores
memory_limit_bytes
gpu_visible_count
python_version
solver_stack_id
environment_lock_sha256
container_runtime_fingerprint
activation_record_sha256
audited_executable_content_head
git_head
```

`hostname`、`cpu_model`、`cpu_affinity_count` 继续逐 manifest 记录，但不再要求六个
独立 scheduler host 相等。每个 task 的完整 runtime identity 仍必须与所属 manifest
精确一致。

已有 main evidence 不重跑。实现先核对恰好 755 task files + 6 manifests 的 membership
和完整 byte tree；任何缺失、多余或 byte mutation 都 fail closed。A1 parent/internal
main entry 均拒绝执行。

只有本次返回 PASS 后，才会另行提交 audit record 和 amendment activation。随后
执行顺序固定为：

```text
verify sealed main bytes
  -> normalized main barrier
  -> derive diagnostic trace trigger
  -> six trace jobs
  -> complete trace barrier
  -> singleton aggregation
```

聚合器已经改为两个 barrier 都 PASS 后才计算 scientific aggregate。

## 未改变边界

Checkpoint、30 graphs、five solver seeds、five methods、P1 limits、SCIP single-thread
设置、task matrix、trace policy、metrics、failure retention、paired bootstrap、Gate
thresholds 和 claim boundary 全部不变。没有 main retry/replacement、model retraining、
test/final access 或 S07 implementation。

## 请重点判断

1. 将物理 hostname/CPU model/host affinity 从跨 shard equality 中移除，是否是合理的
   orchestration compatibility 修复，而不是降低注册资源或改变科学协议；
2. 同 lineage 的所有 methods/seeds 保持同 shard，是否维持 paired comparison 公平性；
3. 761-file seal、exact membership/bytes 和 main-rerun prohibition 是否足以避免
   retry-until-success、删失败样本或 evidence mutation；
4. legacy main evidence 是否被明确绑定 base activation/head，而 future trace 被绑定
   新 amendment activation，没有伪称旧 tasks 使用新协议；
5. normalized equality、per-task exact manifest identity 和负向测试是否仍 fail closed；
6. 两个 barrier 先于 aggregation 的顺序是否阻止 incomplete trace 下提前产出 Gate；
7. A1 是否确实没有修改模型、实例、seed、limit、metric、statistics 或 Gate；
8. `execution_authorized: false` 是否保证 PASS 后仍须独立 activation，且完整 S06
   Gate/S06 tag/S07/test-final 仍未授权。

## 本地证据

- targeted S06 suite：21 passed；
- complete frozen-stack Steiner suite：123 passed、2 skips（既有 optional PACE 和
  当前无 CUDA 的既有 real-CUDA test）；
- 真实 sealed-main replay：761/761 file tree、6/6 manifest barrier PASS；
- 模拟三种物理 host metadata 可以通过，任何 normalized key mismatch 仍 FAIL；
- byte mutation、unexpected file、wrong-shard、task/manifest runtime mismatch、main
  rerun 均 FAIL；
- compileall、SCIP 8.0.4 wrapper、shell syntax、validate-only、diff-check 全 PASS；
- no trace、no aggregate/Gate summary、no test/final access。

请返回 `PASS`、`CONDITIONAL PASS` 或 `FAIL`，并逐项列出 blocking findings。只有
PASS 才允许登记独立 amendment activation 和运行六个 trace jobs；PASS 不等于 S06
scientific PASS，不允许创建 S06 audited tag，不授权 S07 或 test/final。
