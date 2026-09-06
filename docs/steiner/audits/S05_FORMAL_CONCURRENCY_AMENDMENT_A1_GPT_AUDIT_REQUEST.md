# S05 formal concurrency amendment A1 — GPT expedited audit request

请只读审计 S05 formal-v1 的 execution-only concurrency amendment，不修改仓库，
也不要重新审计或改变已经冻结的 teacher、训练超参数、数据和 Gate。

## 固定审计对象

- branch：`research/steiner-migration`
- audited base protocol content：`d1717a7ecb6043efd71678175a92325ac9ff4208`
- formal implementation / experiment run head：
  `13a8a83761f660c96cb238471de2e2c0a2d54a8b`
- amendment content head：`f302fdc42695a1d3fc82e09499b71859ec3d69b0`
- substantive range：
  `13a8a83761f660c96cb238471de2e2c0a2d54a8b..f302fdc42695a1d3fc82e09499b71859ec3d69b0`
- amendment YAML：
  `configs/steiner/experiments/s05_teacher_il_formal_v1_concurrency_a1.yml`
- amendment explanation：
  `docs/steiner/phases/S05/S05_FORMAL_CONCURRENCY_AMENDMENT_A1.md`
- amendment YAML SHA-256：
  `8ae53206bb55c30e052f489687200a0ca5cea787da7bcf8c0d49f9f9ce49451a`
- amendment explanation SHA-256：
  `b0662d3cb5c0ab5a320f3e6208a8ddef040b37063fb6e2710c30b3aa924cdd29`
- base YAML SHA-256：
  `c843f76d69c07b1c8a8093ab6f1426656084b2de2f4e9e0d777f1af32cd4c811`
- base explanation SHA-256：
  `6b121bb215ad9ad5444ce2c8328dc7049b3abdfd0a7bf191fec888441ec572c9`

## 唯一请求的变更

```text
training.max_concurrent_training_jobs: 2 -> 5
```

平台不是一台只有两张卡的共享服务器，而是可以为每个自定义作业分配一张独立
V100。A1 允许五个 preregistered seed 各运行一个独立单卡进程并在墙钟时间上重叠。
它不是 DDP，也不让任何 seed 共享 model、optimizer、RNG 或 artifact path。

每个作业仍固定为：1 张 `Tesla V100-SXM2-32GB`、8 CPU、32 GiB RAM、一个进程，
并执行既有 deterministic CUDA preflight。seeds 仍严格为
`[101,202,303,404,505]`，全部从头训练且失败不得替换或择优重跑。

## 请核对

1. machine-readable `allowed_overrides` 是否只有上述一个路径和值变化；
2. teacher tasks/data、640 states、40 epochs、optimizer、CUDA determinism、
   checkpoint selection、validation roles、statistics、Gate、failure policy、seed 202
   handoff 和 test/final prohibition 是否全部保持不变；
3. 五个 scheduler jobs 是否仍保持单 seed 语义，不构成 data/model parallel；
4. 已在 base run head 上启动的 315-task teacher 是否不受此 scheduling-only
   amendment 影响；101/202 在原 v1 上限内启动后也无需重跑；
5. amendment 当前是否保持 `execution_authorized: false`，只有 GPT `PASS` 和单独
   committed activation record 才能把全局并发上限提高到 5；
6. 是否仍要求五个作业消费同一 teacher manifest、使用同一 formal run head，并
   写入互斥的 seed artifact paths。

## 已有验证与当前边界

- complete Steiner suite：94 passed、1 expected PACE skip；
- 原始 audited v1 YAML/Markdown 未修改；
- formal teacher 正在 base run head `13a8a83...` 上运行；
- formal training 尚未开始，test/final 未访问，完整 S05 Gate 仍
  `NOT_EVALUATED`；
- amendment metadata commit 不作为五个 seed 混用不同训练代码版本的理由。

请只返回 `PASS`、`CONDITIONAL PASS` 或 `FAIL` 并列出 blocking findings。
只有 `PASS` 才允许在 teacher Gate PASS 后最多五个独立 seed 作业同时运行；该
PASS 不授权 S06，也不等于完整 S05 PASS。
