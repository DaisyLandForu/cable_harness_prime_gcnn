# S03/S04 CPU 与内存前置检查

状态：S03 `CONDITIONAL_PASS`；S04 `PASS_CPU_ONLY`

探测时间：2026-09-03 02:01:14 UTC

机器可读记录：`configs/steiner/resource_preflight_20260903.yml`

## 结论

当前环境足以开始 S03 的小并发 pilot，也足以完成 S04，但不能把“可见 48 个
逻辑 CPU”当成可用 48 核。cgroup 的 `cpu.max` 是 `801000 100000`，实际只有约
8.01 核 CPU 配额。S03 冻结计划的 6 个单线程 SCIP worker 可以放入该配额，
还剩约 2 核供调度、采集和主进程使用。

内存 cgroup 上限为 68,720,525,312 bytes（65,537 MiB，约 64.001 GiB），且没有
swap。冻结协议允许 6 workers × 8,192 MiB = 49,152 MiB，并预留 16,384 MiB；
两者合计 65,536 MiB，距离 cgroup 上限只有 1 MiB 名义余量。这个上限组合用于
判定 Gate，不能当作安全的直接启动配置，因为 RSS 瞬时峰值和非 worker 内存仍
可能触发 OOM。

因此 S03 必须按 1 → 3 → 6 workers 逐级运行。每一级记录单 worker RSS p50/p95、
`memory.current`、`memory.events` 和失败/skip。只有 p95 不超过 8,192 MiB、6-worker
预计总 RSS 不超过 49,152 MiB，且加当前不可回收占用和 16,384 MiB 预留后仍严格
小于 `memory.max`，才放行 6 workers。不能通过删除 OOM 样本或降低 Gate 放行。

S04 的范围是 19/5/1 schema、动作映射、未训练 B0、deterministic forward 和
CPU inference 计时，不训练模型。因此当前约 8 核/64 GiB 环境足够，GPU 不是
S04 Gate 的进入条件。首次 CUDA 训练前仍需重新申请并探测 GPU。

## 探测事实

| 项目 | 结果 | 解释 |
|---|---:|---|
| 可见 CPU | 48 logical / 24 physical | 不是实际调度额度 |
| cgroup CPU 配额 | 8.01 cores | 并发规划以此为准 |
| cgroup 内存上限 | 65,537 MiB | 正式资源上限 |
| 探测时 `memory.current` | 约 8.38 GiB | 其中约 7.74 GiB 是可回收 file cache |
| swap | 0 | OOM 没有交换空间缓冲 |
| `memory.events` | low/high/max/oom/oom_kill 均为 0 | 探测前没有本 cgroup 压力/OOM 事件 |
| 工作区可用空间 | 约 2.4 PiB | 足够；raw 数据仍不得进 Git |
| `/tmp` 可用空间 | 约 303 GiB | 足够当前 pilot 临时产物 |

这些是单次快照，不替代 S03 的逐实例和逐 worker 实测。每次正式 batch 前必须
重查 cgroup 限额、当前内存、`memory.events`、磁盘和负载。

## 调度规则

1. SCIP 始终 `threads=1`，不要按 48 个可见逻辑 CPU 启动 worker。
2. 第一个 batch 用 1 worker；记录 peak/p95 RSS 后再开 3 workers。
3. 3-worker 无 OOM/pressure 且投影满足契约后，才允许开 6 workers。
4. 若触发 MCF 的 8,192 MiB/p95、49,152 MiB/6-worker 或 build-time/flow-var
   阈值，按契约登记 SCF 决策；不能静默缩小样本。
5. 原始逐状态日志、LP、数据和资源 trace 写入 ignored artifact 目录，只提交
   聚合表与 manifest。
