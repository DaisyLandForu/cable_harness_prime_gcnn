# S03/S04 CPU 与内存前置检查

状态：S03 `PASS_WITH_REQUIRED_RAMP`；S04 `PASS_CPU_ONLY`

最新探测时间：2026-09-03 06:13:54 UTC

机器可读记录：`configs/steiner/resource_preflight_20260903.yml`、
`configs/steiner/resource_preflight_s03_resume_20260903.yml`

## 结论

正式运行环境和换机恢复环境都足以完成 S03/S04，但不能把可见逻辑 CPU 当成
实际调度配额。两次机器可读前检的 cgroup `cpu.max` 都是
`2401000 100000`，即约 24.01 核。S03 冻结计划的 6 个单线程 SCIP worker 有
充分 CPU 余量。

内存 cgroup 上限为 137,440,002,048 bytes（131,073 MiB，约 128 GiB），且没有
swap。冻结协议允许 6 workers × 8,192 MiB = 49,152 MiB，并预留 16,384 MiB；
两者合计 65,536 MiB，低于 cgroup 上限。较早的 8.01 核/65,537 MiB 观察属于
换机前历史快照，不再代表 S03 正式运行资源。

S03 仍按预注册的 1 → 3 → 6 workers 逐级运行。每一级记录单 worker RSS p50/p95、
`memory.current`、`memory.events` 和失败/skip。只有 p95 不超过 8,192 MiB、6-worker
预计总 RSS 不超过 49,152 MiB，且加当前不可回收占用和 16,384 MiB 预留后仍严格
小于 `memory.max`，才放行 6 workers。不能通过删除 OOM 样本或降低 Gate 放行。

S04 的范围是 19/5/1 schema、动作映射、未训练 B0、deterministic forward 和
CPU inference 计时，不训练模型。因此 24.01 核/128 GiB 环境足够，GPU 不是
S04 Gate 的进入条件。恢复环境没有 GPU；首次 CUDA 训练前仍需重新申请并探测。

## 探测事实

| 项目 | 结果 | 解释 |
|---|---:|---|
| 正式/恢复可见 CPU | 80/48 logical | 不是实际调度额度 |
| cgroup CPU 配额 | 24.01 cores | 两次前检一致 |
| cgroup 内存上限 | 131,073 MiB | 两次前检一致 |
| `memory.current` | 正式约 18.3 GiB；恢复约 2.46 GiB | 点时值，不代替 worker RSS |
| swap | 0 | OOM 没有交换空间缓冲 |
| `memory.events` | low/high/max/oom/oom_kill 均为 0 | 探测前没有本 cgroup 压力/OOM 事件 |
| 工作区可用空间 | 正式前检约 2.4 PiB | 足够；raw 数据仍不得进 Git |
| `/tmp` 可用空间 | 正式前检约 707 GiB | 足够当前 pilot 临时产物 |

这些是单次快照，不替代 S03 的逐实例和逐 worker 实测。每次正式 batch 前必须
重查 cgroup 限额、当前内存、`memory.events`、磁盘和负载。

## 调度规则

1. SCIP 始终 `threads=1`，不要按可见逻辑 CPU 数启动 worker。
2. 第一个 batch 用 1 worker；记录 peak/p95 RSS 后再开 3 workers。
3. 3-worker 无 OOM/pressure 且投影满足契约后，才允许开 6 workers。
4. 若触发 MCF 的 8,192 MiB/p95、49,152 MiB/6-worker 或 build-time/flow-var
   阈值，按契约登记 SCF 决策；不能静默缩小样本。
5. 原始逐状态日志、LP、数据和资源 trace 写入 ignored artifact 目录，只提交
   聚合表与 manifest。
