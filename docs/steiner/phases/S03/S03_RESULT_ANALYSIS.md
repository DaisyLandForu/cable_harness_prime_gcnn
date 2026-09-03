# S03 Result Analysis

## 结论

当前 `rooted_mcf_v1` 与预注册 synthetic train 范围能够形成可学习的 edge
branching 任务，动作映射和 6-worker 内存均满足冻结要求。**本地 Gate S03：
PASS**；所有 MCF-to-SCF triggers 均为 false。S03 不包含模型训练，也不能据此
声称任何 learned policy 优于 SCIP。

## Gate 结果

| 条件 | 阈值 | 结果 | 判断 |
|---|---:|---:|---|
| 主参考实例至少 5 次合法决策 | ≥60% | 21/30 = 70% | PASS |
| 非平凡实例合法决策中位数 | ≥10 | 127（27 个实例） | PASS |
| strong-branch 有效 state | ≥60% | 17/20 = 85% | PASS |
| 有效 state 全 tie | ≤40% | 2/17 = 11.76% | PASS |
| action → 原图 edge 映射 | 100% | 2,415,538/2,415,538 = 100% | PASS |
| fresh-worker RSS p95 | ≤8,192 MB | 1,505.08 MB | PASS |
| worker ramp/正式矩阵 | 完整 | 10/10、90/90 | PASS |

正式任务状态为 44 optimal、46 timelimit、0 solver error、0 memory limit。
Timelimit 是 P1 预算结果，全部保留并参与 branchability/resource 汇总。

## MCF 资源范围

- 最大 continuous flow variables：288,204，低于 1,000,000 trigger。
- build p95：21.006 s，最大 22.053 s，低于 60 s trigger。
- worker RSS p95：1,505.08 MB，最大 1,574.02 MB，低于 8,192 MB。
- 六 worker 投影：9,030.49 MB，低于 49,152 MB。
- 两次运行前和恢复前 cgroup memory events 均为 0。

因此在本次覆盖到的 160 nodes/48 terminals 和 288,204 flow variables 范围内，
没有理由在 S03 切换 SCF。这个结论不外推到更大或更稠密的实例。

## 分族 branchability

按 `relpscost` 主参考，每族 6 个实例中达到至少 5 次决策的数量：

| family | 达标 | 观察 |
|---|---:|---|
| sparse Erdős–Rényi | 6/6 | 三个规模都有稳定分支 |
| grid with holes | 5/6 | small 有一个仅 1 次；medium/large 很强 |
| random geometric | 4/6 | small/medium 有信号；两个 large 均为 0 |
| bridge bottleneck | 4/6 | small 仅 1/2 次；medium/large 达标 |
| community block | 2/6 | medium 强；small/large 不稳定或不足 |

三个零决策实例未删除：两个 large geometric 在 root LP 上运行至 timelimit，
约 149k/154k LP iterations，未进入合法分支 callback；一个 small community
在 root node 直接最优。它们分别代表“root LP 太重”和“实例太容易”，不能用于
高效 teacher collection。

## 交给后续数据阶段的参数范围

S03 建议的首轮 branchable development/training pool 是：

- sparse：当前 small/medium/large 全范围；
- geometric：small/medium，暂不使用当前 large-high MCF 桶；
- grid：优先 medium/large；small 只作为保留诊断桶；
- community：当前 medium-mid；small/large 保留为困难/边界诊断；
- bridge：medium/large；small 保留为易例诊断。

这不是删除失败样本：全部 30 个实例仍在 S03 结果中。后续正式 split/manifest
必须显式包含 family/bucket，并避免让易例或单一 family 主导 teacher states。

## Strong-branch 信号

预注册 10 个 small/relpscost task，每个最多 2 个 state，期望 20。实际观察
17 个且 17 个全部有效；缺少的 3 个来自一个 root-solved community task（缺 2）
和一个只有一次分支的 grid task（缺 1），按计划计入分母。两个 all-tie state
均来自同一个 bridge small 易例，其余 15 个 state 有 score 差异。

Native integration test 同时要求 `sb_calls_delta >= evaluated` 和
`sb_lp_iterations_delta > 0`，因此本结论不是 Ecole 0.8.1 常数 score 的替代统计。

## 不能推出的结论与风险

- 46/90 timelimit 说明 P1 很难，但 P1 关闭 presolve/cuts/heuristics，不能当作
  P2 production performance，也不能用它比较 formulation 优劣。
- 规模和 terminal ratio 在三个桶中同步增加，不能分离二者的独立因果效应。
- 只使用 synthetic train、solver seed 0 和 20 个预期 strong states；S05 仍需
  独立 teacher-quality/learning-curve Gate。
- 结果跨两个相同 cgroup 预算但不同 CPU 型号的 host；不得用 wall/build time
  做基线速度排名。固定资源阈值余量很大，因此不改变本阶段安全结论。
- S00--S03 GPT audit 均为 NOT_RUN；本地 PASS 不是独立审计 PASS。
- SCIP 8.0.4 wrapper、公共数据再分发许可风险和旧航空 4 个独立失败仍存在。
