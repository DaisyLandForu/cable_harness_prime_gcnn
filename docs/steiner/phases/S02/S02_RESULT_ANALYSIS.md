# S02 Result Analysis

## 结论

S02 已证明当前 classic undirected positive-cost SPG 输入能稳定转换为
`rooted_mcf_v1`，并在 curated toy、五类随机小图和一个预注册 PACE odd
public-development instance 上同时通过三重检查：穷举/公开 optimum、SCIP
objective、独立 selected-edge connectivity/objective checker。

这足以支持进入“候选是否形成有意义 branching task”的下一阶段审计，但本次
请求止于 S02，未开始 S03。

## 关键结果解释

- curated toy 5/5、随机 family 5/5 均完全一致，说明 flow balance、linking、
  edge cost 和 checker 的基础语义没有出现已知反例。
- PACE `instance001` 的 503 = 503 且 checker feasible，给出了不同于 toy
  generator 的外部 known-optimum 证据。
- canonical graph hash、变量映射、problem metadata、synthetic manifest 和
  final content manifest 均可重复，支持后续 state/action lineage 审计。
- PACE 样例在 presolve/default P0 下 1 node 即证最优。这证明 correctness，
  **不证明**存在足够 branch decisions；branchability 必须由 S03 的 P1 pilot
  和预注册阈值判断。
- 本样例 MCF 仅 560 vars/399 constraints，不能推出大实例的 memory/build
  time 满足 SCF trigger；S03 才测 p95 build/RSS 和 million-flow-var 条件。

## Gate S02

| 条件 | 证据 | 结论 |
|---|---|---|
| curated toy 100% | 5/5 brute force = MCF = checker | 满足 |
| 随机小图交叉验证 | 5 families 全部一致 | 满足 |
| public known optimum | PACE odd instance001：503 = 503 | 满足 |
| parser 无静默降级 | unknown/variant/count/EOF/graph validity failure tests | 满足 |
| hash/变量映射可复现 | determinism、manifest、metadata tests | 满足 |
| checker 独立验证 SCIP | toy、random、PACE 均 feasible/objective match | 满足 |
| split/lineage | frozen seed ranges + crossing guard | 满足 |
| final content 双层封存 | selector 106 entries；content 338 members；learning runs 0 | 满足 |
| S00/S01 invariant | 同批 tests 全部通过 | 满足 |
| 无旧栈/大产物污染 | boundary/staged path/size checks | 满足 |

最终 staged checks 已通过。**本地 Gate S02：PASS**。本次请求止于 S02，
不会开始 S03。

## 不能推出的结论

- 不能推出 MCF 对大型实例资源可接受或优于 SCF。
- 不能推出 SCIP 会产生足够多的合法 fractional binary `x_e` candidates。
- 不能推出 GCNN observation、teacher label、IL 或 RL 已可用。
- 不能推出任何 learned policy 改善 solver；本阶段没有模型和 GPU training。
- 不能把一个 PACE odd 样例的求解时间当作 benchmark 性能结论。
- final content lock 不是 final evaluation；任何利用其结构、objective 或求解
  结果做选择都仍被禁止。

## 已知风险

1. parser 有意只支持经典 undirected SPG；variant 会失败，需 S10 新契约。
2. S03 可能因 branchability 或资源阈值失败，从而阻止后续学习阶段。
3. SCIP 8.0.4 依赖显式 `LD_LIBRARY_PATH`，默认 SCIP 9.2.2 禁止混入。
4. SteinLib/DIMACS 来源页未提供可确认的显式 redistribution license；当前只
   提交 hash，不提交 raw 数据，公开发布前仍需许可证审查。
5. S00/S01/S02 GPT 审计均未运行；本地 PASS 不能替代独立审计。
6. 旧航空 regression 的 4 个既有失败仍是仓库级风险，但不在 S02 diff。
