# S03 Audit Packet

## 审计对象

- base SHA：`91c30a48e6a06019d16d8b7529fe2d35bfa708fa`
- content head SHA：`495d699cceefd243d4ab4c510be051f9df94833a`
- phase head SHA：本 metadata commit 的 annotated local tag target；精确 SHA 由最终 handoff 报告
- substantive range：`91c30a48e6a06019d16d8b7529fe2d35bfa708fa..495d699cceefd243d4ab4c510be051f9df94833a`
- branch：`research/steiner-migration`
- planned local tag：`steiner-s03-local-gate-v1`
- prior/current GPT audit：S00--S03 均为 NOT_RUN
- remote/PR：content commit 后只允许 fast-forward push；不 merge、不创建阶段分支

## 需求映射

| 需求 | 代码/配置 | 测试/证据 |
|---|---|---|
| 五族/规模/terminal pilot | S03 YAML + task expansion | 30 instances / 90 tasks tests |
| P1/seed/split 锁定 | strict loader + wrapper | config negative tests、all train seeds |
| 合法候选与 edge mapping | Python observer | transformed-name/mapping tests + 2,415,538 candidates |
| native strong branch | C++ probe | real SB calls/LP iterations integration test |
| fresh worker RSS | subprocess-per-task runner | 90 resource records、p95 summary |
| 1→3→6 与断点续跑 | runner/tmux/fingerprint | 10 ramp shards、换机恢复记录 |
| MCF/SCF trigger | deterministic aggregator | machine-readable trigger booleans |
| 失败/timeout 不丢弃 | atomic shards + aggregate | 46 timelimit、3 missing expected SB states |

## 允许变更范围

```text
Makefile
plans/STEINER_RL_BRANCHING_MIGRATION_MASTER_PLAN.md
configs/steiner/experiments/s03_branchability_pilot_v1.yml
configs/steiner/resource_preflight*.yml
python/steiner_branching/solver/**
scripts/steiner/run_s03_*.{py,sh}
tools/steiner_s03_sb_probe.cpp
tests/steiner/test_s03_branchability.py
tests/steiner/test_pre_s03_readiness.py
docs/steiner/STATUS.md
docs/steiner/phases/S03/**
configs/steiner/git_governance_v1.yml  # metadata only
```

明确禁止 stage 现有 `artifacts/**`、`build/**`、`results/audit/**`、旧航空
scripts、用户的 `docs/figures/**`、未跟踪实验 config 或 raw S03 shards/logs。

## 核心复现

```text
CONDA_PREFIX=/home/duweiyue25/conda/envs/rl4scip make steiner-s03-probe
scripts/steiner/run_with_scip804.sh --verify-only
scripts/steiner/run_with_scip804.sh --python -m pytest -q tests/steiner

# 可中断/恢复；完整运行很长
scripts/steiner/run_s03_tmux.sh steiner-s03 6
```

聚合 summary 的 config SHA-256 为
`cab4d8d96b02f427b8fedba6698cb9ee68b7e8a27059f8eaf319a0ede96ac1f1`；
raw evidence 位于 ignored
`results/steiner/raw/s03/s03-branchability-pilot-v1/`，不进入 Git。

## Gate 证据与审计重点

- 90/90 formal、10/10 ramp；44 optimal、46 timelimit、0 missing/error/memlimit。
- branchable fraction 70%；nontrivial median 127。
- strong valid 85%；valid all-tie 11.76%；缺失的 3/20 保留在分母。
- edge mapping 2,415,538/2,415,538，callback errors 0。
- RSS p95 1,505.08 MB；build p95 21.006 s；max flows 288,204；SCF trigger
  全 false。
- 两个 CPU host 的结果只用于固定 Gate，不用于 wall-time baseline 排名。
- final test 未访问，training runs 0；S04 未开始。

## 给 GPT 审计者的问题

1. observer 返回 `DIDNOTRUN`、priority candidate slice 和 node de-dup 是否真正
   不改变 default/relpscost/mostinf 的动作语义？
2. transformed name normalization 是否可能把非原始 edge 误映射为合法 action？
3. native probe 的 valid/all-tie 定义、真实 SB call assertion 和 missing-state
   分母是否保守且与 S00 契约一致？
4. fresh-process `ru_maxrss`、nearest-rank p95 和 six-worker projection 是否足以
   支持当前 MCF 范围，跨 host 是否需要额外限制？
5. 30-instance 对角 bucket 设计能否支持本报告的参数范围，但不被误读为独立
   node-count/terminal-ratio 因果结论？
6. 46 个 timeout、3 个零决策实例、3 个 missing expected SB states 是否均被
   正确保留，没有 success-only filtering？
7. staged diff 是否排除了所有既有航空/build/artifact/用户改动和 raw data？

## 建议结论

在最终 staged boundary/size/test check 和 SHA 回填通过后，建议本地 S03 PASS；
只允许下一阶段使用本文建议的 branchable development range。GPT 独立结论仍为
NOT_RUN，不能创建 audited tag，也不能把 S03 解释为 learned-policy 性能证据。
