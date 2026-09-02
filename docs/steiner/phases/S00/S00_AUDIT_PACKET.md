# S00 Audit Packet

## 审计对象

- 阶段：S00 研究契约与环境冻结
- base SHA：`88ade1ac614fb12f882a10ba9b5d35b15c7b4d01`
- content head SHA：`PENDING_GATE_COMMIT`（本地 Gate PASS 后冻结；审计本文件所在 branch 的记录值）
- commit range：`88ade1ac614fb12f882a10ba9b5d35b15c7b4d01..CONTENT_HEAD_SHA`
- branch：`research/steiner-s00-contract`
- remote/PR：push 前 pending；不创建 PR、不 merge
- 主方案：`plans/STEINER_RL_BRANCHING_MIGRATION_MASTER_PLAN.md`
- 阶段计划：`docs/steiner/phases/S00/S00_PLAN.md`

`content head SHA` 将指向包含契约、配置、测试、结果文档的提交；随后仅允许一个 audit-metadata commit 把上述 placeholder 换成该 SHA。这样避免 Git commit 不能在自身内容中自引用其 SHA 的循环。

## 需求到证据映射

| S00 需求 | 实现/证据 | 测试 |
|---|---|---|
| 固定 SPG/MCF/root/`x_e` | research contract §1；ADR 0001 | action Gate test |
| 固定表示 baseline | contract §2；ADR 0002 | required baseline test |
| 固定 IL/RL 路线 | contract §3；ADR 0003 | baseline/checkpoint test |
| 固定 P0--P4/limits/seeds | `protocols_v1.yml` | protocol IDs/limits/seeds test |
| 固定 split/final guard | split/final manifests | split disjoint + final seal/hash tests |
| 固定指标/统计 | contract §6；ADR 0004；protocols | metrics/statistics test |
| 记录环境版本并决定是否升级 | environment lock | selected/conflicting stack test |
| 建立状态和阶段登记 | `STATUS.md` | required documents test |
| 不运行 final learned model | final manifest learning runs 0；无 result artifacts | final seal/hash test |
| 保留无关用户改动 | S00 plan 初始 status；显式 stage 清单 | staged diff/Git status review |

## 变更文件范围

审计应只看到：

```text
plans/STEINER_RL_BRANCHING_MIGRATION_MASTER_PLAN.md
configs/steiner/**
docs/steiner/**
tests/steiner/test_s00_contract.py
```

不应看到 `build/**`、`artifacts/environment/**`、旧 `scripts/**`、旧航空源码、checkpoint、大数据或 raw state logs。工作区中这些路径的既有改动不属于 commit range。

## 一键复现

```text
/home/duweiyue25/conda/envs/rl4scip/bin/python -m unittest -v tests/steiner/test_s00_contract.py
git diff --check 88ade1ac614fb12f882a10ba9b5d35b15c7b4d01..CONTENT_HEAD_SHA
git diff --name-only 88ade1ac614fb12f882a10ba9b5d35b15c7b4d01..CONTENT_HEAD_SHA
```

本阶段没有 solver/learning reproduction command，因为没有运行这些实验。

## Hash 索引

- final canonical selector：`8c0324c1a82485c2187825977fe2807e31512a6435e2f58f6a1d17babbfbddd1`
- environment binary/library/profile hashes：`configs/steiner/environment.lock.yml`
- 配置/测试文件 SHA-256：本地 Gate 收尾后记录在下表。

| 文件 | SHA-256 |
|---|---|
| `environment.lock.yml` | `674afe7a696fb34ed14c070b77bb8b8adac90970e164172c485c45245080ebdb` |
| `protocols_v1.yml` | `bb72bf985a915cc3c4456c6d9c3a63bfaecc4efbcda29dc9f1e2abbf425189fc` |
| `split_policy_v1.yml` | `ea8703a052f86fc5f2dea995d20b58e6ffce93c2ab64274dc8aaaeec3cc3c593` |
| `final_test_v1.yml` | `1d79aa6ada3026454729da1f6b2b915546012568be7a4dd25d123a3c6eeb9118` |
| `test_s00_contract.py` | `035586ec43e56f76258cbc7f80c7d5567b3dfe84508b4615322c6420a98e9f5c` |

## Gate 证据

- Contract tests：8 passed / 0 failed / 0 skipped。
- final membership：106 canonical entries，sealed，learning runs 0。
- Gate 阈值没有因测试结果调整。
- final test、solver experiment、training 均未运行。
- 详细 PASS/FAIL 矩阵见 `S00_RESULT_ANALYSIS.md`。

## 已知缺陷和风险

1. SCIP 8.0.4 prefix executable mode 与 library search path 当前损坏；S00 记录但不修复用户 artifact。
2. 当前 shell CUDA unavailable；训练前必须重新探测。
3. final selector 是 membership hash；数据 content hashes 需 S02 下载后补充。发现 source mismatch 必须 FAIL，不得替换实例。
4. SCIP-Jack 具体版本尚未接入；只能使用已冻结的 P3/P4角色，不能宣称已有可运行 profile。
5. 主方案最初是用户未跟踪文件；阶段提交会首次纳入，审计者应确认它与 S00 configs 一致。

## 给 GPT 审计者的问题

1. MCF/root/action 和 SCF 触发条件是否足够明确，且没有提前实现 S01/S02？
2. 以 suite/member selector SHA 冻结 final membership、S02 再补 content SHA 的两层设计是否防止样本替换？
3. P0--P4 limits、seed、baseline、主指标和统计口径是否存在可事后选择的缝隙？
4. 当前 environment runtime 失败是否被如实暴露，且是否只构成后续执行风险而非 S00 记录 Gate 的 blocker？
5. commit range 是否只包含允许路径，并完整排除既有用户改动和生成产物？

## 建议结论

本地 Gate 收尾检查通过后建议：PASS。最终结论由 GPT 只读审计者给出；S01 在 GPT 审计通过前保持 NOT_STARTED。
