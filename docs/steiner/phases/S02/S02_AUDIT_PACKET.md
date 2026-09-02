# S02 Audit Packet

## 审计对象

- base SHA：`35a90ec5e52e2fad8301e3441ff6b286c7701d04`
- content head SHA：`19c7f46b91a1d05c46dbdeeba00bf863b37a7f5a`
- substantive range：`35a90ec5e52e2fad8301e3441ff6b286c7701d04..19c7f46b91a1d05c46dbdeeba00bf863b37a7f5a`
- branch：`research/steiner-s02-formulation`
- prior GPT audit：S00 NOT_RUN；S01 NOT_RUN
- remote/PR：未执行远端写入；未创建 PR、未 merge

## 需求映射

| 需求 | 代码/配置 | 测试/证据 |
|---|---|---|
| strict `.gr/.stp` | `data/{pace,steinlib,load}.py` | parser negative matrix |
| canonical/parallel/hash | `data/canonical.py` | determinism/property tests |
| five generators/split | `data/{generate,split,manifest,write}.py` | family/config/lineage tests |
| governed download/license | `data/download.py` + CLI | even guard + PACE hash |
| rooted MCF/metadata | `milp/{mcf,naming}.py` | type/count/name/hash tests |
| independent validation | `milp/validate.py` | toy/random/PACE cross-check |
| operational CLIs | `scripts/steiner/*.py` | subprocess integration tests |
| final content lock | checksum JSON + byte-only CLI | 338 count/hash/import guard |

## 允许变更范围

```text
.gitignore
python/steiner_branching/data/**
python/steiner_branching/milp/**
configs/steiner/data/**
configs/steiner/splits/final_test_v1.yml
configs/steiner/splits/final_test_content_v1.json
scripts/steiner/**
tests/steiner/**
docs/steiner/STATUS.md
docs/steiner/phases/S02/**
```

不允许 stage `artifacts/**`、`build/**`、`results/audit/**`、legacy aviation
scripts 的用户 mode 变化、下载 raw data、生成 LP/result 或其他无关用户文件。

## 核心复现

```text
LD_LIBRARY_PATH="$PWD/artifacts/environment/phase4/scip804_prefix/lib" \
STEINER_PACE_DEV_ROOT=/tmp/steiner-s02-gate.Ckzl19/pace \
PYTHONPATH=python /home/duweiyue25/conda/envs/rl4scip/bin/python \
  -m pytest -q tests/steiner

LD_LIBRARY_PATH="$PWD/artifacts/environment/phase4/scip804_prefix/lib" \
/home/duweiyue25/conda/envs/rl4scip/bin/python \
  scripts/steiner/check_solution.py \
  /tmp/steiner-s02-gate.Ckzl19/pace/Track1/instance001.gr \
  --known-objective 503

# 使用新的空 cache 目录重新核对远端 source bytes；不会解析或求解。
/home/duweiyue25/conda/envs/rl4scip/bin/python \
  scripts/steiner/lock_final_content.py \
  --output /tmp/final_test_content_v1.json \
  --cache-dir /tmp/new-empty-steiner-final-cache
```

## Gate 证据与审计重点

- 最终 tests：52 passed、0 failed、0 skipped。
- public dev objective：PACE odd 503 = SCIP 503，checker feasible。
- final content lock：338/338，manifest SHA-256
  `9cb8117abb00859d5a2a0bb179f4fd03e824b731e3d5c17c9f09ef04b8f67236`。
- final operation：`byte_hash_only_no_parse_no_solve`；learning runs 0。
- 所有早期失败和 skip 均在 test report 保留。
- staged diff/path/size/final test 已在 content commit 前 PASS；无 staged 文件
  超过 1 MiB。

本审计包之后只允许 metadata commit 更新 `STATUS.md`、本文件、结果分析和命令
记录；审计者应把 content head 作为 substantive diff 终点。

## 给 GPT 审计者的问题

1. parser state machine 是否还存在接受 variant、trailing content 或计数覆盖的
   silent fallback？
2. canonical ordering/hash 是否在平行边、重编号和相同权重情况下稳定？
3. MCF 的每 commodity flow balance 与 `f_uv + f_vu <= x_e` 是否正确，且
   continuous flow 未被误标为未来 action？
4. checker 是否真正独立于 SCIP constraints，并正确处理 unknown/duplicate/
   disconnected/objective mismatch？
5. final content-lock CLI 是否只做 bytes/tar member hashing，未泄露 final
   structure/objective 到实现或结果选择？
6. 338-member manifest、archive/member SHA 和 source notice 是否足以满足 S00
   两层封存；SteinLib/DIMACS license 风险是否需要在发布前升级为 blocker？
7. staged range 是否只含允许路径，且没有吸收用户 dirty artifacts/mode changes？

## 建议结论

最终 staged check 通过后建议 PASS；S02 只证明 formulation/data correctness，
不应提前给出 branchability、性能或 learned-policy 主张。最终结论由只读审计者
决定。
