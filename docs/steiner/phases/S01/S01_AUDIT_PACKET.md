# S01 Audit Packet

## 审计对象

- base SHA：`a0bf0e3c1a702e1c85384f864defc86abbda29a5`
- content head SHA：`05b42791226347d31647547c344ef46c9dc4e87d`
- substantive range：`a0bf0e3c1a702e1c85384f864defc86abbda29a5..05b42791226347d31647547c344ef46c9dc4e87d`
- branch：`research/steiner-s01-scaffold`
- S00 audit：NOT_RUN（用户明确要求继续）
- remote/PR：未执行远端写入；未创建 PR、未 merge

## 需求映射

| 需求 | 代码 | 测试 |
|---|---|---|
| 独立包和子包 | `python/steiner_branching/**` | clean import |
| typed dataclass | `contracts.py` | contract round-trip/invariants |
| strict config | `config.py` + smoke YAML | unknown/missing/version tests |
| logging/seed/artifacts | `runtime.py` | deterministic/path/idempotence tests |
| 旧栈不变 | 无旧路径 diff | boundary check + old regression report |

## 允许变更范围

```text
python/steiner_branching/**
configs/steiner/scaffold_smoke.yml
scripts/steiner/README.md
tests/steiner/conftest.py
tests/steiner/test_scaffold.py
docs/steiner/STATUS.md
docs/steiner/phases/S01/**
```

## 复现

```text
/home/duweiyue25/conda/envs/rl4scip/bin/python -m pytest -q \
  tests/steiner/test_scaffold.py tests/steiner/test_s00_contract.py
PYTHONPATH=python /home/duweiyue25/conda/envs/rl4scip/bin/python -c \
  'import steiner_branching; print(steiner_branching.__version__)'
git diff --check a0bf0e3c1a702e1c85384f864defc86abbda29a5..05b42791226347d31647547c344ef46c9dc4e87d
```

本审计包之后只允许 metadata commit 更新上述 SHA；审计者应同时确认 `05b42791226347d31647547c344ef46c9dc4e87d..research/steiner-s01-scaffold` 仅含 `STATUS.md`、本文件和命令记录。

## Gate 证据与风险

- 专属/S00 tests：14 passed。
- 旧 regression：59 passed、4 个既有失败，详见 test report。
- final test 未运行；没有 data/checkpoint/build/raw logs。
- 风险：S00 缺 GPT audit；旧 Prim tests 与 build permissions 未修复。

## 给审计者的问题

1. contracts 是否只冻结 S01 必需 invariant，没有提前隐藏实现 S02？
2. strict loader/path validation 是否存在 traversal 或 silent fallback？
3. 4 个旧 regression 失败是否已正确归因且没有被 S01 diff 掩盖？
4. diff 是否完全隔离旧航空栈和用户改动？

## 建议结论

最终 staged check 通过后建议 PASS；最终结论由只读审计者决定。
