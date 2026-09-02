# S01 Test Report

## 环境

- branch：`research/steiner-s01-scaffold`
- base：`a0bf0e3c1a702e1c85384f864defc86abbda29a5`
- Python：`/home/duweiyue25/conda/envs/rl4scip/bin/python` 3.11.15
- GPU：未申请、未使用

## S01 与 S00 测试

```text
/home/duweiyue25/conda/envs/rl4scip/bin/python -m pytest -q \
  tests/steiner/test_scaffold.py tests/steiner/test_s00_contract.py
```

退出码 0：14 passed，0 failed，0 skipped，33.44 s。

额外检查：

- `compileall`：PASS，退出码 0。
- clean import probe：PASS，输出 `0.1.0`；`ecole`、`pyscipopt`、`rl_branching` 均未被导入。
- `git diff -- python/rl_branching src/rl CMakeLists.txt`：空。
- 最终 staged `git diff --check`、允许路径和 1 MB 文件检查：PASS；复跑 14 passed，退出码 0。

## 旧航空 regression

```text
LD_LIBRARY_PATH="$PWD/artifacts/environment/phase4/scip804_prefix/lib" \
PYTHONPATH=python /home/duweiyue25/conda/envs/rl4scip/bin/python \
  -m pytest -q tests/python
```

退出码 1：59 passed，4 failed，0 skipped，45.49 s。失败全部保留：

1. `test_parse_z_and_grown_sets`：base `build_grown_sets` 在未给 `lower_bounds` 时返回空，与旧测试期望不一致。
2. `test_prim_variable_features`：base 六维 component ratio 与旧测试期望不一致。
3. `test_dsu_sixdim_scip_fixture_matches_cpp_extractor`：`build/graph_probe` permission denied。
4. `test_scip_tree_help_exposes_remapped_seed_triple`：`build/scip_tree` permission denied。

前两项所在源码不在 S01 diff；后两项来自阶段开始时已记录的用户 build mode 改动。它们是既有风险，不是 S01 新回归；本阶段没有修改或跳过这些测试来制造全绿结果。

## 未执行

- solver/parser/MCF tests：S01 尚无实现，留给 S02。
- final test：按封存规则 skipped by design；learning runs 保持 0。

## Hash

| 文件 | SHA-256 |
|---|---|
| `python/steiner_branching/config.py` | `cf3182d554efe2e572f4a02be966dda118d17c068363b39642ff049ba14d208a` |
| `python/steiner_branching/contracts.py` | `68e20ee57f374ce7ae37197206c25f473fb0d0f36a859bfccbe798e3ccf4dca2` |
| `python/steiner_branching/runtime.py` | `563055f556be9d54c5463ddd853e55eda945fe1898abbc4bccadf1f63a521bf9` |
| `tests/steiner/test_scaffold.py` | `9b65b414b571c2fb3cd2f000700ed98d35cbc412e00d38ff6b472572d14d3be8` |
