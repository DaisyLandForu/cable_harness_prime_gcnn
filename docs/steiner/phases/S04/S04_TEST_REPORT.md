# S04 Test Report

## 环境与冻结输入

- branch：`research/steiner-migration`
- base SHA：`931c7ae05c299c54bbdf59ecd458b64c7ca42282`
- stack：SCIP 8.0.4、PySCIPOpt 4.3.0、Ecole 0.8.1、PyTorch 2.5.1+cu121
- host：Intel Xeon Silver 4214；24.01-core cgroup、128 GiB RAM；GPU 不可见且
  S04 不需要 GPU
- protocol/split/seeds：P1/train；solver 0、model 404、generator 100300
- final test access：0；learning runs：0

## 最终验证

```text
scripts/steiner/run_with_scip804.sh --verify-only
bash -n scripts/steiner/run_with_scip804.sh
scripts/steiner/run_with_scip804.sh --python -m py_compile \
  scripts/steiner/run_s04_b0_snapshot.py \
  python/steiner_branching/models/milp_gcnn.py \
  python/steiner_branching/solver/bipartite_observation.py \
  python/steiner_branching/solver/graph_state.py
scripts/steiner/run_with_scip804.sh --python -m pytest -q tests/steiner
scripts/steiner/run_with_scip804.sh --python \
  scripts/steiner/run_s04_b0_snapshot.py --snapshot-output <tmp>/snapshot.json \
  --summary-output <tmp>/summary.json
cmp docs/steiner/phases/S04/S04_FORWARD_SNAPSHOT.json <tmp>/snapshot.json
git diff --check
```

- wrapper/version verification：PASS，退出码 0。
- shell/Python syntax checks：PASS，退出码 0。
- final S04 定向测试：7 passed、0 failed、0 skipped，26.04 s。
- 完整 Steiner suite：代码收尾时 75 passed、1 skipped，31.05 s；文档/契约
  regression 更新后最终复跑 75 passed、0 failed、1 skipped，29.83 s，退出码 0。
- 唯一 skip：未设置 `STEINER_PACE_DEV_ROOT` 的既有 PACE odd development
  integration；S04 不依赖该下载数据。wrapper 内的 S04 real-SCIP test 没有 skip。
- 正式 snapshot 二次生成后 byte-for-byte 相同；两份文件 SHA-256 均为
  `ac2ce0c14b134245221af5140a3008f3ec6067f8867491e7cc0d0b50e2036f2c`。
- `git diff --check`：PASS。

## 失败、诊断与修正

1. 首次 inline API probe 因未通过 wrapper/`PYTHONPATH` 进入，报
   `ModuleNotFoundError: steiner_branching`；改用 canonical wrapper 后继续。
2. 首轮真实 S04 targeted suite：5 passed、1 failed，34.05 s。真实 Ecole state
   有 1,962 个 NaN，精确落在 981 variables 的 incumbent/average-incumbent 两列。
   S04 增加只允许这两列的 versioned zero sentinel，并为其他 NaN/Inf 增加失败
   测试；同时把 immutable NumPy→Torch conversion 改为复制，消除只读警告。
3. sentinel 修复后的 mean prototype：7 passed，26.08 s。复核标准 baseline 后在
   正式冻结前改为 sum aggregation；最终配置再次 7 passed，26.04 s。
4. 收尾时误用不存在的 `scripts/steiner/scip804.sh`，shell 退出 127；随后改用
   已固化的 `run_with_scip804.sh`。另一次把 Python runner 误交给 `bash -n`，该项
   报 Python 第 16 行 shell syntax error；正确的 shell/Python syntax 命令均通过。

以上失败都发生在正式 Gate 判定前，未删除记录、未改变 split/seed/阈值，也未
访问 final test。

## 产物与 Gate

- deterministic snapshot file SHA-256：
  `ac2ce0c14b134245221af5140a3008f3ec6067f8867491e7cc0d0b50e2036f2c`
- deterministic snapshot canonical SHA-256：
  `d7ed96707bb625b64c4251840b348e78cae518d1ead1018c80e540ac04b3d536`
- Gate summary file SHA-256：
  `f687460a8e6bbae9b0159cc1a851bff97cc3814ef9f800e848dbc6d01d08c633`
- schema SHA-256：
  `f47a74b08a3ab88f07733f17b8932a2bf6878ed271321938c121845f44d037fc`
- model-state SHA-256：
  `42b63f31ccbb0b5416db53d38e9ca0072daedbfba3d84d30094a9d443ac3795c`

Gate checks 7/7 true；本地 Gate **PASS**。旧航空全仓测试未运行，因为已登记的
4 个失败属于独立 backlog，S04 没有修改旧航空路径。
