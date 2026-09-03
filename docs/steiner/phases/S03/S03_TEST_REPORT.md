# S03 Test Report

## 环境

- branch：`research/steiner-migration`
- base SHA：`91c30a48e6a06019d16d8b7529fe2d35bfa708fa`
- solver stack：SCIP 8.0.4、PySCIPOpt 4.3.0、Ecole 0.8.1，只经
  `scripts/steiner/run_with_scip804.sh`
- protocol：P1、solver seed 0、1 thread、600 s、200,000 nodes、8,192 MB
- initial host：24.01-core cgroup、128 GiB RAM、Gold 6148；GPU 未使用
- resume host：24.01-core cgroup、128 GiB RAM、Silver 4214；无 GPU
- formal config SHA-256：
  `cab4d8d96b02f427b8fedba6698cb9ee68b7e8a27059f8eaf319a0ede96ac1f1`

## 正式实验验收

- worker ramp：1-task/1-worker、3-task/3-worker、6-task/6-worker，10/10 shards。
- formal：90/90 valid shards；44 optimal、46 timelimit；0 missing、0 solver
  error、0 memlimit。
- aggregate 由原始 shards 独立重算后与 committed
  `S03_GATE_SUMMARY.json` 完全相等。
- final test：未读取、未解析、未求解；所有 generator seeds 均在 frozen train
  范围 `100000..199999`。

## 最终测试命令

```text
CONDA_PREFIX=/home/duweiyue25/conda/envs/rl4scip make steiner-s03-probe
scripts/steiner/run_with_scip804.sh --verify-only
bash -n scripts/steiner/run_s03_tmux.sh
scripts/steiner/run_with_scip804.sh --python -m compileall -q \
  python/steiner_branching scripts/steiner/run_s03_branchability.py
scripts/steiner/run_with_scip804.sh --python -m pytest -q \
  tests/steiner/test_s03_branchability.py \
  tests/steiner/test_pre_s03_readiness.py \
  tests/steiner/test_scip804_wrapper.py
scripts/steiner/run_with_scip804.sh --python -m pytest -q tests/steiner
```

- S03/resource/wrapper 定向测试：16 passed、0 failed、0 skipped，3.19 s。
- 完整 Steiner suite：68 passed、0 failed、1 skipped，5.72 s，退出码 0；
  唯一 skip 是未提供 `STEINER_PACE_DEV_ROOT` 的既有 public PACE integration。
- native integration 会实际构建 MCF/CIP 并断言 mapping 完整、state valid、
  strong-branch calls 不少于候选数且 strong-branch LP iterations 增加。
- C++ `-Wall -Wextra` build、wrapper verification、`bash -n`、`compileall` 和
  `git diff --check` 均 PASS。

## 失败、修正与 skipped 记录

1. 首轮 S03 unit tests：2 passed、3 failed。原因是 YAML 的裸 `off` 被解析为
   boolean；正式运行前改为 quoted string，未改阈值。
2. 首个 native smoke：SCIP parameter type assertion，return code -6；修正
   `limits/restarts` 的 setter 后通过。
3. 第二个 native smoke：mapping 0 和 SB call 0；分别是 edge-name 长度
   off-by-one 与 idempotent probe 未计真实 SB call。正式运行前修正并增加
   regression/integration assertions。
4. 初次长任务在换服务器前留下 66/90 formal；恢复前逐一验证 fingerprint，
   只补缺失 task。有效 shard 的 `SKIP` 记录保留在 raw tmux log。
5. 正式求解 46 个 timelimit，全部在 aggregate 分母中；没有为 Gate 删除。
6. 完整回归中 public PACE integration 在未提供 `STEINER_PACE_DEV_ROOT` 时有
   1 个既有预期 skip；它不是 S03 数据源或正式实验的一部分。
7. 检查结果时环境没有 `jq`，命令退出 127；改用锁定 Python 读取 JSON，未改
   数据或 Gate。

旧航空 4 个既有 regression 未在本阶段运行或修复，继续属于独立 workstream。
