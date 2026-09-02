# S02 Test Report

## 环境

- branch：`research/steiner-s02-formulation`
- base：`35a90ec5e52e2fad8301e3441ff6b286c7701d04`
- Python：`/home/duweiyue25/conda/envs/rl4scip/bin/python` 3.11.15
- PySCIPOpt：4.3.0；SCIP：8.0.4（`Model.version()` 输出 `8.04`）
- profile：P0 `correctness-v1`，solver seed 0、1 thread、60 s、10,000
  nodes、4,096 MB
- GPU：未申请、未探测、未使用；S02 无训练

## 最终 Gate 测试

```text
LD_LIBRARY_PATH="$PWD/artifacts/environment/phase4/scip804_prefix/lib" \
STEINER_PACE_DEV_ROOT=/tmp/steiner-s02-gate.Ckzl19/pace \
PYTHONPATH=python /home/duweiyue25/conda/envs/rl4scip/bin/python \
  -m pytest -q tests/steiner
```

退出码 0：**52 passed，0 failed，0 skipped，2.89 s**。pytest 收集 52 个
test items，包括 S00/S01 全部契约测试。

测试范围：

- parser failure state machine、parallel edge、canonical hash/mapping；
- 五个冻结 generator families、strict config、seed split、lineage guard；
- curated toy 5/5 与随机五族小图 5/5：brute force = SCIP MCF = checker；
- binary/continuous variable type、数量、命名、metadata determinism；
- generate/build/check CLI 和 even-final download guard；
- PACE Track 1 odd `instance001` known optimum；
- final selector/content lock 的 seal、338-member count、hash、byte-only
  implementation guard 和 learning-runs=0。

无 `STEINER_PACE_DEV_ROOT` 的预期行为另行复跑：51 passed、1 skipped，退出码
0；唯一 skip 是需要显式 public-development data path 的 PACE integration。
该 skip 没有被用于最终 Gate，最终 Gate 提供数据后为 0 skipped。

`compileall`、clean import 和 boundary check 均 PASS。clean import 输出 0.1.0，
且未加载 `ecole`、`pyscipopt` 或旧 `rl_branching`。S02 base 到当前的
`python/rl_branching`、`src/rl`、`CMakeLists.txt` content diff 为空。

## Public development correctness run

- source revision：`4df73cea9c311faea7d03e6d6bffa8733c34a1aa`
- `Track1/instance001.gr`：953 bytes，SHA-256
  `76ea79c05e41de49a8c3b24c953d512155a5ff7ad40109fb77c915ec6e7efed1`
- `track1.csv`：SHA-256
  `a528be515e4d77abf3ef968989f619a62ab5eddc61f099a874c6f8527c39fdaa`
- PACE `LICENSE`：SHA-256
  `e67b2338fcfeafe14c69719a7095918d7c779230e7adbe1e464f61efba31073c`
- canonical graph：53 nodes、80 edges、4 terminals，graph SHA-256
  `799da0f03420c57b477552b7c0e2f7c5e44e828aaee8c93b5ccaaa4e0686d367`
- MCF：80 binary edge vars、480 continuous flow vars、399 constraints；
  metadata SHA-256
  `4857a76bde4313ce9470dbb23a4448faf706ce6f7d7d08c0b7621b705e2b919b`
- published objective 503；SCIP objective 503；独立 checker feasible；status
  optimal；1 node；记录到的 SCIP solving time 0.029875 s。

## CLI/manifest 证据

- 五族 synthetic manifest：5 instances，SHA-256
  `418e2812c9fe6c3600bb0bac56e1bb1025d642a8a7bc61552c814f31007e221e`；
  两个独立 output roots byte-identical。
- triangle LP：3 binary、6 continuous；problem metadata SHA-256
  `e496d50e46df6401037b2df55da378bd7a7131c3be8b87d66956100af7a1b3cb`。
- sealed final content manifest：338 members，SHA-256
  `9cb8117abb00859d5a2a0bb179f4fd03e824b731e3d5c17c9f09ef04b8f67236`；
  从 cache 原子重建两次，hash 相同。
- SteinLib archives：D `0b7c6bd8...fe0c4`、E `10808df5...47582`、
  I320 `9b45ab6d...8e1d1`、2R `e04f7d86...ce2dc`、DIW
  `7db56598...f700`；完整值在 content manifest。
- DIMACS `SPG.tgz`：SHA-256
  `bea9b2aa49bb233fc2a23c978803f64009f44b70b6728c5606432019911282a0`。

## 失败、修正与 skipped 记录

1. 首轮新增测试：33 passed、4 failed。三个失败来自测试样例误含孤立节点，
   一个来自 `high_cost.stp` 错把真实 optimum 4 写成 6；修正测试定义，未改
   Gate、模型或删样本。
2. 首次 download manifest 把 PACE instance relative path 写成 basename；修正
   为 `Track1/instance001.gr` 后重新下载，bytes/hash 不变。
3. SCIP version probe 调用了 PySCIPOpt 4.3.0 不存在的 `getMajorVersion()`，退出
   1；改用公开 `Model.version()`，得到 8.04。该失败不属于 solver run。
4. final content lock 的首次顺序下载在缓存 46 个文件后手动中断，退出 130；
   改为固定 8 worker，并从 URL-hash cache 续跑；最终 338/338，无 skip。
5. `jq` 汇总 probe 因环境未安装而退出 127；随后只用 `rg`/`sha256sum` 检查，
   不影响 manifest。
6. 无 PACE path 的 1 个预期 skip 已单独记录；最终 Gate 不含 skip。
7. 首次 staged-size shell probe 使用了 zsh 特殊变量名 `path`，导致循环内
   `git` 不可见；改用 `staged_file` 后 PASS，确认没有 staged 文件超过 1 MiB。

S01 报告中的旧航空 regression 仍有 4 个 stage 前失败。S02 未修改旧源码，
没有为了全绿去修改这些断言或用户 build permissions；本阶段不重复把它们
描述为 PASS。

## Final-test 使用声明

final bytes 仅由 `lock_final_content.py` 读取并 hash；脚本静态测试禁止出现
`pyscipopt`、`parse_pace`、`parse_steinlib` 或 `build_mcf`。没有 objective、
solution、result artifact、learning run 或参数选择；`learning_runs_total=0`。
