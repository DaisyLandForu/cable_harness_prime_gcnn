# S03 Changelog

## 完成内容

- 冻结 `s03-branchability-pilot-v1`：仅使用 synthetic train seeds
  `100300..100329`，覆盖五个图族、三个规模/terminal 桶、两个 replicate 和
  `scip_default`/`relpscost`/`mostinf` 三个基线，共 90 个正式任务。
- 新增不干预求解的 PySCIPOpt branchrule observer。它只读取 priority LP
  candidates、按 SCIP node number 去重并返回 `DIDNOTRUN`，让目标基线继续负责
  实际分支。
- 对每个候选规范化 transformed `t_` 前缀，并与 S02 的
  `stp_x_e########` metadata 做全量映射；continuous flow 变量不进入动作集合。
- 新增 SCIP 8.0.4 C++ 原生 strong-branch 探针。它调用
  `SCIPgetVarStrongbranchFrac`，要求真实 strong-branch call、完整候选映射、
  有效上下分支和有限 score，避免使用已知可能退化的 Ecole score 路径。
- 新增 fresh-process、atomic JSON shard、config/task fingerprint、自动断点续跑、
  1 → 3 → 6 worker 放量、nearest-rank 汇总和机器可读 Gate 判定。
- 新增 tmux 启动脚本；长任务中断或换机后重复同一命令只复用指纹一致的 shard。
- 记录两次 24.01-core/128-GiB 资源验收。恢复服务器无 GPU；S03 全程 CPU-only。
- 完成 90/90 formal 和 10/10 ramp；保留 44 optimal、46 timelimit、3 个主参考
  零决策实例和 3 个未取得的预期 strong states。

## 接口与构建变化

- Python：`steiner_branching.solver.branchability` 提供 strict config、task
  expansion、candidate observer、native output parser、fresh-worker execution、
  resumable shard 和 Gate aggregation。
- CLI：`scripts/steiner/run_s03_branchability.py`；tmux：
  `scripts/steiner/run_s03_tmux.sh`。
- Native：`tools/steiner_s03_sb_probe.cpp`；构建目标：
  `make steiner-s03-probe`，二进制只生成到 ignored `build/`。
- 配置：`configs/steiner/experiments/s03_branchability_pilot_v1.yml` 和两份
  resource preflight。

## 正式运行前的失败与修正

1. 首轮新增测试因 YAML 1.1 把未加引号的 `off` 解析为 boolean，得到
   2 passed、3 failed；只给配置值加引号，没有改 Gate。
2. 首个 native smoke 把 SCIP 8.0.4 的 `limits/restarts` 当 Longint 设置，触发
   parameter type assertion、return code -6；改为 `SCIPsetIntParam`。
3. 第二个 native smoke 暴露 canonical edge name 长度 off-by-one，且
   `idempotent=true` 时 strong-branch call 统计为 0。正式运行前修正名字校验，
   将预注册值冻结为 `idempotent=false`，并把真实 SB call 增加设为有效条件。
4. 上述 smoke 全在 `/tmp/steiner-s03-smoke*`，未进入正式 shard 或 Gate。

## 执行偏差与恢复

- 初次 tmux 从 2026-09-03 04:18 UTC 运行；换服务器后先验证 66/90 formal 和
  10/10 ramp shard 的 config/task hash，再只补 24 个缺失任务。最终 100 个
  shard 全部匹配 config SHA-256
  `cab4d8d96b02f427b8fedba6698cb9ee68b7e8a27059f8eaf319a0ede96ac1f1`。
- 结果跨 Intel Gold 6148 与 Silver 4214 两个 CPU host。S03 不用 wall time
  排名分支策略；资源值只与宽松的固定安全阈值比较。该限制必须随结果保留。
- 初版 plan 中“配置必须先 commit”与“只有 Gate PASS 才 commit”冲突；实际在
  正式运行前以 canonical SHA-256 冻结，在本地 Gate PASS 后才提交。没有改动
  split、seed、实例、profile 或 Gate 数值。

## 未完成

- 未实现 S04 observation/B0，也未采 S05 teacher dataset 或训练模型。
- 未运行 sealed final test；learning runs 仍为 0。
- 未修改旧航空源码、测试、build 或运行脚本；其 4 个既有失败仍在独立 backlog。
