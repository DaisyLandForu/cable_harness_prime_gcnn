# S01 Changelog

## 完成内容

- 建立独立 `python/steiner_branching` package 及 data/milp/solver/models/learning/evaluation 子包。
- 新增 immutable `SteinerGraph`、`ProblemMetadata`、`GraphSchema`、`RunManifest` 及其基础映射类型和 invariant。
- 新增严格 YAML loader；unknown、missing、schema mismatch 均显式失败。
- 新增统一 UTC logging、Python/NumPy/Torch seed 和安全 artifact layout。
- 新增 S01 smoke config、CLI 目录说明和 6 个 scaffold tests。
- 保持 `python/rl_branching`、C++ 和航空配置零改动。

## 接口变化

- package version：`steiner_branching 0.1.0`。
- 公共入口：`ScaffoldConfig`、`SteinerEdge`、`SteinerGraph`、`EdgeVariableMetadata`、`ProblemMetadata`、`GraphSchema`、`RunManifest`、`ArtifactLayout`、`configure_logging`、`seed_everything`。
- `configs/steiner/scaffold_smoke.yml` 是 S01 最小严格配置。

## 数据与迁移

无数据迁移、无 benchmark 下载、无 solver run、无 checkpoint。S01 不创建持久 artifact；测试只使用 pytest 临时目录。

## 与主方案的偏差

- S00 GPT audit 尚未执行。用户明确要求继续 S01--S02，因此从 S00 本地/远端 PASS commit 开始；审计状态仍记录为 NOT_RUN。
- 旧航空全量 regression 存在 4 个 stage 前已有失败，未为使 S01 Gate 通过而修改旧实现或测试。

## 未完成

- parser、canonicalization、generator、manifest/split 实现属于 S02。
- MCF、solution checker、known-optimum cross-check 属于 S02。
- observation/model/teacher/learning 未开始。
