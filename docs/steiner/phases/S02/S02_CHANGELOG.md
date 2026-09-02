# S02 Changelog

## 完成内容

- 实现 PACE `.gr` 与经典 SteinLib `.stp` 的严格 SPG parser；unknown
  section、directed arc、root/prize variant、重复 declaration、计数错误、
  非正权、self-loop、不连通、缺失/重复 EOF 均显式失败。
- 实现 canonical node/edge/terminal mapping、平行边稳定 ID、graph SHA-256
  和 `.gr/.stp` 严格 extension dispatch。
- 实现五个冻结合成图族、严格 YAML 配置、seed-to-split、dataset manifest、
  lineage leakage guard 和确定性 `.stp` writer。
- 实现 odd-development-only PACE downloader；even selector 在网络访问前拒绝，
  下载记录包含固定 revision、相对路径、bytes 和 SHA-256。
- 实现 `rooted_mcf_v1`：每边一个 binary `x`、每 commodity/无向边两个
  continuous flow、flow balance、双向合计 link constraint、稳定变量命名和
  `ProblemMetadata`。
- 实现独立 selected-edge checker、小图穷举器和 SCIP solve/check bridge。
- 新增 download/generate/build/check 四个常规 CLI，以及只读取 bytes、绝不
  导入 parser/solver 的 final content-lock CLI。
- 为 sealed final selectors 锁定 338 个实例 bytes：PACE even 100、SteinLib
  5 个 archive/188 个成员、DIMACS archive/50 个成员；raw cache 不入 Git，
  checksum manifest 入 Git，learning runs 保持 0。
- 增加大数据、生成数据、solver artifacts、checkpoint 和 raw logs 的精确
  `.gitignore` 规则。

## 接口变化

- data：`load_graph`、`parse_pace`、`parse_steinlib`、`canonicalize_raw`、
  `generate_graph`、`DatasetManifest`、`split_for_synthetic_seed`。
- MILP：`build_mcf`、`configure_p0`、`check_selected_edges`、
  `brute_force_optimum`、`solve_and_validate`。
- 变量契约：`stp_x_e########` 和 `stp_f_t####_a########`；continuous flow
  不属于未来 branching action。
- 配置：`configs/steiner/data/synthetic_v1.yml` 和
  `configs/steiner/splits/final_test_content_v1.json`。

## 数据与迁移

- public development：只运行 PACE 2018 Track 1 odd `instance001.gr`。
- sealed final：只做 byte/archive/member hashing；未导入 parser、未建模、
  未求解、未读取 objective、未产生 result artifact。
- 所有下载、生成 `.stp`、LP、metadata、solver result 和 cache 均在 `/tmp`；
  Git 只保留 toy fixtures 和聚合 checksum/config manifest。

## 与计划/契约的偏差

- 初版 S02 plan 写成“不下载 final selectors”，与 S00 已冻结的
  `content_lock_policy` 冲突。收尾审查时按上位契约纠正为“只下载 bytes 并
  hash，禁止解析/求解”；修改发生在任何 learned run 前，未改变 selector、
  Gate 或实验结果。
- S00/S01 GPT 审计仍为 NOT_RUN；用户明确要求继续完成 S01--S02，未把缺失
  审计改写为 PASS。

## 未完成

- branchability、candidate coverage、SCF resource trigger 属于 S03，未开始。
- observation、teacher、GCNN、IL、RL 和 GPU training 均未开始。
- SteinLib/DIMACS 来源页没有提供可确认的显式 redistribution license；raw
  data 不提交，后续公开分发前必须单独完成法务/许可证核实。
