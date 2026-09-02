# S02 数据解析、生成器与 MCF correctness 计划

## 阶段目标

从严格解析的 PACE `.gr`、经典 SteinLib `.stp` 和确定性合成图生成 canonical `SteinerGraph`，构建 `rooted_mcf_v1` PySCIPOpt 模型，并用独立 solution checker、小图穷举和 PACE odd known optimum 交叉验证。

## 非目标

- 不解析、求解或运行 PACE even、SteinLib final families 或 DIMACS final
  bundle；依照 S00 上位契约，只允许下载原始 bytes，记录 archive/member
  SHA-256 后丢弃本地 raw cache，且不得把内容用于任何实现选择或调参。
- 不执行 branchability pilot、strong branching、observation、GCNN 或训练。
- 不实现 SCF；只有 S03 资源触发后才允许。
- 不修改旧航空栈或 SCIP/C++ artifacts。

## 输入、基线与环境

- 起始/工作 base：S01 metadata head `35a90ec5e52e2fad8301e3441ff6b286c7701d04`。
- 工作 branch：`research/steiner-s02-formulation`。
- S01 local Gate：PASS；S01 GPT audit/remote push 尚未执行。
- Solver：冻结 SCIP 8.0.4 shared library + PySCIPOpt 4.3.0；命令显式设置 `LD_LIBRARY_PATH=artifacts/environment/phase4/scip804_prefix/lib`，不使用系统 SCIP 9.2.2。
- P0：seed 0、1 thread、60 s、10,000 nodes、4,096 MB。
- 外部开发实例：只允许 PACE 2018 Track 1 odd `instance001.gr`，source revision `4df73cea9c311faea7d03e6d6bffa8733c34a1aa`；known objective 来自同 revision `track1.csv`。

## 计划文件

```text
python/steiner_branching/data/{canonical,pace,steinlib,load,generate,manifest,split,write}.py
python/steiner_branching/milp/{mcf,naming,validate}.py
scripts/steiner/{download_data,generate_data,build_milp,check_solution,lock_final_content}.py
tests/steiner/fixtures/{path,triangle,star,parallel,high_cost}.stp
tests/steiner/test_{parsers,determinism,generator,manifest,mcf,solution_checker,pace_dev,cli,load}.py
configs/steiner/data/synthetic_v1.yml
configs/steiner/splits/final_test_content_v1.json
docs/steiner/phases/S02/*
```

必要时只为大型 S02 artifacts 增加精确 `.gitignore` 规则；toy fixtures 和汇总 JSON 必须可提交。

## 实现契约

- parser 只接受 undirected positive-cost SPG 的 Graph/Terminals/Comment sections；arc/prize/rooted/未知 problem sections 显式失败。
- 规范节点 ID 为 `0..n-1`；root 是最小 canonical terminal。
- 平行边保留唯一 edge ID；self-loop、非正/非有限 cost、计数不符、不连通和非法 terminal 拒绝。
- graph hash 来自 canonical nodes/edges/terminals，不依赖临时路径。
- MCF 变量命名为 `stp_x_e########`、`stp_f_t####_a########`；flow 全部 continuous，只有 x binary。
- 每 commodity 每无向边有两个方向 flow；flow balance 和双向和不超过 x。
- checker 独立于 model constraints，用 selected edge IDs 检查 terminal connectivity 和 objective。
- brute force 只用于小图并有 edge-count hard limit。
- split 按 base lineage，重复 lineage 跨 split 必须失败；synthetic seed range 使用 S00 manifest。
- download CLI 默认只支持 PACE odd development；final selector 必须拒绝。
- final content lock 使用独立 byte-only CLI；该入口不导入 parser/solver，
  校验 sealed/learning-runs=0 后只生成 archive/member checksum manifest。

## 测试矩阵

| 类别 | 必测内容 |
|---|---|
| parser | PACE/SteinLib、comments、counts、unknown section、arc、nonpositive、disconnected |
| canonical | 重复运行 hash/edge mapping 相同、平行边保留、重编号 optimum invariant |
| generator | 5 个预注册 families、同 seed bitwise manifest 相同、连通/terminal 合法 |
| split/manifest | seed range、lineage leakage、canonical JSON/SHA |
| MCF | variable types/counts/names、balance/link constraints、toy objective |
| checker | feasible/infeasible edge sets、objective mismatch、unknown edge、cycle 可行 |
| property | 增大非负边权不会产生更低 brute-force optimum |
| integration | `.gr/.stp -> SCIP -> solve -> checker` |
| public dev | PACE Track1 odd instance001 objective 等于同 revision known optimum |
| final seal | byte-only archive/member hash、338 count、learning runs 0、无 parser/solver import |
| regression | S00/S01 tests 继续通过；旧栈 diff 为空 |

## Gate S02

全部 curated toy 100% 正确；parser 无静默降级；同输入 hash/变量映射可复现；SCIP MCF objective 与穷举一致；选定 PACE odd 小实例 objective 与公布 optimum 一致；checker 验证 SCIP solution；S00/S01 tests 通过；final archive/member content lock 完整且 learned runs 仍为 0；提交不含下载数据、LP/CIP、build、checkpoint 或原始状态日志。

任一 objective mismatch、parser silent fallback、hash 不确定或 checker failure 均判 FAIL 并停止，不开始 S03。

## 外部副作用

- 允许把 PACE odd `instance001.gr`、`track1.csv` 和 license 下载到 `data/steiner/raw` 或临时目录，记录 SHA 后保持 Git ignored/untracked。
- 允许为 S00 `content_lock_policy` 把 final selector bytes 下载到临时目录；
  只提交 checksum manifest，不解析、不求解、不生成结果或学习运行。
- 允许在临时目录生成 `.lp`、metadata 和 solver logs；不提交这些产物。
- 不运行 final selectors，不 push/merge/force-push，除非用户另行明确授权。
