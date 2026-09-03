# S04 Changelog

## 完成内容

- 新增 `milp_bipartite_v1`：固定 Ecole 风格 variable/constraint/edge
  features 为 19/5/1，所有数组复制后只读，shape、identity、finite 和 candidate
  invariants fail closed。
- 新增合法动作绑定：只接受 SCIP 当前 action set 中 fractional binary 变量，
  transformed `t_` 名称必须精确回映射到 `problem_meta.json` 的
  `stp_x_e########` 与唯一 `edge_id`；支持平行边。
- 新增一轮 B0 的 candidate exact closure：候选变量、其相邻 constraints、这些
  constraints 相邻的全部 variables，以及稳定 local/global identity map。
- 新增航空无关的 `MilpBipartiteGCNN`：19/5/1 输入、64 维 embedding、64 维
  hidden、一轮 variable→constraint→variable、sum aggregation、candidate logits。
- 新增 strict B0 YAML、真实 SCIP 8.0.4 CPU snapshot runner、deterministic forward
  snapshot、Gate summary 和 7 个 S04 测试。
- 将公共 naming normalization 提取到 `milp/naming.py`，S03 observer 与 S04
  共用；`configure_p1` 改成公开接口，协议内容没有变化。
- 补充 S04 CLI README、状态/主方案，并校正研究契约与资源说明中已经过时的
  S00 inventory 时态和换机前资源数字。此文档修订没有改变 Gate、split、seed、
  metric 或 scientific claim；相应 contract regression 已更新为 v1.3/resource
  fact 断言。

## Schema、配置与接口

- schema ID：`milp_bipartite_v1`
- feature widths：variable 19、constraint 5、edge 1
- model ID：`b0_milp_gcnn_v1`
- config canonical SHA-256：
  `056ce49bce41c731138a83b3befbc97e006585311bcf8f5298532c2d86f830dc`
- config file SHA-256：
  `9360f5893103adcf3b12baa3f8b2d1d5e0549791c7da68060e43542257ed1fc2`
- solver stack：`scip804-ecole081-pyscipopt430`
- snapshot：P1、train、solver seed 0、model seed 404、generator seed 100300、
  CPU one thread

Ecole 在无 incumbent 时会把 variable features 13/14 写为 NaN。B0 没有额外的
missingness channel，因此 v1 对且仅对这两个字段使用显式 zero sentinel；其他
任意 NaN/Inf 仍立即失败。该约定已写入源码注释和 regression test。

## 与计划的偏差及失败保留

- 最初本地原型使用 mean aggregation。核对标准 Gasse baseline 后，在冻结配置
  和正式 snapshot 之前改为 scatter-add/sum；先前原型不作为 Gate 证据，Gate
  阈值、split 和 seed 均未修改。
- 首次真实 Ecole 定向测试发现 1 个失败：981 个变量的 features 13/14 共 1,962
  个预期 NaN。增加上述 versioned sentinel 和异常 NaN/Inf negative test 后通过。
- 首次 inline 探针漏设 Python import path，随后使用 canonical wrapper；另有两次
  命令书写错误（不存在的 wrapper 名称、把 Python 文件交给 `bash -n`）。它们均
  记录在 test report/commands，没有伪装成测试或 Gate 失败。
- 没有删除失败样本、skip、candidate 或 snapshot；没有降低 `1e-5` parity Gate。

## 数据与未完成项

- 无数据迁移。只生成一个 frozen synthetic-train 实例并取前三个真实 SCIP
  branch states；没有读取 validation、test、PACE even 或 DIMACS final。
- 没有 teacher collection、loss、optimizer、训练、checkpoint、GPU、TorchScript
  或 C++ deployment；这些均不属于 S04。
- 旧航空源码和其 4 个既有 regression 失败未修改，仍在独立 backlog。
- SteinLib/DIMACS raw bytes、build、checkpoint、raw node logs 均未提交。
- GPT 联合审计仍是 NOT_RUN；S05 在 S00--S04 联合审计 PASS 前保持阻塞。

## Remediation v2：canonical probindex identity

- 提交并原样保留首次联合 GPT 审计请求与返回；审计结论为
  **CONDITIONAL PASS**，唯一 blocking finding 是 Ecole row 与 transformed
  variable identity 的顺序假设未被显式验证。
- 新增 `solver/scip_identity.py`：使用 SCIP 8.0.4 的公开
  `SCIPvarGetProbindex()`，以 probindex 而不是 `getVars()` list position 组装
  Ecole row names；每次 extraction 验证完整双射并 fail closed。
- C bridge 绑定 repository-frozen 绝对 prefix 和 `libscip.so.8.0` checksum，同时
  要求 wrapper 导出的 stack/version identity；拒绝系统 SCIP 和其他动态库。
- 新增 permutation、missing/count mismatch、duplicate、out-of-range、错误 stack、
  错误 prefix 负向测试；真实 frozen SCIP integration 额外覆盖 parallel edges。
- 重新生成三个 frozen states：2,943/2,943 variable rows identity 完整，31/31
  actions mapping，full/closure max error 0，argmax 3/3；snapshot bytes 未变化。
- schema 仍为 19/5/1，没有增加 missingness bit；没有重跑 S03、修改航空、采集
  teacher、训练、读取 validation/final 或降低 Gate。
