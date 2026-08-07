# 阶段 2：航空布线实例数据集报告

日期：2026-08-04  
环境：Conda `rl4scip`，SCIP 8.0.4 / SoPlex 6.0.4  
配置：`configs/dataset/phase2.json`

## 1. 本阶段目标与边界

本阶段把真实航空布线实例和业务语义一致的确定性合成实例组织为按原始 MILP 划分的数据集，并为每个实例导出 CIP、metadata 与 baseline 指标。没有把 B&B 状态随机拆分到不同集合，没有训练模型，也没有实现自定义 branchrule。

交付规模：24 个实例，CIP 总大小约 198 MB。

## 2. 修改文件

| 文件 | 内容 |
|---|---|
| `configs/dataset/phase2.json` | split、合成规模、seed、baseline 资源配置 |
| `scripts/instance_generator.py` | 航空布线语义一致的确定性生成器 |
| `scripts/build_dataset.py` | 生成源 CSV、导出 CIP、运行/复用 baseline、生成 metadata 和 manifest；支持断点续跑 |
| `scripts/validate_dataset.py` | hash、split、seed、规模、可复现性和 SCIP CIP 读取检查 |
| `code/data/synthesis/dataext.py` | 为旧生成器增加 `--seed`、输出目录和死循环保护 |
| `code/scip_tree.cpp` | 支持 6 列合成 pair schema；JSON 增加中心图节点、边和 commodity 数量 |
| `README.md` | 数据集构建与验证命令 |

没有修改任何真实 `code/data/edges-*.csv` 或 `pairs-*.csv` 文件，也没有改变 MILP 数学表达式。

## 3. 为什么不直接使用旧合成文件

旧 `dataext.py` 把 `E` 前缀节点称为 end node，并直接把 pair 端点设为 `E`。当前 C++ 模型的实际语义是：

- `N/M` 加纯数字：主干中心节点；
- `E`：入口中心节点，属于中心图；
- 其他前缀：线缆叶端点，通过 access edge 连接到 `E`。

因此旧合成文件不能保证 pair 端点存在 `leaf_to_center` 映射，不适合作为正式训练数据。本阶段保留旧脚本并增加 deterministic seed，但正式数据集使用新的 `instance_generator.py`。

旧脚本的默认小规模还存在一个原有死循环：只有两个中心节点时，唯一中心边已用于连通链，脚本仍尝试随机生成额外不同中心边。本阶段改为枚举可用边并安全截断。同 seed 双次运行的 edge/pair CSV 已逐字节一致。

## 4. 新合成器的业务结构

每个生成实例包含：

1. 由 `N0...` 构成的连通主干链；
2. 在主干节点之间增加确定性随机的额外拓扑边；
3. 每个 `E` 入口节点连接一个主干节点；
4. 每根线缆有两个唯一 `L...` 叶端点；
5. 每个叶端点通过 access edge 连接到一个 `E`；
6. 每根线缆选择唯一的入口节点对，避免多个输入 pair 被模型聚合成同一 commodity。

变化的是实际存在的业务参数：主干节点数、入口节点数、中心边密度、线缆数、拓扑和物理边长。没有通过给同一个 MILP 随机扰动 objective 来扩充样本。线缆权重使用固定周期 `[1e-7, 2e-7, 4e-7]`，随机性来自场景拓扑和物理边长。

| 规格 | 主干节点 | 入口节点 | 额外主干边 | 线缆 | MILP 变量 | 整数变量 | 约束 |
|---|---:|---:|---:|---:|---:|---:|---:|
| small | 8 | 6 | 4 | 6 | 522 | 160 | 1,023 |
| medium | 24 | 12 | 20 | 18 | 3,626 | 512 | 8,363 |
| large | 60 | 20 | 60 | 45 | 20,377 | 1,292 | 49,675 |

所有配置使用 `copy_num=4`，与真实 baseline 一致。

## 5. Split 设计

| split | 真实场景 | 合成 seed | 数量 |
|---|---|---|---:|
| train | TESTA10（real_06）、TESTA02（real_07） | small/medium/large 各 101、102、103 | 11 |
| validation | TESTA03（real_08） | small/medium/large 各 201 | 4 |
| test | TESTJK01（real_09） | small/medium/large 各 301 | 4 |
| transfer | F0001（real_01）、全部 TESTA01 变体（real_02～05） | 无 | 5 |

设计理由：

- train/validation/test 对每个合成规模使用完全相同的业务参数，仅场景 seed 不同，可以做同分布泛化检查。
- 真实场景 ID 不跨 split。TESTA01 的四个相关变体整体进入 transfer，避免相关场景泄漏。
- train、validation、test 都有真实实例，最终评估不只依赖合成问题。
- transfer 是更大规模或真实结构分布变化：包含 F0001、TESTA01 系列和两个 large MILP。
- 当前最大 real_04 有 326,502 个变量、5,168 个整数变量、863,691 条约束，只在 transfer，不进入训练。

split 单位始终是完整原始 MILP。后续采集的同一实例所有 B&B 状态必须继承该 split。

## 6. 构建与复现命令

编译：

```bash
conda run -n rl4scip make
```

完整构建或从断点继续：

```bash
conda run -n rl4scip python scripts/build_dataset.py \
  --config configs/dataset/phase2.json \
  --binary build/scip_tree \
  --instances-dir data/instances \
  --generated-dir data/generated \
  --results-dir results/dataset \
  --manifest data/instances/manifest.csv \
  --resume
```

`--resume` 对 CIP、build JSON 和 metadata 均完整的实例只读取 metadata；部分完成实例只补缺失的 baseline 或 metadata。阶段 2 曾在 validation 的 `syn_large_s201` baseline 中断，续跑时前 14 个完整实例均被跳过，断点实例从缺失步骤恢复。

验证：

```bash
conda run -n rl4scip python scripts/validate_dataset.py \
  --config configs/dataset/phase2.json \
  --manifest data/instances/manifest.csv \
  --scip-binary /home/duweiyue25/SCIP/scipoptsuite-8.0.4/build/bin/scip
```

真实输出：

```text
Dataset validation passed: 24 instances
Split counts: train=11, validation=4, test=4, transfer=5
```

## 7. 输出结构

```text
data/
  generated/<split>/<instance>_{edges,pairs}.csv
  instances/
    train/*.cip, *.json
    validation/*.cip, *.json
    test/*.cip, *.json
    transfer/*.cip, *.json
    manifest.csv
results/dataset/
  raw/build/*.json, *.log
  raw/baseline/*.json, *.log
  validation.log
```

每份 metadata JSON 包含：split、场景、source type、生成 seed、baseline seed、业务图规模、生成参数、源 CSV 路径和 SHA-256、CIP SHA-256、变量/整数变量/约束/commodity 数量、constraint-variable ratio，以及 baseline 状态、界、gap、时间和节点。

manifest 满足以下主要字段：split、instance_id、seed、size、业务参数、变量数、整数变量数、约束数、baseline 时间、baseline 节点，并额外记录 source type/scenario、commodity、状态、gap 和产物路径。

## 8. Baseline 画像

统一资源：SCIP default、seed 0、30 秒、单线程、`copy_num=4`。

- 24 个实例中 15 个 optimal，9 个 time limit。
- 15 个新合成实例中 12 个 optimal。
- small 全部在约 0.05 秒内根节点求优。
- medium 全部求优，时间约 0.4～4.9 秒。
- large 的 train seed 102/103 求优；train seed 101、validation seed 201、test seed 301 在 30 秒超时，后两者保留有限 gap。
- 真实 real_06、real_08、real_09 求优；其余使用阶段 1 的受控时限结果。
- real_04 的 30 秒仍主要用于 presolve，baseline nodes 为 0。它是 transfer 压力测试，不能用于短时 branching 比较。

合成 large 不是人为保证“容易”或“困难”；不同 seed 导致不同拓扑，真实保留了可学习的难度差异。

## 9. 验收检查

自动验证完成：

- manifest 恰好 24 个唯一 instance ID；
- 24 个 CIP 与 24 个 metadata 均存在且非空；
- 所有源 CSV 和 CIP 的 SHA-256 与 metadata 一致；
- 15 个合成实例全部重新生成，edge/pair 文件哈希逐一一致；
- train/validation/test 的 small、medium、large 均覆盖；
- 合成 size/seed 在 split 之间无重叠；
- 真实 source scenario 不跨 split；
- 最大真实实例位于 transfer；
- 所有 MILP 的变量、整数变量、约束和 commodity 均为正；
- 24 个 CIP 均由独立 SCIP 8.0.4 CLI 成功重新读取，零失败。

## 10. 风险与后续使用规则

1. 真实训练实例只有 real_06、real_07，合成到真实的分布差异仍是主要风险。训练和验证报告必须分别统计 synthetic 与 real。
2. 旧 synthesis 文件不进入正式 manifest；它们只能用于历史对照，不能与新生成器产物混用。
3. 原 C++ 模型仍把 edge weight 存为整数，真实 CSV 小数被截断。这是既有模型行为，本阶段未改变。
4. 相关 TESTA01 变体都在 transfer，因此 transfer 内部不是独立同分布样本；统计时应同时给出场景级结果。
5. CIP 共约 198 MB，训练环境应按 episode 按需读取，不能一次把全部 SCIP 模型常驻内存。
6. 阶段 4 采集 B&B transition 时，必须按 manifest split 加载完整实例，严禁重新随机划分状态。

## 11. 验收结论

| 条件 | 状态 |
|---|---|
| deterministic seed | 通过：旧脚本和新生成器均字节级复现 |
| small/medium/large | 通过 |
| train/validation/test 同规格不同 seed | 通过 |
| transfer 为更大规模或结构变化 | 通过 |
| 最大实例不进入训练 | 通过：real_04 在 transfer |
| 无实例/场景/seed 泄漏 | 通过 |
| 每个实例 CIP + JSON | 通过 |
| manifest 字段与 baseline | 通过 |
| SCIP 可重新读取 | 通过：24/24 |

阶段 2 到此完成，等待用户批准后才进入阶段 3。
