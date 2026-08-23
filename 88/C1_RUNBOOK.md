# C1 Runbook — Expert Ranking Dataset

## 目标

在 **train split only** 上采集带完整 soft ranking 的 branching states：

- 每条保存 `teacher_scores` / `teacher_ranks`（不是只存 winner）
- rollout mixture：`expert` / `epsilon_expert` / `random`（缓解 imitation covariate shift）
- Gate：`n_samples >= 30000` 才能进最终 C2 训练

禁止：`validation` / `test` / `transfer` 实例进采集。

## Step 0 — 环境

```bash
cd /data/hanchengcheng/hcc_1/du/cable_harness_prim_gcnn

source /data/hanchengcheng/miniconda3/etc/profile.d/conda.sh
conda activate /data/hanchengcheng/envs/rl4scip

export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export SCIPLIB="$PWD/artifacts/environment/phase4/scip804_prefix/lib"
export TORCH_LIB=/data/hanchengcheng/envs/rl4scip/lib/python3.11/site-packages/torch/lib
export LD_LIBRARY_PATH="$SCIPLIB:$TORCH_LIB:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

## 关键：v1 pilot 标签失效

首轮 pilot（`c1_branch_ranking_pilot`）采集流程 OK，但  
`StrongBranchingScores` 在本数据上几乎全是常数 `1e-12`（SCIP product-score 地板），  
**expert 实际退化为字典序平局打破**。旧 458 SB demos 同样有此问题。

已改为 `teacher_mode: auto`：SB 无信息时回退 **Pseudocosts**（+微小 fractionality）。  
输出目录：`artifacts/datasets/c1_branch_ranking_pilot_v2`。

## Step 1 — 先跑 Pilot v2（必须）

```bash
mkdir -p results/c1_dataset/logs
nohup bash scripts/run_c1_ranking_pilot.sh \
  > results/c1_dataset/logs/nohup_pilot_v2.log 2>&1 &
echo $!
tail -f results/c1_dataset/logs/nohup_pilot_v2.log
```

看 `results/c1_dataset/DATASET_REPORT_PILOT.md`：

- `n_samples > 0`
- **`gate label quality: PASS`**（informative_score_frac ≥ 0.9）
- `teacher_used` 主要是 `pseudocost_fallback` 或真正有区分度的 `sb`
- 无明显大面积 `scip_error`
- depth buckets 不只全是 `0`

**label quality FAIL 时禁止 wave1。**

## Step 2 — Wave-1 扩量（Pilot 通过后）

```bash
nohup bash scripts/run_c1_ranking_wave1.sh \
  > results/c1_dataset/logs/nohup_wave1.log 2>&1 &
echo $!
tail -f results/c1_dataset/logs/nohup_wave1.log
```

可用 `WORKERS=8`（机器空闲可到 12–16）：

```bash
WORKERS=12 nohup bash scripts/run_c1_ranking_wave1.sh \
  > results/c1_dataset/logs/nohup_wave1.log 2>&1 &
```

支持 `--resume`：中断后重复执行同一脚本即可续跑。

## Step 3 — Gate

```bash
python scripts/analyze_c1_dataset.py \
  --dataset-dir artifacts/datasets/c1_branch_ranking
cat results/c1_dataset/DATASET_REPORT.md
```

- `gate_30k=true` → 可进入 C2  
- 否则：增大 `seeds` / 加 train 合成实例 / 提高 `node_limit`，再 resume

## 说明

- Wave-1 默认 `store_graph: false`：先保证 30k ranking 标签吞吐；GCNN 全图可在 C2 前对子集补采。
- 旧 458 SB demos 保留作 frozen baseline，不再当主训练集。
- **不要**开始 C2/C5 大训练，直到 C1 gate 通过并经你确认。
