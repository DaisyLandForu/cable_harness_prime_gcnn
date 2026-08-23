# C1.1 Runbook — Teacher Repair（当前要跑的）

**不要**执行 `scripts/run_c1_ranking_wave1.sh`。

## 命令

```bash
cd /data/hanchengcheng/hcc_1/du/cable_harness_prim_gcnn

source /data/hanchengcheng/miniconda3/etc/profile.d/conda.sh
conda activate /data/hanchengcheng/envs/rl4scip

export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export SCIPLIB="$PWD/artifacts/environment/phase4/scip804_prefix/lib"
export TORCH_LIB=/data/hanchengcheng/envs/rl4scip/lib/python3.11/site-packages/torch/lib
export LD_LIBRARY_PATH="$SCIPLIB:$TORCH_LIB:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

mkdir -p results/c1_dataset/teacher_repair/logs
nohup bash scripts/run_c1_1_teacher_repair.sh \
  > results/c1_dataset/teacher_repair/logs/nohup_c11.log 2>&1 &
echo $!
tail -f results/c1_dataset/teacher_repair/logs/nohup_c11.log
```

## 看哪些文件

- `results/c1_dataset/teacher_repair/C1_TEACHER_REPAIR_REPORT.md`
- `results/c1_dataset/teacher_repair/teacher_quality.json`
- `results/c1_dataset/teacher_repair/fallback_analysis.csv`
- `results/c1_dataset/teacher_repair/native_probe_summary.json`
- `docs/C1_TEACHER_REPAIR_REPORT.md`

## 如何判断

| 检查 | 期望 |
|------|------|
| shard 重分类 | 大量 `strong_degenerate_floor`（不是 candidate mismatch） |
| Ecole live | `action_lp_overlap == n_action`；`vanillafullstrong` 调用为 0 或 SB 无区分度 |
| native probe | `sb_lp_iters_delta > 0` 且 `score_range > 0` 才说明 native 路径可用 |

C1.1 **本轮脚本只做审计+probe**；把 native SB 接到 collector、重采 500–1000 是下一小步（等你确认报告后再做）。
