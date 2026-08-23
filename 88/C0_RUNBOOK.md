# C0 Runbook（请按顺序执行）

工作目录：

```bash
cd /data/hanchengcheng/hcc_1/du/cable_harness_prim_gcnn
```

环境：

```bash
source /data/hanchengcheng/miniconda3/etc/profile.d/conda.sh
conda activate /data/hanchengcheng/envs/rl4scip
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
export SCIPLIB="$PWD/artifacts/environment/phase4/scip804_prefix/lib"
export TORCH_LIB=/data/hanchengcheng/envs/rl4scip/lib/python3.11/site-packages/torch/lib
export LD_LIBRARY_PATH="$SCIPLIB:$TORCH_LIB:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

## Step 0 — 编译（必须）

```bash
make -j8 CXX=/usr/bin/g++
./build/scip_tree --help | grep -E 'bias-mode|prim'
```

应看到 `--rl-bias-mode`。

## Step 1 — 单元测试 / parity

```bash
python -m pytest tests/python/test_prim_bias.py tests/python/test_prim_parity.py -q
```

## Step 2 — C0.2 Prim 拆解评测（核心）

建议用 `nohup`（约 30 jobs；`real_01/05` 单次可达 600s）：

```bash
mkdir -p results/c0_prim_decomposition/logs
nohup bash scripts/run_c0_prim_decomposition.sh \
  > results/c0_prim_decomposition/logs/nohup_outer.log 2>&1 &
echo $!
tail -f results/c0_prim_decomposition/logs/nohup_outer.log
```

或前台：

```bash
bash scripts/run_c0_prim_decomposition.sh
```

矩阵：`real_01/05/08` × {gcnn, z-bias, root-z-bias, full-prim, topology-only} × seeds（探索 0,1；最终判断再用 0–4）。  
输出：`results/c0_prim_decomposition/`  
脚本末尾会自动跑 `analyze_c0_decomposition.py` 与 `analyze_c0_audit.py`。

## Step 3 — C0.1 审计汇总（依赖 Step2 的 branch logs）

```bash
python scripts/analyze_c0_audit.py \
  --input-glob 'results/c0_prim_decomposition/raw/**/*.branches.csv' \
  --raw-results results/c0_prim_decomposition/raw_results.csv \
  --output-dir results/c0_audit
```

输出：`decision_logs.csv` / `q_prim_scale_analysis.csv` / `C0_AUDIT_REPORT.md`

## Step 4 — 阅读结论

重点文件：

- `results/c0_prim_decomposition/C0_PRIM_DECOMPOSITION.md`
- `results/c0_audit/C0_AUDIT_REPORT.md`
- `88/NOTES_AND_PITFALLS.md`

**若 `root-z-bias ≈ full-prim` 复现 real_01 收益，必须在报告中明确写出：收益主要来自 root variable-family prior。**

## 尚未自动完成（C0 后续可选）

- 从 live SCIP dump 20 个 depth 的 C++/Python bitwise snapshot（`scripts/run_c0_parity_snapshots.sh`，需较长求解）
- 5 seeds 终局确认（把 `C0_SEEDS=0,1,2,3,4`）

## 明确不要做

- 不要跑 Stage C both_in mask  
- 不要启动 B 加长训练 / C5 RL  
- 不要用 real_09 选超参  
