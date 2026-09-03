# S03 Commands

所有命令从仓库根目录执行。不要直接调用系统 SCIP 9.x。

## 版本与资源核实

```bash
git branch --show-current
git rev-parse HEAD
git status --short
cat /sys/fs/cgroup/cpu.max
cat /sys/fs/cgroup/memory.max
cat /sys/fs/cgroup/memory.events
scripts/steiner/run_with_scip804.sh --verify-only
```

## 构建与测试

```bash
CONDA_PREFIX=/home/duweiyue25/conda/envs/rl4scip make steiner-s03-probe
bash -n scripts/steiner/run_s03_tmux.sh
scripts/steiner/run_with_scip804.sh --python -m compileall -q \
  python/steiner_branching scripts/steiner/run_s03_branchability.py
scripts/steiner/run_with_scip804.sh --python -m pytest -q \
  tests/steiner/test_s03_branchability.py \
  tests/steiner/test_pre_s03_readiness.py \
  tests/steiner/test_scip804_wrapper.py
scripts/steiner/run_with_scip804.sh --python -m pytest -q tests/steiner
```

## 可恢复正式运行

```bash
scripts/steiner/run_s03_tmux.sh steiner-s03 6
tmux attach -t steiner-s03
```

在 tmux 中按 `Ctrl-B`、再按 `D` 只会 detach，不会停止任务。换服务器后使用
新的本机 session 名重复同一命令即可；runner 只跳过 config/task fingerprint
一致的 atomic shard：

```bash
scripts/steiner/run_s03_tmux.sh steiner-s03-resume 6
tail -f results/steiner/raw/s03/s03-branchability-pilot-v1/tmux.log
find results/steiner/raw/s03/s03-branchability-pilot-v1/shards \
  -maxdepth 1 -name 'formal--*.json' | wc -l
```

不用 tmux 的等价前台命令：

```bash
scripts/steiner/run_with_scip804.sh --python \
  scripts/steiner/run_s03_branchability.py --max-workers 6
```

raw shards、CIP 和 tmux log 位于 ignored `results/steiner/raw/s03/`。不得提交。

## Gate 与 Git 边界

机器可读结果：

```bash
sed -n '1,260p' docs/steiner/phases/S03/S03_GATE_SUMMARY.json
git diff --check
git diff --cached --name-only
git diff --cached --check
git rev-list --left-right --count HEAD...origin/research/steiner-migration
```

只有本地 Gate PASS 且远端不 ahead/diverged 才允许：

```bash
git push origin research/steiner-migration
```

不得使用 `merge`、`rebase`、`commit --amend` 或 `push --force`；命令和文档中
不得写入 PAT。
