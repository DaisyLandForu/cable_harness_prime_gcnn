# S05 formal job guide

Protocol: `s05-teacher-il-formal-v1`. At most two training seeds may run at
once. Every seed is one independent single-GPU process; DDP and shared
optimizer/state are prohibited.

## Required order

1. Complete the 315-task CPU teacher job and require teacher Gate PASS.
2. Run seeds 101 and 202 concurrently.
3. After both stop successfully, run seeds 303 and 404 concurrently.
4. Run seed 505.
5. Aggregate all five reports. Do not begin S06 unless the formal Gate passes.

## Persistent interactive host: tmux

Teacher collection (24 CPU, 128 GiB RAM, no GPU, 6 workers):

```bash
cd /home/duweiyue25/SCIP_Merge/cable_harness_prim_gcnn
scripts/steiner/run_s05_formal_teacher_tmux.sh steiner-s05-formal-teacher 6
tmux attach -t steiner-s05-formal-teacher
```

First two seeds on the current two-V100 host:

```bash
scripts/steiner/run_s05_formal_seed_tmux.sh steiner-s05-formal-101 0 101
scripts/steiner/run_s05_formal_seed_tmux.sh steiner-s05-formal-202 1 202
```

Later two-seed wave:

```bash
scripts/steiner/run_s05_formal_seed_tmux.sh steiner-s05-formal-303 0 303
scripts/steiner/run_s05_formal_seed_tmux.sh steiner-s05-formal-404 1 404
```

Final seed:

```bash
scripts/steiner/run_s05_formal_seed_tmux.sh steiner-s05-formal-505 0 505
```

Each seed needs one Tesla V100-SXM2-32GB, 8 CPU cores and 32 GiB RAM. A tmux
launcher performs the CUDA preflight before training and writes a disjoint log,
report and checkpoint path. It refuses to overwrite an existing seed artifact.

## Scheduler-managed non-interactive job

Use the original `scip` image, repository/user-space mount, Bash shell and the
repository as working directory. A non-interactive scheduler job must keep the
training command in the foreground; starting a detached tmux and allowing the
job entry process to exit can cause the scheduler to destroy the allocation.
The scheduler itself protects the process from an SSH disconnect.

Per-seed form values:

| field | value |
|---|---|
| job name | `s05-formal-seed-303` (replace seed as needed) |
| CPU | 8 |
| RAM | 32 GiB |
| GPU | 1 x Tesla V100 32GB |
| image | existing `scip` image |
| shell | Bash |
| mount | persistent user space containing the repository and formal raw data |
| working directory | `/home/duweiyue25/SCIP_Merge/cable_harness_prim_gcnn` |

Foreground startup command for a one-GPU allocation (the assigned GPU appears
inside the job as device 0):

```bash
cd /home/duweiyue25/SCIP_Merge/cable_harness_prim_gcnn
export CUDA_VISIBLE_DEVICES=0
scripts/steiner/run_s05_formal_seed_batch.sh 303
```

Replace only the final seed with 404 or 505. Do not run more than two formal
seed jobs globally at the same time. All jobs must see the same shared formal
teacher manifest and the same Git run head.

## Final aggregation

After all five seed reports exist:

```bash
scripts/steiner/run_with_scip804.sh --python \
  scripts/steiner/aggregate_s05_formal_training.py
```

The aggregator resamples 30 base-graph lineages, not 320 correlated states,
and evaluates every frozen S05 formal Gate without accessing test/final data.
