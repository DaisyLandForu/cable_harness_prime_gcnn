# S05 non-interactive batch job guide

## Current state

- S04 remediation audit: PASS; `steiner-s04-audited-v2` exists locally.
- S05 teacher pilot manifest: NOT_RUN / absent.
- S05 pilot training report: NOT_RUN / absent.
- Formal teacher state count: not frozen; formal collection/training must not start
  until the pilot learning curve has been analyzed.

All jobs below must use the same repository checkout and Git HEAD. The teacher
manifest records the HEAD and training fails closed if the checkout changes.
The repository mount must therefore be persistent and shared by the CPU and GPU
jobs.

## Job A: teacher pilot

- job name: `s05-teacher-pilot`
- CPU: 12 cores (24 is safe but does not increase the frozen 6-worker limit)
- memory: 64 GiB (128 GiB is a conservative alternative)
- GPU: 0
- image: the existing `scip` image containing the locked `rl4scip` environment
- shell: `bash`
- file mount: user space at `/home/duweiyue25`
- working directory:
  `/home/duweiyue25/SCIP_Merge/cable_harness_prim_gcnn`
- startup command:

```bash
set -euo pipefail
cd /home/duweiyue25/SCIP_Merge/cable_harness_prim_gcnn
scripts/steiner/run_s05_teacher_batch.sh 6
```

Do not submit GPU training until this job exits with code 0 and
`results/steiner/raw/s05/s05-teacher-il-pilot-v1/manifest.json` reports
`status: completed`.

## Jobs B1/B2: parallel pilot training on two GPUs

Each job requests one scheduler-managed GPU. Do not override
`CUDA_VISIBLE_DEVICES`; each container uses its assigned logical `cuda:0`.

Common resources:

- CPU: 4 cores
- memory: 16 GiB
- GPU: 1 V100 32 GiB
- image/shell/mount/working directory: same as Job A

Job B1:

```bash
set -euo pipefail
cd /home/duweiyue25/SCIP_Merge/cable_harness_prim_gcnn
scripts/steiner/run_s05_pilot_seed_batch.sh 101 303
```

Job B2:

```bash
set -euo pipefail
cd /home/duweiyue25/SCIP_Merge/cable_harness_prim_gcnn
scripts/steiner/run_s05_pilot_seed_batch.sh 202
```

These jobs write disjoint checkpoints and reports. They may run concurrently
after Job A succeeds.

## Job C: strict pilot aggregation

- job name: `s05-pilot-aggregate`
- CPU: 2 cores
- memory: 4 GiB
- GPU: 0
- image/shell/mount/working directory: same as Job A
- start only after B1 and B2 both exit with code 0
- startup command:

```bash
set -euo pipefail
cd /home/duweiyue25/SCIP_Merge/cable_harness_prim_gcnn
scripts/steiner/run_with_scip804.sh --python \
  scripts/steiner/aggregate_s05_pilot_training.py \
  --input results/steiner/s05/s05-teacher-il-pilot-v1/pilot_training_shards/seeds-101-303.json \
  --input results/steiner/s05/s05-teacher-il-pilot-v1/pilot_training_shards/seeds-202.json
```

The aggregate fails if a registered seed or learning-curve run is missing,
duplicated, failed, or tied to a different Git/config/teacher fingerprint.

## Formal five-seed boundary

The registered formal training seeds are `101, 202, 303, 404, 505`. Once the
pilot Gate passes and the formal teacher state count/config are frozen, they may
be executed as five independent one-GPU jobs, or as two one-GPU jobs containing
three and two seeds. With only two GPUs, no more than two jobs execute at once.
The current pilot CLI deliberately rejects formal seeds 404/505, so the formal
jobs cannot accidentally start before that prerequisite work is complete.
