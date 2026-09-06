# S05 pilot-v3 CUDA determinism remediation

Status: v3 teacher and three-seed pilot training COMPLETE; pilot Gate PASS

## Retained pilot-v2 failure

Pilot-v2 teacher collection at Git head `52c4e50fe950a316df1509dbe63e8f2a083e2c42`
completed 12/12 tasks: 149/192 valid states, 85 valid train states, 64 valid
validation-IID states, 3,939/3,939 mapped candidates, zero all-tie valid states
and zero split leakage. Its manifest SHA-256 is
`ad7915a3d603bc844dd0f0c279612c36c487513f09d3ee38649f8b19815230af`.

All nine registered GPU runs (three curve sizes for seeds 101/202/303) finished
40 epochs and wrote checksum-addressed checkpoints, but failed the unchanged
exact reload Gate. Reload logit errors ranged from `9.5367431640625e-07` to
`1.52587890625e-05`; all three failed seed reports and checkpoints remain under
the v2 paths and must not be relabeled or mixed with v3.

Failed report SHA-256 values are: seed 101
`020cbdefc284c3671689c06b67880e780f3ce5f8ecf28d28f9e843af441edce3`,
seed 202
`07bc88f84e65650ff74a9a537e87c5425d1747546164cd8aaadf5b6d3f3fae89`,
and seed 303
`7fbb57df9dca5734c875aa84606d9c6ff374e978d8232c04280898f4d1c0aa77`.

Diagnosis proved that saved/reloaded state dicts and normalization were bit
exact, while repeated inference by the same loaded model on one V100 varied by
up to `3.814697265625e-06`. The cause was incomplete CUDA determinism controls,
not teacher corruption, OOM, candidate mapping, or checkpoint corruption.

## Frozen v3 change

Pilot-v3 changes only the training-runtime determinism contract and artifact
namespace:

- `CUBLAS_WORKSPACE_CONFIG=:4096:8` is exported before either CUDA process;
- Python enables `torch.use_deterministic_algorithms(True)`, disables cuDNN
  benchmark mode and requires deterministic cuDNN behavior;
- GPU preflight and checkpoint manifests record the complete contract;
- checkpoint reload still requires state/normalization bit equality and
  `reload_max_absolute_error == 0.0`;
- missing or altered controls fail closed.

The 12 teacher tasks, split/profile, teacher seed 1001, pilot training seeds
101/202/303, 16/32/64 curve, 40 epochs, optimizer, model, metrics, and every Gate
threshold are identical to v2. Final-test access remains prohibited.

## Immutable inputs before run

- remediation content commit:
  `7fa85ff7b0d37f14d4223d396a49fb96138eb8cd`
- v3 config file SHA-256:
  `ce5b076493640863ed6f02049bd7a05df5accdad51603645cca373984b102c9e`
- v3 canonical config SHA-256:
  `cccb611deba26726772680416f49b7404c281e3520ee813dde4b2b4a100f178f`
- raw/report/checkpoint roots end in `s05-teacher-il-pilot-v3`.

Because teacher manifests bind the exact Git head, v3 recollects the same fixed
12 tasks after this amendment is committed. Reusing or editing the v2 manifest
under a new code identity is forbidden.

## Verification before registration

- S05 targeted: 12 passed, including real V100 deterministic training,
  repeated inference, checkpoint reload, and missing-environment rejection.
- Complete Steiner suite: 90 passed, 1 expected PACE-development skip.
- A separate real V100 smoke run under the registered controls completed a
  deterministic training step and returned repeated/reloaded max error 0.

## Completed v3 evidence

The commit-bound teacher rerun completed 12/12 tasks with the same aggregate
counts as v2: 149/192 valid, 85 train-valid, 64 validation-valid, zero all-tie,
3,939/3,939 mapped and zero leakage. Its manifest SHA-256 is
`9ba08c0f30f1014396c7a3825b6bb3247fe4d6e0d2dab04c180ae26cb307d36f`.

Seeds 101/202/303 completed all 9 curve runs. Every checkpoint reloaded with
bit-exact state/normalization and `reload_max_absolute_error=0.0`. The strict
aggregate report SHA-256 is
`73aa463a25d10cfd28a1e17e2396ad76eda4b945d0e2f04e7d775571bcded390`.

Mean validation normalized SB regret was 0.596116 / 0.496956 / 0.447868 for
16/32/64 states, versus random 0.685977. The primary curve improves as data
grows and its across-seed sample standard deviation falls to 0.022333 at 64.
All nine individual runs beat random. Seed 303 is slightly better at 32 than 64
on its own, and top-3/correlation are not monotone, so this is pilot evidence,
not a saturation or online-solver claim.

Pilot Gate: **PASS**. Full S05 scientific Gate: **NOT_EVALUATED** because the
formal teacher state count, formal five-seed run and preregistered significance
analysis have not been frozen/executed. S06 remains blocked.
