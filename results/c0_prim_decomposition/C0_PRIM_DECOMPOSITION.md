# C0 Prim Decomposition Report

## Claim

**TOPOLOGY_CONNECTIVITY**

topology-only ≈ full-prim (3.362× vs 3.340×) while root-z does not explain the gain (0.992×).

## Protocol

- Methods: gcnn / z-bias / root-z-bias / full-prim / topology-only
- Focus instances: real_01, real_05, real_08
- real_09 must NOT be used for selecting λ or mode
- Exploratory seeds in config (expand to 5 for final judgment)

## Method × Instance (shifted-geomean wall)

```
method       full-prim       gcnn  root-z-bias  topology-only     z-bias
instance_id                                                             
real_01      24.596231  82.154194    82.856959      24.435915  84.592804
real_05      69.520259  45.773465    32.422963      68.261740  20.212147
real_08      19.577667  21.642474    12.860885      20.460548  13.438310
```

## Speedup vs gcnn

```
instance_id        method  speedup_vs_gcnn  wall_sgm  gcnn_wall_sgm
    real_01        z-bias         0.971172 84.592804      82.154194
    real_01   root-z-bias         0.991518 82.856959      82.154194
    real_01     full-prim         3.340113 24.596231      82.154194
    real_01 topology-only         3.362027 24.435915      82.154194
    real_05        z-bias         2.264651 20.212147      45.773465
    real_05   root-z-bias         1.411761 32.422963      45.773465
    real_05     full-prim         0.658419 69.520259      45.773465
    real_05 topology-only         0.670558 68.261740      45.773465
    real_08        z-bias         1.610506 13.438310      21.642474
    real_08   root-z-bias         1.682814 12.860885      21.642474
    real_08     full-prim         1.105467 19.577667      21.642474
    real_08 topology-only         1.057766 20.460548      21.642474
```

## Interpretation rules

1. If `root-z-bias ≈ full-prim` on real_01 → Phase-A gain is mainly **root z-family prior**.
2. If `topology-only ≈ full-prim` and root-z fails → connectivity prior matters.
3. If only `z-bias` (all depths) works → variable-family prior, not Prim growth.
4. Do **not** proceed to hard-mask `both_in` based on these results alone.

## Artifacts

- `results/c0_prim_decomposition/method_instance_summary.csv`
- `results/c0_prim_decomposition/speedup_vs_gcnn.csv`
- raw: `results/c0_prim_decomposition/raw_results.csv`

```json
{
  "claim": "TOPOLOGY_CONNECTIVITY",
  "detail": "topology-only ≈ full-prim (3.362× vs 3.340×) while root-z does not explain the gain (0.992×)."
}
```
