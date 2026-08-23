# C1 Branch Ranking Dataset Report

- samples: **132**
- gate ≥30k: **FAIL**
- gate label quality (informative scores ≥90%): **PASS**
- informative_score_frac: 1.000
- teacher_used: `{'pseudocost_fallback': 113, 'sb': 19}`
- shards: 12 (empty=0)
- followed expert frac: 0.598
- top1/top2 teacher gap tiny (<1e-3) frac: 1.000

## Depth buckets

```
{
  "0": 12,
  "1-2": 24,
  "3-5": 40,
  "6-10": 56
}
```

## By instance

```
{
  "real_06": 68,
  "syn_medium_s101": 64
}
```

## By rollout policy

```
{
  "expert": 45,
  "epsilon_expert": 43,
  "random": 44
}
```

## Notes

- Soft labels (`teacher_scores` / `teacher_ranks`) are primary; do not treat near-ties as hard negatives.
- If `teacher_used` is mostly `pseudocost_fallback`, SB labels were degenerate on this family.
- Validation/test/transfer instances must not appear here.
- Full bipartite graphs are optional (`store_graph`); wave-1 defaults to candidate ranking features for throughput.
