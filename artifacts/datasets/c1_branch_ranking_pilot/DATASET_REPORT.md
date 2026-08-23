# C1 Branch Ranking Dataset Report

- samples: **151**
- gate ≥30k: **FAIL**
- shards: 12 (empty=0)
- followed expert frac: 0.609
- top1/top2 teacher gap tiny (<1e-3) frac: 1.000

## Depth buckets

```
{
  "0": 12,
  "1-2": 25,
  "3-5": 37,
  "6-10": 77
}
```

## By instance

```
{
  "real_06": 69,
  "syn_medium_s101": 82
}
```

## By rollout policy

```
{
  "expert": 50,
  "epsilon_expert": 57,
  "random": 44
}
```

## Notes

- Soft labels (`teacher_scores` / `teacher_ranks`) are primary; do not treat near-ties as hard negatives.
- Validation/test/transfer instances must not appear here.
- Full bipartite graphs are optional (`store_graph`); wave-1 defaults to candidate ranking features for throughput.
