# S05 formal-v2 B1 ordering-key remediation

Status: **implemented locally; GPT re-audit pending; execution unauthorized**

The first v2 audit accepted the methodology but returned `CONDITIONAL PASS`
because the YAML used `canonical_instance_content_sha256` while the sealed
formal-v1 manifest, `TeacherSample`, explanation and existing selector use the
literal field `graph_sha256`.

The remediation changes only these two declarations:

```text
primary_order:
  state_index, graph_sha256, teacher_seed, semantic_sha256

fallback_order:
  state_index, bucket_order, graph_sha256, teacher_seed, semantic_sha256
```

The explanation now explicitly forbids aliases. A static schema test checks
that every declared ordering field exists in a formal manifest record, with
`bucket_order` derived only from the frozen `bucket_id` ordering. A synthetic
fixture matching the sealed bucket availability verifies that arbitrary input
permutation produces the same 640 selected semantic identities and the exact
frozen composition `56/72/0`, `125/3`, `64/64`, `128`, `60/68`. An unknown
ordering field fails closed.

No selector implementation or v2 manifest was executed. Formal-v1 remains
FAIL, training remains NOT_RUN, and all state budgets, seeds, hyperparameters,
validation/statistics/Gates, concurrency semantics and claim restrictions are
unchanged.
