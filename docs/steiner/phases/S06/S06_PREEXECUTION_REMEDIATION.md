# S06 pre-execution CONDITIONAL PASS remediation

Status: first remediation historical record. The focused external review
closed B1/B4 but found residual B2/B3 immutability issues. Those residuals are
handled by `S06_PREEXECUTION_REMEDIATION_V2.md`. Formal S06 execution remains
unauthorized.

## Fixed audit object

- original audited implementation head:
  `9866a3743ff1c2d6ef580aedf1f91629db0c6049`;
- original evidence/process head:
  `bcd9e33e9a09be856fc5f61078ccb4875ea325c1`;
- original audit verdict: `CONDITIONAL_PASS`;
- audit source SHA-256:
  `7fd45e09a0218f6c3e983e07c2dd0968334a5cbd9257851f653d811c5e1eaf27`;
- formal S06 results before and after remediation: `NOT_RUN`.

## B1 — terminal failure semantics

Policy and runtime exceptions now produce a complete terminal result with a
stable failure class, `solved=false` and `par2_seconds=1200`. The result stays
in the random/mostinf PAR-2 pair when the paired task exists. PDI is explicitly
`null` because no valid integral exists; it is never imputed. The missing PDI,
solver-error count and incomplete completed matrix all independently prevent a
scientific Gate PASS.

Negative tests cover invalid action, edge mapping, non-finite score and generic
solver/runtime failure. The aggregate regression verifies 150 retained PAR-2
pairs, 149 available PDI pairs and an overall failed Gate after one injected
failure.

## B2 — registered runtime identity

Every formal shard must satisfy all of the following before task launch:

- at least eight effective CPU cores after affinity and cgroup quota;
- at least 96 GiB available under the cgroup/host memory limit;
- zero visible CUDA devices;
- frozen SCIP 8.0.4 stack and frozen environment-lock checksum;
- runtime fingerprint equal to the value bound in the post-audit activation;
- activation checksum present and audited executable content unchanged.

The parent writes the exact identity into its phase manifest and passes it to
each isolated task process. The task validates and records it, and the barrier
requires byte-equivalent task/manifest identity plus one compatible identity
across all six shards. Hostname is intentionally not a compatibility key.

The runtime fingerprint covers the frozen environment lock, Python executable
bytes, Python/numpy/PyYAML/Torch/Ecole/PySCIPOpt versions and solver stack. It
does not include task outcomes or hardware identity.

## B3 — immutable aggregation

The finalizer now owns a non-blocking `aggregate.lock`. Direct calls to the
Python aggregator are rejected unless the launcher holds that lock. Before
reading barriers or writing output, aggregation refuses an existing formal
summary or run manifest. Re-aggregation therefore requires a separately
audited remediation rather than overwriting evidence.

## B4 — evidence packaging

The updated `S06_AUDIT_PACKET.md` is included in the new fixed remediation
object and maps all four findings to code and tests. The focused re-audit
request will bind the exact remediation head and file hashes after commit.

## Unchanged scientific design

The checkpoint, 30 graphs, five solver seeds, methods, P1 limits, six-shard
partition, trace policy, bootstrap, Gate thresholds and claim boundary are
unchanged. No checkpoint was loaded for a formal solve, no model was retrained,
no formal shard was executed and no test/final selector was accessed.

## Verification

- targeted S06 suite: `17 passed`;
- complete frozen-stack Steiner suite: `120 passed, 1 expected PACE skip`;
- Python compilation, shell syntax and `git diff --check`: PASS.

Historical local conclusion at `a8fbc098...`: B1--B4 appeared closed. The
subsequent focused review superseded that local assessment with B1/B4 CLOSED
and B2/B3 OPEN; it is retained rather than rewritten as a PASS.
