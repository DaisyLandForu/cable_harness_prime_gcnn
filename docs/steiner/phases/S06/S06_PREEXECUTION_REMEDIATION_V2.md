# S06 pre-execution B2/B3 immutability remediation

Status: implemented and locally verified; focused external re-audit required.
Formal S06 execution remains unauthorized.

## Fixed history

- first remediation head:
  `a8fbc0986068171344387ceb68e1ee5ab5bbefbc`;
- first focused audit-request head:
  `98a535d19924c80df47fcb3f8153d7fdf89ccf86`;
- focused verdict: `CONDITIONAL_PASS`;
- audit source SHA-256:
  `fa63f4cce005cccd78c872dad152680eaccec5ee521d6cc93ca7772b73eef611`;
- B1/B4: `CLOSED`; B2/B3: `OPEN_AT_AUDITED_HEAD`;
- formal S06 results: `NOT_RUN`.

## B2 — committed activation bytes

The activation loader now resolves the requested path, reads its local bytes,
requires the resolved file to remain inside the repository and obtains the
same path from `git show HEAD:<relative-path>`. Git lookup must succeed and the
committed bytes must exactly equal the local bytes before JSON parsing and
authorization checks proceed.

Consequently, changing `container_runtime_fingerprint` or any other activation
field without committing it cannot authorize a different runtime. The runtime
identity's activation SHA is now derived only from a record already proven to
be byte-identical to `HEAD`.

## B3 — Python-owned aggregate lock

The environment-variable guard and shell-owned lock were removed. `_aggregate`
now opens the formal run's `locks/aggregate.lock`, obtains
`LOCK_EX | LOCK_NB`, and holds the file descriptor until barrier validation,
summary writing and run-manifest writing finish. A second shell finalizer and a
direct Python caller contend for exactly the same kernel lock and cannot both
enter the evidence-writing section.

The existing refusal to overwrite `S06_GATE_SUMMARY.json` or the formal run
manifest remains unchanged.

## Tests

- a committed synthetic activation is accepted;
- modifying its local runtime fingerprint while Git continues to expose the
  original bytes is rejected;
- two direct Python aggregate-lock contenders cannot both obtain the lock;
- existing formal aggregate outputs remain immutable;
- all B1/B4 and scientific-protocol tests remain present.

## Unchanged boundaries

No YAML scientific field changed. The checkpoint, instances, solver seeds,
methods, P1 limits, 750-task matrix, trace policy, bootstrap, Gate thresholds
and claim boundary are unchanged. No formal task, model retraining, S07 work or
test/final access occurred.
