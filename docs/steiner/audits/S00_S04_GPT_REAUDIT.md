# S00--S04 GPT Re-audit — S04 probindex remediation v2

- date: 2026-09-03 UTC
- audit type: independent read-only remediation review
- branch: `research/steiner-migration`
- v1 phase head: `123e4f3daaef1b8d15f2cc8f02a06f6edd6887fc`
- remediation content head: `4ab54ffa2b80f06ac8a9ecfe662a04df7899b072`
- remediation range:
  `123e4f3daaef1b8d15f2cc8f02a06f6edd6887fc..4ab54ffa2b80f06ac8a9ecfe662a04df7899b072`
- request: `docs/steiner/audits/S00_S04_GPT_REAUDIT_REQUEST.md`
- conclusion: **PASS**
- blocking finding B1: **CLOSED**

## Auditor conclusion

The remediation no longer uses the Python list position returned by
`getVars(transformed=True)` as variable identity. It reconstructs the complete
row-name vector by SCIP probindex and enforces:

```text
Ecole row i -> SCIP probindex i -> SCIP_VAR -> transformed name
  -> original stp_x name -> edge_id
```

`variable_names_by_probindex()` requires an exact `0..n_vars-1` bijection and
fails on count mismatch, non-integer/out-of-range/duplicate/missing probindex,
or empty/duplicate names. `SteinerNodeBipartite.extract()` applies this invariant
at every real observation extraction, not only in the snapshot runner.

The bridge is bound to the repository-frozen SCIP 8.0.4 absolute library path.
It checks stack ID, SCIP version, exact `SCIPOPTDIR`, and the pinned
`libscip.so.8.0` SHA-256 before loading `SCIPvarGetProbindex()`. There is no
system-soname fallback.

The auditor accepted the permutation/count/duplicate/out-of-range/wrong-stack/
wrong-prefix tests, the parallel-edge and frozen SCIP/Ecole integration, and the
committed Gate evidence: 3/3 states, 2,943/2,943 probindex rows, 31/31 legal
actions, full/closure max error 0, argmax 3/3, and 8/8 checks including
`probindex_identity_complete: true`.

## Independent-runtime scope

The auditor performed commit/diff, source, frozen API-contract, test-source and
committed Gate-JSON review. Their environment could not execute the repository's
frozen SCIP/Ecole runtime, so `78 passed, 1 skipped` and snapshot SHA recomputation
were explicitly marked `NOT_VERIFIED(runtime)`. This did not reopen B1 because
the unchecked list-order assumption was structurally removed and no conflict
with committed runtime evidence was found.

## Non-blocking note

Keep the v1 immutable anchors in the S04 audit packet, and record the v2 audited
anchor separately instead of overwriting history.

## Authorization resulting from PASS

The auditor permits creation of the S04 audited tag and releases the audit
blocker for S05 teacher collection and imitation learning. CUDA/resource
preflight and the teacher pilot remain mandatory S05 prerequisites.
