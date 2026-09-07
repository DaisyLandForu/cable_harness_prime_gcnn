# S05 formal-v1 teacher failure record

Formal run head: `13a8a83761f660c96cb238471de2e2c0a2d54a8b`
Manifest SHA-256: `2bb3b4875d173571df5e4fb7e9c1e1d3f4615708308e891f4e4ccdbaac4c149c`
Result: **FAIL; training not started**

All 315/315 tasks completed and there were no task failures. The run produced
3,431 observed states, 3,207 valid states and 88,549/88,549 mapped candidates.
Valid-state fraction was 63.630952%, all-tie among valid states was 0.436545%,
and role/split leakage was zero. Those checks passed.

The only failed check was `all_state_quotas_met`. Validation-select reached
160/160 and validation-gate reached 320/320, but train reached only 527/640.

| train stratum | eligible / target | shortage |
|---|---:|---:|
| sparse Erdős–Rényi large-high | 0 / 48 | 48 |
| random geometric medium-mid | 3 / 64 | 61 |
| bridge bottleneck medium-mid | 60 / 64 | 4 |

No task, state or failure record was deleted. The automatic seed-101/202 watcher
correctly refused to start training after seeing the failed teacher Gate.
Formal-v1 remains failed even if a later audited revision succeeds.
