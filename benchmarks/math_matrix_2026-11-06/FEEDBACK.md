# Representative feedback workload, not a full latency matrix

Only f32 exp, ln, log2 and sqrt are measured here. All operations' throughput
and available widths are covered separately. Integer-rounding fixed points
and general power were not added as purported instruction-latency measurements.

Each scalar case performs 1024 dependent evaluations from initial 0.5, storing
all results. Each x8 case performs 128 dependent vector evaluations, i.e. eight
independent lane chains, also storing 1,024 results. Values quickly converge to
fixed points. Mapping is part of the measured loop:

- exp: `exp(-abs(a))`;
- ln/log2: `op(abs(a)+1.1)`;
- sqrt: `sqrt(abs(a)+0.1)`.

The mapping's arithmetic, loads/stores and cold assertions are included. The
numbers are amortized feedback-workload costs, not isolated instruction
latencies. `ns/value = mean/1024`; scalar step cost is also `mean/1024`, while
vector step cost is `mean/128 = 8*ns/value`. An x8 step performs eight elementary
operations in parallel. Do not compare scalar ns/step with vector ns/value
as though they measured the same dependency depth.

Every body keeps the whole recurrence in one feature region. Correct Magemath
uses the matching-context array entry via attuned!, rather than repeatedly
crossing a generic CPU-feature boundary. Scalar/std, correctly rounded scalar,
correct packed paths and available approximation families are separate rows.
Only native Intel/Ryzen/Neoverse feedback was measured; no WASM feedback claim.
Per-build source archives distinguish these later binaries from throughput.

Intel V3 examples (ns/value; ns/dependent step):

| Recurrence/candidate | ns/value | ns/step |
|---|---:|---:|
| exp / std/baseline/x1 | 6.5938 | 6.5938 |
| exp / magemath_portable_midp/v3/x8 | 1.3571 | 10.8568 |
| exp / magemath_correct_proof/v3/x8 | 2.4793 | 19.8342 |

These include mapping/storage and fixed-point behavior; see each platform
feedback CSV for paired intervals rather than ranking close point estimates.
