# Fixtures and observed accuracy

Exact-representable fixtures passed for all five binary operations, tokenless
and explicit-token APIs, and all six generic widths. Numerators differ in every
lane across the entire width; denominators vary through positive nonzero powers
of two. Division remains exact and remainder includes nonzero cases. The
horizontal trait has an additional exact assertion. Native records 180 binary
fixture operations per run (three tiers); ARM and WASM record 120 (two tiers).
Counts describe the matrix, not a proof of general algebraic correctness.

`analyze.py` independently evaluates the actual output bit triples with MPFR at
512-bit precision. Polynomial uses the exact f32/f64 representation of the
leading 1/6 coefficient. Its other coefficients are 1/2, 1 and 1. MPFR checks
that polynomial and dot intermediates are exact. Division and remainder use
directed endpoints, which must round to the same destination bits. Reference
inputs were checked byte-for-byte across candidates before caching by workload
and band. No NaN or infinity inputs occur in this finite corpus.

Each kernel has a timed-input band and a finite mixed-magnitude band with
exponents from -20 through 20. The latter includes signed products and
cancellation. Dot compares one output; other kernels compare 1,024 outputs.

| Platform | Case/band/mode rows | Outputs | Correctly rounded bits | Nonfinite outputs | Maximum representable steps |
|---|---:|---:|---:|---:|---:|
| x86 | 552 | 393,384 | 377,192 | 0 | 208 |
| ARM | 368 | 262,256 | 253,648 | 0 | 208 |
| WASM | 368 | 262,256 | 247,088 | 0 | 432 |

These aggregate observations mix ordinary, fused, scalar-loop, external and
local-map controls. They are not an accuracy ranking. Raw per-candidate CSVs and
input/output bit triples are archived by platform and build mode. WASM fixtures
and observations ran under official Wasmtime 49.0.5, matching the embedded
49.0.5 timing engine version.

For a concrete cancellation observation, mixed f32 dot is 80 representable steps
from the rounded reference with ordinary scalar accumulation, 16 with external
tokenless scalar accumulation on x86/WASM, and 64 with external WASM128
accumulation. External V3 and NEON remain at 80. Mixed f64 ordinary scalar is
144 steps; external tokenless scalar is 48 on x86/ARM and 144 on WASM.
External V3/NEON are 208 steps and external WASM128 is 144. Both LTO modes agree
on these observations. Reassociation can change the result without improving it.

The distance metric counts ordered adjacent floating-point representations
between the observed value and the **rounded** MPFR reference. Signed zeros
are adjacent in this ordering; subnormal spacing is the format's minimum
positive subnormal (2^-149 for f32, 2^-1074 for f64). This is not a bound on
true-real error measured in ULPs. Correctly-rounded bit agreement is a separate
column. Algebraic methods provide no accuracy, evaluation-order or
bit-reproducibility guarantee; this finite snapshot establishes no global bound.
