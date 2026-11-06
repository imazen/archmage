# Independent numerical validation

Every timed math workload uses the same 1024 input bit patterns across candidates
and targets. Ordinary values are shuffled deterministically. Edge-heavy values
include min/max finite, min subnormal, both neighbors of 1, ties and signed zero;
the separate specials dataset includes NaNs and infinities. This is a workload
validation snapshot, not exhaustive testing, a maximum-error search or global proof.
The [preceding study](../portable_f32_2026-11-06/README.md) owns larger directed
MPFR and hard-case corpora for exp/ln/log2/pow; it is not silently rerun or
claimed as new coverage here.

`analyze-values.py` uses independent MPFR at 256 bits with downward/upward
endpoints. Both endpoints must convert to identical nearest-even f32/f64 bits
(or both NaN); otherwise analysis fails. Exact-bit eligibility compares those
reference bits, including signed zero, excluding NaN payload/sign. Cache keys
use raw x/y bits and precision, so signed zeros never collapse.

True-real error is bracketed using outward f64 endpoints, with subtraction and
division widened outward with nextafter. The ULP denominator is
`2^max(min_subnormal_exp, floor(log2(abs(real_result)))-fraction_bits)`;
minimum-subnormal exponents are -149/-1074, fraction bits 23/52. Zero uses
minimum-subnormal spacing. f64 endpoint bounds can be coarse around rounding
boundaries. Finite-vs-nonfinite and signed-zero errors are recorded separately.
Full raw accuracy rows remain in the artifact archive.

Correctly rounded Zenmath/Magemath candidates matched every checked record
across 4568 output files on Intel, Ryzen, Neoverse and strict WASM. This is
sample evidence, not a replacement for their broader existing contract tests.
Approximation candidates are timed with their actual error/special behavior,
including out-of-domain results: speed never implies full-range accuracy.

Metallic 0.3.1 general pow on the WASM special dataset traps with invalid integer
conversion. The isolated 23 other metallic op/dataset checks completed. That
one unavailable timing cell has a retained crash log and explicit trap manifest;
it is never filled by another implementation. Initial CLI validation used Wasmtime 40.0.1. The identical retained validation
module was then executed under official Wasmtime 49.0.5, matching the embedded
timing engine version. All 1,046 successful files (1,071,104 records) were
byte-identical between engines, including output NaN bits. The independent
MPFR checker was rerun on 49.0.5 output: no correctly-rounded finding. The same
metallic pow/special trap remained. These are corpus observations, not a global
cross-engine proof. Binary/module SHA256s and full logs are archived.
