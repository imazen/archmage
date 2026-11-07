# Certified gamma power specialization

Power retains the existing nearest-even, gradual-underflow result contract.
Only the exact f32 exponent bit patterns `0x4019999a` and `0x3ed55555` are
eligible. The respective positive input bit ranges are
`0x3db71074..=0x3f800000` and `0x3b475041..=0x3f800000`.
Everything else continues the existing full-range CORE implementation.

The private scalar candidate uses sixteen dyadic centers and a degree-five
Estrin graph in binary64. Packed candidates use the approved global degree
thirteen/decode and degree fifteen/encode graphs through Zenmath's private
intrinsic backend. Both use separate multiply/add rounding. Uniform-exponent
packed calls mask unused lanes to a valid filler before reduction. Each invalid
or certificate-rejected lane uses the original reference power independently.
Magemath's precise adapters delegate to this shared intrinsic implementation.

For each admitted exponent band and center, the offline directed MPFR proof
accounts for coefficient rounding, table rounding, analytic truncation and the
exact evaluation graph. The selected error pad is added/subtracted in binary64;
each endpoint is widened by one adjacent binary64 value. A value is accepted
only when both endpoints round to the same f32 bits. The helper returns `None`
on rejection; no power rescue calls back into that helper. The independent
`scalar::reference::powf` remains the original CORE algorithm.

The ordinary scalar facade checks the candidate before runtime CPU selection.
Proof contexts check it inside the context. A separate private core-only entry
avoids repeating the guard after ordinary selection. Accurate policy rescue
shares the candidate; the reference policy remains independent. Cargo family
unification does not change these result bits.

The source and proof originate in gamma study commit
`1e6cc8e336ffb76727235d98c2aea988f4132bdd`. Its
[global bound](../../benchmarks/zenmath_gamma_opt_2026-11-06/BOUND.md),
[piecewise bound](../../benchmarks/zenmath_gamma_opt_2026-11-06/PIECEWISE-BOUND.md)
and [vector bound](../../benchmarks/zenmath_gamma_opt_2026-11-06/VECTOR-BOUND.md)
explain the certificate. `gamma-tables.json` identifies the frozen source,
binary and proof hashes. `tools/extract-gamma-finalists.py` checks every
selected literal against both proof records and archived compiled Rust.
Normal regeneration uses the resulting 902 bit literals in
`tools/gamma-bits.json`; no floating-point literal is hand rounded.

The admitted domains contain 29,945,741 decode and 70,823,872 encode values.
Production native V3 scalar/packed-four, V4x packed-eight, NEON scalar/packed
and strict SIMD128 WASM scalar/packed paths pass direct exact-bit comparison
for every admitted value, with the same raw little-endian XXH3 output hashes
as the frozen study. Eight-input NEON/WASM adapters comprise two four-input
blocks; they are not eight-wide hardware registers. The validation harness
also retains fixed-chunk digests and a separately labeled ordered aggregate.
This bounded two-exponent gate is not a full binary-power proof over 2^64 pairs.

Performance remains selective. The frozen experimental C0 decode/bits/64
composition lost to std (1.1443833 versus 0.9674650 ns/value; paired all-round
95% difference interval +0.1694829..+0.1840226 ns/value), although fixed-power
measurements won. Production guard, dynamic-exponent and whole-transfer
measurements are required before choosing the final automatic policy; the
experiment's fixed-exponent ratios do not establish those costs.
