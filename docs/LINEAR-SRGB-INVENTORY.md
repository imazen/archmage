# Linear-sRGB math and configured replacements

Read-only source inventory, 2026-11-10, clean linear-srgb revision
[`8900ffc87cd50b249bae19c99d874431e055d888`](https://github.com/imazen/linear-srgb/tree/8900ffc87cd50b249bae19c99d874431e055d888).
No linear-srgb source or dependency was changed for this inspection.

| Use | Existing graph | Configurable status / equal-cost prospect |
|---|---|---|
| sRGB decode | Degree-4/4 rational, eight Horner FMA operations and a divide; piecewise linear toe/clamps | No matching Magemath transfer fit. A fixed-coefficient, fixed-degree inline P/Q kernel can retain this arithmetic, but needs codegen and throughput evidence. Generic pow is a different graph. |
| sRGB encode | sqrt then degree-4/4 rational, toe/clamps | Basic sqrt already exists in Magetypes. Same P/Q opportunity; coefficients and sqrt placement must remain fit-specific. |
| Extended sRGB | Degree-6/6 rational; encode fitted on sqrt(input) | Same opportunity, twelve Horner FMA operations plus divide, and encode sqrt. |
| fast_log2f | Exponent reduction plus libjxl-style degree-2/2 rational | Magemath LOWP has the same structure, but one coefficient differs by one f32 ULP. Closest measured replacement. |
| fast_pow2f | Reduced degree-3/3 rational plus exponent reconstruction | Configured exp2 LOWP is a cubic polynomial; MIDP is a higher-degree polynomial. Neither preserves this graph or fit. |
| BT.709 | Piecewise fast_powf = log2 × exponent → rational exp2 | Configured powf exists; measure complete curves and their boundary/roundtrip error before replacing. |
| HLG | sqrt for low segment, fast_log2f or fast_pow2f for high segment | sqrt is available. Log2 is close; exp2 needs a new comparison. |
| PQ | Piecewise rational fits on transformed input, including sqrt(sqrt(x)) for encode; tiny scalar segments use powf | Preserve dedicated fits. A polynomial evaluator can share mechanics; generic pow alone does not express their complete cost or error. |

Sources: [sRGB coefficients and scalar evaluator](https://github.com/imazen/linear-srgb/blob/8900ffc87cd50b249bae19c99d874431e055d888/src/rational_poly.rs),
[SIMD sRGB](https://github.com/imazen/linear-srgb/blob/8900ffc87cd50b249bae19c99d874431e055d888/src/simd.rs),
[transfer fast math](https://github.com/imazen/linear-srgb/blob/8900ffc87cd50b249bae19c99d874431e055d888/src/tf/fast_math.rs),
[BT.709](https://github.com/imazen/linear-srgb/blob/8900ffc87cd50b249bae19c99d874431e055d888/src/tf/bt709.rs),
[HLG](https://github.com/imazen/linear-srgb/blob/8900ffc87cd50b249bae19c99d874431e055d888/src/tf/hlg.rs),
[PQ](https://github.com/imazen/linear-srgb/blob/8900ffc87cd50b249bae19c99d874431e055d888/src/tf/pq.rs).

The scalar sRGB rational evaluator uses f64 intermediates with f32 coefficients;
SIMD uses f32 intermediates. Sharing a configurable evaluator will not itself
resolve this contract difference. Replacing P/Q with P*recip(Q) also introduces
an extra rounding step, even when the reciprocal is HIGHP.

## Measured log2 comparison

A mirror of the exact linear-srgb log2 graph on current Magetypes takes
5.765597 ns per 64 inputs, versus 5.712658 ns for configured LOWP,
BackendDependent, AssumeValid, Caller, V3 x8 on Zen 5. Fifty interleaved rounds
are admission-clean in this cell; the all-retained-pair 95% time-ratio interval
is [0.978026, 1.000227], within the 2% cell bound. This is not a timing of the
published linear-srgb binary, and does not qualify the whole primitive suite,
whose reciprocal throughput comparison failed admission.

The log2 numerator's quadratic coefficient is `0x3f3e11c7` in linear-srgb and
`0x3f3e11c6` in Magemath; the other five coefficients match. Therefore it is a
nearby fit, not a bit-identical replacement. Selecting the original coefficient
at compile time could preserve instruction count; it would still need complete
curve accuracy and tail validation. No new fit-selection API was added here.
See the [study protocol/results](../benchmarks/primitive_variants_2026-11-10/README.md).

## Confirmed existing tail mismatch

Tracked in [linear-srgb issue 32](https://github.com/imazen/linear-srgb/issues/32).

The actual unmodified linear-srgb crate, built from the isolated source revision
above with its own dependencies, fails this reproducer:

```rust
let mut values = [0.125_f32; 17];
linear_srgb::default::srgb_to_linear_slice(&mut values);
assert_eq!(values[0].to_bits(), values[16].to_bits());
// SIMD: 0x3c6b22d7; scalar tail: 0x3c6b22d8
```

`just primitive-linear-original CHECKOUT DURABLE_OUTPUT` verifies a clean source
checkout, archives it to an isolated copy, builds that crate and runs this
assertion. Its nonzero exit preserves the actual mismatch. The source checkout
is untouched. The separate `primitive-linear-inventory` coefficient mirror also
finds a one-bit difference at 0.9. These are witnesses, not an exhaustive count.

The next transfer-specific step should choose one scalar/SIMD arithmetic contract
and preserve the existing monotonicity, toe, endpoint and roundtrip requirements.
Using f64 everywhere or f32 everywhere has different performance/quality effects;
this quick inventory does not silently choose either or change the owning crate.
