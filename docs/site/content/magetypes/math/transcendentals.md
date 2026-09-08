+++
title = "Transcendentals: Domains and Fixups"
weight = 1
+++

## Real call chain: gamma decoding in linear-srgb

The following complete chain is taken from `zen/linear-srgb` at
`6bae33ee657d0cc29eaae6fd869894ec78a41b1c`:
[`gamma_to_linear_slice` → `incant!` → `gamma_to_linear_slice_tier`](https://github.com/imazen/linear-srgb/blob/6bae33ee657d0cc29eaae6fd869894ec78a41b1c/src/simd.rs#L1313),
then `pow_midp` for full vectors and the
[scalar tail helper](https://github.com/imazen/linear-srgb/blob/6bae33ee657d0cc29eaae6fd869894ec78a41b1c/src/scalar.rs#L300).
Only the scalar helper's name/path is changed to make this excerpt standalone.
This example requires the default `w512` feature; `avx512` enables its native V4
variant. The other tiers polyfill the same logical 16-lane shape.

```rust
use archmage::prelude::*;

fn gamma_to_linear_scalar(encoded: f32, gamma: f32) -> f32 {
    if encoded <= 0.0 {
        0.0
    } else if encoded >= 1.0 {
        1.0
    } else {
        encoded.powf(gamma)
    }
}

#[archmage::magetypes(define(f32x16), v4(cfg(avx512)), v3, neon, wasm128, scalar)]
fn gamma_to_linear_slice_tier(token: Token, values: &mut [f32], gamma: f32) {
    let (chunks, remainder) = values.as_chunks_mut::<16>();
    for chunk in chunks {
        let v = f32x16::from_array(token, *chunk);
        let clamped = v.max(f32x16::zero(token)).min(f32x16::splat(token, 1.0));
        *chunk = clamped.pow_midp(gamma).to_array();
    }
    for v in remainder {
        *v = gamma_to_linear_scalar(*v, gamma);
    }
}

pub fn gamma_to_linear_slice(values: &mut [f32], gamma: f32) {
    incant!(
        gamma_to_linear_slice_tier(values, gamma),
        [v4, v3, neon, wasm128, scalar]
    )
}


let mut values = [0.5; 19];
gamma_to_linear_slice(&mut values, 2.2);
for value in values { assert!((value - 0.5_f32.powf(2.2)).abs() < 1e-4); }
```

For finite normalized samples and positive gamma, this decodes a simple power
transfer curve. It is not the piecewise sRGB transfer function. The vector body
uses a polynomial approximation, while the tail uses scalar `powf`: do not
promise bit identity between them. Test the consumer's error budget, especially
around chunk boundaries. The code and all lengths through two chunks plus a
tail are tested in `magetypes/tests/doc_examples.rs`.

## Exact calls and runtime repair

| Call | Additional work versus its unchecked polynomial | Domain / behavior |
|---|---|---|
| `v.log2_midp()` | Three compare/blend stages | Repair zero to negative infinity, negatives to NaN, positive infinity to infinity. This is not a universal guarantee of NaN payload or subnormal accuracy. |
| `v.exp2_midp()` | Two min/max clamps, two comparisons, two blends | Clamp the polynomial input, return zero below −126 and infinity at/above 128. Subnormal outputs are deliberately not constructed. NaN behavior inherits backend min/max differences. |
| `v.ln_midp()` / `v.log10_midp()` | `log2_midp` repairs plus scaling | Same logarithm domain restrictions. |
| `v.exp_midp()` | Input scaling plus `exp2_midp` repairs | Same clamping policy after scaling. |
| `v.pow_midp(gamma)` | Composes repaired log/exp and scaling | Not a full scalar `powf` replacement over all negative bases, exponents, NaNs and infinities. |
| `v.sigmoid_midp()` / `v.silu_midp()` | Their documented exponential arithmetic and division | Use division internally; do not infer raw reciprocal-estimate special cases. |
| `_unchecked` variants | Omit the corresponding domain repair | Memory-safe methods with numerical preconditions; the name does not authorize raw memory access. |

These counts describe the expression graph. Inlining, constant folding, width
polyfills, and ISA lowering change actual instructions and CPU cost. Isolated
checked-versus-unchecked timings for these transcendental methods are not yet
recorded in the [ISA explorer](../../../isa-explorer/). Do not substitute the
reciprocal repair percentages for transcendental overhead.

`_lowp` and `_midp` select approximation families, not one cross-ISA ULP guarantee
for every input. FMA, range reduction, conversion behavior, and the input domain
all matter. See [precision levels](@/magetypes/math/precision.md) and the
[ISA contracts](@/magetypes/isa-quirks.md).

## Width and backend selection

The complete example above uses [`f32x16`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x16.html) and requires `w512` (enabled by default).
Native V4 additionally needs `avx512`; other listed tiers use polyfills. These
are the supported listed tokens, not every token in the registry. The f32 and
f64 method sets also differ; follow the selected type's API reference.

For [`f32x4`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x4.html) / [`f32x8`](https://docs.rs/magetypes/latest/magetypes/simd/generic/struct.f32x8.html) transcendental bodies, use V3, NEON, WASM, and scalar.
Current V4-family narrow conversion bounds are not supplied merely by the
hardware's superset features. Keeping V3 on AVX-512 hardware is valid; widening
to sixteen lanes is an algorithm and performance decision, not automatic macro
behavior.

Polyfills can split elementwise arithmetic efficiently, but reductions,
shuffles, and live temporaries have different costs. There is no established
universal 1.5–2× overhead or a guarantee that widening is free. Inspect assembly
and measure representative inputs before changing the kernel's shape.
