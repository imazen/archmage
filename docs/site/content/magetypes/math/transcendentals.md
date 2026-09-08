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

## Platform Coverage

- **x86-64**: All functions available on `f32x4`, `f32x8`, `f64x2`, `f64x4`
- **AArch64**: Full support via NEON polynomial approximations
- **WASM**: Most functions available; some use scalar fallback internally

The implementations use platform-tuned polynomial coefficients for best accuracy per instruction count.

## Known limitation: `f32x4` / `f32x8` transcendentals on AVX-512 tokens

Transcendentals on `f32x4<T>` / `f32x8<T>` are bounded by `T: F32x4Convert` / `T: F32x8Convert`. Today these traits are implemented for `X64V3Token`, `NeonToken`, `Wasm128Token`, and `ScalarToken` — **not** for `X64V4Token`, `X64V4xToken`, or `Avx512Fp16Token`.

The `f32x16<T>` path is unaffected: `F32x16Convert` is implemented for **every** token — `X64V3Token`, `X64V4Token`, `X64V4xToken`, `NeonToken`, `Wasm128Token`, and `ScalarToken` (only `Avx512Fp16Token` is missing). On AVX-512 silicon `f32x16` runs at native 512-bit width via `X64V4Token`; on every other platform the same `f32x16` code path runs through polyfills (two `f32x8` ops on V3, four `f32x4` ops on NEON / WASM, scalar lanes on `ScalarToken`). The full transcendental family (`pow_*`, `log2_*`, `exp2_*`, `ln_*`, `exp_*`, `log10_*`) is therefore available on `f32x16<T>` everywhere.

**Practical effect.** A `#[magetypes(...)]` body that calls `pow_midp` / `log2_midp` / etc. on an `f32x8` cannot include `v4` in its tier list — the trait bound rejects V4 tokens at compile time. AVX-512 hardware running such a kernel either:

1. Falls back to the V3 dispatch tier (256-bit AVX2 lanes — correct, slightly less throughput than AVX-512), or
2. Requires writing a parallel `f32x16` body for the V4 tier.

Tracked as [issue #45](https://github.com/imazen/archmage/issues/45). The fix is mechanical (delegate W128/W256 narrow backends from V4-family tokens through to V3 via `.v3()`, since V4 ⊃ V3 — same pattern as the existing `x86_v4_f32_delegated.rs`), but the build-time cost is non-trivial:

| Approach | magetypes self-build delta | Trait redesign? |
|---|---|---|
| Hand-written / codegenned per-method delegation | ~+1.8s (≈+85% on a 2.1s release build) — measured: ~3ms per `#[inline(always)]` shell × ~2000 shells | No |
| Trait redesign with default methods + `DelegateSource` associated type | ~zero in magetypes, monomorphized at use site downstream | Yes — significant change to all backend traits and impls |
| Feature-gate the delegation (`features = ["v4-narrow-delegation"]`) | Zero by default; opt-in pays the +1.8s | No |
| Widen kernels to `f32x16` where possible | Zero; works today | No, but doubles kernel surface for `f32x8`-shaped algorithms |

Until #45 lands, the recommended workarounds are (in order of decreasing ergonomics):

1. **Widen to `f32x16` if the kernel allows it.** `F32x16Convert` is impl'd on every token (V3 / V4 / V4x / NEON / WASM / scalar), so a single `f32x16<T>` kernel runs everywhere — natively at 512-bit on AVX-512, polyfilled to 2× `f32x8` on V3, 4× `f32x4` on NEON / WASM, scalar lanes on `ScalarToken`. No parallel kernels needed. *Caveats below.*
2. **Drop V4 from the tier list** and let the V3 dispatch arm handle AVX-512 hardware (correct, slightly less throughput).
3. **Hand-write an `_v4x` slot via `#[arcane]`** for the kernels where 512-bit width measurably pays off, slotted alongside the `#[magetypes]`-generated variants by suffix convention. Use this only after profiling shows the gain.

### `f32x16` polyfill overhead by operation

When the `f32x16` workaround runs on non-AVX-512 hardware, the polyfill is `[f32x8; 2]` (V3) or `[f32x4; 4]` (NEON / WASM). What that costs depends on which operations the kernel uses:

- **Pure compute (`add`, `mul`, `fma`, polynomial bodies of transcendentals):** ~zero overhead. Each method maps componentwise to N× native ops; LLVM inlines through cleanly. The assembly is identical to hand-rolling N× native-width code. Most transcendental-heavy kernels (color conversion, tone mapping, gamma curves, AgX / BT.2408 / BT.2446 / HLG) live here.
- **Reductions (`reduce_add`, `reduce_min`, `reduce_max`):** ~1.5-2× overhead. The polyfill does N sub-reductions + scalar combine, vs a smarter native version that could tree-reduce in SIMD first. A kernel that reduces every iteration pays this each loop pass.
- **Cross-lane shuffles / blends across the high/low halves:** can be expensive — extract + reassemble across sub-vectors. Heavy lane-permutation kernels may want to stay native.
- **Memory layout:** 16 elements per iteration instead of 4 / 8. Usually a *win* (less loop overhead, more ILP). On V3 (16 ymm registers) a transcendental with many live temporaries can quadruple register pressure and start spilling — measure before assuming it's free.
- **Splat / scalar setup:** ~free. Constants get CSE'd; only register copies remain.

**Rule of thumb:** for compute-bound per-element kernels (most transcendental work), widen to `f32x16` — it's effectively free on every platform. For reduction-heavy or shuffle-heavy kernels, or anything that's already register-bound on V3, profile before widening.
