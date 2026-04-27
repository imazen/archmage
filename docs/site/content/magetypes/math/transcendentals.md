+++
title = "Transcendentals"
weight = 1
+++

Magetypes provides SIMD implementations of common math functions. These are polynomial approximations tuned per platform — faster than calling scalar `f32::exp()` in a loop.

## Exponential Functions

```rust
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Convert,
};

#[inline(always)]
fn exponentials<T: F32x8Convert>(token: T) {
    let v = f32x8::<T>::splat(token, 3.0);

    // Base-2 exponential: 2^x
    let result = v.exp2_midp();   // [8.0; 8]

    // Natural exponential: e^x
    let result = v.exp_midp();    // [e^3; 8] ~ [20.09; 8]
}
```

## Logarithms

```rust
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Convert,
};

#[inline(always)]
fn logarithms<T: F32x8Convert>(token: T) {
    let v = f32x8::<T>::splat(token, 8.0);

    // Base-2 logarithm: log2(x)
    let result = v.log2_midp();   // [3.0; 8]

    // Natural logarithm: ln(x)
    let result = v.ln_midp();     // [ln(8); 8] ~ [2.08; 8]

    // Base-10 logarithm: log10(x)
    let result = v.log10_midp();  // [log10(8); 8] ~ [0.90; 8]
}
```

## Power

```rust
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Convert,
};

#[inline(always)]
fn power<T: F32x8Convert>(token: T) {
    let base = f32x8::<T>::splat(token, 2.0);

    // x^n (computed as exp2(n * log2(x)))
    let result = base.pow_midp(3.0);  // [8.0; 8]
}
```

Note: `pow` takes a scalar exponent, not a vector.

## Roots

```rust
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Convert,
};

#[inline(always)]
fn roots<T: F32x8Convert>(token: T) {
    let v = f32x8::<T>::splat(token, 9.0);

    // Square root (hardware instruction on all platforms)
    let result = v.sqrt();            // [3.0; 8]

    // Cube root (polynomial approximation)
    let result = v.cbrt_midp();       // [cbrt(9); 8] ~ [2.08; 8]

    // Reciprocal square root: 1/sqrt(x)
    let result = v.rsqrt();           // [1/3; 8] ~ [0.33; 8]
}
```

`sqrt()` maps to a single hardware instruction (`vsqrtps` on x86, `fsqrt` on ARM). It's exact, not an approximation.

## Precision Variants

Most transcendentals come in multiple precision levels. See [Precision Levels](@/magetypes/math/precision.md) for the full breakdown.

```rust
// given token: T where T: F32x8Convert
let v = f32x8::<T>::splat(token, 2.0);

let fast     = v.exp2_lowp();   // ~12-bit precision, fastest
let balanced = v.exp2_midp();   // ~20-bit precision
```

There are no unsuffixed "full precision" transcendentals. `_midp` is the highest precision level for polynomial approximations. For exact results, use `sqrt()` (which is a hardware instruction, not an approximation).

## Domain Errors

Invalid inputs produce NaN or infinity, matching IEEE 754 behavior:

```rust
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Convert,
};

#[inline(always)]
fn domain_errors<T: F32x8Convert>(token: T) {
    let v = f32x8::<T>::from_array(token, [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // sqrt of negative -> NaN
    let sqrt = v.sqrt();   // [NaN, 0.0, 1.0, 1.41, 1.73, 2.0, 2.24, 2.45]

    // log of non-positive -> NaN or -inf
    let log = v.ln_midp(); // [NaN, -inf, 0.0, 0.69, 1.10, 1.39, 1.61, 1.79]
}
```

## Example: Gaussian Function

```rust
use archmage::{arcane, SimdToken};
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Convert,
};

#[arcane(import_intrinsics)]
fn gaussian<T: F32x8Convert>(token: T, x: &[f32; 8], sigma: f32) -> [f32; 8] {
    let v = f32x8::<T>::from_array(token, *x);
    let sigma_v = f32x8::<T>::splat(token, sigma);
    let two = f32x8::<T>::splat(token, 2.0);

    // exp(-x^2 / (2 * sigma^2))
    let x_sq = v * v;
    let two_sigma_sq = two * sigma_v * sigma_v;
    let exponent = -(x_sq / two_sigma_sq);
    let result = exponent.exp_midp();  // Good precision, fast

    result.to_array()
}
```

## Example: Softmax

```rust
use archmage::{arcane, SimdToken};
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Convert,
};

#[arcane(import_intrinsics)]
fn softmax<T: F32x8Convert>(token: T, logits: &[f32; 8]) -> [f32; 8] {
    let v = f32x8::<T>::from_array(token, *logits);

    // Subtract max for numerical stability
    let max = v.reduce_max();
    let shifted = v - f32x8::<T>::splat(token, max);

    // exp(x - max)
    let exp = shifted.exp_midp();

    // Normalize
    let sum = exp.reduce_add();
    let result = exp / f32x8::<T>::splat(token, sum);

    result.to_array()
}
```

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
