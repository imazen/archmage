# zenmath

Scalar math with defined bits on every target: no dependencies, `no_std`, and
`#![forbid(unsafe_code)]`. It is the scalar half of `magemath`, for callers who need
portable math without SIMD types. Optional Archmage acceleration adds intrinsic-only
array/slice kernels without a Magetypes dependency; [design and current limits](docs/ACCELERATION.md).

**Status: draft, unpublished.**

| function | notes |
|---|---|
| `sqrtf`, `sqrt`, `floorf`, `floor`, `ceilf`, `ceil`, `roundf`, `round`, `roundevenf`, `roundeven` | correctly rounded, same bits as `std` on finite inputs |
| CORE-MATH f32/f64 exp, log, power, trigonometric and selected hyperbolic functions | 33 safe Rust ports; [coverage, assumptions and validation](docs/CORE_MATH.md) |
| `fmaf`, `fma` | fused multiply-add, correctly rounded; software except `std` builds with compile-time x86 FMA or AArch64 NEON |
| `log2_midp_f32`, `exp2_midp_f32`, `ln_midp_f32`, `exp_midp_f32`, `log10_midp_f32`, `pow_midp_f32`, `cbrt_midp_f32`, `cbrt_midp_precise_f32`, `sigmoid_midp_f32`, `silu_midp_f32` | scalar forms of `magemath::portable`, same bits on every tier |
| `log2_lowp_f32`, `exp2_lowp_f32`, `ln_lowp_f32`, `exp_lowp_f32`, `log10_lowp_f32`, `pow_lowp_f32`, `cbrt_lowp_f32` | same, at lower precision |
| `log2_lowp_f64`, `exp2_lowp_f64`, `ln_lowp_f64`, `exp_lowp_f64`, `log10_lowp_f64`, `pow_lowp_f64` | clamped approximations, not full-range libm replacements; no vector form yet |

The explicitly named `lowp`/`midp` functions are approximations with published bounds
([`docs/transcendentals.md`](../docs/transcendentals.md)), not libm replacements for
callers who need the correctly rounded value.

`scalar::unfused` is a draft mirror family using separate multiply/add for
polynomial steps. It needs no software FMA and is independent of the `std`
feature. It mirrors `magemath::unfused` and the two-rounding `relaxed` backends;
`scalar` continues to mirror fused `portable`. The final inclusion of `unfused`
is pending the rounding decision. Special-value rules and accuracy bounds are
shared; finite result bits can differ.

## Features

| feature | contents |
|---|---|
| *(default)* | `no_std` core, no dependencies |
| `std` | `fmaf` and `fma` use hardware `mul_add` with compile-time x86 FMA or AArch64 NEON; otherwise it retains software FMA. The results are identical; the feature changes only how they are computed |
| `archmage` | optional proof/compile-time-only accurate packed f32 exp/ln/log2/pow; no runtime probing from unifying archmage/std |
| `runtime-dispatch` | selective ordinary scalar acceleration and once-per-array/slice native selection; implies std and archmage/std |
| `avx512` | V4x proofs: four-input path shares V3 width; eight-input/wide paths use eight native f64 lanes |
| `special` | placeholder for erf, gamma and friends; empty today |
| `exact` | placeholder for bit-exact fdlibm and glibc ports; empty today |

The [existing implementation inventory](docs/INVENTORY.md) records the
zenmetrics-math ports, scalar special functions, versioned compatibility targets
and proposed extraction order.

## Draft compile-time power behavior

`PowBehavior` selects per-operation power semantics through validated const bits.
Existing power calls keep their results. Correctly rounded, Midp and Lowp policies
share scalar/packed behavior; the near-one-ULP prototype is not a supported preset.
See [the power behavior guide](../magemath/docs/POW_BEHAVIOR.md) for modes, precedence and packed entry points.

The [acceleration record](docs/ACCELERATION.md) lists exact ordinary names, proof APIs, measured policy limits and independent reference access.
