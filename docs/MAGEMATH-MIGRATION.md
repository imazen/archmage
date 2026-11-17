# Choosing Magemath over the flat Magetypes approximations

[Magemath](../magemath) (new in the 0.9.32 workspace, crate version 0.1.0)
holds configurable `f32` power, exponential, logarithm and cube-root
approximations. It depends on Magetypes vectors and Archmage proofs; Magetypes
does not depend on Magemath.

**Nothing in Magetypes changes or is deprecated by this crate.** The flat
methods published in 0.9.31 (`pow_midp`, `exp2_lowp`, `log2_midp`, `cbrt_midp`,
their `_unchecked` / `_precise` spellings, and the rest) keep their arithmetic
and remain supported. Migrate a kernel when you need what Magemath adds:

- an explicit accepted domain, checked or assumed, with whole-vector rejection
  (`Err` with a lane mask) or exact first-rejected slice index instead of
  silently undefined results;
- a choice of precision (`Lowp`, `Midp`, and `Highp` for `powf`), portability
  (`BitExact` across backends, or `BackendDependent` native arithmetic), and
  execution (`Execution::Upgrade` once per operation, or `Execution::Caller`);
- scalar and slice entry points that produce the same bits as the vector entry
  points for the same options and token, so SIMD tails match the vector body.

## Equivalents

| Flat Magetypes method | Inline vector (`SimdMath`) | Operator struct |
|---|---|---|
| `pow_lowp*`, `pow_midp*` | `v.try_powf_with::<CFG>(n)` / `v.powf::<CFG>(n)` | `magemath::powf::Powf::<CFG>` |
| `exp2_lowp*`, `exp2_midp*` | `v.try_exp2_with::<CFG>()` / `v.exp2::<CFG>()` | `magemath::expf::Expf::<CFG>` |
| `exp_lowp*`, `exp_midp*` | `v.try_exp_with::<CFG>()` / `v.exp::<CFG>()` | `magemath::expf::Expf::<CFG>` |
| `log2_lowp*`, `log2_midp*` | `v.try_log2_with::<CFG>()` / `v.log2::<CFG>()` | `magemath::logf::Logf::<CFG>` |
| `ln_lowp*`, `ln_midp*` | `v.try_ln_with::<CFG>()` / `v.ln::<CFG>()` | `magemath::logf::Logf::<CFG>` |
| `log10_lowp*`, `log10_midp*` | `v.try_log10_with::<CFG>()` / `v.log10::<CFG>()` | `magemath::logf::Logf::<CFG>` |
| `log1p` compositions | `v.try_log1p_with::<CFG>()` / `v.log1p::<CFG>()` | `magemath::logf::Logf::<CFG>` |
| `cbrt_lowp*`, `cbrt_midp*` | `v.try_cbrt_with::<CFG>()` / `v.cbrt::<CFG>()` | `magemath::cbrtf::Cbrtf::<CFG>` |

`SimdMath` works on `f32x4<T>`, `f32x8<T>`, and (with `w512`) `f32x16<T>`,
reusing the token already stored in the vector (`v.token()`). Every operator
struct (`Powf<CFG>`, `Expf<CFG>`, `Logf<CFG>`, `Cbrtf<CFG>`) also provides
scalar, infallible scalar, `*_with_token`, array (`[f32; N]`), slice,
`*_slice_with_token`, and `*_simd` methods, while `options.runtime()` provides
the matching dynamic runtime adapter. Default-option free helpers (`try_powf`,
`try_powf_slice`, `try_exp2`, `try_ln`, `try_cbrt`, ...) cover common defaults
without defining a `const`.

An infallible `SimdMath` method (`v.powf::<CFG>(n)`, `v.exp2::<CFG>()`, ...) or
`*_infallible` scalar method fails to compile unless `CFG` sets `.unchecked()`
or `.full_domain()`. Removing checks via `.unchecked()` (`Validation::AssumeValid`)
is safe Rust, but inputs outside the stated domain give unspecified numbers.

## Options builders in brief

- **powf (`PowfOptions`):** positive normal bases by default; `.admit_zero()`,
  `.gamma()` (`0` or `[2^-16, 2^16]`, exponent `[0, 4]`), `.positive_gamma()`,
  custom `.base_range(lo, hi)` and `.exponent_range(lo, hi)` (outward-rounded
  to 25-bit `f32` boundaries; elides range checks when `(E + 1) * N <= 120`),
  `.clamp_output()`, `.lowp()` / `.midp()` / `.highp()`, and `.full_domain().highp()`.
- **expf (`ExpfOptions`):** checked finite inputs reject NaN and infinity by
  default; `.full_domain()` propagates IEEE specials, and `.input_range(lo, hi)`
  elides underflow/overflow clamps when `[lo, hi]` stays inside `[-126, 127.5]`
  (or scaled by `LOG2_E` for `exp`).
- **logf (`LogfOptions`):** accepts positive normal finite inputs (`> -1` for
  `log1p`) by default; `.full_domain()` propagates IEEE specials (`0 -> -inf`,
  `< 0 -> NaN`, `+inf -> +inf`), and `.input_range(lo, hi)` validates `[lo, hi]`
  in one SIMD comparison pair. LOWP `log1p` can return a negative value for tiny
  positive arguments; use MIDP there.
- **cbrtf (`CbrtfOptions`):** handles `±0.0` and normal finite `|x| < f32::MAX / 3`
  by default, rejecting subnormals, larger magnitudes, infinities and NaN;
  `.full_domain()` rescales subnormals and large normals and propagates `±0.0`,
  `±inf` and NaN, while `.input_range(lo, hi)` elides the denormal check when the
  interval stays strictly normal away from zero.

All guarantees assume round-to-nearest with gradual underflow. NaN payloads and
floating-point exception flags are not part of any contract.

## Execution modes and keeping the caller's token in tails

`Execution::Upgrade` (the default, `.upgrade()`) upgrades once per operation when
runtime SIMD is wider or faster than the caller's token — including upgrading
`X64V3Token` to `X64V4xToken`/`X64V4Token` under `avx512` for slices (16-lane
`zmm` loop), 16-lane vectors, and 8-lane `Highp` `powf`, while keeping isolated
4/8-lane `Midp`/`Lowp` `V3` calls inline. Use `.caller()` (`Execution::Caller`)
inside a manual per-tier SIMD loop so body and tail stay on the exact same tier:

```rust
use archmage::ScalarToken;
use magetypes::simd::generic::f32x4;
use magemath::SimdMath;
use magemath::powf::{Powf, PowfOptions};

const GAMMA_CALLER: u128 = PowfOptions::new().gamma().midp().caller().bits();

let token = ScalarToken; // A native kernel passes its own token here.
let input = f32x4::from_array_t(token, [0.0, 0.25, 0.5, 1.0]);
let body = input.try_powf_with::<GAMMA_CALLER>(2.2).unwrap();
let tail = Powf::<GAMMA_CALLER>::scalar_with_token(input.token(), 0.5, 2.2).unwrap();
assert_eq!(body.to_array()[2].to_bits(), tail.to_bits());
```

Do not substitute `ScalarToken` for a native token in the tail of a
`.backend_dependent().caller()` configuration: the scalar path may round differently.

## Features

Magemath is `no_std` by default. `std` enables runtime CPU detection for
`Execution::Upgrade` and `std` `Highp` fallbacks; `w512` enables 16-lane vectors;
`avx512` adds native AVX-512 tiers and 16-lane slice kernels and implies `w512`.
