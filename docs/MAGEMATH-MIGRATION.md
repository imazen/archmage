# Choosing Magemath over the flat Magetypes approximations

[Magemath](../magemath) (new in the 0.9.32 workspace, crate version 0.1.0)
holds configurable f32 power, exponential, logarithm and cube-root
approximations. It depends on Magetypes vectors and Archmage proofs; Magetypes
does not depend on Magemath.

**Nothing in Magetypes changes or is deprecated by this crate.** The flat
methods published in 0.9.31 (`pow_midp`, `exp2_lowp`, `log2_midp`, `cbrt_midp`,
their `_unchecked` / `_precise` spellings, and the rest) keep their arithmetic
and remain supported. Migrate a kernel when you need what Magemath adds:

- an explicit accepted domain, checked or assumed, with whole-vector rejection
  (`Err` with a lane mask) instead of silently undefined results;
- a choice of precision (LOWP, MIDP, and HIGHP for pow) and of portability
  (bit-identical across backends, or backend-native arithmetic);
- scalar entry points that produce the same bits as the vector entry points for
  the same flags and token, so SIMD tails match the vector body.

## Equivalents

| Flat Magetypes method | Magemath family |
|---|---|
| `pow_lowp*`, `pow_midp*` | `magemath::powf::try_powf_f32xN_with` |
| `exp2_lowp*`, `exp2_midp*` | `magemath::expf::try_exp2_f32xN_with` |
| `exp_lowp*`, `exp_midp*` | `magemath::expf::try_exp_f32xN_with` |
| `log2_lowp*`, `log2_midp*` | `magemath::logf::try_log2_f32xN_with` |
| `ln_lowp*`, `ln_midp*` | `magemath::logf::try_ln_f32xN_with` |
| `log10_lowp*`, `log10_midp*` | `magemath::logf::try_log10_f32xN_with` |
| `log1p` compositions | `magemath::logf::try_log1p_f32xN_with` |
| `cbrt_lowp*`, `cbrt_midp*` | `magemath::cbrtf::try_cbrt_f32xN_with` |

`N` is 4 or 8, and 16 with the `w512` feature. Every family also has a plain
scalar entry (`try_powf_with::<FLAGS>(x, y)`), a token-preserving scalar entry
for tails (`try_powf_with_token::<FLAGS, _>(token, x, y)`), array entries
(`try_*_array_with`) and runtime-option entries (`try_*_runtime`).

A flat method's name does not map mechanically to a profile. An infallible
method accepts every input and returns unspecified values outside its range;
a checked profile rejects them. Choose the domain your code actually
establishes. `AssumeValid` validation removes the checks; it is safe Rust, but
inputs outside the stated domain give unspecified numbers.

## Profiles in brief

- **powf:** `PositiveNormal`, `NonNegativeNormal`, `Gamma`, `PositiveGamma`
  and `Full` (HIGHP) domains. MIDP/LOWP approximate `exp2(n * log2(x))`, so the
  error grows with the magnitude of the product. HIGHP evaluates a canonical
  binary64 graph; `STD_HIGHP` delegates to std and needs the `std` feature.
- **expf:** checked finite inputs reject NaN and infinity and keep the existing
  clamps. Exp2 MIDP's 1.9-ULP target applies below 127.5, not at the clamp rail.
- **logf:** `Positive` accepts positive normal finite inputs. `Specials` keeps
  the historical special-value results, including inaccurate subnormals. MIDP
  targets 4.5 ULP on normal inputs; LOWP has an absolute-error budget and large
  relative error near 1. LOWP log1p can return a negative value for tiny
  positive arguments; use MIDP there.
- **cbrtf:** `Bounded` handles `±0.0` and normal finite `|x| < f32::MAX / 3`,
  rejecting subnormals, larger magnitudes, infinities and NaN. `Full` rescales
  subnormals and large normals and propagates `±0.0`, `±inf` and NaN. LOWP uses
  one Halley step (≤ 259 ULP), MIDP two (≤ 3.2 ULP).

All guarantees assume round-to-nearest with gradual underflow. NaN payloads and
floating-point exception flags are not part of any contract.

## Keep the caller's token, including in tails

Scalar and array entries may select a backend once per call. In a typed kernel,
pass the token you already hold to the vector entry and use `*_with_token` for
the remainder with the same flags:

```rust
use archmage::ScalarToken;
use magetypes::simd::generic::f32x4;
use magemath::powf::{PowfExecution, PowfOptions, try_powf_f32x4_with, try_powf_with_token};

const FLAGS: u32 = PowfOptions::PORTABLE_GAMMA_MIDP
    .with_execution(PowfExecution::Caller)
    .bits();
let token = ScalarToken; // A native kernel passes its own token here.
let input = f32x4::from_array_t(token, [0.0, 0.25, 0.5, 1.0]);
let body = try_powf_f32x4_with::<FLAGS, _>(token, input, 2.2).unwrap();
let tail = try_powf_with_token::<FLAGS, _>(token, 0.5, 2.2).unwrap();
assert_eq!(body.to_array()[2].to_bits(), tail.to_bits());
```

Do not substitute `ScalarToken` for a native token in the tail of a
backend-portability profile: the scalar path may round differently.

## Features

Magemath is `no_std` by default. `std` enables runtime CPU detection for the
auto-selecting entries; `w512` enables 16-lane vectors; `avx512` adds the native
AVX-512 tiers and implies `w512`.
