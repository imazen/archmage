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
| `pow_lowp*`, `pow_midp*` | `v.try_powf_with::<CFG>(n)` / `v.powf::<CFG>(n)` / `v.powf_const::<CFG>()` | `magemath::powf::Powf::<CFG>` |
| `exp2_lowp*`, `exp2_midp*` | `v.try_exp2_with::<CFG>()` / `v.exp2::<CFG>()` | `magemath::expf::Expf::<CFG>` |
| `exp_lowp*`, `exp_midp*` | `v.try_exp_with::<CFG>()` / `v.exp::<CFG>()` | `magemath::expf::Expf::<CFG>` |
| `log2_lowp*`, `log2_midp*` | `v.try_log2_with::<CFG>()` / `v.log2::<CFG>()` | `magemath::logf::Logf::<CFG>` |
| `ln_lowp*`, `ln_midp*` | `v.try_ln_with::<CFG>()` / `v.ln::<CFG>()` | `magemath::logf::Logf::<CFG>` |
| `log10_lowp*`, `log10_midp*` | `v.try_log10_with::<CFG>()` / `v.log10::<CFG>()` | `magemath::logf::Logf::<CFG>` |
| `log1p` compositions | `v.try_log1p_with::<CFG>()` / `v.log1p::<CFG>()` | `magemath::log1pf::Log1pf::<CFG>` |
| `cbrt_lowp*`, `cbrt_midp*` | `v.try_cbrt_with::<CFG>()` / `v.cbrt::<CFG>()` | `magemath::cbrtf::Cbrtf::<CFG>` |

`SimdMath` works on `f32x4<T>`, `f32x8<T>`, and (with `w512`) `f32x16<T>`,
reusing the token already stored in the vector (`v.token()`). Every operator
struct (`Powf<CFG>`, `Expf<CFG>`, `Logf<CFG>`, `Log1pf<CFG>`, `Cbrtf<CFG>`) also
provides scalar, infallible scalar, `*_with_token`, array (`[f32; N]`), slice,
`*_slice_with_token`, and `*_simd` methods (plus `*_const` methods on `Powf<CFG>`
and `PowfRuntime` when `PowfOptions::const_exponent(n)` is set), while
`options.runtime()` provides the matching dynamic runtime adapter. Default-option
free helpers (`try_powf`, `try_powf_slice`, `try_exp2`, `try_ln`, `try_log1p`,
`try_cbrt`, ...) cover common defaults without defining a `const`.

An infallible `SimdMath` method (`v.powf::<CFG>(n)`, `v.powf_const::<CFG>()`,
`v.exp2::<CFG>()`, ...) or
`*_infallible` scalar method fails to compile unless `CFG` is infallible
(`.clamp_out_of_bounds()`, `.assume_in_bounds()` / `.unchecked()`, or
`.admit_all_floats()` / `.full_domain()`). Removing checks via
`.assume_in_bounds()` / `.unchecked()` (`Validation::AssumeValid`) is safe Rust,
but inputs outside the stated domain give unspecified numbers; use
`.clamp_out_of_bounds()` (`Validation::ClampOutOfBounds`) when out-of-range
lanes should clamp safely into the configured domain (`NaN -> NaN`).

## Options builders in brief

Every builder separates **what inputs are supported** (via discrete `#[non_exhaustive]`
per-family `*Domain` enums, custom outward-rounded `bf16` bounds, and `Subnormals`)
from **what happens out of bounds** (`.error_on_out_of_bounds()` / `.checked()`,
`.clamp_out_of_bounds()`, `.assume_in_bounds()` / `.unchecked()`). Subnormal
rescaling (`Subnormals::Normalize`, `.admit_subnormals()`) is opt-in across all
bounded domains (`Subnormals::Reject` by default); `.flush_or_clamp_subnormals()` /
`.shove_subnormals()` (`Subnormals::FlushOrClamp`) shoves nonzero subnormals to
`±0.0` (when `0` is admitted) or the lower normal bound (when `0` is excluded),
even under `Validation::Checked`:

- **powf (`PowfOptions`, `PowfDomain`):** `PositiveNormals` by default, plus
  `NonNegative` (`.admit_zero()`), `Unit`, `PositiveUnit`, `UnitEncode`
  (`.srgb_encode_domain()`), `SdrGamma` (`.unit_gamma_domain()`), `HdrGamma`
  (`.gamma()`), `PositiveGamma`, `PqRatio`, and `AllFloats` (`.full_domain().highp()`).
  Custom `.base_range(lo, hi)` and `.exponent_range(lo, hi)` pack outward-rounded
  `bf16` bounds (`MAX_BF16 = 0x7f7f_0000`). `.const_exponent(n: f32)` specializes
  for a compile-time exponent (`v.powf_const::<CFG>()`, `Powf::<CFG>::slice_const`,
  ...), eliminating runtime exponent validation, proving range-check elision at
  compile time when the configured domain bounds keep outputs normal, and lowering
  `0, 0.25, ±0.5, ±1, 1.5, ±2, 3, 4` to exact IEEE hardware multiply/sqrt/divide
  sequences (`f64`-promoted on `Highp`; checked `Midp`/`Lowp`/`Highp` still report
  `OutsideApproximationRange` when a result leaves the normal output range and
  `elides_range_check()` is false).
- **expf (`ExpfOptions`, `ExpfDomain`):** `Finite` by default (accepting `±0.0` and
  subnormals without extra rescaling: `1.0` on `Midp`, and within `Lowp`'s
  `<= 0.030%` relative-error envelope on `Lowp`), plus `NormalOutput`
  (`.unclamped_exp2_range()` / `.unclamped_exp_range()`), `NonPositive`
  (`[-MAX, 0.0]`, elides upper overflow clamp), `SignedUnit`, `SingleOctave`
  (`.small_exp2_range()` / `.small_exp_range()`, elides integer `2^k` reduction),
  `PositiveOctave`, `AllFloats`, and custom `bf16` `.input_range(lo, hi)`.
- **logf (`LogfOptions`, `LogfDomain`):** `PositiveNormals` by default, plus
  `NonNegative` (`.admit_zero()`), `Unit`, `PositiveUnit`, `Hdr`, `PositiveHdr`,
  `SingleOctave` (`.near_one_range()`, elides exponent extraction), `AllFloats`,
  and custom `bf16` `.input_range(lo, hi)`.
- **log1pf (`Log1pfOptions`, `Log1pfDomain`):** dedicated `pub mod log1pf` for
  `ln(1 + x)` centered at `0` with singularity at `-1`: `GreaterThanNegOne` by
  default, plus `FromNegOne` (`.admit_neg_one()`), `NonNegative`, `Unit`,
  `OpenSignedUnit`, `SignedUnit`, `AroundZero` (`[-0.25, 0.25]`, elides exponent
  extraction and `1 + x >= 2` Kahan branch), `SingleOctave`, `AllFloats`, and
  `bf16` `.input_range(lo, hi)`. Both `Midp` and `Lowp` preserve `ln(1 + x) == x`
  on admitted zeros and subnormals; on tiny positive normal inputs, LOWP `log1p`
  has ~`1.3e-6` polynomial intercept near `0`, so prefer MIDP when exact small-`x`
  normal sign preservation matters.
- **cbrtf (`CbrtfOptions`, `CbrtfDomain`):** `Bounded` (`±0.0` and normal
  `|x| < f32::MAX / 3`) by default, plus `Unit` (`[0, 1]`), `SignedUnit`
  (`.unit_range()`, `[-1, 1]`), `PositiveNormals`, `AllFloats`, and custom `bf16`
  `.input_range(lo, hi)` (`|bound| <= MAX_CBRT_BF16 = 0x7eaa_0000`). Subnormals
  remain opt-in on `Unit` and `SignedUnit` unless `.admit_subnormals()` or
  `.flush_or_clamp_subnormals()` is enabled.

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
