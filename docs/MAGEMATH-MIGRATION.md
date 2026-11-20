# Choosing Magemath over the flat Magetypes approximations

[Magemath](../magemath) (new in the 0.9.32 workspace, crate version 0.1.0)
holds configurable `f32` power, exponential, logarithm and cube-root
approximations. It depends on Magetypes vectors and Archmage proofs; Magetypes
does not depend on Magemath.

Full guides on the documentation site:

- [Magemath overview & API](https://imazen.github.io/archmage/magemath/)
- [Tuning Performance & Subnormals](https://imazen.github.io/archmage/magemath/tuning-and-subnormals/)
- [Magetypes Fast Paths & Contract Tiers](https://imazen.github.io/archmage/magemath/magetypes-fast-paths/)

## Two opposite configurations: why explicit domains pay off

On `4,096` normalized `f32` samples (`x^(1.0 / 2.4)` on AMD Ryzen 9 9950X3D, Zen 5, single core pinned via `taskset -c 8`, no `-C target-cpu`, 16-lane AVX-512 slice loop via `Execution::Upgrade`; full data in [`benchmarks/magemath_slices_2026-11-17/RESULTS.md`](../benchmarks/magemath_slices_2026-11-17/RESULTS.md)):

```rust
use magemath::powf::{Powf, PowfDomain, PowfOptions};

// 1. Broad defensive config: unranged [0, MAX], runtime exponent validation,
//    * 2^23 subnormal normalization, internal y*log2(x) range check -> 1.52 µs ± 2 ns
const DEFENSIVE: u128 = PowfOptions::new()
    .domain(PowfDomain::NonNegative)
    .admit_subnormals()
    .bits();

// 2. Tight domain-proven config: [0, 1] unit domain, compile-time exponent 1/2.4
//    (range-check elision proven at compile time), unchecked -> 666.5 ns ± 2 ns (2.28x faster)
const FAST: u128 = PowfOptions::new()
    .domain(PowfDomain::Unit)
    .const_exponent(1.0 / 2.4)
    .assume_in_bounds()
    .bits();

let src: Vec<f32> = (0..4096).map(|i| (i as f32) / 4095.0).collect();
let mut wide_out = vec![0.0_f32; 4096];
let mut fast_out = vec![0.0_f32; 4096];
Powf::<DEFENSIVE>::slice(&src, &mut wide_out, 1.0 / 2.4).unwrap();
Powf::<FAST>::slice_const(&src, &mut fast_out).unwrap();
assert_eq!(wide_out, fast_out);
```

Keeping `Validation::Checked` on `[0, 1]` with `.const_exponent(1.0 / 2.4).reject_subnormals()` still runs in **`827.1 ns ± 3 ns`** (**`1.84x` faster** while rejecting `< 0`, `> 1`, `±inf`, and `NaN` with exact index), and algebraic `.const_exponent(0.5)` lowers to `vsqrtps` in **`251.8 ns ± 1 ns`** (**`6.0x` faster**). Across all five families (`powf`, `exp2`, `log2`, `log1p`, `cbrt`), tightening domains and skipping unneeded checks yields **`1.72x` to `3.15x`** speedups (`2.28x` on `powf`, `2.56x`–`3.15x` on `exp2`, `2.65x` on `log2`, `2.45x` on `log1p`, `1.72x` on `cbrt`).

## When to keep Magetypes vs migrate to Magemath

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
(`.clamp_out_of_bounds()`, `.clamp_finite()`, `.assume_in_bounds()` / `.unchecked()`, or
`.admit_all_floats()` / `.full_domain()`). Removing checks via
`.assume_in_bounds()` / `.unchecked()` (`Validation::AssumeValid`) is safe Rust,
but inputs outside the stated domain give unspecified numbers; use
`.clamp_out_of_bounds()` (`Validation::ClampOutOfBounds`) when out-of-range
lanes should clamp into the configured domain and propagate NaN.
`.clamp_finite()` (`Validation::ClampFinite`) offers the same clamping for
finite and infinite numbers with unspecified NaN results.

## Options builders in brief

Every builder separates **what inputs are supported** (via discrete `#[non_exhaustive]`
per-family `*Domain` enums, custom outward-rounded `bf16` bounds, and `Subnormals`)
from **what happens out of bounds** (`.error_on_out_of_bounds()` / `.checked()`,
`.clamp_out_of_bounds()`, `.clamp_finite()`, `.assume_in_bounds()` / `.unchecked()`).
Subnormal rejection and normalization are explicit under bounded domains
(`.reject_subnormals()` and `.admit_subnormals()`). By default
`Subnormals::FlushOrClamp` moves nonzero subnormals to
`±0.0` (when `0` is admitted) or the lower normal bound (when `0` is excluded),
even under `Validation::Checked`. The all-floats domains keep their native
special-value and subnormal behavior.

- **powf (`PowfOptions`, `PowfDomain`):** `PositiveNormals` by default, plus
  `NonNegative` (`.admit_zero()`), `Unit`, `PositiveUnit`, `UnitEncode`
  (`.srgb_encode_domain()`), `SdrGamma` (`.unit_gamma_domain()`), `HdrGamma`
  (`.gamma()`), `PositiveGamma`, `PqRatio`, and `AllFloats` (`.full_domain().highp()`).
  Custom `.base_range(lo, hi)` and `.exponent_range(lo, hi)` pack outward-rounded
  `bf16` bounds (`MAX_BF16 = 0x7f7f_0000`). `.const_exponent(n: f32)` specializes
  for a compile-time exponent (`v.powf_const::<CFG>()`, `Powf::<CFG>::slice_const`,
  ...), eliminating runtime exponent validation, proving range-check elision at
  compile time when the configured domain bounds keep outputs normal, and lowering
  `0, 0.25, ±0.5, ±1, 1.5, ±2, 3, 4` to IEEE hardware multiply/sqrt/divide
  sequences (using `f64` intermediates for compound steps on `Highp`; checked
  `Midp`/`Lowp`/`Highp` still report `OutsideApproximationRange` when a positive
  base underflows or overflows `[MIN_POSITIVE, MAX]` and `elides_range_check()`
  is false, while preserving `+0.0` from an admitted `+0.0` base with `n > 0`).
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
  `.input_range(lo, hi)` (`|bound| <= MAX_CBRT_BF16 = 0x7eaa_0000`). Like `powf`,
  `logf`, and `log1pf`, `CbrtfOptions::new()` defaults to
  `Subnormals::FlushOrClamp` (flushing nonzero subnormals to `±0.0` when `0.0` is
  in range or to the nearest normal endpoint otherwise); `.reject_subnormals()`
  and `.admit_subnormals()` (`* 2^24` / `* 2^-8` exact rescaling) are explicit opt-ins.

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
