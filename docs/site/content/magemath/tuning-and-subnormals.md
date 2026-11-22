+++
title = "Tuning Performance & Subnormals"
description = "Why SIMD transcendentals pay extra for IEEE-754 subnormals, when inputs cannot be subnormal, and how to strip unneeded checks"
weight = 1
+++

Magemath separates **what inputs a kernel accepts** from **how out-of-bounds lanes behave**. Every check, clamp, or rescaling step you turn off disappears at compile time through `const` dead-code elimination on `const CFG: u128`.

## 1. What subnormals are (and why SIMD kernels care)

### IEEE-754 `f32` anatomy

An IEEE-754 `f32` packs a 1-bit sign, an 8-bit biased exponent (`(bits >> 23) & 0xff`), and a 23-bit fraction:

- **Normal numbers** have biased exponent `1..=254`, representing `(-1)^s × 1.fraction × 2^(e - 127)`. Every normal `f32` carries an implicit leading `1` bit in its significand. The smallest positive normal is `f32::MIN_POSITIVE = 2^-126 ≈ 1.1754944e-38` (`0x0080_0000`).
- **Subnormal (denormal) numbers** have biased exponent `0` and a nonzero fraction (`0x0000_0001..=0x007f_ffff`), representing `(-1)^s × 0.fraction × 2^-126` with **no** implicit leading `1`. They span `1.4012985e-45` (`2^-149`) up to `1.1754942e-38`, filling the underflow gap between `f32::MIN_POSITIVE` and `0.0` gradually instead of dropping straight to zero.

### Why fast SIMD transcendentals break on subnormals without extra work

Vector `log2`, `ln`, `log10`, `powf`, and `cbrt` extract the binary exponent directly from the float's magnitude bits.

- On a normal float, `abs_bits >> 23` gives the biased exponent in one vector shift.
- On **any** nonzero subnormal from `1.17e-38` down to `1.40e-45`, the 8-bit exponent field is `0`. `bits >> 23` reads `0` regardless of how many leading zeros sit in the 23-bit fraction, so an unrescaled `log2` polynomial treats `1e-42` (`≈ 2^-139.5`) as if it were near `2^-127` and returns `~ -127` instead of `-139.5`. (`exp2(x)` and `exp(x)` do not extract the input's exponent — for any subnormal `x`, `2^x ≈ 1 + x ln 2` rounds directly to `1.0` on `Midp`, so `ExpfOptions` needs no subnormal pre-pass.)

`logf` and `powf` multiply subnormal lanes by `2^23` before extracting the exponent and compensate for that shift afterward. `cbrtf` places the subnormal fraction bits at the exponent of `1.0`, then subtracts `1.0`. This gives the magnitude scaled by `2^126` exactly. The cube-root graph compensates by subtracting 42 from its output exponent. Vectors without zeros or subnormals skip this normalization.

### When you cannot have subnormals vs when you can

In most media, imaging, and ML pipelines, nonzero subnormals are impossible by representation:

- **`u8` pixels divided by `255.0`:** smallest nonzero value is `1.0 / 255.0 = 3.9216e-3` (`~2^-8`).
- **`u16` pixels divided by `65535.0`:** smallest nonzero value is `1.0 / 65535.0 = 1.5259e-5` (`~2^-16`, matching the `1.0 / 65_536.0` lower bound of `PowfDomain::SdrGamma`, `HdrGamma`, and `LogfDomain::Hdr`).
- **24-bit PCM audio (`2^-23 ≈ 1.192e-7`) or clamped linear light (`>= 1e-6`):** the smallest nonzero magnitude is more than 30 orders of magnitude above `f32::MIN_POSITIVE` (`1.175e-38`).

Subnormals **do** appear and are worth handling in:

- **Unthresholded exponential decay** (IIR audio filters or reverb tails decaying toward silence without a zero gate).
- **Products or ratios of unnormalized scientific quantities** (for example squaring `1e-20` before a log or cube root).
- **Arbitrary untrusted `f32` streams** where lossless IEEE classification across the entire `f32` bit space is part of the API contract (`*Domain::AllFloats` or `.admit_subnormals()`).

### Comparing the three `Subnormals` policies

`PowfOptions::new()`, `LogfOptions::new()`, `Log1pfOptions::new()`, and `CbrtfOptions::new()` all default to [`Subnormals::FlushOrClamp`](https://docs.rs/magemath/latest/magemath/enum.Subnormals.html) so checked evaluation does not fail if floating-point noise near zero produces a stray subnormal:

| Policy | Builder method | Behavior under `Validation::Checked` | Behavior under `AssumeValid` |
|---|---|---|---|
| `Subnormals::FlushOrClamp` *(default)* | `.flush_or_clamp_subnormals()` | Flushes to signed zero when the domain admits it, or clamps to the applicable normal bound | Caller supplies values within the post-flush domain |
| `Subnormals::Reject` | `.reject_subnormals()` | Rejects nonzero subnormals with an exact lane mask or slice index | Caller supplies normal values or admitted zeros |
| `Subnormals::Normalize` | `.admit_subnormals()` | Normalizes subnormals for accuracy down to `2^-149`. `log1pf` uses the identity at subnormal inputs | Keeps subnormal support |

`CbrtfDomain::Bounded` can flush subnormals during output reconstruction because it admits zero of either sign. For a custom interval such as `.input_range(0.01, 4.0)`, the bounds already rule out subnormals.

```rust
use magemath::cbrtf::{Cbrtf, CbrtfDomain, CbrtfErrorKind, CbrtfOptions};

// Default on SignedUnit is Subnormals::FlushOrClamp: flushes ±subnormal -> ±0.0
const FLUSH: u128 = CbrtfOptions::new()
    .domain(CbrtfDomain::SignedUnit)
    .bits();
const REJECT: u128 = CbrtfOptions::new()
    .domain(CbrtfDomain::SignedUnit)
    .reject_subnormals()
    .bits();
const NORMALIZE: u128 = CbrtfOptions::new()
    .domain(CbrtfDomain::SignedUnit)
    .admit_subnormals()
    .bits();

let sub = f32::from_bits(4); // 2^-147 ≈ 5.605e-45 (nonzero subnormal)

// 1. FlushOrClamp (default): maps -sub to -0.0 without failing Checked validation
assert_eq!(Cbrtf::<FLUSH>::scalar(-sub).unwrap().to_bits(), (-0.0_f32).to_bits());

// 2. Reject: flags nonzero subnormals as InvalidInput
assert_eq!(Cbrtf::<REJECT>::scalar(sub).unwrap_err().kind, CbrtfErrorKind::InvalidInput);

// 3. Normalize preserves the exact root of 2^-147.
let exact = Cbrtf::<NORMALIZE>::scalar(sub).unwrap();
let expected = f32::from_bits((127 - 49) << 23);
assert_eq!(exact.to_bits(), expected.to_bits());
```

---

## 2. Step-by-step knob guide for skipping unneeded checks

All timings below are whole-slice runs (`4,096` `f32` values, 16-lane AVX-512 `Execution::Upgrade`, no `-C target-cpu`) on AMD Ryzen 9 9950X3D from [`benchmarks/magemath_slices_2026-11-17/RESULTS.md`](https://github.com/imazen/archmage/blob/main/benchmarks/magemath_slices_2026-11-17/RESULTS.md) (`cb7c9580ca62`).

### Step 1: Narrow the `*Domain` (or `bf16` range) to elide reduction and clamp stages

Tightening the domain does more than change what `Checked` rejects — when the domain bounds prove an internal overflow clamp, integer `2^k` reconstruction, or Kahan branch is unreachable, Magemath drops those vector instructions at compile time:

- **`ExpfDomain::NormalOutput` (`[-126.0, 127.0]` for `exp2`, `[-87.0, 88.0]` for `exp`):** elides the internal `[-126, 128]` output underflow/overflow clamp (`343.2 ns` vs `464.3 ns` on `ExpfDomain::Finite`, **`-26.1%`**, `1.35x` faster).
  - *Watch out for the `127.5` endpoint:* `.input_range(-126.0, 127.5)` runs in **`570.3 ns`** (**`+22.8%` slower** than `Finite`). Because `Midp` splits `x` with `round_ties_even(x)`, `round_ties_even(127.5) == 128.0`, which would overflow `(xi + 127) << 23` if unclamped. Any upper bound `>= 127.5` forces the kernel to keep its internal `xi.min(127.0)` clamp *plus* the custom `[lo, hi]` range check. Keep custom `exp2` upper bounds strictly below `127.5` (note `.input_range(lo, hi)` rounds outward to `bf16` via `floor_bf16` / `ceil_bf16`).
- **`ExpfDomain::SingleOctave` (`[-0.5, 0.5]` for `exp2`, `[-0.34657356, 0.34657356]` for `exp`):** `round_ties_even(x)` is identically `0`, so the kernel drops `round`, `sub`, float-to-int conversion, bit-shift, and `2^k` scale multiplication completely — **`257.4 ns`** checked (**`1.80x` faster** than `Finite`) and **`181.0 ns`** unchecked (**`2.56x` faster**, `<= 0.78 ULP`).
- **`LogfDomain::SingleOctave` (`[√2/2, √2]` = `[0.70710677, 1.4142134]` on `Midp`):** the unbiased binary exponent is `0` across the entire interval, eliding integer exponent extraction and `i32 -> f32` conversion — **`318.2 ns`** checked (`1.35x` faster than `PositiveNormals` `429.2 ns`) and **`217.1 ns`** unchecked (**`1.98x` faster** than `PositiveNormals`, `2.65x` faster than `.admit_subnormals()`).
- **`Log1pfDomain::AroundZero` (`[-0.25, 0.25]`):** `1 + x` stays inside `[0.75, 1.25]`, eliding both integer exponent extraction and the `1 + x >= 2` Kahan compensation branch — **`533.5 ns`** checked (**`1.59x` faster** than `GreaterThanNegOne` `849.7 ns`) and **`347.5 ns`** unchecked (**`2.45x` faster**).
- **`PowfDomain` transfer presets & `.const_exponent(n)`:** `UnitEncode`, `SdrGamma` (`{+0.0} ∪ [2^-16, 1.0]`), `HdrGamma` (`{+0.0} ∪ [2^-16, 2^16]`), and `PqRatio` (`[107/128, 1.0]`, exponent `[0, 80]`) statically bound `|n * log2(x)| <= 125.0`, eliding per-lane `OutsideApproximationRange` checks (`830.4 ns` on `HdrGamma` vs `1.33 µs` on unranged `NonNegative`). Adding `.const_exponent(n)` removes runtime exponent validation and lowers algebraic exponents (`0, 0.25, ±0.5, ±1, 1.5, ±2, 3, 4`) directly to IEEE `sqrt`, `mul`, and `div` instructions (`251.8 ns` for `0.5`, `365.6 ns` for `2.0`).

```rust
use magemath::expf::{Expf, ExpfDomain, ExpfOptions};
use magemath::log1pf::{Log1pf, Log1pfDomain, Log1pfOptions};
use magemath::powf::{Powf, PowfDomain, PowfOptions};

// SingleOctave [-0.5, 0.5]: elides output clamps and integer 2^k reconstruction
const SMALL_EXP2: u128 = ExpfOptions::new()
    .domain(ExpfDomain::SingleOctave)
    .assume_in_bounds()
    .bits();

// AroundZero [-0.25, 0.25]: elides exponent extraction and Kahan 1+x >= 2 branch
const SMALL_LOG1P: u128 = Log1pfOptions::new()
    .domain(Log1pfDomain::AroundZero)
    .assume_in_bounds()
    .bits();

// Algebraic const_exponent(0.5): lowers directly to IEEE hardware sqrt
const UNIT_SQRT: u128 = PowfOptions::new()
    .domain(PowfDomain::Unit)
    .const_exponent(0.5)
    .bits();

assert_eq!(Expf::<SMALL_EXP2>::exp2_infallible(0.0), 1.0);
assert_eq!(Log1pf::<SMALL_LOG1P>::scalar_infallible(0.0), 0.0);
assert_eq!(Powf::<UNIT_SQRT>::scalar_const(0.25).unwrap(), 0.5);
```

### Step 2: Pick the lightest `Validation` mode your caller contract justifies

| `Validation` variant | Builder method | Return type | Out-of-bounds finite / `±inf` | `NaN` input | When to pick it |
|---|---|---|---|---|---|
| `Checked` *(default)* | `.error_on_out_of_bounds()` / `.checked()` | `Result<_, Error>` | Rejects vector (`lane_mask`) or slice (`index`) | Rejects | Untrusted external inputs or boundary assertions |
| `ClampOutOfBounds` | `.clamp_out_of_bounds()` | Infallible | Clamps into `[lo, hi]` (and saturates `Midp`/`Lowp` `powf` output) | Propagates `NaN -> NaN` | Noisy upstream floats (for example filter overshoot below `0.0` or above `1.0`) where `NaN` must remain `NaN` |
| `ClampFinite` | `.clamp_finite()` | Infallible | Same domain clamp as `ClampOutOfBounds` | Unspecified float (no UB) | Upstream already excluded `NaN` (for example decoded integer pixels + bicubic ringing), **skipping the `NaN` mask save and restore** |
| `AssumeValid` | `.assume_in_bounds()` / `.unchecked()` | Infallible | Unspecified float (never memory UB; `#![forbid(unsafe_code)]`) | Unspecified float | Caller just clamped or normalized the lanes (`20%`–`49%` faster than `Checked` across families) |

```rust
use archmage::ScalarToken;
use magetypes::simd::generic::f32x4;
use magemath::SimdMath;
use magemath::expf::{Expf, ExpfDomain, ExpfErrorKind, ExpfOptions};

const CHECKED: u128 = ExpfOptions::new().domain(ExpfDomain::SignedUnit).bits();
const CLAMP_NAN: u128 = ExpfOptions::new()
    .domain(ExpfDomain::SignedUnit)
    .clamp_out_of_bounds()
    .bits();
const CLAMP_FINITE: u128 = ExpfOptions::new()
    .domain(ExpfDomain::SignedUnit)
    .clamp_finite()
    .bits();

// Checked rejects out-of-bounds lanes and reports every failing lane in `lane_mask`:
let v = f32x4::from_array_t(ScalarToken, [0.0, 2.5, -0.5, f32::NAN]);
let err = v.try_exp2_with::<CHECKED>().unwrap_err();
assert_eq!(err.kind, ExpfErrorKind::InvalidInput);
assert_eq!(err.lane_mask, 0b1010); // lanes 1 and 3 failed [-1.0, 1.0]

// ClampOutOfBounds clamps 2.5 -> 1.0 (2^1 = 2.0) and preserves NaN -> NaN:
let clamped = v.exp2::<CLAMP_NAN>().to_array();
assert_eq!(clamped[1], 2.0);
assert!(clamped[3].is_nan());

// ClampFinite skips saving/restoring the NaN mask when inputs are known finite:
let finite = f32x4::from_array_t(ScalarToken, [-5.0, 0.0, 0.5, 5.0]);
let fast_clamped = finite.exp2::<CLAMP_FINITE>().to_array();
assert_eq!(fast_clamped[0], 0.5);
assert_eq!(fast_clamped[3], 2.0);
assert_eq!(Expf::<CLAMP_FINITE>::exp2_infallible(10.0), 2.0);
```

### Step 3: Choose `Portability::BitExact` vs `Portability::BackendDependent`

- **`Portability::BitExact` *(default, `.bit_exact()`)*:** evaluates one fused polynomial graph across every target so x86 AVX2, x86 AVX-512, AArch64 NEON, WASM SIMD128, and `ScalarToken` produce bit-for-bit identical accepted outputs. On x86 V3/V4 and AArch64 NEON, hardware FMA already rounds once in a single instruction, so **`BitExact` costs nothing extra on native desktop, server, and mobile SIMD tiers**. On `ScalarToken` (without compile-time FMA) and strict `Wasm128Token` (or `relaxed-simd` engines whose cached startup probe rounds twice), `BitExact` runs vectorized branchless software single-rounding FMA (`f64` product with round-to-odd adjustment on rare halfway/subnormal lanes).
- **`Portability::BackendDependent` *( `.backend_dependent()` )*:** evaluates polynomials with `mul_add_relaxed` (1-instruction FMA on x86 V3/V4 and NEON, plain `a * b + c` on `ScalarToken` and strict WASM, and platform `std` `f32::powf` on `Highp` `powf`). Switch to `.backend_dependent()` when 1–3 ULP cross-ISA drift on scalar/WASM fallbacks is acceptable and WASM or scalar fallback speed matters.

### Step 4: Choose `Precision` and `Execution`

- **Precision (`Lowp`, `Midp`, `Highp`):**
  - `Midp` *(default, `.midp()`)*: refitted degree-6 `exp2` (`<= 1.9 ULP` below `127.5`, `<= 0.78 ULP` on `SingleOctave`), degree-6 odd-rational `log2`/`ln`/`log10` (`<= 4.5 ULP` over positive normals, `<= 2.9 ULP` on `SingleOctave`), Kahan-compensated `log1p` (`<= 2.9 ULP` around zero), division-free Estrin `cbrt` (1.47 ULP maximum measured in `BitExact`), and 100% exact 8/10/12-bit sRGB round-trips on `powf`.
  - `Lowp` *( `.lowp()` )*: refitted degree-3 `exp2` (`<= 0.030%` relative error on `[-126, 127.99]`), rational `P2/Q2` `log2` (`~6e-6` absolute error), and division-free Horner `cbrt` (180.67 ULP maximum measured in `BitExact`). Note that `Lowp` `log1p` has a `~1.3e-6` polynomial intercept on tiny positive normals near zero (while preserving `0.0` and admitted subnormals), so keep `Midp` when tiny normal signs near zero matter.
  - `Highp` *( `.highp()`, `powf` only )*: table-based binary64 polynomial under `BitExact`, or `std` `f32::powf` under `BackendDependent` (requires `std`). Required for `PowfDomain::AllFloats`.
- **Execution (`Upgrade` vs `Caller`):**
  - `Execution::Upgrade` *(default, `.upgrade()`)*: upgrades once per operation when runtime CPU detection (`std`) exposes a wider or faster tier — notably widening `ScalarToken` and `X64V3Token` slice calls to 16-lane AVX-512 `zmm` loops when `avx512` is enabled, while keeping isolated 4/8-lane `Midp`/`Lowp` `V3` vector calls inline.
  - `Execution::Caller` *( `.caller()` )*: disables autonomous runtime tier upgrades so inline `SimdMath` vector calls and `*_with_token` scalar tails inside your own `#[attune]` loop stay locked to the caller's token.

Next: see [Magetypes Fast Paths & Contract Tiers](@/magemath/magetypes-fast-paths.md) for the same check-skipping discipline on primitive vector conversions, reciprocals, min/max/clamp, FMA reductions, and `#[attune]` boundaries.
