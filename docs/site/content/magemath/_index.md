+++
title = "Magemath"
description = "Configurable scalar and SIMD transcendental approximations with explicit domain and subnormal contracts"
sort_by = "weight"
weight = 3

[extra]
sidebar = true
+++

`magemath` (`0.1`) provides configurable `f32` `powf`, `expf` (`exp2`, `exp`), `logf` (`log2`, `ln`, `log10`), `log1pf` (`log1p`), and `cbrtf` (`cbrt`) approximations built on [Magetypes](@/magetypes/_index.md) vectors and [Archmage](@/attunement/_index.md) capability proofs.

## Two opposite configurations: `1.52 µs` vs `666.5 ns`

Consider encoding `4,096` normalized linear-light samples (`x in [0, 1]`) with an sRGB-like power exponent `1.0 / 2.4`. If the kernel cannot assume anything about the caller's range or exponent, it has to validate the exponent at runtime, normalize subnormals on the critical path, check `y * log2(x)` against approximation overflow/underflow bounds, and check the whole slice for errors. Stating the actual domain and constant exponent removes all of that work at compile time:

```rust
use archmage::ScalarToken;
use magetypes::simd::generic::f32x8;
use magemath::SimdMath;
use magemath::powf::{Powf, PowfDomain, PowfOptions};

// 1. Broad defensive configuration:
//    Unranged [0, MAX] domain, runtime exponent validation, * 2^23 subnormal
//    normalization on the critical path, internal y * log2(x) output range check,
//    and whole-slice error checking.
//    Zen 5 (Ryzen 9 9950X3D, 16-lane AVX-512 slice): 1.52 µs ± 2 ns / 4,096 f32
const DEFENSIVE: u128 = PowfOptions::new()
    .domain(PowfDomain::NonNegative)
    .admit_subnormals()
    .bits();

// 2. Tight domain-proven configuration:
//    [0, 1] unit domain, compile-time constant exponent 1/2.4 (zero runtime
//    exponent prep, static output range-check elision), subnormals skipped
//    under AssumeValid, and infallible vector evaluation.
//    Zen 5 (Ryzen 9 9950X3D, 16-lane AVX-512 slice): 666.5 ns ± 2 ns / 4,096 f32 (2.28x faster)
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

// Inline on a Magetypes vector, infallible `powf_const` returns `f32x8` directly:
let chunk = f32x8::from_slice_t(ScalarToken, &src[..8]);
assert_eq!(&chunk.powf_const::<FAST>().to_array(), &fast_out[..8]);
```

You do not have to jump straight to unchecked execution to collect most of the speedup:

- Keeping `Validation::Checked` on `[0, 1]` with `.const_exponent(1.0 / 2.4)` and `.reject_subnormals()` runs in **`827.1 ns ± 3 ns`** (**`1.84x` faster** than `DEFENSIVE`, `-46%` time) while still rejecting any `< 0`, `> 1`, subnormal, `±inf`, or `NaN` lane with its exact slice index.
- Algebraic constant exponents bypass the `exp2(n * log2(x))` polynomial graph entirely: checked `.domain(PowfDomain::Unit).const_exponent(0.5)` lowers to hardware `vsqrtps` in **`251.8 ns ± 1 ns`** (**`6.0x` faster**), and `.const_exponent(2.0)` lowers to `vmulps` in **`365.6 ns ± 2 ns`**.

Across all five families, narrowing the domain and dropping unneeded runtime repairs cuts slice execution time by `1.72x` to `3.15x` on AMD Ryzen 9 9950X3D (Zen 5, single core pinned via `taskset -c 8`, release build without `-C target-cpu`, `magemath` `avx512` enabled for 16-lane `zmm` slice loops, `4,096` `f32` values per call; full data in [`benchmarks/magemath_slices_2026-11-17/RESULTS.md`](https://github.com/imazen/archmage/blob/main/benchmarks/magemath_slices_2026-11-17/RESULTS.md)):

| Family | Broad / IEEE configuration | Time (`4,096` `f32`) | Tight / unchecked configuration | Time (`4,096` `f32`) | Speedup |
|---|---|---:|---|---:|---:|
| `powf` (`x^(1/2.4)`) | `NonNegative + admit_subnormals()` | `1.52 µs ± 2 ns` | `Unit + const_exponent(1.0 / 2.4) + assume_in_bounds()` | `666.5 ns ± 2 ns` | **`2.28x`** (`-56%`) |
| `exp2` (`2^x`) | `Finite` (`464.3 ns`) / `.input_range(-126.0, 127.5)` (`570.3 ns`) | `464.3`–`570.3 ns` | `SingleOctave + assume_in_bounds()` | `181.0 ns ± 2 ns` | **`2.56x`–`3.15x`** |
| `log2` (`log2(x)`) | `PositiveNormals + admit_subnormals()` | `575.2 ns ± 3 ns` | `SingleOctave + assume_in_bounds()` | `217.1 ns ± 1 ns` | **`2.65x`** (`-62%`) |
| `log1p` (`ln(1 + x)`) | `GreaterThanNegOne` | `849.7 ns ± 4 ns` | `AroundZero + assume_in_bounds()` | `347.5 ns ± 1 ns` | **`2.45x`** (`-59%`) |
| `cbrt` (`cbrt(x)`) | `AllFloats` | `1.10 µs ± 2 ns` | `SignedUnit + assume_in_bounds()` | `640.3 ns ± 5 ns` | **`1.72x`** (`-42%`) |

## Guide structure

1. [Tuning Performance & Subnormals](@/magemath/tuning-and-subnormals.md) — IEEE-754 subnormal anatomy, why SIMD bit-extraction polynomials pay `+31.5%` to `+41.5%` to normalize subnormals, when inputs cannot be subnormal by representation, and a step-by-step knob guide across domains, validation, portability, and precision.
2. [Magetypes Fast Paths & Contract Tiers](@/magemath/magetypes-fast-paths.md) — The matching contract-relaxing methods directly on `magetypes` vectors (`to_i32_fast`, `recip`/`rcp_approx`, `min_fast`/`clamp`, `mul_add_relaxed` vs `a * b + acc`, flat `*_unchecked` transcendentals, and `#[attune(wrap, suppress_const_test)]`).

## Installation and Cargo features

Pin matching `archmage` and `magetypes` versions alongside `magemath`:

```toml
[dependencies]
archmage = "=0.9.32"
magetypes = "=0.9.32"
magemath = { version = "0.1", features = ["std"] }
```

`magemath` is `#![no_std]` and `#![forbid(unsafe_code)]` by default:

- `std`: enables runtime CPU feature detection for `Execution::Upgrade` and `std` `Highp` `powf` fallbacks.
- `w512`: implements `SimdVector` / `SimdMath` and array width `16` for `f32x16<T>`.
- `avx512`: implies `w512` and enables native 16-lane AVX-512 `zmm` slice loops and `V3 -> V4/V4x` runtime widening.

## How Magemath is organized

Each function family exposes a fluent `const` options builder, a const-generic operator struct specialized by `const CFG: u128`, a `.runtime()` dynamic adapter, inline vector methods on [`SimdMath`](https://docs.rs/magemath/latest/magemath/trait.SimdMath.html), and default-option free functions:

| Module | Operations | Builder & domain enum | Const operator & runtime adapter | Default helpers |
|---|---|---|---|---|
| [`magemath::powf`](https://docs.rs/magemath/latest/magemath/powf/index.html) | `x^n`, `x^CONST` | `PowfOptions`, `PowfDomain` | `Powf<CFG>`, `PowfRuntime` | `try_powf`, `try_powf_slice` |
| [`magemath::expf`](https://docs.rs/magemath/latest/magemath/expf/index.html) | `exp2(x)`, `exp(x)` | `ExpfOptions`, `ExpfDomain` | `Expf<CFG>`, `ExpfRuntime` | `try_exp2`, `try_exp2_slice`, `try_exp`, `try_exp_slice` |
| [`magemath::logf`](https://docs.rs/magemath/latest/magemath/logf/index.html) | `log2(x)`, `ln(x)`, `log10(x)` | `LogfOptions`, `LogfDomain` | `Logf<CFG>`, `LogfRuntime` | `try_log2`, `try_log2_slice`, `try_ln`, `try_ln_slice`, `try_log10`, `try_log10_slice` |
| [`magemath::log1pf`](https://docs.rs/magemath/latest/magemath/log1pf/index.html) | `ln(1 + x)` | `Log1pfOptions`, `Log1pfDomain` | `Log1pf<CFG>`, `Log1pfRuntime` | `try_log1p`, `try_log1p_slice` |
| [`magemath::cbrtf`](https://docs.rs/magemath/latest/magemath/cbrtf/index.html) | `cbrt(x)` | `CbrtfOptions`, `CbrtfDomain` | `Cbrtf<CFG>`, `CbrtfRuntime` | `try_cbrt`, `try_cbrt_slice` |

Option builders pack all settings into a `u128` word via `.bits()`, carrying a per-family tag so passing an `ExpfOptions` word to `Powf<CFG>` fails at compile time. When options depend on runtime configuration (such as file headers or user parameters), call `.runtime()` on the same builder instead of `.bits()`:

```rust
use magemath::expf::{Expf, ExpfDomain, ExpfOptions, try_exp2};

// Default checked Midp BitExact helper:
assert_eq!(try_exp2(3.0).unwrap(), 8.0);

// Compile-time specialization via `.bits()`:
const NORMAL_OUT: u128 = ExpfOptions::new()
    .domain(ExpfDomain::NormalOutput)
    .bits();
assert_eq!(Expf::<NORMAL_OUT>::exp2(-2.0).unwrap(), 0.25);

// Dynamic specialization via `.runtime()` on the same option type:
let rt = ExpfOptions::new().domain(ExpfDomain::SingleOctave).runtime();
let mut dst = [0.0_f32; 4];
rt.exp2_slice(&[-0.5, 0.0, 0.25, 0.5], &mut dst).unwrap();
assert_eq!(dst[1], 1.0);
```

## Which entry point should you call?

Every operator struct (`Powf<CFG>`, `Expf<CFG>`, `Logf<CFG>`, `Log1pf<CFG>`, `Cbrtf<CFG>`) provides scalar (`scalar` / `exp2` / `ln`), infallible scalar (`*_infallible`), token-preserving scalar (`*_with_token`), fixed-array (`array::<N>` for `N` in `1, 4, 8, 16`), whole-slice (`slice` / `exp2_slice` / `ln_slice` and `*_slice_with_token`), and vector (`simd`) entry points.

| Work | Entry point |
| --- | --- |
| Fused pixel or row loop | Import `SimdMath` and select `.caller()`. Use `try_powf_with`, `try_powf_const`, `try_exp2_with`, `try_log2_with`, `try_log1p_with`, or `try_cbrt_with` for checked options. Start with four lanes on WASM, eight on V3 or NEON, and sixteen on V4/V4x. Match scalar tails with `*_with_token`, or load a padded tail and store its live prefix. |
| Standalone buffer, LUT, or plane pass | Use `Powf::<CFG>::slice` for a runtime exponent, `slice_const` for a configured exponent, and the sibling `*_slice` methods. Each call selects a kernel once. `Execution::Upgrade` can widen to 16 lanes with `avx512`. |
| One value or a cold scalar path | Use `*_with_token` with `.caller()`. Pass the held proof for a matching tail or `ScalarToken` for a scalar-only entry. |
| Several gamma powers | `Highp` scalar slices share a kernel across exponents with the same structural options. For `Midp`, use a runtime exponent with `slice` if you want to keep a shared entry. Use `slice_const` when a fixed exponent pays for its specialization. |

For exact arithmetic on strict or unfused WASM, see the [WASM portability choices](@/magemath/tuning-and-subnormals.md#wasm-portability-backenddependent-vs-bitexact).

When transforming a buffer, call the **slice entry point** rather than looping over `array::<8>` chunks. Under the default `Execution::Upgrade`, a slice call probes CPU features once at the entry boundary and runs a 16-lane AVX-512 `zmm` loop (when `avx512` is enabled) plus a padded remainder pass. Calling `array::<8>` inside a caller loop re-runs dispatch selection and array marshaling every 8 floats.

On Ryzen 9 9950X3D (`len = 4096`, inputs in `[0.01, 3.91)`, [`RESULTS.md`](https://github.com/imazen/archmage/blob/main/benchmarks/magemath_slices_2026-11-17/RESULTS.md) commit `d538e9bd06f2`), slice entry points beat per-chunk `array::<8>` calls by **`2.7x` to `19.2x`** (`927.8 ns` vs `10.35 µs` on `pow`, `519.2 ns` vs `9.96 µs` on `exp2`, `497.0 ns` vs `1.33 µs` on `log2`), and beat an 8-lane `#[attune]` loop by `1.48x` to `2.35x` because `Execution::Upgrade` widens the slice loop to 16 lanes. At the 8-lane Zen 4 baseline (`1cb042ba3741`, Ryzen 9 7900X without `avx512`), slice entries and hand-written 8-lane `#[attune]` loops are within `3%` of each other while beating per-chunk `array::<8>` calls by `4.6x` to `12x`.

## Inline vector kernels and matching scalar tails

In a fused SIMD loop, import [`magemath::SimdMath`](https://docs.rs/magemath/latest/magemath/trait.SimdMath.html). Each method reuses the proof carried inside the vector. Use `.caller()` and the same token and options for the scalar tail. This example shows the eight-lane loop.

```rust
use archmage::prelude::*;
use magemath::SimdMath;
use magemath::powf::{Powf, PowfDomain, PowfError, PowfOptions};

const HDR_CALLER: u128 = PowfOptions::new()
    .domain(PowfDomain::HdrGamma)
    .caller()
    .bits();

#[attune(_*_t, define(f32x8))]
fn apply_hdr_gamma_tier(
    token: Token,
    src: &[f32],
    dst: &mut [f32],
    gamma: f32,
) -> Result<(), PowfError> {
    assert_eq!(src.len(), dst.len());
    let (sc, st) = f32x8::partition_slice_t(token, src);
    let (dc, dt) = f32x8::partition_slice_mut_t(token, dst);
    for (s, d) in sc.iter().zip(dc.iter_mut()) {
        f32x8::load_t(token, s).try_powf_with::<HDR_CALLER>(gamma)?.store(d);
    }
    for (&x, y) in st.iter().zip(dt.iter_mut()) {
        *y = Powf::<HDR_CALLER>::scalar_with_token(token, x, gamma)?;
    }
    Ok(())
}

pub fn apply_hdr_gamma(src: &[f32], dst: &mut [f32], gamma: f32) -> Result<(), PowfError> {
    attuned!(apply_hdr_gamma_tier(src, dst, gamma))
}

let src = [0.5_f32; 11]; // 1 full 8-lane chunk + 3 tail lanes
let mut dst = [0.0_f32; 11];
apply_hdr_gamma(&src, &mut dst, 2.2).unwrap();
assert_eq!(dst[0].to_bits(), dst[10].to_bits());
```

## Compile-time contract guardrails

Infallible `SimdMath` methods (`v.powf::<CFG>(n)`, `v.powf_const::<CFG>()`, `v.exp2::<CFG>()`, `v.ln::<CFG>()`, `v.log1p::<CFG>()`, `v.cbrt::<CFG>()`) and `*_infallible` scalar methods assert `opts.is_infallible()` in a `const` block. Passing a checked configuration fails at compile time instead of unwrapping at runtime:

```rust,compile_fail
use archmage::ScalarToken;
use magetypes::simd::generic::f32x4;
use magemath::{SimdMath, powf::PowfOptions};

const CHECKED: u128 = PowfOptions::new().gamma().bits();
let v = f32x4::splat_t(ScalarToken, 0.5);
// Compile error: CHECKED uses Validation::Checked; use `try_powf_with` or make CFG infallible.
let _ = v.powf::<CHECKED>(2.2);
```

Compile-time checks also reject calling `powf::<CFG>(n)` when `CFG` sets `.const_exponent(...)` (preventing a runtime argument from being silently ignored), calling `*_const` without `.const_exponent(...)`, or combining `PowfDomain::AllFloats` with `Midp`/`Lowp` instead of `Highp`.
