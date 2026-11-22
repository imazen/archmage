# magemath [![CI](https://img.shields.io/github/actions/workflow/status/imazen/archmage/ci.yml?style=flat-square&label=CI)](https://github.com/imazen/archmage/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/magemath?style=flat-square)](https://crates.io/crates/magemath) [![lib.rs](https://img.shields.io/crates/v/magemath?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/magemath) [![docs.rs](https://img.shields.io/docsrs/magemath?style=flat-square)](https://docs.rs/magemath) [![MSRV](https://img.shields.io/badge/MSRV-1.89-blue?style=flat-square)](https://github.com/imazen/archmage/blob/main/MSRV.md) [![license](https://img.shields.io/crates/l/magemath?style=flat-square)](#license)

[Guide](https://imazen.github.io/archmage/magemath/) · [Tuning & Subnormals](https://imazen.github.io/archmage/magemath/tuning-and-subnormals/) · [Magemath API](https://docs.rs/magemath/latest/magemath/)

`magemath` is a configurable scalar and SIMD `f32` transcendental approximation crate (`powf`, `exp2`, `exp`, `log2`, `ln`, `log10`, `log1p`, `cbrt`), built on [Magetypes](https://crates.io/crates/magetypes) vectors and [Archmage](https://crates.io/crates/archmage) capability proofs. `#![no_std]` and `#![forbid(unsafe_code)]` by default; enable `std` for runtime CPU detection and `avx512` for 16-lane AVX-512 slice widening.

## Two opposite configurations: `1.52 µs` vs `666.5 ns`

Stating your actual input domain and exponent removes subnormal rescaling, runtime exponent validation, and internal range clamps at compile time:

```rust
use archmage::ScalarToken;
use magetypes::simd::generic::f32x8;
use magemath::SimdMath;
use magemath::powf::{Powf, PowfDomain, PowfOptions};

// 1. Broad defensive config: unranged [0, MAX], runtime exponent validation,
//    * 2^23 subnormal normalization, internal y*log2(x) range check, error reporting.
//    Zen 5 (Ryzen 9 9950X3D, 16-lane AVX-512 slice): 1.52 µs ± 2 ns / 4,096 f32
const DEFENSIVE: u128 = PowfOptions::new()
    .domain(PowfDomain::NonNegative)
    .admit_subnormals()
    .bits();

// 2. Tight domain-proven config: [0, 1] unit domain, compile-time constant 1/2.4
//    (proving output range-check elision at compile time), unchecked in-bounds.
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

let chunk = f32x8::from_slice_t(ScalarToken, &src[..8]);
assert_eq!(&chunk.powf_const::<FAST>().to_array(), &fast_out[..8]);
```

Keeping `Validation::Checked` on `[0, 1]` with `.const_exponent(1.0 / 2.4).reject_subnormals()` runs in **`827.1 ns ± 3 ns`** (**`1.84x` faster** while rejecting `< 0`, `> 1`, `±inf`, or `NaN` with exact index), and `.const_exponent(0.5)` lowers to `vsqrtps` in **`251.8 ns ± 1 ns`** (**`6.0x` faster**).

Measured whole-slice throughput (`4,096` `f32` values, AMD Ryzen 9 9950X3D Zen 5, single core pinned via `taskset -c 8`, release build without `-C target-cpu`, `avx512` enabled; methodology and full matrix in [`benchmarks/magemath_slices_2026-11-17/RESULTS.md`](https://github.com/imazen/archmage/blob/main/benchmarks/magemath_slices_2026-11-17/RESULTS.md)):

| Family | Broad / IEEE configuration | Time (`4,096` `f32`) | Tight / unchecked configuration | Time (`4,096` `f32`) | Speedup |
|---|---|---:|---|---:|---:|
| `powf` (`x^(1/2.4)`) | `NonNegative + admit_subnormals()` | `1.52 µs ± 2 ns` | `Unit + const_exponent(1.0 / 2.4) + assume_in_bounds()` | `666.5 ns ± 2 ns` | **`2.28x`** |
| `exp2` (`2^x`) | `Finite` (`464.3 ns`) / `.input_range(-126.0, 127.5)` (`570.3 ns`) | `464.3`–`570.3 ns` | `SingleOctave + assume_in_bounds()` | `181.0 ns ± 2 ns` | **`2.56x`–`3.15x`** |
| `log2` (`log2(x)`) | `PositiveNormals + admit_subnormals()` | `575.2 ns ± 3 ns` | `SingleOctave + assume_in_bounds()` | `217.1 ns ± 1 ns` | **`2.65x`** |
| `log1p` (`ln(1 + x)`) | `GreaterThanNegOne` | `849.7 ns ± 4 ns` | `AroundZero + assume_in_bounds()` | `347.5 ns ± 1 ns` | **`2.45x`** |
| `cbrt` (`cbrt(x)`) | `AllFloats` | `1.10 µs ± 2 ns` | `SignedUnit + assume_in_bounds()` | `640.3 ns ± 5 ns` | **`1.72x`** |

## Quick start

```toml
[dependencies]
archmage = "=0.9.32"
magetypes = "=0.9.32"
magemath = { version = "0.1", features = ["std"] }
```

```rust
use magemath::powf::{Powf, PowfOptions, try_powf_slice};

const GAMMA: u128 = PowfOptions::new().gamma().midp().bits();

let result = Powf::<GAMMA>::array([0.0, 0.125, 0.5, 1.0], 2.2).unwrap();
assert_eq!(result[0], 0.0);
assert_eq!(result[3], 1.0);

let src = [0.0, 0.125, 0.5, 1.0, 2.0];
let mut dst = [0.0; 5];
Powf::<GAMMA>::slice(&src, &mut dst, 2.2).unwrap();

// Default positive-normal Midp convenience helpers:
let mut pos_dst = [0.0; 3];
try_powf_slice(&[0.25, 0.5, 2.0], &mut pos_dst, 2.2).unwrap();
```

Each family has one fluent `const` options builder ([`PowfOptions`](https://docs.rs/magemath/latest/magemath/powf/struct.PowfOptions.html), [`ExpfOptions`](https://docs.rs/magemath/latest/magemath/expf/struct.ExpfOptions.html), [`LogfOptions`](https://docs.rs/magemath/latest/magemath/logf/struct.LogfOptions.html), [`Log1pfOptions`](https://docs.rs/magemath/latest/magemath/log1pf/struct.Log1pfOptions.html), [`CbrtfOptions`](https://docs.rs/magemath/latest/magemath/cbrtf/struct.CbrtfOptions.html)) with discrete `#[non_exhaustive]` `*Domain` presets, explicit [`Subnormals`](https://docs.rs/magemath/latest/magemath/enum.Subnormals.html) policies (`FlushOrClamp` default, `Reject` via `.reject_subnormals()`, opt-in `Normalize` via `.admit_subnormals()`), outward-rounded `bf16` custom range/clamp bounds, and (for `powf`) compile-time `.const_exponent(n)` specialization — ending in `.bits()` for `const CFG: u128` (`Powf<CFG>`, `Expf<CFG>`, `Logf<CFG>`, `Log1pf<CFG>`, `Cbrtf<CFG>`, and inline vector methods via [`SimdMath`](https://docs.rs/magemath/latest/magemath/trait.SimdMath.html)) or `.runtime()` for dynamic options.

## Inline SIMD kernels and matching scalar tails

Inside an `#[attune(_*_t, define(f32x8))]` kernel, import [`SimdMath`](https://docs.rs/magemath/latest/magemath/trait.SimdMath.html) to call configured operations inline on any Magetypes `f32` vector (`self.token()` is reused automatically), and pair `.caller()` with `*_with_token` so scalar tails produce identical bits:

```rust
use archmage::prelude::*;
use magemath::SimdMath;
use magemath::powf::{Powf, PowfError, PowfOptions};

const GAMMA_CALLER: u128 = PowfOptions::new().gamma().midp().caller().bits();

#[attune(_*_t, define(f32x8))]
fn powf_slice_impl(token: Token, src: &[f32], dst: &mut [f32], exp: f32) -> Result<(), PowfError> {
    assert_eq!(src.len(), dst.len(), "powf slice lengths differ");
    let (sc, st) = f32x8::partition_slice_t(token, src);
    let (dc, dt) = f32x8::partition_slice_mut_t(token, dst);
    for (s, d) in sc.iter().zip(dc.iter_mut()) {
        f32x8::load_t(token, s).try_powf_with::<GAMMA_CALLER>(exp)?.store(d);
    }
    for (&x, y) in st.iter().zip(dt.iter_mut()) {
        *y = Powf::<GAMMA_CALLER>::scalar_with_token(token, x, exp)?;
    }
    Ok(())
}

pub fn powf_slice(src: &[f32], dst: &mut [f32], exp: f32) -> Result<(), PowfError> {
    attuned!(powf_slice_impl(src, dst, exp))
}

let src = [0.5_f32; 11];
let mut dst = [0.0_f32; 11];
powf_slice(&src, &mut dst, 2.2).unwrap();
assert_eq!(dst[0].to_bits(), dst[10].to_bits());
```

## Limitations and when to use an alternative

- **Bounded approximations, not universal `< 1 ULP` libm:** `Midp` and `Lowp` trade full-range `< 1 ULP` rounding for deterministic SIMD approximations (`PowfPrecision::Highp` adds a binary64 table path or `std` `f32::powf` delegation for `powf`). When you need `< 1 ULP` IEEE-754 libm behavior across arbitrary `f32` or `f64` inputs, use `std::f32` / `libm` or `sleef`.
- **`f32` only:** `f64` transcendentals are not implemented here, and hardware `sqrt` stays directly on [Magetypes](https://docs.rs/magetypes/latest/magetypes/).
- **Flat Magetypes methods remain supported:** if you only need an inline unchecked polynomial inside an existing `magetypes` loop without cross-ISA bit identity, `const_exponent` lowering, or scalar tail helpers, the flat methods on `magetypes` (`pow_midp`, `exp2_lowp`, `cbrt_midp`, ...) are unchanged and not deprecated. See [choosing and migrating](https://github.com/imazen/archmage/blob/main/docs/MAGEMATH-MIGRATION.md) and [Magetypes Fast Paths](https://imazen.github.io/archmage/magemath/magetypes-fast-paths/).

## License

MIT OR Apache-2.0
