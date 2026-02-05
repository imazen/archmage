# archmage

[![CI](https://github.com/imazen/archmage/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/archmage/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/archmage.svg)](https://crates.io/crates/archmage)
[![docs.rs](https://docs.rs/archmage/badge.svg)](https://docs.rs/archmage)
[![codecov](https://codecov.io/gh/imazen/archmage/graph/badge.svg)](https://codecov.io/gh/imazen/archmage)
[![MSRV](https://img.shields.io/crates/msrv/archmage.svg)](https://crates.io/crates/archmage)
[![License](https://img.shields.io/crates/l/archmage.svg)](https://github.com/imazen/archmage#license)

> Safely invoke your intrinsic power, using the tokens granted to you by the CPU.

Zero-cost capability tokens that prove CPU features exist at runtime, making SIMD safe to use in Rust. Works across x86-64, ARM, and WASM.

```toml
[dependencies]
archmage = "0.4"
magetypes = "0.4"
```

## Usage

### High-level: the prelude

The prelude gives you the best SIMD types for whatever platform you're compiling on. No `#[cfg]` blocks needed.

```rust
use archmage::SimdToken;
use magetypes::prelude::*;

fn magnitude_squared(data: &[f32]) -> f32 {
    if let Some(token) = RecommendedToken::summon() {
        let chunks = data.chunks_exact(LANES);
        let remainder = chunks.remainder();

        let mut acc = F32Vec::splat(token, 0.0);
        for chunk in chunks {
            let v = F32Vec::from_slice(token, chunk);
            acc = v.mul_add(v, acc);
        }

        let mut sum: f32 = acc.reduce_add();
        for &x in remainder {
            sum += x * x;
        }
        sum
    } else {
        data.iter().map(|x| x * x).sum()
    }
}
```

`RecommendedToken` is `X64V3Token` on x86-64 (AVX2+FMA), `NeonToken` on ARM, `Simd128Token` on WASM. `F32Vec` and `LANES` match: `f32x8`/8, `f32x4`/4, `f32x4`/4 respectively.

### Multi-platform codegen: `#[magetypes]`

Write one function with placeholder types. The macro generates platform-specific variants. `incant!` dispatches to the best one at runtime.

```rust
use archmage::{incant, magetypes};

#[magetypes]
fn magnitude_squared(token: Token, data: &[f32]) -> f32 {
    let _ = token;
    let chunks = data.chunks_exact(LANES);
    let remainder = chunks.remainder();
    let mut acc = f32xN::splat(token, 0.0);
    for chunk in chunks {
        let v = f32xN::from_slice(token, chunk);
        acc = v.mul_add(v, acc);
    }
    let mut sum: f32 = acc.reduce_add();
    for &x in remainder {
        sum += x * x;
    }
    sum
}

// Dispatches to magnitude_squared_v3, _v4, _neon, _wasm128, or _scalar
pub fn magnitude_squared_api(data: &[f32]) -> f32 {
    incant!(magnitude_squared(data))
}
```

`Token`, `f32xN`, and `LANES` are replaced with concrete types for each variant. The scalar fallback always compiles.

### Low-level: raw intrinsics with `#[arcane]`

When you need specific SIMD instructions, `#[arcane]` makes them safe. It sets `#[target_feature]` on the generated inner function, which means value-based intrinsics are safe (Rust 1.85+). Memory operations use `safe_unaligned_simd` for reference-based safety.

```toml
[dependencies]
archmage = "0.4"
safe_unaligned_simd = "0.2"
```

```rust
use archmage::{Desktop64, SimdToken, arcane};
use std::arch::x86_64::*;

#[arcane]
fn dot_product(_token: Desktop64, a: &[f32], b: &[f32]) -> f32 {
    // safe_unaligned_simd loads are safe inside #[arcane]
    let va = safe_unaligned_simd::x86_64::_mm256_loadu_ps(&a[..8]);
    let vb = safe_unaligned_simd::x86_64::_mm256_loadu_ps(&b[..8]);
    // Value-based intrinsics are safe inside #[arcane]
    let product = _mm256_mul_ps(va, vb);
    // horizontal sum...
    # let mut out = [0.0f32; 8];
    # safe_unaligned_simd::x86_64::_mm256_storeu_ps(&mut out, product);
    # out.iter().sum()
}

fn main() {
    if let Some(token) = Desktop64::summon() {
        let result = dot_product(token, &[1.0; 8], &[2.0; 8]);
    }
}
```

## How tokens work

SIMD intrinsics are unsafe because calling AVX2 on a CPU without AVX2 is undefined behavior. Tokens fix this: they're zero-sized types that can only be created after runtime CPUID detection succeeds.

```rust
// summon() checks CPUID, returns Some only if features are present
if let Some(token) = Desktop64::summon() {
    // Token exists = CPU has AVX2 + FMA. Pass it to #[arcane] functions.
}
```

If you compile with `-C target-cpu=haswell` (or any target that includes the token's features), `summon()` becomes a no-op, the `else` branch is dead code, and the compiler eliminates both.

## Token reference

### x86-64

| Token | Features | CPUs |
|-------|----------|------|
| `X64V2Token` | SSE4.2, POPCNT | Nehalem 2008+; Windows 11 baseline |
| `X64V3Token` / `Desktop64` | AVX2, FMA, BMI2 | Haswell 2013+, Zen 1+; ~95% of desktops |

### x86-64 AVX-512 (requires `avx512` feature)

| Token | Features | CPUs |
|-------|----------|------|
| `X64V4Token` / `Server64` | AVX-512 F/BW/CD/DQ/VL | Skylake-X 2017+, Zen 4+ |
| `Avx512ModernToken` | + VBMI2, VNNI, BF16 | Ice Lake 2019+, Zen 4+ |
| `Avx512Fp16Token` | + FP16 | Sapphire Rapids 2023+ |

Intel 12th-14th gen consumer CPUs do not have AVX-512.

### ARM

| Token | Features | CPUs |
|-------|----------|------|
| `NeonToken` / `Arm64` | NEON | All AArch64 (baseline) |
| `NeonAesToken` | + AES | ARMv8 with crypto |
| `NeonSha3Token` | + SHA3 | ARMv8.2+ |
| `NeonCrcToken` | + CRC | Most ARMv8 |

### WASM

| Token | Features |
|-------|----------|
| `Simd128Token` | WASM SIMD |

## Token hierarchy and traits

Higher tokens can extract lower ones:

```rust
if let Some(v4) = X64V4Token::summon() {
    let v3: X64V3Token = v4.v3();  // v4 implies v3
    let v2: X64V2Token = v4.v2();  // v4 implies v2
}
```

Use trait bounds for generic code:

```rust
use archmage::{HasX64V2, arcane};

#[arcane]
fn process<T: HasX64V2>(_token: T, data: &[u8]) {
    // SSE4.2 intrinsics available here
}
```

| Trait | Meaning |
|-------|---------|
| `SimdToken` | Base trait for all tokens |
| `HasX64V2` | SSE4.2 + POPCNT |
| `HasX64V4` | AVX-512 (requires `avx512` feature) |
| `Has128BitSimd` | 128-bit vectors |
| `Has256BitSimd` | 256-bit vectors |
| `Has512BitSimd` | 512-bit vectors |
| `HasNeon` | ARM NEON |
| `HasNeonAes` | NEON + AES |
| `HasNeonSha3` | NEON + SHA3 |

## Compile-time dispatch with `IntoConcreteToken`

For generic dispatch where the token type is a type parameter, `IntoConcreteToken` lets you branch at compile time. The compiler monomorphizes away non-matching branches.

```rust
use archmage::{IntoConcreteToken, SimdToken, ScalarToken};

fn process<T: IntoConcreteToken>(token: T, data: &mut [f32]) {
    if let Some(t) = token.as_x64v3() {
        process_avx2(t, data);
    } else if let Some(t) = token.as_neon() {
        process_neon(t, data);
    } else if let Some(_) = token.as_scalar() {
        process_scalar(data);
    }
}
```

Or use `incant!` in passthrough mode to do this automatically:

```rust
fn process<T: IntoConcreteToken>(token: T, data: &mut [f32]) {
    incant!(some_magetypes_fn(data) with token)
}
```

## SIMD types (magetypes)

| Width | Float | Signed | Unsigned | Token |
|-------|-------|--------|----------|-------|
| 128-bit | `f32x4`, `f64x2` | `i8x16`..`i64x2` | `u8x16`..`u64x2` | `X64V3Token` / `NeonToken` / `Simd128Token` |
| 256-bit | `f32x8`, `f64x4` | `i8x32`..`i64x4` | `u8x32`..`u64x4` | `X64V3Token` |
| 512-bit | `f32x16`, `f64x8` | `i8x64`..`i64x8` | `u8x64`..`u64x8` | `X64V4Token` |

**Construction** (token required): `splat`, `from_array`, `from_slice`, `zero`, `load`

**Extraction**: `to_array`, `as_array`, `store`, `reduce_add`, `reduce_min`, `reduce_max`

**Arithmetic**: `+`, `-`, `*`, `/`, `mul_add`, `mul_sub`

**Math**: `sqrt`, `abs`, `floor`, `ceil`, `round`, `min`, `max`, `clamp`, `recip`, `rsqrt`

**Transcendentals**: `log2_lowp`, `exp2_lowp`, `ln_lowp`, `exp_lowp`, `pow_lowp` (and `_midp` variants)

**Comparison**: `simd_eq`, `simd_ne`, `simd_lt`, `simd_le`, `simd_gt`, `simd_ge`

**Bitwise/shifts**: `&`, `|`, `^`, `shl::<N>`, `shr::<N>`, `shr_arithmetic::<N>`

## Cross-platform code

All tokens compile on all platforms. `summon()` returns `None` on unsupported architectures, so dispatch chains work everywhere:

```rust
use archmage::{Desktop64, Arm64, SimdToken};

fn process(data: &mut [f32]) {
    if let Some(token) = Desktop64::summon() {
        process_avx2(token, data);
    } else if let Some(token) = Arm64::summon() {
        process_neon(token, data);
    } else {
        process_scalar(data);
    }
}
```

## Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | yes | Standard library support |
| `macros` | yes | `#[arcane]`, `#[magetypes]`, `incant!` |
| `bytemuck` | yes | Pod/Zeroable casts for SIMD types |
| `avx512` | no | AVX-512 tokens and 512-bit types |

## License

MIT OR Apache-2.0

## AI-Generated Code Notice

Developed with Claude (Anthropic). Review critical paths before production use.
