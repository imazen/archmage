# archmage

[![Crates.io](https://img.shields.io/crates/v/archmage.svg)](https://crates.io/crates/archmage)
[![Documentation](https://docs.rs/archmage/badge.svg)](https://docs.rs/archmage)
[![CI](https://github.com/imazen/archmage/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/archmage/actions/workflows/ci.yml)
[![License](https://img.shields.io/crates/l/archmage.svg)](https://github.com/imazen/archmage#license)

> Safely invoke your intrinsic power, using the tokens granted to you by the CPU.

**archmage** provides zero-cost capability tokens that prove CPU features are available at runtime, making raw SIMD intrinsics safe to call via the `#[arcane]` macro.

## Quick Start

```toml
[dependencies]
archmage = "0.3"
safe_unaligned_simd = "0.2"  # For safe memory operations
```

```rust
use archmage::{Desktop64, SimdToken, arcane};
use std::arch::x86_64::*;

#[arcane]
fn square(_token: Desktop64, data: &[f32; 8]) -> [f32; 8] {
    let v = safe_unaligned_simd::x86_64::_mm256_loadu_ps(data);
    let squared = _mm256_mul_ps(v, v);
    let mut out = [0.0f32; 8];
    safe_unaligned_simd::x86_64::_mm256_storeu_ps(&mut out, squared);
    out
}

fn main() {
    if let Some(token) = Desktop64::summon() {
        let result = square(token, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        println!("{:?}", result); // [1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]
    }
}
```

## How It Works

SIMD intrinsics are unsafe for two reasons:
1. **Feature availability**: Calling AVX2 instructions on a CPU without AVX2 is undefined behavior
2. **Memory operations**: Load/store intrinsics use raw pointers

archmage solves #1 with **capability tokens** - zero-sized types that can only be created after runtime CPU detection succeeds:

```rust
// summon() checks CPUID and returns Some only if features are available
if let Some(token) = Desktop64::summon() {
    // Token exists = CPU definitely has AVX2 + FMA
}
```

The `#[arcane]` macro transforms your function to enable `#[target_feature]`, which makes value-based intrinsics safe (Rust 1.85+):

```rust
#[arcane]
fn example(token: Desktop64, data: &[f32; 8]) -> [f32; 8] {
    let v = safe_unaligned_simd::x86_64::_mm256_loadu_ps(data);  // Safe!
    let result = _mm256_mul_ps(v, v);  // Safe! (value-based)
    // ...
}
```

For memory operations (#2), use the `safe_unaligned_simd` crate which provides reference-based alternatives.

## Token Reference

### x86-64 Tokens

Start with `Desktop64` for most applications:

| Token | Features | CPU Support |
|-------|----------|-------------|
| **`Desktop64`** | AVX2 + FMA + BMI2 | Intel Haswell 2013+, AMD Zen 1 2017+ |
| `X64V2Token` | SSE4.2 + POPCNT | Intel Nehalem 2008+, AMD Bulldozer 2011+ |
| `X64V3Token` | AVX2 + FMA + BMI2 | Same as Desktop64 (alias) |

Individual feature tokens for fine-grained control:

| Token | Features |
|-------|----------|
| `Avx2FmaToken` | AVX2 + FMA |
| `Avx2Token` | AVX2 only |
| `FmaToken` | FMA only |
| `AvxToken` | AVX |
| `Sse42Token` | SSE4.2 |
| `Sse41Token` | SSE4.1 |

### x86-64 AVX-512 Tokens (requires `avx512` feature)

```toml
[dependencies]
archmage = { version = "0.3", features = ["avx512"] }
```

| Token | Features | CPU Support |
|-------|----------|-------------|
| **`X64V4Token`** | AVX-512 F/BW/CD/DQ/VL | Intel Skylake-X 2017+, AMD Zen 4 2022+ |
| `Avx512ModernToken` | + VBMI2, VNNI, BF16, etc. | Intel Ice Lake 2019+, AMD Zen 4+ |
| `Avx512Fp16Token` | + FP16 | Intel Sapphire Rapids 2023+ |

Note: Intel 12th-14th gen consumer CPUs do NOT have AVX-512.

### ARM Tokens

| Token | Features | CPU Support |
|-------|----------|-------------|
| **`Arm64`** | NEON | All AArch64 (baseline) |
| `NeonToken` | NEON | Same as Arm64 (alias) |
| `NeonAesToken` | NEON + AES | ARM with crypto extensions |
| `NeonSha3Token` | NEON + SHA3 | ARMv8.2+ |
| `ArmCryptoToken` | AES + SHA2 + CRC | Most ARMv8 CPUs |
| `ArmCrypto3Token` | + SHA3 | ARMv8.4+ (M1/M2/M3, Graviton 2+) |

### WASM Tokens

| Token | Features |
|-------|----------|
| `Simd128Token` | WASM SIMD |

## Token Hierarchy

Tokens form a hierarchy. Higher-level tokens can extract lower-level ones:

```rust
if let Some(v3) = X64V3Token::summon() {
    let v2: X64V2Token = v3.v2();           // v3 implies v2
    let avx2_fma: Avx2FmaToken = v3.avx2_fma();
    let avx2: Avx2Token = v3.avx2();
    let fma: FmaToken = v3.fma();
    let sse42: Sse42Token = v3.sse42();
}
```

## Trait Bounds

Use trait bounds for generic SIMD code:

```rust
use archmage::{HasX64V2, SimdToken, arcane};

// Accept any token with at least v2 features
#[arcane]
fn process<T: HasX64V2>(_token: T, data: &[u8]) {
    // SSE4.2 intrinsics available
}
```

**Available traits:**

| Trait | Meaning |
|-------|---------|
| `SimdToken` | Base trait for all tokens |
| `HasX64V2` | Has SSE4.2 + POPCNT |
| `HasX64V4` | Has AVX-512 (requires `avx512` feature) |
| `Has128BitSimd` | Has 128-bit vectors |
| `Has256BitSimd` | Has 256-bit vectors |
| `Has512BitSimd` | Has 512-bit vectors |
| `HasNeon` | Has ARM NEON |
| `HasNeonAes` | Has NEON + AES |
| `HasNeonSha3` | Has NEON + SHA3 |

## Cross-Platform Code

All tokens compile on all platforms. `summon()` returns `None` on unsupported architectures:

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

## SIMD Types

archmage provides token-gated SIMD types with ergonomic operators:

```rust
use archmage::{Desktop64, SimdToken, simd::f32x8};

if let Some(token) = Desktop64::summon() {
    let a = f32x8::splat(token, 2.0);
    let b = f32x8::from_array(token, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let c = a * b + a;  // Operators work naturally
    let result = c.sqrt();
    println!("{:?}", result.to_array());
}
```

### Available Types

| Width | Float | Signed Int | Unsigned Int | Token Required |
|-------|-------|------------|--------------|----------------|
| 128-bit | `f32x4`, `f64x2` | `i8x16`, `i16x8`, `i32x4`, `i64x2` | `u8x16`, `u16x8`, `u32x4`, `u64x2` | `X64V3Token` |
| 256-bit | `f32x8`, `f64x4` | `i8x32`, `i16x16`, `i32x8`, `i64x4` | `u8x32`, `u16x16`, `u32x8`, `u64x4` | `X64V3Token` |
| 512-bit | `f32x16`, `f64x8` | `i8x64`, `i16x32`, `i32x16`, `i64x8` | `u8x64`, `u16x32`, `u32x16`, `u64x8` | `X64V4Token` |

### Operations

**Construction** (requires token): `splat`, `from_array`, `load`, `zero`

**Extraction**: `to_array`, `as_array`, `store`, `raw`

**Arithmetic**: `+`, `-`, `*`, `/` and assignment variants

**Bitwise**: `&`, `|`, `^` and assignment variants

**Math** (float): `sqrt`, `abs`, `floor`, `ceil`, `round`, `min`, `max`, `clamp`, `mul_add`, `mul_sub`, `recip`, `rsqrt`

**Transcendentals** (float): `log2_lowp`, `log2_midp`, `exp2_lowp`, `exp2_midp`, `ln_lowp`, `ln_midp`, `exp_lowp`, `exp_midp`, `pow_lowp`, `pow_midp`, `cbrt_midp`

**Comparison**: `simd_eq`, `simd_ne`, `simd_lt`, `simd_le`, `simd_gt`, `simd_ge`

**Reduction**: `reduce_add`, `reduce_min`, `reduce_max`

**Integer**: `shl::<N>`, `shr::<N>`, `shr_arithmetic::<N>`

## Feature Flags

| Feature | Description |
|---------|-------------|
| `std` (default) | Standard library support |
| `macros` (default) | `#[arcane]` macro |
| `avx512` | AVX-512 tokens |
| `__composite` | Transpose, dot product (unstable) |
| `__wide` | `wide` crate integration (unstable) |

## Testing Fallback Paths

Set `ARCHMAGE_DISABLE=1` to force `summon()` to return `None`:

```bash
ARCHMAGE_DISABLE=1 cargo test
```

## License

MIT OR Apache-2.0

## AI-Generated Code Notice

Developed with Claude (Anthropic). Review critical paths before production use.
