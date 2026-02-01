# archmage

[![CI](https://github.com/imazen/archmage/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/archmage/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/archmage.svg)](https://crates.io/crates/archmage)
[![docs.rs](https://docs.rs/archmage/badge.svg)](https://docs.rs/archmage)
[![codecov](https://codecov.io/gh/imazen/archmage/graph/badge.svg)](https://codecov.io/gh/imazen/archmage)
[![MSRV](https://img.shields.io/crates/msrv/archmage.svg)](https://crates.io/crates/archmage)
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

Use `X64V3Token` (or its alias `Desktop64`) for most applications:

| Token | Features | CPU Support |
|-------|----------|-------------|
| `X64V2Token` | SSE4.2 + POPCNT | Windows 11 minimum, Nehalem 2008+ |
| **`X64V3Token`** | AVX2 + FMA + BMI2 | 95%+ of CPUs, Haswell 2013+, Zen 1+ |
| `Desktop64` | AVX2 + FMA + BMI2 | Alias for X64V3Token |

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
| `NeonCrcToken` | NEON + CRC | Most ARMv8 CPUs |

### WASM Tokens

| Token | Features |
|-------|----------|
| `Simd128Token` | WASM SIMD |

## Target Selection

### Choosing Your Baseline

**x86-64-v2** is the minimum requirement for Windows 11, making it a safe baseline for distributed binaries. However, **95%+ of desktop/laptop CPUs** from the last decade support x86-64-v3 (AVX2+FMA), so optimizing for v3 covers nearly all users.

| Target | Use Case | Coverage |
|--------|----------|----------|
| x86-64-v2 | Maximum compatibility (Windows 11 minimum) | ~100% |
| **x86-64-v3** | Recommended for most apps | ~95%+ |
| x86-64-v4 | Server/HPC workloads | Xeon, Zen 4+ |

For most applications, compile a v2 baseline and add v3-optimized paths:

```rust
if let Some(token) = X64V3Token::summon() {
    fast_path(token, data);  // 95%+ of users
} else {
    baseline_path(data);      // Fallback
}
```

### Compile-Time Optimization

When you compile with `-C target-cpu=native` or specify target features that match or exceed a token's requirements, **runtime detection is eliminated**:

```rust
// Compiled with RUSTFLAGS="-C target-cpu=haswell"
if let Some(token) = X64V3Token::summon() {  // Always succeeds, check optimized away
    process(token, data);
} else {
    fallback(data);  // Dead code, optimized away entirely
}
```

This means:
- `summon()` becomes a no-op returning `Some`
- The `else` branch is eliminated by the optimizer
- Zero runtime overhead for feature detection

Build for your deployment target and let the compiler eliminate unused paths.

## Token Hierarchy

Tokens form a hierarchy. Higher-level tokens can extract lower-level ones:

```rust
if let Some(v3) = X64V3Token::summon() {
    let v2: X64V2Token = v3.v2();  // v3 implies v2
}

if let Some(v4) = X64V4Token::summon() {
    let v3: X64V3Token = v4.v3();  // v4 implies v3
    let v2: X64V2Token = v4.v2();  // v4 implies v2
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

The companion crate `magetypes` provides token-gated SIMD types with ergonomic operators:

```toml
[dependencies]
magetypes = "0.1"
```

```rust
use archmage::{Desktop64, SimdToken};
use magetypes::simd::f32x8;

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

## License

MIT OR Apache-2.0

## AI-Generated Code Notice

Developed with Claude (Anthropic). Review critical paths before production use.
