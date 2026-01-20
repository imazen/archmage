# archmage

> Safely invoke your intrinsic power, using the tokens granted to you by the CPU. Cast primitive magics faster than any mage alive.

**archmage** provides capability tokens that prove CPU feature availability at compile time, making raw SIMD intrinsics safe to call via the `#[arcane]` macro.

## The Problem

Raw SIMD intrinsics in Rust have two safety concerns:

1. **Feature availability**: Calling `_mm256_add_ps` on a CPU without AVX2 is undefined behavior
2. **Memory safety**: `_mm256_loadu_ps(ptr)` dereferences a raw pointer

Rust 1.85+ made value-based intrinsics safe inside `#[target_feature]` functions, but:
- Calling those functions is still `unsafe`
- Memory operations remain unsafe regardless

## The Solution: Capability Tokens

```rust
use archmage::{Avx2Token, SimdToken, arcane};
use std::arch::x86_64::*;

#[arcane]
fn double(token: Avx2Token, data: &[f32; 8]) -> [f32; 8] {
    // Memory ops need unsafe (raw pointers)
    let v = unsafe { _mm256_loadu_ps(data.as_ptr()) };

    // Arithmetic intrinsics are SAFE - token proves AVX2!
    let doubled = _mm256_add_ps(v, v);

    let mut out = [0.0f32; 8];
    unsafe { _mm256_storeu_ps(out.as_mut_ptr(), doubled) };
    out
}

fn main() {
    if let Some(token) = Avx2Token::try_new() {
        let result = double(token, &[1.0; 8]);
        // result = [2.0; 8]
    }
}
```

The `#[arcane]` macro wraps your function with `#[target_feature]`, making all value-based intrinsics safe. The token proves the caller verified feature availability.

## What archmage Provides

### 1. Capability Tokens

Zero-sized proof types that verify CPU features at runtime:

```rust
// Runtime detection
if let Some(token) = Avx2Token::try_new() {
    // AVX2 guaranteed available
}

// Inside multiversioned code
let token = avx2_token!();  // Safe in multiversion context
```

**Feature Tokens:**
| Token | Features |
|-------|----------|
| `Sse2Token` | SSE2 (baseline x86-64) |
| `Sse41Token` | SSE4.1 |
| `AvxToken` | AVX |
| `Avx2Token` | AVX2 |
| `FmaToken` | FMA |
| `Avx2FmaToken` | AVX2 + FMA |

**Profile Tokens (x86-64 microarchitecture levels):**
| Token | Features | Hardware |
|-------|----------|----------|
| `X64V2Token` | SSE4.2 + POPCNT | Nehalem 2008+ |
| `X64V3Token` | AVX2 + FMA + BMI2 | Haswell 2013+, Zen 1+ |
| `X64V4Token` | AVX-512 (F/BW/CD/DQ/VL) | Xeon 2017+, Zen 4+ |

**ARM Tokens:**
| Token | Features | Hardware |
|-------|----------|----------|
| `NeonToken` | NEON | All AArch64 |
| `SveToken` | SVE | Graviton 3, Apple M-series |
| `Sve2Token` | SVE2 | ARMv9: Cortex-X2+, Graviton 4 |

### 2. Safe Load/Store (feature = "ops")

Memory operations use references instead of raw pointers:

```rust
use archmage::ops;

if let Some(token) = Avx2Token::try_new() {
    let data = [1.0f32; 8];
    let v = ops::load_f32x8(token, &data);  // Safe!

    let mut out = [0.0f32; 8];
    ops::store_f32x8(token, &mut out, v);   // Safe!
}
```

### 3. Composite Operations (feature = "composite")

High-level SIMD algorithms built on tokens:

```rust
use archmage::composite::{transpose_8x8, dot_product_f32, hsum_f32x8};

if let Some(token) = Avx2FmaToken::try_new() {
    // 8x8 matrix transpose (critical for DCT)
    let mut block = [0.0f32; 64];
    transpose_8x8(token.avx2(), &mut block);

    // Dot product with FMA
    let a = vec![1.0f32; 1024];
    let b = vec![2.0f32; 1024];
    let result = dot_product_f32(token, &a, &b);
}
```

### 4. Generic API with Operation Traits

Write generic code that works with any implementing token:

```rust
use archmage::simd_ops::{Transpose8x8, DotProduct, HorizontalOps};

fn process<T: Transpose8x8 + DotProduct + HorizontalOps>(
    token: T,
    block: &mut [f32; 64],
    data: &[f32],
) {
    token.transpose_8x8(block);                // Specialized per token
    let dot = token.dot_product_f32(data, data);
    let sum = token.sum_f32(data);
}

// Works with Avx2Token, X64V3Token, etc.
if let Some(token) = X64V3Token::try_new() {
    process(token, &mut block, &data);
}
```

The compiler selects the optimized implementation for each token at compile time.

## When to Use archmage

archmage is for when you need **specific instructions** that neither `wide` nor LLVM autovectorization will produce:

- Complex shuffles and permutes
- Exact FMA sequences for numerical precision
- DCT butterflies and signal processing
- Gather/scatter operations
- Bit manipulation (BMI1/BMI2)

For portable SIMD without manual intrinsics, use the `wide` crate instead.

| Approach | When to Use |
|----------|-------------|
| **wide** | Portable code, don't want to think about target features |
| **archmage + intrinsics** | Need specific instructions, complex algorithms |

## Feature Flags

```toml
[dependencies]
archmage = "0.1"
```

| Feature | Description |
|---------|-------------|
| `std` (default) | Enable std library support |
| `macros` (default) | Enable `#[arcane]` attribute macro (alias: `#[simd_fn]`) |
| `ops` | Safe load/store operations |
| `composite` | Higher-level ops (transpose, dot product) - implies `ops` |
| `wide` | Integration with the `wide` crate |
| `safe-simd` | Integration with `safe_unaligned_simd` |
| `full` | Enable all optional features |

## License

MIT OR Apache-2.0

## AI-Generated Code Notice

Developed with Claude (Anthropic). Not all code manually reviewed. Review critical paths before production use.
