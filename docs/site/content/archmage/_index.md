+++
title = "Archmage"
description = "Safe SIMD via capability tokens for Rust"
sort_by = "weight"
weight = 1

[extra]
sidebar = true
+++

# Archmage

> Safely invoke your intrinsic power, using the tokens granted to you by the CPU. Cast primitive magics faster than any mage alive.

Rust 1.85 made value-based SIMD intrinsics safe inside `#[target_feature]` functions. [`safe_unaligned_simd`](https://crates.io/crates/safe_unaligned_simd) made memory intrinsics safe by taking references instead of raw pointers. **Archmage** fills the last gap: proving at the type level that the CPU actually supports those features, so you never call a `#[target_feature]` function on hardware that can't run it.

You prove CPU feature availability once with a **capability token**, then write safe code that the compiler optimizes into raw SIMD instructions. No `unsafe` needed for the SIMD work itself.

## Zero Overhead

Archmage generates identical assembly to bare `#[target_feature]` + `unsafe` code. The safety abstractions compile away entirely. The only thing that costs performance is calling `#[arcane]` from the wrong place (4-6x depending on workload). See [Target-Feature Boundaries](@/archmage/concepts/target-feature-boundaries.md) and [The #\[rite\] Macro](@/archmage/concepts/rite.md) for the fix.

## The Problem

Raw SIMD in Rust requires `unsafe`:

```rust
use std::arch::x86_64::*;

// Every. Single. Call.
unsafe {
    let a = _mm256_loadu_ps(data.as_ptr());
    let b = _mm256_set1_ps(2.0);
    let c = _mm256_mul_ps(a, b);
    _mm256_storeu_ps(out.as_mut_ptr(), c);
}
```

This is tedious and error-prone. Miss a feature check? Undefined behavior on older CPUs.

## The Solution

Archmage separates **proof of capability** from **use of capability**:

```rust
use archmage::prelude::*;

#[arcane(import_intrinsics)]
fn multiply(_token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    // Safe! Token proves AVX2+FMA. Intrinsics in scope from import_intrinsics.
    let a = _mm256_loadu_ps(data);
    let b = _mm256_set1_ps(2.0);
    let c = _mm256_mul_ps(a, b);
    let mut out = [0.0f32; 8];
    _mm256_storeu_ps(&mut out, c);
    out
}

fn main() {
    // Runtime check happens ONCE here
    if let Some(token) = X64V3Token::summon() {
        let result = multiply(token, &[1.0; 8]);
        println!("{:?}", result);
    }
}
```

## Key Concepts

1. **Tokens** are zero-sized proof types. `X64V3Token::summon()` returns `Some(token)` only if the CPU supports AVX2+FMA. See [`token-registry.toml`](https://github.com/imazen/archmage/blob/main/token-registry.toml) for the complete token-to-feature mapping.

2. **`#[arcane(import_intrinsics)]`** generates a `#[target_feature]` function and auto-imports intrinsics. Inside, SIMD intrinsics are safe (Rust 1.85+). Descriptive alias: `#[token_target_features_boundary]`.

3. **`#[rite(import_intrinsics)]`** adds `#[target_feature]` + `#[inline]` directly and auto-imports intrinsics — no wrapper, no boundary. Descriptive alias: `#[token_target_features]`.

4. **Dispatch once, loop inside**: Call `summon()` at your API boundary, put loops inside `#[arcane(import_intrinsics)]`, use `#[rite(import_intrinsics)]` for everything called from SIMD code. Each `#[arcane]` call crosses a `#[target_feature]` boundary that LLVM can't optimize across.

5. **`#![forbid(unsafe_code)]` compatible**: Combine archmage tokens + `#[arcane]`/`#[rite]` + `safe_unaligned_simd` for memory operations, and your downstream crate needs zero `unsafe`.

## Supported Platforms

| Platform | Tokens | Register Width |
|----------|--------|----------------|
| x86-64 | `X64V1Token`, `X64V2Token`, `X64CryptoToken`, `X64V3Token`, `X64V3CryptoToken`, `X64V4Token`/`Server64`, `X64V4xToken`, `Avx512Fp16Token` | 128-512 bit |
| AArch64 | `NeonToken`/`Arm64`, `Arm64V2Token`, `Arm64V3Token`, `NeonAesToken`, `NeonSha3Token`, `NeonCrcToken` | 128 bit |
| WASM | `Wasm128Token`, `Wasm128RelaxedToken` | 128 bit |

## Magetypes (Exploratory)

[Magetypes](/magetypes/) is our companion crate that provides ergonomic SIMD vector types with natural Rust operators (`f32x8`, `i32x4`, etc.). It's an **exploratory crate** — the API may change between releases. Archmage itself is stable and does not depend on magetypes.

## Next Steps

- [Installation](@/archmage/getting-started/installation.md) — Add archmage to your project
- [Your First SIMD Function](@/archmage/getting-started/first-simd.md) — Write real SIMD code
- [Understanding Tokens](@/archmage/getting-started/tokens.md) — Learn the token system

## Resources

- **[Intrinsics Browser](https://imazen.github.io/archmage/intrinsics/)** — Search 12,000+ SIMD intrinsics by token, architecture, safety, and stability
