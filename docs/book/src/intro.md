# Archmage & Magetypes

> Safely invoke your intrinsic power, using the tokens granted to you by the CPU.
> Cast primitive magics faster than any mage alive.

Rust 1.85 made value-based SIMD intrinsics safe inside `#[target_feature]` functions. **Archmage** fills the last gap: proving at the type level that the CPU actually supports those features, so you never call a `#[target_feature]` function on hardware that can't run it. The `import_intrinsics` option on `#[arcane]`/`#[rite]` brings safe memory operations into scope alongside `core::arch` intrinsics, so memory loads and stores take references instead of raw pointers.

You prove CPU feature availability once with a **capability token**, then write safe code that the compiler optimizes into raw SIMD instructions. No `unsafe` needed for the SIMD work itself.

**Magetypes** provides SIMD vector types (`f32x8`, `i32x4`, etc.) with natural Rust operators that integrate with archmage tokens.

## Zero Overhead

Archmage generates identical assembly to bare `#[target_feature]` + `unsafe` code. The safety abstractions compile away entirely. The only thing that costs performance is calling `#[arcane]` from the wrong place (4-6x depending on workload). See the [performance guide](../../PERFORMANCE.md) for full benchmark data, or [Target-Feature Boundaries](./concepts/token-hoisting.md) and [The #\[rite\] Macro](./concepts/rite.md) for the fix.

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
    // Safe! Token proves AVX2+FMA, import_intrinsics provides safe memory ops
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

2. **`#[arcane]`** generates a `#[target_feature]` inner function. Inside, SIMD intrinsics are safe (Rust 1.85+). Descriptive alias: `#[token_target_features_boundary]`.

3. **`#[rite]`** adds `#[target_feature]` + `#[inline]` directly — no wrapper, no boundary. Three modes: token-based (`#[rite]`), tier-based (`#[rite(v3)]` — no token needed), or multi-tier (`#[rite(v3, v4, neon)]` — generates suffixed variants). Descriptive alias: `#[token_target_features]`.

4. **Dispatch once, loop inside**: Call `summon()` at your API boundary, put loops inside `#[arcane]`, use `#[rite]` for helpers (token-based, tier-based, or multi-tier). Each `#[arcane]` call crosses a `#[target_feature]` boundary that LLVM can't optimize across.

5. **`#![forbid(unsafe_code)]` compatible**: Combine archmage tokens + `#[arcane(import_intrinsics)]`/`#[rite(import_intrinsics)]` for safe memory operations, and your downstream crate needs zero `unsafe`.

## Supported Platforms

| Platform | Tokens | Register Width |
|----------|--------|----------------|
| x86-64 | `X64V1Token`, `X64V2Token`, `X64CryptoToken`, `X64V3Token`, `X64V3CryptoToken`, `X64V4Token`/`Server64`, `X64V4xToken`, `Avx512Fp16Token` | 128-512 bit |
| AArch64 | `NeonToken`/`Arm64`, `Arm64V2Token`, `Arm64V3Token`, `NeonAesToken`, `NeonSha3Token`, `NeonCrcToken` | 128 bit |
| WASM | `Wasm128Token`, `Wasm128RelaxedToken` | 128 bit |

## Next Steps

- [Installation](./getting-started/installation.md) — Add archmage to your project
- [Your First SIMD Function](./getting-started/first-simd.md) — Write real SIMD code
- [Understanding Tokens](./getting-started/tokens.md) — Learn the token system

## Resources

- **[Intrinsics Browser](../intrinsics/)** — Search 12,000+ SIMD intrinsics by token, architecture, safety, and stability
- [SIMD Reference](../simd_reference/) — ASM-verified patterns, per-token listings, cross-platform differences
