# Archmage & Magetypes

> Safely invoke your intrinsic power, using the tokens granted to you by the CPU.
> Cast primitive magics faster than any mage alive.

**Archmage** makes SIMD programming in Rust safe and ergonomic. Instead of scattering `unsafe` blocks throughout your code, you prove CPU feature availability once with a **capability token**, then write safe code that the compiler optimizes into raw SIMD instructions.

**Magetypes** provides SIMD vector types (`f32x8`, `i32x4`, etc.) with natural Rust operators that integrate with archmage tokens.

## Zero Overhead

Archmage is **never slower than equivalent unsafe code**. The safety abstractions exist only at compile time. At runtime, you get the exact same assembly as hand-written `#[target_feature]` + `unsafe` code.

```
Benchmark: 1000 iterations of 8-float vector operations
  Manual unsafe code:     570 ns
  #[rite] in #[arcane]:   572 ns  ← identical
  #[arcane] in loop:     2320 ns  ← wrong pattern (see below)
```

The key is using the right pattern: put loops inside `#[arcane]`, use `#[rite]` for helpers. See [Token Hoisting](./concepts/token-hoisting.md) and [The #\[rite\] Macro](./concepts/rite.md).

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

#[arcane]
fn multiply(_token: Desktop64, data: &[f32; 8]) -> [f32; 8] {
    // Safe! Token proves AVX2+FMA, safe_unaligned_simd takes references
    let a = _mm256_loadu_ps(data);
    let b = _mm256_set1_ps(2.0);
    let c = _mm256_mul_ps(a, b);
    let mut out = [0.0f32; 8];
    _mm256_storeu_ps(&mut out, c);
    out
}

fn main() {
    // Runtime check happens ONCE here
    if let Some(token) = Desktop64::summon() {
        let result = multiply(token, &[1.0; 8]);
        println!("{:?}", result);
    }
}
```

## Key Concepts

1. **Tokens** are zero-sized proof types. `Desktop64::summon()` returns `Some(token)` only if the CPU supports AVX2+FMA.

2. **`#[arcane]`** generates a `#[target_feature]` inner function. Inside, SIMD intrinsics are safe.

3. **Dispatch once, loop inside**: Call `summon()` at your API boundary, put loops inside `#[arcane]`, use `#[rite]` for helpers. Each `#[arcane]` call crosses a `#[target_feature]` boundary that LLVM can't optimize across.

## Supported Platforms

| Platform | Tokens | Register Width |
|----------|--------|----------------|
| x86-64 | `X64V2Token`, `X64V3Token`/`Desktop64`, `X64V4Token`/`Server64` | 128-512 bit |
| AArch64 | `NeonToken`/`Arm64`, `NeonAesToken`, `NeonSha3Token` | 128 bit |
| WASM | `Wasm128Token` | 128 bit |

## Next Steps

- [Installation](./getting-started/installation.md) — Add archmage to your project
- [Your First SIMD Function](./getting-started/first-simd.md) — Write real SIMD code
- [Understanding Tokens](./getting-started/tokens.md) — Learn the token system
