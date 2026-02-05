# Archmage & Magetypes

> Safely invoke your intrinsic power, using the tokens granted to you by the CPU.
> Cast primitive magics faster than any mage alive.

**Archmage** makes SIMD programming in Rust safe and ergonomic. Instead of scattering `unsafe` blocks throughout your code, you prove CPU feature availability once with a **capability token**, then write safe code that the compiler optimizes into raw SIMD instructions.

**Magetypes** provides SIMD vector types (`f32x8`, `i32x4`, etc.) with natural Rust operators that integrate with archmage tokens.

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
use archmage::{Desktop64, SimdToken, arcane};
use std::arch::x86_64::*;

#[arcane]
fn multiply(token: Desktop64, data: &[f32; 8]) -> [f32; 8] {
    // Safe! The token proves AVX2+FMA are available
    let a = _mm256_loadu_ps(data.as_ptr());  // safe_unaligned_simd version
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

3. **Token hoisting**: Call `summon()` once at your API boundary, pass the token through. Don't summon in hot loops.

## Supported Platforms

| Platform | Tokens | Register Width |
|----------|--------|----------------|
| x86-64 | `X64V2Token`, `X64V3Token`/`Desktop64`, `X64V4Token`/`Server64` | 128-512 bit |
| AArch64 | `NeonToken`/`Arm64`, `NeonAesToken`, `NeonSha3Token` | 128 bit |
| WASM | `Simd128Token` | 128 bit |

## Next Steps

- [Installation](./getting-started/installation.md) — Add archmage to your project
- [Your First SIMD Function](./getting-started/first-simd.md) — Write real SIMD code
- [Understanding Tokens](./getting-started/tokens.md) — Learn the token system
