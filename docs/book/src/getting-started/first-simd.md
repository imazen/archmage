# Your First SIMD Function

Let's write a function that squares 8 floats in parallel using AVX2.

## The Code

```rust
use archmage::{Desktop64, SimdToken, arcane};
use magetypes::f32x8;

#[arcane]
fn square_f32x8(token: Desktop64, data: &[f32; 8]) -> [f32; 8] {
    let v = f32x8::from_array(token, *data);
    let squared = v * v;
    squared.to_array()
}

fn main() {
    if let Some(token) = Desktop64::summon() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let output = square_f32x8(token, &input);
        println!("{:?}", output);
        // [1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]
    } else {
        println!("AVX2 not available");
    }
}
```

## What's Happening

1. **`Desktop64::summon()`** — Runtime check via CPUID. Returns `Some(token)` if CPU has AVX2+FMA.

2. **`#[arcane]`** — Tells the compiler to use AVX2 instructions in this function. The token parameter proves the check happened.

3. **`f32x8`** — A vector of 8 floats. Operations compile to single AVX2 instructions.

4. **`v * v`** — Compiles to `vmulps` (one instruction, 8 multiplies in parallel).

## Key Points

- **`Desktop64`** = AVX2 + FMA + BMI1 + BMI2 (Intel Haswell 2013+, AMD Zen 2017+)
- **Token is zero-sized** — no runtime overhead passing it around
- **`#[arcane]`** is required for the compiler to use SIMD instructions

## Using Raw Intrinsics

If you need direct control over instruction selection, use `std::arch` intrinsics with `safe_unaligned_simd`:

```toml
[dependencies]
archmage = "0.4"
safe_unaligned_simd = "0.2"
```

```rust
use archmage::{Desktop64, SimdToken, arcane};
use std::arch::x86_64::*;
use safe_unaligned_simd::x86_64 as safe_simd;

#[arcane]
fn square_f32x8(token: Desktop64, data: &[f32; 8]) -> [f32; 8] {
    let v = safe_simd::_mm256_loadu_ps(data);
    let squared = _mm256_mul_ps(v, v);

    let mut out = [0.0f32; 8];
    safe_simd::_mm256_storeu_ps(&mut out, squared);
    out
}
```

`safe_unaligned_simd` provides safe wrappers for memory operations. Arithmetic intrinsics like `_mm256_mul_ps` are safe inside `#[arcane]`.

## Next Steps

- [Understanding Tokens](./tokens.md) — Learn about the token hierarchy
- [Token Hoisting](../concepts/token-hoisting.md) — Why summon() goes at the top, loops go inside
- [The #\[rite\] Macro](../concepts/rite.md) — Zero-overhead helpers called from `#[arcane]`
