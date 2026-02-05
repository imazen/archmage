# Your First SIMD Function

Let's write a function that squares 8 floats in parallel using AVX2.

## The Recommended Way

Use `archmage::prelude::*` which includes `safe_unaligned_simd` for memory operations:

```rust
use archmage::prelude::*;

#[arcane]
fn square_f32x8(_token: Desktop64, data: &[f32; 8]) -> [f32; 8] {
    // safe_unaligned_simd takes references - fully safe!
    let v = _mm256_loadu_ps(data);
    let squared = _mm256_mul_ps(v, v);

    let mut out = [0.0f32; 8];
    _mm256_storeu_ps(&mut out, squared);
    out
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

## Using magetypes

For the most ergonomic experience, use magetypes' vector types:

```rust
use archmage::{Desktop64, SimdToken};
use magetypes::simd::f32x8;

fn square_f32x8(token: Desktop64, data: &[f32; 8]) -> [f32; 8] {
    let v = f32x8::from_array(token, *data);
    let squared = v * v;  // Natural operator!
    squared.to_array()
}

fn main() {
    if let Some(token) = Desktop64::summon() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let output = square_f32x8(token, &input);
        println!("{:?}", output);
    }
}
```

## What `#[arcane]` Does

The macro transforms your function:

```rust
// You write:
#[arcane]
fn square(token: Desktop64, data: &[f32; 8]) -> [f32; 8] {
    // body
}

// Macro generates:
fn square(token: Desktop64, data: &[f32; 8]) -> [f32; 8] {
    #[target_feature(enable = "avx2,fma,bmi1,bmi2")]
    #[inline]
    fn __inner(token: Desktop64, data: &[f32; 8]) -> [f32; 8] {
        // body - intrinsics are safe here!
    }
    // SAFETY: token proves CPU support
    unsafe { __inner(token, data) }
}
```

The token parameter proves you checked CPU features. The macro enables those features for the inner function, making intrinsics safe to call.

## Key Points

1. **`Desktop64`** = AVX2 + FMA + BMI1 + BMI2 (Haswell 2013+, Zen 1+)
2. **`summon()`** does runtime CPU detection
3. **`#[arcane]`** makes intrinsics safe inside the function
4. **Token is zero-sized** — no runtime overhead passing it around
