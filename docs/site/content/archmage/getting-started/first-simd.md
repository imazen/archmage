+++
title = "Your First SIMD Function"
weight = 2
+++

# Your First SIMD Function

Let's write a function that squares 8 floats in parallel using AVX2.

## The Recommended Way

Use `#[arcane(import_intrinsics)]` which auto-imports all intrinsics with safe memory ops:

```rust
use archmage::prelude::*;

#[arcane(import_intrinsics)]
fn square_f32x8(_token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    // import_intrinsics brings types, value ops, and safe memory ops into scope
    let v = _mm256_loadu_ps(data);   // Takes &[f32; 8], not *const f32
    let squared = _mm256_mul_ps(v, v);

    let mut out = [0.0f32; 8];
    _mm256_storeu_ps(&mut out, squared);  // Takes &mut [f32; 8], not *mut f32
    out
}

fn main() {
    if let Some(token) = X64V3Token::summon() {
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let output = square_f32x8(token, &input);
        println!("{:?}", output);
        // [1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0, 64.0]
    } else {
        println!("AVX2 not available");
    }
}
```

## Using magetypes (exploratory)

For a more ergonomic experience, [magetypes](/magetypes/) provides generic vector types with natural operators. Write one function, run it on any backend. Note that magetypes is an exploratory crate — its API may change between releases:

```rust
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Backend,
};

#[inline(always)]
fn square_f32x8<T: F32x8Backend>(token: T, data: &[f32; 8]) -> [f32; 8] {
    let v = f32x8::<T>::from_array(token, *data);
    let squared = v * v;  // Natural operator!
    squared.to_array()
}
```

Call it with any backend token — `X64V3Token` for AVX2, `NeonToken` for ARM, `ScalarToken` as fallback:

```rust
use archmage::{X64V3Token, SimdToken};

fn main() {
    if let Some(token) = X64V3Token::summon() {
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
#[arcane(import_intrinsics)]
fn square(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    // body — intrinsics in scope from import_intrinsics
}

// Macro generates (sibling mode — the default):
#[cfg(target_arch = "x86_64")]
#[doc(hidden)]
#[target_feature(enable = "avx2,fma,bmi1,bmi2,...")]
fn __arcane_square(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    use archmage::intrinsics::x86_64::*;
    // body — intrinsics are safe here!
}

#[cfg(target_arch = "x86_64")]
fn square(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    // SAFETY: token proves CPU support
    unsafe { __arcane_square(token, data) }
}
```

The sibling function is `fn` (not `unsafe fn`) — `#![forbid(unsafe_code)]` compatible. The token parameter proves you checked CPU features. The macro enables those features for the sibling, making intrinsics safe to call.

## Adding Helpers

For functions called from within SIMD code, use `#[rite]` instead of `#[arcane]`. You can specify the tier directly instead of taking a token parameter:

```rust
use archmage::prelude::*;

#[arcane(import_intrinsics)]
fn process(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    let v = _mm256_loadu_ps(data);
    let squared = square(v);     // #[rite(v3)] — inlines, no token needed
    let mut out = [0.0f32; 8];
    _mm256_storeu_ps(&mut out, squared);
    out
}

// Tier-based: specify features via tier name, no token parameter
#[rite(v3, import_intrinsics)]
fn square(v: __m256) -> __m256 {
    _mm256_mul_ps(v, v)
}
```

`#[rite(v3)]` generates the same `#[target_feature]` as taking an `X64V3Token` parameter. Use it when the function doesn't need the token for anything else. Taking the token parameter works just as well — and can be easier to remember if you already have the token in scope. For multi-tier code, `#[rite(v3, v4, neon)]` generates suffixed variants (`square_v3`, `square_v4`, `square_neon`) from a single function body.

Since Rust 1.85, calling a `#[target_feature]` function from another function with matching features is safe — that's why `square(v)` needs no `unsafe` block inside the `#[arcane]` function. The caller's V3 features are a superset of the callee's V3 features, so the compiler verifies safety at compile time.

## Key Points

1. **`X64V3Token`** = AVX2 + FMA + BMI1 + BMI2 (Haswell 2013+, Zen 1+)
2. **`summon()`** does runtime CPU detection
3. **`#[arcane(import_intrinsics)]`** at entry points, **`#[rite(v3, import_intrinsics)]`** for internal helpers
4. **Token is zero-sized** — no runtime overhead passing it around
5. **Memory ops take references** — `_mm256_loadu_ps` takes `&[f32; 8]` instead of `*const f32` (via `import_intrinsics`)
6. **`#![forbid(unsafe_code)]` compatible** — archmage + `#[arcane]`/`#[rite]` macros mean your crate needs zero `unsafe`
