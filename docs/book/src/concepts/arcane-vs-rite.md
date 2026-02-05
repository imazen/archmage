# When to Use #[arcane] vs #[rite]

Both macros enable SIMD instructions in your function. The difference is how they handle calls from outside SIMD context.

## TL;DR

| Macro | Use When | Overhead |
|-------|----------|----------|
| `#[rite]` | Called from other SIMD functions | Zero |
| `#[arcane]` | Called from non-SIMD code (tests, public API) | Minimal wrapper |

**Default to `#[rite]`.** Use `#[arcane]` only at entry points.

## The Rule

```rust
// ENTRY POINT: Called from non-SIMD code after summon()
//              Use #[arcane] here
if let Some(token) = Desktop64::summon() {
    process_simd(token, data);  // ← #[arcane]
}

// INTERNAL HELPERS: Called from other SIMD functions
//                   Use #[rite] here
#[rite]
fn helper(token: Desktop64, chunk: &[f32; 8]) -> f32 { ... }
```

## Why This Matters

`#[rite]` generates a direct `#[target_feature]` function:

```rust
// #[rite] generates:
#[target_feature(enable = "avx2,fma,...")]
#[inline]
fn helper(token: Desktop64, data: &[f32; 8]) -> f32 {
    // your code
}
```

`#[arcane]` wraps that in a safe outer function:

```rust
// #[arcane] generates:
fn process(token: Desktop64, data: &[f32; 8]) -> f32 {
    #[target_feature(enable = "avx2,fma,...")]
    #[inline]
    unsafe fn __inner(token: Desktop64, data: &[f32; 8]) -> f32 {
        // your code
    }
    unsafe { __inner(token, data) }
}
```

The wrapper is needed because calling a `#[target_feature]` function from normal code requires `unsafe`. But when you're already inside a `#[target_feature]` context (like another `#[rite]` function), the wrapper is pure overhead.

## Example: Correct Usage

```rust
use archmage::{Desktop64, SimdToken, rite};
use magetypes::f32x8;

// PUBLIC API
pub fn normalize(data: &mut [f32]) {
    if let Some(token) = Desktop64::summon() {
        normalize_simd(token, data);
    } else {
        normalize_scalar(data);
    }
}

// ENTRY POINT: called from non-SIMD context
// Could use #[arcane] here, but #[rite] works too since we're
// just calling it once from the dispatch point
#[rite]
fn normalize_simd(token: Desktop64, data: &mut [f32]) {
    let len = compute_length(token, data);  // calls #[rite]
    if len > 0.0 {
        scale_vector(token, data, 1.0 / len);  // calls #[rite]
    }
}

// HELPER: called from normalize_simd
#[rite]
fn compute_length(token: Desktop64, data: &[f32]) -> f32 {
    let mut sum = f32x8::zero(token);
    for chunk in data.chunks_exact(8) {
        let v = f32x8::from_slice(token, chunk);
        sum = v.mul_add(v, sum);
    }
    sum.reduce_add().sqrt()
}

// HELPER: called from normalize_simd
#[rite]
fn scale_vector(token: Desktop64, data: &mut [f32], scale: f32) {
    let scale_v = f32x8::splat(token, scale);
    for chunk in data.chunks_exact_mut(8) {
        let v = f32x8::from_slice(token, chunk);
        (v * scale_v).store_slice(chunk);
    }
}

fn normalize_scalar(data: &mut [f32]) {
    let len: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();
    if len > 0.0 {
        for x in data {
            *x /= len;
        }
    }
}
```

All the SIMD functions use `#[rite]`. They inline into each other with zero overhead.

## When You Need #[arcane]

**From tests:**

If you want to call a SIMD function directly from a test, use `#[arcane]`:

```rust
#[arcane]
fn compute_length(token: Desktop64, data: &[f32]) -> f32 {
    // SIMD implementation
    let mut sum = f32x8::zero(token);
    for chunk in data.chunks_exact(8) {
        let v = f32x8::from_slice(token, chunk);
        sum = v.mul_add(v, sum);
    }
    sum.reduce_add().sqrt()
}

#[test]
fn test_compute_length() {
    if let Some(token) = Desktop64::summon() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = compute_length(token, &data);
        assert!((result - 14.28).abs() < 0.1);
    }
}
```

The `#[arcane]` wrapper makes the function callable from non-SIMD contexts like tests.

**From callbacks or trait implementations:**

```rust
trait Processor {
    fn process(&self, data: &mut [f32]);
}

struct SimdProcessor {
    token: Desktop64,
}

impl Processor for SimdProcessor {
    fn process(&self, data: &mut [f32]) {
        // Trait method isn't #[target_feature], so we need #[arcane]
        process_impl(self.token, data);
    }
}

#[arcane]  // Called from non-target_feature context
fn process_impl(token: Desktop64, data: &mut [f32]) {
    // ...
}
```

## Summary

1. **Use `#[rite]` by default** — for all SIMD functions
2. **Use `#[arcane]` at boundaries** — when called from non-SIMD code
3. **All `#[rite]` functions inline** — zero overhead when composing SIMD code
4. **The performance difference is real** — 4x slower if you use `#[arcane]` everywhere

Benchmark results:
```
#[arcane] in loop:    2.32 µs  (wrapper overhead each iteration)
#[rite] in #[rite]:   572 ns   (full inlining)
```
