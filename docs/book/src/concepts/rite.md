# The #[rite] Macro

`#[rite]` is for **inner helpers** that are only called from `#[arcane]` functions. It adds `#[target_feature]` without creating a wrapper function.

## When to Use #[rite]

**Use `#[arcane]` for:** Entry points where you summon the token or receive it from external callers.

**Use `#[rite]` for:** Internal helpers that are always called from an `#[arcane]` context.

```rust
use archmage::{arcane, rite, Desktop64};
use std::arch::x86_64::*;

// ENTRY POINT: receives token from caller
#[arcane]
pub fn dot_product(token: Desktop64, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let products = mul_vectors(token, a, b);  // Calls #[rite] helper
    horizontal_sum(token, products)
}

// INNER HELPER: only called from #[arcane] context
#[rite]
fn mul_vectors(_token: Desktop64, a: &[f32; 8], b: &[f32; 8]) -> __m256 {
    let va = unsafe { _mm256_loadu_ps(a.as_ptr()) };
    let vb = unsafe { _mm256_loadu_ps(b.as_ptr()) };
    _mm256_mul_ps(va, vb)  // Safe inside #[target_feature]!
}

// INNER HELPER: only called from #[arcane] context
#[rite]
fn horizontal_sum(_token: Desktop64, v: __m256) -> f32 {
    let sum = _mm256_hadd_ps(v, v);
    let sum = _mm256_hadd_ps(sum, sum);
    let low = _mm256_castps256_ps128(sum);
    let high = _mm256_extractf128_ps::<1>(sum);
    _mm_cvtss_f32(_mm_add_ss(low, high))
}
```

## What It Generates

```rust
// Your code:
#[rite]
fn helper(_token: Desktop64, v: __m256) -> __m256 {
    _mm256_add_ps(v, v)
}

// Generated (NO wrapper function):
#[target_feature(enable = "avx2,fma,bmi1,bmi2")]
#[inline]
fn helper(_token: Desktop64, v: __m256) -> __m256 {
    _mm256_add_ps(v, v)
}
```

Compare to `#[arcane]` which creates:
```rust
fn helper(_token: Desktop64, v: __m256) -> __m256 {
    #[target_feature(enable = "avx2,fma,bmi1,bmi2")]
    #[inline]
    unsafe fn __inner(_token: Desktop64, v: __m256) -> __m256 {
        _mm256_add_ps(v, v)
    }
    unsafe { __inner(_token, v) }
}
```

## Why This Works (Rust 1.85+)

Since Rust 1.85, calling a `#[target_feature]` function from another function with matching or superset features is **safe**—no `unsafe` block needed:

```rust
#[target_feature(enable = "avx2,fma")]
fn outer(data: &[f32; 8]) -> f32 {
    inner_add(data) + inner_mul(data)  // Safe! No unsafe needed!
}

#[target_feature(enable = "avx2")]
#[inline]
fn inner_add(data: &[f32; 8]) -> f32 { /* ... */ }

#[target_feature(enable = "avx2")]
#[inline]
fn inner_mul(data: &[f32; 8]) -> f32 { /* ... */ }
```

The caller's features (`avx2,fma`) are a superset of the callee's (`avx2`), so the compiler knows the call is safe.

## Direct Calls Require Unsafe

If you call a `#[rite]` function from outside a `#[target_feature]` context, you need `unsafe`:

```rust
#[test]
fn test_helper() {
    if let Some(token) = Desktop64::summon() {
        // Direct call from test (no target_feature) requires unsafe
        let result = unsafe { helper(token, data) };
        assert_eq!(result, expected);
    }
}
```

This is correct—the test function doesn't have `#[target_feature]`, so the compiler can't verify safety at compile time. The `unsafe` block says "I checked at runtime via `summon()`."

## Benefits

1. **Zero wrapper overhead**: No extra function call indirection
2. **Better inlining**: LLVM sees the actual function, not a wrapper
3. **Cleaner stack traces**: No `__inner` functions in backtraces
4. **Syntactic sugar**: No need to manually maintain feature strings

## Choosing Between #[arcane] and #[rite]

| Situation | Use |
|-----------|-----|
| Public API function | `#[arcane]` |
| Called from non-SIMD code | `#[arcane]` |
| Called from tests directly | `#[arcane]` (or `#[rite]` + unsafe) |
| Internal helper in a SIMD module | `#[rite]` |
| Composable building blocks | `#[rite]` |
| Functions that must be safe to call | `#[arcane]` |

## Composing Helpers

`#[rite]` helpers compose naturally:

```rust
#[rite]
fn complex_op(token: Desktop64, a: &[f32; 8], b: &[f32; 8], c: &[f32; 8]) -> f32 {
    let ab = mul_vectors(token, a, b);       // Calls another #[rite]
    let vc = load_vector(token, c);          // Calls another #[rite]
    let sum = add_vectors_raw(token, ab, vc); // Calls another #[rite]
    horizontal_sum(token, sum)                // Calls another #[rite]
}
```

All helpers inline into the caller with zero overhead.

## Options

### Custom Inline Behavior

```rust
// Default: #[inline]
#[rite]
fn normal_helper(token: Desktop64, v: __m256) -> __m256 { /* ... */ }

// Force inline (requires nightly + feature)
#[rite(inline_always)]
fn hot_helper(token: Desktop64, v: __m256) -> __m256 { /* ... */ }
```
