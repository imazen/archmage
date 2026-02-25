+++
title = "Methods with #[arcane]"
weight = 1
+++

Using `#[arcane]` on methods requires special handling because of how the macro transforms the function body.

## The Problem

```rust
impl MyType {
    // This won't work as expected!
    #[arcane]
    fn process(&self, token: Desktop64) -> f32 {
        self.data[0]  // Error: `self` not available in inner function
    }
}
```

The macro generates an inner function where `self` becomes a regular parameter.

## The Solution: `_self = Type`

Use the `_self` argument and reference `_self` in your code:

```rust
use archmage::{Desktop64, arcane};

struct Vector8([f32; 8]);

impl Vector8 {
    #[arcane(_self = Vector8)]
    fn magnitude(&self, token: Desktop64) -> f32 {
        // Use _self, not self
        let v = _mm256_loadu_ps(&_self.0);
        let sq = _mm256_mul_ps(v, v);
        let sum = _mm256_hadd_ps(sq, sq);
        let sum = _mm256_hadd_ps(sum, sum);
        let low = _mm256_castps256_ps128(sum);
        let high = _mm256_extractf128_ps::<1>(sum);
        _mm_cvtss_f32(_mm_sqrt_ss(_mm_add_ss(low, high)))
    }
}
```

## All Receiver Types

### `&self` (Shared Reference)

```rust
impl Vector8 {
    #[arcane(_self = Vector8)]
    fn dot(&self, token: Desktop64, other: &Self) -> f32 {
        let a = _mm256_loadu_ps(&_self.0);
        let b = _mm256_loadu_ps(&other.0);
        let mul = _mm256_mul_ps(a, b);
        let sum = _mm256_hadd_ps(mul, mul);
        let sum = _mm256_hadd_ps(sum, sum);
        let low = _mm256_castps256_ps128(sum);
        let high = _mm256_extractf128_ps::<1>(sum);
        _mm_cvtss_f32(_mm_add_ss(low, high))
    }
}
```

### `&mut self` (Mutable Reference)

```rust
impl Vector8 {
    #[arcane(_self = Vector8)]
    fn normalize(&mut self, token: Desktop64) {
        let v = _mm256_loadu_ps(&_self.0);
        let sq = _mm256_mul_ps(v, v);
        let sum = _mm256_hadd_ps(sq, sq);
        let sum = _mm256_hadd_ps(sum, sum);
        let low = _mm256_castps256_ps128(sum);
        let high = _mm256_extractf128_ps::<1>(sum);
        let len = _mm_cvtss_f32(_mm_sqrt_ss(_mm_add_ss(low, high)));
        if len > 0.0 {
            let inv = _mm256_set1_ps(1.0 / len);
            let normalized = _mm256_mul_ps(v, inv);
            _mm256_storeu_ps(&mut _self.0, normalized);
        }
    }
}
```

### `self` (By Value)

```rust
impl Vector8 {
    #[arcane(_self = Vector8)]
    fn scaled(self, token: Desktop64, factor: f32) -> Self {
        let v = _mm256_loadu_ps(&_self.0);
        let s = _mm256_set1_ps(factor);
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(&mut out, _mm256_mul_ps(v, s));
        Vector8(out)
    }
}
```

## Trait Implementations

Works with traits too:

```rust
trait SimdOps {
    fn double(&self, token: Desktop64) -> Self;
}

impl SimdOps for Vector8 {
    #[arcane(_self = Vector8)]
    fn double(&self, token: Desktop64) -> Self {
        let v = _mm256_loadu_ps(&_self.0);
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(&mut out, _mm256_add_ps(v, v));
        Vector8(out)
    }
}
```

## Why `_self`?

The name `_self` reminds you that:

1. You're not using the normal `self` keyword
2. The macro has transformed the function
3. You need to be explicit about the type

It's a deliberate choice to make the transformation visible.

## Generated Code

```rust
// You write:
impl Vector8 {
    #[arcane(_self = Vector8)]
    fn sum(&self, token: Desktop64) -> f32 {
        let v = _mm256_loadu_ps(&_self.0);
        let sum = _mm256_hadd_ps(v, v);
        let sum = _mm256_hadd_ps(sum, sum);
        let low = _mm256_castps256_ps128(sum);
        let high = _mm256_extractf128_ps::<1>(sum);
        _mm_cvtss_f32(_mm_add_ss(low, high))
    }
}

// Macro generates:
impl Vector8 {
    fn sum(&self, token: Desktop64) -> f32 {
        #[target_feature(enable = "avx2,fma,bmi1,bmi2")]
        #[inline]
        fn __inner(_self: &Vector8, token: Desktop64) -> f32 {
            let v = _mm256_loadu_ps(&_self.0);
            let sum = _mm256_hadd_ps(v, v);
            let sum = _mm256_hadd_ps(sum, sum);
            let low = _mm256_castps256_ps128(sum);
            let high = _mm256_extractf128_ps::<1>(sum);
            _mm_cvtss_f32(_mm_add_ss(low, high))
        }
        unsafe { __inner(self, token) }
    }
}
```

## Common Patterns

### Builder Pattern

```rust
impl ImageProcessor {
    #[arcane(_self = ImageProcessor)]
    fn with_brightness(self, token: Desktop64, amount: f32) -> Self {
        let mut result = _self;
        // Process brightness with SIMD...
        result
    }

    #[arcane(_self = ImageProcessor)]
    fn with_contrast(self, token: Desktop64, amount: f32) -> Self {
        let mut result = _self;
        // Process contrast with SIMD...
        result
    }
}

// Usage
let processed = processor
    .with_brightness(token, 1.2)
    .with_contrast(token, 1.1);
```

### Mutable Iteration

```rust
impl Buffer {
    #[arcane(_self = Buffer)]
    fn process_all(&mut self, token: Desktop64) {
        for chunk in _self.data.chunks_exact_mut(8) {
            let arr: &mut [f32; 8] = chunk.try_into().unwrap();
            let v = _mm256_loadu_ps(arr as &[f32; 8]);
            let result = _mm256_mul_ps(v, v);
            _mm256_storeu_ps(arr, result);
        }
    }
}
```
