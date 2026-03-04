+++
title = "Methods with #[arcane]"
weight = 1
+++

`#[arcane]` works naturally on methods. Sibling expansion (the default) keeps both functions in the same `impl` block, so `self` and `Self` work without any special handling.

## Inherent Methods (Default — Just Works)

For methods on your own types, `#[arcane]` works like it does on free functions:

```rust
use archmage::{X64V3Token, arcane};

struct Vector8([f32; 8]);

impl Vector8 {
    #[arcane(import_intrinsics)]
    fn magnitude(&self, token: X64V3Token) -> f32 {
        // self and Self work naturally!
        let sum: f32 = self.0.iter().map(|x| x * x).sum();
        sum.sqrt()
    }

    #[arcane(import_intrinsics)]
    fn scale(&mut self, token: X64V3Token, factor: f32) {
        for v in self.0.iter_mut() {
            *v *= factor;
        }
    }

    #[arcane(import_intrinsics)]
    fn into_sum(self, token: X64V3Token) -> f32 {
        self.0.iter().sum()
    }

    // Self in return type works too
    #[arcane(import_intrinsics)]
    fn doubled(&self, token: X64V3Token) -> Self {
        let mut data = [0.0f32; 8];
        for i in 0..8 {
            data[i] = self.0[i] * 2.0;
        }
        Self(data)
    }

    // Self::CONSTANT works
    #[arcane(import_intrinsics)]
    fn with_offset(&self, token: X64V3Token) -> [f32; 8] {
        let mut out = self.0;
        for v in out.iter_mut() {
            *v += Self::OFFSET;
        }
        out
    }

    const OFFSET: f32 = 10.0;
}
```

### What Gets Generated

```rust
impl Vector8 {
    // Sibling: both live in the same impl block
    #[cfg(target_arch = "x86_64")]
    #[doc(hidden)]
    #[target_feature(enable = "avx2,fma,...")]
    unsafe fn __arcane_magnitude(&self, token: X64V3Token) -> f32 {
        let sum: f32 = self.0.iter().map(|x| x * x).sum();
        sum.sqrt()
    }

    #[cfg(target_arch = "x86_64")]
    fn magnitude(&self, token: X64V3Token) -> f32 {
        unsafe { self.__arcane_magnitude(token) }
    }
}
```

Both functions are in the same `impl` block, so `self`, `Self`, and associated constants resolve correctly.

## Trait Implementations (Require `_self = Type`)

Sibling expansion adds `__arcane_fn` to the impl block — but that method isn't in the trait definition. The compiler rejects it. **Trait impls must use nested mode** via `_self = Type`.

`_self = Type` implies nested mode and renames `self` → `_self` with the concrete type:

```rust
use archmage::{X64V3Token, arcane};

trait SimdOps {
    fn compute(&self, token: X64V3Token) -> f32;
    fn transform(&self, token: X64V3Token) -> Self;
}

struct Point { x: f32, y: f32 }

impl SimdOps for Point {
    #[arcane(_self = Point, import_intrinsics)]
    fn compute(&self, token: X64V3Token) -> f32 {
        // Use _self, not self (nested mode renames self → _self)
        _self.x * _self.x + _self.y * _self.y
    }

    #[arcane(_self = Point, import_intrinsics)]
    fn transform(&self, token: X64V3Token) -> Self {
        Point {
            x: _self.x * 2.0,
            y: _self.y * 2.0,
        }
    }
}
```

### What Gets Generated

```rust
impl SimdOps for Point {
    fn compute(&self, token: X64V3Token) -> f32 {
        #[cfg(target_arch = "x86_64")]
        #[target_feature(enable = "avx2,fma,...")]
        #[inline]
        fn __inner(_self: &Point, token: X64V3Token) -> f32 {
            use core::arch::x86_64::*;
            use safe_unaligned_simd::x86_64::*;
            _self.x * _self.x + _self.y * _self.y
        }
        #[cfg(target_arch = "x86_64")]
        { unsafe { __inner(self, token) } }
    }
}
```

The inner `fn` can't have a `self` receiver (Rust doesn't allow that in inner function items), so the macro passes `self` as a regular parameter named `_self` with the concrete type you specified.

### Why `_self`?

The name reminds you:

1. You're not using the normal `self` keyword
2. The macro has transformed `self` into a concrete-typed parameter
3. You need `_self = Type` to tell the macro what type `self` is

### All Receiver Types with `_self`

#### `&self` (Shared Reference)

```rust
#[arcane(_self = Vector8, import_intrinsics)]
fn dot(&self, token: X64V3Token, other: &Self) -> f32 {
    let a = _mm256_loadu_ps(&_self.0);
    let b = _mm256_loadu_ps(&other.0);
    // ...
}
```

#### `&mut self` (Mutable Reference)

```rust
#[arcane(_self = Vector8, import_intrinsics)]
fn normalize(&mut self, token: X64V3Token) {
    // _self is &mut Vector8
    _mm256_storeu_ps(&mut _self.0, normalized);
}
```

#### `self` (By Value)

```rust
#[arcane(_self = Vector8, import_intrinsics)]
fn scaled(self, token: X64V3Token, factor: f32) -> Self {
    // _self is Vector8 (owned)
    let v = _mm256_loadu_ps(&_self.0);
    // ...
    Vector8(out)
}
```

## When to Use Which

| Context | Approach | `self`/`Self` |
|---------|----------|---------------|
| Inherent method (`impl MyType`) | `#[arcane(import_intrinsics)]` (default sibling) | Just works |
| Trait impl (`impl Trait for Type`) | `#[arcane(_self = Type, import_intrinsics)]` (nested) | Use `_self` |
| Explicit nested | `#[arcane(nested, import_intrinsics)]` | Use `_self` if accessing self |
| Free function | `#[arcane(import_intrinsics)]` (default sibling) | N/A |

## Common Patterns

### Builder Pattern

```rust
impl ImageProcessor {
    #[arcane(import_intrinsics)]
    fn with_brightness(self, token: X64V3Token, amount: f32) -> Self {
        // self/Self work naturally in sibling mode
        let mut result = self;
        // Process brightness with SIMD...
        result
    }
}

let processed = processor
    .with_brightness(token, 1.2)
    .with_contrast(token, 1.1);
```

### Mutable Iteration

```rust
impl Buffer {
    #[arcane(import_intrinsics)]
    fn process_all(&mut self, token: X64V3Token) {
        for chunk in self.data.chunks_exact_mut(8) {
            let arr: &mut [f32; 8] = chunk.try_into().unwrap();
            let v = _mm256_loadu_ps(arr as &[f32; 8]);
            let result = _mm256_mul_ps(v, v);
            _mm256_storeu_ps(arr, result);
        }
    }
}
```
