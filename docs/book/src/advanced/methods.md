# Methods with #[arcane]

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
use magetypes::f32x8;

struct Vector8([f32; 8]);

impl Vector8 {
    #[arcane(_self = Vector8)]
    fn magnitude(&self, token: Desktop64) -> f32 {
        // Use _self, not self
        let v = f32x8::from_array(token, _self.0);
        (v * v).reduce_add().sqrt()
    }
}
```

## All Receiver Types

### `&self` (Shared Reference)

```rust
impl Vector8 {
    #[arcane(_self = Vector8)]
    fn dot(&self, token: Desktop64, other: &Self) -> f32 {
        let a = f32x8::from_array(token, _self.0);
        let b = f32x8::from_array(token, other.0);
        (a * b).reduce_add()
    }
}
```

### `&mut self` (Mutable Reference)

```rust
impl Vector8 {
    #[arcane(_self = Vector8)]
    fn normalize(&mut self, token: Desktop64) {
        let v = f32x8::from_array(token, _self.0);
        let len = (v * v).reduce_add().sqrt();
        if len > 0.0 {
            let inv = f32x8::splat(token, 1.0 / len);
            let normalized = v * inv;
            _self.0 = normalized.to_array();
        }
    }
}
```

### `self` (By Value)

```rust
impl Vector8 {
    #[arcane(_self = Vector8)]
    fn scaled(self, token: Desktop64, factor: f32) -> Self {
        let v = f32x8::from_array(token, _self.0);
        let s = f32x8::splat(token, factor);
        Vector8((v * s).to_array())
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
        let v = f32x8::from_array(token, _self.0);
        Vector8((v + v).to_array())
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
    fn process(&self, token: Desktop64) -> f32 {
        f32x8::from_array(token, _self.0).reduce_add()
    }
}

// Macro generates:
impl Vector8 {
    fn process(&self, token: Desktop64) -> f32 {
        #[target_feature(enable = "avx2,fma,bmi1,bmi2")]
        #[inline]
        unsafe fn __inner(_self: &Vector8, token: Desktop64) -> f32 {
            f32x8::from_array(token, _self.0).reduce_add()
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
        // Process brightness...
        result
    }

    #[arcane(_self = ImageProcessor)]
    fn with_contrast(self, token: Desktop64, amount: f32) -> Self {
        let mut result = _self;
        // Process contrast...
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
            let v = f32x8::from_slice(token, chunk);
            let result = v * v;
            result.store_slice(chunk);
        }
    }
}
```
