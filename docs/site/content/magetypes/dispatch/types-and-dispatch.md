+++
title = "Types and Dispatch"
weight = 1
+++

Magetypes vectors work with archmage's [`incant!`](@/archmage/dispatch/incant.md) macro for multi-platform dispatch. Write platform-specific variants using the right types for each architecture, dispatch with a single call.

## The Pattern

Each platform gets a function variant using its native vector types. `incant!` picks the best available at runtime:

```rust
use archmage::{arcane, incant, X64V3Token, NeonToken, SimdToken};
use magetypes::simd::{f32x8, f32x4};

// x86-64: use f32x8 (256-bit AVX2)
#[arcane]
fn sum_v3(token: X64V3Token, data: &[f32; 8]) -> f32 {
    f32x8::from_array(token, *data).reduce_add()
}

// AArch64: use f32x4 (128-bit NEON) — process in two halves
#[arcane]
fn sum_neon(token: NeonToken, data: &[f32; 8]) -> f32 {
    let a = f32x4::from_slice(token, &data[0..4]);
    let b = f32x4::from_slice(token, &data[4..8]);
    a.reduce_add() + b.reduce_add()
}

// Scalar fallback — no SIMD, no token
fn sum_scalar(data: &[f32; 8]) -> f32 {
    data.iter().sum()
}

// Dispatch: tries v3 -> neon -> wasm128 -> scalar
pub fn sum(data: &[f32; 8]) -> f32 {
    incant!(sum(data))
}
```

## When Algorithms Differ Per Platform

Different register widths often mean different algorithms. On x86-64 with AVX2, you can process 8 floats in one operation. On ARM NEON, you process 4. The algorithm shape changes:

```rust
#[arcane]
fn dot_product_v3(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let va = f32x8::from_array(token, *a);
    let vb = f32x8::from_array(token, *b);
    (va * vb).reduce_add()
}

#[arcane]
fn dot_product_neon(token: NeonToken, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    // Two 4-wide multiplies, two reductions, final add
    let sum1 = {
        let va = f32x4::from_slice(token, &a[0..4]);
        let vb = f32x4::from_slice(token, &b[0..4]);
        (va * vb).reduce_add()
    };
    let sum2 = {
        let va = f32x4::from_slice(token, &a[4..8]);
        let vb = f32x4::from_slice(token, &b[4..8]);
        (va * vb).reduce_add()
    };
    sum1 + sum2
}

fn dot_product_scalar(a: &[f32; 8], b: &[f32; 8]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn dot_product(a: &[f32; 8], b: &[f32; 8]) -> f32 {
    incant!(dot_product(a, b))
}
```

## Using Polyfill Types Instead

If your algorithm works the same regardless of register width, you can use the [polyfill types](@/magetypes/cross-platform/polyfills.md) and write one implementation:

```rust
// f32x8 on ARM is polyfilled with two f32x4 ops
#[arcane]
fn sum_v3(token: X64V3Token, data: &[f32; 8]) -> f32 {
    f32x8::from_array(token, *data).reduce_add()
}

#[arcane]
fn sum_neon(token: NeonToken, data: &[f32; 8]) -> f32 {
    // f32x8 works on ARM too — polyfilled with two NEON ops
    f32x8::from_array(token, *data).reduce_add()
}
```

The polyfill version is simpler to write but may be slightly less efficient than a hand-tuned native implementation. For most code, the difference is negligible.

## The #[magetypes] Macro

When the function body is truly platform-independent (only the token type changes), use [`#[magetypes]`](@/archmage/dispatch/magetypes-macro.md) to generate all variants automatically:

```rust
use archmage::magetypes;

#[magetypes]
fn validate(token: Token, threshold: f32) -> bool {
    // Token is replaced with X64V3Token, NeonToken, ScalarToken, etc.
    threshold > 0.0
}

// Generates: validate_v3, validate_neon, validate_wasm128, validate_scalar
// Ready for incant!:
pub fn validate(threshold: f32) -> bool {
    incant!(validate(threshold))
}
```

`#[magetypes]` does text substitution — `Token` becomes the concrete token type for each variant. It's useful when the token is the only platform-dependent part. When your function body uses platform-specific types like `f32x8` or different algorithms per platform, write the variants manually.

## Passthrough Dispatch

When you already have a token and want to dispatch to specialized variants without re-summoning:

```rust
use archmage::{incant, IntoConcreteToken};

fn process_inner<T: IntoConcreteToken>(token: T, data: &[f32]) -> f32 {
    incant!(token => compute(data))
    // Uses IntoConcreteToken to check what the token actually is
}
```

See [`incant!` Passthrough Mode](@/archmage/dispatch/incant.md) for details.
