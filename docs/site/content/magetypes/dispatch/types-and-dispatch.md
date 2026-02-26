+++
title = "Types and Dispatch"
weight = 1
+++

Magetypes vectors work with archmage's [`incant!`](@/archmage/dispatch/incant.md) macro for multi-platform dispatch. The generic backend pattern lets you write one function body that works across all architectures — `incant!` selects the best available token at runtime.

## The Generic Pattern

Write your SIMD function once with a generic backend bound. `incant!` dispatches to it with the best available token:

```rust
use archmage::{arcane, incant};
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Backend,
};

#[arcane]
fn sum_impl<T: F32x8Backend>(token: T, data: &[f32; 8]) -> f32 {
    f32x8::<T>::from_array(token, *data).reduce_add()
}

pub fn sum(data: &[f32; 8]) -> f32 {
    incant!(sum_impl(data))
}
```

`incant!` tries tokens from best to worst (V4 → V3 → NEON → WASM → scalar) and calls the first available variant. With the generic pattern, the same function body handles all of them.

## When Algorithms Differ Per Platform

Sometimes you want separate implementations to exploit architecture-specific strengths — different register widths, native instruction sequences, or algorithm shapes:

```rust
use archmage::{arcane, incant, X64V3Token, NeonToken};
use magetypes::simd::generic::{f32x8, f32x4};

// x86-64: use f32x8 (256-bit AVX2)
#[arcane]
fn dot_product_v3(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let va = f32x8::<X64V3Token>::from_array(token, *a);
    let vb = f32x8::<X64V3Token>::from_array(token, *b);
    (va * vb).reduce_add()
}

// AArch64: use f32x4 (128-bit NEON) — process in two halves
#[arcane]
fn dot_product_neon(token: NeonToken, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let sum1 = {
        let va = f32x4::<NeonToken>::from_slice(token, &a[0..4]);
        let vb = f32x4::<NeonToken>::from_slice(token, &b[0..4]);
        (va * vb).reduce_add()
    };
    let sum2 = {
        let va = f32x4::<NeonToken>::from_slice(token, &a[4..8]);
        let vb = f32x4::<NeonToken>::from_slice(token, &b[4..8]);
        (va * vb).reduce_add()
    };
    sum1 + sum2
}

fn dot_product_scalar(a: &[f32; 8], b: &[f32; 8]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn dot_product(a: &[f32; 8], b: &[f32; 8]) -> f32 {
    incant!(dot_product(a, b), [v3, neon, wasm128])
}
```

When the algorithm is the same on every platform, the generic pattern is cleaner. When you need platform-tuned implementations, write concrete variants and name them explicitly.

## Using Polyfill Types

If your algorithm works the same regardless of register width, the generic pattern automatically uses polyfills on narrower hardware. There's nothing special to do — `f32x8::<NeonToken>` is already the polyfill:

```rust
use archmage::{arcane, incant};
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Backend,
};

// One function — works on all platforms
#[arcane]
fn sum_impl<T: F32x8Backend>(token: T, data: &[f32; 8]) -> f32 {
    // On AVX2: one 256-bit operation
    // On NEON: two 128-bit NEON operations (polyfill)
    // On WASM: two 128-bit SIMD128 operations (polyfill)
    f32x8::<T>::from_array(token, *data).reduce_add()
}

pub fn sum(data: &[f32; 8]) -> f32 {
    incant!(sum_impl(data))
}
```

The polyfill version is simpler to write than separate platform variants. For most code, the performance difference is negligible.

## The #[magetypes] Macro

When the function body is truly platform-independent (only the token type changes), use [`#[magetypes]`](@/archmage/dispatch/magetypes-macro.md) to generate all variants automatically:

```rust
use archmage::{magetypes, incant};

#[magetypes]
fn validate(token: Token, threshold: f32) -> bool {
    // Token is replaced with X64V3Token, NeonToken, ScalarToken, etc.
    threshold > 0.0
}

// Generates: validate_v3, validate_neon, validate_wasm128, validate_scalar
// Ready for incant!:
pub fn validate(threshold: f32) -> bool {
    incant!(validate(threshold), [v3, neon, wasm128])
}
```

`#[magetypes]` does text substitution — `Token` becomes the concrete token type for each variant. It's useful when the token is the only platform-dependent part. When your function body uses specific SIMD operations that differ by platform, use the generic pattern with `F32x8Backend` bounds, or write concrete variants manually.

## Passthrough Dispatch

When you already have a token and want to dispatch to specialized variants without re-summoning:

```rust
use archmage::{incant, IntoConcreteToken};

fn process_inner<T: IntoConcreteToken>(token: T, data: &[f32]) -> f32 {
    incant!(compute(data) with token, [v3, neon, wasm128])
    // Uses IntoConcreteToken to check what the token actually is
}
```

See [`incant!` Passthrough Mode](@/archmage/dispatch/incant.md) for details.
