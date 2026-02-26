+++
title = "Your First Types"
weight = 2
+++

Every magetypes vector requires a token for construction. The pattern is: **write a generic function over a backend trait, summon a token, call the function.**

## The Pattern

Write the computation as a generic function bounded on the backend trait. The token parameter gates construction; once you have the vector, operators work without it.

```rust
use archmage::{Desktop64, SimdToken};
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Backend,
};

fn scale_and_sum<T: F32x8Backend>(token: T, input: &[f32; 8]) -> f32 {
    // 1. Construct: token is the first argument, turbofish selects the backend
    let a = f32x8::<T>::from_array(token, *input);
    let b = f32x8::<T>::splat(token, 2.0);

    // 2. Operate: natural Rust operators
    let c = a * b;

    // 3. Extract: get scalar results back
    c.reduce_add()
}

fn main() {
    // 4. Summon: prove the CPU supports AVX2+FMA
    if let Some(token) = Desktop64::summon() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0f32];
        let result = scale_and_sum(token, &data);
        println!("{}", result);  // 72.0
    }
}
```

The turbofish `f32x8::<T>` is required on constructors because Rust can't infer the backend from a consumed token value. Once the vector exists, methods like `*`, `reduce_add()`, and `to_array()` resolve without annotation.

## Why Tokens?

The token proves CPU support exists. Without it, constructing an `f32x8` on a CPU without AVX2 would produce garbage or crash. The type system prevents this at compile time — you cannot call `f32x8::<T>::splat()` without a token of type `T`, and `T` can only be summoned if the CPU supports the required features.

Tokens are zero-sized. Passing them around costs nothing at runtime. Construction functions need the token; once you have the vector, operations like `+`, `*`, `reduce_add()` don't need it again.

## Summon Once, Use Many

You don't need to summon a token every time you construct a vector. Summon once at the dispatch boundary, pass the token into generic SIMD code:

```rust
use archmage::{Desktop64, SimdToken, arcane};
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Backend,
};

fn process_data<T: F32x8Backend>(token: T, input: &[f32; 8]) -> f32 {
    let a = f32x8::<T>::from_array(token, *input);
    let b = f32x8::<T>::splat(token, 0.5);  // Same token, no re-detection
    let scaled = a * b;
    scaled.reduce_add()
}

#[arcane]
fn process_avx2(token: Desktop64, input: &[f32; 8]) -> f32 {
    process_data(token, input)
}

fn main() {
    if let Some(token) = Desktop64::summon() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0f32];
        let result = process_avx2(token, &data);
        println!("sum of halved values: {}", result);  // 18.0
    }
}
```

`#[arcane]` generates `#[target_feature]` attributes from the token type, so SIMD intrinsics are safe inside the function. See [The #\[arcane\] Macro](@/archmage/concepts/arcane.md) for details.

## Different Tokens, Different Types

The backend type parameter determines what hardware you're targeting. The same generic function works across all of them:

```rust
use archmage::{Desktop64, ScalarToken, SimdToken};
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Backend,
};

fn sum8<T: F32x8Backend>(token: T, data: &[f32; 8]) -> f32 {
    f32x8::<T>::from_array(token, *data).reduce_add()
}

fn dispatch_sum8(data: &[f32; 8]) -> f32 {
    // Scalar is always available — no summon needed
    let scalar = ScalarToken::new();

    #[cfg(target_arch = "x86_64")]
    if let Some(token) = Desktop64::summon() {
        return sum8(token, data);
    }

    sum8(scalar, data)
}
```

On AArch64, `NeonToken` backs the NEON implementation. On x86-64, `Desktop64` gives AVX2+FMA. On any platform, `ScalarToken` falls back to portable scalar code. The generic function `sum8` compiles correctly for each.

## Concrete Backends

When you need to name a specific backend type (e.g., in a `use` or a type annotation), the `backends` module exports short aliases:

```rust
use magetypes::simd::backends::{x64v3, neon, scalar};
use magetypes::simd::generic::f32x8;

// Explicit concrete types for type annotations or static dispatch
type F32x8Avx2  = f32x8<x64v3>;
type F32x8Neon  = f32x8<neon>;
type F32x8Scalar = f32x8<scalar>;
```

These aliases (`x64v3`, `neon`, `wasm128`, `scalar`, etc.) are simply re-exports of the archmage token types under shorter names.

## Implementation Name

Concrete specializations expose `implementation_name()` to confirm which backend is active:

```rust
use magetypes::simd::generic::f32x8;

fn show_impl() {
    #[cfg(target_arch = "x86_64")]
    println!("{}", f32x8::<archmage::X64V3Token>::implementation_name());
    // "x86::v3::f32x8"
}
```

This is an associated function on the concrete specialization, not on the generic type or on a vector value.

## Type Properties

All magetypes SIMD types are:

- **Copy** — pass by value freely, no moves
- **Clone** — explicit `.clone()` works
- **Debug** — `println!("{:?}", v)` for debugging
- **Send + Sync** — safe to share across threads

```rust
use magetypes::simd::{generic::f32x8, backends::F32x8Backend};

fn copy_example<T: F32x8Backend>(token: T) {
    let a = f32x8::<T>::splat(token, 1.0);
    let b = a;       // Copy, not move
    let c = a + b;   // Both still valid
    let _ = c;
}
```

## Performance Note

The generic pattern (`f32x8::<T>`) produces **identical assembly** to concrete types (`f32x8::<x64v3>`) when called from inside `#[arcane]` or `#[rite]`. All backend methods are `#[inline(always)]` — LLVM inlines them fully. There is zero abstraction cost.

The only requirement: your generic function must be called from within a `#[target_feature]` context (via `#[arcane]` or `#[rite]`). Without it, intrinsics become function calls and performance drops ~18x. See [Polyfills — Performance](@/magetypes/cross-platform/polyfills.md#performance-generic-concrete-inside-arcane) for benchmark data.

## Next Steps

- [Type Overview](@/magetypes/types/overview.md) — full list of available types per platform
- [Arithmetic & Comparisons](@/magetypes/operations/operators.md) — operators, FMA, min/max
- [Reductions](@/magetypes/operations/reductions.md) — `reduce_add`, `reduce_max`, `reduce_min`
