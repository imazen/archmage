+++
title = "Your First Types"
weight = 2
+++

Every magetypes vector requires a token for construction. The pattern is: **summon a token, construct vectors, operate, extract results.**

## The Pattern

```rust
use archmage::{Desktop64, SimdToken};
use magetypes::simd::f32x8;

fn main() {
    // 1. Summon: prove the CPU supports AVX2+FMA
    if let Some(token) = Desktop64::summon() {

        // 2. Construct: token is the first argument
        let a = f32x8::from_array(token, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let b = f32x8::splat(token, 2.0);

        // 3. Operate: natural Rust operators
        let c = a * b;

        // 4. Extract: get scalar results back
        let result: [f32; 8] = c.to_array();
        println!("{:?}", result);
        // [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
    }
}
```

## Why Tokens?

The token proves CPU support exists. Without it, constructing an `f32x8` on a CPU without AVX2 would produce garbage or crash. The type system prevents this at compile time — you cannot call `f32x8::splat()` without a token that guarantees the right features.

Tokens are zero-sized. Passing them around costs nothing at runtime. Construction functions need the token; once you have the vector, operations like `+`, `*`, `reduce_add()` don't need it again.

## Summon Once, Use Many

You don't need to summon a token every time you construct a vector. Summon once, pass it around:

```rust
use archmage::{Desktop64, SimdToken, arcane};
use magetypes::simd::f32x8;

#[arcane]
fn process_data(token: Desktop64, input: &[f32; 8]) -> f32 {
    let a = f32x8::from_array(token, *input);
    let b = f32x8::splat(token, 0.5);  // Same token, no re-detection
    let scaled = a * b;
    scaled.reduce_add()
}

fn main() {
    if let Some(token) = Desktop64::summon() {
        let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = process_data(token, &data);
        println!("sum of halved values: {}", result);
    }
}
```

`#[arcane]` generates `#[target_feature]` attributes from the token type, so SIMD intrinsics are safe inside the function. See [The #\[arcane\] Macro](@/archmage/concepts/arcane.md) for details.

## Different Tokens, Different Types

The token determines what hardware you're targeting:

```rust
use archmage::{X64V2Token, Desktop64, NeonToken, SimdToken};
use magetypes::simd::{f32x4, f32x8};

fn example_x86_128bit() {
    // 128-bit SSE4.2 — available on most x86-64 CPUs (2008+)
    if let Some(token) = X64V2Token::summon() {
        let v = f32x4::from_array(token, [1.0, 2.0, 3.0, 4.0]);
        println!("sum: {}", v.reduce_add());
    }
}

fn example_x86_256bit() {
    // 256-bit AVX2+FMA — Haswell 2013+, Zen 1+
    if let Some(token) = Desktop64::summon() {
        let v = f32x8::from_array(token, [1.0; 8]);
        println!("sum: {}", v.reduce_add());
    }
}

fn example_arm() {
    // 128-bit NEON — all 64-bit ARM
    if let Some(token) = NeonToken::summon() {
        let v = f32x4::from_array(token, [1.0, 2.0, 3.0, 4.0]);
        println!("sum: {}", v.reduce_add());
    }
}
```

## Type Properties

All magetypes SIMD types are:

- **Copy** — pass by value freely, no moves
- **Clone** — explicit `.clone()` works
- **Debug** — `println!("{:?}", v)` for debugging
- **Send + Sync** — safe to share across threads

```rust
let a = f32x8::splat(token, 1.0);
let b = a;       // Copy, not move
let c = a + b;   // Both still valid
```

## Next Steps

- [Type Overview](@/magetypes/types/overview.md) — full list of available types per platform
- [Arithmetic & Comparisons](@/magetypes/operations/operators.md) — operators, FMA, min/max
- [Reductions](@/magetypes/operations/reductions.md) — `reduce_add`, `reduce_max`, `reduce_min`
