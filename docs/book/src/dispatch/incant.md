# incant! Macro

`incant!` automates dispatch to suffixed function variants. Write one call, get automatic fallback through capability tiers.

## Basic Usage

```rust
use archmage::{incant, rite, X64V3Token, NeonToken};
use magetypes::prelude::*;

// Define variants with standard suffixes
#[cfg(target_arch = "x86_64")]
#[rite]
fn sum_v3(token: X64V3Token, data: &[f32]) -> f32 {
    let mut total = f32x8::zero(token);
    for chunk in data.chunks_exact(8) {
        total = total + f32x8::from_slice(token, chunk);
    }
    total.reduce_add() + data.chunks_exact(8).remainder().iter().sum::<f32>()
}

#[cfg(target_arch = "aarch64")]
#[rite]
fn sum_neon(token: NeonToken, data: &[f32]) -> f32 {
    let mut total = f32x4::zero(token);
    for chunk in data.chunks_exact(4) {
        total = total + f32x4::from_slice(token, chunk);
    }
    total.reduce_add() + data.chunks_exact(4).remainder().iter().sum::<f32>()
}

fn sum_scalar(data: &[f32]) -> f32 {
    data.iter().sum()
}

// Dispatch automatically
pub fn sum(data: &[f32]) -> f32 {
    incant!(sum(data))
    // Tries: sum_v4 → sum_v3 → sum_neon → sum_wasm128 → sum_scalar
}
```

## How It Works

`incant!` expands to a dispatch chain that tries each variant in order:

```rust
// incant!(process(data)) expands to approximately:
'__incant: {
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(token) = X64V4Token::summon() {
        break '__incant process_v4(token, data);
    }

    #[cfg(target_arch = "x86_64")]
    if let Some(token) = X64V3Token::summon() {
        break '__incant process_v3(token, data);
    }

    #[cfg(target_arch = "aarch64")]
    if let Some(token) = NeonToken::summon() {
        break '__incant process_neon(token, data);
    }

    #[cfg(target_arch = "wasm32")]
    if let Some(token) = Simd128Token::summon() {
        break '__incant process_wasm128(token, data);
    }

    process_scalar(data)
}
```

## Suffix Conventions

| Suffix | Token | Platform |
|--------|-------|----------|
| `_v4` | `X64V4Token` | x86-64 AVX-512 |
| `_v3` | `X64V3Token` | x86-64 AVX2+FMA |
| `_v2` | `X64V2Token` | x86-64 SSE4.2 |
| `_neon` | `NeonToken` | AArch64 |
| `_wasm128` | `Simd128Token` | WASM |
| `_scalar` | — | Fallback (always tried last) |

Define only the variants you need — `incant!` skips missing ones.

## Passthrough Mode

When you already have a token and want to dispatch to specialized variants:

```rust
use archmage::{incant, IntoConcreteToken};

fn outer<T: IntoConcreteToken>(token: T, data: &[f32]) -> f32 {
    // Passthrough: use existing token to pick best variant
    incant!(token => process(data))
}
```

This uses `IntoConcreteToken` to check the token's type and dispatch accordingly, without re-summoning.

## Complete Example

```rust
use archmage::{incant, rite, X64V3Token, NeonToken, SimdToken};
use magetypes::prelude::*;

// Public API
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    incant!(dot_product(a, b))
}

// AVX2 variant
#[cfg(target_arch = "x86_64")]
#[rite]
fn dot_product_v3(token: X64V3Token, a: &[f32], b: &[f32]) -> f32 {
    let mut sum = f32x8::zero(token);
    let chunks_a = a.chunks_exact(8);
    let chunks_b = b.chunks_exact(8);

    for (ca, cb) in chunks_a.clone().zip(chunks_b.clone()) {
        let va = f32x8::from_slice(token, ca);
        let vb = f32x8::from_slice(token, cb);
        sum = va.mul_add(vb, sum);
    }

    let mut total = sum.reduce_add();
    for (x, y) in chunks_a.remainder().iter().zip(chunks_b.remainder()) {
        total += x * y;
    }
    total
}

// NEON variant
#[cfg(target_arch = "aarch64")]
#[rite]
fn dot_product_neon(token: NeonToken, a: &[f32], b: &[f32]) -> f32 {
    let mut sum = f32x4::zero(token);
    let chunks_a = a.chunks_exact(4);
    let chunks_b = b.chunks_exact(4);

    for (ca, cb) in chunks_a.clone().zip(chunks_b.clone()) {
        let va = f32x4::from_slice(token, ca);
        let vb = f32x4::from_slice(token, cb);
        sum = va.mul_add(vb, sum);
    }

    let mut total = sum.reduce_add();
    for (x, y) in chunks_a.remainder().iter().zip(chunks_b.remainder()) {
        total += x * y;
    }
    total
}

// Scalar fallback
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
```

## When to Use incant!

**Use incant! when:**
- You have multiple platform-specific implementations
- You want automatic fallback through tiers
- Function signatures are similar across variants

**Use manual dispatch when:**
- You need custom fallback logic
- Variants have different signatures
- You want more explicit control
