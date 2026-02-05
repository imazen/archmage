# Manual Dispatch

The simplest dispatch pattern: check for tokens explicitly, call the appropriate implementation.

## Basic Pattern

```rust
use archmage::{Desktop64, SimdToken};

pub fn process(data: &mut [f32]) {
    if let Some(token) = Desktop64::summon() {
        process_avx2(token, data);
    } else {
        process_scalar(data);
    }
}

#[arcane]
fn process_avx2(token: Desktop64, data: &mut [f32]) {
    // AVX2 implementation
}

fn process_scalar(data: &mut [f32]) {
    // Scalar fallback
}
```

## Multi-Tier Dispatch

Check from highest to lowest capability:

```rust
use archmage::{X64V4Token, X64V3Token, X64V2Token, SimdToken};

pub fn process(data: &mut [f32]) {
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    if let Some(token) = X64V4Token::summon() {
        return process_v4(token, data);
    }

    #[cfg(target_arch = "x86_64")]
    if let Some(token) = X64V3Token::summon() {
        return process_v3(token, data);
    }

    #[cfg(target_arch = "x86_64")]
    if let Some(token) = X64V2Token::summon() {
        return process_v2(token, data);
    }

    process_scalar(data);
}
```

## Cross-Platform Dispatch

```rust
use archmage::{Desktop64, NeonToken, Simd128Token, SimdToken};

pub fn process(data: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = Desktop64::summon() {
        return process_x86(token, data);
    }

    #[cfg(target_arch = "aarch64")]
    if let Some(token) = NeonToken::summon() {
        return process_arm(token, data);
    }

    #[cfg(target_arch = "wasm32")]
    if let Some(token) = Simd128Token::summon() {
        return process_wasm(token, data);
    }

    process_scalar(data);
}
```

## When to Use Manual Dispatch

**Pros:**
- Explicit and readable
- Full control over fallback logic
- Easy to understand

**Cons:**
- Verbose for many tiers
- Easy to forget a tier
- Repeated boilerplate

For complex dispatch with many tiers, consider [`incant!`](./incant.md) or [`IntoConcreteToken`](./into-concrete.md).

## Avoiding Common Mistakes

### Don't Dispatch in Hot Loops

```rust
// WRONG
for chunk in data.chunks_mut(8) {
    if let Some(token) = Desktop64::summon() {  // CPUID every iteration!
        process_chunk(token, chunk);
    }
}

// RIGHT
if let Some(token) = Desktop64::summon() {
    for chunk in data.chunks_mut(8) {
        process_chunk(token, chunk);  // Token hoisted
    }
} else {
    for chunk in data.chunks_mut(8) {
        process_chunk_scalar(chunk);
    }
}
```

### Don't Forget Early Returns

```rust
// WRONG - falls through to scalar even when SIMD available
if let Some(token) = Desktop64::summon() {
    process_avx2(token, data);
    // Missing return!
}
process_scalar(data);  // Always runs!

// RIGHT
if let Some(token) = Desktop64::summon() {
    return process_avx2(token, data);
}
process_scalar(data);
```
