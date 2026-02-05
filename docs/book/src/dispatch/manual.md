# Manual Dispatch

The simplest dispatch pattern: check for tokens explicitly, call the appropriate implementation.

## Basic Pattern

```rust
use archmage::{Desktop64, SimdToken, rite};
use magetypes::f32x8;

pub fn process(data: &mut [f32]) {
    if let Some(token) = Desktop64::summon() {
        process_simd(token, data);
    } else {
        process_scalar(data);
    }
}

#[rite]
fn process_simd(token: Desktop64, data: &mut [f32]) {
    for chunk in data.chunks_exact_mut(8) {
        let v = f32x8::from_slice(token, chunk);
        (v * v).store_slice(chunk);
    }
}

fn process_scalar(data: &mut [f32]) {
    for x in data { *x *= *x; }
}
```

That's it. No `#[cfg(target_arch)]` needed — this compiles and runs everywhere.

## Cross-Platform Dispatch

On unsupported architectures, `summon()` returns `None`. Your dispatch logic handles it naturally:

```rust
use archmage::{Desktop64, Arm64, Simd128Token, SimdToken, rite};

pub fn process(data: &mut [f32]) {
    // x86-64 AVX2
    if let Some(token) = Desktop64::summon() {
        return process_x86(token, data);
    }

    // ARM NEON
    if let Some(token) = Arm64::summon() {
        return process_arm(token, data);
    }

    // WASM SIMD
    if let Some(token) = Simd128Token::summon() {
        return process_wasm(token, data);
    }

    // Scalar fallback
    process_scalar(data);
}

#[cfg(target_arch = "x86_64")]
#[rite]
fn process_x86(token: Desktop64, data: &mut [f32]) { /* ... */ }

#[cfg(target_arch = "aarch64")]
#[rite]
fn process_arm(token: Arm64, data: &mut [f32]) { /* ... */ }

#[cfg(target_arch = "wasm32")]
#[rite]
fn process_wasm(token: Simd128Token, data: &mut [f32]) { /* ... */ }

fn process_scalar(data: &mut [f32]) { /* ... */ }
```

## Tiered x86 Dispatch

Check from highest to lowest capability:

```rust
use archmage::{X64V4Token, Desktop64, X64V2Token, SimdToken, rite};

pub fn process(data: &mut [f32]) {
    // AVX-512 (requires avx512 cargo feature)
    #[cfg(feature = "avx512")]
    if let Some(token) = X64V4Token::summon() {
        return process_v4(token, data);
    }

    // AVX2+FMA
    if let Some(token) = Desktop64::summon() {
        return process_v3(token, data);
    }

    // SSE4.2
    if let Some(token) = X64V2Token::summon() {
        return process_v2(token, data);
    }

    process_scalar(data);
}
```

## When to Use Manual Dispatch

**Use manual dispatch when:**
- You have 2-3 tiers
- Different tiers have different APIs or algorithms
- You want explicit, readable control flow

**Consider [`incant!`](./incant.md) when:**
- You have many tiers with similar signatures
- You want automatic best-available selection

## Common Mistakes

### Don't Forget Early Returns

```rust
// WRONG — falls through!
if let Some(token) = Desktop64::summon() {
    process_simd(token, data);
    // Missing return!
}
process_scalar(data);  // Always runs!

// RIGHT
if let Some(token) = Desktop64::summon() {
    return process_simd(token, data);
}
process_scalar(data);
```

### Don't Put Dispatch Inside Loops

```rust
// WRONG — wrapper overhead every iteration
if let Some(token) = Desktop64::summon() {
    for chunk in data.chunks_mut(8) {
        process_chunk(token, chunk);  // #[rite] call from non-#[rite] context
    }
}

// RIGHT — put the loop inside the SIMD function
if let Some(token) = Desktop64::summon() {
    process_all(token, data);
}

#[rite]
fn process_all(token: Desktop64, data: &mut [f32]) {
    for chunk in data.chunks_exact_mut(8) {
        // Now we're calling #[rite] from #[rite] context — full inlining
        process_chunk(token, chunk);
    }
}
```
