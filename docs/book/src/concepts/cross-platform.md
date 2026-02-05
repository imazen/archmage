# Cross-Platform Code

Your archmage code compiles on all platforms. On unsupported platforms, `summon()` returns `None` and SIMD functions become unreachable stubs.

## The Simple Case

This works everywhere without `#[cfg]`:

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

On x86-64: `Desktop64::summon()` returns `Some(token)`, takes the SIMD path.
On ARM/WASM: Returns `None`, takes scalar path.

## Multi-Platform SIMD

For SIMD on multiple platforms, use `#[cfg]` for the implementations:

```rust
use archmage::{Desktop64, Arm64, SimdToken, rite};

pub fn process(data: &mut [f32]) {
    // x86-64
    if let Some(token) = Desktop64::summon() {
        return process_x86(token, data);
    }

    // ARM
    if let Some(token) = Arm64::summon() {
        return process_arm(token, data);
    }

    // Fallback
    process_scalar(data);
}

#[cfg(target_arch = "x86_64")]
#[rite]
fn process_x86(token: Desktop64, data: &mut [f32]) {
    use magetypes::simd::x86::f32x8;
    for chunk in data.chunks_exact_mut(8) {
        let v = f32x8::from_slice(token, chunk);
        (v * v).store_slice(chunk);
    }
}

#[cfg(target_arch = "aarch64")]
#[rite]
fn process_arm(token: Arm64, data: &mut [f32]) {
    use magetypes::simd::arm::f32x4;
    for chunk in data.chunks_exact_mut(4) {
        let v = f32x4::from_slice(token, chunk);
        (v * v).store_slice(chunk);
    }
}

fn process_scalar(data: &mut [f32]) {
    for x in data { *x *= *x; }
}
```

## Why Stubs Are Safe

On ARM, the `#[rite]` function for x86 becomes:

```rust
fn process_x86(token: Desktop64, data: &mut [f32]) {
    unreachable!("Desktop64 cannot exist on this architecture")
}
```

This can never execute because:
1. `Desktop64::summon()` returns `None` on ARM
2. You can't construct `Desktop64` any other way (safely)

## Token Existence vs Availability

Token **types** exist everywhere — only `summon()` behavior differs:

```rust
// On ARM, this compiles:
use archmage::{Desktop64, SimdToken};

// But this returns None:
let result = Desktop64::summon();  // None on ARM

// Check at compile time:
match Desktop64::guaranteed() {
    Some(true) => "Guaranteed available",
    Some(false) => "Wrong architecture",
    None => "Runtime check needed",
}
```

## ScalarToken

`ScalarToken` succeeds everywhere — use it for fallbacks:

```rust
use archmage::{ScalarToken, SimdToken};

let token = ScalarToken::summon().unwrap();  // Always works
// Or:
let token = ScalarToken;
```

## Using incant! for Cross-Platform

The `incant!` macro handles dispatch automatically:

```rust
use archmage::incant;

pub fn process(data: &[f32]) -> f32 {
    incant!(process(data))
    // Tries: process_v4 → process_v3 → process_neon → process_wasm128 → process_scalar
}

#[cfg(target_arch = "x86_64")]
#[rite]
fn process_v3(token: X64V3Token, data: &[f32]) -> f32 { ... }

#[cfg(target_arch = "aarch64")]
#[rite]
fn process_neon(token: NeonToken, data: &[f32]) -> f32 { ... }

fn process_scalar(data: &[f32]) -> f32 {
    data.iter().sum()
}
```

Only define the variants you need — `incant!` skips missing ones.
