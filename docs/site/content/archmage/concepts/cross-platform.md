+++
title = "Cross-Platform Behavior"
weight = 5
+++

Archmage lets you write x86 SIMD code that compiles on ARM and vice versa. `#[arcane]` and `#[rite]` automatically cfg-gate their output to the matching architecture — you don't need to add `#[cfg(target_arch)]` yourself.

## Default: Cfg-Out

`#[arcane]` and `#[rite]` only emit code on the matching architecture. On other architectures, no function is generated — no dead code, no stubs.

```rust
use archmage::prelude::*;

// #[arcane] wraps output in #[cfg(target_arch = "x86_64")] — you don't need to
#[arcane(import_intrinsics)]
fn avx2_kernel(_token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    let v = _mm256_loadu_ps(data);  // In scope from import_intrinsics
    // ...
}
```

The function simply doesn't exist on ARM/WASM. This is fine for `incant!` dispatch (which cfg-gates its calls automatically) and for manual dispatch behind `summon()`. The only case where it matters is direct references to the function *by name* outside of dispatch — those need either `stub` or a `#[cfg]` guard on the *call site*.

## `incant!` — No Guards Needed

`incant!` wraps every tier call in `#[cfg(target_arch)]` internally, so the cfg'd-out functions are never referenced on wrong platforms. Just write the functions and dispatch:

```rust
use archmage::incant;

#[archmage::arcane]
fn process_v3(token: archmage::X64V3Token, data: &mut [f32]) { /* AVX2 */ }

#[archmage::arcane]
fn process_neon(token: archmage::NeonToken, data: &mut [f32]) { /* NEON */ }

fn process_scalar(_token: archmage::ScalarToken, data: &mut [f32]) { /* fallback */ }

pub fn process(data: &mut [f32]) {
    incant!(process(data), [v3, neon, scalar])
}
```

No `#[cfg]` on the function definitions. No stubs. `incant!` handles it.

## Manual Dispatch with `summon()`

Manual dispatch also works without stubs or `#[cfg]`, because the function is only reachable through a token that can't be obtained on the wrong architecture:

```rust
pub fn process(data: &mut [f32]) {
    if let Some(token) = X64V3Token::summon() {
        return process_avx2(token, data);  // only compiles on x86_64
    }
    process_scalar(data);
}
```

Wait — that won't compile on ARM because `process_avx2` doesn't exist. Two options:

### Option A: Use `stub`

```rust
#[arcane(stub, import_intrinsics)]
fn process_avx2(token: X64V3Token, data: &mut [f32]) { /* ... */ }

#[arcane(stub, import_intrinsics)]
fn process_neon(token: NeonToken, data: &mut [f32]) { /* ... */ }

// Both referenced by name — stubs make this compile everywhere
pub fn process(data: &mut [f32]) {
    if let Some(token) = X64V3Token::summon() {
        return process_avx2(token, data);
    }
    if let Some(token) = NeonToken::summon() {
        return process_neon(token, data);
    }
    process_scalar(data);
}
```

### Option B: Guard the call site

```rust
#[arcane(import_intrinsics)]
fn process_avx2(token: X64V3Token, data: &mut [f32]) { /* ... */ }

#[arcane(import_intrinsics)]
fn process_neon(token: NeonToken, data: &mut [f32]) { /* ... */ }

pub fn process(data: &mut [f32]) {
    // #[cfg] on the CALL, not the function definition
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = X64V3Token::summon() {
        return process_avx2(token, data);
    }
    #[cfg(target_arch = "aarch64")]
    if let Some(token) = NeonToken::summon() {
        return process_neon(token, data);
    }
    process_scalar(data);
}
```

### Option C: Use `incant!` (recommended)

```rust
// No #[cfg], no stubs, no ceremony
pub fn process(data: &mut [f32]) {
    incant!(process(data), [v3, neon, scalar])
}
```

## `#[rite]` Also Supports `stub`

`#[rite(stub)]` works the same way for `#[rite]` functions. This applies to all three `#[rite]` modes — token-based, tier-based (`#[rite(v3, stub)]`), and multi-tier (`#[rite(v3, v4, neon, stub)]` — each variant gets its own stub on wrong architectures):

```rust
#[rite(v3, stub)]
fn helper(val: f32) -> f32 {
    val * 2.0
}
```

## Token Existence vs Token Availability

All token **types** exist on all platforms:

```rust
// These types compile on ARM:
use archmage::{X64V3Token, X64V4Token};

// But summon() returns None:
assert!(X64V3Token::summon().is_none());  // On ARM

// And compiled_with() tells you:
assert_eq!(X64V3Token::compiled_with(), Some(false));  // Wrong arch
```

## The ScalarToken Escape Hatch

`ScalarToken` works everywhere:

```rust
use archmage::{ScalarToken, SimdToken};

let token = ScalarToken::summon().unwrap();
let token = ScalarToken;  // or just construct it
```

## Testing Cross-Platform Code

```rust
#[test]
fn test_scalar_fallback() {
    let token = ScalarToken;
    let result = process_with_token(token, &data);
    assert_eq!(result, expected);
}

#[test]
fn test_avx2_path() {
    // summon() returns None on non-x86, so test is skipped naturally
    if let Some(token) = X64V3Token::summon() {
        let result = process_with_token(token, &data);
        assert_eq!(result, expected);
    }
}
```

## Migration from Stub Default

If your code previously relied on `#[arcane]` generating stubs on wrong architectures:

1. **Use `incant!`**: Handles cfg-gating automatically (recommended)
2. **Add `stub`**: `#[arcane(stub)]` restores the old behavior
3. **Guard the call site**: `#[cfg]` on the reference, not the function
