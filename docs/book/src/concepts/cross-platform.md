# Cross-Platform Stubs

Archmage lets you write x86 SIMD code that compiles on ARM and vice versa. Functions become unreachable stubs on non-matching architectures.

## How It Works

When you write:

```rust
use archmage::prelude::*;

#[arcane]
fn avx2_kernel(_token: Desktop64, data: &[f32; 8]) -> [f32; 8] {
    // x86-64 SIMD code - safe_unaligned_simd takes references
    let v = _mm256_loadu_ps(data);
    // ...
}
```

On **x86-64**, you get the real implementation.

On **ARM/WASM**, you get:

```rust
fn avx2_kernel(token: Desktop64, data: &[f32; 8]) -> [f32; 8] {
    unreachable!("Desktop64 cannot exist on this architecture")
}
```

## Why This Is Safe

The stub can never execute because:

1. `Desktop64::summon()` returns `None` on ARM
2. You can't construct `Desktop64` any other way (safely)
3. The only path to `avx2_kernel` is through a token you can't obtain

```rust
fn process(data: &[f32; 8]) -> [f32; 8] {
    if let Some(token) = Desktop64::summon() {
        avx2_kernel(token, data)  // Never reached on ARM
    } else {
        scalar_fallback(data)     // ARM takes this path
    }
}
```

## Writing Cross-Platform Libraries

Structure your code with platform-specific implementations:

```rust
// Public API - works everywhere
pub fn process(data: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = Desktop64::summon() {
        return process_avx2(token, data);
    }

    #[cfg(target_arch = "aarch64")]
    if let Some(token) = NeonToken::summon() {
        return process_neon(token, data);
    }

    process_scalar(data);
}

#[cfg(target_arch = "x86_64")]
#[arcane]
fn process_avx2(token: Desktop64, data: &mut [f32]) {
    // AVX2 implementation
}

#[cfg(target_arch = "aarch64")]
#[arcane]
fn process_neon(token: NeonToken, data: &mut [f32]) {
    // NEON implementation
}

fn process_scalar(data: &mut [f32]) {
    // Works everywhere
}
```

## Token Existence vs Token Availability

All token **types** exist on all platforms:

```rust
// These types compile on ARM:
use archmage::{Desktop64, X64V3Token, X64V4Token};

// But summon() returns None:
assert!(Desktop64::summon().is_none());  // On ARM

// And guaranteed() tells you:
assert_eq!(Desktop64::guaranteed(), Some(false));  // Wrong arch
```

This enables cross-platform code without `#[cfg]` soup:

```rust
// Compiles everywhere, dispatches at runtime
fn process<T: IntoConcreteToken>(token: T, data: &[f32]) {
    if let Some(t) = token.as_x64v3() {
        process_v3(t, data);
    } else if let Some(t) = token.as_neon() {
        process_neon(t, data);
    } else {
        process_scalar(data);
    }
}
```

## The ScalarToken Escape Hatch

`ScalarToken` works everywhere:

```rust
use archmage::{ScalarToken, SimdToken};

// Always succeeds, any platform
let token = ScalarToken::summon().unwrap();

// Or just construct it
let token = ScalarToken;
```

Use it for fallback paths that need a token for API consistency:

```rust
fn must_have_token<T: SimdToken>(token: T, data: &[f32]) -> f32 {
    // ...
}

// On platforms without SIMD:
let result = must_have_token(ScalarToken, &data);
```

## Testing Cross-Platform Code

Test your dispatch logic without needing every CPU:

```rust
#[test]
fn test_scalar_fallback() {
    // Force scalar path even on AVX2 machine
    let token = ScalarToken;
    let result = process_with_token(token, &data);
    assert_eq!(result, expected);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_avx2_path() {
    if let Some(token) = Desktop64::summon() {
        let result = process_with_token(token, &data);
        assert_eq!(result, expected);
    }
}
```
