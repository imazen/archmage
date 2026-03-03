# Cross-Platform Behavior

Archmage lets you write x86 SIMD code that compiles on ARM and vice versa. By default, functions are **cfg'd out** on non-matching architectures (no dead code). Opt into unreachable stubs with `#[arcane(stub)]` when you need cross-arch dispatch without `#[cfg]` guards.

## Default: Cfg-Out

By default, `#[arcane]` and `#[rite]` only emit code on the matching architecture. On other architectures, the function simply doesn't exist.

```rust
use archmage::prelude::*;

// This function only exists on x86_64
#[arcane]
fn avx2_kernel(_token: Desktop64, data: &[f32; 8]) -> [f32; 8] {
    let v = _mm256_loadu_ps(data);
    // ...
}
```

Code that references `avx2_kernel` on ARM won't compile — the function isn't there. Use `#[cfg(target_arch)]` guards or `incant!` for cross-platform dispatch.

## Opt-In Stubs with `stub`

`#[arcane(stub)]` generates an `unreachable!()` stub on non-matching architectures. The stub compiles but can never execute.

```rust
#[arcane(stub)]
fn avx2_kernel(_token: Desktop64, data: &[f32; 8]) -> [f32; 8] {
    let v = _mm256_loadu_ps(data);
    // ...
}
```

On **ARM/WASM**, you get:

```rust
fn avx2_kernel(token: Desktop64, data: &[f32; 8]) -> [f32; 8] {
    unreachable!("Desktop64 cannot exist on this architecture")
}
```

### Why Stubs Are Safe

The stub can never execute because:

1. `Desktop64::summon()` returns `None` on ARM
2. You can't construct `Desktop64` any other way (safely)
3. The only path to `avx2_kernel` is through a token you can't obtain

### When to Use Stubs

Use `#[arcane(stub)]` when you reference both x86 and ARM functions in the same dispatch block without `#[cfg]` guards:

```rust
#[arcane(stub)]
fn process_avx2(token: Desktop64, data: &mut [f32]) { /* ... */ }

#[arcane(stub)]
fn process_neon(token: NeonToken, data: &mut [f32]) { /* ... */ }

// Both referenced without #[cfg] — stubs make this compile everywhere
pub fn process(data: &mut [f32]) {
    if let Some(token) = Desktop64::summon() {
        return process_avx2(token, data);
    }
    if let Some(token) = NeonToken::summon() {
        return process_neon(token, data);
    }
    process_scalar(data);
}
```

## Preferred: Use `incant!` or `#[cfg]`

The recommended approach is to use `incant!` (which cfg-gates dispatch calls automatically) or explicit `#[cfg]` guards — no stubs needed:

```rust
use archmage::incant;

#[cfg(target_arch = "x86_64")]
#[arcane]
fn process_v3(token: archmage::X64V3Token, data: &mut [f32]) { /* ... */ }

#[cfg(target_arch = "aarch64")]
#[arcane]
fn process_neon(token: archmage::NeonToken, data: &mut [f32]) { /* ... */ }

fn process_scalar(_token: archmage::ScalarToken, data: &mut [f32]) { /* ... */ }

pub fn process(data: &mut [f32]) {
    incant!(process(data), [v3, neon])
}
```

## `#[rite]` Also Supports `stub`

`#[rite(stub)]` works the same way for inner helpers.

## Token Existence vs Token Availability

All token **types** exist on all platforms:

```rust
// These types compile on ARM:
use archmage::{Desktop64, X64V3Token, X64V4Token};

// But summon() returns None:
assert!(Desktop64::summon().is_none());  // On ARM

// And compiled_with() tells you:
assert_eq!(Desktop64::compiled_with(), Some(false));  // Wrong arch
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
    if let Some(token) = Desktop64::summon() {
        let result = process_with_token(token, &data);
        assert_eq!(result, expected);
    }
}
```

## Migration from Stub Default

If your code previously relied on `#[arcane]` generating stubs on wrong architectures:

1. **Add `stub`**: Change `#[arcane]` to `#[arcane(stub)]`
2. **Add `#[cfg]` guards**: Wrap cross-arch references in `#[cfg(target_arch)]`
3. **Use `incant!`**: Let the dispatch macro handle cfg-gating (recommended)
