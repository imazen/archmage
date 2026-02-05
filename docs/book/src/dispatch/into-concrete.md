# IntoConcreteToken Trait

`IntoConcreteToken` enables compile-time dispatch via monomorphization. Each token type returns `Some(self)` for its own type and `None` for others.

## Basic Usage

```rust
use archmage::{IntoConcreteToken, SimdToken, X64V3Token, NeonToken, ScalarToken};

fn process<T: IntoConcreteToken>(token: T, data: &mut [f32]) {
    // Compiler eliminates non-matching branches via monomorphization
    if let Some(t) = token.as_x64v3() {
        process_avx2(t, data);
    } else if let Some(t) = token.as_neon() {
        process_neon(t, data);
    } else if let Some(_) = token.as_scalar() {
        process_scalar(data);
    }
}
```

When called with `X64V3Token`, the compiler sees:
- `as_x64v3()` → `Some(token)` (takes this branch)
- `as_neon()` → `None` (eliminated)
- `as_scalar()` → `None` (eliminated)

## Available Methods

```rust
pub trait IntoConcreteToken: SimdToken {
    fn as_x64v2(self) -> Option<X64V2Token> { None }
    fn as_x64v3(self) -> Option<X64V3Token> { None }
    fn as_x64v4(self) -> Option<X64V4Token> { None }      // requires avx512
    fn as_avx512_modern(self) -> Option<Avx512ModernToken> { None }
    fn as_neon(self) -> Option<NeonToken> { None }
    fn as_neon_aes(self) -> Option<NeonAesToken> { None }
    fn as_neon_sha3(self) -> Option<NeonSha3Token> { None }
    fn as_wasm128(self) -> Option<Wasm128Token> { None }
    fn as_scalar(self) -> Option<ScalarToken> { None }
}
```

Each concrete token overrides its own method to return `Some(self)`.

## Upcasting with IntoConcreteToken

You can check if a token supports higher capabilities:

```rust
fn maybe_use_avx512<T: IntoConcreteToken>(token: T, data: &mut [f32]) {
    // Check if we actually have AVX-512
    if let Some(v4) = token.as_x64v4() {
        fast_path_avx512(v4, data);
    } else if let Some(v3) = token.as_x64v3() {
        normal_path_avx2(v3, data);
    }
}
```

**Note**: This creates an LLVM optimization boundary. The generic caller and feature-enabled callee have different target settings. Do this dispatch at entry points, not in hot code.

## Dispatch Order

Check from highest to lowest capability:

```rust
fn dispatch<T: IntoConcreteToken>(token: T, data: &[f32]) -> f32 {
    // Highest first
    #[cfg(feature = "avx512")]
    if let Some(t) = token.as_x64v4() {
        return process_v4(t, data);
    }

    if let Some(t) = token.as_x64v3() {
        return process_v3(t, data);
    }

    if let Some(t) = token.as_neon() {
        return process_neon(t, data);
    }

    if let Some(t) = token.as_wasm128() {
        return process_wasm(t, data);
    }

    // Scalar fallback
    process_scalar(data)
}
```

## vs incant!

| Feature | `IntoConcreteToken` | `incant!` |
|---------|---------------------|-----------|
| Dispatch style | Explicit if/else | Macro-generated |
| Token passing | Token already obtained | Summons tokens |
| Flexibility | Full control | Convention-based |
| Verbosity | More code | Less code |

Use `IntoConcreteToken` when you already have a token and need to specialize. Use `incant!` for entry-point dispatch.

## Example: Library with Generic Token API

```rust
use archmage::{IntoConcreteToken, SimdToken, arcane};

/// Public API accepts any token
pub fn transform<T: IntoConcreteToken>(token: T, data: &mut [f32]) {
    if let Some(t) = token.as_x64v3() {
        transform_avx2(t, data);
    } else if let Some(t) = token.as_neon() {
        transform_neon(t, data);
    } else {
        transform_scalar(data);
    }
}

#[arcane]
fn transform_avx2(token: X64V3Token, data: &mut [f32]) {
    // AVX2 implementation
}

#[arcane]
fn transform_neon(token: NeonToken, data: &mut [f32]) {
    // NEON implementation
}

fn transform_scalar(data: &mut [f32]) {
    // Scalar fallback
}
```

Callers can pass any token:

```rust
if let Some(token) = Desktop64::summon() {
    transform(token, &mut data);  // Uses AVX2 path
}

// Or force scalar for testing
transform(ScalarToken, &mut data);
```
