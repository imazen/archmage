# The #[magetypes] Macro

`#[magetypes]` generates platform-specific function variants by replacing `Token` with each concrete token type. It works with `incant!` to provide both generation and dispatch.

## When to Use

Use `#[magetypes]` for functions that don't use platform-specific SIMD types — where `Token` is the only platform-dependent part. If your function body uses `f32x8` or `__m256`, write the variants manually and use `incant!` directly.

## Basic Usage

```rust
use archmage::magetypes;

#[magetypes]
fn process(token: Token, data: &[f32]) -> f32 {
    // Token is replaced with X64V3Token, NeonToken, ScalarToken, etc.
    // Generates: process_v3, process_neon, process_scalar, ...
    data.iter().sum()
}
```

The macro generates one function per platform variant, each with the concrete token type substituted in:

| Generated Function | Token Type |
|-------------------|------------|
| `process_v3` | `X64V3Token` |
| `process_v4` | `X64V4Token` |
| `process_neon` | `NeonToken` |
| `process_wasm128` | `Wasm128Token` |
| `process_scalar` | `ScalarToken` |

These suffixed functions are exactly what `incant!` expects:

```rust
pub fn process(data: &[f32]) -> f32 {
    incant!(process(data))
}
```

## What Gets Replaced

`#[magetypes]` does text substitution on the token stream. Inside a `#[magetypes]` function:

| Placeholder | Replaced With |
|-------------|---------------|
| `Token` | Concrete token type (`X64V3Token`, `NeonToken`, etc.) |
| `f32xN` | Platform-native f32 vector (`f32x8`, `f32x4`, etc.) |
| `LANES` | Lane count for the platform (`8`, `4`, etc.) |

Case-sensitive — `Token` is replaced, `token` is not.

## `#[magetypes]` vs Manual Variants

**Use `#[magetypes]`** when the function body is platform-independent (only the token type changes):

```rust
#[magetypes]
fn validate(token: Token, threshold: f32) -> bool {
    // Token is the only platform-dependent part
    threshold > 0.0
}
```

**Write manual variants** when the function body uses platform-specific types or different algorithms per platform:

```rust
#[cfg(target_arch = "x86_64")]
fn process_v3(token: X64V3Token, data: &[f32]) -> f32 {
    // Uses f32x8 — 8-wide AVX2 algorithm
    let v = f32x8::from_array(token, data[..8].try_into().unwrap());
    v.reduce_add()
}

fn process_scalar(_token: ScalarToken, data: &[f32]) -> f32 {
    // Completely different algorithm
    data.iter().sum()
}
```

In practice, SIMD functions almost always need different types and algorithms per platform, so `incant!` with manual variants is the more common pattern.
