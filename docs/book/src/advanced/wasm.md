# WASM SIMD

WebAssembly SIMD128 provides 128-bit vectors in the browser and WASI environments.

## Setup

Enable SIMD128 in your build:

```bash
RUSTFLAGS="-Ctarget-feature=+simd128" cargo build --target wasm32-unknown-unknown
```

Or in `.cargo/config.toml`:

```toml
[target.wasm32-unknown-unknown]
rustflags = ["-Ctarget-feature=+simd128"]
```

## The Token

```rust
use archmage::{Simd128Token, SimdToken};

fn check_wasm_simd() {
    if let Some(token) = Simd128Token::summon() {
        process_simd(token, &data);
    } else {
        process_scalar(&data);
    }
}
```

**Note**: On WASM, `Simd128Token::summon()` succeeds if the binary was compiled with SIMD128 support. There's no runtime feature detection in WASM—the capability is determined at compile time.

## Available Types

| Type | Elements |
|------|----------|
| `f32x4` | 4 × f32 |
| `f64x2` | 2 × f64 |
| `i32x4` | 4 × i32 |
| `i64x2` | 2 × i64 |
| `i16x8` | 8 × i16 |
| `i8x16` | 16 × i8 |
| `u32x4` | 4 × u32 |
| `u64x2` | 2 × u64 |
| `u16x8` | 8 × u16 |
| `u8x16` | 16 × u8 |

## Basic Usage

```rust
use archmage::{Simd128Token, arcane};
use magetypes::simd::f32x4;

#[arcane]
fn dot_product(token: Simd128Token, a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let va = f32x4::from_array(token, *a);
    let vb = f32x4::from_array(token, *b);
    (va * vb).reduce_add()
}
```

## Cross-Platform Code

Write once, run on x86, ARM, and WASM:

```rust
use archmage::{Desktop64, NeonToken, Simd128Token, SimdToken, incant};

// Define platform-specific implementations
#[cfg(target_arch = "x86_64")]
#[arcane]
fn sum_v3(token: Desktop64, data: &[f32; 8]) -> f32 {
    use magetypes::simd::f32x8;
    f32x8::from_array(token, *data).reduce_add()
}

#[cfg(target_arch = "aarch64")]
#[arcane]
fn sum_neon(token: NeonToken, data: &[f32; 8]) -> f32 {
    use magetypes::simd::f32x4;
    let a = f32x4::from_slice(token, &data[0..4]);
    let b = f32x4::from_slice(token, &data[4..8]);
    a.reduce_add() + b.reduce_add()
}

#[cfg(target_arch = "wasm32")]
#[arcane]
fn sum_wasm128(token: Simd128Token, data: &[f32; 8]) -> f32 {
    use magetypes::simd::f32x4;
    let a = f32x4::from_slice(token, &data[0..4]);
    let b = f32x4::from_slice(token, &data[4..8]);
    a.reduce_add() + b.reduce_add()
}

fn sum_scalar(data: &[f32; 8]) -> f32 {
    data.iter().sum()
}

// Public API
pub fn sum(data: &[f32; 8]) -> f32 {
    incant!(sum(data))
}
```

## WASM-Specific Considerations

### No Runtime Detection

Unlike x86/ARM, WASM doesn't have runtime feature detection. The SIMD support is baked in at compile time:

```rust
// On WASM, this is always the same result
// (based on compile-time -Ctarget-feature=+simd128)
let has_simd = Simd128Token::summon().is_some();
```

### Browser Compatibility

WASM SIMD is supported in:
- Chrome 91+ (May 2021)
- Firefox 89+ (June 2021)
- Safari 16.4+ (March 2023)
- Node.js 16.4+

For older browsers, provide a non-SIMD fallback WASM binary.

### Relaxed SIMD

WASM also has "relaxed SIMD" with even more instructions. As of 2024, this requires additional flags:

```bash
RUSTFLAGS="-Ctarget-feature=+simd128,+relaxed-simd" cargo build
```

## Example: Image Processing in Browser

```rust
use wasm_bindgen::prelude::*;
use archmage::{Simd128Token, SimdToken, arcane};
use magetypes::simd::u8x16;

#[wasm_bindgen]
pub fn brighten_image(pixels: &mut [u8], amount: u8) {
    if let Some(token) = Simd128Token::summon() {
        brighten_simd(token, pixels, amount);
    } else {
        brighten_scalar(pixels, amount);
    }
}

#[arcane]
fn brighten_simd(token: Simd128Token, pixels: &mut [u8], amount: u8) {
    let add = u8x16::splat(token, amount);

    for chunk in pixels.chunks_exact_mut(16) {
        let v = u8x16::from_slice(token, chunk);
        let bright = v.saturating_add(add);
        bright.store_slice(chunk);
    }

    // Handle remainder
    for pixel in pixels.chunks_exact_mut(16).into_remainder() {
        *pixel = pixel.saturating_add(amount);
    }
}

fn brighten_scalar(pixels: &mut [u8], amount: u8) {
    for pixel in pixels {
        *pixel = pixel.saturating_add(amount);
    }
}
```

## Testing WASM Code

Use `wasm-pack test`:

```bash
wasm-pack test --node
```

Or test natively with the scalar fallback:

```rust
#[test]
fn test_sum() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let result = sum(&data);
    assert_eq!(result, 36.0);
}
```
