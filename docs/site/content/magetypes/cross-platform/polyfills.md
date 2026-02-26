+++
title = "Polyfills"
weight = 1
+++

Magetypes lets you write code using wider types (like `f32x8`) even on hardware with narrower registers (128-bit NEON or WASM SIMD128). The polyfill layer handles this transparently.

## How It Works

On x86-64 with AVX2, `f32x8` maps directly to a single 256-bit `__m256` register. On AArch64 NEON (128-bit registers), the same `f32x8` type is implemented as two `f32x4` operations internally. Every method — `+`, `reduce_add()`, `splat()`, etc. — works identically regardless of the underlying implementation.

The generic type `f32x8<T>` is parameterized by a backend token. Different backends produce different implementations:

```rust
use magetypes::simd::{
    generic::f32x8,
    backends::{x64v3, neon, scalar},
};

// x86-64 AVX2: maps to a single __m256 register
let a = f32x8::<x64v3>::splat(token, 1.0);

// AArch64 NEON: internally two f32x4 NEON operations
let a = f32x8::<neon>::splat(token, 1.0);

// Scalar fallback: eight individual f32 values
let a = f32x8::<scalar>::splat(token, 1.0);
```

Write your code once using a generic backend bound — the right implementation is selected at the call site:

```rust
use magetypes::simd::{
    generic::f32x8,
    backends::F32x8Backend,
};

fn example<T: F32x8Backend>(token: T) {
    let a = f32x8::<T>::splat(token, 1.0);
    let b = f32x8::<T>::splat(token, 2.0);
    let c = a + b;       // Two vaddq_f32 on NEON, one vaddps on AVX2
    let sum = c.reduce_add();
}
```

## Pick the Right Size for Your Algorithm

The polyfill approach means you pick the vector width that matches your algorithm, not your hardware:

- **Processing 8 floats at a time?** Use `f32x8`. On ARM, it's two NEON ops — still faster than scalar.
- **Processing 4 floats at a time?** Use `f32x4`. Native on all platforms.
- **Processing 16 floats at a time?** Use `f32x16` (requires `avx512` feature). On ARM, it's four NEON ops.

Wider polyfills have overhead (2x or 4x the instruction count) but the overhead is constant and predictable. For data-parallel workloads, using `f32x8` on ARM is still substantially faster than scalar `f32` code.

## implementation_name()

Every magetypes vector has an `implementation_name()` associated function that returns a string identifying the actual implementation. It's an associated function, not a method — call it on the type, not on a value:

```rust
use magetypes::simd::{
    generic::f32x8,
    backends::{x64v3, neon},
};

println!("{}", f32x8::<x64v3>::implementation_name());
// "x86::v3::f32x8"

println!("{}", f32x8::<neon>::implementation_name());
// "polyfill::neon::f32x8"
```

| Platform | f32x8 implementation_name |
|----------|--------------------------|
| x86-64 (AVX2) | `"x86::v3::f32x8"` |
| AArch64 (NEON) | `"polyfill::neon::f32x8"` |
| WASM | `"polyfill::wasm128::f32x8"` |

For native-width types, the prefix reflects the platform directly:

| Platform | f32x4 implementation_name |
|----------|--------------------------|
| x86-64 | `"x86::v2::f32x4"` |
| AArch64 | `"arm::neon::f32x4"` |
| WASM | `"wasm::wasm128::f32x4"` |

This is useful for debugging and logging — you can verify which code path is actually running.

## Polyfill Tiers

| Width | x86-64 | AArch64 | WASM |
|-------|--------|---------|------|
| 128-bit (f32x4, i32x4, ...) | Native (SSE/AVX) | Native (NEON) | Native (SIMD128) |
| 256-bit (f32x8, i32x8, ...) | Native (AVX2) | Polyfill (2x NEON) | Polyfill (2x SIMD128) |
| 512-bit (f32x16, i32x16, ...) | Native (AVX-512)* | Polyfill (4x NEON) | Polyfill (4x SIMD128) |

*512-bit types require the `avx512` feature flag.

## What's Polyfilled, What's Not

The polyfill layer covers:
- All arithmetic operators (`+`, `-`, `*`, `/`, negation)
- FMA (`mul_add`, `mul_sub`)
- Comparisons (`simd_lt`, `simd_eq`, etc.)
- Reductions (`reduce_add`, `reduce_max`, `reduce_min`)
- Construction and extraction (`from_array`, `to_array`, `splat`, etc.)
- Transcendentals (`exp`, `log2`, `ln`, etc.)
- Bitwise operations
- Conversions

The API is identical. The only difference is the number of hardware instructions emitted.
