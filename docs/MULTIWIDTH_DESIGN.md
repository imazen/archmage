# Multiwidth Macro Design

## Goal

Write SIMD code once, have it automatically specialized for SSE, AVX2, and AVX-512.

## Usage

```rust
use archmage::multiwidth;

#[multiwidth]
mod my_kernels {
    use crate::simd::*;  // imports f32xN, Token, LANES_F32, etc.

    pub fn normalize(token: Token, data: &mut [f32]) {
        let chunks = data.chunks_exact_mut(LANES_F32);
        for chunk in chunks {
            let v = f32xN::load(token, chunk.try_into().unwrap());
            let normalized = v * f32xN::splat(token, 1.0 / 255.0);
            normalized.store(chunk.try_into().unwrap());
        }
    }

    pub fn dot_product(token: Token, a: &[f32], b: &[f32]) -> f32 {
        let mut sum = f32xN::zero(token);
        for (ca, cb) in a.chunks_exact(LANES_F32).zip(b.chunks_exact(LANES_F32)) {
            let va = f32xN::load(token, ca.try_into().unwrap());
            let vb = f32xN::load(token, cb.try_into().unwrap());
            sum = sum + va * vb;
        }
        sum.reduce_add()
    }
}

// Generated output:
pub mod my_kernels {
    pub mod sse {
        use archmage::simd::sse::*;
        pub fn normalize(token: Token, data: &mut [f32]) { ... }
        pub fn dot_product(token: Token, a: &[f32], b: &[f32]) -> f32 { ... }
    }

    pub mod avx2 {
        use archmage::simd::avx2::*;
        pub fn normalize(token: Token, data: &mut [f32]) { ... }
        pub fn dot_product(token: Token, a: &[f32], b: &[f32]) -> f32 { ... }
    }

    #[cfg(feature = "avx512")]
    pub mod avx512 {
        use archmage::simd::avx512::*;
        pub fn normalize(token: Token, data: &mut [f32]) { ... }
        pub fn dot_product(token: Token, a: &[f32], b: &[f32]) -> f32 { ... }
    }

    /// Runtime dispatcher - picks the best available implementation
    pub fn normalize(data: &mut [f32]) {
        #[cfg(feature = "avx512")]
        if let Some(token) = archmage::X64V4Token::try_new() {
            return avx512::normalize(token, data);
        }
        if let Some(token) = archmage::X64V3Token::try_new() {
            return avx2::normalize(token, data);
        }
        if let Some(token) = archmage::Sse41Token::try_new() {
            return sse::normalize(token, data);
        }
        // Scalar fallback
        for x in data.iter_mut() {
            *x *= 1.0 / 255.0;
        }
    }
}
```

## Key Design Decisions

### 1. Module-level macro, not function-level

Using `#[multiwidth] mod name { ... }` instead of per-function because:
- Functions often share helper functions
- Shared constants and types
- Cleaner organization

### 2. Token parameter convention

Functions use `token: Token` where `Token` is aliased per-width:
- `sse::Token = Sse41Token`
- `avx2::Token = X64V3Token`
- `avx512::Token = X64V4Token`

### 3. Generic type aliases

Inside the module, these types are available:
- `f32xN`, `f64xN` - float vectors
- `i8xN` through `i64xN` - signed integer vectors
- `u8xN` through `u64xN` - unsigned integer vectors
- `LANES_F32`, `LANES_32`, etc. - lane count constants
- `Token` - the token type for construction

### 4. Dispatcher generation

The macro generates a dispatcher function that:
1. Tries AVX-512 first (if feature enabled)
2. Falls back to AVX2
3. Falls back to SSE
4. Falls back to scalar (if provided)

### 5. Scalar fallback

Optional `#[scalar_fallback]` attribute on functions provides fallback:

```rust
#[multiwidth]
mod kernels {
    #[scalar_fallback(normalize_scalar)]
    pub fn normalize(token: Token, data: &mut [f32]) { ... }
}

fn normalize_scalar(data: &mut [f32]) {
    for x in data { *x *= 1.0 / 255.0; }
}
```

## Implementation Plan

1. **Phase 1**: Width namespaces (DONE)
   - `simd::sse`, `simd::avx2`, `simd::avx512` with type aliases

2. **Phase 2**: Basic macro
   - Parse module, duplicate for each width
   - Replace `use crate::simd::*` with width-specific import

3. **Phase 3**: Dispatcher generation
   - Generate runtime dispatch functions
   - Support scalar fallback

4. **Phase 4**: ARM support
   - `simd::neon` namespace
   - Cross-platform dispatch

## Alternative: Per-function macro

```rust
#[multiwidth(sse, avx2, avx512)]
#[arcane]
fn process(token: impl HasSimd, data: &mut [f32]) {
    // Token type determines which f32xN is used
}
```

This is more complex because the token type needs to drive type inference.
