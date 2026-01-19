# archmage

Type-safe SIMD capability tokens for Rust. Isolates `unsafe` to token construction, enabling safe SIMD code at usage sites.

## The Problem

Using SIMD safely in Rust is hard:

1. **`#[target_feature]` functions require unsafe to call** - even from multiversioned functions
2. **`cfg!(target_feature)` is crate-level** - doesn't optimize inside `#[target_feature]` functions
3. **Runtime dispatch loses type safety** - no compile-time proof features are available
4. **Crates like `wide` use cfg-gating** - they emit SSE even inside AVX2-enabled functions

## The Solution: Capability Tokens + `#[simd_fn]`

Tokens are zero-sized proof types. The `#[simd_fn]` macro makes raw intrinsics safe:

```rust
use archmage::{Avx2Token, SimdToken, simd_fn};
use std::arch::x86_64::*;

// #[simd_fn] makes raw intrinsics safe - token proves AVX2 is available!
#[simd_fn]
fn double_avx2(token: Avx2Token, data: &[f32; 8]) -> [f32; 8] {
    let v = unsafe { _mm256_loadu_ps(data.as_ptr()) };  // load needs unsafe (pointer)
    let doubled = _mm256_add_ps(v, v);                   // arithmetic is safe!
    let mut out = [0.0f32; 8];
    unsafe { _mm256_storeu_ps(out.as_mut_ptr(), doubled) };
    out
}

fn main() {
    if let Some(token) = Avx2Token::try_new() {
        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let output = double_avx2(token, &input);
        // output = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
    }
}
```

The macro expands to:
```rust
fn double_avx2(token: Avx2Token, data: &[f32; 8]) -> [f32; 8] {
    #[target_feature(enable = "avx2")]
    unsafe fn __inner(data: &[f32; 8]) -> [f32; 8] { /* body */ }
    let _ = &token;  // prove we have the token
    unsafe { __inner(data) }  // SAFETY: token proves avx2 available
}
```

## Usage Patterns

### Pattern 1: `#[simd_fn]` with Raw Intrinsics (Most Ergonomic)

Use `#[simd_fn]` to write raw intrinsics with token-based safety:

```rust
use archmage::{Avx2FmaToken, SimdToken, simd_fn};
use std::arch::x86_64::*;

#[simd_fn]
fn fma_kernel(token: Avx2FmaToken, a: &[f32; 8], b: &[f32; 8], c: &[f32; 8]) -> [f32; 8] {
    let va = unsafe { _mm256_loadu_ps(a.as_ptr()) };
    let vb = unsafe { _mm256_loadu_ps(b.as_ptr()) };
    let vc = unsafe { _mm256_loadu_ps(c.as_ptr()) };

    // FMA intrinsic is safe - #[simd_fn] adds target_feature
    let result = _mm256_fmadd_ps(va, vb, vc);  // a * b + c

    let mut out = [0.0f32; 8];
    unsafe { _mm256_storeu_ps(out.as_mut_ptr(), result) };
    out
}
```

Profile tokens automatically enable all required features:

```rust
#[simd_fn]
fn v3_kernel(token: X64V3Token, data: &mut [f32]) {
    // AVX2 + FMA + BMI1 + BMI2 all enabled!
}
```

### Pattern 2: Token-Gated Wrapper Operations

archmage provides wrapped operations that require tokens:

```rust
use archmage::{Avx2Token, Avx2FmaToken, SimdToken, ops};

fn example() {
    if let Some(token) = Avx2FmaToken::try_new() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = [2.0f32; 8];

        // Load (safe via token)
        let va = ops::load_f32x8(token.avx2(), &a);
        let vb = ops::load_f32x8(token.avx2(), &b);

        // Arithmetic (safe via token)
        let sum = ops::add_f32x8(token.avx2(), va, vb);
        let product = ops::mul_f32x8(token.avx2(), va, vb);

        // FMA: a * b + c (safe via token)
        let ones = ops::set1_f32x8(token.avx2(), 1.0);
        let fma_result = ops::fmadd_f32x8(token.fma(), va, vb, ones);

        // Shuffle/permute (safe via token)
        let shuffled = ops::shuffle_f32x8::<0b00_01_10_11>(token.avx2(), va, vb);
        let blended = ops::blend_f32x8::<0b10101010>(token.avx2(), va, vb);

        // Store result
        let mut out = [0.0f32; 8];
        ops::store_f32x8(token.avx2(), &mut out, fma_result);
    }
}
```

### Pattern 2: Profile Tokens (match multiversed presets)

Use profile tokens for x86-64 microarchitecture levels:

```rust
use archmage::{X64V3Token, SimdToken, ops};

fn process_v3(data: &mut [f32]) {
    // X64V3 = AVX2 + FMA + BMI2 (Haswell 2013+, Zen 1+)
    if let Some(v3) = X64V3Token::try_new() {
        // Extract any sub-token you need
        let avx2 = v3.avx2();
        let fma = v3.fma();
        let avx2_fma = v3.avx2_fma();  // Combined token

        for chunk in data.chunks_exact_mut(8) {
            let v = ops::load_f32x8(avx2, chunk.try_into().unwrap());
            let squared = ops::mul_f32x8(avx2, v, v);
            ops::store_f32x8(avx2, chunk.try_into().unwrap(), squared);
        }
    }
}
```

Available profile tokens:
| Token | Features | Hardware |
|-------|----------|----------|
| `X64V2Token` | SSE4.2 + POPCNT | Nehalem 2008+, Bulldozer 2011+ |
| `X64V3Token` | AVX2 + FMA + BMI2 | Haswell 2013+, Zen 1 2017+ |
| `X64V4Token` | AVX-512 (F/BW/CD/DQ/VL) | Xeon 2017+, Zen 4 2022+ |

### Pattern 3: Inside Multiversioned Functions

Use token macros inside `#[multiversed]` functions:

```rust
use archmage::{x64v3_token, ops};
use multiversed::multiversed;

#[multiversed]
fn kernel(data: &mut [f32]) {
    // Token creation - justified by multiversed context
    let token = x64v3_token!();
    let avx2 = token.avx2();

    for chunk in data.chunks_exact_mut(8) {
        let v = ops::load_f32x8(avx2, chunk.try_into().unwrap());
        let result = ops::mul_f32x8(avx2, v, v);
        ops::store_f32x8(avx2, chunk.try_into().unwrap(), result);
    }
}
```

### Pattern 4: Raw Intrinsics (Manual Alternative)

If you can't use `#[simd_fn]`, you can write the pattern manually:

```rust
use archmage::{Avx2Token, SimdToken};
use std::arch::x86_64::*;

fn advanced_shuffle(token: Avx2Token, data: &[f32; 8]) -> [f32; 8] {
    #[target_feature(enable = "avx2")]
    unsafe fn inner(data: &[f32; 8]) -> [f32; 8] {
        let v = _mm256_loadu_ps(data.as_ptr());
        let permuted = _mm256_permute_ps::<0b00_01_10_11>(v);
        let shuffled = _mm256_shuffle_ps::<0b10_11_00_01>(v, permuted);
        let mut out = [0.0f32; 8];
        _mm256_storeu_ps(out.as_mut_ptr(), shuffled);
        out
    }
    let _ = token;  // proves AVX2 available
    unsafe { inner(data) }
}
```

**Prefer `#[simd_fn]`** - it does this automatically and is less error-prone.

### Pattern 5: With `wide` Crate

Enable the `wide` feature for portable SIMD types:

```rust
use archmage::{Avx2Token, Avx2FmaToken, SimdToken};
use wide::f32x8;

fn with_wide() {
    if let Some(token) = Avx2FmaToken::try_new() {
        let a = f32x8::splat(2.0);
        let b = f32x8::splat(3.0);
        let c = f32x8::splat(1.0);

        // Token-gated wide operations
        let sum = token.add_f32x8_wide(a, b);      // 5.0
        let product = token.mul_f32x8_wide(a, b);  // 6.0
        let fma = token.fma_f32x8_wide(a, b, c);   // 2*3+1 = 7.0
    }
}
```

### Pattern 6: Composite Operations

High-level operations built on tokens:

```rust
use archmage::{Avx2Token, Avx2FmaToken, SimdToken, composite};

fn composites() {
    if let Some(token) = Avx2FmaToken::try_new() {
        // 8x8 matrix transpose
        let mut matrix: [f32; 64] = core::array::from_fn(|i| i as f32);
        composite::transpose_8x8(token.avx2(), &mut matrix);

        // Dot product with FMA
        let a: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let b: Vec<f32> = vec![1.0; 1024];
        let dot = composite::dot_product_f32(token, &a, &b);

        // Horizontal operations
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let sum = composite::hsum_f32x8(token.avx2(), &data);   // 36.0
        let max = composite::hmax_f32x8(token.avx2(), &data);   // 8.0
        let min = composite::hmin_f32x8(token.avx2(), &data);   // 1.0
    }
}
```

## Available Operations

### Memory Operations
- `load_f32x8`, `store_f32x8` - 8-wide float load/store
- `load_i32x8`, `store_i32x8` - 8-wide int load/store

### Arithmetic
- `add_f32x8`, `sub_f32x8`, `mul_f32x8`, `div_f32x8`
- `add_i32x8`, `sub_i32x8`, `mullo_i32x8`
- `fmadd_f32x8`, `fmsub_f32x8`, `fnmadd_f32x8` (FMA)
- `min_f32x8`, `max_f32x8`

### Shuffle/Permute
- `shuffle_f32x8`, `permute_f32x8`, `permute2_f32x8`
- `blend_f32x8`, `unpacklo_f32x8`, `unpackhi_f32x8`
- `hadd_f32x8` - horizontal add

### Bitwise
- `and_f32x8`, `or_f32x8`, `xor_f32x8`, `andnot_f32x8`

### Conversion
- `cvt_i32x8_f32x8` - int to float
- `cvtt_f32x8_i32x8` - float to int (truncate)

### Creation
- `zero_f32x8`, `set1_f32x8` - create vectors

## Feature Flags

```toml
[dependencies]
archmage = { version = "0.1", features = ["wide", "safe-simd"] }
```

| Feature | Description |
|---------|-------------|
| `std` (default) | Enable std library support |
| `macros` (default) | Enable `#[simd_fn]` attribute macro |
| `wide` | Integration with `wide` crate |
| `safe-simd` | Integration with `safe_unaligned_simd` |

## Why Not Just Use...

### `wide` crate directly?
`wide` uses `cfg!(target_feature)` which is evaluated at **crate level**. Inside a multiversioned function, `wide` still emits SSE code even when AVX2 is available.

### `safe_arch` crate?
Same problem - cfg-gated at crate level, incompatible with runtime dispatch.

### Raw intrinsics?
Every call site needs `unsafe`. With tokens, unsafe is isolated to token creation.

### `pulp` crate?
Runtime dispatch, but still unsafe at call sites. Tokens provide type-level proof.

## ARM Support

ARM tokens follow the same pattern:

```rust
use archmage::{NeonToken, SveToken, Sve2Token, SimdToken};

fn arm_example() {
    // NEON is baseline for AArch64 (always available)
    let neon = NeonToken::try_new().unwrap();

    // SVE (Graviton 3, Apple M-series, A64FX)
    if let Some(sve) = SveToken::try_new() {
        let neon = sve.neon();  // Extract sub-token
    }

    // SVE2 (ARMv9: Cortex-X2/A710+, Graviton 4)
    if let Some(sve2) = Sve2Token::try_new() {
        let sve = sve2.sve();
        let neon = sve2.neon();
    }
}
```

## License

MIT OR Apache-2.0

## AI-Generated Code Notice

Developed with Claude (Anthropic). Not all code manually reviewed. Review critical paths before production use.
