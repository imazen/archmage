# archmage

[![CI](https://github.com/imazen/archmage/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/archmage/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/archmage.svg)](https://crates.io/crates/archmage)
[![docs.rs](https://docs.rs/archmage/badge.svg)](https://docs.rs/archmage)
[![codecov](https://codecov.io/gh/imazen/archmage/graph/badge.svg)](https://codecov.io/gh/imazen/archmage)
[![MSRV](https://img.shields.io/crates/msrv/archmage.svg)](https://crates.io/crates/archmage)
[![License](https://img.shields.io/crates/l/archmage.svg)](https://github.com/imazen/archmage#license)

> Safely invoke your intrinsic power, using the tokens granted to you by the CPU.

**Zero overhead.** Archmage generates identical assembly to hand-written unsafe code. The safety abstractions exist only at compile time—at runtime, you get raw SIMD instructions with no wrapper overhead.

```toml
[dependencies]
archmage = "0.4"
magetypes = "0.4"
```

## Raw intrinsics with `#[arcane]`

```rust
use archmage::prelude::*;

#[arcane]
fn dot_product(_token: Desktop64, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let va = _mm256_loadu_ps(a);
    let vb = _mm256_loadu_ps(b);
    let mul = _mm256_mul_ps(va, vb);
    let mut out = [0.0f32; 8];
    _mm256_storeu_ps(&mut out, mul);
    out.iter().sum()
}

fn main() {
    if let Some(token) = Desktop64::summon() {
        println!("{}", dot_product(token, &[1.0; 8], &[2.0; 8]));
    }
}
```

`summon()` checks CPUID. `#[arcane]` enables `#[target_feature]`, making intrinsics safe (Rust 1.85+). The prelude re-exports `safe_unaligned_simd` functions directly — `_mm256_loadu_ps` takes `&[f32; 8]`, not a raw pointer. Compile with `-C target-cpu=haswell` to elide the runtime check.

## Inner helpers with `#[rite]`

**`#[rite]` should be your default.** Use `#[arcane]` only at entry points.

```rust
use archmage::prelude::*;

// Entry point: use #[arcane]
#[arcane]
fn dot_product(token: Desktop64, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let products = mul_vectors(token, a, b);  // Calls #[rite] helper
    horizontal_sum(token, products)
}

// Inner helper: use #[rite] (no wrapper overhead)
#[rite]
fn mul_vectors(_: Desktop64, a: &[f32; 8], b: &[f32; 8]) -> __m256 {
    let va = _mm256_loadu_ps(a);
    let vb = _mm256_loadu_ps(b);
    _mm256_mul_ps(va, vb)
}

#[rite]
fn horizontal_sum(_: Desktop64, v: __m256) -> f32 {
    let sum = _mm256_hadd_ps(v, v);
    let sum = _mm256_hadd_ps(sum, sum);
    let low = _mm256_castps256_ps128(sum);
    let high = _mm256_extractf128_ps::<1>(sum);
    _mm_cvtss_f32(_mm_add_ss(low, high))
}
```

`#[rite]` adds `#[target_feature]` + `#[inline]` without a wrapper function. Since Rust 1.85+, calling `#[target_feature]` functions from matching contexts is safe—no `unsafe` needed between `#[arcane]` and `#[rite]` functions.

**Performance rule:** Never call `#[arcane]` from `#[arcane]`. Use `#[rite]` for any function called exclusively from SIMD code.

### Why this matters

Processing 1000 8-float vector additions:

| Pattern | Time |
|---------|------|
| `#[rite]` helper called from `#[arcane]` | 572 ns |
| `#[arcane]` called from loop | 2320 ns (4x slower) |

The difference is wrapper overhead. `#[rite]` inlines fully; `#[arcane]` generates an inner function call per invocation.

## SIMD types with `magetypes`

```rust
use archmage::{Desktop64, SimdToken};
use magetypes::simd::f32x8;

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    if let Some(token) = Desktop64::summon() {
        let mut sum = f32x8::zero(token);
        for (a_chunk, b_chunk) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
            let va = f32x8::load(token, a_chunk.try_into().unwrap());
            let vb = f32x8::load(token, b_chunk.try_into().unwrap());
            sum = va.mul_add(vb, sum);
        }
        sum.reduce_add()
    } else {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }
}
```

`f32x8` wraps `__m256` with token-gated construction and natural operators.

## Multi-platform with `#[magetypes]`

```rust
use archmage::{incant, magetypes, SimdToken};

#[magetypes]
fn sum_squares(token: Token, data: &[f32]) -> f32 {
    let chunks = data.chunks_exact(LANES);
    let mut acc = f32xN::zero(token);
    for chunk in chunks {
        let v = f32xN::from_array(token, chunk.try_into().unwrap());
        acc = v.mul_add(v, acc);
    }
    acc.reduce_add() + chunks.remainder().iter().map(|x| x * x).sum::<f32>()
}

fn sum_squares_api(data: &[f32]) -> f32 {
    incant!(sum_squares(data))
}
```

`#[magetypes]` generates `_v3` (AVX2), `_v4` (AVX-512), `_neon`, `_wasm128`, and `_scalar` variants. `Token`, `f32xN`, and `LANES` are replaced with concrete types. `incant!` dispatches to the best available at runtime.

## Tokens

| Token | Alias | Features |
|-------|-------|----------|
| `X64V2Token` | | SSE4.2, POPCNT |
| `X64V3Token` | `Desktop64` | AVX2, FMA, BMI2 |
| `X64V4Token` | `Server64` | AVX-512 (requires `avx512` feature) |
| `NeonToken` | `Arm64` | NEON |
| `Wasm128Token` | | WASM SIMD |
| `ScalarToken` | | Always available |

All tokens compile on all platforms. `summon()` returns `None` on unsupported architectures. Detection is cached: ~1.3 ns after first call, 0 ns with `-Ctarget-cpu=haswell` (compiles away).

## The prelude

`use archmage::prelude::*` gives you:

- Tokens: `Desktop64`, `Arm64`, `ScalarToken`, etc.
- Traits: `SimdToken`, `IntoConcreteToken`, `HasX64V2`, etc.
- Macros: `#[arcane]`, `#[rite]`, `#[magetypes]`, `incant!`
- Intrinsics: `core::arch::*` for your platform
- Memory ops: `safe_unaligned_simd` functions (reference-based, no raw pointers)

## Feature flags

| Feature | Default | |
|---------|---------|---|
| `std` | yes | Standard library |
| `macros` | yes | `#[arcane]`, `#[magetypes]`, `incant!` |
| `safe_unaligned_simd` | yes | Re-exports via prelude |
| `avx512` | no | AVX-512 tokens |

## License

MIT OR Apache-2.0
