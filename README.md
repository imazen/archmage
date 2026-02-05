# archmage

[![CI](https://github.com/imazen/archmage/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/archmage/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/archmage.svg)](https://crates.io/crates/archmage)
[![docs.rs](https://docs.rs/archmage/badge.svg)](https://docs.rs/archmage)
[![codecov](https://codecov.io/gh/imazen/archmage/graph/badge.svg)](https://codecov.io/gh/imazen/archmage)
[![MSRV](https://img.shields.io/crates/msrv/archmage.svg)](https://crates.io/crates/archmage)
[![License](https://img.shields.io/crates/l/archmage.svg)](https://github.com/imazen/archmage#license)

> Safely invoke your intrinsic power, using the tokens granted to you by the CPU.

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
    let va = safe_simd::_mm256_loadu_ps(a);
    let vb = safe_simd::_mm256_loadu_ps(b);
    let mul = _mm256_mul_ps(va, vb);
    let mut out = [0.0f32; 8];
    safe_simd::_mm256_storeu_ps(&mut out, mul);
    out.iter().sum()
}

fn main() {
    if let Some(token) = Desktop64::summon() {
        println!("{}", dot_product(token, &[1.0; 8], &[2.0; 8]));
    }
}
```

`summon()` checks CPUID. `#[arcane]` enables `#[target_feature]`, making intrinsics safe (Rust 1.85+). Memory ops use `safe_simd`. Compile with `-C target-cpu=haswell` to elide the runtime check.

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
| `Simd128Token` | | WASM SIMD |
| `ScalarToken` | | Always available |

All tokens compile on all platforms. `summon()` returns `None` on unsupported architectures.

## The prelude

`use archmage::prelude::*` gives you:

- Tokens: `Desktop64`, `Arm64`, `ScalarToken`, etc.
- Traits: `SimdToken`, `IntoConcreteToken`, `HasX64V2`, etc.
- Macros: `#[arcane]`, `#[magetypes]`, `incant!`
- Intrinsics: `core::arch::*` for your platform
- Memory ops: `safe_simd::*` (`safe_unaligned_simd` re-export)

## Feature flags

| Feature | Default | |
|---------|---------|---|
| `std` | yes | Standard library |
| `macros` | yes | `#[arcane]`, `#[magetypes]`, `incant!` |
| `safe_unaligned_simd` | yes | Re-exports via prelude |
| `avx512` | no | AVX-512 tokens |

## License

MIT OR Apache-2.0
