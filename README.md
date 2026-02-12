# archmage

[![CI](https://github.com/imazen/archmage/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/archmage/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/archmage.svg)](https://crates.io/crates/archmage)
[![docs.rs](https://docs.rs/archmage/badge.svg)](https://docs.rs/archmage)
[![codecov](https://codecov.io/gh/imazen/archmage/graph/badge.svg)](https://codecov.io/gh/imazen/archmage)
[![MSRV](https://img.shields.io/crates/msrv/archmage.svg)](https://crates.io/crates/archmage)
[![License](https://img.shields.io/crates/l/archmage.svg)](https://github.com/imazen/archmage#license)

> Safely invoke your intrinsic power, using the tokens granted to you by the CPU.

**Zero overhead.** Archmage generates identical assembly to hand-written unsafe code. The safety abstractions exist only at compile time—at runtime, you get raw SIMD instructions. Calling an `#[arcane]` function costs exactly the same as calling a bare `#[target_feature]` function directly.

```toml
[dependencies]
archmage = "0.5"
magetypes = "0.5"
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

// Inner helper: use #[rite] (inlines into #[arcane] — features match)
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

Processing 1000 8-float vector additions ([full benchmark details](docs/PERFORMANCE.md)):

| Pattern | Time | Why |
|---------|------|-----|
| `#[rite]` in `#[arcane]` | 547 ns | Features match — LLVM inlines |
| `#[arcane]` per iteration | 2209 ns (4x) | Target-feature boundary per call |
| Bare `#[target_feature]` (no archmage) | 2222 ns (4x) | Same boundary — archmage adds nothing |

The 4x penalty comes from LLVM's `#[target_feature]` optimization boundary, not from archmage. Bare `#[target_feature]` has the same cost. With real workloads (DCT-8), the boundary costs up to 6.2x. Use `#[rite]` for helpers called from SIMD code — it inlines into callers with matching features, eliminating the boundary.

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

## Runtime dispatch with `incant!`

Write platform-specific variants with concrete types, then dispatch at runtime:

```rust
use archmage::incant;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::f32x8;

#[cfg(target_arch = "x86_64")]
const LANES: usize = 8;

/// AVX2 path — processes 8 floats at a time.
#[cfg(target_arch = "x86_64")]
fn sum_squares_v3(token: archmage::X64V3Token, data: &[f32]) -> f32 {
    let chunks = data.chunks_exact(LANES);
    let mut acc = f32x8::zero(token);
    for chunk in chunks {
        let v = f32x8::from_array(token, chunk.try_into().unwrap());
        acc = v.mul_add(v, acc);
    }
    acc.reduce_add() + chunks.remainder().iter().map(|x| x * x).sum::<f32>()
}

/// Scalar fallback.
fn sum_squares_scalar(_token: archmage::ScalarToken, data: &[f32]) -> f32 {
    data.iter().map(|x| x * x).sum()
}

/// Public API — dispatches to the best available at runtime.
fn sum_squares(data: &[f32]) -> f32 {
    incant!(sum_squares(data))
}
```

`incant!` looks for `_v3`, `_v4`, `_neon`, `_wasm128`, and `_scalar` suffixed functions, and dispatches to the best one the CPU supports. Each variant uses concrete SIMD types for its platform; the scalar fallback uses plain math.

### `#[magetypes]` for simple cases

If your function body doesn't use SIMD types (only `Token`), `#[magetypes]` can generate the variants for you by replacing `Token` with the concrete token type for each platform:

```rust
use archmage::magetypes;

#[magetypes]
fn process(token: Token, data: &[f32]) -> f32 {
    // Token is replaced with X64V3Token, NeonToken, ScalarToken, etc.
    // But SIMD types like f32x8 are NOT replaced — use incant! pattern
    // for functions that need different types per platform.
    data.iter().sum()
}
```

For functions that use platform-specific SIMD types (`f32x8`, `f32x4`, etc.), write the variants manually and use `incant!` as shown above.

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
