+++
title = "#[autoversion]"
weight = 6
+++

`#[autoversion]` generates architecture-specific function variants and a runtime dispatcher from a single annotated function. Write scalar code, let the compiler auto-vectorize it for each target.

## Quick start

```rust
use archmage::autoversion;

#[autoversion]
fn sum_of_squares(data: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &x in data {
        sum += x * x;
    }
    sum
}

// Call directly — no token, no unsafe:
let result = sum_of_squares(&my_data);
```

The macro generates one variant per architecture tier — each compiled with `#[target_feature]` via `#[arcane]` — plus a dispatcher that calls the best variant the CPU supports.

On x86-64 with AVX2+FMA, that loop compiles to `vfmadd231ps` — fused multiply-add on 8 floats per cycle. On aarch64 with NEON, you get `fmla`. The `_scalar` fallback compiles without SIMD target features as a safety net.

## Two forms: tokenless and explicit

### Tokenless (recommended for new code)

No `SimdToken` parameter. The dispatcher keeps your exact signature:

```rust
#[autoversion]
fn normalize(data: &mut [f32], scale: f32) {
    for x in data.iter_mut() {
        *x = (*x - 128.0) * scale;
    }
}

// Caller sees: fn normalize(data: &mut [f32], scale: f32)
normalize(&mut data, 0.5);
```

### Explicit SimdToken

When you write `_token: SimdToken`, the dispatcher preserves that parameter as `impl SimdToken` — it accepts any token:

```rust
use archmage::SimdToken;

#[autoversion]
fn dot_product(_token: SimdToken, a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

// Caller sees: fn dot_product(_: impl SimdToken, a: &[f32], b: &[f32]) -> f32
dot_product(ScalarToken, &a, &b);  // any token works
```

The explicit form exists for **incant! nesting** — when the autoversioned function is a scalar fallback called by `incant!`, which passes `ScalarToken`. See [Nesting with incant!](#nesting-with-incant) below.

## What gets generated

With default tiers, `#[autoversion] fn process(data: &[f32]) -> f32` generates:

| Function | Token | Architecture | Feature gate |
|----------|-------|-------------|-------------|
| `process_v4` | `X64V4Token` | x86-64 | `avx512` |
| `process_v3` | `X64V3Token` | x86-64 | — |
| `process_neon` | `NeonToken` | aarch64 | — |
| `process_wasm128` | `Wasm128Token` | wasm32 | — |
| `process_scalar` | `ScalarToken` | all | — |
| **`process`** | — | all | — |

The dispatcher (`process`) does runtime CPU feature detection via `Token::summon()` and calls the best match. When compiled with `-Ctarget-cpu=native`, detection compiles away entirely.

{% mermaid() %}
flowchart TD
    CALL["process(data)"] --> V4{"X64V4Token::summon()?"}
    V4 -->|Some| PV4["process_v4(token, data)"]
    V4 -->|None / wrong arch| V3{"X64V3Token::summon()?"}
    V3 -->|Some| PV3["process_v3(token, data)"]
    V3 -->|None / wrong arch| NEON{"NeonToken::summon()?"}
    NEON -->|Some| PN["process_neon(token, data)"]
    NEON -->|None / wrong arch| WASM{"Wasm128Token::summon()?"}
    WASM -->|Some| PW["process_wasm128(token, data)"]
    WASM -->|None / wrong arch| PS["process_scalar(ScalarToken, data)"]

    style CALL fill:#5a3d1e,color:#fff
    style PV4 fill:#2d5a27,color:#fff
    style PV3 fill:#2d5a27,color:#fff
    style PN fill:#2d5a27,color:#fff
    style PW fill:#2d5a27,color:#fff
    style PS fill:#1a4a6e,color:#fff
{% end %}

Variants are always **private** — only the dispatcher gets the original function's visibility. Within the same module, variants are accessible for testing and benchmarking.

Variants for other architectures are excluded at compile time by `#[cfg(target_arch)]`. On x86-64, only `_v4`, `_v3`, and `_scalar` exist in the binary.

## Name collision patterns

`#[autoversion]` generates functions named `{fn_name}_{tier_suffix}`. If you have other functions with those names, you get collisions:

```rust
// COLLISION: autoversion generates process_v3, but you also defined one
#[autoversion]
fn process(data: &[f32]) -> f32 { data.iter().sum() }

#[arcane]
fn process_v3(_token: X64V3Token, data: &[f32]) -> f32 { /* hand-written */ }
// ERROR: duplicate definition of process_v3
```

**Resolution:** Use different base names for hand-written and autoversioned code. For the pattern where you want hand-written SIMD for some tiers and auto-vectorization for others, use [incant! nesting](#nesting-with-incant).

### Generated names by tier

| Tier | Suffix | Example |
|------|--------|---------|
| `v1` | `_v1` | `process_v1` |
| `v2` | `_v2` | `process_v2` |
| `x64_crypto` | `_x64_crypto` | `process_x64_crypto` |
| `v3` | `_v3` | `process_v3` |
| `v3_crypto` | `_v3_crypto` | `process_v3_crypto` |
| `v4` | `_v4` | `process_v4` |
| `v4x` | `_v4x` | `process_v4x` |
| `neon` | `_neon` | `process_neon` |
| `arm_v2` | `_arm_v2` | `process_arm_v2` |
| `arm_v3` | `_arm_v3` | `process_arm_v3` |
| `neon_aes` | `_neon_aes` | `process_neon_aes` |
| `neon_sha3` | `_neon_sha3` | `process_neon_sha3` |
| `neon_crc` | `_neon_crc` | `process_neon_crc` |
| `wasm128` | `_wasm128` | `process_wasm128` |
| `wasm128_relaxed` | `_wasm128_relaxed` | `process_wasm128_relaxed` |
| `scalar` | `_scalar` | `process_scalar` |

## Nesting with incant!

The most powerful pattern: hand-written SIMD intrinsics for specific tiers, autoversioned auto-vectorization as the fallback.

```rust
use archmage::prelude::*;

/// Top-level dispatcher — hand-written V4, autoversioned fallback
pub fn process(data: &[f32]) -> f32 {
    incant!(process(data), [v4, scalar])
}

/// Hand-written AVX-512 intrinsics
#[arcane(import_intrinsics)]
fn process_v4(_token: X64V4Token, data: &[f32]) -> f32 {
    // ... AVX-512 intrinsics ...
    todo!()
}

/// Autoversioned scalar fallback — gets V3 auto-vectorization for free.
/// The explicit SimdToken makes the dispatcher accept ScalarToken,
/// which is exactly what incant! passes to the scalar fallback.
#[autoversion(v3, neon)]
fn process_scalar(_token: SimdToken, data: &[f32]) -> f32 {
    data.iter().sum()
}
```

How it works:

1. `incant!` tries `process_v4` first (hand-written AVX-512)
2. If AVX-512 isn't available, falls through to `process_scalar(ScalarToken, data)`
3. `process_scalar` is an autoversioned dispatcher — internally tries V3 auto-vectorization, then NEON, then true scalar

The explicit `SimdToken` parameter is the key. `incant!` always passes `ScalarToken` to the scalar fallback. With tokenless autoversion, the dispatcher wouldn't accept a token argument. With explicit `SimdToken`, autoversion produces a dispatcher that takes `ScalarToken` — the signatures match.

### Nesting for methods

```rust
impl ImageBuffer {
    pub fn sharpen(&mut self, radius: f32) {
        incant!(ImageBuffer::sharpen(self, radius) with ScalarToken, [v4, scalar])
    }

    #[arcane(import_intrinsics)]
    fn sharpen_v4(&mut self, _token: X64V4Token, radius: f32) {
        // AVX-512 sharpen kernel
    }

    #[autoversion(v3, neon)]
    fn sharpen_scalar(&mut self, _token: SimdToken, radius: f32) {
        // Auto-vectorized fallback
        for pixel in self.data.iter_mut() {
            *pixel = (*pixel * radius).clamp(0.0, 1.0);
        }
    }
}
```

## Explicit tiers

```rust
#[autoversion(v3, neon)]
fn process(data: &[f32]) -> f32 { ... }
```

Only generates the listed tiers plus `scalar` (always implicit). Use this when you don't need every platform, or when you want tiers beyond the defaults.

Default tiers (when no list given): **v4, v3, neon, wasm128, scalar**.

## Feature-gated tiers

### Per-tier gates: `tier(feature)`

```rust
#[autoversion(v4(avx512), v3, neon)]
fn process(_token: SimdToken, data: &[f32]) -> f32 { ... }
```

The `v4` variant and its dispatch arm are wrapped in `#[cfg(feature = "avx512")]` — checked against the **calling crate's** features, not archmage's. If the crate doesn't define `avx512`, v4 is silently excluded.

### Whole-dispatch gate: `cfg(feature)`

```rust
#[autoversion(cfg(simd))]
fn process(_token: SimdToken, data: &[f32]) -> f32 { ... }
```

With `--features simd`: full dispatch (v4 → v3 → neon → wasm128 → scalar).
Without: direct scalar call, zero dispatch overhead.

This generates two dispatcher bodies:
- `#[cfg(feature = "simd")]` — full dispatch with runtime detection
- `#[cfg(not(feature = "simd"))]` — immediate scalar call

### Combined

```rust
#[autoversion(v4(avx512), v3, neon, cfg(simd))]
```

## Methods

### Inherent methods — `self` works naturally

```rust
impl ImageBuffer {
    #[autoversion]
    fn normalize(&mut self, gamma: f32) {
        for pixel in &mut self.data {
            *pixel = (*pixel / 255.0).powf(gamma);
        }
    }
}

buffer.normalize(2.2);
```

All receiver types work: `self`, `&self`, `&mut self`. The generated variants use `#[arcane]` in sibling mode where `self`/`Self` resolve naturally.

### `_self = Type` — for trait method delegation

Required when you need `#[arcane]`'s nested mode (trait impls can't expand to sibling functions):

```rust
impl MyType {
    #[autoversion(_self = MyType)]
    fn compute_impl(&self, _token: SimdToken, data: &[f32]) -> f32 {
        _self.weights.iter().zip(data).map(|(w, d)| w * d).sum()
    }
}
```

Use `_self` (not `self`) in the body. Non-scalar variants get `#[arcane(_self = Type)]`; the scalar variant gets `let _self = self;` injected.

### Trait methods — delegation pattern

Trait methods can't use `#[autoversion]` directly. Delegate:

```rust
trait Processor {
    fn process(&self, data: &[f32]) -> f32;
}

impl Processor for MyType {
    fn process(&self, data: &[f32]) -> f32 {
        self.process_impl(data) // delegate to autoversioned method
    }
}

impl MyType {
    #[autoversion]
    fn process_impl(&self, data: &[f32]) -> f32 {
        self.weights.iter().zip(data).map(|(w, d)| w * d).sum()
    }
}
```

## Const generics

```rust
#[autoversion]
fn sum_array<const N: usize>(data: &[f32; N]) -> f32 {
    let mut sum = 0.0f32;
    for &x in data { sum += x; }
    sum
}

let result = sum_array(&[1.0, 2.0, 3.0, 4.0]);
// Variant call with turbofish:
let s = sum_array_scalar::<4>(ScalarToken, &data);
```

Works with multiple const generics, type generics, lifetimes, and all combinations thereof.

## Benchmarking variants

Variants are real functions. Call them directly to measure auto-vectorization speedup:

```rust
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use archmage::{ScalarToken, SimdToken};

#[autoversion]
fn sum_squares(_token: SimdToken, data: &[f32]) -> f32 {
    data.iter().map(|&x| x * x).fold(0.0f32, |a, b| a + b)
}

fn bench(c: &mut Criterion) {
    let data: Vec<f32> = (0..4096).map(|i| i as f32 * 0.01).collect();
    let mut group = c.benchmark_group("sum_squares");

    // Dispatched — picks best available at runtime
    group.bench_function("dispatched", |b| {
        b.iter(|| sum_squares(ScalarToken, black_box(&data)))
    });

    // Scalar baseline
    group.bench_function("scalar", |b| {
        b.iter(|| sum_squares_scalar(ScalarToken, black_box(&data)))
    });

    // Specific tier
    #[cfg(target_arch = "x86_64")]
    if let Some(t) = archmage::X64V3Token::summon() {
        group.bench_function("v3_avx2_fma", |b| {
            b.iter(|| sum_squares_v3(t, black_box(&data)));
        });
    }

    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
```

For tight numeric loops on x86-64, the `_v3` variant typically runs 4-8x faster than `_scalar` because `#[target_feature]` unlocks auto-vectorization that the baseline build can't use.

## When to use what

| | `#[autoversion]` | `#[magetypes]` + `incant!` | Manual dispatch |
|---|---|---|---|
| **Placeholder** | `SimdToken` (optional) | `Token` | None |
| **Generates variants** | Yes | Yes (`#[magetypes]`) | No |
| **Generates dispatcher** | Yes | No (`incant!` separately) | No |
| **Body touched** | No (signature only) | Yes (text substitution) | N/A |
| **Best for** | Scalar auto-vectorization | Hand-written SIMD types | Different algorithms per arch |
| **Lines of code** | 1 attribute | 2+ | Many |

**Use `#[autoversion]`** for scalar code the compiler can auto-vectorize — tight numeric loops, element-wise transforms, reductions.

**Use `#[magetypes]` + `incant!`** when you need `f32x8`, `u8x32`, and hand-tuned SIMD per architecture.

**Use manual dispatch** when different tiers need fundamentally different algorithms.

**Nest them** when some tiers need hand-written intrinsics and others benefit from auto-vectorization.
