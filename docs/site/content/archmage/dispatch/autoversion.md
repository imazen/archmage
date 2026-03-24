+++
title = "#[autoversion] Macro"
weight = 6
+++

`#[autoversion]` generates architecture-specific function variants and a runtime dispatcher from a single annotated function. Write scalar code, let the compiler auto-vectorize it for each target.

## How It Works

You write a function with a `SimdToken` placeholder parameter. The macro generates one copy per architecture tier — each compiled with `#[target_feature]` via `#[arcane]` — plus a dispatcher that removes the token parameter and calls the best variant the CPU supports.

```rust
use archmage::SimdToken;

#[autoversion]
fn sum_of_squares(_token: SimdToken, data: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &x in data {
        sum += x * x;
    }
    sum
}

// Call directly — no token, no unsafe:
let result = sum_of_squares(&my_data);
```

The `_token` parameter exists so the macro knows where to substitute concrete token types. You don't use it in the body. Each variant gets `#[arcane]` → `#[target_feature(enable = "avx2,fma,...")]`, which unlocks auto-vectorization for that feature set.

On x86-64 with AVX2+FMA, that loop compiles to `vfmadd231ps` — fused multiply-add on 8 floats per cycle. On aarch64 with NEON, you get `fmla`. The `_scalar` fallback compiles without SIMD target features as a safety net.

## Parameters

### No parameters (defaults)

```rust
#[autoversion]
fn process(_token: SimdToken, data: &[f32]) -> f32 { ... }
```

Generates variants for the default tier list: **v4, v3, neon, wasm128, scalar**.

- `process_v4` — AVX-512 (behind `#[cfg(feature = "avx512")]`)
- `process_v3` — AVX2+FMA
- `process_neon` — aarch64 NEON
- `process_wasm128` — WASM SIMD128
- `process_scalar` — no SIMD (always available)
- `process` — **dispatcher** (SimdToken param removed)

### Explicit tier list

```rust
#[autoversion(v3, neon)]
fn process(_token: SimdToken, data: &[f32]) -> f32 { ... }
```

Only generates the listed tiers plus `scalar` (always implicit). Use this when you don't need every platform, or when you want tiers beyond the defaults.

### Methods with self receivers

For inherent methods, `self` works naturally — no special parameters needed:

```rust
impl ImageBuffer {
    #[autoversion]
    fn normalize(&mut self, _token: SimdToken, gamma: f32) {
        for pixel in &mut self.data {
            *pixel = (*pixel / 255.0).powf(gamma);
        }
    }
}

// Call normally — no token:
buffer.normalize(2.2);
```

All receiver types work: `self`, `&self`, `&mut self`. The generated variants use `#[arcane]` in sibling mode, where `self`/`Self` resolve naturally.

### `_self = Type` (trait method delegation)

Only needed when delegating from a trait impl, which requires `#[arcane]`'s nested mode:

```rust
impl MyType {
    #[autoversion(_self = MyType)]
    fn compute_impl(&self, _token: SimdToken, data: &[f32]) -> f32 {
        _self.weights.iter().zip(data).map(|(w, d)| w * d).sum()
    }
}
```

Use `_self` (not `self`) in the body when using this form. Non-scalar variants get `#[arcane(_self = Type)]`; the scalar variant gets `let _self = self;` injected.

### Combined: tiers + `_self`

```rust
#[autoversion(v3, neon, _self = MyType)]
fn compute_impl(&self, _token: SimdToken, data: &[f32]) -> f32 {
    _self.weights.iter().zip(data).map(|(w, d)| w * d).sum()
}
```

Order doesn't matter — tiers and `_self` can be mixed freely, separated by commas.

## Available Tiers

| Tier | Suffix | Token | Architecture | Cargo Feature |
|------|--------|-------|-------------|---------------|
| `v1` | `_v1` | `X64V1Token` | x86-64 | — |
| `v2` | `_v2` | `X64V2Token` | x86-64 | — |
| `x64_crypto` | `_x64_crypto` | `X64CryptoToken` | x86-64 | — |
| `v3` | `_v3` | `X64V3Token` | x86-64 | — |
| `v3_crypto` | `_v3_crypto` | `X64V3CryptoToken` | x86-64 | — |
| `v4` | `_v4` | `X64V4Token` | x86-64 | `avx512` |
| `v4x` | `_v4x` | `X64V4xToken` | x86-64 | `avx512` |
| `arm_v3` | `_arm_v3` | `Arm64V3Token` | aarch64 | — |
| `arm_v2` | `_arm_v2` | `Arm64V2Token` | aarch64 | — |
| `neon_aes` | `_neon_aes` | `NeonAesToken` | aarch64 | — |
| `neon_sha3` | `_neon_sha3` | `NeonSha3Token` | aarch64 | — |
| `neon_crc` | `_neon_crc` | `NeonCrcToken` | aarch64 | — |
| `neon` | `_neon` | `NeonToken` | aarch64 | — |
| `wasm128_relaxed` | `_wasm128_relaxed` | `Wasm128RelaxedToken` | wasm32 | — |
| `wasm128` | `_wasm128` | `Wasm128Token` | wasm32 | — |
| `scalar` | `_scalar` | `ScalarToken` | all | — |

`scalar` is always appended implicitly. You don't list it, but you don't need to define `_scalar` separately — `#[autoversion]` generates it.

Tiers requiring a cargo feature (like `v4`/`v4x` requiring `avx512`) are only compiled when that feature is enabled. Within the same architecture, tiers are tried highest-priority first.

## Default Tiers

When you write `#[autoversion]` with no tier list, you get: **v4, v3, neon, wasm128, scalar**.

This covers the common case: AVX-512 when available (with cargo feature opt-in), AVX2+FMA as the x86-64 workhorse, NEON for ARM, WASM SIMD for browsers, and scalar as the universal fallback.

If you need finer-grained tiers (e.g., `v2` for SSE4.2, `arm_v2` for modern ARM compute, or `wasm128_relaxed`), specify them explicitly.

## Generated Code

{% mermaid() %}
flowchart TD
    CALL["process(data)"] --> V4{"X64V4Token::summon()?<br/>(x86, avx512 feature)"}
    V4 -->|Some| PV4["process_v4(token, data)"]
    V4 -->|None / wrong arch| V3{"X64V3Token::summon()?<br/>(x86)"}
    V3 -->|Some| PV3["process_v3(token, data)"]
    V3 -->|None / wrong arch| NEON{"NeonToken::summon()?<br/>(aarch64)"}
    NEON -->|Some| PN["process_neon(token, data)"]
    NEON -->|None / wrong arch| WASM{"Wasm128Token::summon()?<br/>(wasm32)"}
    WASM -->|Some| PW["process_wasm128(token, data)"]
    WASM -->|None / wrong arch| PS["process_scalar(ScalarToken, data)"]

    style CALL fill:#5a3d1e,color:#fff
    style PV4 fill:#2d5a27,color:#fff
    style PV3 fill:#2d5a27,color:#fff
    style PN fill:#2d5a27,color:#fff
    style PW fill:#2d5a27,color:#fff
    style PS fill:#1a4a6e,color:#fff
{% end %}

The dispatcher uses labeled blocks (`'__dispatch: { ... break '__dispatch result; }`) rather than `return`, so it works as an expression and can be embedded in larger functions.

Variants are always private — only the dispatcher gets the original function's visibility. Within the same module, you can call variants directly for testing or benchmarking.

Variants for other architectures are excluded at compile time by `#[cfg(target_arch)]`. On x86-64, only `_v4`, `_v3`, and `_scalar` are compiled.

## SimdToken Replacement

`#[autoversion]` replaces the `SimdToken` type in the function signature with the concrete token for each variant. Only the parameter's type changes — the function body is untouched.

The token variable (whatever you named it — `token`, `_token`, `_t`, or even `_`) keeps working because its type comes from the signature. If you pass it to functions that accept concrete tokens or trait bounds, it resolves correctly.

This is different from `#[magetypes]`, which does text substitution of `Token` throughout the entire function body. `#[autoversion]` is lighter — it only touches the signature, so compile times stay low.

## Trait Methods

Trait methods can't use `#[autoversion]` directly — proc macro attributes on trait impl items can't expand to sibling functions. Delegate to an autoversioned inherent method:

```rust
trait Processor {
    fn process(&self, data: &[f32]) -> f32;
}

impl Processor for MyType {
    fn process(&self, data: &[f32]) -> f32 {
        self.process_impl(data)
    }
}

impl MyType {
    #[autoversion(_self = MyType)]
    fn process_impl(&self, _token: SimdToken, data: &[f32]) -> f32 {
        _self.weights.iter().zip(data).map(|(w, d)| w * d).sum()
    }
}
```

## Wildcard Token Parameters

If you don't reference the token at all, use `_`:

```rust
#[autoversion]
fn sum(_: SimdToken, data: &[f32]) -> f32 {
    data.iter().sum()
}
```

The macro generates an internal name for the wildcard so it can pass it to each variant.

## When to Use

**Use `#[autoversion]` when** you have scalar code that the compiler can auto-vectorize — tight numeric loops, element-wise transforms, reductions. You get multi-platform SIMD with zero intrinsics.

**Use `#[magetypes]` + `incant!` when** you need hand-written SIMD with `f32x8`, `u8x32`, and architecture-specific algorithms.

**Use manual dispatch when** different tiers need fundamentally different algorithms or APIs.

| | `#[autoversion]` | `#[magetypes]` + `incant!` |
|---|---|---|
| Placeholder | `SimdToken` | `Token` |
| Generates variants | Yes | Yes (`#[magetypes]`) |
| Generates dispatcher | Yes | No (you write `incant!`) |
| Body touched | No (signature only) | Yes (text substitution) |
| Best for | Scalar auto-vectorization | Hand-written SIMD |

## Benchmarking Variants

The generated variants are real, visible functions. You can call them individually to measure the auto-vectorization speedup:

```rust
use criterion::{Criterion, black_box, criterion_group, criterion_main};
use archmage::SimdToken;

#[autoversion]
fn sum_squares(_token: SimdToken, data: &[f32]) -> f32 {
    data.iter().map(|&x| x * x).fold(0.0f32, |a, b| a + b)
}

fn bench(c: &mut Criterion) {
    let data: Vec<f32> = (0..4096).map(|i| i as f32 * 0.01).collect();
    let mut group = c.benchmark_group("sum_squares");

    group.bench_function("dispatched", |b| {
        b.iter(|| sum_squares(black_box(&data)))
    });

    group.bench_function("scalar", |b| {
        b.iter(|| sum_squares_scalar(archmage::ScalarToken, black_box(&data)))
    });

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
