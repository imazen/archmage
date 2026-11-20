# Dispatch

Choosing the right SIMD implementation at runtime.

## `attuned!` and `#[attune]` — Automatic Dispatch

Generate per-tier proof wrappers with `#[attune(_*_t, ...)]` (or a single callable dispatcher with `#[attune(dispatch)]`) and dispatch with `attuned!`:

```rust
use archmage::prelude::*;

#[attune(_*_t, define(f32x8))]
fn sum_8(token: Token, data: &[f32; 8]) -> f32 {
    f32x8::load_t(token, data).reduce_add()
}

pub fn sum(data: &[f32; 8]) -> f32 {
    attuned!(sum_8(data))
    // Tries: sum_8_v3_t → sum_8_neon_t → sum_8_wasm128_t → sum_8_scalar_t
}
```

You can also write separate per-tier implementations and dispatch across them with `attuned!`:

```rust
use archmage::prelude::*;

#[attune(import_intrinsics)]
fn dot_v3_t(_token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let va = _mm256_loadu_ps(a);
    let vb = _mm256_loadu_ps(b);
    let prod = _mm256_mul_ps(va, vb);
    let hi = _mm256_extractf128_ps::<1>(prod);
    let sum128 = _mm_add_ps(_mm256_castps256_ps128(prod), hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    _mm_cvtss_f32(_mm_add_ss(sums, shuf2))
}

#[attune]
fn dot_scalar_t(_token: ScalarToken, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn dot(a: &[f32; 8], b: &[f32; 8]) -> f32 {
    attuned!(dot(a, b), [_v3, _scalar])
}
```

### Default tiers and AVX-512

Elided `attuned!(work(data))` dispatches across the portable default set:

1. `_v3_t` — `X64V3Token` (x86-64)
2. `_neon_t` — `NeonToken` (AArch64)
3. `_wasm128_t` — `Wasm128Token` (WASM)
4. `_scalar_t` — `ScalarToken` (fallback, always available)

Unlike legacy `incant!`, elided `attuned!` never adds `_v4` or `_v4x` implicitly, even when `avx512` is enabled. Add AVX-512 explicitly on both definition and call:

```rust
use archmage::prelude::*;

#[attune(_*_t, +v4(cfg(avx512)), define(f32x16))]
fn scale(token: Token, data: &mut [f32; 16]) {
    (f32x16::load_t(token, data) * f32x16::splat_t(token, 2.0)).store(data);
}

pub fn scale_in_place(data: &mut [f32; 16]) {
    attuned!(scale(data), [_v4(cfg(avx512)), _v3, _neon, _wasm128, _scalar]);
}
```

Candidates are sorted by registry priority (`v4x` > `v4` > `v3`), not textual list order. Cross-architecture variants are excluded via `#[cfg(target_arch)]` at compile time.

### Suffix conventions

| Direct body | Proof wrapper | Token | Platform |
|-------------|---------------|-------|----------|
| `_v4x` | `_v4x_t` | `X64V4xToken` | x86-64 Ice Lake / Zen 4+ (`cfg(avx512)`) |
| `_v4` | `_v4_t` | `X64V4Token` | x86-64 AVX-512 (`cfg(avx512)`) |
| `_v3` | `_v3_t` | `X64V3Token` | x86-64 AVX2+FMA |
| `_neon` | `_neon_t` | `NeonToken` | AArch64 |
| `_wasm128` | `_wasm128_t` | `Wasm128Token` | WASM |
| `_scalar` | `_scalar_t` | `ScalarToken` | All platforms (fallback) |

Outside a feature context, `attuned!` calls the `_t` proof wrappers. Inside an `#[attune]` body whose features already cover a candidate tier, `attuned!` calls a `_*` direct body with no runtime probe; use `reattune!` when you want to probe for a stronger tier from inside an already-covered context.

### Passing an existing token (`using(token)`)

When you already hold a token (concrete or generic `IntoConcreteToken`) and want to select a variant without probing CPUID again, pass `using(token)`:

```rust
use archmage::prelude::*;

#[attune(_*_t)]
fn process(_token: Token, data: &[f32]) -> f32 {
    data.iter().sum()
}

fn inner<T: IntoConcreteToken>(token: T, data: &[f32]) -> f32 {
    attuned!(process(data), [_v3, _neon, _wasm128, _scalar], using(token))
}
```

`using(token)` evaluates its argument once, matches by exact token type via `IntoConcreteToken` (or direct proof subsumption for concrete tokens), and never probes the CPU.

## Manual Dispatch

For custom control flow or per-tier signatures, dispatch manually:

```rust
use archmage::prelude::*;

pub fn process(data: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = X64V3Token::summon() {
        return process_avx2(token, data);
    }
    process_scalar(data);
}
```

### When to use manual dispatch

- Different function signatures per platform (for example, different state structs)
- Custom fallback or hybrid tier logic
- Combining results across multiple feature checks

## Enter the Boundary Once

Cross into a proof wrapper (`#[attune(wrap)]` or `_tier_t`) once at the API boundary, put loops inside it, and mark helpers as direct feature bodies (`#[attune(v3)]` or `#[attune(_*)]`):

```rust
use archmage::prelude::*;

// WRONG: target-feature boundary every iteration (4x slower)
fn process_all_slow(points: &[[f32; 8]]) {
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = X64V3Token::summon() { // hoisting summon() doesn't help!
        for p in points {
            let _ = process_one_wrapped(token, p); // boundary crossing every iteration
        }
    }
}

// RIGHT: one boundary crossing, loop inside
pub fn process_all(points: &[[f32; 8]]) {
    #[cfg(target_arch = "x86_64")]
    if let Some(token) = X64V3Token::summon() {
        return process_all_simd(token, points);
    }
    process_all_scalar(points);
}

#[attune(wrap, import_intrinsics)]
fn process_all_simd(_token: X64V3Token, points: &[[f32; 8]]) {
    for p in points {
        let _ = process_one(p); // #[attune(v3)] helper, inlines
    }
}

#[attune(v3, import_intrinsics)]
fn process_one(p: &[f32; 8]) -> __m256 {
    _mm256_loadu_ps(p)
}
```

The cost isn't `summon()` (~1.3 ns cached). It's the `#[target_feature]` boundary — LLVM cannot inline a stronger-feature callee into a weaker-feature caller. A bare `#[target_feature]` function without archmage pays the same cost: 4x on simple adds, up to 6.2x on real workloads (verified in `benches/asm_inspection.rs`). Direct `#[attune(v3)]` helpers share the caller's target features and inline into the same LLVM optimization region. See the [full benchmark data](../../../PERFORMANCE.md).
