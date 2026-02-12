# Dispatch

Choosing the right SIMD implementation at runtime.

## `incant!` — Automatic Dispatch

`incant!` dispatches to suffixed function variants, trying the highest tier first:

```rust
use archmage::{incant, arcane, X64V3Token, NeonToken};

#[arcane]
fn sum_v3(token: X64V3Token, data: &[f32; 8]) -> f32 { /* AVX2 */ }

#[arcane]
fn sum_neon(token: NeonToken, data: &[f32; 4]) -> f32 { /* NEON */ }

fn sum_scalar(data: &[f32]) -> f32 { data.iter().sum() }

pub fn sum(data: &[f32; 8]) -> f32 {
    incant!(sum(data))
    // Tries: sum_v4 → sum_v3 → sum_neon → sum_wasm128 → sum_scalar
}
```

### Dispatch order

1. `_v4` — `X64V4Token` (x86-64, requires `avx512` feature)
2. `_v3` — `X64V3Token` (x86-64)
3. `_v2` — `X64V2Token` (x86-64)
4. `_neon` — `NeonToken` (AArch64)
5. `_wasm128` — `Wasm128Token` (WASM)
6. `_scalar` — No token (fallback)

Missing variants are skipped at compile time. You only need `_scalar` plus whatever platforms you support.

### Suffix conventions

| Suffix | Token | Platform |
|--------|-------|----------|
| `_v4` | `X64V4Token` | x86-64 AVX-512 |
| `_v3` | `X64V3Token` | x86-64 AVX2+FMA |
| `_v2` | `X64V2Token` | x86-64 SSE4.2 |
| `_neon` | `NeonToken` | AArch64 |
| `_wasm128` | `Wasm128Token` | WASM |
| `_scalar` | — | Fallback |

### Passthrough mode

When you already have a token and want to dispatch to specialized variants without re-summoning:

```rust
fn inner<T: IntoConcreteToken>(token: T, data: &[f32]) -> f32 {
    incant!(with token => process(data))
}
```

This uses `IntoConcreteToken` to check the token's actual type and dispatch to the best matching suffix.

## Manual Dispatch

For more control, dispatch manually:

```rust
pub fn process(data: &mut [f32]) {
    if let Some(token) = Desktop64::summon() {
        process_avx2(token, data);
    } else {
        process_scalar(data);
    }
}
```

### When to use manual dispatch

- Different function signatures per platform (e.g., different chunk sizes)
- Custom fallback logic
- Need to combine results from multiple tiers
- Performance-critical dispatch where you want to control the check order

## The Dispatch-Once Rule

Dispatch at the API boundary, not inside hot loops:

```rust
// WRONG: 42% regression from target-feature boundary per iteration
fn process_all(points: &[[f32; 8]]) {
    for p in points {
        if let Some(token) = Desktop64::summon() {
            process_one(token, p);  // #[arcane] boundary every iteration
        }
    }
}

// RIGHT: zero overhead
fn process_all(points: &[[f32; 8]]) {
    if let Some(token) = Desktop64::summon() {
        process_all_simd(token, points);  // One boundary, loop inside
    } else {
        process_all_scalar(points);
    }
}

#[arcane]
fn process_all_simd(token: Desktop64, points: &[[f32; 8]]) {
    for p in points {
        process_one(token, p);  // #[rite] helper, inlines
    }
}
```

The cost isn't `summon()` (~1.3 ns cached). It's the `#[target_feature]` boundary — LLVM can't optimize across mismatched target features. Each `#[arcane]` call transitions between LLVM optimization regions. Inside the `#[arcane]` function, use `#[rite]` helpers which inline freely.
