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
3. `_neon` — `NeonToken` (AArch64)
4. `_wasm128` — `Wasm128Token` (WASM)
5. `_scalar` — `ScalarToken` (fallback, always required)

Cross-architecture variants are excluded by `#[cfg(target_arch)]` at compile time. On x86-64, you need `_v3` and `_scalar` (plus `_v4` if the `avx512` feature is enabled). You don't need `_neon` or `_wasm128`.

### Suffix conventions

| Suffix | Token | Platform |
|--------|-------|----------|
| `_v4` | `X64V4Token` | x86-64 AVX-512 (requires `avx512` feature) |
| `_v3` | `X64V3Token` | x86-64 AVX2+FMA |
| `_neon` | `NeonToken` | AArch64 |
| `_wasm128` | `Wasm128Token` | WASM |
| `_scalar` | `ScalarToken` | Always required |

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

## Enter `#[arcane]` Once

Enter `#[arcane]` at the API boundary, put loops inside it, use `#[rite]` for helpers:

```rust
// WRONG: target-feature boundary every iteration (4x slower)
fn process_all(points: &[[f32; 8]]) {
    let token = Desktop64::summon().unwrap(); // hoisting doesn't help!
    for p in points {
        process_one(token, p);  // #[arcane] boundary every iteration
    }
}

// RIGHT: one boundary, loop inside
fn process_all(points: &[[f32; 8]]) {
    if let Some(token) = Desktop64::summon() {
        process_all_simd(token, points);
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

The cost isn't `summon()` (~1.3 ns cached). It's the `#[target_feature]` boundary — LLVM can't inline across mismatched target features. A bare `#[target_feature]` function without archmage has the same cost: 4x on simple adds, up to 6.2x on real workloads (verified in `benches/asm_inspection.rs`). The fix is `#[rite]` — it inlines into callers with matching features, keeping everything in one LLVM optimization region. See the [full benchmark data](../../../PERFORMANCE.md).
