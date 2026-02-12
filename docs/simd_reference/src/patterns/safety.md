# Safety Model

## The Short Version

1. **Tokens prove CPU features exist.** You can't construct one without `summon()` succeeding.
2. **`#[arcane]` generates `#[target_feature]` code.** The macro wraps your function body in an inner function with the right target features enabled.
3. **Inside `#[target_feature]`, most intrinsics are safe.** Rust 1.85+ made value-based intrinsics safe in this context.
4. **Only memory operations need `unsafe`.** Raw pointer loads/stores, or use `safe_unaligned_simd` to avoid `unsafe` entirely.

## What's safe inside `#[arcane]`

All value-based intrinsics — arithmetic, comparison, shuffle, bitwise, conversion, reduction:

```rust
#[arcane]
fn example(token: Desktop64, a: __m256, b: __m256) -> __m256 {
    // All safe — no `unsafe` needed:
    let sum = _mm256_add_ps(a, b);
    let product = _mm256_mul_ps(a, b);
    let fma = _mm256_fmadd_ps(a, b, sum);
    let mask = _mm256_cmp_ps::<_CMP_GT_OQ>(a, b);
    let blended = _mm256_blendv_ps(a, b, mask);
    let shuffled = _mm256_permute_ps::<0b10_11_00_01>(a);
    let zero = _mm256_setzero_ps();
    let broadcast = _mm256_set1_ps(42.0);
    fma
}
```

## What still needs `unsafe` (or `safe_unaligned_simd`)

Raw pointer operations:

```rust
#[arcane]
fn load_raw(_token: Desktop64, ptr: *const f32) -> __m256 {
    // Raw pointer — needs unsafe
    unsafe { _mm256_loadu_ps(ptr) }
}

#[arcane]
fn load_safe(_token: Desktop64, data: &[f32; 8]) -> __m256 {
    // Reference-based — no unsafe needed
    safe_unaligned_simd::x86_64::_mm256_loadu_ps(data)
}
```

## How `#[arcane]` works

The macro generates a safe outer function wrapping an unsafe inner:

```rust
// You write:
#[arcane]
fn kernel(token: X64V3Token, data: &[f32; 8]) -> f32 {
    let v = _mm256_setzero_ps();
    // ...
}

// Macro generates:
fn kernel(_token: X64V3Token, data: &[f32; 8]) -> f32 {
    #[target_feature(enable = "avx2,fma,...")]
    unsafe fn __simd_inner_kernel(data: &[f32; 8]) -> f32 {
        let v = _mm256_setzero_ps();  // Safe inside #[target_feature]
        // ...
    }
    // SAFETY: Token existence proves CPU support was verified via summon()
    unsafe { __simd_inner_kernel(data) }
}
```

The outer function is safe. The `unsafe` call to the inner function is justified by the token's existence.

## `#[rite]` vs `#[arcane]`

| | `#[arcane]` | `#[rite]` |
|---|---|---|
| Adds `#[target_feature]` | Yes (via wrapper) | Yes (directly) |
| Safe to call from anywhere | Yes | No — must be called from matching `#[target_feature]` context |
| Overhead | ~4x in hot loop | Zero (inlines) |
| Use for | Entry points (after `summon()`) | Internal helpers |

**Rule of thumb:** `#[arcane]` at the boundary, `#[rite]` for everything else.

```rust
// Entry point — called from non-SIMD code
#[arcane]
pub fn process(token: Desktop64, data: &mut [f32]) {
    for chunk in data.chunks_exact_mut(8) {
        process_chunk(token, chunk.try_into().unwrap());
    }
}

// Internal helper — inlines into the #[arcane] caller
#[rite]
fn process_chunk(_: Desktop64, chunk: &mut [f32; 8]) {
    // ... SIMD work ...
}
```

## Concrete tokens beat generics

Generic bounds create LLVM optimization barriers:

```rust
// BAD: LLVM can't inline across the generic boundary
#[arcane]
fn process<T: HasX64V2>(token: T, data: &[f32]) -> f32 { ... }

// GOOD: Full inlining, single target_feature region
#[arcane]
fn process(token: X64V3Token, data: &[f32]) -> f32 { ... }
```

`#[target_feature]` changes LLVM's compilation target for that function. A generic caller and a feature-enabled callee have mismatched targets, preventing cross-function optimization.

**Downcasting is free:** Passing `X64V4Token` to a function expecting `X64V3Token` preserves the inlining chain.

**Upcasting via `IntoConcreteToken` is safe but creates a boundary:** The generic dispatch function has baseline target features, the concrete callee has extended features. LLVM can't optimize across that mismatch.

## The soundness invariant

```
features_enabled_by_arcane(Token) ⊆ features_checked_by_summon(Token)
```

This is verified by `cargo xtask validate` — it reads the token registry and checks that every feature `#[arcane]` enables is also checked by the corresponding `summon()` implementation.

## Cross-architecture compilation

On the wrong architecture, `#[arcane]` generates an unreachable stub:

```rust
// On ARM, this generates:
fn kernel(_token: X64V3Token, _data: &[f32; 8]) -> f32 {
    unreachable!("X64V3Token cannot exist on this architecture")
}
```

The stub compiles but can never execute — `X64V3Token::summon()` returns `None` on ARM, so you can never obtain the token needed to call the function.
