# Safety Model

## The Short Version

1. **Tokens prove CPU features exist.** You can't construct one without `summon()` succeeding.
2. **`#[arcane]` generates `#[target_feature]` code.** The macro wraps your function body in an inner function with the right target features enabled.
3. **Inside `#[target_feature]`, most intrinsics are safe.** Rust 1.87+ made value-based intrinsics safe in this context.
4. **Only memory operations need `unsafe`.** Raw pointer loads/stores, or use `import_intrinsics` to get safe memory ops that take references instead of raw pointers.

## What's safe inside `#[arcane]`

All value-based intrinsics — arithmetic, comparison, shuffle, bitwise, conversion, reduction:

```rust
#[arcane(import_intrinsics)]
fn example(token: X64V3Token, a: __m256, b: __m256) -> __m256 {
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

## What still needs `unsafe` (or safe memory ops via `import_intrinsics`)

Raw pointer operations:

```rust
use archmage::prelude::*;

#[arcane(import_intrinsics)]
fn load_safe(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    let value = _mm256_loadu_ps(data);
    let mut output = [0.0; 8];
    _mm256_storeu_ps(&mut output, value);
    output
}
```

## How `#[arcane]` works

The macro generates a safe outer function wrapping a target-feature inner:

```rust
// You write:
#[arcane(import_intrinsics)]
fn kernel(token: X64V3Token, data: &[f32; 8]) -> f32 {
    let v = _mm256_setzero_ps();
    // ...
}
```

The macro generates the target-feature boundary and its justified internal call.
User code does not implement that boundary; see the expansion tests for exact output.

The outer function is safe. The `unsafe` call to the inner function is justified by the token's existence.

## `#[rite]` vs `#[arcane]`

| | `#[arcane]` | `#[rite]` |
|---|---|---|
| Adds `#[target_feature]` | Yes (via wrapper) | Yes (directly) |
| Safe to call from anywhere | Yes | No — must be called from matching `#[target_feature]` context |
| Overhead | 4-6x in hot loop ([details](../../../PERFORMANCE.md)) | Zero (inlines) |
| Use for | Entry points (from non-SIMD code) | Internal helpers (from SIMD code) |

**Rule of thumb:** `#[arcane]` at the boundary, `#[rite]` for everything else. `#[rite]` works in three modes: token-based (`#[rite]`), tier-based (`#[rite(v3)]` — no token needed), or multi-tier (`#[rite(v3, v4, neon)]` — generates suffixed variants).

```rust
// Entry point — called from non-SIMD code
#[arcane(import_intrinsics)]
pub fn process(token: X64V3Token, data: &mut [f32]) {
    for chunk in data.chunks_exact_mut(8) {
        process_chunk(chunk.try_into().unwrap());  // no token needed
    }
}

// Internal helper — tier-based, inlines into the #[arcane] caller
#[rite(v3, import_intrinsics)]
fn process_chunk(chunk: &mut [f32; 8]) {
    // ... SIMD work ...
}
```

## Generic kernels need a generated context

Use `#[magetypes]` at the dispatched entry. A reusable `T: F32x8Backend` helper
can inline into each concrete variant with `#[inline(always)]`. Monomorphized
generic calls are statically resolved; generic bounds are not inherently an
indirect call or optimization barrier. An inline attribute alone does not
supply target features.

**Downcasting is free:** Passing `X64V4Token` to a function expecting `X64V3Token` preserves the inlining chain.

**Upcasting via `IntoConcreteToken` is safe but creates a boundary:** The generic dispatch function has baseline target features, the concrete callee has extended features. LLVM can't optimize across that mismatch.

## The soundness invariant

```
features_enabled_by_arcane(Token) ⊆ features_checked_by_summon(Token)
```

This is verified by `cargo xtask validate` — it reads the token registry and checks that every feature `#[arcane]` enables is also checked by the corresponding `summon()` implementation.

## Cross-architecture compilation

On the wrong architecture, `#[arcane]` and `#[rite]` cfg-out the function entirely — no code is emitted. This means direct call sites must be guarded:

```rust
// This function only exists on x86_64 — cfg'd out on ARM/WASM
#[arcane(import_intrinsics)]
fn kernel(token: X64V3Token, data: &[f32; 8]) -> f32 {
    // ...
}

// Option 1: Guard the call site with #[cfg]
#[cfg(target_arch = "x86_64")]
if let Some(token) = X64V3Token::summon() {
    kernel(token, &data);
}

// Option 2: Use incant! which handles cfg-gating automatically
pub fn process(data: &[f32; 8]) -> f32 {
    incant!(kernel(data))
}
```

`incant!` is the recommended approach for cross-arch dispatch — it wraps each tier call in `#[cfg(target_arch)]` blocks automatically, so you never need manual cfg guards at call sites.
