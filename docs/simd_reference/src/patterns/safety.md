# Safety Model

## The Short Version

1. **Tokens prove CPU features exist.** Safe code acquires one through `summon()`, extraction from a stronger token (`v4.v3()`), or `from_context()` inside a covering feature body.
2. **`#[attune(wrap)]` and `_tier_t` generate `#[target_feature]` wrappers.** The macro wraps your function body in a private `#[target_feature]` function with the token's features enabled.
3. **Inside `#[target_feature]`, value-based intrinsics are safe.** Rust 1.87+ made value-based intrinsics safe in matching target-feature contexts.
4. **Only raw-pointer and processor-state intrinsics remain `unsafe`.** Use `import_intrinsics` for safe reference-based load/store wrappers (`&[T; N]` / `&mut [T; N]`).

## What's safe inside `#[attune]`

All value-based intrinsics — arithmetic, comparison, shuffle, bitwise, conversion, reduction:

```rust
use archmage::prelude::*;

#[attune(wrap, import_intrinsics)]
fn example(_token: X64V3Token, a: __m256, b: __m256) -> __m256 {
    // All safe — no `unsafe` needed:
    let sum = _mm256_add_ps(a, b);
    let product = _mm256_mul_ps(a, b);
    let fma = _mm256_fmadd_ps(a, b, sum);
    let mask = _mm256_cmp_ps::<_CMP_GT_OQ>(a, b);
    let blended = _mm256_blendv_ps(a, b, mask);
    let _shuffled = _mm256_permute_ps::<0b10_11_00_01>(blended);
    let _zero = _mm256_setzero_ps();
    let _broadcast = _mm256_set1_ps(42.0);
    _mm256_add_ps(fma, product)
}
```

## Safe memory ops via `import_intrinsics`

Raw pointer operations remain `unsafe`, while `import_intrinsics` brings safe reference-based memory ops into scope:

```rust
use archmage::prelude::*;

#[attune(wrap, import_intrinsics)]
fn load_safe(_token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    let value = _mm256_loadu_ps(data);
    let mut output = [0.0; 8];
    _mm256_storeu_ps(&mut output, value);
    output
}
```

## How `#[attune(wrap)]` works

A proof wrapper (`#[attune(wrap)]` or `#[attune] fn kernel_v3_t(...)`) generates a safe outer function calling a private `#[target_feature]` inner function:

```rust
use archmage::prelude::*;

// You write:
#[attune(wrap, import_intrinsics)]
fn kernel(_token: X64V3Token, data: &[f32; 8]) -> f32 {
    let v = _mm256_loadu_ps(data);
    _mm256_cvtss_f32(_mm256_castps256_ps128(v))
}
```

The macro authenticates the token type and emits the `#[target_feature]` boundary call internally. User code contains no `unsafe`; see [`docs/ATTUNE.md`](../../../ATTUNE.md) and the expansion tests for exact output.

## Proof wrappers vs direct feature bodies

| | Proof wrapper (`#[attune(wrap)]`, `_tier_t`) | Direct body (`#[attune(v3)]`, `_tier`, `_*`) |
|---|---|---|
| Adds `#[target_feature]` | Yes (via private inner function) | Yes (directly on the function) |
| Safe to call from anywhere | Yes (with token proof) | No — caller must have matching or superset `#[target_feature]` |
| Overhead across feature mismatch | 4–6x in hot loop ([details](../../../PERFORMANCE.md)) | Zero (inlines into matching caller) |
| Legacy equivalent | `#[arcane]` | `#[rite]` / `#[rite(v3)]` |
| Use for | Entry points from non-SIMD code | Internal helpers inside SIMD code |

**Rule of thumb:** enter a proof wrapper once at the boundary, and use direct feature bodies (`#[attune(v3)]` or multi-tier `#[attune(_v3, _neon)]`) for internal helpers.

```rust
use archmage::prelude::*;

// Entry point — called from non-SIMD code
#[attune(wrap, import_intrinsics)]
pub fn process(_token: X64V3Token, data: &mut [f32]) {
    for chunk in data.chunks_exact_mut(8) {
        process_chunk(chunk.try_into().unwrap());  // no token needed
    }
}

// Internal helper — direct V3 body, inlines into the caller
#[attune(v3, import_intrinsics)]
fn process_chunk(chunk: &mut [f32; 8]) {
    let v = _mm256_loadu_ps(chunk);
    _mm256_storeu_ps(chunk, _mm256_add_ps(v, v));
}
```

## Generic kernels need a generated context

Use `#[attune(_*_t, define(f32x8))]` (or `#[attune(dispatch, define(f32x8))]`) at the dispatched entry. A reusable `T: F32x8Backend` helper inlines into each concrete variant when marked `#[inline(always)]`. Monomorphized generic calls are statically resolved; generic bounds are not inherently an indirect call or optimization barrier, though an inline attribute alone does not supply target features.

**Downcasting is free:** Passing `token.v3()` from an `X64V4Token` caller to a function expecting `X64V3Token` preserves inlining because V4 is a superset of V3.

**Upcasting via `IntoConcreteToken` is safe but crosses a boundary:** A baseline generic function calling into a concrete higher-tier wrapper crosses into stronger target features, which LLVM cannot inline back into the weaker caller.

## The soundness invariant

```text
features_enabled_by_attune(Token) ⊆ features_checked_by_summon(Token)
```

`cargo xtask validate` reads `token-registry.toml` and verifies that every feature enabled by the macros for a token is checked by its `summon()` implementation.

## Cross-architecture compilation

On unsupported architectures, `#[attune]` omits the architecture-specific function via `#[cfg(target_arch)]`. Direct call sites to single-architecture functions must be guarded, or dispatched with `attuned!`:

```rust
use archmage::prelude::*;

// Emitted only on x86_64 — omitted on ARM/WASM
#[attune(wrap, import_intrinsics)]
fn kernel_x86(_token: X64V3Token, data: &[f32; 8]) -> f32 {
    _mm256_cvtss_f32(_mm256_castps256_ps128(_mm256_loadu_ps(data)))
}

// Option 1: Guard a single-arch call site with #[cfg]
#[cfg(target_arch = "x86_64")]
if let Some(token) = X64V3Token::summon() {
    let _ = kernel_x86(token, &data);
}

// Option 2: Generate portable variants and use attuned!
#[attune(_*_t, define(f32x8))]
fn kernel(token: Token, data: &[f32; 8]) -> f32 {
    f32x8::load_t(token, data).reduce_add()
}

pub fn process(data: &[f32; 8]) -> f32 {
    attuned!(kernel(data))
}
```

`attuned!` and `#[attune(dispatch)]` guard each candidate call with `#[cfg(target_arch)]` automatically, so portable call sites need no manual `#[cfg]` attributes.
