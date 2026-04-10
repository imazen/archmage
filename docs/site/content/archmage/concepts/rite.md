+++
title = "The #[rite] Macro"
weight = 4
+++

<sub>(alias: `#[token_target_features]`)</sub>

`#[rite]` is an advanced alternative to `#[arcane]` for internal SIMD helpers. It adds `#[target_feature]` + `#[inline]` directly to your function — no wrapper, no boundary.

For most code, `#[arcane]` works for both entry points and helpers (LLVM inlines the wrapper when features match). `#[rite]` is useful when you want explicit `#[target_feature]` without a wrapper, or when generating multi-tier suffixed variants.

## Three Modes

`#[rite]` works in three modes:

```rust
// 1. Token-based: token parameter determines features
#[rite(import_intrinsics)]
fn helper(_token: X64V3Token, data: &[f32; 8]) -> __m256 {
    _mm256_loadu_ps(data)
}

// 2. Tier-based (single): tier name determines features, no token needed
#[rite(v3, import_intrinsics)]
fn helper(data: &[f32; 8]) -> __m256 {
    _mm256_loadu_ps(data)
}

// 3. Multi-tier: generates suffixed variants for each tier
#[rite(v3, v4, import_intrinsics)]
fn helper(data: &[f32; 8]) -> __m256 {
    _mm256_loadu_ps(data)
}
// Produces: helper_v3() and helper_v4()
```

Single-tier and token-based generate one function with identical attributes. Multi-tier generates a suffixed copy of the function for each tier, each compiled with different `#[target_feature]` attributes.

**Use tier-based** (`#[rite(v3)]`) when the function doesn't need the token for anything else. **Use token-based** when you pass the token to other functions or magetypes constructors. The token form can be easier to remember if you already have the token in scope — but both produce identical machine code.

## How It Works

{% mermaid() %}
flowchart LR
    A["Your code:<br/>#[rite(v3, import_intrinsics)]<br/>fn process(data: &[f32; 8])"] --> B["Macro adds:<br/>#[target_feature(...)]<br/>#[inline]<br/>+ auto-imports intrinsics"]

    style A fill:#1a4a6e,color:#fff
    style B fill:#2d5a27,color:#fff
{% end %}

No inner function. Just attributes on your function. LLVM inlines it into any caller with matching `#[target_feature]` — keeping everything in one optimization region.

{% mermaid() %}
flowchart TD
    PUB["Public API<br/>(no SIMD features)"] --> ARC["#[arcane] entry point<br/>(creates safe wrapper)"]
    ARC --> H1["#[rite] fn<br/>(inlines fully)"]
    ARC --> H2["#[rite] fn<br/>(inlines fully)"]
    H1 --> H3["#[rite] fn<br/>(inlines fully)"]

    PUB -.->|"unsafe needed<br/>if calling #[rite]<br/>directly"| H1

    style PUB fill:#5a3d1e,color:#fff
    style ARC fill:#2d5a27,color:#fff
    style H1 fill:#1a4a6e,color:#fff
    style H2 fill:#1a4a6e,color:#fff
    style H3 fill:#1a4a6e,color:#fff
{% end %}

## When to Use `#[rite]` vs `#[arcane]`

For most code, `#[arcane]` works everywhere — LLVM inlines the wrapper when features match (V3→V3 = zero overhead). `#[rite]` is available when you want direct `#[target_feature]` + `#[inline]` without a wrapper.

| Caller | `#[arcane]` | `#[rite]` |
|--------|-------------|-----------|
| From non-SIMD code | Required (generates safe wrapper) | N/A |
| From `#[arcane]`/`#[rite]` with matching features | Works (wrapper inlined away) | Also works (no wrapper) |

```rust
use archmage::prelude::*;

// ENTRY POINT: receives token from caller
#[arcane(import_intrinsics)]
pub fn dot_product(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let products = mul_vectors(a, b);       // tier-based — no token needed
    horizontal_sum(token, products)          // token-based — passes token on
}

// Tier-based: no token parameter, just specify the tier
#[rite(v3, import_intrinsics)]
fn mul_vectors(a: &[f32; 8], b: &[f32; 8]) -> __m256 {
    _mm256_mul_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b))
}

// Token-based: same behavior, token threaded through
#[rite(import_intrinsics)]
fn horizontal_sum(_token: X64V3Token, v: __m256) -> f32 {
    let sum = _mm256_hadd_ps(v, v);
    let sum = _mm256_hadd_ps(sum, sum);
    let low = _mm256_castps256_ps128(sum);
    let high = _mm256_extractf128_ps::<1>(sum);
    _mm_cvtss_f32(_mm_add_ss(low, high))
}
```

## What It Generates

<details>
<summary>Token-based expansion (click to expand)</summary>

```rust
// Your code:
#[rite(import_intrinsics)]
fn helper(_token: X64V3Token, v: __m256) -> __m256 {
    _mm256_add_ps(v, v)
}

// Generated (NO wrapper function):
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,bmi1,bmi2,...")]
#[inline]
fn helper(_token: X64V3Token, v: __m256) -> __m256 {
    use archmage::intrinsics::x86_64::*;
    _mm256_add_ps(v, v)
}
```

</details>

<details>
<summary>Tier-based expansion (click to expand)</summary>

```rust
// Your code:
#[rite(v3, import_intrinsics)]
fn helper(v: __m256) -> __m256 {
    _mm256_add_ps(v, v)
}

// Generated — identical attributes, no token parameter:
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,bmi1,bmi2,...")]
#[inline]
fn helper(v: __m256) -> __m256 {
    use archmage::intrinsics::x86_64::*;
    _mm256_add_ps(v, v)
}
```

The tier name (`v3`) maps to the same features as `X64V3Token`. No token appears in the generated code.

</details>

<details>
<summary>Multi-tier expansion (click to expand)</summary>

```rust
// Your code:
#[rite(v3, v4, import_intrinsics)]
fn process(data: &[f32; 4]) -> f32 {
    let v = _mm_loadu_ps(data);
    let sum = _mm_hadd_ps(v, v);
    let sum = _mm_hadd_ps(sum, sum);
    _mm_cvtss_f32(sum)
}

// Generated — two suffixed variants:
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,bmi1,bmi2,...")]
#[inline]
fn process_v3(data: &[f32; 4]) -> f32 {
    use archmage::intrinsics::x86_64::*;
    let v = _mm_loadu_ps(data);
    let sum = _mm_hadd_ps(v, v);
    let sum = _mm_hadd_ps(sum, sum);
    _mm_cvtss_f32(sum)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,...")]
#[inline]
fn process_v4(data: &[f32; 4]) -> f32 {
    use archmage::intrinsics::x86_64::*;
    let v = _mm_loadu_ps(data);
    let sum = _mm_hadd_ps(v, v);
    let sum = _mm_hadd_ps(sum, sum);
    _mm_cvtss_f32(sum)
}
```

The original function name disappears — only the suffixed variants exist. Each is compiled with different `#[target_feature]` attributes, so LLVM auto-vectorizes each with the available instructions for that tier.

</details>

<details>
<summary>Compare to #[arcane] which creates a wrapper</summary>

```rust
fn helper(_token: X64V3Token, v: __m256) -> __m256 {
    #[target_feature(enable = "avx2,fma,bmi1,bmi2,...")]
    #[inline]
    fn __inner(_token: X64V3Token, v: __m256) -> __m256 {
        _mm256_add_ps(v, v)
    }
    unsafe { __inner(_token, v) }
}
```

</details>

## Multi-Tier `#[rite]`

When you specify more than one tier, `#[rite]` generates a separate suffixed function for each:

```rust
use archmage::prelude::*;

// One function body, three compiled variants
#[rite(v3, v4, neon)]
fn scale(data: &[f32; 4], factor: f32) -> [f32; 4] {
    [
        data[0] * factor,
        data[1] * factor,
        data[2] * factor,
        data[3] * factor,
    ]
}
// Generates: scale_v3(), scale_v4(), scale_neon()
```

Each variant gets:
- Its own `#[target_feature]` (v3 gets AVX2+FMA, v4 gets AVX-512, neon gets NEON)
- `#[cfg(target_arch)]` gating (v3/v4 get `x86_64`, neon gets `aarch64`)
- `#[inline]` for inlining into callers

### Calling from `#[arcane]`

The primary use case is calling the right variant from within SIMD code:

```rust
#[arcane(import_intrinsics)]
fn entry(token: X64V3Token, data: &[f32; 4]) -> [f32; 4] {
    scale_v3(data, 2.0)  // Safe! Caller has V3 features, callee needs V3
}
```

Since Rust 1.86, a `#[target_feature]` function can safely call another `#[target_feature]` function when the caller has matching or superset features. The `#[arcane]` wrapper gives the caller V3 features, so calling `scale_v3()` (which also needs V3) requires no `unsafe`.

### When to use multi-tier

Multi-tier is useful when:
- You want LLVM to auto-vectorize the same scalar loop at different feature levels
- You're writing `incant!`-style dispatch helpers without `#[magetypes]`
- You need variants of a utility function for different platforms

For intrinsics-heavy code where each tier uses different instructions, write separate functions per tier instead.

### Multi-tier with options

All options (`import_intrinsics`, `import_magetypes`) work with multi-tier:

```rust
#[rite(v3, v4, import_intrinsics)]
fn process(data: &[f32; 4]) -> f32 {
    let v = _mm_loadu_ps(data);
    let sum = _mm_hadd_ps(v, v);
    let sum = _mm_hadd_ps(sum, sum);
    _mm_cvtss_f32(sum)
}
// Generates: process_v3(), process_v4() on x86_64
// Cfg'd out on other architectures
```

## Why This Works (Rust 1.86+)

Since Rust 1.86, calling a `#[target_feature]` function from another function with matching or superset features is **safe** — no `unsafe` block needed. This is what makes `#[rite]` functions callable from `#[arcane]` or other `#[rite]` functions without `unsafe`:

```rust
#[target_feature(enable = "avx2,fma")]
fn outer(data: &[f32; 8]) -> f32 {
    inner_add(data) + inner_mul(data)  // Safe! No unsafe needed!
}

#[target_feature(enable = "avx2")]
#[inline]
fn inner_add(data: &[f32; 8]) -> f32 { /* ... */ }

#[target_feature(enable = "avx2")]
#[inline]
fn inner_mul(data: &[f32; 8]) -> f32 { /* ... */ }
```

The caller's features (`avx2,fma`) are a superset of the callee's (`avx2`), so the compiler knows the call is safe.

This applies equally to multi-tier variants. When `process_v3()` is called from an `#[arcane]` function with V3 features, or from another `#[rite(v3)]` function, the call is safe because the features match.

## Direct Calls Require Unsafe

If you call a `#[rite]` function from outside a `#[target_feature]` context, you need `unsafe`:

```rust
#[test]
fn test_helper() {
    if let Some(token) = X64V3Token::summon() {
        // Direct call from test (no target_feature) requires unsafe
        let result = unsafe { helper(token, data) };
        assert_eq!(result, expected);
    }
}
```

This is correct—the test function doesn't have `#[target_feature]`, so the compiler can't verify safety at compile time. The `unsafe` block says "I checked at runtime via `summon()`."

The same applies to multi-tier variants — calling `process_v3()` from a test or non-SIMD context requires `unsafe`.

## Benefits

1. **No target-feature boundary**: Inlines into callers with matching features
2. **Better inlining**: LLVM sees the actual function with matching target attributes
3. **Cleaner stack traces**: No `__inner` functions in backtraces
4. **Syntactic sugar**: No need to manually maintain feature strings

## Choosing Between #[arcane] and #[rite]

**`#[arcane]` works everywhere** — entry points and helpers alike. When features match, LLVM inlines the wrapper away. `#[rite]` is an alternative for when you want explicit `#[target_feature]` + `#[inline]` without a wrapper.

| Situation | Recommended | Why |
|-----------|-------------|-----|
| Entry point / public API | `#[arcane]` | Generates safe wrapper for non-SIMD callers |
| Internal helper | `#[arcane]` or `#[rite]` | Both work; arcane wrapper inlines when features match |
| Multi-tier auto-vectorization | `#[rite(v3, v4, neon)]` | Generates suffixed variants from one body |
| Tokenless helper (no threading) | `#[rite(v3)]` | Cleaner — no token parameter needed |

## When to Use Tier-Based vs Token-Based

| Situation | Recommended |
|-----------|------------|
| Pure intrinsics helper (no magetypes, no forwarding) | `#[rite(v3)]` — cleaner, no token to thread |
| Uses magetypes types (`f32x8::load(token, ...)`) | `#[rite]` with token — magetypes needs the token |
| Passes token to other token-based `#[rite]` functions | `#[rite]` with token — already have it |
| Deep in a call chain, token is just passed through unused | `#[rite(v3)]` — drop the ceremony |
| Same body should compile for multiple tiers | `#[rite(v3, v4)]` — generates suffixed variants |

Both single-tier and token-based produce identical machine code. Multi-tier produces one copy per tier, each compiled with different features. The choice is about what you need.

## Tier Names

The tier name maps to the same features as the corresponding token. All `incant!` tier names work. Tier names accept the `_` prefix — `_v3` is identical to `v3`, matching the suffix on generated function names:

| Tier | Token | Architecture | Features |
|------|-------|-------------|----------|
| `v1` | `X64V1Token` | x86_64 | SSE, SSE2 (baseline) |
| `v2` | `X64V2Token` | x86_64 | + SSE4.2, POPCNT |
| `v3` | `X64V3Token` | x86_64 | + AVX2, FMA, BMI2 |
| `v4` / `avx512` | `X64V4Token` | x86_64 | + AVX-512 |
| `v4x` | `X64V4xToken` | x86_64 | + modern AVX-512 extensions |
| `neon` | `NeonToken` | aarch64 | NEON |
| `arm_v2` | `Arm64V2Token` | aarch64 | + CRC, RDM, DotProd, FP16 |
| `arm_v3` | `Arm64V3Token` | aarch64 | + SHA3, I8MM, BF16 |
| `wasm128` | `Wasm128Token` | wasm32 | SIMD128 |
| `x64_crypto` | `X64CryptoToken` | x86_64 | V2 + PCLMULQDQ, AES-NI |
| `v3_crypto` | `X64V3CryptoToken` | x86_64 | V3 + VPCLMULQDQ, VAES |

These are the same tier names used by `incant!` and `#[autoversion]`.

## Composition

`#[rite]` functions compose naturally — both token-based and tier-based:

```rust
// Tier-based helpers — no tokens needed
#[rite(v3, import_intrinsics)]
fn mul_vectors(a: &[f32; 8], b: &[f32; 8]) -> __m256 {
    _mm256_mul_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b))
}

#[rite(v3, import_intrinsics)]
fn load_vector(c: &[f32; 8]) -> __m256 {
    _mm256_loadu_ps(c)
}

#[rite(v3, import_intrinsics)]
fn add_raw(a: __m256, b: __m256) -> __m256 {
    _mm256_add_ps(a, b)
}

// Compose them — no tokens threaded through
#[rite(v3, import_intrinsics)]
fn complex_op(a: &[f32; 8], b: &[f32; 8], c: &[f32; 8]) -> __m256 {
    let ab = mul_vectors(a, b);
    let vc = load_vector(c);
    add_raw(ab, vc)
}
```

All `#[rite]` functions inline into the caller — no target-feature boundary, one optimization region.

## Cross-Architecture Behavior

Like `#[arcane]`, `#[rite]` cfg's out functions on non-matching architectures by default. On the wrong architecture, no function is emitted — no dead code.

```rust
// Only exists on x86_64 — cfg'd out on ARM/WASM
#[rite(v3, import_intrinsics)]
fn helper(v: __m256) -> __m256 {
    _mm256_add_ps(v, v)
}
```

For multi-tier, each variant gets its own `#[cfg(target_arch)]` guard:

```rust
#[rite(v3, neon)]
fn portable_helper(x: f32, y: f32) -> f32 { x + y }
// portable_helper_v3() on x86_64, portable_helper_neon() on aarch64
```

Use `incant!` for cross-arch dispatch — it handles all cfg gating automatically.

## Auto-Imports

`#[rite]` supports `import_intrinsics` and `import_magetypes`. Both work with all modes:

```rust
// Tier-based with auto-imports
#[rite(v3, import_intrinsics, import_magetypes)]
fn helper(data: &[f32; 8]) -> f32 {
    // core::arch::x86_64::* and magetypes::simd::v3::* both in scope
    let v = _mm256_loadu_ps(data);
    let _ = _mm256_add_ps(v, v);
    0.0
}

// Token-based with auto-imports (same behavior)
#[rite(import_intrinsics, import_magetypes)]
fn helper_with_token(token: X64V3Token, data: &[f32; 8]) -> f32 {
    let v = f32x8::load(token, data);
    v.reduce_add()
}
```

For most code, `#[arcane(import_intrinsics)]` works for both entry points and helpers. `#[rite]` is an alternative when you want direct `#[target_feature]` without a wrapper.

See [#\[arcane\] Options](@/archmage/concepts/arcane.md#import-intrinsics) for the full namespace mapping.

## Inlining Behavior

`#[rite]` uses `#[inline]` which is sufficient for full inlining when called from matching `#[target_feature]` context. This applies equally to all `#[rite]` modes — token-based, tier-based, and multi-tier variants all emit the same `#[inline]` attribute.

Benchmarks show `#[rite]` with `#[inline]` performs identically to manually inlined code — 547 ns vs 544 ns on 1000 8-float vector adds. `#[arcane]` calling `#[arcane]` with matching features also hits 547 ns — the wrapper inlines away. See the [full benchmark data](https://github.com/imazen/archmage/blob/main/docs/PERFORMANCE.md).

`#[inline(always)]` combined with `#[target_feature]` is not allowed on stable Rust, but we don't need it — `#[inline]` works perfectly.

## Options Reference

All options combine freely. Order doesn't matter.

| Option | Effect |
|--------|--------|
| `v3`, `neon`, `v4`, ... | Tier name — sets features (one = single function, multiple = suffixed variants) |
| `import_intrinsics` | Injects `use archmage::intrinsics::{arch}::*` (safe memory ops) |
| `import_magetypes` | Injects magetypes SIMD type imports |

**Examples:**

```rust
#[rite(v3)]                                    // single tier
#[rite(v3, import_intrinsics)]                 // single tier + safe intrinsics
#[rite(v3, v4, neon)]                          // multi-tier (generates _v3, _v4, _neon)
#[rite(v3, v4, import_intrinsics)]             // multi-tier + safe intrinsics
#[rite(import_intrinsics)]                     // token-based (reads token from params)
```
