# archmage

[![CI](https://github.com/imazen/archmage/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/archmage/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/archmage.svg)](https://crates.io/crates/archmage)
[![docs.rs](https://docs.rs/archmage/badge.svg)](https://docs.rs/archmage)
[![codecov](https://codecov.io/gh/imazen/archmage/graph/badge.svg)](https://codecov.io/gh/imazen/archmage)
[![MSRV](https://img.shields.io/crates/msrv/archmage.svg)](https://crates.io/crates/archmage)
[![License](https://img.shields.io/crates/l/archmage.svg)](https://github.com/imazen/archmage#license)

**[Browse 12,000+ SIMD Intrinsics →](https://imazen.github.io/archmage/intrinsics/)** · [Docs](https://imazen.github.io/archmage/) · [Magetypes](https://imazen.github.io/archmage/magetypes/) · [API Docs](https://docs.rs/archmage)

Archmage lets you write SIMD code in Rust without `unsafe`. It works on x86-64, AArch64, and WASM.

```toml
[dependencies]
archmage = "0.9"
```

## The problem

Raw SIMD in Rust requires `unsafe` for every intrinsic call:

```rust
use std::arch::x86_64::*;

// Every. Single. Call.
unsafe {
    let a = _mm256_loadu_ps(data.as_ptr());      // unsafe: raw pointer
    let b = _mm256_set1_ps(2.0);                  // unsafe: needs target_feature
    let c = _mm256_mul_ps(a, b);                  // unsafe: needs target_feature
    _mm256_storeu_ps(out.as_mut_ptr(), c);         // unsafe: raw pointer
}
```

Miss a feature check and you get undefined behavior on older CPUs. Wrap everything in `unsafe` and hope for the best.

## The solution

```rust
use archmage::prelude::*;  // tokens, traits, macros, intrinsics, safe memory ops

// X64V3Token = AVX2 + FMA (Haswell 2013+, Zen 1+)
#[arcane]
fn multiply(_token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    let a = _mm256_loadu_ps(data);          // safe: takes &[f32; 8], not *const f32
    let b = _mm256_set1_ps(2.0);            // safe: inside #[target_feature]
    let c = _mm256_mul_ps(a, b);            // safe: value-based (Rust 1.85+)
    let mut out = [0.0f32; 8];
    _mm256_storeu_ps(&mut out, c);          // safe: takes &mut [f32; 8]
    out
}

fn main() {
    // Runtime CPU check — returns None if AVX2+FMA unavailable
    if let Some(token) = X64V3Token::summon() {
        let result = multiply(token, &[1.0; 8]);
        println!("{:?}", result);
    }
}
```

No `unsafe` anywhere. Your crate can use `#![forbid(unsafe_code)]`.

## What changed in Rust 1.85

Rust 1.85 (Feb 2025) stabilized `target_feature_11`, which changed two things:

1. **`#[target_feature]` functions can be declared safe** (Rust 2024 edition). Previously they had to be `unsafe fn`. Now they're just `fn` — but calling them from code without matching features still requires `unsafe`.

2. **Value-based intrinsics are safe inside `#[target_feature]` functions.** `_mm256_add_ps`, `_mm256_mul_ps`, shuffles, compares — anything that doesn't touch a pointer is safe to call when the compiler knows the CPU features are enabled. Only pointer-based operations (`_mm256_loadu_ps(ptr)`, `_mm256_storeu_ps(ptr, v)`) remain unsafe.

This means the only `unsafe` left in SIMD code is: (a) the one call site where you cross from normal code into a `#[target_feature]` function, and (b) memory operations that take raw pointers. Archmage eliminates both.

## How archmage makes it zero-unsafe

```
┌─────────── Your crate: #![forbid(unsafe_code)] ───────────┐
│                                                            │
│   summon() ──→ Token ──→ #[arcane] fn ──→ #[rite] fn      │
│    (CPUID)    (proof)   (entry point)   (inlines freely)  │
│                              │                             │
│                   prelude / import_intrinsics               │
│                    ┌────────┴────────┐                     │
│                    │ core::arch    safe_unaligned_simd     │
│                    │ (value ops    (memory ops             │
│                    │  = safe)       = safe)                │
│                    └─────────────────┘                     │
│                                                            │
│   Result: all intrinsics are safe. No unsafe in your code. │
└────────────────────────────────────────────────────────────┘
```

Three pieces eliminate the remaining `unsafe`:

1. **Tokens** prove the CPU has the right features. `X64V3Token::summon()` checks CPUID and returns `Some(token)` only if AVX2+FMA are available. The token is zero-sized — passing it around costs nothing at runtime. Detection is cached (~1.3 ns), or compiles away entirely with `-Ctarget-cpu=haswell`.

2. **`#[arcane]` / `#[rite]`** read the token type from your function signature and generate `#[target_feature(enable = "avx2,fma,...")]`. The `#[arcane]` macro generates the `unsafe` call to cross the `#[target_feature]` boundary *inside its own generated code* — your crate never writes `unsafe`. `#[rite]` functions called from matching `#[target_feature]` contexts don't need `unsafe` at all (Rust 1.85+). Both macros handle `#[cfg(target_arch)]` gating automatically.

3. **Safe memory ops** from [`safe_unaligned_simd`](https://crates.io/crates/safe_unaligned_simd) (by [okaneco](https://github.com/okaneco)) shadow `core::arch`'s pointer-based versions. `_mm256_loadu_ps` takes `&[f32; 8]` instead of `*const f32`. Same function names, safe signatures. These are included in the prelude, or you can use `#[arcane(import_intrinsics)]` to scope them to a single function.

All tokens compile on all platforms. On the wrong architecture, `summon()` returns `None`. You rarely need `#[cfg(target_arch)]` in your code.

### Import styles

`use archmage::prelude::*` imports everything — tokens, traits, macros, all platform intrinsics, and safe memory ops. The examples in this README use the prelude for brevity:

```rust
use archmage::prelude::*;

#[arcane]  // intrinsics already in scope from prelude
fn example(_: X64V3Token, data: &[f32; 8]) -> __m256 {
    _mm256_loadu_ps(data)
}
```

If you don't want thousands of intrinsic names at module scope, use selective imports with `import_intrinsics` to scope them to the function body:

```rust
use archmage::{X64V3Token, SimdToken, arcane};

#[arcane(import_intrinsics)]  // injects intrinsics inside this function only
fn example(_: X64V3Token, data: &[f32; 8]) -> core::arch::x86_64::__m256 {
    _mm256_loadu_ps(data)  // in scope from import_intrinsics
}
```

Both import the same combined intrinsics module — using both is just duplication. The prelude is simpler; `import_intrinsics` is more explicit.

## `#[arcane]` vs `#[rite]`: entry point vs internal

**`#[rite]` should be your default.** Use `#[arcane]` only at the entry point — the first call from non-SIMD code.

`#[rite]` works three ways:

```rust
use archmage::prelude::*;

// Entry point: #[arcane] — safe wrapper for non-SIMD callers
#[arcane]
fn dot_product(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let products = mul_vectors(a, b);       // tier-based — no token!
    horizontal_sum(token, products)          // token-based — passes token
}

// Tier-based #[rite]: specify tier, no token parameter
#[rite(v3)]
fn mul_vectors(a: &[f32; 8], b: &[f32; 8]) -> __m256 {
    _mm256_mul_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b))
}

// Token-based #[rite]: reads features from token type
#[rite]
fn horizontal_sum(_: X64V3Token, v: __m256) -> f32 {
    let sum = _mm256_hadd_ps(v, v);
    let sum = _mm256_hadd_ps(sum, sum);
    let low = _mm256_castps256_ps128(sum);
    let high = _mm256_extractf128_ps::<1>(sum);
    _mm_cvtss_f32(_mm_add_ss(low, high))
}
```

Both `#[rite(v3)]` and `#[rite]` with an `X64V3Token` parameter produce identical `#[target_feature]` output. The tier-based form is shorter; the token form can be easier to remember if you already have the token in scope.

### Multi-tier `#[rite]`

When you specify multiple tiers, `#[rite]` generates a suffixed variant for each:

```rust
use archmage::prelude::*;

// Generates process_v3() and process_v4() — each with correct #[target_feature]
#[rite(v3, v4)]
fn process(data: &[f32; 4]) -> f32 {
    data[0] + data[1] + data[2] + data[3]
}
```

Call the suffixed variant from a matching `#[arcane]` or `#[rite]` context — since Rust 1.85, a `#[target_feature]` function can safely call another `#[target_feature]` function when both have matching (or superset) features:

```rust
#[arcane]
fn entry(token: X64V3Token, data: &[f32; 4]) -> f32 {
    process_v3(data)  // safe — caller has V3 features, callee needs V3
}
```

This is useful when writing a single function body that should be compiled for multiple tiers — the compiler applies different `#[target_feature]` attributes to each copy, and LLVM auto-vectorizes each variant with the available instructions.

Tier names: `v1`, `v2`, `v3`, `v4`, `neon`, `arm_v2`, `wasm128`, etc. — the same names used by `incant!`.

### Why two macros?

`#[arcane]` generates a safe wrapper that crosses the `#[target_feature]` boundary — LLVM can't optimize across it. `#[rite]` adds `#[target_feature]` + `#[inline]` directly, so LLVM inlines it into the caller. Same features = no boundary.

Processing 1000 8-float vector additions ([full benchmark details](docs/PERFORMANCE.md)):

| Pattern | Time | Why |
|---------|------|-----|
| `#[rite]` inside `#[arcane]` | 547 ns | Features match — LLVM inlines |
| `#[arcane]` per iteration | 2209 ns (4x) | Target-feature boundary per call |
| Bare `#[target_feature]` (no archmage) | 2222 ns (4x) | Same boundary — archmage adds nothing |

The 4x penalty is LLVM's `#[target_feature]` boundary, not archmage overhead. Bare `#[target_feature]` without archmage has the same cost. With real workloads (DCT-8), the boundary costs up to 6.2x.

**The rule:** `#[arcane]` once at the entry point, `#[rite]` for everything called from SIMD code.

For trait impls, use `#[arcane(_self = Type)]` — a nested inner-function approach (since sibling would add methods not in the trait definition).

## Auto-vectorization with `#[autoversion]`

Don't want to write intrinsics? Write plain scalar code and let the compiler vectorize it:

```rust
use archmage::prelude::*;

#[autoversion]
fn sum_of_squares(_token: SimdToken, data: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &x in data {
        sum += x * x;
    }
    sum
}

// Call directly — no token needed, no unsafe:
let result = sum_of_squares(&my_data);
```

`#[autoversion]` generates a separate copy of your function for each architecture tier — each compiled with `#[target_feature]` to unlock the auto-vectorizer — plus a runtime dispatcher that picks the best one. On x86-64 with AVX2+FMA, that loop compiles to `vfmadd231ps` (8 floats per cycle). On ARM, you get `fmla`. The `_scalar` fallback compiles without SIMD features as a safety net.

The `_token: SimdToken` parameter is a placeholder — you don't use it in the body. The macro replaces it with concrete token types (`X64V3Token`, `NeonToken`, etc.) for each variant.

**What gets generated** (default tiers):

- `sum_of_squares_v4(token: X64V4Token, ...)` — AVX-512 (with `avx512` feature)
- `sum_of_squares_v3(token: X64V3Token, ...)` — AVX2+FMA
- `sum_of_squares_neon(token: NeonToken, ...)` — AArch64 NEON
- `sum_of_squares_wasm128(token: Wasm128Token, ...)` — WASM SIMD
- `sum_of_squares_scalar(token: ScalarToken, ...)` — no SIMD
- `sum_of_squares(data: &[f32]) -> f32` — **dispatcher** (token param removed)

Explicit tiers: `#[autoversion(v3, neon)]`. `scalar` is always implicit.

For inherent methods, `self` works naturally — no special parameters needed. For trait method delegation, use `#[autoversion(_self = MyType)]` and `_self` in the body. See the [full parameter reference](https://imazen.github.io/archmage/archmage/dispatch/autoversion/) or the [API docs](https://docs.rs/archmage/latest/archmage/attr.autoversion.html).

**When to use which:**

| | `#[autoversion]` | `#[arcane]` + `#[rite]` |
|---|---|---|
| You write | Scalar loops | SIMD intrinsics |
| Vectorization | Compiler auto-vectorizes | You choose the instructions |
| Lines of code | 1 attribute | Manual variant + dispatch |
| Best for | Simple numeric loops | Hand-tuned SIMD kernels |

## SIMD types with `magetypes`

`magetypes` provides ergonomic SIMD vector types (`f32x8`, `i32x4`, etc.) with natural Rust operators. It's an exploratory companion crate — the API may change between releases.

```toml
[dependencies]
archmage = "0.9"
magetypes = "0.9"
```

```rust
use archmage::prelude::*;
use magetypes::simd::f32x8;

fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    if let Some(token) = X64V3Token::summon() {
        dot_product_simd(token, a, b)
    } else {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }
}

#[arcane]
fn dot_product_simd(token: X64V3Token, a: &[f32], b: &[f32]) -> f32 {
    let mut sum = f32x8::zero(token);
    for (a_chunk, b_chunk) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        let va = f32x8::load(token, a_chunk.try_into().unwrap());
        let vb = f32x8::load(token, b_chunk.try_into().unwrap());
        sum = va.mul_add(vb, sum);
    }
    sum.reduce_add()
}
```

`f32x8` wraps `__m256` on x86 with AVX2. On ARM/WASM, it's polyfilled with two `f32x4` operations — same API, automatic fallback. The `#[arcane]` wrapper lets LLVM optimize the entire loop as a single SIMD region.

## Runtime dispatch with `incant!` <sub>(alias: `dispatch_variant!`)</sub>

Write platform-specific variants with concrete types, then dispatch at runtime:

```rust
use archmage::incant;
#[cfg(target_arch = "x86_64")]
use magetypes::simd::f32x8;

#[cfg(target_arch = "x86_64")]
fn sum_squares_v3(token: archmage::X64V3Token, data: &[f32]) -> f32 {
    let chunks = data.chunks_exact(8);
    let mut acc = f32x8::zero(token);
    for chunk in chunks {
        let v = f32x8::from_array(token, chunk.try_into().unwrap());
        acc = v.mul_add(v, acc);
    }
    acc.reduce_add() + chunks.remainder().iter().map(|x| x * x).sum::<f32>()
}

fn sum_squares_scalar(_token: archmage::ScalarToken, data: &[f32]) -> f32 {
    data.iter().map(|x| x * x).sum()
}

/// Dispatches to the best available at runtime.
fn sum_squares(data: &[f32]) -> f32 {
    incant!(sum_squares(data), [v3])
}
```

Each variant's first parameter is the matching token type — `_v3` takes `X64V3Token`, `_neon` takes `NeonToken`, etc. A `_scalar` variant (taking `ScalarToken`) is always required as the fallback.

`incant!` wraps each tier's call in `#[cfg(target_arch)]` and `#[cfg(feature)]` guards, so you only define variants for architectures you target. With no explicit tier list, `incant!` dispatches to `v3`, `neon`, `wasm128`, and `scalar` by default (plus `v4` if the `avx512` feature is enabled).

Known tiers: `v1`, `v2`, `x64_crypto`, `v3`, `v3_crypto`, `v4`, `v4x`, `arm_v2`, `arm_v3`, `neon`, `neon_aes`, `neon_sha3`, `neon_crc`, `wasm128`, `scalar`.

If you already have a token, use `with` to dispatch on its concrete type: `incant!(func(data) with token)`.

## Tokens

| Token | Alias | Features | Hardware |
|-------|-------|----------|----------|
| `X64V1Token` | `Sse2Token` | SSE, SSE2 | x86_64 baseline (always available) |
| `X64V2Token` | | + SSE4.2, POPCNT | Nehalem 2008+ |
| `X64CryptoToken` | | V2 + PCLMULQDQ, AES-NI | Westmere 2010+ |
| `X64V3Token` | — | + AVX2, FMA, BMI2 | Haswell 2013+, Zen 1+ |
| `X64V3CryptoToken` | | V3 + VPCLMULQDQ, VAES | Zen 3+ 2020, Alder Lake 2021+ |
| `X64V4Token` | `Server64` | + AVX-512 (requires `avx512` feature) | Skylake-X 2017+, Zen 4+ |
| `NeonToken` | `Arm64` | NEON | All 64-bit ARM |
| `Arm64V2Token` | | + CRC, RDM, DotProd, FP16, AES, SHA2 | A55+, M1+, Graviton 2+ |
| `Arm64V3Token` | | + FHM, FCMA, SHA3, I8MM, BF16 | A510+, M2+, Snapdragon X |
| `Wasm128Token` | | WASM SIMD | Compile-time only |
| `ScalarToken` | | (none) | Always available |

Higher tokens subsume lower ones: `X64V4Token` → `X64V3Token` → `X64V2Token` → `X64V1Token`. Downcasting is free (zero-cost). `#[arcane(stub)]` generates unreachable stubs on non-matching architectures when you need cross-arch dispatch without `#[cfg]` guards. `incant!` handles cfg-gating automatically.

See [`token-registry.toml`](token-registry.toml) for the complete mapping of tokens to CPU features.

## Testing SIMD dispatch paths

`for_each_token_permutation` tests every `incant!` dispatch path on your native hardware — no cross-compilation needed. It disables tokens one at a time, running your closure at each tier from "all SIMD enabled" down to "scalar only":

```rust
use archmage::testing::{for_each_token_permutation, CompileTimePolicy};

#[test]
fn sum_squares_matches_across_tiers() {
    let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let expected: f32 = data.iter().map(|x| x * x).sum();

    let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let result = sum_squares(&data);
        assert!(
            (result - expected).abs() < 1e-1,
            "mismatch at tier: {perm}"
        );
    });

    assert!(report.permutations_run >= 2, "expected multiple tiers");
}
```

On an AVX-512 machine this runs 5–7 permutations; on Haswell, 3. Tokens the CPU doesn't have are skipped.

If you compiled with `-Ctarget-cpu=native`, the compiler bakes feature detection into the binary and tokens can't be disabled at runtime. Use the `testable_dispatch` feature to force runtime detection in CI:

```toml
[dev-dependencies]
archmage = { version = "0.9", features = ["testable_dispatch"] }
```

For manual single-token testing, `lock_token_testing()` serializes against parallel tests. See the [testing docs](https://imazen.github.io/archmage/archmage/testing/dispatch-testing/) for `CompileTimePolicy::Fail`, env-var integration, and `dangerously_disable_token_process_wide`.

## Feature flags

| Feature | Default | |
|---------|---------|---|
| `std` | yes | Standard library (required for runtime detection) |
| `macros` | yes | `#[arcane]`, `#[rite]`, `#[autoversion]`, `#[magetypes]`, `incant!` |
| `avx512` | no | AVX-512 tokens (`X64V4Token`, `X64V4xToken`, `Avx512Fp16Token`) |
| `testable_dispatch` | no | Makes token disabling work with `-Ctarget-cpu=native` |

## Acknowledgments

- **[`safe_unaligned_simd`](https://crates.io/crates/safe_unaligned_simd)** by [okaneco](https://github.com/okaneco) — Reference-based wrappers for every SIMD load/store intrinsic across x86, ARM, and WASM. This crate closed the last `unsafe` gap: `_mm256_loadu_ps` taking `*const f32` was the one thing you couldn't make safe without a wrapper. Archmage depends on it and re-exports its functions through `import_intrinsics`, shadowing `core::arch`'s pointer-based versions automatically.

## License

MIT OR Apache-2.0
