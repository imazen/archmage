# archmage [![CI](https://img.shields.io/github/actions/workflow/status/imazen/archmage/ci.yml?style=flat-square)](https://github.com/imazen/archmage/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/archmage?style=flat-square)](https://crates.io/crates/archmage) [![lib.rs](https://img.shields.io/crates/v/archmage?style=flat-square&label=lib.rs&color=blue)](https://lib.rs/crates/archmage) [![docs.rs](https://img.shields.io/docsrs/archmage?style=flat-square)](https://docs.rs/archmage) [![license](https://img.shields.io/crates/l/archmage?style=flat-square)](https://github.com/imazen/archmage#license)

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

## How Rust enforces SIMD safety

Rust 1.85 (Feb 2025) changed the rules for `#[target_feature]` functions:

```
  Rust's #[target_feature] call rules (1.85+, 2024 edition)

  ┌─────────────────────────┐         ┌──────────────────────────────┐
  │  fn normal_code()       │ unsafe  │ #[target_feature(avx2, fma)] │
  │                         │────────▶│ fn simd_work()               │
  │  (no target features)   │         │                              │
  └─────────────────────────┘         └──────────────┬───────────────┘
                                                     │
          Calling simd_work() from                   │ safe
          normal_code() requires                     │ (subset of
          unsafe { }. The caller                     ▼ caller's features)
          has fewer features.          ┌──────────────────────────────┐
                                       │ #[target_feature(avx2)]      │
                                       │ fn simd_helper()             │
                                       │                              │
                                       └──────────────────────────────┘

  Caller has same or superset features? Safe call. No unsafe needed.
  Caller has fewer features?            Rust requires an unsafe block.
```

Inside a `#[target_feature]` function, **value-based intrinsics are safe** — `_mm256_add_ps`, shuffles, compares, anything that doesn't touch a pointer. Only pointer-based memory ops (`_mm256_loadu_ps(ptr)`) remain unsafe.

Two gaps remain:

1. **The boundary crossing.** The first call from normal code into a `#[target_feature]` function requires `unsafe`. Someone has to verify the CPU actually has those features.

2. **Memory operations.** `_mm256_loadu_ps` takes `*const f32`. Raw pointers need `unsafe`.

## How archmage closes both gaps

Archmage makes the boundary crossing **sound** by tying it to runtime CPU detection. You can't cross without proof.

```
  Your crate: #![forbid(unsafe_code)]

  ┌────────────────────────────────────────────────────────────┐
  │                                                            │
  │  1. summon()          Checks CPUID. Returns Some(token)    │
  │     X64V3Token        only if AVX2+FMA are present.        │
  │                       Token is zero-sized proof.           │
  │          │                                                 │
  │          ▼                                                 │
  │  2. #[arcane] fn      Reads token type from signature.     │
  │     entry(token, ..)  Generates #[target_feature] sibling. │
  │                       Wraps the unsafe call internally —   │
  │          │            your code never writes unsafe.        │
  │          ▼                                                 │
  │  3. #[rite] fns       Same #[target_feature] as caller.    │
  │     + plain fns       Safe calls — Rust allows it.         │
  │                       Inline freely, no boundary.          │
  │          │                                                 │
  │          ▼                                                 │
  │  4. Intrinsics        Value ops: safe (Rust 1.85+)         │
  │     in scope          Memory ops: safe_unaligned_simd      │
  │                       takes &[f32; 8], not *const f32      │
  │                                                            │
  │  Result: all intrinsics are safe. No unsafe in your code.  │
  └────────────────────────────────────────────────────────────┘
```

**Tokens are grouped by common CPU tiers.** `X64V3Token` covers AVX2+FMA+BMI2 — the set that Haswell (2013) and Zen 1+ share. `NeonToken` covers AArch64 NEON. `Arm64V2Token` covers CRC+RDM+DotProd+FP16+AES+SHA2 — the set that Apple M1, Cortex-A55+, and Graviton 2+ share. You pick a tier, not individual features. `summon()` checks all features in the tier atomically; it either succeeds (every feature present) or returns `None`. The token is zero-sized — passing it costs nothing. Detection is cached (~1.3 ns), or compiles away entirely with `-Ctarget-cpu=haswell`.

**`#[arcane]` is the trampoline.** It generates a sibling function with `#[target_feature(enable = "avx2,fma,...")]` and an `#[inline(always)]` wrapper that calls it through `unsafe`. The macro generates the `unsafe` block, not you. Since the token's existence proves the features are present, the call is sound. From inside the `#[arcane]` function, you can use intrinsics directly (value ops are safe) and call `#[rite]` functions with matching features (safe under Rust 1.85+). Both macros handle `#[cfg(target_arch)]` gating automatically.

**[`safe_unaligned_simd`](https://crates.io/crates/safe_unaligned_simd)** (by [okaneco](https://github.com/okaneco)) closes the memory gap. It shadows `core::arch`'s pointer-based load/store functions with reference-based versions — `_mm256_loadu_ps` takes `&[f32; 8]` instead of `*const f32`. Same names, safe signatures. Archmage re-exports these through `import_intrinsics`, so the safe versions are in scope automatically.

All tokens compile on all platforms. On the wrong architecture, `summon()` returns `None`. You rarely need `#[cfg(target_arch)]` in your code.

### Why `#![forbid(unsafe_code)]` matters for AI-written SIMD

AI is a patient compiler. It can write SIMD intrinsics, run benchmarks, iterate on hot loops, and try instruction sequences that no human would bother testing. Constraining AI to `#![forbid(unsafe_code)]` means it can't introduce undefined behavior — the type system catches unsound calls at compile time. The result is hand-tuned SIMD that's both fast and provably safe.

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

## Which macro do I use?

```
                    Writing a SIMD function?
                    ┌───────────┴───────────┐
            Hand-written                Scalar code,
            intrinsics              compiler auto-vectorizes
                │                           │
        Called from                  Need manual control
        non-SIMD code?              over dispatch?
     (after summon(), main)
        ┌─────┴─────┐              ┌─────┴─────┐
       YES          NO             NO          YES
        │            │              │            │
  ┌─────▼──────┐ ┌──▼────────┐ ┌───▼─────────┐ ┌▼────────────────┐
  │ #[arcane]  │ │ #[rite]   │ │#[autoversion]│ │ #[arcane] per   │
  │            │ │           │ │             │ │ variant +       │
  │ Entry from │ │ Inlines   │ │ Generates   │ │ incant!()       │
  │ normal code│ │ into the  │ │ variants +  │ │                 │
  │ One per    │ │ caller's  │ │ dispatcher  │ │ You write each  │
  │ hot path   │ │ LLVM      │ │ from one    │ │ #[arcane] fn    │
  │            │ │ region    │ │ scalar body │ │ and dispatch    │
  └─────┬──────┘ └──▲──┬────┘ └─────────────┘ └─────────────────┘
        │           │  │
        └───────────┘  └──▶ also calls #[rite]
     #[arcane] calls       and plain #[inline(always)]
     #[rite] helpers       fns (get caller's features)
```

Plain `#[inline(always)]` functions with no macro also work — if they inline into an `#[arcane]` or `#[rite]` caller, LLVM compiles them with the caller's features for free.

## `#[arcane]` vs `#[rite]`: entry point vs internal

**`#[rite]` should be your default.** Use `#[arcane]` only at the entry point — the first call from non-SIMD code.

### Two `#[rite]` syntaxes — same output

```rust
use archmage::{X64V3Token, rite};
use core::arch::x86_64::__m256;

// Tier-based: specify the tier name, no token parameter needed
#[rite(v3, import_intrinsics)]
fn mul_vectors(a: &[f32; 8], b: &[f32; 8]) -> __m256 {
    _mm256_mul_ps(_mm256_loadu_ps(a), _mm256_loadu_ps(b))
}

// Token-based: reads features from the token type in the signature
#[rite(import_intrinsics)]
fn horizontal_sum(_: X64V3Token, v: __m256) -> f32 {
    let sum = _mm256_hadd_ps(v, v);
    let sum = _mm256_hadd_ps(sum, sum);
    let low = _mm256_castps256_ps128(sum);
    let high = _mm256_extractf128_ps::<1>(sum);
    _mm_cvtss_f32(_mm_add_ss(low, high))
}
```

Both produce identical `#[target_feature(enable = "avx2,fma,...")]` output. The tier-based form is shorter; the token form is handy when you already have the token in scope.

### `#[arcane]` at the entry point

```rust
use archmage::{X64V3Token, SimdToken, arcane};

#[arcane(import_intrinsics)]
fn dot_product(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let products = mul_vectors(a, b);       // #[rite(v3)] — no token needed
    horizontal_sum(token, products)          // #[rite] — reads token type
}

fn main() {
    if let Some(token) = X64V3Token::summon() {
        let result = dot_product(token, &[1.0; 8], &[2.0; 8]);
        println!("{result}");
    }
}
```

`#[arcane]` generates an `#[inline(always)]` wrapper that crosses the `#[target_feature]` boundary via `unsafe` — internally, in generated code your crate never sees. From inside, you call `#[rite]` functions freely (same features = safe call).

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

### Multi-tier `#[rite]` (rare)

Occasionally you have scalar code called from inside SIMD functions where `#[inline(always)]` isn't viable and you don't want `#[autoversion]`'s dispatch overhead. In that case, give `#[rite]` multiple tiers and it generates a suffixed variant for each:

```rust
use archmage::prelude::*;

// Generates normalize_v3() and normalize_neon() —
// each compiled with different #[target_feature], auto-vectorized separately
#[rite(v3, neon)]
fn normalize(data: &mut [f32]) {
    let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if max != 0.0 {
        for x in data.iter_mut() { *x /= max; }
    }
}
```

Call the matching suffixed variant from an `#[arcane]` or `#[rite]` context:

```rust
use archmage::prelude::*;

#[arcane]
fn process(token: X64V3Token, data: &mut [f32]) {
    normalize_v3(data);  // safe — caller has V3 features, callee needs V3
}
```

These are `#[target_feature]` functions — you can't call them from non-SIMD code or dispatch them via `incant!`. They're for internal use within a SIMD call chain.

## Auto-vectorization with `#[autoversion]`

Don't want to write intrinsics? Write plain scalar code and let the compiler vectorize it:

```rust
use archmage::prelude::*;

#[autoversion]
fn sum_of_squares(data: &[f32]) -> f32 {
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

The macro auto-injects a hidden `SimdToken` parameter internally — you don't need to add one. The generated dispatcher has your original signature. You can optionally write `_token: SimdToken` if you prefer the explicit style, but it's not required.

**What gets generated** (default tiers):

- `sum_of_squares_v4(token: X64V4Token, ...)` — AVX-512 (with `avx512` feature)
- `sum_of_squares_v3(token: X64V3Token, ...)` — AVX2+FMA
- `sum_of_squares_neon(token: NeonToken, ...)` — AArch64 NEON
- `sum_of_squares_wasm128(token: Wasm128Token, ...)` — WASM SIMD
- `sum_of_squares_scalar(token: ScalarToken, ...)` — no SIMD
- `sum_of_squares(data: &[f32]) -> f32` — **dispatcher** (token param removed)

Explicit tiers: `#[autoversion(v3, neon)]`. `scalar` is always implicit. Use modifiers to tweak defaults: `#[autoversion(+arm_v2)]` adds a tier, `#[autoversion(-wasm128)]` removes one. Gate a tier on a Cargo feature: `#[autoversion(v4(cfg(avx512)), v3, neon)]`.

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

## Tier naming conventions

`incant!` and `#[autoversion]` dispatch to suffixed functions. `incant!(sum(data))` calls `sum_v3`, `sum_neon`, etc. These suffixes correspond to tokens:

| Suffix | Token | Arch | Key features |
|--------|-------|------|------|
| `_v1` | `X64V1Token` | x86_64 | SSE2 (baseline) |
| `_v2` | `X64V2Token` | x86_64 | + SSE4.2, POPCNT |
| `_v3` | `X64V3Token` | x86_64 | + AVX2, FMA, BMI2 |
| `_v4` | `X64V4Token` | x86_64 | + AVX-512 |
| `_neon` | `NeonToken` | aarch64 | NEON |
| `_arm_v2` | `Arm64V2Token` | aarch64 | + CRC, RDM, DotProd, AES, SHA2 |
| `_arm_v3` | `Arm64V3Token` | aarch64 | + SHA3, I8MM, BF16 |
| `_wasm128` | `Wasm128Token` | wasm32 | SIMD128 |
| `_scalar` | `ScalarToken` | any | No SIMD (always available) |

By default, `incant!` tries `_v4` (if the `avx512` feature is enabled), `_v3`, `_neon`, `_wasm128`, then `_scalar`. You can restrict to specific tiers: `incant!(sum(data), [v3, neon, scalar])`. Tier names accept the `_` prefix — `_v3` is identical to `v3`, matching the suffix on generated function names.

Instead of restating the entire default list, use modifiers: `[+arm_v2]` adds a tier, `[-wasm128]` removes one, `[+v4]` makes v4 unconditional. Gate a tier on a Cargo feature with `v4(cfg(avx512))` (shorthand: `v4(avx512)`). All entries must be modifiers or all plain — mixing is a compile error.

Always include `scalar` (or `default`) in explicit tier lists — it documents the fallback path. (Currently auto-appended if omitted; will become a compile error in v1.0.)

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
    incant!(sum_squares(data), [v3, scalar])
}
```

Each variant's first parameter is the matching token type — `_v3` takes `X64V3Token`, `_neon` takes `NeonToken`, etc.

**`_scalar` is mandatory.** `incant!` always emits an unconditional call to `fn_scalar(ScalarToken, ...)` as the final fallback. If the `_scalar` function doesn't exist, you get a compile error — not a runtime failure. Always include `scalar` in explicit tier lists (e.g., `[v3, neon, scalar]`) to document this dependency. Currently `scalar` is auto-appended if omitted; this will become a compile error in v1.0.

`incant!` wraps each tier's call in `#[cfg(target_arch)]` and `#[cfg(feature)]` guards, so you only define variants for architectures you target. With no explicit tier list, `incant!` dispatches to `v3`, `neon`, `wasm128`, and `scalar` by default (plus `v4` if the `avx512` feature is enabled).

Gate a tier on a Cargo feature with the `tier(cfg(feature))` syntax: `incant!(sum(data), [v4(cfg(avx512)), v3, neon, scalar])`. The shorthand `v4(avx512)` also works.

Use modifiers to tweak the default tier list without restating it: `[+arm_v2]` adds a tier, `[-wasm128]` removes one. All entries must be modifiers or all plain.

Known tiers: `v1`, `v2`, `x64_crypto`, `v3`, `v3_crypto`, `v4`, `v4x`, `arm_v2`, `arm_v3`, `neon`, `neon_aes`, `neon_sha3`, `neon_crc`, `wasm128`, `scalar`.

If you already have a token, use `with` to dispatch on its concrete type: `incant!(func(data) with token, [v3, neon, scalar])`. This uses `IntoConcreteToken` for compile-time monomorphized dispatch — no runtime summon.

### Token position

Use `Token` to mark where the summoned token is placed: `incant!(process(data, Token), [v3, scalar])` puts the token last. Without `Token`, the token is prepended.

### Zero-overhead nesting

Inside `#[arcane]`, `#[rite]`, or `#[autoversion]` bodies, `incant!` is automatically rewritten to a direct call — no runtime dispatch. The rewriter handles downcasting, upgrade attempts, and feature-gated tiers. See the [dispatch docs](https://imazen.github.io/archmage/archmage/dispatch/incant/) for details.

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

`for_each_token_permutation` tests every `incant!` dispatch path on your native hardware — no cross-compilation needed. It disables tokens one at a time, running your closure at each combination from "all SIMD enabled" down to "scalar only":

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

On an AVX-512 machine this runs 5–7 permutations; on Haswell, 3. Tokens the CPU doesn't have are skipped automatically.

### The `testable_dispatch` feature

When you compile with `-Ctarget-cpu=native` (or `-Ctarget-cpu=haswell`, etc.), Rust bakes the feature checks into the binary at compile time. `X64V3Token::summon()` compiles to a constant `Some(token)` — it can't be disabled at runtime, and `for_each_token_permutation` can't test fallback paths.

The `testable_dispatch` feature forces runtime detection even when compile-time detection would succeed. Enable it in dev-dependencies for full permutation coverage:

```toml
[dev-dependencies]
archmage = { version = "0.9", features = ["testable_dispatch"] }
```

`CompileTimePolicy` controls what happens when a token can't be disabled:

| Policy | Behavior |
|--------|----------|
| `Warn` | Skip the token, collect a warning in the report |
| `WarnStderr` | Same, plus print each warning to stderr |
| `Fail` | Panic — use in CI with `testable_dispatch` where full coverage is expected |

### Serializing parallel tests with `lock_token_testing()`

Token disabling is process-wide — if `cargo test` runs tests in parallel, one test disabling `X64V3Token` would break another test that expects it to be available. `for_each_token_permutation` acquires an internal mutex automatically. If you need to disable tokens manually (via `dangerously_disable_token_process_wide`), wrap your test in `lock_token_testing()` to serialize against other permutation tests:

```rust
use archmage::testing::lock_token_testing;
use archmage::{X64V3Token, SimdToken};

#[test]
fn manual_disable_test() {
    let _lock = lock_token_testing();
    let baseline = my_function(&data);
    X64V3Token::dangerously_disable_token_process_wide(true).unwrap();
    let fallback = my_function(&data);
    X64V3Token::dangerously_disable_token_process_wide(false).unwrap();
    assert_eq!(baseline, fallback);
}
```

The lock is reentrant — `for_each_token_permutation` called from within a `lock_token_testing` scope works without deadlock.

For the full testing API, see the [testing docs](https://imazen.github.io/archmage/archmage/testing/dispatch-testing/).

## Feature flags

| Feature | Default | |
|---------|---------|---|
| `std` | yes | Standard library (required for runtime detection) |
| `macros` | yes | No-op (macros are always available). Kept for backwards compatibility |
| `avx512` | no | AVX-512 tokens (`X64V4Token`, `X64V4xToken`, `Avx512Fp16Token`) |
| `testable_dispatch` | no | Makes token disabling work with `-Ctarget-cpu=native` |

## Acknowledgments

- **[`safe_unaligned_simd`](https://crates.io/crates/safe_unaligned_simd)** by [okaneco](https://github.com/okaneco) — Reference-based wrappers for every SIMD load/store intrinsic across x86, ARM, and WASM. This crate closed the last `unsafe` gap: `_mm256_loadu_ps` taking `*const f32` was the one thing you couldn't make safe without a wrapper. Archmage depends on it and re-exports its functions through `import_intrinsics`, shadowing `core::arch`'s pointer-based versions automatically.

## Image tech I maintain

| | |
|:--|:--|
| State of the art codecs<sup>[1]</sup> | [zenjpeg] · [zenpng] · [zenwebp] · [zengif] · [zenavif] ([rav1d-safe] · [zenrav1e] · [zenavif-parse] · [zenavif-serialize]) · [zenjxl] ([jxl-encoder] · [zenjxl-decoder]) · [zentiff] · [zenbitmaps] · [heic] · [zenraw] · [zenpdf] · [ultrahdr] · [mozjpeg-rs] · [webpx] |
| Compression | [zenflate] · [zenzop] |
| Processing | [zenresize] · [zenfilters] · [zenquant] · [zenblend] |
| Metrics | [zensim] · [fast-ssim2] · [butteraugli] · [resamplescope-rs] · [codec-eval] · [codec-corpus] |
| Pixel types & color | [zenpixels] · [zenpixels-convert] · [linear-srgb] · [garb] |
| Pipeline | [zenpipe] · [zencodec] · [zencodecs] · [zenlayout] · [zennode] |
| ImageResizer | [ImageResizer] (C#) — 24M+ NuGet downloads across all packages |
| [Imageflow][] | Image optimization engine (Rust) — [.NET][imageflow-dotnet] · [node][imageflow-node] · [go][imageflow-go] — 9M+ NuGet downloads across all packages |
| [Imageflow Server][] | [The fast, safe image server](https://www.imazen.io/) (Rust+C#) — 552K+ NuGet downloads, deployed by Fortune 500s and major brands |

<sup>[1]</sup> <sub>as of 2026</sub>

### General Rust awesomeness

**archmage** · [magetypes] · [enough] · [whereat] · [zenbench] · [cargo-copter]

[And other projects](https://www.imazen.io/open-source) · [GitHub @imazen](https://github.com/imazen) · [GitHub @lilith](https://github.com/lilith) · [lib.rs/~lilith](https://lib.rs/~lilith) · [NuGet](https://www.nuget.org/profiles/imazen) (over 30 million downloads / 87 packages)

[zenjpeg]: https://crates.io/crates/zenjpeg
[zenpng]: https://crates.io/crates/zenpng
[zenwebp]: https://crates.io/crates/zenwebp
[zengif]: https://crates.io/crates/zengif
[zenavif]: https://crates.io/crates/zenavif
[rav1d-safe]: https://crates.io/crates/rav1d-safe
[zenrav1e]: https://crates.io/crates/zenrav1e
[zenavif-parse]: https://crates.io/crates/zenavif-parse
[zenavif-serialize]: https://crates.io/crates/zenavif-serialize
[zenjxl]: https://crates.io/crates/zenjxl
[jxl-encoder]: https://crates.io/crates/jxl-encoder
[zenjxl-decoder]: https://crates.io/crates/zenjxl-decoder
[zentiff]: https://crates.io/crates/zentiff
[zenbitmaps]: https://crates.io/crates/zenbitmaps
[heic]: https://crates.io/crates/heic
[zenraw]: https://crates.io/crates/zenraw
[zenpdf]: https://crates.io/crates/zenpdf
[ultrahdr]: https://crates.io/crates/ultrahdr
[mozjpeg-rs]: https://crates.io/crates/mozjpeg-rs
[webpx]: https://crates.io/crates/webpx
[zenflate]: https://crates.io/crates/zenflate
[zenzop]: https://crates.io/crates/zenzop
[zenresize]: https://crates.io/crates/zenresize
[zenfilters]: https://crates.io/crates/zenfilters
[zenquant]: https://crates.io/crates/zenquant
[zenblend]: https://crates.io/crates/zenblend
[zensim]: https://crates.io/crates/zensim
[fast-ssim2]: https://crates.io/crates/fast-ssim2
[butteraugli]: https://crates.io/crates/butteraugli
[resamplescope-rs]: https://crates.io/crates/resamplescope-rs
[codec-eval]: https://crates.io/crates/codec-eval
[codec-corpus]: https://crates.io/crates/codec-corpus
[zenpixels]: https://crates.io/crates/zenpixels
[zenpixels-convert]: https://crates.io/crates/zenpixels-convert
[linear-srgb]: https://crates.io/crates/linear-srgb
[garb]: https://crates.io/crates/garb
[zenpipe]: https://crates.io/crates/zenpipe
[zencodec]: https://crates.io/crates/zencodec
[zencodecs]: https://crates.io/crates/zencodecs
[zenlayout]: https://crates.io/crates/zenlayout
[zennode]: https://crates.io/crates/zennode
[ImageResizer]: https://imageresizing.net
[Imageflow]: https://github.com/imazen/imageflow
[imageflow-dotnet]: https://www.nuget.org/packages/Imageflow.AllPlatforms
[imageflow-node]: https://www.npmjs.com/package/@imazen/imageflow-node
[imageflow-go]: https://github.com/imazen/imageflow-go
[Imageflow Server]: https://github.com/imazen/imageflow-dotnet-server
[magetypes]: https://crates.io/crates/magetypes
[enough]: https://crates.io/crates/enough
[whereat]: https://crates.io/crates/whereat
[zenbench]: https://crates.io/crates/zenbench
[cargo-copter]: https://crates.io/crates/cargo-copter

## License

MIT OR Apache-2.0
