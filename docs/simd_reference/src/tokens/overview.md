# Token Overview

Tokens are zero-sized proofs that the CPU supports a specific set of SIMD features. You get a token from `summon()`, and you pass it to functions that need those features. No token, no SIMD — the type system enforces this at compile time.

## The Core API

```rust
use archmage::{X64V3Token, SimdToken};

// Runtime detection — returns Some(token) if CPU has AVX2+FMA
if let Some(token) = X64V3Token::summon() {
    process_simd(token, &mut data);
} else {
    process_scalar(&mut data);
}
```

All tokens implement the `SimdToken` trait:

```rust
pub trait SimdToken: Copy + Clone + Send + Sync + 'static {
    const NAME: &'static str;

    /// Compile-time check: Some(true) if guaranteed, Some(false) if wrong arch, None if unknown
    fn compiled_with() -> Option<bool>;

    /// Runtime detection with atomic caching (~1.3 ns cached, 0 ns when compiled away)
    fn summon() -> Option<Self>;
}
```

## Token Hierarchy

### x86-64

| Token | Aliases | Features | CPUs |
|-------|---------|----------|------|
| `X64V1Token` | `Sse2Token` | SSE, SSE2 (the x86-64 baseline) | All x86-64 CPUs |
| `X64V2Token` | — | + SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT, CMPXCHG16B | Nehalem 2008+, Bulldozer 2011+ |
| `X64CryptoToken` | — | V2 + PCLMULQDQ, AES | Westmere 2010+, Bulldozer 2011+ |
| `X64V3Token` | `Desktop64` | + AVX, AVX2, FMA, BMI1, BMI2, F16C, LZCNT, MOVBE | Haswell 2013+, Zen 1 2017+ |
| `X64V3CryptoToken` | — | V3 + PCLMULQDQ, AES, VPCLMULQDQ, VAES | Zen 3+ 2020, Alder Lake 2021+ |
| `X64V3GfniCryptoToken` | — | V3 Crypto + GFNI | Alder/Raptor/Meteor/Arrow/Lunar Lake, Sierra Forest, Zen 4+ |
| `X64V4Token` | `Server64`, `Avx512Token` | V3 + PCLMULQDQ, AES, AVX-512 F/BW/CD/DQ/VL | Skylake-X 2017+, Zen 4 2022+ |
| `X64V4xToken` | `Avx512ModernToken` | + VPOPCNTDQ, IFMA, VBMI, VNNI, VBMI2, BITALG, VPCLMULQDQ, GFNI, VAES | Ice Lake 2019+, Zen 4 2022+ |
| `Avx512Fp16Token` | — | V4 + AVX-512 FP16 | Sapphire Rapids 2023+ |

Each higher tier is a superset. An `X64V4Token` converts to the lower tokens with `.v3()` or `.v2()`, and satisfies bounds such as `impl HasX64V2` (both free).

**The `avx512` cargo feature** is needed for `import_intrinsics` with the AVX-512 tokens (their safe memory operations) and for magetypes' native 512-bit backends. The tokens themselves, `summon()`, and `#[attune(wrap)]` without `import_intrinsics` work without it.

### AArch64

| Token | Aliases | Features | CPUs |
|-------|---------|----------|------|
| `NeonToken` | `Arm64` | NEON (128-bit SIMD) | All 64-bit ARM |
| `Arm64V2Token` | — | + CRC, RDM, DotProd, FP16, AES, SHA2 | A55+, M1+, Graviton 2+ |
| `Arm64V3Token` | — | + FHM, FCMA, SHA3, I8MM, BF16 | A510+, M2+, Snapdragon X, Graviton 3+ |
| `NeonAesToken` | — | NEON + AES | Most ARMv8 with crypto |
| `NeonSha3Token` | — | NEON + SHA3 | ARMv8.2+ with SHA3 |
| `NeonCrcToken` | — | NEON + CRC | Most ARMv8 |

NEON is baseline on AArch64 — `NeonToken::summon()` always succeeds. Arm64V2/V3 are compute tiers (archmage-defined, not ARM architecture versions). The crypto tokens (AES, SHA3, CRC) are independent leaf tokens for single-feature checks.

### WASM

| Token | Features | Notes |
|-------|----------|-------|
| `Wasm128Token` | SIMD128 | Compile with `-Ctarget-feature=+simd128` |
| `Wasm128RelaxedToken` | SIMD128, relaxed SIMD | Compile with `-Ctarget-feature=+simd128,+relaxed-simd` |

### Universal

| Token | Features | Notes |
|-------|----------|-------|
| `ScalarToken` | None | Always available, used as the fallback by `attuned!` and `#[attune(dispatch)]` |

## Detection Behavior

### `summon()` — Runtime Detection

```rust
// ~1.3 ns (cached via AtomicU8)
if let Some(token) = X64V3Token::summon() { ... }
```

Each token has a static `AtomicU8` cache: 0 = unknown, 1 = unavailable, 2 = available. First call does the CPUID check and caches the result. Subsequent calls read the atomic.

### `compiled_with()` — Compile-Time Check

```rust
match X64V3Token::compiled_with() {
    Some(true)  => { /* compiled with -Ctarget-cpu=haswell, summon() is a no-op */ }
    Some(false) => { /* wrong architecture — token can never exist */ }
    None        => { /* need runtime check */ }
}
```

### When Detection Compiles Away

| Build Flags | Effect |
|------------|--------|
| `-Ctarget-cpu=haswell` | `X64V3Token::summon()` → always `Some`, zero-cost |
| `-Ctarget-cpu=skylake-avx512` | `Server64::summon()` → always `Some`, zero-cost |
| `-Ctarget-cpu=native` | All available tokens compile away |
| Default | Runtime CPUID check, cached |

## Zero-Sized Types

All tokens are zero-sized. Passing them has no runtime cost:

```rust
assert_eq!(std::mem::size_of::<X64V3Token>(), 0);
assert_eq!(std::mem::size_of::<NeonToken>(), 0);
assert_eq!(std::mem::size_of::<ScalarToken>(), 0);
```

## Cross-Architecture Behavior

Every token type exists on every architecture. On unsupported architectures, `summon()` returns `None` and `compiled_with()` returns `Some(false)`. `#[attune]` omits inapplicable target-architecture functions via `#[cfg(target_arch = ...)]` rather than generating `unreachable!()` stubs.

`attuned!` and `#[attune(dispatch)]` guard candidate calls automatically. Direct calls to a single-architecture function from ungated code need a `#[cfg(target_arch)]` guard:

```rust
use archmage::prelude::*;
use magetypes::simd::generic::f32x8;

// Emitted only on x86_64; omitted on ARM and WASM
#[attune(wrap)]
fn x86_kernel(token: X64V3Token, a: &[f32; 8], b: &[f32; 8], c: &[f32; 8]) -> [f32; 8] {
    let va = f32x8::load_t(token, a);
    let vb = f32x8::load_t(token, b);
    let vc = f32x8::load_t(token, c);
    va.mul_add_relaxed(vb, vc).to_array()
}

#[cfg(target_arch = "x86_64")]
if let Some(token) = X64V3Token::summon() {
    let out = x86_kernel(token, &a, &b, &c);
}
```

## Tier Traits

Tier traits provide generic bounds across token families:

```rust
fn needs_v2(token: impl HasX64V2) { ... }       // X64V2Token, X64V3Token, X64V4Token, ...
fn needs_v4(token: impl HasX64V4) { ... }       // X64V4Token, X64V4xToken, ...
fn needs_neon(token: impl HasNeon) { ... }      // NeonToken, Arm64V2Token, Arm64V3Token, ...
fn needs_arm_v2(token: impl HasArm64V2) { ... } // Arm64V2Token, Arm64V3Token
fn needs_arm_v3(token: impl HasArm64V3) { ... } // Arm64V3Token
```

For x86 V3 (the recommended baseline), use `X64V3Token` directly — no trait needed.

**Note:** On an `#[attune(wrap)]` boundary, a tier-trait bound enables only that bound's features (for example, `impl HasX64V2` enables V2 even when passed an `X64V4Token`). Use concrete tokens at entry points for the tier you want. See [Safety Model](../patterns/safety.md) for details.

## Disabling Tokens

For testing, tokens can be disabled process-wide:

```rust
// Requires `testable_dispatch` feature
X64V3Token::dangerously_disable_token_process_wide();
assert!(X64V3Token::summon().is_none());

// Re-enable
X64V3Token::dangerously_enable_token_process_wide();
```

This is for testing fallback paths. Don't use it in production.
