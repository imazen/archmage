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

**The `avx512` cargo feature** is needed for `import_intrinsics` with the AVX-512 tokens (their safe memory operations) and for magetypes' native 512-bit backends. The tokens themselves, `summon()`, and `#[arcane]` without `import_intrinsics` work without it.

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
| `ScalarToken` | None | Always available, used by `incant!` fallback |

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

## Cross-Architecture Stubs

Every token type compiles on every architecture. On the wrong arch, `summon()` returns `None` and `#[arcane]` functions generate `unreachable!()` stubs:

```rust
// This compiles on ARM — it just can't be called
#[arcane(import_intrinsics)]
fn x86_kernel(token: X64V3Token, data: &[f32; 8]) -> f32 { ... }

// On ARM: summon() returns None, kernel is never reached
if let Some(token) = X64V3Token::summon() {
    x86_kernel(token, &data);
}
```

## Tier Traits

Tier traits provide generic bounds across token families:

```rust
fn needs_v2(token: impl HasX64V2) { ... }     // X64V2Token, X64V3Token, X64V4Token, ...
fn needs_v4(token: impl HasX64V4) { ... }     // X64V4Token, X64V4xToken, ...
fn needs_neon(token: impl HasNeon) { ... }     // NeonToken, Arm64V2Token, Arm64V3Token, ...
fn needs_arm_v2(token: impl HasArm64V2) { ... } // Arm64V2Token, Arm64V3Token
fn needs_arm_v3(token: impl HasArm64V3) { ... } // Arm64V3Token
```

For x86 V3 (the recommended baseline), use `X64V3Token` directly — no trait needed.

**Warning:** Generic bounds create LLVM optimization barriers. Use concrete tokens for hot paths. See [Safety Model](../patterns/safety.md) for details.

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
