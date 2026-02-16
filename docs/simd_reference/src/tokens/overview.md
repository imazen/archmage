# Token Overview

Tokens are zero-sized proofs that the CPU supports a specific set of SIMD features. You get a token from `summon()`, and you pass it to functions that need those features. No token, no SIMD — the type system enforces this at compile time.

## The Core API

```rust
use archmage::{Desktop64, SimdToken};

// Runtime detection — returns Some(token) if CPU has AVX2+FMA
if let Some(token) = Desktop64::summon() {
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
| `X64V2Token` | — | SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT | Nehalem 2008+, Bulldozer 2011+ |
| `X64V3Token` | `Desktop64`, `Avx2FmaToken` | + AVX, AVX2, FMA, BMI1, BMI2, F16C, MOVBE | Haswell 2013+, Zen 1 2017+ |
| `X64V4Token` | `Server64`, `Avx512Token` | + AVX-512 F/BW/CD/DQ/VL | Skylake-X 2017+, Zen 4 2022+ |
| `X64V4xToken` | — | + VPOPCNTDQ, IFMA, VBMI, VNNI, BF16, VBMI2, BITALG, VPCLMULQDQ, GFNI, VAES | Ice Lake 2019+, Zen 4 2022+ |
| `Avx512Fp16Token` | — | AVX-512 FP16 | Sapphire Rapids 2023+ |

Each higher tier is a superset. If you have `X64V4Token`, you can pass it to any function expecting `X64V3Token` or `X64V2Token` (downcast is free).

**Requires `avx512` feature:** `X64V4Token`, `X64V4xToken`, `Avx512Fp16Token`.

### AArch64

| Token | Aliases | Features | CPUs |
|-------|---------|----------|------|
| `NeonToken` | `Arm64` | NEON (128-bit SIMD) | All 64-bit ARM |
| `NeonAesToken` | — | + AES | Most ARMv8 with crypto |
| `NeonSha3Token` | — | + SHA3 | ARMv8.2+ with SHA3 |
| `NeonCrcToken` | — | + CRC | Most ARMv8 |

NEON is baseline on AArch64 — `NeonToken::summon()` always succeeds.

### WASM

| Token | Features | Notes |
|-------|----------|-------|
| `Wasm128Token` | SIMD128 | Compile with `-Ctarget-feature=+simd128` |

### Universal

| Token | Features | Notes |
|-------|----------|-------|
| `ScalarToken` | None | Always available, used by `incant!` fallback |

## Detection Behavior

### `summon()` — Runtime Detection

```rust
// ~1.3 ns (cached via AtomicU8)
if let Some(token) = Desktop64::summon() { ... }
```

Each token has a static `AtomicU8` cache: 0 = unknown, 1 = unavailable, 2 = available. First call does the CPUID check and caches the result. Subsequent calls read the atomic.

### `compiled_with()` — Compile-Time Check

```rust
match Desktop64::compiled_with() {
    Some(true)  => { /* compiled with -Ctarget-cpu=haswell, summon() is a no-op */ }
    Some(false) => { /* wrong architecture — token can never exist */ }
    None        => { /* need runtime check */ }
}
```

### When Detection Compiles Away

| Build Flags | Effect |
|------------|--------|
| `-Ctarget-cpu=haswell` | `Desktop64::summon()` → always `Some`, zero-cost |
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
#[arcane]
fn x86_kernel(token: X64V3Token, data: &[f32; 8]) -> f32 { ... }

// On ARM: summon() returns None, kernel is never reached
if let Some(token) = X64V3Token::summon() {
    x86_kernel(token, &data);
}
```

## Tier Traits

Two tier traits exist for generic bounds:

```rust
fn needs_v2(token: impl HasX64V2) { ... }  // X64V2Token, X64V3Token, X64V4Token, ...
fn needs_v4(token: impl HasX64V4) { ... }  // X64V4Token, X64V4xToken, ...
fn needs_neon(token: impl HasNeon) { ... } // NeonToken, NeonAesToken, ...
```

For V3 (the recommended baseline), use `X64V3Token` directly — no trait needed.

**Warning:** Generic bounds create LLVM optimization barriers. Use concrete tokens for hot paths. See [Safety Model](../patterns/safety.md) for details.

## Disabling Tokens

For testing, tokens can be disabled process-wide:

```rust
// Requires `testable_dispatch` feature
Desktop64::dangerously_disable_token_process_wide();
assert!(Desktop64::summon().is_none());

// Re-enable
Desktop64::dangerously_enable_token_process_wide();
```

This is for testing fallback paths. Don't use it in production.
