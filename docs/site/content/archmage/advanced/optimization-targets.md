+++
title = "Which Tokens to Target"
weight = 1
+++

You have limited time and a codebase to ship. Here's a practical guide to which archmage tokens are worth implementing, in what order, and why.

## The short version

Target these tokens, in this priority order:

1. **X64V3Token** -- AVX2+FMA. Your x86 baseline. Covers every Intel since 2013 and every AMD since 2017.
2. **NeonToken** -- ARM NEON. Covers all 64-bit ARM: Apple Silicon, Graviton, Snapdragon, Ampere.
3. **X64V4Token** / **X64V4xToken** -- AVX-512. Big wins for data-parallel workloads. Zen 4+, Skylake-X+.
4. **Arm64V2Token** -- The broad modern ARM tier. M1+, Graviton 2+, Cortex-A55+. Worth it if you use RDM, DotProd, AES, or SHA2.
5. **Wasm128Token** -- WASM SIMD128. All modern browsers since 2021.

Crypto tokens (`X64CryptoToken`, `X64V3CryptoToken`, `NeonAesToken`, `NeonSha3Token`) are their own story -- target them when you need AES, GHASH, or SHA operations specifically.

Everything gets a `ScalarToken` fallback via `incant!`.

## Why this order

### X64V3Token first

X64V3 (AVX2 + FMA + BMI1/2) is the practical x86 baseline. Every PC sold since roughly 2015 has it. Steam hardware surveys consistently show 95%+ AVX2 support. It gives you 256-bit vectors and fused multiply-add, which is where the bulk of SIMD performance comes from.

Unless you're targeting embedded x86 or very old servers, X64V3 is where you start. You can skip V1 and V2 entirely -- the performance gap between SSE2 and AVX2 is large enough that V2 is rarely worth a separate implementation. Let the scalar fallback handle ancient hardware.

**X64V1Token** (`Sse2Token`) is compile-time guaranteed on x86_64 -- `summon()` always succeeds, and `compiled_with()` always returns `Some(true)`. You still need it if you're writing SSE2-specific code (explicit use of `_mm_*` intrinsics), but for most workloads, go straight to V3.

### NeonToken second

NEON is baseline on all AArch64 processors. In practice, `NeonToken::summon()` succeeds on every 64-bit ARM chip that exists. ARM without NEON is ARMv7 (32-bit) territory.

NEON gives you 128-bit vectors. That's half the width of AVX2, but ARM's register file is deeper (32 x 128-bit registers vs x86's 16 x 256-bit), and NEON instructions have good throughput. For many workloads, a NEON implementation gets you 80% of the AVX2 speedup.

If you use `incant!` with `[v3, neon]`, you cover x86 and ARM with two implementations plus a scalar fallback. That's the sweet spot for most libraries.

### AVX-512 third

`X64V4Token` (AVX-512 F/BW/CD/DQ/VL) and `X64V4xToken` (+ VNNI, VBMI, IFMA, etc.) are worth adding *after* your V3 and NEON implementations work. AVX-512 gives you:

- 512-bit vectors (2x the work of AVX2 per instruction)
- Per-lane masking (eliminates branches in SIMD code)
- Gather/scatter (much faster than AVX2's versions)
- Conflict detection, ternary logic, and other operations with no AVX2 equivalent

V4 vs V4x: V4 is the conservative choice (Skylake-X 2017+, Zen 4 2022+). V4x adds modern extensions that are useful for ML (VNNI), cryptography (GFNI, VAES), and bit manipulation (VBMI, VPOPCNTDQ). Use V4x when those specific instructions help your workload; otherwise V4 is fine.

Require the `avx512` cargo feature in your dependency:

```toml
[dependencies]
archmage = { version = "0.8", features = ["avx512"] }
```

### Arm64V2Token fourth

Arm64V2 adds CRC, RDM (rounding doubling multiply), DotProd, FP16, AES, and SHA2 on top of NEON. This covers Apple M1+, Graviton 2+, and all Cortex-A55+ cores.

Whether you need this depends on your workload. If you're doing general floating-point SIMD, plain NEON is sufficient -- Arm64V2's additions are mostly integer, crypto, and specialized operations. Target Arm64V2 when you specifically benefit from:

- **RDM**: Audio/DSP with fixed-point math
- **DotProd**: ML inference, quantized operations (note: dotprod intrinsics are currently nightly-only in Rust)
- **AES/SHA2**: Cryptography
- **FP16**: Half-precision compute (95 of 210 intrinsics are stable)

**Windows ARM64 caveat**: Windows doesn't expose rdm or fp16 through its runtime detection API (`IsProcessorFeaturePresent`). This means `Arm64V2Token::summon()` returns `None` on Windows ARM64, even on Snapdragon X chips that definitely have these features. If you need NEON on Windows ARM64, use `NeonToken` -- it works everywhere. This is a Windows API limitation, not an archmage bug.

### WASM last (but not least)

WASM SIMD128 has been supported in all major browsers since mid-2021 (Chrome 91, Firefox 89, Safari 16.4). It provides 128-bit vectors -- same width as SSE2 and NEON.

WASM is a special case in several ways:

**No runtime dispatch.** SIMD128 availability is decided at compile time via `-Ctarget-feature=+simd128`. `Wasm128Token::summon()` compiles down to a constant -- there's no CPUID or feature probing.

**No unsafe overhead.** On wasm32, `#[target_feature(enable = "simd128")]` functions are safe. The wasm validation model traps deterministically on unsupported instructions -- there's no undefined behavior from a feature mismatch. As of archmage 0.8.10, `#[arcane]` on wasm32 skips the unsafe wrapper entirely and emits the function directly with `#[target_feature]` + `#[inline]`.

**Compile-time only.** You either build with SIMD128 or you don't. For browser deployment, ship two WASM binaries (SIMD and non-SIMD) and feature-detect in JavaScript, or rely on the scalar fallback.

Build with SIMD128:

```bash
RUSTFLAGS="-Ctarget-feature=+simd128" cargo build --target wasm32-unknown-unknown
```

## Crypto tokens

The crypto tokens don't fit neatly into the "target in order" model because they're orthogonal to compute tiers. You target them when your workload specifically needs:

| Token | What it adds | When to use |
|-------|-------------|-------------|
| `X64CryptoToken` | AES-NI, PCLMULQDQ (128-bit) | AES encrypt/decrypt, CRC32C via CLMUL, GHASH |
| `X64V3CryptoToken` | VAES, VPCLMULQDQ (256-bit) | Same ops but 2x throughput, Zen 3+ / Alder Lake+ |
| `NeonAesToken` | AES rounds + polynomial multiply | ARM AES/GHASH |
| `NeonSha3Token` | SHA3 instructions | ARM SHA3 |
| `NeonCrcToken` | CRC32 instructions | ARM CRC32 |

These are leaf tokens -- use them directly when needed, don't try to tier-fallback through them.

## What about Arm64V3?

`Arm64V3Token` adds FHM, FCMA, SHA3, I8MM, and BF16 on top of Arm64V2. This is Apple M2+, Cortex-A510+, Snapdragon X, Graviton 3+.

Right now, most of these features have zero or nightly-only intrinsics in stable Rust (fhm has none at all, i8mm and fcma are all nightly, bf16 has none). There's not much you can do with Arm64V3 until Rust stabilizes more ARM intrinsics. Check back when that changes.

The notable exception is SHA3 -- if you need it, use `NeonSha3Token` directly.

## The `incant!` pattern

For most libraries, this is all you need:

```rust
use archmage::{arcane, incant};

pub fn process(data: &mut [f32]) {
    incant!(process(data), [v3, neon, wasm128])
}

#[arcane]
fn process_v3(token: X64V3Token, data: &mut [f32]) {
    // AVX2 + FMA implementation
}

#[arcane]
fn process_neon(token: NeonToken, data: &mut [f32]) {
    // NEON implementation
}

#[arcane]
fn process_wasm128(token: Wasm128Token, data: &mut [f32]) {
    // WASM SIMD128 implementation
}

fn process_scalar(_token: ScalarToken, data: &mut [f32]) {
    // Fallback
}
```

Three SIMD implementations + scalar covers every platform. Add `v4` when you have AVX-512 code worth shipping:

```rust
pub fn process(data: &mut [f32]) {
    incant!(process(data), [v4, v3, neon, wasm128])
}
```

## Don't over-tier

Every tier is code you write, test, and maintain. A V2 (SSE4.2) implementation between V3 and scalar is almost never worth it -- the performance gap between scalar and SSE4.2 is smaller than between SSE4.2 and AVX2, and the hardware that has SSE4.2 but not AVX2 is rare.

Similarly, don't write separate V1 and V2 tiers unless you have a specific reason. If your workload benefits from SIMD at all, it benefits from 256-bit vectors (V3) much more than from 128-bit (V1/V2).

The tiers that earn their keep: V3, NEON, V4 (if you use it), scalar. Everything else is diminishing returns.
