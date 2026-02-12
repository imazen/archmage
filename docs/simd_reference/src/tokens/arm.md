# AArch64 Tokens

## NeonToken / Arm64

**Features:** NEON (Advanced SIMD)
**CPUs:** All 64-bit ARM processors
**Register width:** 128-bit, 32 registers (V0-V31)

```rust
use archmage::{Arm64, SimdToken, arcane};

// Always succeeds on AArch64 — NEON is baseline
if let Some(token) = Arm64::summon() {
    process_neon(token, &data);
}
```

**Alias:** `Arm64` (preferred)

NEON is always available on AArch64. Unlike x86 where you need to check for AVX2, `Arm64::summon()` always returns `Some` on 64-bit ARM. It still requires runtime detection because the same binary might run on x86 (where it returns `None`).

### Key differences from x86 SSE

| Aspect | x86 SSE | ARM NEON |
|--------|---------|----------|
| Registers | 16 (XMM0-15) | 32 (V0-V31) |
| FMA | Requires separate FMA flag | Always available |
| Integer types | One type (`__m128i`) | Separate per signedness and width |
| Float bitwise | Direct (`_mm_and_ps`) | Cast to int, operate, cast back |
| Reciprocal estimate | ~12 bits | ~8 bits (needs extra Newton-Raphson) |

### FMA is always available

On x86, FMA requires V3 (Haswell+). On ARM, FMA is part of baseline NEON:

```rust
#[arcane]
fn fma_neon(token: Arm64, a: f32x4, b: f32x4, c: f32x4) -> f32x4 {
    a.mul_add(b, c)  // Uses native vfmaq_f32
}
```

### 256-bit on ARM

NEON only has 128-bit registers. For `f32x8` and wider types, magetypes uses polyfills — two 128-bit operations per 256-bit operation:

```rust
use magetypes::simd::f32x8;

#[arcane]
fn process(token: Arm64, data: &[f32; 8]) -> f32 {
    let v = f32x8::load(token, data);  // 2× vld1q_f32 internally
    v.reduce_add()                      // 2× horizontal sum + scalar add
}
```

## NeonAesToken

**Features:** NEON + AES
**CPUs:** Most ARMv8 with cryptography extensions

```rust
if let Some(token) = NeonAesToken::summon() {
    // AES-NI equivalent: hardware AES rounds
}
```

Available on most ARM server and mobile SoCs with crypto extensions. Use for AES encryption/decryption and GCM modes.

## NeonSha3Token

**Features:** NEON + SHA3
**CPUs:** ARMv8.2+ with SHA3 extension

```rust
if let Some(token) = NeonSha3Token::summon() {
    // SHA3/Keccak acceleration
}
```

Less widely available than AES. Check with `summon()`.

## NeonCrcToken

**Features:** NEON + CRC
**CPUs:** Most ARMv8 processors

```rust
if let Some(token) = NeonCrcToken::summon() {
    // CRC32 hardware acceleration
}
```

## SVE/SVE2: Not Supported

Scalable Vector Extension (SVE/SVE2) is **not supported** in archmage:

- Variable vector length (128-2048 bits) complicates the fixed-size type model
- Rust stable doesn't support SVE intrinsics
- Only available on a few server chips (Graviton 3+, Fujitsu A64FX, Neoverse)
- No consumer hardware ships SVE

When SVE ships widely and Rust adds stable support, archmage will add tokens. Until then, use NEON.

## NEON Intrinsic Naming

NEON uses a different naming scheme than x86:

```
v[operation][modifier]q_[type]
```

| Part | Meaning | Example |
|------|---------|---------|
| `v` | Vector prefix | Always present |
| operation | `add`, `mul`, `min`, etc. | `vadd` |
| modifier | `l` (long), `n` (narrow), `p` (pairwise), `a` (accumulate) | `vaddl` (widening add) |
| `q` | Quadword (128-bit) | Always present for our types |
| type | `f32`, `s32`, `u8`, etc. | `vaddq_f32` |

See [Naming Conventions](../intrinsics/naming.md) for the full mapping.
