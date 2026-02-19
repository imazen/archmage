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

## Arm64V2Token

**Features:** NEON + CRC, RDM, DotProd, FP16, AES, SHA2
**CPUs:** Cortex-A55+, Apple M1+, Graviton 2+, Snapdragon 8 Gen 1+
**Trait:** `HasArm64V2`

The broadest modern ARM compute tier. Covers virtually all ARM chips shipping since 2017.

```rust
use archmage::{Arm64V2Token, SimdToken, arcane};

if let Some(token) = Arm64V2Token::summon() {
    process_v2(token, &data);
}

#[arcane]
fn process_v2(token: Arm64V2Token, data: &[f32]) {
    // DotProd (vdotq), FP16 (vcvt), CRC, AES, SHA2 all available
    // RDM (rounding doubling multiply) for fixed-point DSP
}
```

### What V2 adds over baseline NEON

| Feature | Instructions | Use case |
|---------|-------------|----------|
| CRC32 | `__crc32*` | Hash tables, checksums |
| RDM | `vqrdmlah`, `vqrdmlsh` | Fixed-point audio/DSP |
| DotProd | `vdotq_s32`, `vdotq_u32` | Int8 ML inference |
| FP16 | `vcvt_f16_f32`, etc. | Half-precision conversion |
| AES | `vaeseq_u8`, `vaesdq_u8` | Encryption |
| SHA2 | `vsha256hq_u32`, etc. | Hashing |

### Which chips get V2 but not V3

- **Apple M1** — the notable high-end chip stuck at V2 (lacks i8mm/bf16)
- **Cortex-A55** LITTLE cores — still used in budget 2025 phones (lacks fhm/fcma/i8mm/bf16)
- **Graviton 2** (Neoverse N1) — AWS's first custom ARM server chip

## Arm64V3Token

**Features:** All V2 + FHM, FCMA, SHA3, I8MM, BF16
**CPUs:** Cortex-A510+, Apple M2+, Snapdragon X Elite/Plus, Graviton 3+, Cobalt 100
**Trait:** `HasArm64V3`

The full modern feature set. Requires A510+ LITTLE cores.

```rust
use archmage::{Arm64V3Token, SimdToken, arcane};

if let Some(token) = Arm64V3Token::summon() {
    process_v3(token, &data);
}

#[arcane]
fn process_v3(token: Arm64V3Token, data: &[f32]) {
    // I8MM (matrix multiply), BF16, SHA3, FCMA, FHM all available
    // Plus everything from V2
}
```

### What V3 adds over V2

| Feature | Instructions | Use case |
|---------|-------------|----------|
| I8MM | `vmmlaq_s32`, `vusdotq_s32` | Int8 matrix multiply (ML) |
| BF16 | `vbfmmlaq_f32`, `vcvtq_f32_bf16` | BFloat16 ML training/inference |
| SHA3 | `vsha512hq_u64`, etc. | SHA-512, SHA3 hashing |
| FCMA | `vcmlaq_f32` | Complex number multiply-add |
| FHM | `vfmlalq_f32` | FP16-to-FP32 fused multiply (ML) |

### Compute tier hierarchy

```
NeonToken / Arm64          — Baseline NEON (all AArch64)
├── Arm64V2Token           — + CRC, RDM, DotProd, FP16, AES, SHA2
│   └── Arm64V3Token       — + FHM, FCMA, SHA3, I8MM, BF16
├── NeonAesToken           — NEON + AES (crypto leaf)
├── NeonSha3Token          — NEON + SHA3 (crypto leaf)
└── NeonCrcToken           — NEON + CRC (crypto leaf)
```

V3 is a superset of V2, which is a superset of baseline NEON. Pass an `Arm64V3Token` to any function expecting `Arm64V2Token` or `NeonToken`.

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
