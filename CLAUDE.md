# archmage

> Safely invoke your intrinsic power, using the tokens granted to you by the CPU. Cast primitive magics faster than any mage alive.

## CRITICAL: Token/Trait Design (DO NOT MODIFY)

### LLVM x86-64 Microarchitecture Levels

| Level | Features | Token | Trait |
|-------|----------|-------|-------|
| **v1** | SSE, SSE2 (baseline) | None | None (always available) |
| **v2** | + SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT | `X64V2Token` | `HasX64V2` |
| **v3** | + AVX, AVX2, FMA, BMI1, BMI2, F16C | `X64V3Token` / `Desktop64` / `Avx2FmaToken` | Use token directly |
| **v4** | + AVX512F, AVX512BW, AVX512CD, AVX512DQ, AVX512VL | `X64V4Token` / `Avx512Token` | `HasX64V4` |
| **Modern** | + VPOPCNTDQ, IFMA, VBMI, VNNI, BF16, VBMI2, BITALG, VPCLMULQDQ, GFNI, VAES | `Avx512ModernToken` | Use token directly |
| **FP16** | AVX512FP16 (independent) | `Avx512Fp16Token` | Use token directly |

### AArch64 Tokens

| Token | Features | Trait |
|-------|----------|-------|
| `NeonToken` / `Arm64` | neon + fp16 (always together) | `HasNeon` (baseline) |
| `NeonAesToken` | + aes | `HasNeonAes` |
| `NeonSha3Token` | + sha3 | `HasNeonSha3` |
| `ArmCryptoToken` | aes + sha2 + crc | Use token directly |
| `ArmCrypto3Token` | + sha3 | Use token directly |

**PROHIBITED:** NO SVE/SVE2 - hasn't shipped in consumer hardware.

### Rules

1. **NO granular x86 traits** - No `HasSse`, `HasSse2`, `HasAvx`, `HasAvx2`, `HasFma`, `HasAvx512f`, `HasAvx512bw`, etc.
2. **Use tier tokens** - `X64V2Token`, `Avx2FmaToken`, `X64V4Token`, `Avx512ModernToken`
3. **Single trait per tier** - `HasX64V2`, `HasX64V4` only
4. **NEON includes fp16** - They always appear together on AArch64
5. **NO SVE** - `SveToken`, `Sve2Token`, `HasSve`, `HasSve2` are PROHIBITED

---

## CRITICAL: Documentation Examples

### Always prefer `#[arcane]` over manual `#[target_feature]`

**DO NOT write examples with manual `#[target_feature]` + unsafe wrappers.** The `#[arcane]` macro does this automatically and is the correct pattern for archmage.

```rust
// WRONG - manual #[target_feature] wrapping
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn process_inner(data: &[f32]) -> f32 { ... }

#[cfg(target_arch = "x86_64")]
fn process(token: Avx2FmaToken, data: &[f32]) -> f32 {
    unsafe { process_inner(data) }
}

// CORRECT - use #[arcane] (it generates the above automatically)
#[cfg(target_arch = "x86_64")]
#[arcane]
fn process(token: Avx2FmaToken, data: &[f32]) -> f32 {
    // This function body is compiled with #[target_feature(enable = "avx2,fma")]
    // Intrinsics and operators inline properly into single SIMD instructions
    ...
}
```

### Use `safe_unaligned_simd` inside `#[arcane]` functions

**Use `safe_unaligned_simd` directly inside `#[arcane]` functions.** The calls are safe because the target features match.

```rust
// WRONG - raw pointers need unsafe
let v = unsafe { _mm256_loadu_ps(data.as_ptr()) };

// CORRECT - use safe_unaligned_simd (safe inside #[arcane])
let v = safe_unaligned_simd::x86_64::_mm256_loadu_ps(data);
```

## Quick Start

```bash
cargo test                    # Run tests
cargo test --all-features     # Test with all integrations
cargo clippy --all-features   # Lint
```

## Core Insight: Rust 1.85+ Changed Everything

As of Rust 1.85, **value-based intrinsics are safe inside `#[target_feature]` functions**:

```rust
#[target_feature(enable = "avx2")]
unsafe fn example() {
    let a = _mm256_setzero_ps();           // SAFE!
    let b = _mm256_add_ps(a, a);           // SAFE!
    let c = _mm256_fmadd_ps(a, a, a);      // SAFE!

    // Only memory ops remain unsafe (raw pointers)
    let v = unsafe { _mm256_loadu_ps(ptr) };  // Still needs unsafe
}
```

This means we **don't need to wrap** arithmetic, shuffle, compare, bitwise, or other value-based intrinsics. Only:
1. **Tokens** - Prove CPU features are available
2. **`#[arcane]` macro** - Enable `#[target_feature]` via token proof
3. **`safe_unaligned_simd`** - Reference-based memory operations (user adds as dependency)

## How `#[arcane]` Works

The macro generates an inner function with `#[target_feature]`:

```rust
// You write:
#[arcane]
fn my_kernel(token: Avx2FmaToken, data: &[f32; 8]) -> [f32; 8] {
    let v = _mm256_setzero_ps();  // Safe!
    // ...
}

// Macro generates:
fn my_kernel(token: Avx2FmaToken, data: &[f32; 8]) -> [f32; 8] {
    #[target_feature(enable = "avx2,fma")]
    unsafe fn inner(data: &[f32; 8]) -> [f32; 8] {
        let v = _mm256_setzero_ps();  // Safe inside #[target_feature]!
        // ...
    }
    // SAFETY: Token proves CPU support was verified via try_new()
    unsafe { inner(data) }
}
```

## Friendly Aliases

| Alias | Token | What it means |
|-------|-------|---------------|
| `Desktop64` | `X64V3Token` | AVX2 + FMA (Haswell 2013+, Zen 1+) |
| `Server64` | `X64V4Token` | + AVX-512 (Xeon 2017+, Zen 4+) |
| `Arm64` | `NeonToken` | NEON + FP16 (all 64-bit ARM) |

```rust
use archmage::{Desktop64, SimdToken, arcane};

#[arcane]
fn process(token: Desktop64, data: &mut [f32; 8]) {
    // AVX2 + FMA intrinsics safe here
}

if let Some(token) = Desktop64::summon() {
    process(token, &mut data);
}
```

## Directory Structure

```
src/
├── lib.rs              # Main exports
├── tokens/
│   ├── mod.rs          # SimdToken trait, tier traits (HasX64V2, HasX64V4)
│   ├── x86.rs          # x86 token types
│   ├── arm.rs          # ARM token types
│   └── wasm.rs         # WASM token types
├── composite/          # Higher-level operations (__composite feature)
├── integrate/          # wide crate integration (__wide feature)
└── simd/               # Auto-generated SIMD types (wide-like ergonomics)
xtask/
└── src/main.rs         # SIMD type generator
```

## Token Hierarchy

**x86:**
- `X64V2Token` - SSE4.2 + POPCNT (Nehalem 2008+)
- `X64V3Token` / `Desktop64` / `Avx2FmaToken` - AVX2 + FMA + BMI2 (Haswell 2013+, Zen 1+)
- `X64V4Token` / `Avx512Token` - + AVX-512 F/BW/CD/DQ/VL (Skylake-X 2017+, Zen 4+)
- `Avx512ModernToken` - + modern extensions (Ice Lake 2019+, Zen 4+)
- `Avx512Fp16Token` - + FP16 (Sapphire Rapids 2023+)

**ARM:**
- `NeonToken` / `Arm64` - NEON + FP16 (baseline)
- `NeonAesToken` - + AES
- `NeonSha3Token` - + SHA3
- `ArmCryptoToken` - AES + SHA2 + CRC
- `ArmCrypto3Token` - + SHA3

## Tier Traits

Only two tier traits exist for generic bounds:

```rust
fn requires_v2(token: impl HasX64V2) { ... }
fn requires_v4(token: impl HasX64V4) { ... }
fn requires_neon(token: impl HasNeon) { ... }
```

For v3 (AVX2+FMA), use `Avx2FmaToken` directly - it's the recommended baseline.

## Safe Memory Operations

Use `safe_unaligned_simd` directly inside `#[arcane]` functions:

```rust
use archmage::{Desktop64, SimdToken, arcane};

#[arcane]
fn process(_token: Desktop64, data: &[f32; 8]) -> [f32; 8] {
    // safe_unaligned_simd calls are SAFE inside #[arcane]
    let v = safe_unaligned_simd::x86_64::_mm256_loadu_ps(data);
    let squared = _mm256_mul_ps(v, v);
    let mut out = [0.0f32; 8];
    safe_unaligned_simd::x86_64::_mm256_storeu_ps(&mut out, squared);
    out
}
```

## License

MIT OR Apache-2.0
