# archmage

[![CI](https://github.com/imazen/archmage/actions/workflows/ci.yml/badge.svg)](https://github.com/imazen/archmage/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/archmage.svg)](https://crates.io/crates/archmage)
[![docs.rs](https://docs.rs/archmage/badge.svg)](https://docs.rs/archmage)
[![codecov](https://codecov.io/gh/imazen/archmage/graph/badge.svg)](https://codecov.io/gh/imazen/archmage)
[![MSRV](https://img.shields.io/crates/msrv/archmage.svg)](https://crates.io/crates/archmage)
[![License](https://img.shields.io/crates/l/archmage.svg)](https://github.com/imazen/archmage#license)

> Safely invoke your intrinsic power, using the tokens granted to you by the CPU.

```toml
[dependencies]
archmage = "0.4"
```

```rust
use archmage::prelude::*;

#[arcane]
fn dot_product(_token: Desktop64, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    let va = safe_simd::_mm256_loadu_ps(a);
    let vb = safe_simd::_mm256_loadu_ps(b);
    let mul = _mm256_mul_ps(va, vb);
    let mut tmp = [0.0f32; 8];
    safe_simd::_mm256_storeu_ps(&mut tmp, mul);
    tmp.iter().sum()
}

fn main() {
    if let Some(token) = Desktop64::summon() {
        println!("{}", dot_product(token, &[1.0; 8], &[2.0; 8]));
    }
}
```

`summon()` checks CPUID at runtime. `#[arcane]` enables `#[target_feature]` on the inner function, making value-based intrinsics safe (Rust 1.85+). Loads and stores go through `safe_simd` (reference-based, no raw pointers). If you compile with `-C target-cpu=haswell`, the runtime check is elided entirely.

The prelude re-exports tokens, traits, macros, `core::arch::*` intrinsics, and `safe_unaligned_simd` as `safe_simd` — all for the current platform.

## Tokens

All tokens compile on all platforms; `summon()` returns `None` on unsupported architectures.

| Token | Alias | Features | CPUs |
|-------|-------|----------|------|
| `X64V2Token` | | SSE4.2, POPCNT | Nehalem 2008+ |
| `X64V3Token` | `Desktop64` | AVX2, FMA, BMI2 | Haswell 2013+, Zen 1+ |
| `X64V4Token` | `Server64` | AVX-512 F/BW/CD/DQ/VL | Skylake-X 2017+, Zen 4+ |
| `Avx512ModernToken` | | + VBMI2, VNNI, BF16 | Ice Lake 2019+, Zen 4+ |
| `Avx512Fp16Token` | | + FP16 | Sapphire Rapids 2023+ |
| `NeonToken` | `Arm64` | NEON | All AArch64 |
| `NeonAesToken` | | + AES | ARMv8 with crypto |
| `NeonSha3Token` | | + SHA3 | ARMv8.2+ |
| `NeonCrcToken` | | + CRC | Most ARMv8 |
| `Simd128Token` | | WASM SIMD | |
| `ScalarToken` | | (none) | Always available |

AVX-512 tokens require the `avx512` feature. Higher tokens imply lower ones (`v4.v3()`, `v4.v2()`).

## Traits

| Trait | Meaning |
|-------|---------|
| `SimdToken` | All tokens |
| `IntoConcreteToken` | Compile-time dispatch via monomorphization |
| `HasX64V2` | SSE4.2+ |
| `HasX64V4` | AVX-512 (requires `avx512`) |
| `Has128BitSimd` / `Has256BitSimd` / `Has512BitSimd` | Vector width |
| `HasNeon` / `HasNeonAes` / `HasNeonSha3` | ARM tiers |

## Feature flags

| Feature | Default | |
|---------|---------|---|
| `std` | yes | Standard library support |
| `macros` | yes | `#[arcane]`, `#[magetypes]`, `incant!` |
| `safe-simd` | yes | Re-exports `safe_unaligned_simd` via prelude |
| `avx512` | no | AVX-512 tokens |

## License

MIT OR Apache-2.0

## AI-Generated Code Notice

Developed with Claude (Anthropic). Review critical paths before production use.
