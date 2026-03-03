# Token Support by CPU

Real-world token availability measured on GitHub Actions runners and local hardware.
Data collected via `cargo run --example cpu_survey --features "avx512"`.

## Quick Reference

### x86_64

| CPU | Highest Token | V1 | V2 | Crypto | V3 | V3Crypto | V4 | V4x | FP16 |
|-----|--------------|:--:|:--:|:------:|:--:|:--------:|:--:|:---:|:----:|
| Intel Core i7-8700B (Coffee Lake, 2018) | V3 | + | + | + | + | | | | |
| AMD EPYC 7763 (Zen 3 Milan, 2021) | V3Crypto | + | + | + | + | + | | | |
| AMD Ryzen 9 7950X (Zen 4, 2022) | V4x | + | + | + | + | + | + | + | |
| Intel Xeon Platinum 8370C (Ice Lake, 2021) | V4x | + | + | + | + | + | + | + | |
| Intel Xeon w9-3595X (Sapphire Rapids, 2023) | FP16 | + | + | + | + | + | + | + | + |

### AArch64

| CPU | Highest Token | Neon | AES | SHA3 | CRC | V2 | V3 |
|-----|--------------|:----:|:---:|:----:|:---:|:--:|:--:|
| Apple M1 (2020) | V2 | + | + | + | + | + | |
| Apple M2+ (2022+) | V3 | + | + | + | + | + | + |
| AWS Graviton 2 (Neoverse N1, 2020) | V2 | + | + | | + | + | |
| AWS Graviton 3+ (Neoverse V1, 2022) | V3 | + | + | + | + | + | + |
| Azure Cobalt 100 (Neoverse N2, 2024) | V3 | + | + | + | + | + | + |
| Qualcomm Snapdragon X (Oryon, 2024) | V3 | + | + | + | + | + | + |
| Azure Cobalt 100 on **Windows** | **Neon** | + | + | | + | | |

## Detailed Feature Tables

### x86_64 Features

Features grouped by archmage token tier. A `+` means detected at runtime.

| Feature | Description | Coffee Lake i7-8700B | Zen 3 EPYC 7763 | Zen 4 Ryzen 7950X | Ice Lake Xeon 8370C |
|---------|-------------|:--------------------:|:----------------:|:------------------:|:-------------------:|
| | | macOS Intel | Linux/Windows | WSL2 | Linux |
| **V1 (baseline)** | | | | | |
| sse | Streaming SIMD Extensions | + | + | + | + |
| sse2 | 128-bit integer/double SIMD | + | + | + | + |
| **V2 (Nehalem 2008+)** | | | | | |
| sse3 | Horizontal add, complex math | + | + | + | + |
| ssse3 | Byte shuffle, absolute value | + | + | + | + |
| sse4.1 | Blend, dot product, round | + | + | + | + |
| sse4.2 | String compare, CRC32 | + | + | + | + |
| popcnt | Population count | + | + | + | + |
| cmpxchg16b | 128-bit compare-and-swap | + | + | + | + |
| **Crypto** | | | | | |
| aes | AES-NI (128-bit AES rounds) | + | + | + | + |
| pclmulqdq | Carry-less multiply (GF(2)) | + | + | + | + |
| **V3 (Haswell 2013+)** | | | | | |
| avx | 256-bit float SIMD | + | + | + | + |
| avx2 | 256-bit integer SIMD | + | + | + | + |
| fma | Fused multiply-add | + | + | + | + |
| bmi1 | Bit Manipulation Instructions 1 | + | + | + | + |
| bmi2 | PDEP/PEXT bit deposit/extract | + | + | + | + |
| f16c | Half-precision float conversion | + | + | + | + |
| lzcnt | Leading zero count | + | + | + | + |
| movbe | Move data after byte swap | + | + | + | + |
| **V3Crypto** | | | | | |
| vaes | Vectorized AES (256/512-bit) | | + | + | + |
| vpclmulqdq | Vectorized CLMUL (256/512-bit) | | + | + | + |
| **V4 (Skylake-X 2017+)** | | | | | |
| avx512f | AVX-512 Foundation | | | + | + |
| avx512bw | Byte/Word operations | | | + | + |
| avx512cd | Conflict Detection | | | + | + |
| avx512dq | Doubleword/Quadword | | | + | + |
| avx512vl | Vector Length (128/256-bit) | | | + | + |
| **V4x (Ice Lake 2019+)** | | | | | |
| avx512vpopcntdq | Vector POPCNT DW/QW | | | + | + |
| avx512ifma | Integer FMA (52-bit) | | | + | + |
| avx512vbmi | Vector Byte Manipulation | | | + | + |
| avx512vbmi2 | Compress/expand byte | | | + | + |
| avx512bitalg | Bit Algorithms (POPCNT byte) | | | + | + |
| avx512vnni | Vector Neural Network (INT8 dot) | | | + | + |
| **FP16 (Sapphire Rapids 2023+)** | | | | | |
| avx512fp16 | Native Float16 arithmetic | | | | |
| **Other AVX-512** | | | | | |
| avx512bf16 | BFloat16 conversion/dot | | | + | |
| avx512vp2intersect | VP2INTERSECT | | | | |
| **Crypto (beyond V2)** | | | | | |
| sha | SHA-1/SHA-256 acceleration | | + | + | + |
| sha512 | SHA-512 acceleration | | | | |
| gfni | Galois Field arithmetic | | | + | + |
| sm3 | ShangMi 3 hash | | | | |
| sm4 | ShangMi 4 cipher | | | | |
| kl | Key Locker | | | | |
| widekl | Wide Key Locker | | | | |
| **VEX ML (Alder Lake+)** | | | | | |
| avxvnni | AVX-VNNI (INT8 dot, VEX) | | | | |
| avxifma | AVX-IFMA (52-bit FMA, VEX) | | | | |
| avxneconvert | BF16/FP16 no-except convert | | | | |
| avxvnniint8 | Signed INT8 dot product | | | | |
| avxvnniint16 | INT16 dot product | | | | |
| **Misc** | | | | | |
| adx | Multi-precision add-carry | + | + | + | + |
| rdrand | Hardware RNG | + | + | + | + |
| rdseed | Hardware random seed | + | + | + | + |
| sse4a | AMD EXTRQ/INSERTQ | | + | + | |
| tbm | Trailing Bit Manipulation | | | | |

### AArch64 Features

| Feature | Description | Apple M1 (macOS) | Cobalt 100 (Linux) | Cobalt 100 (Windows) |
|---------|-------------|:----------------:|:-------------------:|:--------------------:|
| **Neon (baseline)** | | | | |
| neon | 128-bit SIMD | + | + | + |
| **CRC** | | | | |
| crc | CRC32 instructions | + | + | + |
| **AES** | | | | |
| aes | AES encrypt/decrypt | + | + | + |
| sha2 | SHA-1/SHA-256 | + | + | + |
| **SHA3** | | | | |
| sha3 | SHA-3/SHA-512 (EOR3/RAX1) | + | + | |
| **V2 features** | | | | |
| rdm | Rounding Double Multiply | + | + | |
| dotprod | Integer dot product | + | + | + |
| fp16 | Half-precision arithmetic | + | + | |
| **V3 features** | | | | |
| fhm | FP16 multiply-accum to FP32 | + | + | |
| fcma | Complex multiply-add | + | + | |
| bf16 | BFloat16 (BFDOT/BFMMLA) | | + | |
| i8mm | Int8 matrix multiply | | + | |
| **Other compute** | | | | |
| jsconv | JavaScript FJCVTZS | + | + | + |
| frintts | FRINT32Z/FRINT64Z rounding | + | + | |
| sm4 | ShangMi 4 cipher | | + | |
| **Atomics / memory** | | | | |
| lse | Large System Extensions | + | + | + |
| rcpc | Release-Consistent PC | + | + | + |
| rcpc2 | RCPC2 (LDAPR seq-cst) | + | + | |
| **SVE (not used by archmage)** | | | | |
| sve | Scalable Vector Extension | | + | + |
| sve2 | SVE2 | | + | + |
| sve2-aes | SVE2 AES | | | |
| sve2-bitperm | SVE2 bit permutation | | + | + |
| sve2-sha3 | SVE2 SHA3 | | + | + |
| sve2-sm4 | SVE2 SM4 | | + | + |
| f32mm | SVE FP32 matrix multiply | | | |
| f64mm | SVE FP64 matrix multiply | | | |
| **Security** | | | | |
| bti | Branch Target Identification | | | |
| mte | Memory Tagging Extension | | | |
| dit | Data Independent Timing | + | | |
| sb | Speculation Barrier | + | + | |
| ssbs | Speculative Store Bypass Safe | + | | |
| paca | Pointer Auth (address key) | + | + | |
| pacg | Pointer Auth (generic key) | + | + | |
| **System** | | | | |
| dpb | Data Cache Clean to PoP | + | + | |
| dpb2 | Data Cache Clean to PoDP | + | + | |
| rand | Hardware RNG (RNDR) | | | |
| flagm | Condition flag manipulation | + | + | |
| tme | Transactional Memory | | | |

## Why Apple M1 Gets V2 but Not V3

The M1 has most V3 features (sha3, fhm, fcma) but **lacks bf16 and i8mm**. Archmage's Arm64V3Token requires all of: fhm, fcma, sha3, i8mm, bf16. The M2 and later add bf16 + i8mm, so they qualify for V3.

## Windows ARM64 Detection Limitations

Windows `IsProcessorFeaturePresent` only exposes: neon, crc, dotprod, aes, sha2.

Features **not detectable** on Windows ARM64 despite hardware support:
rdm, fp16, fhm, fcma, bf16, i8mm, sha3, frintts, rcpc2, dit, sb, ssbs, paca, pacg, dpb, dpb2, flagm.

This means Azure Cobalt 100 (Neoverse N2) — which has **every** archmage ARM feature — only gets `NeonToken` on Windows. The same chip on Linux gets `Arm64V3Token`.

Paradoxically, Windows *does* detect SVE/SVE2 features (sve2-bitperm, sve2-sha3, sve2-sm4) while missing basic NEON extensions like rdm. This is a Rust `std_detect` limitation tied to the Windows API surface.

## GitHub Actions Runner Summary

| Runner | CPU | OS | Highest Token |
|--------|-----|----|---------------|
| `ubuntu-latest` (x64) | Intel Xeon Platinum 8370C | Linux | **X64V4xToken** |
| `windows-latest` | AMD EPYC 7763 | Windows | X64V3CryptoToken |
| `macos-26-intel` | Intel Core i7-8700B | macOS | X64V3Token |
| `macos-14` | Apple M1 (Virtual) | macOS | Arm64V2Token |
| `macos-latest` | Apple M1 (Virtual) | macOS | Arm64V2Token |
| `ubuntu-24.04-arm` | Neoverse N2 (Cobalt 100) | Linux | **Arm64V3Token** |
| `windows-11-arm` | Neoverse N2 (Cobalt 100) | Windows | NeonToken |

**Note**: `ubuntu-latest` x64 draws from a mixed pool. Some jobs get Intel Xeon 8370C (AVX-512), others get AMD EPYC 7763 (no AVX-512). Token-level tests that require AVX-512 may pass or fail non-deterministically on this runner. The SDE jobs provide deterministic emulation for specific CPU levels.

## Running the Survey

```bash
# On your machine
cargo run --example cpu_survey --features "avx512"

# CI uploads cpu-survey artifacts from every runner automatically
```
