# Feature Status

Tracks CPU target features relative to archmage token coverage. Updated March 2026.

**MSRV: Rust 1.89** (archmage, archmage-macros, magetypes, xtask).

The source of truth for what archmage tokens enable is `token-registry.toml`.

---

## Stable and in archmage tokens

Features with stable target features AND stable intrinsics, enabled by one or
more archmage tokens via `#[target_feature]`.

### x86_64

| Feature | Stable since | Token(s) | Notes |
|---------|-------------|----------|-------|
| `sse` | 1.27 | All x86 tokens | x86_64 baseline |
| `sse2` | 1.27 | All x86 tokens | x86_64 baseline |
| `sse3` | 1.27 | V2+ | |
| `ssse3` | 1.27 | V2+ | |
| `sse4.1` | 1.27 | V2+ | |
| `sse4.2` | 1.27 | V2+ | |
| `popcnt` | 1.27 | V2+ | |
| `cmpxchg16b` | 1.69 | V2+ | 128-bit atomic CAS |
| `avx` | 1.27 | V3+ | 256-bit float |
| `avx2` | 1.27 | V3+ | 256-bit integer |
| `fma` | 1.27 | V3+ | Fused multiply-add |
| `bmi1` | 1.27 | V3+ | Bit manipulation |
| `bmi2` | 1.27 | V3+ | Bit manipulation |
| `f16c` | 1.68 | V3+ | FP16 conversion |
| `lzcnt` | 1.27 | V3+ | Leading zero count |
| `movbe` | 1.70 | V3+ | Byte-swap load/store |
| `pclmulqdq` | 1.27 | X64CryptoToken, V3Crypto+ | 128-bit carry-less multiply |
| `aes` | 1.27 | X64CryptoToken, V3Crypto+ | AES-NI |
| `vpclmulqdq` | 1.89 | X64V3CryptoToken, V4x | 256/512-bit carry-less multiply |
| `vaes` | 1.89 | X64V3CryptoToken, V4x | 256/512-bit AES |
| `avx512f` | 1.89 | V4+ | AVX-512 Foundation |
| `avx512bw` | 1.89 | V4+ | Byte/Word operations |
| `avx512cd` | 1.89 | V4+ | Conflict Detection |
| `avx512dq` | 1.89 | V4+ | Dword/Qword operations |
| `avx512vl` | 1.89 | V4+ | 128/256-bit AVX-512 |
| `avx512vpopcntdq` | 1.89 | V4x | Vector popcount |
| `avx512ifma` | 1.89 | V4x | Integer fused multiply-add |
| `avx512vbmi` | 1.89 | V4x | Vector byte manipulation |
| `avx512vbmi2` | 1.89 | V4x | Vector byte manipulation 2 |
| `avx512bitalg` | 1.89 | V4x | Bit algorithms |
| `avx512vnni` | 1.89 | V4x | Vector neural network |
| `gfni` | 1.89 | V4x | Galois field operations |
| `avx512fp16` | 1.89 | Avx512Fp16Token | Half-precision arithmetic |

### AArch64 — full intrinsic support

| Feature | Stable since | Token(s) | Stable intrinsics | Notes |
|---------|-------------|----------|-------------------|-------|
| `neon` | 1.61 | All ARM tokens | ~1540 | Baseline SIMD |
| `aes` | 1.61 | NeonAesToken, Arm64V2+ | ~37 | AES + polynomial multiply |
| `sha2` | 1.61 | Arm64V2+ | ~10 | SHA-1/SHA-256 |
| `sha3` | 1.61 | NeonSha3Token, Arm64V3 | 22 | SHA-512/SHA-3 |
| `crc` | 1.61 | NeonCrcToken, Arm64V2+ | 8 | CRC32 |
| `rdm` | 1.61 | Arm64V2+ | 36 | Rounding double multiply |

### AArch64 — target feature only (intrinsics pending)

These features have stable target features and are in our tokens, so `#[arcane]`
emits the correct `#[target_feature]` and LLVM can auto-vectorize with them. But
users cannot call `std::arch` intrinsics for these on stable Rust. The token is
still useful — it just can't be used for direct intrinsic calls yet.

| Feature | Stable since | Token(s) | Intrinsic status | Notes |
|---------|-------------|----------|------------------|-------|
| `dotprod` | 1.61 | Arm64V2+ | ALL nightly (`stdarch_neon_dotprod`) | SDOT/UDOT. Critical for ML inference. |
| `fp16` | 1.61 | Arm64V2+ | 95/210 stable | Conversion + FMA stable, arithmetic mostly nightly. |
| `fcma` | 1.61 | Arm64V3 | ALL nightly (`stdarch_neon_fcma`) | Complex number FMA. |
| `i8mm` | 1.61 | Arm64V3 | ALL nightly (`stdarch_neon_i8mm`) | Int8 matrix multiply. ML inference. |
| `fhm` | 1.61 | Arm64V3 | None exist in stdarch | FMLAL/FMLSL. |
| `bf16` | 1.61 | Arm64V3 | None exist in stdarch | BFloat16. |

### WebAssembly

| Feature | Stable since | Token(s) | Notes |
|---------|-------------|----------|-------|
| `simd128` | 1.54 | Wasm128Token | ~100+ intrinsics |
| `relaxed-simd` | 1.82 | Wasm128RelaxedToken | 28 relaxed instructions |

---

## Stable but not in archmage

These features are stable in rustc and have no technical barriers to adoption.
We haven't added them because each new token adds API surface, detection logic,
and maintenance burden.

**Design question: cargo feature gates.** Archmage isn't just about SIMD — tokens
can prove any CPU capability. But crypto, key management, and other specialized
tokens probably belong behind cargo feature flags (e.g., `features = ["crypto"]`,
`features = ["ml"]`) to keep the default API focused. The core SIMD compute
tokens (V1-V4x, Neon, WASM) should remain always-available.

### x86_64 — SIMD compute

These have stable intrinsics and real use cases for SIMD compute or ML workloads.

| Feature | Stable since | Intrinsics | Hardware | Candidate token | Notes |
|---------|-------------|-----------|----------|----------------|-------|
| `avx512bf16` | 1.89 | ~24 stable | Cooper Lake, Sapphire Rapids+, Zen 4+ | Avx512Bf16Token (V4 + bf16) | BFloat16 dot product + conversion. NOT on Ice Lake/Tiger Lake/Rocket Lake, so doesn't belong in V4x. |
| `avxvnni` | 1.89 | stable | Alder Lake+, Zen 5 | V3 + avxvnni | VNNI without AVX-512. For CPUs that dropped AVX-512. |
| `avxifma` | 1.89 | stable | Sierra Forest+, Zen 5 | — | Integer FMA without AVX-512. |
| `avxneconvert` | 1.89 | stable | Sierra Forest+ | — | FP conversion without AVX-512. |
| `avxvnniint8` | 1.89 | stable | Sierra Forest+ | — | VNNI 8-bit without AVX-512. |
| `avxvnniint16` | 1.89 | stable | Arrow Lake+ | — | VNNI 16-bit without AVX-512. |

### x86_64 — crypto / specialized

Stable, have intrinsics, but specialized. Should be behind a cargo feature flag.

| Feature | Stable since | Intrinsics | Hardware | Notes |
|---------|-------------|-----------|----------|-------|
| `sha` | 1.27 | ~10 stable | Zen 1+, Goldmont+, Ice Lake+ | SHA-1/SHA-256. Not in any psABI level. |
| `sha512` | 1.89 | stable | Lunar Lake, Arrow Lake, Zen 5 | SHA-512. Implies AVX2. |
| `sm3` | 1.89 | stable | Niche (Chinese market CPUs) | ShangMi 3 hash. |
| `sm4` | 1.89 | stable | Niche (Chinese market CPUs) | ShangMi 4 cipher. |
| `kl` | 1.89 | ~11 stable | Alder Lake (S-SKU), Sapphire Rapids | Intel Key Locker — hardware-bound AES key wrapping. |
| `widekl` | 1.89 | stable | Same as `kl` | Wide Key Locker — 384-bit key variants. |

### x86_64 — non-SIMD, out of scope

Stable but outside archmage's domain. No plans to add tokens.

| Feature | Stable since | Notes |
|---------|-------------|-------|
| `rdrand` | 1.27 | Hardware RNG. |
| `rdseed` | 1.27 | Hardware RNG seed. |
| `adx` | 1.61 | Multi-precision add-carry. Scalar arithmetic. |
| `fxsr` | 1.27 | FPU state save/restore. OS-level. |
| `xsave` | 1.27 | Extended state save. OS-level. |
| `xsavec` | 1.27 | Extended state save (compacted). OS-level. |
| `xsaveopt` | 1.27 | Extended state save (optimized). OS-level. |
| `xsaves` | 1.27 | Extended state save (supervisor). OS-level. |
| `sse4a` | 1.91 | AMD SSE4a. Legacy AMD-only (Phenom/Bulldozer). |
| `tbm` | 1.91 | AMD Trailing Bit Manipulation. Piledriver only, discontinued. |
| `avx512vp2intersect` | 1.89 | Tiger Lake only, removed from all later CPUs. Dead end. |

### AArch64 — stable target features without archmage tokens

Most are system-level, not compute. SVE target features are stable but useless
without intrinsics — see "Looking forward" section.

| Feature | Stable since | Category | Notes |
|---------|-------------|----------|-------|
| `sm4` | 1.61 | Crypto | SM3/SM4. Niche. Would need a cargo feature. |
| `jsconv` | 1.61 | Scalar | JavaScript FP conversion (FJCVTZS). Single instruction. |
| `lse` | 1.61 | Atomics | Large System Extension atomics (CAS/SWP). |
| `rcpc` | 1.61 | Memory ordering | Acquire semantics (LDAPR). |
| `rcpc2` | 1.61 | Memory ordering | RCPC with immediates. |
| `flagm` | 1.61 | Scalar | Flag manipulation. |
| `dpb` / `dpb2` | 1.61 | Cache | Data persistence barriers. |
| `rand` | 1.61 | RNG | Hardware RNG (RNDR/RNDRRS). |
| `ras` | 1.61 | System | Reliability/Availability/Serviceability. |
| `lor` | 1.61 | Memory ordering | Limited Ordering Regions. |
| `bti` | 1.61 | Security | Branch Target Identification. |
| `mte` | 1.61 | Security | Memory Tagging Extension. |
| `dit` | 1.61 | Security | Data-independent timing. |
| `frintts` | 1.61 | Scalar FP | FP-to-int rounding instructions. |
| `sb` / `ssbs` | 1.61 | Security | Speculation barriers. |
| `pan` / `vh` | 1.61 | System | Privileged access / virtualization. OS-level. |
| `pmuv3` / `spe` | 1.61 | System | Performance monitoring / profiling. OS-level. |
| `tme` | 1.61 | System | Transactional Memory Extension. |
| `paca` / `pacg` | 1.61 | Security | Pointer Authentication. Must enable together. |

---

## Looking forward

Features we're watching that are unstable or have incomplete support.

### Want to adopt when ready

| Feature | Arch | Blocker | Notes |
|---------|------|---------|-------|
| `avx10.1` / `avx10.2` | x86 | Unstable target feature (`avx10_target_feature`). No shipping hardware yet (Panther Lake 2026?). | Intel's AVX-512 successor. Consolidates 13 AVX-512 subsets into one feature level. Mandatory 256-bit, optional 512-bit. Natural V4 successor token. |
| AArch64 `dotprod` intrinsics | aarch64 | Nightly-only (`stdarch_neon_dotprod`). Target feature already stable and in Arm64V2. | SDOT/UDOT — critical for ML inference. We have the token, just waiting on intrinsic stabilization. |
| AArch64 `fcma` intrinsics | aarch64 | Nightly-only (`stdarch_neon_fcma`). Target feature already stable and in Arm64V3. | Complex number FMA. Same — token exists, intrinsics don't. |
| AArch64 `i8mm` intrinsics | aarch64 | Nightly-only (`stdarch_neon_i8mm`). Target feature already stable and in Arm64V3. | Int8 matrix multiply. ML inference. |
| AArch64 `bf16` intrinsics | aarch64 | No intrinsics exist in stdarch. Target feature stable and in Arm64V3. | BFloat16. Waiting on stdarch. |
| AArch64 `fhm` intrinsics | aarch64 | No intrinsics exist in stdarch. Target feature stable and in Arm64V3. | FMLAL/FMLSL. Waiting on stdarch. |
| AArch64 remaining `fp16` | aarch64 | 115/210 intrinsics still nightly. | Half-precision arithmetic. Gradual progress. |

### Blocked on language/compiler changes

These require fundamental Rust language work.

| Feature | Arch | Blocker | Tracking | Notes |
|---------|------|---------|----------|-------|
| SVE / SVE2 intrinsics | aarch64 | Scalable vector types need `Sized` reform (RFC 3729). Target features are stable but no `std::arch` SVE intrinsics exist on stable. | [#145052](https://github.com/rust-lang/rust/issues/145052) | 2027+ at earliest. `sve`/`sve2` target features being stable is misleading — you get auto-vectorization but can't call SVE intrinsics. Archmage's NO SVE prohibition remains correct. |
| RISC-V Vector (RVV) | riscv64 | Same scalable type blocker as SVE. | [#145052](https://github.com/rust-lang/rust/issues/145052), [#114544](https://github.com/rust-lang/rust/issues/114544) | Same timeline as SVE. Would need new arch support in archmage. |
| AMX | x86 | Tile register ABI unresolved (`tmm` registers). | [#126622](https://github.com/rust-lang/rust/issues/126622), [#133144](https://github.com/rust-lang/rust/issues/133144) | Matrix multiply accelerator. 9 unstable features. Different programming model — may not fit archmage's token model at all. |

### No current position

Features that exist but we have no plans for.

| Feature | Arch | Status | Notes |
|---------|------|--------|-------|
| `apxf` | x86 | Unstable | Intel APX — doubles GPR count (16 to 32). Compiler codegen only, no intrinsics needed. No SIMD impact. |
| `movrs` | x86 | Unstable | Move with read-shared hint. Minor. |
| `ermsb` | x86 | Unstable | Enhanced REP MOVSB. Compiler hint. |
| `rtm` | x86 | Unstable | Intel TSX (deprecated on consumer CPUs). |
| `x87` | x86 | Unstable | About disabling x87, not enabling. Kernel/embedded use. |
| `xop` | x86 | Unstable | AMD XOP (Bulldozer-era, discontinued). |
| AArch64 SME | aarch64 | Unstable | Scalable Matrix Extension. 11 features. Even further out than SVE — same type system blocker plus matrix tile ABI. |
| AArch64 FP8 | aarch64 | Unstable | 7 features. ML inference. No intrinsics. |
| AArch64 `cssc` | aarch64 | Unstable | Scalar integer abs/min/max/count. ARMv8.9. |
| AArch64 `lse128` | aarch64 | Unstable | 128-bit atomics. |
| AArch64 `mops` | aarch64 | Unstable | Hardware memcpy/memset acceleration. |
| AArch64 `f32mm` / `f64mm` | aarch64 | Stable target feature | Requires SVE. Blocked by scalable type system. |
| AArch64 SVE2 extensions | aarch64 | Mixed | `sve2-aes`, `sve2-bitperm`, `sve2-sha3`, `sve2-sm4` are stable target features. Blocked by same SVE intrinsics blocker. |
| WASM `atomics` | wasm32 | Unstable | SharedArrayBuffer threading. |
| WASM `wide-arithmetic` | wasm32 | Unstable | 128-bit integer ops. Potentially interesting. |

---

## Notable facts

- **`avx512bf16` does NOT belong in V4xToken.** Ice Lake, Tiger Lake, and Rocket Lake have
  all the V4x features but lack BF16. It first appeared on Cooper Lake (Xeon), then Sapphire
  Rapids and Zen 4+. Needs its own token (V4 + bf16) or a higher-tier token.

- **`gfni` is not in any psABI level** and does not require AVX-512 — it only needs SSE2.
  Some CPUs (Tremont, Alder Lake E-cores) have GFNI without AVX-512. We currently only
  offer it via V4xToken. A standalone GFNI token at V2 level could be useful but is niche.

- **`vaes` and `vpclmulqdq` don't require AVX-512 either.** They work with AVX2. We have
  X64V3CryptoToken for exactly this case.

- **No crypto/hash features are in any psABI level.** The psABI explicitly excludes
  "non-general-purpose" extensions: AES, PCLMULQDQ, SHA, RDRAND, and GFNI are all outside
  the v1-v4 level definitions. Archmage's crypto tokens (X64CryptoToken, X64V3CryptoToken)
  are leaf tokens off the main hierarchy for this reason.

- **VEX-encoded ML extensions** (`avxvnni`, `avxifma`, `avxneconvert`, `avxvnniint8`,
  `avxvnniint16`) exist for CPUs that dropped AVX-512 (Alder Lake, Sierra Forest). These
  bring select AVX-512 functionality to V3-class CPUs. Potentially useful for ML inference
  on consumer hardware, but no users have requested them.

- **`avx512vp2intersect`** was implemented in Tiger Lake, then removed from all subsequent
  Intel CPUs. Do not add a token for it.

- **Cargo feature gating strategy.** The VEX ML extensions and crypto/hash tokens should
  be behind cargo features to avoid bloating the default token set. Proposed gates:
  - `crypto` — SHA, SHA-512, SM3, SM4, Key Locker tokens
  - `ml` — BF16, VNNI, IFMA tokens (both AVX-512 and VEX variants)
  - The core compute hierarchy (V1-V4x, Neon/Arm64V2/V3, WASM) stays ungated
