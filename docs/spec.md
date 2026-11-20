# Archmage Specification

> **Primary macro interface (`0.9.32+`):** [`docs/ATTUNE.md`](ATTUNE.md) specifies the unified `#[attune]`, `attuned!`, and `reattune!` macros for feature bodies, proof wrappers, families, and dispatch. Legacy `#[arcane]`, `#[rite]`, `#[magetypes]`, `#[autoversion]`, and `incant!` remain supported and share the same token registry and boundary emitters described below.

This document is the authoritative reference for archmage's token architecture, safety model, and intrinsic classification. It captures design decisions and open questions so context survives across sessions.

## 1. Token Architecture

Archmage uses zero-sized **capability tokens** to prove CPU features are available at compile time. A token can only be constructed via:

- `summon()` — runtime feature detection, returns `Option<Self>`
- `from_context()` — callable safely from a matching or superset target-feature context on the native architecture; otherwise the caller must establish the features.
- `forge_token_dangerously()` — deprecated compatibility alias; use `from_context()`.

Tokens are `Copy + Clone + Send + Sync + 'static`. They carry no data — the type itself is the proof.

### 1.1 Token Registry

`token-registry.toml` is the single source of truth for all token definitions. Everything else — macro feature maps, xtask codegen, validation — derives from it. Run `just generate` to regenerate all derived code, `just validate-registry` to check the registry, and `just validate-tokens` to verify `summon()` checks match the registry.

### 1.2 Current Token Set

#### x86_64 (9 tokens)

| Token | Aliases | Features | Hardware |
|-------|---------|----------|----------|
| `X64V1Token` | `Sse2Token` | sse, sse2 (x86_64 baseline) | All x86_64 CPUs |
| `X64V2Token` | — | + sse3, ssse3, sse4.1, sse4.2, popcnt, cmpxchg16b | Nehalem 2008+, Bulldozer 2011+ |
| `X64CryptoToken` | — | v2 + pclmulqdq, aes | Westmere 2010+, Bulldozer 2011+ |
| `X64V3Token` | `Desktop64` (`Avx2FmaToken` deprecated) | + avx, avx2, fma, bmi1, bmi2, f16c, lzcnt, movbe | Haswell 2013+, Zen 1 2017+ |
| `X64V3CryptoToken` | — | v3 + vpclmulqdq, vaes | Zen 3+ 2020, Alder Lake 2021+ |
| `X64V3GfniCryptoToken` | — | v3 crypto + gfni | Alder/Raptor/Meteor/Arrow/Lunar Lake, Sierra Forest, Zen 4+ |
| `X64V4Token` | `Avx512Token`, `Server64` | v3 + pclmulqdq, aes, avx512f, avx512bw, avx512cd, avx512dq, avx512vl | Skylake-X 2017+, Zen 4 2022+ |
| `X64V4xToken` | `Avx512ModernToken` | + avx512vpopcntdq, avx512ifma, avx512vbmi, avx512vbmi2, avx512bitalg, avx512vnni, vpclmulqdq, gfni, vaes | Ice Lake 2019+, Zen 4 2022+ |
| `Avx512Fp16Token` | — | v4 + avx512fp16 | Sapphire Rapids 2023+ |

Features are cumulative — each token lists ALL features it enables, not just the delta from the previous tier. This eliminates the class of bugs where "minimal" lists diverge from "cumulative" lists. LLVM deduplicates redundant features in `#[target_feature]`.

`X64V1Token` / `Sse2Token` represents the x86_64 baseline (SSE + SSE2, always available). `summon()` always returns `Some` on x86_64.

#### AArch64 (6 tokens)

**Compute tiers** (archmage-defined, not ARM architecture versions):

| Token | Features | Hardware |
|-------|----------|----------|
| `NeonToken` / `Arm64` | neon | Baseline, all AArch64 |
| `Arm64V2Token` | + crc, rdm, dotprod, fp16, aes, sha2 | Cortex-A55+, Apple M1+, Graviton 2+ |
| `Arm64V3Token` | + fhm, fcma, sha3, i8mm, bf16 | Cortex-A510+, Apple M2+, Snapdragon X, Graviton 3+ |

**Crypto leaves** (independent, not part of compute hierarchy):

| Token | Features | Notes |
|-------|----------|-------|
| `NeonAesToken` | neon, aes | ARMv8-A with Crypto |
| `NeonSha3Token` | neon, sha3 | ARMv8.2-A+ |
| `NeonCrcToken` | neon, crc | ARMv8.1-A+ (most AArch64 CPUs) |

SVE/SVE2 tokens are prohibited — Rust stable doesn't support SVE intrinsics.

#### WASM (2 tokens)

| Token | Features | Notes |
|-------|----------|-------|
| `Wasm128Token` | simd128 | Compile-time only (`#[cfg(target_feature = "simd128")]`) |
| `Wasm128RelaxedToken` | simd128, relaxed-simd | Compile-time only; relaxed operations may differ between engines |

### 1.3 Token Hierarchy

Tokens form a subsumption hierarchy. Higher-tier tokens produce lower-tier tokens through extraction methods (there are no `From`/`Into` conversions between tokens):

```
x86_64:
  X64V4xToken → X64V4Token (v4) → X64V3Token (v3) → X64V2Token (v2) → X64V1Token (v1)
               → X64V3GfniCryptoToken → X64V3CryptoToken → X64V3Token (v3)
                                                       → X64CryptoToken → X64V2Token (v2)
  Avx512Fp16Token → X64V4Token (v4) → X64V3Token (v3) → X64V2Token (v2) → X64V1Token (v1)
  X64CryptoToken → X64V2Token (v2) → X64V1Token (v1)
AArch64:
  Arm64V3Token → Arm64V2Token → NeonToken
  NeonAesToken → NeonToken
  NeonSha3Token → NeonToken
  NeonCrcToken → NeonToken
```

Extraction methods are `.v1()`, `.v2()`, `.x64_crypto()`, `.v3()`, `.v3_crypto()`, `.v3_gfni_crypto()`, `.v4()` (alias `.avx512()`), `.neon()`, `.neon_aes()`, `.neon_sha3()`, `.neon_crc()`, `.arm_v2()` and `.wasm128()`, each on the tokens that cover it. Extraction is free. `IntoConcreteToken` recovers the concrete type of a generic token: `as_x64v3()` returns `Some` only for exactly `X64V3Token`, so it neither upgrades nor downcasts.

### 1.4 Trait Hierarchy

Traits provide generic bounds. Tokens implement the traits matching their capabilities.

**Width traits (DEPRECATED — will be removed):**
```
Has512BitSimd → Has256BitSimd → Has128BitSimd
```
Width traits are misleading (`Has256BitSimd` enables AVX but NOT AVX2 or FMA). Use concrete tokens or tier traits instead.

**x86 tier traits:**
```
HasX64V4 → HasX64V2
```

There is no `HasX64V3` trait. For v3 bounds, use `X64V3Token` directly — it's the recommended baseline for high-performance SIMD code.

**AArch64 traits:**
```
HasArm64V3 → HasArm64V2 → HasNeon
                          → HasNeonAes
HasNeonSha3 → HasNeon
HasArm64V3 → HasNeonSha3
```

**Trait implementations by token:**

| Token | HasX64V2 | HasX64V4 | HasNeon | HasNeonAes | HasNeonSha3 | HasArm64V2 | HasArm64V3 |
|-------|----------|----------|---------|------------|-------------|------------|------------|
| X64V1Token | | | | | | | |
| X64V2Token | x | | | | | | |
| X64CryptoToken | x | | | | | | |
| X64V3Token | x | | | | | | |
| X64V3CryptoToken | x | | | | | | |
| X64V3GfniCryptoToken | x | | | | | | |
| X64V4Token | x | x | | | | | |
| X64V4xToken | x | x | | | | | |
| Avx512Fp16Token | x | x | | | | | |
| NeonToken | | | x | | | | |
| Arm64V2Token | | | x | x | | x | |
| Arm64V3Token | | | x | x | x | x | x |
| NeonAesToken | | | x | x | | | |
| NeonSha3Token | | | x | | x | | |
| NeonCrcToken | | | x | | | | |
| Wasm128Token | | | | | | | |
| Wasm128RelaxedToken | | | | | | | |

### 1.5 Cross-Platform Behavior

All token types are defined on all architectures. On unsupported architectures, `summon()` returns `None`. This enables cross-platform code that compiles everywhere and dispatches only on matching hardware.

**Cfg-out default:** `#[attune]` (and legacy `#[arcane]` / `#[rite]`) emits architecture-specific functions only on the matching `target_arch`. On unsupported architectures, no function is generated — less dead code, cleaner binaries. Direct call sites referencing a single-architecture function by name must use `#[cfg(target_arch)]` guards, or dispatch through `attuned!` / `#[attune(dispatch)]` (or legacy `incant!`), which cfg-gate candidate calls automatically. No manual `#[cfg(target_arch)]` is needed on `#[attune]` function definitions.

**`attuned!` and `#[attune(dispatch)]` are the primary dispatch mechanisms** — they wrap each candidate call in `#[cfg(target_arch)]` blocks automatically, so portable call sites need no manual `#[cfg]` guards.

Stub modules for token types (not macro-generated functions): `x86_stubs.rs`, `arm_stubs.rs`, `wasm_stubs.rs`.

## 2. Safety Model

### 2.1 The Soundness Invariant

For every token `T`:

```
target_features_in_macro(T) ⊆ features_checked_in_summon(T)
```

`#[attune]` (and legacy `#[arcane]` / `#[rite]`) resolves the token type or tier selector against `token-registry.toml` and generates `#[target_feature(enable = "...")]`. If `summon()` checked fewer features than the macro enables, it would hand out tokens on CPUs that lack an enabled feature, and the compiler could emit instructions the CPU does not support — **undefined behavior** (illegal instructions, crashes, silent data corruption).

The reverse direction (summon checks more than the macro enables) is safe, only conservative.

### 2.2 Rust 1.86-1.87 Intrinsic Safety

Rust 1.86 stabilized `target_feature_11` (safe `fn` with `#[target_feature]`, safe cross-calls between matching contexts). Rust 1.87 declared value-based `std::arch` intrinsics safe. Together, **value-based intrinsics are safe inside `#[target_feature]` functions**:

```rust
use archmage::prelude::*;

#[attune(wrap, import_intrinsics)]
fn example(_token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    let a = _mm256_loadu_ps(data);
    let b = _mm256_add_ps(a, a);
    let mut out = [0.0; 8];
    _mm256_storeu_ps(&mut out, b);
    out
}
```

This means archmage doesn't need to wrap value intrinsics. `#[attune(wrap)]` (and `_tier_t` proof wrappers) generate `#[target_feature]` inner functions; value-based intrinsics are safe inside them.

Only pointer-based operations and processor-state access remain unsafe:
- Loads from `*const T` (`_mm256_loadu_ps`, etc.)
- Stores to `*mut T` (`_mm256_storeu_ps`, etc.)
- Gather/scatter operations
- MXCSR, transactional-memory and state save/restore intrinsics

`_mm_prefetch` takes a pointer and is safe: a prefetch never faults.

For safe memory access, use `import_intrinsics` which provides reference-based alternatives (accepts `&[T; N]`/`&mut [T; N]` instead of raw pointers).

### 2.3 How `#[attune]` (and Legacy `#[arcane]` / `#[rite]`) Work

`#[attune]` determines `#[target_feature]` attributes from an explicit tier (`#[attune(v3)]`), a tier suffix (`#[attune] fn work_v3(...)` or `work_v3_t(...)`), a family selector (`#[attune(_*_t, ...)]`, `#[attune(dispatch)]`), or a proof parameter with `wrap` (`#[attune(wrap)] fn work(token: X64V3Token, ...)`). Legacy `#[arcane]` and `#[rite]` parse the token parameter or tier directly. All mappings come from `token-registry.toml` (see [`docs/ATTUNE.md`](ATTUNE.md) for the full grammar).

Passing matching features through a call hierarchy lets LLVM inline freely with no optimization boundary. When a callee requires features its caller lacks — a stronger token, or a caller bounded by a tier trait and compiled at that lower tier — LLVM hits a target-feature boundary and cannot inline across it, costing 4–6x in hot loops (see `docs/PERFORMANCE.md`).

#### Sibling Expansion (default for proof wrappers)

`#[attune(wrap)]` (and `_tier_t` proof wrappers, plus legacy `#[arcane]`) generates two functions at the same scope:

```rust
use archmage::prelude::*;

// Input:
#[attune(wrap, import_intrinsics)]
fn kernel(_token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    let v = _mm256_loadu_ps(data);
    let mut out = [0.0; 8];
    _mm256_storeu_ps(&mut out, v);
    out
}
```

The macro generates the `#[inline(always)]` safe outer wrapper and a private `#[doc(hidden)]` `__arcane_*` sibling carrying `#[target_feature]`.
User code does not implement that boundary; see the expansion tests for exact output.

Both functions live at the same scope level. For inherent receiver methods, both are in the same `impl` block, so `self`, `Self`, and associated constants resolve naturally; receiverless associated functions pass `in_impl` so sibling calls resolve as `Self::name`.

#### Nested Expansion (`in_trait`, `nested`, or `_self = Type`)

For trait impls (where sibling expansion would add methods not in the trait definition), use `#[attune(wrap, in_trait)]` (receiverless) or `#[attune(wrap, _self = Type)]` (with receiver; legacy `#[arcane(nested)]` / `#[arcane(_self = Type)]`):

The macro emits a block-local `#[target_feature]` function plus a token-justified boundary call. Callers use `#[attune]` (or legacy `#[arcane]` / `#[magetypes]`), and helpers use a matching direct feature context.

#### Key `#[attune]` forms

| Form | Effect |
|------|--------|
| `#[attune(wrap)]` | Proof wrapper from parameter token/trait bound; sibling expansion, cfg-out on wrong arch |
| `#[attune] fn f_v3_t(...)` | Inferred V3 proof wrapper (inserts token parameter if omitted) |
| `#[attune(v3)]` / `#[attune] fn f_v3(...)` | Direct V3 feature body (`#[target_feature]` + `#[inline]`, no wrapper) |
| `#[attune(_*, _*_t, dispatch)]` | Portable family generation (`v3`, `neon`, `wasm128`, `scalar`; add AVX-512 with `+v4(cfg(avx512))`) |
| `#[attune(wrap, in_impl)]` | Receiverless associated function inside an inherent `impl` |
| `#[attune(wrap, _self = Type)]` | Trait impl method with receiver (implies nested expansion) |

#### Direct Feature Bodies (`#[attune(v3)]`, `_*`, and legacy `#[rite]`)

Direct feature bodies apply `#[target_feature]` + `#[inline]` directly to the function, with no wrapper. When the caller already has matching or superset features, LLVM inlines freely — no boundary. Use proof wrappers (`#[attune(wrap)]` / `_tier_t`) at entry points from non-SIMD code, and direct feature bodies (`#[attune(v3)]` / `#[attune(_v3, _neon)]` / `#[attune(_*)]`) for helpers called within SIMD code.

Direct feature bodies cannot be applied directly to safe trait impl methods because Rust rejects `#[target_feature]` on safe trait methods; use `#[attune(wrap, _self = Type)]` or delegate to a free function there.

## 3. Intrinsic Safety Classification

The Rust 1.98.0 intrinsic database (`docs/intrinsics/complete_intrinsics.csv`)
marks 626 x86 intrinsics `unsafe`. Inside a `#[target_feature]` function that
covers their features, all the others are safe to call. The `unsafe` ones fall
into two groups, and the database does not record which.

### 3.1 Memory access through a raw pointer (576 of the 626)

Loads and stores (`_mm*_loadu_*`, `_mm*_storeu_*`, `_mm*_stream_*`), masked and
expanding loads, compressing stores, `_mm_maskmoveu_si128`, gathers and
scatters, mask-register loads (`_load_mask16`), the AVX-NE-CONVERT reads
(`_mm*_bcstnebf16_ps`, `_mm*_cvtneebf16_ps` and relatives), AMX tile loads and
stores (`_tile_loadd`, `_tile_stored`), and cache control (`_mm_clflush`).

Contiguous loads and stores become safe by taking `&[T; N]` / `&mut [T; N]`
instead of a pointer; `import_intrinsics` provides these. A reference does not
make gather or scatter safe, because each lane's offset is data: magetypes
bounds every lane in `simd_storage.rs`. `_mm_prefetch` also takes a pointer but
is already safe, because a prefetch never faults.

### 3.2 Processor state (the other 50)

MXCSR access (`_mm_getcsr`, `_mm_setcsr`, `_MM_SET_FLUSH_ZERO_MODE` and
relatives), transactional memory (`_xbegin`, `_xend`, `_xabort`, `_xtest`),
extended control registers (`_xgetbv`, `_xsetbv`), the Key Locker key load
(`_mm_loadiwkey`), and AMX tile operations (`_tile_dpbssd`, `_tile_zero`,
`_tile_release` and relatives), which act on tile registers that
`_tile_loadconfig` must have configured. State save and restore (`_xsave`,
`_xrstor`, `_fxrstor`) belongs here too, though it is counted in 3.1 because it
takes a pointer. Changing this state affects code the caller does not control,
so no reference-based signature makes these safe.

### 3.3 Open Question: Safety Annotation

Should the CSV gain a `safety_reason` column (`pointer_deref` | `state`) to enable:

1. Automatic generation of reference-based safe wrappers for 3.1
2. Validation that magetypes never exposes 3.2 without explicit `unsafe`
3. Documentation of WHY each intrinsic is unsafe

This would require auditing all 626 unsafe x86 intrinsics.

## 4. Feature Detection

### 4.1 Token Detection

Token detection combines compile-time and runtime checks:

1. `compiled_with()` — returns `Some(true)` if the features are enabled for the whole crate (`-Ctarget-cpu`, `-Ctarget-feature`) and `testable_dispatch` is off, `Some(false)` on the wrong architecture, and `None` when only runtime detection can tell. A function's own `#[target_feature]` attribute does not change it: `cfg(target_feature)` is evaluated per crate.
2. `summon()` — returns `Some(token)` if the CPU supports the required features. Uses atomic caching for fast repeated calls (~1.3 ns). When `compiled_with()` returns `Some(true)`, `summon()` compiles away entirely.

Inside an `#[attune]` feature body (or legacy `#[arcane]` / `#[rite]`), `summon()` is still a cached runtime check. To get a token for the function's own tier, call `from_context()`, which rustc accepts there and which compiles to nothing.

### 4.2 WASM Special Case

WASM SIMD128 uses compile-time detection only (`#[cfg(target_feature = "simd128")]`). There is no runtime detection mechanism.

### 4.3 AArch64 NEON Detection

NEON is part of the standard AArch64 targets, so `NeonToken::summon()` is a constant `Some` there (the `testable_dispatch` feature keeps it a runtime check so tests can disable it). The other AArch64 tokens use runtime detection:

- **Linux/Android:** Uses `getauxval(AT_HWCAP)` via `std::arch::is_aarch64_feature_detected!`
- **macOS / Mac Catalyst / aarch64 simulators:** Uses `sysctlbyname` — with an Apple Silicon fallback for the ten M1+ baseline features, applied **only** on these provably-M1+ hosts (works around the macOS 15.x std_detect bug). Device iOS/tvOS/watchOS/visionOS targets use genuine runtime detection and otherwise fail closed — their A7–A12-era hardware does not guarantee the list (see `docs/SOUNDNESS.md` incident log and `tests/apple_fallback_guard.rs`).
- **Windows:** Uses `winarm-cpufeatures` (registry-backed `ID_AA64*_EL1` decoding + `IsProcessorFeaturePresent`) — see Known Platform Detection Issues in CLAUDE.md

## 5. CI Testing

### Intel SDE (x86 emulation)

Intel Software Development Emulator (SDE) allows testing x86 feature paths on any x86 host:

- `sde64 -p4` — Pentium 4 (SSE2 only)
- `sde64 -nhm` — Nehalem (SSE4.2, no AVX)
- `sde64 -hsw` — Haswell (AVX2 + FMA, no AVX-512)
- `sde64 -skx` — Skylake-X (AVX-512)
- `sde64 -icl` — Ice Lake (AVX-512 + extensions)

See `just test-all-cpus` for the full suite.

### Cross-compilation (ARM + WASM)

ARM testing uses `cross` (Docker-based QEMU emulation):
- `cross test --target aarch64-unknown-linux-gnu` for AArch64

WASM testing uses `wasmtime`:
- `cargo test --target wasm32-wasip1` with `-C target-feature=+simd128`

See `just test-cross` and `just test-parity` for full cross-platform suites.

## 6. Token Registry

`token-registry.toml` is the single source of truth. All token code, macro registries, SIMD types, and documentation are generated from it via `cargo xtask generate` (aka `just generate`).

### 6.1 Registry Schema

The registry uses TOML with these table types:

- `[[token]]` — token definitions with name, arch, aliases, deprecated_aliases, features, traits, parents, display_name, short_name, magetypes_namespace, extraction_aliases, doc
- `[[trait]]` — trait definitions with name, features, parents
- `[[width_namespace]]` — width namespace config for simd type re-exports
- `[[polyfill_w256]]` / `[[polyfill_w512]]` — polyfill platform configs

(The former `[[magetypes_file]]` file-to-token mappings are gone: intrinsic
gating is now derived from the code structure itself by the soundness
scanner — see `docs/SOUNDNESS.md`.)

### 6.2 Validation

Loading the registry checks it: no duplicate names or aliases, every referenced trait exists, every parent exists on the same architecture, no parent cycles, and each token's features include those of its traits. `cargo xtask validate-registry` runs only that. `cargo xtask validate` loads the registry and then runs:
1. The soundness scanner (`cargo xtask soundness`): every intrinsic call sits in a context whose features cover it, plus the structural and SAFETY-comment rules
2. A portability check on the committed trybuild `.stderr` snapshots
3. `summon()` source checks exactly the registry features
4. The hand-written AVX-512 f32 delegation covers every method it must
5. The generated backend surface stays within its compile budget
6. The published sub-crates carry both license files

`just check-generated` regenerates everything and fails if the tree changed, which catches drift between the registry and the generated code.

`cargo xtask parity` checks API surface parity across x86/ARM/WASM (currently 0 issues).
