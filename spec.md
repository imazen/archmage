# Archmage Specification

This document is the authoritative reference for archmage's token architecture, safety model, and intrinsic classification. It captures design decisions and open questions so context survives across sessions.

## 1. Token Architecture

Archmage uses zero-sized **capability tokens** to prove CPU features are available at compile time. A token can only be constructed via:

- `summon()` / `summon()` — runtime feature detection, returns `Option<Self>`
- `forge_token_dangerously()` — unsafe, caller guarantees features are present

Tokens are `Copy + Clone + Send + Sync + 'static`. They carry no data — the type itself is the proof.

### 1.1 Token Registry (planned)

`token-registry.toml` is the single source of truth for all token definitions. Everything else — macro feature maps, xtask codegen, validation — derives from it. The registry schema is defined in the plan (see `melodic-skipping-pancake.md`).

### 1.2 Current Token Set

#### x86_64 (5 tokens)

| Token | Aliases | Features | Hardware |
|-------|---------|----------|----------|
| `X64V2Token` | — | sse3, ssse3, sse4.1, sse4.2, popcnt | Nehalem 2008+, Bulldozer 2011+ |
| `X64V3Token` | `Desktop64` | + avx, avx2, fma, bmi1, bmi2, f16c, lzcnt | Haswell 2013+, Zen 1 2017+ |
| `X64V4Token` | `Avx512Token`, `Server64` | + avx512f, avx512bw, avx512cd, avx512dq, avx512vl | Skylake-X 2017+, Zen 4 2022+ |
| `Avx512ModernToken` | — | + avx512vpopcntdq, avx512ifma, avx512vbmi, avx512vbmi2, avx512bitalg, avx512vnni, avx512bf16, vpclmulqdq, gfni, vaes | Ice Lake 2019+, Zen 4 2022+ |
| `Avx512Fp16Token` | — | v4 + avx512fp16 | Sapphire Rapids 2023+ |

Features are cumulative — each token lists ALL features it enables, not just the delta from the previous tier. This eliminates the class of bugs where "minimal" lists diverge from "cumulative" lists. LLVM deduplicates redundant features in `#[target_feature]`.

The v1 level (SSE, SSE2) has no token — it's the x86_64 baseline, always available.

#### AArch64 (4 tokens)

| Token | Aliases | Features | Notes |
|-------|---------|----------|-------|
| `NeonToken` | `Arm64` | neon | Baseline, always available on AArch64 |
| `NeonAesToken` | — | neon, aes | ARMv8-A with Crypto |
| `NeonSha3Token` | — | neon, sha3 | ARMv8.2-A+ |
| `NeonCrcToken` | — | neon, crc | ARMv8.1-A+ (most AArch64 CPUs) |

SVE/SVE2 tokens are prohibited — SVE hasn't shipped in consumer hardware.

#### WASM (1 token)

| Token | Features | Notes |
|-------|----------|-------|
| `Wasm128Token` | simd128 | Compile-time only (`#[cfg(target_feature = "simd128")]`) |

### 1.3 Token Hierarchy

Tokens form a subsumption hierarchy. Higher-tier tokens can produce lower-tier tokens:

```
x86_64:
  Avx512ModernToken → Avx512Token (v4) → X64V3Token (v3) → X64V2Token (v2)
  Avx512Fp16Token   → Avx512Token (v4) → X64V3Token (v3) → X64V2Token (v2)

AArch64:
  NeonAesToken → NeonToken
  NeonSha3Token → NeonToken
  NeonCrcToken → NeonToken
```

Extraction methods: `.v3()`, `.v2()`, `.avx512()`, `.neon()`.

### 1.4 Trait Hierarchy

Traits provide generic bounds. Tokens implement the traits matching their capabilities.

**Width traits:**
```
Has512BitSimd → Has256BitSimd → Has128BitSimd
```

**x86 tier traits:**
```
HasX64V4 → HasX64V2
```

There is no `HasX64V3` trait. For v3 bounds, use `X64V3Token` directly — it's the recommended baseline for high-performance SIMD code.

**AArch64 traits:**
```
HasNeonAes → HasNeon
HasNeonSha3 → HasNeon
```

**Trait implementations by token:**

| Token | Has128Bit | Has256Bit | Has512Bit | HasX64V2 | HasX64V4 | HasNeon | HasNeonAes | HasNeonSha3 |
|-------|-----------|-----------|-----------|----------|----------|---------|------------|-------------|
| X64V2Token | x | | | x | | | | |
| X64V3Token | x | x | | x | | | | |
| X64V4Token | x | x | x | x | x | | | |
| Avx512ModernToken | x | x | x | x | x | | | |
| Avx512Fp16Token | x | x | x | x | x | | | |
| NeonToken | x | | | | | x | | |
| NeonAesToken | x | | | | | x | x | |
| NeonSha3Token | x | | | | | x | | x |
| NeonCrcToken | x | | | | | x | | |
| Wasm128Token | x | | | | | | | |

### 1.5 Cross-Platform Stubs

All token types are defined on all architectures. On unsupported architectures, `summon()` returns `None`. This enables cross-platform code that compiles everywhere but only dispatches on the right arch.

Stub modules: `x86_stubs.rs`, `arm_stubs.rs`, `wasm_stubs.rs`.

## 2. Safety Model

### 2.1 The Soundness Invariant

For every token `T`:

```
target_features_in_macro(T) ⊆ features_checked_in_summon(T)
```

The `#[arcane]` / `#[arcane]` macro reads the token type, looks up features via `token_to_features()`, and generates `#[target_feature(enable = "...")]`. If `summon()` checked fewer features than the macro enables, forging the token would allow calling intrinsics that the CPU doesn't actually support — **undefined behavior** (illegal instructions, crashes, silent data corruption).

The reverse direction (summon checks more than the macro enables) is safe but wasteful.

### 2.2 Rust 1.85+ Intrinsic Safety

As of Rust 1.85 (stabilized `target_feature_11`), **value-based intrinsics are safe inside `#[target_feature]` functions**:

```rust
#[target_feature(enable = "avx2")]
fn example() {
    let a = _mm256_setzero_ps();        // SAFE — value-based
    let b = _mm256_add_ps(a, a);        // SAFE — value-based
    let v = unsafe { _mm256_loadu_ps(ptr) };  // UNSAFE — pointer dereference
}
```

This means archmage doesn't need to wrap most intrinsics. The `#[arcane]` macro generates `#[target_feature]` inner functions; value-based intrinsics are automatically safe inside them.

Only pointer-based operations remain unsafe:
- Loads from `*const T` (`_mm256_loadu_ps`, etc.)
- Stores to `*mut T` (`_mm256_storeu_ps`, etc.)
- Gather/scatter operations
- Prefetch instructions

For safe memory access, use `safe_unaligned_simd` (accepts `&[T]`/`&mut [T]` instead of raw pointers).

### 2.3 How `#[arcane]` Works

```rust
// Input:
#[arcane]
fn kernel(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    let v = _mm256_setzero_ps();  // value-based, safe in target_feature
    // ...
}

// Generated:
fn kernel(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    #[target_feature(enable = "sse3,ssse3,sse4.1,sse4.2,popcnt,avx,avx2,fma,bmi1,bmi2,f16c,lzcnt")]
    unsafe fn inner(data: &[f32; 8]) -> [f32; 8] {
        let v = _mm256_setzero_ps();
        // ...
    }
    // SAFETY: token existence proves CPU support was verified via summon()
    unsafe { inner(data) }
}
```

The macro:
1. Reads the token type from the first parameter
2. Looks up features via `token_to_features()`
3. Generates an inner function with `#[target_feature(enable = "...")]`
4. Calls it unsafely (the token proves safety)
5. Drops the token parameter from the inner function signature

This also works with `impl Trait` bounds, generic parameters, and `_self` for trait methods.

## 3. Intrinsic Safety Classification

The stdarch CSV marks all SIMD intrinsics as `unsafe` without distinguishing WHY they're unsafe. Three categories exist:

### 3.1 Category 1: Pointer Dereference (~525 of 582 unsafe x86 intrinsics)

Takes `*const T` or `*mut T`, dereferences it. Includes:

- **Loads:** `_mm*_loadu_*`, `_mm*_load_*`, `_mm*_lddqu_*`
- **Stores:** `_mm*_storeu_*`, `_mm*_store_*`, `_mm*_stream_*`
- **Gather:** `_mm*_i32gather_*`, `_mm*_i64gather_*`
- **Scatter:** `_mm*_i32scatter_*`, `_mm*_i64scatter_*`
- **Masked:** `_mm*_maskload_*`, `_mm*_maskstore_*`
- **Prefetch:** `_mm_prefetch`

**Wrappable safely** via the `safe_unaligned_simd` pattern: accept `&[T]` or `&[T; N]` instead of `*const T`, validate bounds at the reference level, pass `.as_ptr()` to the intrinsic.

### 3.2 Category 2: Implicit Memory Access (~37 intrinsics)

Reads/writes memory through implicit mechanisms — not through an explicit pointer parameter.

- **AMX tile ops** (~15): `_tile_loadd`, `_tile_stored`, `_tile_dpbuud`, etc. — operate on tile registers that reference memory regions configured via `_tile_loadconfig`.
- **AVX-NE-CONVERT pointer reads** (~12): `_mm*_cvtneps_avx_pbh` variants that read from memory addresses encoded in the instruction.
- **SSE1 legacy pointer ops** (~10): `_mm_loadh_pi`, `_mm_storeh_pi` — legacy `__m64*` pointer operations.

Some of these can be wrapped safely (with careful API design), some cannot.

### 3.3 Category 3: Side Effects / State Mutation (~20 intrinsics)

Modifies CPU state beyond just computing a value.

- **RTM transactional memory** (4): `_xbegin`, `_xend`, `_xabort`, `_xtest` — hardware transactional memory (deprecated on many CPUs).
- **Key Locker crypto** (10): `_mm_aesenc256kl`, `_mm_aesdec256kl`, etc. — hardware-bound encryption with internal key state.
- **XSAVE state save/restore** (4+): `_xsave`, `_xrstor`, `_xsaveopt`, etc. — saves/restores extended processor state.
- **CMPXCHG16B atomic** (1): `_cmpxchg16b` — 128-bit compare-and-swap.

**NOT wrappable** with reference patterns. These intrinsics have inherent side effects that cannot be made safe through API design alone.

### 3.4 Open Question: Safety Annotation

Should the registry or CSV gain a `safety_reason` column (`pointer_deref` | `implicit_memory` | `state_mutation`) to enable:

1. Automatic generation of `safe_unaligned_simd`-style wrappers for Category 1
2. Validation that magetypes never exposes Category 2/3 without explicit `unsafe`
3. Documentation of WHY each intrinsic is unsafe

This would be valuable but requires auditing all ~582 unsafe intrinsics to categorize them.

## 4. Feature Detection

### 4.1 `is_x86_feature_available!` / `is_aarch64_feature_available!`

Archmage provides detection macros that combine compile-time and runtime checks:

1. If the feature is enabled at compile time (via `#[target_feature]`, `-C target-cpu`, or being inside a multiversed function variant), the check is eliminated entirely.
2. Otherwise, falls back to `std::is_x86_feature_detected!` (or architecture equivalent).

This means inside a `#[multiversed]` function compiled for AVX2, `X64V3Token::summon()` compiles to a constant `Some(token)`.

### 4.2 WASM Special Case

WASM SIMD128 uses compile-time detection only (`#[cfg(target_feature = "simd128")]`). There is no runtime detection mechanism.

### 4.3 AArch64 NEON Special Case

NEON is always available on AArch64 (`NeonToken::summon()` always returns `Some`). No runtime check needed.

## 5. QEMU CI Testing

QEMU user-mode emulation can trap SIMD instructions, allowing CI to test non-native feature paths:

- `qemu-x86_64 -cpu Haswell` for x86-64-v3 testing
- `qemu-x86_64 -cpu Skylake-Server-v4` for AVX-512
- `qemu-aarch64 -cpu cortex-a76` for NEON + crypto

This catches feature-detection bugs and instruction encoding errors without native hardware. Tests verify `summon()` returns `Some` and intrinsics execute correctly under emulation.

## 6. Token Registry Consolidation (in progress)

The goal is to replace 10+ independent copies of token-to-feature mappings (in macros, xtask, tests, docs) with a single `token-registry.toml` that everything derives from.

### 6.1 Current State (post Step 1)

- Granular x86 tokens removed (Sse41Token, Avx2Token, FmaToken, etc.)
- ARM composite tokens removed (ArmCryptoToken, ArmCrypto3Token)
- NeonCrcToken added
- Avx2FmaToken deprecated (use X64V3Token or Desktop64 instead)
- CompositeToken trait removed
- Feature lists in `token_to_features()` are complete and correct
- X64V3Token features fixed (was missing sse3, ssse3, sse4.1, sse4.2, popcnt, f16c, lzcnt)

### 6.2 Remaining Steps

- **Step 2:** Create `token-registry.toml` and xtask registry parser
- **Step 3:** Generate `archmage-macros/src/generated_registry.rs` from TOML
- **Step 4:** Wire xtask internals to registry
- **Step 5:** Add `summon()`/`summon()` verification (validate source against registry)
- **Step 7:** Housekeeping (update CLAUDE.md, justfile, bump version)

### 6.3 Registry Schema

The registry uses TOML with these table types:

- `[[token]]` — token definitions with name, arch, aliases, features, traits, optional cargo_feature
- `[[trait]]` — trait definitions with name, features, parents
- `[[width_namespace]]` — width namespace config for simd type re-exports
- `[[magetypes_file]]` — file-to-token validation mappings
- `[[polyfill_w256]]` / `[[polyfill_w512]]` — polyfill platform configs

See the plan file for the full schema.

### 6.4 Validation After Consolidation

`cargo xtask validate` will verify:
1. Generated macro code matches registry
2. `summon()` source checks exactly the registry features
3. Magetypes intrinsic usage is valid under gating tokens
4. Trait hierarchy is consistent (token features ⊇ claimed trait features)
5. Re-running `cargo xtask generate` produces identical output (idempotent)
