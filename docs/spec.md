# Archmage Specification

This document is the authoritative reference for archmage's token architecture, safety model, and intrinsic classification. It captures design decisions and open questions so context survives across sessions.

## 1. Token Architecture

Archmage uses zero-sized **capability tokens** to prove CPU features are available at compile time. A token can only be constructed via:

- `summon()` / `summon()` — runtime feature detection, returns `Option<Self>`
- `forge_token_dangerously()` — unsafe, caller guarantees features are present

Tokens are `Copy + Clone + Send + Sync + 'static`. They carry no data — the type itself is the proof.

### 1.1 Token Registry

`token-registry.toml` is the single source of truth for all token definitions. Everything else — macro feature maps, xtask codegen, validation — derives from it. Run `just generate` to regenerate all derived code, `just validate-registry` to check the registry, and `just validate-tokens` to verify `summon()` checks match the registry.

### 1.2 Current Token Set

#### x86_64 (8 tokens)

| Token | Aliases | Features | Hardware |
|-------|---------|----------|----------|
| `X64V1Token` | `Sse2Token` | sse, sse2 (x86_64 baseline) | All x86_64 CPUs |
| `X64V2Token` | — | + sse3, ssse3, sse4.1, sse4.2, popcnt | Nehalem 2008+, Bulldozer 2011+ |
| `X64CryptoToken` | — | v2 + pclmulqdq, aes | Westmere 2010+, Bulldozer 2011+ |
| `X64V3Token` | — | + avx, avx2, fma, bmi1, bmi2, f16c, lzcnt | Haswell 2013+, Zen 1 2017+ |
| `X64V3CryptoToken` | — | v3 + vpclmulqdq, vaes | Zen 3+ 2020, Alder Lake 2021+ |
| `X64V4Token` | `Avx512Token`, `Server64` | + avx512f, avx512bw, avx512cd, avx512dq, avx512vl | Skylake-X 2017+, Zen 4 2022+ |
| `X64V4xToken` | — | + avx512vpopcntdq, avx512ifma, avx512vbmi, avx512vbmi2, avx512bitalg, avx512vnni, vpclmulqdq, gfni, vaes | Ice Lake 2019+, Zen 4 2022+ |
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

#### WASM (1 token)

| Token | Features | Notes |
|-------|----------|-------|
| `Wasm128Token` | simd128 | Compile-time only (`#[cfg(target_feature = "simd128")]`) |

### 1.3 Token Hierarchy

Tokens form a subsumption hierarchy. Higher-tier tokens can produce lower-tier tokens via `From`/`Into` conversions and extraction methods:

```
x86_64:
  X64V4xToken → X64V4Token (v4) → X64V3Token (v3) → X64V2Token (v2) → X64V1Token (v1)
  Avx512Fp16Token → X64V4Token (v4) → X64V3Token (v3) → X64V2Token (v2) → X64V1Token (v1)
  X64V3CryptoToken → X64V3Token (v3) → X64V2Token (v2) → X64V1Token (v1)
  X64CryptoToken → X64V2Token (v2) → X64V1Token (v1)

AArch64:
  Arm64V3Token → Arm64V2Token → NeonToken
  NeonAesToken → NeonToken
  NeonSha3Token → NeonToken
  NeonCrcToken → NeonToken
```

Extraction methods: `.v1()`, `.v2()`, `.v3()`, `.avx512()`, `.neon()`, `.arm_v2()`. Downcasting is free (zero-cost, same optimization region). Upcasting via `IntoConcreteToken` is safe but creates an LLVM optimization boundary.

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

### 1.5 Cross-Platform Behavior

All token types are defined on all architectures. On unsupported architectures, `summon()` returns `None`. This enables cross-platform code that compiles everywhere but only dispatches on the right arch.

**Cfg-out default:** `#[arcane]` and `#[rite]` only emit code on the matching architecture. On wrong architectures, no function is generated — less dead code, cleaner binaries. Direct *call sites* referencing the function by name must use `#[cfg(target_arch)]` guards, `stub`, or `incant!` (which cfg-gates automatically). No `#[cfg]` is needed on the function *definitions* — the macros handle that.

**Stub opt-in:** `#[arcane(stub)]` and `#[rite(stub)]` generate `unreachable!()` stubs on wrong architectures. Use when cross-arch dispatch references the function without cfg guards. The stub is safe because the token can't be constructed on the wrong architecture.

**`incant!` is unaffected** — it wraps each tier call in `#[cfg(target_arch)]` blocks, so it works correctly with cfg'd-out functions.

Stub modules for token types (not macro-generated functions): `x86_stubs.rs`, `arm_stubs.rs`, `wasm_stubs.rs`.

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

For safe memory access, use `import_intrinsics` which provides reference-based alternatives (accepts `&[T; N]`/`&mut [T; N]` instead of raw pointers).

### 2.3 How `#[arcane]` and `#[rite]` Work

Both macros parse the token type from your function signature to determine which `#[target_feature]` attributes to emit. The token type *is* the feature selector — `X64V3Token` maps to `avx2,fma,...`, `X64V4Token` maps to `avx512f,avx512bw,...`, and so on. This mapping is maintained in `token-registry.toml` and compiled into the proc macro via `token_to_features()`.

Passing the same token type through a call hierarchy means every function gets the same `#[target_feature]` attributes. LLVM sees matching targets and inlines freely — no optimization boundary. When token types mismatch (or generic bounds prevent monomorphization to a concrete type), LLVM hits a target-feature boundary and can't optimize across it, costing 4-6x (see `docs/PERFORMANCE.md`).

#### Sibling Expansion (default)

`#[arcane]` generates two functions at the same scope:

```rust
// Input:
#[arcane(import_intrinsics)]
fn kernel(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    let v = _mm256_setzero_ps();  // In scope from import_intrinsics
    // ...
}

// Generated (x86_64 only — cfg'd out on other architectures):
#[cfg(target_arch = "x86_64")]
#[doc(hidden)]
#[target_feature(enable = "sse3,ssse3,sse4.1,...,avx2,fma,...")]
fn __arcane_kernel(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    let v = _mm256_setzero_ps();
    // ...
}

#[cfg(target_arch = "x86_64")]
fn kernel(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    // SAFETY: token existence proves CPU support was verified via summon()
    unsafe { __arcane_kernel(token, data) }
}
```

Both functions live at the same scope level. For methods, both are in the same `impl` block, so `self`, `Self`, and associated constants resolve naturally.

#### Nested Expansion (`nested` or `_self = Type`)

For trait impls (where sibling would add methods not in the trait definition), use `#[arcane(nested)]` or `#[arcane(_self = Type)]`:

```rust
impl SimdOps for MyType {
    #[arcane(_self = MyType)]
    fn compute(&self, token: X64V3Token) -> f32 {
        _self.data.iter().sum()  // _self replaces self
    }
}

// Generated:
impl SimdOps for MyType {
    fn compute(&self, token: X64V3Token) -> f32 {
        #[target_feature(enable = "...")]
        #[inline]
        fn __inner(_self: &MyType, token: X64V3Token) -> f32 {
            _self.data.iter().sum()
        }
        unsafe { __inner(self, token) }
    }
}
```

#### Options

| Option | Effect |
|--------|--------|
| `#[arcane]` | Sibling expansion, cfg-out on wrong arch |
| `#[arcane(stub)]` | Sibling expansion, unreachable stub on wrong arch |
| `#[arcane(nested)]` | Nested inner function |
| `#[arcane(_self = Type)]` | Implies nested, replaces `self`→`_self` |
| `#[arcane(nested, stub)]` | Nested + stub |
| `#[arcane(inline_always)]` | Force `#[inline(always)]` (nightly only) |

#### `#[rite]`

`#[rite]` applies `#[target_feature]` + `#[inline]` directly to the function, with no wrapper. When the caller already has matching features, LLVM inlines freely — no boundary. **`#[rite]` should be the default.** Use `#[arcane]` only at entry points (the first call from non-SIMD code), and `#[rite]` for everything called from within SIMD code.

`#[rite]` works in three modes:

1. **Token-based** (`#[rite]`): reads the token type from the function signature
2. **Tier-based** (`#[rite(v3)]`): specifies features via tier name, no token parameter needed
3. **Multi-tier** (`#[rite(v3, v4, neon)]`): generates a suffixed variant for each tier (`fn_v3`, `fn_v4`, `fn_neon`), each with its own `#[target_feature]` and `#[cfg(target_arch)]`

Token-based and tier-based produce identical output. Multi-tier generates one function per tier. Since Rust 1.85+, all variants are safe to call from matching `#[arcane]` or `#[rite]` contexts.

`#[rite(stub)]` generates an unreachable stub on wrong architectures (default: cfg-out).

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

**Wrappable safely** via reference-based wrappers: accept `&[T; N]` instead of `*const T`, validate bounds at the reference level, pass `.as_ptr()` to the intrinsic. These are provided by `import_intrinsics`.

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

1. Automatic generation of reference-based safe wrappers for Category 1
2. Validation that magetypes never exposes Category 2/3 without explicit `unsafe`
3. Documentation of WHY each intrinsic is unsafe

This would be valuable but requires auditing all ~582 unsafe intrinsics to categorize them.

## 4. Feature Detection

### 4.1 Token Detection

Token detection combines compile-time and runtime checks:

1. `compiled_with()` — returns `Some(true)` if the feature is enabled at compile time (via `#[target_feature]`, `-Ctarget-cpu`, or being inside an `#[arcane]`/`#[rite]` function). Returns `None` if unknown at compile time.
2. `summon()` — returns `Some(token)` if the CPU supports the required features. Uses atomic caching for fast repeated calls (~1.3 ns). When `compiled_with()` returns `Some(true)`, `summon()` compiles away entirely.

This means inside an `#[arcane]` function compiled for AVX2, `X64V3Token::summon()` compiles to a constant `Some(token)`.

### 4.2 WASM Special Case

WASM SIMD128 uses compile-time detection only (`#[cfg(target_feature = "simd128")]`). There is no runtime detection mechanism.

### 4.3 AArch64 NEON Detection

NEON is virtually universal on AArch64, but `NeonToken::summon()` performs runtime detection (not a constant `Some`). This is because some AArch64 Linux kernels can disable NEON via `HWCAP` flags, and the detection mechanism is architecture-specific:

- **Linux/Android:** Uses `getauxval(AT_HWCAP)` via `std::arch::is_aarch64_feature_detected!`
- **macOS:** Uses `sysctlbyname` — with Apple vendor fallback for features in the Apple target spec baseline
- **Windows:** Uses `IsProcessorFeaturePresent` — limited feature coverage (see Known Platform Detection Issues in CLAUDE.md)

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

- `[[token]]` — token definitions with name, arch, aliases, features, traits, cargo_feature, display_name, short_name, parent, extraction_aliases, doc
- `[[trait]]` — trait definitions with name, features, parents
- `[[width_namespace]]` — width namespace config for simd type re-exports
- `[[magetypes_file]]` — file-to-token validation mappings
- `[[polyfill_w256]]` / `[[polyfill_w512]]` — polyfill platform configs

### 6.2 Validation

`cargo xtask validate` verifies:
1. Generated macro code matches registry
2. `summon()` source checks exactly the registry features
3. Magetypes intrinsic usage is valid under gating tokens
4. Trait hierarchy is consistent (token features ⊇ claimed trait features)
5. Re-running `cargo xtask generate` produces identical output (idempotent)

`cargo xtask parity` checks API surface parity across x86/ARM/WASM (currently 0 issues).
