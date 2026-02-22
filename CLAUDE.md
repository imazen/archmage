# archmage

> Safely invoke your intrinsic power, using the tokens granted to you by the CPU. Cast primitive magics faster than any mage alive.

## CRITICAL: Naming Conventions

**Use the thematic names, not the boring ones:**

| ❌ Don't use | ✅ Use instead | Notes |
|-------------|----------------|-------|
| `#[simd_fn]` | `#[arcane]` | `simd_fn` exists only for migration |
| `try_new()` | `summon()` | `try_new` exists only for migration |

**We are mages, not bureaucrats.** Write `Token::summon()`, not `Token::try_new()`.

### Descriptive Aliases (for AI-assisted coding)

These aliases exist so AI tools can infer behavior from the name. **Prefer the thematic names** in hand-written code, but accept both in reviews and docs.

| Thematic | Descriptive Alias | What it does |
|----------|------------------|--------------|
| `#[arcane]` | `#[token_target_features_boundary]` | Generates safe `#[target_feature]` wrapper (entry point) |
| `#[rite]` | `#[token_target_features]` | Adds `#[target_feature]` + `#[inline]` directly (internal helper) |
| `incant!` | `dispatch_variant!` | Runtime dispatch to architecture-specific variants |

## Reference: CPU Features, Detection, and Dispatch

### The Core Distinction: Compile-Time vs Runtime

| Mechanism | When | Effect |
|-----------|------|--------|
| `#[cfg(target_arch = "...")]` | Compile | Include/exclude code from binary |
| `#[cfg(target_feature = "...")]` | Compile | True only if feature is in target spec |
| `#[cfg(feature = "...")]` | Compile | Cargo feature flag |
| `-Ctarget-cpu=native` | Compile | LLVM assumes current CPU's features |
| `is_x86_feature_detected!()` | Runtime | CPUID instruction |
| `Token::summon()` | Runtime | Archmage's detection (compiles away when guaranteed) |

**Tokens exist everywhere.** `Desktop64`, `Arm64`, etc. compile on all platforms—`summon()` just returns `None` on unsupported architectures. This means **you rarely need `#[cfg(target_arch)]` guards** in user code. The stubs handle cross-compilation cleanly.

### CRITICAL: How the Macros Choose Features

`#[arcane]` and `#[rite]` parse the token type from your function signature to determine which `#[target_feature]` attributes to emit. A function taking `Desktop64` gets `#[target_feature(enable = "avx2,fma,...")]`. A function taking `X64V4Token` gets AVX-512 features. The token type *is* the feature selector.

`#[arcane]` generates a wrapper: an outer function that calls an inner `#[target_feature]` function via `unsafe`. This wrapper is how you cross into SIMD code without writing `unsafe` yourself — but it also creates an LLVM optimization boundary. `#[rite]` applies `#[target_feature]` + `#[inline]` directly, with no wrapper and no boundary.

**`#[rite]` should be the default.** Use `#[arcane]` only at the entry point, and `#[rite]` for everything called from within SIMD code. Passing the same token type through your call hierarchy keeps every function compiled with matching features, so LLVM inlines freely.

### CRITICAL: Target-Feature Boundaries (4x Performance Impact)

**Enter `#[arcane]` once at the top, use `#[rite]` for everything inside.**

LLVM cannot inline across mismatched `#[target_feature]` attributes. Each `#[arcane]` call from non-SIMD code creates an optimization boundary — LLVM can't hoist loads, sink stores, or vectorize across it. This costs 4-6x depending on workload (see `benches/asm_inspection.rs` and `docs/PERFORMANCE.md`). Token hoisting doesn't help — even with the token pre-summoned, calling `#[arcane]` per iteration still hits the boundary.

```rust
// WRONG: #[arcane] boundary every iteration (4x slower)
#[arcane]
fn dist_simd(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> f32 { ... }

fn process_all(points: &[[f32; 8]]) {
    let token = X64V3Token::summon().unwrap(); // hoisted — doesn't help!
    for i in 0..points.len() {
        for j in i+1..points.len() {
            dist_simd(token, &points[i], &points[j]); // boundary per call
        }
    }
}
```

```rust
// RIGHT: one #[arcane] entry, #[rite] helpers inline freely
fn process_all(points: &[[f32; 8]]) {
    if let Some(token) = X64V3Token::summon() {
        process_all_simd(token, points);
    } else {
        process_all_scalar(points);
    }
}

#[arcane]
fn process_all_simd(token: X64V3Token, points: &[[f32; 8]]) {
    for i in 0..points.len() {
        for j in i+1..points.len() {
            dist_simd(token, &points[i], &points[j]); // #[rite] inlines here
        }
    }
}

#[rite]
fn dist_simd(token: X64V3Token, a: &[f32; 8], b: &[f32; 8]) -> f32 {
    // Inlines into process_all_simd — same LLVM optimization region
    ...
}
```

**The rule:** `#[arcane]` at the entry point, `#[rite]` for everything called from SIMD code.

### CRITICAL: Generic Bounds Are Optimization Barriers

**Generic passthrough with trait bounds breaks inlining.** The compiler cannot inline across generic boundaries — each trait-bounded call is a potential indirect call.

```rust
// BAD: Generic bound prevents inlining into caller
#[arcane]
fn process_generic<T: HasX64V2>(token: T, data: &[f32]) -> f32 {
    inner_work(token, data)  // Can't inline — T could be any type
}

#[arcane]
fn inner_work<T: HasX64V2>(token: T, data: &[f32]) -> f32 {
    // Even with #[inline(always)], this may not inline through generic
    ...
}
```

```rust
// GOOD: Concrete token enables full inlining
#[arcane]
fn process_concrete(token: X64V3Token, data: &[f32]) -> f32 {
    inner_work(token, data)  // Fully inlinable — concrete type
}

#[arcane]
fn inner_work(token: X64V3Token, data: &[f32]) -> f32 {
    // Inlines into caller, single #[target_feature] region
    ...
}
```

**Why this matters:**
- `#[target_feature]` functions inline to share the feature-enabled region
- Generic bounds break this chain — each function is a separate compilation unit
- Even `#[inline(always)]` can't force inlining across trait object boundaries

**Downcasting is free:** Pass a higher token to a function expecting a lower one. Nested `#[arcane]` with downcasting preserves the inlining chain:

```rust
#[arcane]
fn v4_kernel(token: X64V4Token, data: &mut [f32]) {
    // Can call V3 functions — V4 is a superset
    let partial = v3_helper(token, &data[..8]);  // Downcasts, still inlines
    // ... AVX-512 specific work ...
}

#[arcane]
fn v3_helper(token: X64V3Token, chunk: &[f32]) -> f32 {
    // AVX2+FMA work — inlines into v4_kernel
    ...
}
```

**Upcasting via `IntoConcreteToken`:** Safe, but creates an LLVM optimization boundary:

```rust
fn process<T: IntoConcreteToken>(token: T, data: &mut [f32]) {
    // Generic caller has baseline LLVM target
    if let Some(v4) = token.as_x64v4() {
        process_v4(v4, data);  // Callee has AVX-512 target — mismatched
    } else if let Some(v3) = token.as_x64v3() {
        process_v3(v3, data);
    }
}
```

The issue: `#[target_feature]` changes LLVM's target for that function. Generic caller and feature-enabled callee have mismatched targets, so LLVM can't optimize across that boundary. Do dispatch once at entry, not deep in hot code.

**The rule:** Use concrete tokens for hot paths. Downcasting (V4→V3) is free. Upcasting via `IntoConcreteToken` is safe but creates optimization boundaries.

### When `-Ctarget-cpu=native` Is Fine

**Use it when:** Building for your own machine or known deployment target.

```bash
RUSTFLAGS="-Ctarget-cpu=native" cargo build --release
```

**Detection compiles away:** With `-Ctarget-cpu=haswell`, `X64V3Token::compiled_with()` returns `Some(true)` and `summon()` becomes a no-op. The compiler elides the check entirely.

**Don't use it when:** Distributing binaries to unknown CPUs.

### How `#[cfg(target_feature)]` Actually Works

```rust
// TRUE only if compiled with -Ctarget-cpu=haswell or -Ctarget-feature=+avx2
#[cfg(target_feature = "avx2")]
fn only_with_avx2_target() { }

// ALWAYS true on x86_64 (baseline)
#[cfg(target_feature = "sse2")]
fn always_on_x86_64() { }
```

Default `x86_64-unknown-linux-gnu` only enables SSE/SSE2. Extended features require `-Ctarget-cpu` or `-Ctarget-feature`.

### The Cargo Feature Trap

**WRONG:** Gating type aliases on cargo features:

```rust
// BAD: Types don't exist unless cargo feature enabled!
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub use crate::simd::x86::w512::f32x16 as F32Vec;
```

This breaks runtime dispatch — types aren't available even if CPU supports AVX-512.

**Cargo features should control:**
- Whether to *attempt* higher tiers at runtime
- Compile-time-only paths for known targets

**Cargo features should NOT control:**
- Whether SIMD types exist

### `#[arcane]`: Cross-Arch Compilation

On wrong architecture, generates unreachable stub:

```rust
// On ARM: stub that compiles but can't be reached
#[cfg(not(target_arch = "x86_64"))]
fn process(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    unreachable!("X64V3Token cannot exist on this architecture")
}
```

### `#[arcane]` with Methods

Use `_self = Type` and reference `_self` in body:

```rust
impl Processor {
    #[arcane(_self = Processor)]
    fn process(&self, token: X64V3Token, data: &[f32; 8]) -> f32 {
        _self.threshold  // Use _self, not self
    }
}
```

### `incant!`: Dispatch Macro

```rust
use archmage::incant;

pub fn sum(data: &[f32]) -> f32 {
    incant!(sum(data))
}

// Default suffixed functions:
// sum_v3(token: X64V3Token, ...)
// sum_v4(token: X64V4Token, ...)     // if feature = "avx512"
// sum_neon(token: NeonToken, ...)
// sum_wasm128(token: Wasm128Token, ...)
// sum_scalar(token: ScalarToken, ...)
```

**Explicit tiers** (only dispatch to these):

```rust
pub fn sum(data: &[f32]) -> f32 {
    incant!(sum(data), [v1, v3, neon])
}
// Requires: sum_v1, sum_v3, sum_neon, sum_scalar
```

Known tiers: `v1`, `v2`, `x64_crypto`, `v3`, `v3_crypto`, `v4`, `v4x`, `arm_v2`, `arm_v3`,
`neon`, `neon_aes`, `neon_sha3`, `neon_crc`, `wasm128`, `scalar`.
Scalar is always implicit.

**Passthrough mode** (already have token):

```rust
fn inner<T: IntoConcreteToken>(token: T, data: &[f32]) -> f32 {
    incant!(process(data) with token)
}

// With explicit tiers:
fn inner<T: IntoConcreteToken>(token: T, data: &[f32]) -> f32 {
    incant!(process(data) with token, [v3, neon])
}
```

### `ScalarToken`

Always-available fallback. Used for:
- `incant!()` convention (`_scalar` suffix)
- Consistent API shape in dispatch

### Fixed-Size Types with Polyfills

**Pick a concrete size. Use polyfills for portability.**

```rust
use magetypes::simd::f32x8;  // Always 8 lanes, polyfilled on ARM/WASM

#[arcane]
fn process(token: X64V3Token, data: &[f32; 8]) -> f32 {
    let v = f32x8::load(token, data);
    v.reduce_add()
}
```

On ARM, `f32x8` is emulated with two `f32x4` operations. The API is identical.

---

## CRITICAL: Token/Trait Design (DO NOT MODIFY)

### LLVM x86-64 Microarchitecture Levels

| Level | Features | Token | Trait |
|-------|----------|-------|-------|
| **v1** | SSE, SSE2 (baseline) | `X64V1Token` / `Sse2Token` | None (always available on x86_64) |
| **v2** | + SSE3, SSSE3, SSE4.1, SSE4.2, POPCNT | `X64V2Token` | `HasX64V2` |
| **crypto** | v2 + PCLMULQDQ, AES-NI | `X64CryptoToken` | Use token directly |
| **v3** | + AVX, AVX2, FMA, BMI1, BMI2, F16C | `X64V3Token` / `Desktop64` | Use token directly |
| **v3_crypto** | v3 + VPCLMULQDQ, VAES | `X64V3CryptoToken` | Use token directly |
| **v4** | + AVX512F, AVX512BW, AVX512CD, AVX512DQ, AVX512VL | `X64V4Token` / `Avx512Token` | `HasX64V4` |
| **V4x** | + VPOPCNTDQ, IFMA, VBMI, VNNI, VBMI2, BITALG, VPCLMULQDQ, GFNI, VAES | `X64V4xToken` | Use token directly |
| **FP16** | AVX512FP16 (independent) | `Avx512Fp16Token` | Use token directly |

### AArch64 Tokens

| Token | Features | Trait |
|-------|----------|-------|
| `NeonToken` / `Arm64` | neon | `HasNeon` |
| `Arm64V2Token` | + crc, rdm, dotprod, fp16, aes, sha2 | `HasArm64V2` |
| `Arm64V3Token` | + fhm, fcma, sha3, i8mm, bf16 | `HasArm64V3` |
| `NeonAesToken` | neon + aes | `HasNeonAes` |
| `NeonSha3Token` | neon + sha3 | `HasNeonSha3` |
| `NeonCrcToken` | neon + crc | Use token directly |

**Arm64 compute tiers** (archmage-defined, not ARM architecture versions):
- **Arm64-v2**: Broadest modern ARM baseline. Cortex-A55+, Apple M1+, Graviton 2+, all post-2017 ARM chips.
- **Arm64-v3**: Full modern feature set. Cortex-A510+, Apple M2+, Snapdragon X, Graviton 3+, Cobalt 100.

The M1 is the notable chip that gets V2 but not V3 (lacks i8mm/bf16). Budget 2025 phones with Cortex-A55 LITTLE cores also max out at V2 (A55 lacks fhm/fcma/i8mm/bf16).

**PROHIBITED:** NO SVE/SVE2 - Rust stable doesn't support SVE intrinsics yet.

### Rules

1. **NO granular x86 traits** - No `HasSse`, `HasSse2`, `HasAvx`, `HasAvx2`, `HasFma`, `HasAvx512f`, `HasAvx512bw`, etc.
2. **Use tier tokens** - `X64V2Token`, `X64V3Token`, `X64V4Token`, `X64V4xToken`
3. **Single trait per tier** - `HasX64V2`, `HasX64V4`, `HasArm64V2`, `HasArm64V3` only
4. **NO SVE** - `SveToken`, `Sve2Token`, `HasSve`, `HasSve2` are PROHIBITED (Rust stable lacks SVE support)
5. **NO WIDTH TRAITS** - `Has128BitSimd`, `Has256BitSimd`, `Has512BitSimd` are DEPRECATED and will be removed:
   - `Has256BitSimd` only enables AVX, **NOT AVX2 or FMA** — misleading and causes suboptimal codegen
   - Use concrete tokens (`X64V3Token`) or feature traits (`HasX64V2`, `HasX64V4`) instead

---

## CRITICAL: Documentation Examples

### Prefer `#[rite]` for internal code, `#[arcane]` only at entry points

**`#[rite]` should be the default.** It adds `#[target_feature]` + `#[inline]` — LLVM inlines it into any caller with matching features.

Use `#[arcane]` only when the function is called from non-SIMD code:
- After `summon()` in a public API
- From tests
- From non-`#[target_feature]` contexts

```rust
// Entry point (called after summon) - use #[arcane]
#[arcane]
pub fn process(token: Desktop64, data: &mut [f32]) {
    for chunk in data.chunks_exact_mut(8) {
        process_chunk(token, chunk);  // Calls #[rite]
    }
}

// Internal helper (called from #[arcane] or #[rite]) - use #[rite]
#[rite]
fn process_chunk(_: Desktop64, chunk: &mut [f32; 8]) {
    // ...
}
```

### Never use manual `#[target_feature]`

**DO NOT write examples with manual `#[target_feature]` + unsafe wrappers.** Use `#[arcane]` or `#[rite]` instead.

```rust
// WRONG - manual #[target_feature] wrapping
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn process_inner(data: &[f32]) -> f32 { ... }

#[cfg(target_arch = "x86_64")]
fn process(token: X64V3Token, data: &[f32]) -> f32 {
    unsafe { process_inner(data) }
}

// CORRECT - use #[arcane] (generates #[target_feature] + stubs on other arches)
#[arcane]
fn process(token: X64V3Token, data: &[f32]) -> f32 {
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
just generate                 # Regenerate all generated code
just validate-registry        # Validate token-registry.toml
just validate-tokens          # Validate magetypes safety + summon() checks
just parity                   # Check API parity across x86/ARM/WASM
just soundness                # Static intrinsic soundness verification
just miri                     # Run magetypes under Miri (detects UB)
just audit                    # Scan for safety-critical code
just intrinsics-refresh       # Re-extract intrinsics from current Rust
just ci                       # Run ALL checks (must pass before push/publish)
```

## CI and Publishing Rules

**ABSOLUTE REQUIREMENT: Run `just ci` (or `just all` or `cargo xtask all`) before ANY push or publish.**

```bash
just ci    # or: just all, cargo xtask ci, cargo xtask all
```

**NEVER run `git push` or `cargo publish` until this passes. No exceptions.**

CI checks (all must pass):
1. `cargo xtask generate` — regenerate all code
2. **Clean worktree check** — no uncommitted changes after generation (HARD FAIL)
3. `cargo xtask validate` — intrinsic safety + summon() feature verification
4. `cargo xtask parity` — parity check (0 issues remaining)
5. `cargo clippy --features "std macros avx512"` — zero warnings
6. `cargo test --features "std macros avx512"` — all tests pass
7. `cargo fmt --check` — code is formatted

**Note:** Parity check reports 0 issues. All W128 types have identical APIs across x86/ARM/WASM.

If ANY check fails:
- Do NOT push
- Do NOT publish
- Fix the issue first
- Re-run `just ci` until it passes

**Git tags are MANDATORY for every publish.** After `cargo publish`, immediately create tags:

```bash
git tag v{version}                        # archmage
git tag archmage-macros-v{version}        # archmage-macros
git tag magetypes-v{version}              # magetypes
git push origin v{version} archmage-macros-v{version} magetypes-v{version}
```

Publish order (respect dependency chain): `archmage-macros` → `archmage` → `magetypes`.

## Source of Truth: token-registry.toml

All token definitions, feature sets, trait mappings, and width configurations
live in `token-registry.toml`. Everything else is derived:

- `src/tokens/generated/` — token structs, traits, stubs, generated by xtask
- `archmage-macros/src/generated/` — macro registry, generated by xtask
- `magetypes/src/simd/generated/` — SIMD types, generated by xtask
- `docs/generated/` — intrinsics reference docs, generated by xtask
- `xtask/src/main.rs` validation — reads registry at runtime
- `cargo xtask validate` — verifies summon() checks match registry
- `cargo xtask parity` — checks API parity across architectures

To add/modify tokens: edit `token-registry.toml`, then `just generate`.

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

**`#![forbid(unsafe_code)]` compatible**: Downstream crates can use `#![forbid(unsafe_code)]` when combining archmage tokens + `#[arcane]`/`#[rite]` macros + `safe_unaligned_simd` for memory operations.

## How `#[arcane]` Works

The macro generates an inner function with `#[target_feature]`:

```rust
// You write:
#[arcane]
fn my_kernel(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    let v = _mm256_setzero_ps();  // Safe!
    // ...
}

// Macro generates:
fn my_kernel(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    #[target_feature(enable = "avx2,fma")]
    fn inner(data: &[f32; 8]) -> [f32; 8] {
        let v = _mm256_setzero_ps();  // Safe inside #[target_feature]!
        // ...
    }
    // SAFETY: Calling #[target_feature] fn from non-matching context.
    // Token proves CPU support was verified via summon().
    unsafe { inner(data) }
}
```

## Friendly Aliases

| Alias | Token | What it means |
|-------|-------|---------------|
| `Sse2Token` | `X64V1Token` | SSE + SSE2 (x86_64 baseline — always available) |
| `Desktop64` | `X64V3Token` | AVX2 + FMA (Haswell 2013+, Zen 1+) |
| `Server64` | `X64V4Token` | + AVX-512 (Xeon 2017+, Zen 4+) |
| `Arm64` | `NeonToken` | NEON (all 64-bit ARM) |

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
token-registry.toml          # THE source of truth for all token/trait/feature data
spec.md                      # Architecture spec and safety model documentation
archmage/                    # Main crate: tokens, macros, detect
├── src/
│   ├── lib.rs              # Main exports
│   ├── tokens/             # SIMD capability tokens
│   │   ├── mod.rs          # SimdToken trait definition only
│   │   └── generated/      # Generated from token-registry.toml
│   │       ├── mod.rs      # cfg-gated module routing + re-exports
│   │       ├── traits.rs   # Marker traits (Has128BitSimd, HasX64V2, etc.)
│   │       ├── x86.rs      # x86 tokens (v2, v3) + detection
│   │       ├── x86_avx512.rs  # AVX-512 tokens (v4, v4x, fp16)
│   │       ├── arm.rs      # ARM tokens + detection
│   │       ├── wasm.rs     # WASM tokens + detection
│   │       ├── x86_stubs.rs   # x86 stubs (summon → None)
│   │       ├── arm_stubs.rs   # ARM stubs
│   │       └── wasm_stubs.rs  # WASM stubs
archmage-macros/             # Proc-macro crate (#[arcane], #[rite], #[magetypes], incant!)
└── src/
    ├── lib.rs              # Macro implementation
    └── generated/          # Generated from token-registry.toml
        ├── mod.rs          # Re-exports
        └── registry.rs     # Token→features mappings
magetypes/                   # SIMD types crate (depends on archmage)
├── src/
│   ├── lib.rs              # Exports simd module
│   └── simd/
│       ├── mod.rs          # Re-exports from generated/
│       └── generated/      # Auto-generated SIMD types
│           ├── x86/        # x86-64 types (w128, w256, w512)
│           ├── arm/        # AArch64 types (w128)
│           ├── wasm/       # WASM types (w128)
│           └── polyfill.rs # Width emulation
docs/
└── generated/              # Auto-generated reference docs
    ├── x86-intrinsics-by-token.md
    ├── aarch64-intrinsics-by-token.md
    └── memory-ops-reference.md
xtask/                       # Code generator and validation
└── src/
    ├── main.rs             # Generates everything, validates safety, parity check
    ├── registry.rs         # token-registry.toml parser
    └── token_gen.rs        # Token/trait code generator
```

## CRITICAL: Codegen Style Rules — NO `writeln!` CHAINS

**THIS IS MANDATORY. ALL codegen MUST use `formatdoc!` from the `indoc` crate.**

`writeln!` chains are the single biggest readability problem in our codegen. They turn 10 lines of readable Rust into 40 lines of string-escaping noise where you can't see the generated code's structure. Every `{{` and `}}` is a bug waiting to happen. Every `.unwrap()` is visual clutter. Stop it.

### The rule

1. **Use `formatdoc!` with raw strings** for any block of generated code (2+ lines)
2. **Use `format!` or string literals** for single-line fragments only
3. **`writeln!` is BANNED** except for trivial single-line output to stdout/stderr (like progress messages)
4. **`indoc` is already a dependency** of xtask — there is zero excuse

### Pattern: `formatdoc!` with `push_str`

```rust
use indoc::formatdoc;

// WRONG — unreadable writeln! soup (actual current state of token_gen.rs)
writeln!(code, "impl SimdToken for {name} {{").unwrap();
writeln!(code, "    const NAME: &'static str = \"{display_name}\";").unwrap();
writeln!(code, "").unwrap();
writeln!(code, "    fn compiled_with() -> Option<bool> {{").unwrap();
writeln!(code, "        #[cfg(all({cfg_all}))]").unwrap();
writeln!(code, "        {{ Some(true) }}").unwrap();
writeln!(code, "        #[cfg(not(all({cfg_all})))]").unwrap();
writeln!(code, "        {{ None }}").unwrap();
writeln!(code, "    }}").unwrap();
writeln!(code, "}}").unwrap();

// CORRECT — you can actually READ the generated code
code.push_str(&formatdoc! {r#"
    impl SimdToken for {name} {{
        const NAME: &'static str = "{display_name}";

        fn compiled_with() -> Option<bool> {{
            #[cfg(all({cfg_all}))]
            {{ Some(true) }}
            #[cfg(not(all({cfg_all})))]
            {{ None }}
        }}
    }}
"#});
```

### Pattern: conditional blocks

```rust
// Build a section conditionally, then splice it in
let cascade_code = if !descendants.is_empty() {
    let mut lines = String::new();
    for desc in &descendants {
        lines.push_str(&formatdoc! {r#"
            super::{module}::{cache}.store(v, Ordering::Relaxed);
            super::{module}::{disabled}.store(disabled, Ordering::Relaxed);
        "#, module = desc.module, cache = desc.cache_var, disabled = desc.disabled_var});
    }
    lines
} else {
    String::new()
};

code.push_str(&formatdoc! {r#"
    pub fn disable(disabled: bool) {{
        CACHE.store(if disabled {{ 1 }} else {{ 0 }}, Ordering::Relaxed);
        {cascade_code}
    }}
"#});
```

### Pattern: helper functions that return String

```rust
fn gen_compiled_with(name: &str, cfg_all: &str) -> String {
    formatdoc! {r#"
        fn compiled_with() -> Option<bool> {{
            #[cfg(all({cfg_all}))]
            {{ Some(true) }}
            #[cfg(not(all({cfg_all})))]
            {{ None }}
        }}
    "#}
}
```

### For magetypes method generation

Use the helpers in `xtask/src/simd_types/types.rs`:

```rust
use super::types::{gen_unary_method, gen_binary_method, gen_scalar_method};

code.push_str(&gen_unary_method("Compute absolute value", "abs", "Self(_mm256_abs_epi32(self.0))"));
code.push_str(&gen_binary_method("Add two vectors", "add", "Self(_mm256_add_epi32(self.0, other.0))"));
code.push_str(&gen_scalar_method("Extract first element", "first", "i32", "_mm_cvtsi128_si32(self.0)"));
```

### Enforcement

When touching ANY codegen file, convert `writeln!` chains to `formatdoc!` in the same commit. Don't add new `writeln!` chains. Existing `writeln!` chains in `token_gen.rs` (297 occurrences!) and `main.rs` (33 occurrences) are tech debt — convert them progressively.

## Token Hierarchy

**x86:**
- `X64V1Token` / `Sse2Token` - SSE + SSE2 (x86_64 baseline — always available)
- `X64V2Token` - SSE4.2 + POPCNT (Nehalem 2008+)
- `X64CryptoToken` - V2 + PCLMULQDQ + AES-NI (Westmere 2010+)
- `X64V3Token` / `Desktop64` - AVX2 + FMA + BMI2 (Haswell 2013+, Zen 1+)
- `X64V3CryptoToken` - V3 + VPCLMULQDQ + VAES (Zen 3+ 2020, Alder Lake 2021+)
- `X64V4Token` / `Avx512Token` - + AVX-512 F/BW/CD/DQ/VL (Skylake-X 2017+, Zen 4+)
- `X64V4xToken` - + modern extensions (Ice Lake 2019+, Zen 4+)
- `Avx512Fp16Token` - + FP16 (Sapphire Rapids 2023+)

**ARM (compute tiers):**
- `NeonToken` / `Arm64` - NEON (virtually all AArch64, requires runtime detection)
- `Arm64V2Token` - + CRC, RDM, DotProd, FP16, AES, SHA2 (A55+, M1+, Graviton 2+)
- `Arm64V3Token` - + FHM, FCMA, SHA3, I8MM, BF16 (A510+, M2+, Snapdragon X, Graviton 3+)

**ARM (crypto leaves):**
- `NeonAesToken` - neon + AES
- `NeonSha3Token` - neon + SHA3
- `NeonCrcToken` - neon + CRC

## Tier Traits

```rust
fn requires_v2(token: impl HasX64V2) { ... }
fn requires_v4(token: impl HasX64V4) { ... }
fn requires_neon(token: impl HasNeon) { ... }
fn requires_arm_v2(token: impl HasArm64V2) { ... }
fn requires_arm_v3(token: impl HasArm64V3) { ... }
```

For x86 v3 (AVX2+FMA), use `X64V3Token` directly - it's the recommended baseline.

## SIMD Types (magetypes crate)

Token-gated SIMD types live in the **magetypes** crate. **Use fixed-size types:**

```rust
use archmage::{X64V3Token, SimdToken, arcane};
use magetypes::simd::f32x8;

pub fn process(data: &[f32; 8]) -> f32 {
    if let Some(token) = X64V3Token::summon() {
        process_simd(token, data)
    } else {
        data.iter().sum()
    }
}

#[arcane]
fn process_simd(token: X64V3Token, data: &[f32; 8]) -> f32 {
    let a = f32x8::load(token, data);
    let b = f32x8::splat(token, 2.0);
    let c = a * b;
    c.reduce_add()
}
```

On ARM/WASM, `f32x8` is polyfilled with two `f32x4` operations. Pick the size that fits your algorithm.

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

## Pending Work

### API Parity Status (0 issues — complete!)

**Current state:** All W128 types have identical APIs across x86/ARM/WASM. Reduced from 270 → 0 parity issues (100%).

Run `cargo xtask parity` to verify.

### Known Cross-Architecture Behavioral Differences

These are documented semantic differences between architectures. Tests must account for them; they are not bugs to fix.

| Issue | x86 | ARM | WASM | Workaround |
|-------|-----|-----|------|------------|
| Bitwise operators (`&`, `\|`, `^`) on integers | Trait impls (operators work) | Methods only | Methods only | Use `.and()`, `.or()`, `.xor()` methods |
| `shr` for signed integers | Logical (zero-fill) | Arithmetic (sign-extend) | Arithmetic (sign-extend) | Use `shr_arithmetic` for portable sign-extending shift |
| `blend` signature | `(mask, true, false)` | `(mask, true, false)` | `(self, other, mask)` | Avoid in portable code; use bitcast + comparison verification |
| `interleave_lo/hi` | f32x4 only | f32x4 only | f32x4 only | Only use on f32x4, not integer types |

### Long-Term

- **Generator test fixtures**: Add example input/expected output pairs to each xtask generator (SIMD types, width dispatch, tokens, macro registry). These serve as both documentation of expected output and cross-platform regression tests — run on x86, ARM, and WASM to catch codegen divergence.

- ~~**Target-feature boundary overhead benchmark**~~: Done. See `benches/asm_inspection.rs` and `docs/PERFORMANCE.md`. Key results:
  - Simple vector add (1000 x 8-float): `#[rite]` in `#[arcane]` 547 ns, `#[arcane]` per iteration 2209 ns (4x), bare `#[target_feature]` 2222 ns (4x, identical)
  - DCT-8 (100 rows x 8 dot products): `#[rite]` in `#[arcane]` 61 ns, `#[arcane]` per row 376 ns (6.2x), bare `#[target_feature]` 374 ns (6.2x, identical)
  - Cross-token nesting: downgrade (V4→V3, V3→V2) is free, upgrade (V2→V3, V3→V4) costs 4x, all patterns match bare `#[target_feature]`

  Key insight: the overhead is from the `#[target_feature]` optimization boundary, NOT from wrappers or archmage abstractions. The cost scales with computational density (4x simple add, 6.2x DCT-8). Feature direction matters: downgrades are free (superset enables inlining), upgrades hit the boundary.

- ~~**summon() caching**~~: **Implemented!** See `benches/summon_overhead.rs`. Results after adding atomic caching:
  - `Desktop64::summon()` (cached): ~1.3 ns (was 2.6 ns — **2x faster**)
  - `X64V4xToken::summon()` (cached): ~1.3 ns (was 7.2 ns — **6x faster**)
  - With `-Ctarget-cpu=haswell`: 0 ns (compiles away entirely)

  Implementation: Each token has a static `AtomicU8` cache (0=unknown, 1=unavailable, 2=available). Compile-time `#[cfg(target_feature)]` guard skips the cache entirely when features are guaranteed.

### safe_unaligned_simd Gaps (discovered in rav1d-safe refactoring)

Found during pal.rs refactoring to use `#[arcane]` + `safe_unaligned_simd`:

- **SOLVED: Created `partial_simd` module in rav1d-safe** with `Is64BitsUnaligned` trait:
  ```rust
  // Safe functions with #[target_feature] - callable from #[arcane] without unsafe!
  #[target_feature(enable = "sse2")]
  pub fn mm_loadl_epi64<T: Is64BitsUnaligned>(src: &T) -> __m128i {
      unsafe { _mm_loadl_epi64(ptr::from_ref(src).cast()) }
  }
  // Trait: [u8; 8], [i16; 4], [i32; 2], u64, i64, f64, etc.
  ```
  - Generates identical `vmovq` instructions (zero overhead)
  - Pattern ready for upstream to safe_unaligned_simd

- **Verified: No overhead from slice-to-array conversion**
  - `slice[..32].try_into().unwrap()` optimizes away completely
  - `safe_simd::_mm256_loadu_si256(arr)` → same `vmovdqu` as raw pointer

### Completed

- ~~**Type implementation verification**~~: Done. Added `implementation_name() -> &'static str` to all magetypes vectors. Uses tier-based naming: `"x86::v3::f32x8"`, `"x86::v4::f32x16"`, `"arm::neon::f32x4"`, `"wasm::wasm128::f32x4"`, `"polyfill::v3::f32x8"`, `"polyfill::v3_512::f32x16"`, `"polyfill::neon::f32x8"`. Test in `tests/exhaustive_intrinsics.rs`.
- ~~**WASM u64x2 ordering comparisons**~~: Done. Added simd_lt/le/gt/ge via bias-to-signed polyfill (XOR with i64::MIN, then i64x2_lt/gt). Parity: 4 → 0.
- ~~**x86 byte shift polyfills**~~: Done. Added i8x16/u8x16 shl, shr, shr_arithmetic for all x86 widths. Uses 16-bit shift + byte mask (~2 instructions). AVX-512 shr_arithmetic uses mask registers. Parity: 9 → 4.
- ~~**All actionable parity issues**~~: Done. Closed 28 remaining issues: extend/pack ops (17), RGBA pixel ops (4), i64/u64 polyfill math (7). Parity: 37 → 9 (0 actionable).
- ~~**ARM/WASM block ops**~~: Done. ARM uses native vzip1q/vzip2q, WASM uses i32x4_shuffle. Both gained interleave_lo/hi, interleave, deinterleave_4ch, interleave_4ch, transpose_4x4, transpose_4x4_copy. Parity: 47 → 37.
- ~~**WASM cbrt + f64x2 log10_lowp**~~: Done. WASM f32x4 gained cbrt_midp/cbrt_midp_precise (scalar initial guess + Newton-Raphson). WASM f64x2 gained log10_lowp via scalar fallback.
- ~~**ARM transcendentals + x86 missing variants**~~: Done. ARM f32x4 has full lowp+midp transcendentals (log2, exp2, ln, exp, log10, pow, cbrt) with all variant coverage. ARM f64x2 has lowp transcendentals via scalar fallback. x86 gained lowp _unchecked aliases, midp _precise variants, and log10_midp family. Parity: 80 → 47.
- ~~**API surface parity detection tool**~~: Done. Use `cargo xtask parity` to detect API variances between x86/ARM/WASM.
- ~~**Move generated files to subfolder**~~: Done. All generated code now lives in `generated/` subfolders.
- ~~**Merge WASM transcendentals from `feat/wasm128`**~~: Done (354dc2b). All `_unchecked` and `_precise` variants now generated.
- ~~**ARM comparison ops**~~: Done. Added simd_eq, simd_ne, simd_lt, simd_le, simd_gt, simd_ge, blend.
- ~~**ARM bitwise ops**~~: Done. Added not, shl, shr for all integer types.
- ~~**ARM boolean reductions**~~: Done. Added all_true, any_true, bitmask for all integer types.
- ~~**x86 boolean reductions**~~: Done. Added all_true, any_true, bitmask for all integer types (128/256/512-bit).
- ~~**WASM token-gated casting methods**~~: Done. Added cast_slice, cast_slice_mut, as_bytes, as_bytes_mut, from_bytes, from_bytes_owned (token-gated replacements for bytemuck, NOT actual Pod/Zeroable implementations).
- ~~**ARM reduce_add for unsigned**~~: Done. Extended reduce_add to all integer types including unsigned.
- ~~**Approximations (rcp, rsqrt) for ARM/WASM**~~: Done. ARM uses native vrecpe/vrsqrte, WASM uses division.
- ~~**mul_sub for ARM/WASM**~~: Done. ARM uses vfma with negation, WASM uses mul+sub.
- ~~**Type conversions for ARM/WASM**~~: Done. Added to_i32x4, to_i32x4_round, from_i32x4, to_f32x4, to_i32x4_low.
- ~~**shr_arithmetic for ARM/WASM**~~: Done. Added for i8x16, i16x8, i32x4.

## Suboptimal Intrinsics (needs faster-path overloads)

Track places where we use polyfills or slower instruction sequences because the base token lacks a native intrinsic, but a higher token would have one. Each entry should get a method overload that accepts the higher token for the fast path.

| Method | Token (slow) | Polyfill | Token (fast) | Native Intrinsic | Status |
|--------|-------------|----------|-------------|------------------|--------|
| f32 cbrt initial guess | all tokens | scalar extract + bit hack | — | No SIMD cbrt exists; consider SIMD bit hack via integer ops | Low priority |

**Rules for this section:**
- Only add entries when you've verified the faster intrinsic exists and is correct.
- The overload should take the higher token as a parameter (e.g., `fn min_fast(self, other: Self, _: X64V4Token) -> Self`).
- Or use trait bounds: `fn min<T: HasX64V4>(self, other: Self, _: T) -> Self` for the fast path.
- Remove entries when the fast-path overload is implemented.

### Completed fast-path overloads

All i64/u64 min/max/abs now have `_fast` variants that take `X64V4Token`:
- `i64x2::min_fast`, `max_fast`, `abs_fast`
- `u64x2::min_fast`, `max_fast`
- `i64x4::min_fast`, `max_fast`, `abs_fast`
- `u64x4::min_fast`, `max_fast`

## License

MIT OR Apache-2.0
