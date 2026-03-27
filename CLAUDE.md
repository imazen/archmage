# archmage

> Safely invoke your intrinsic power, using the tokens granted to you by the CPU. Cast primitive magics faster than any mage alive.

## CRITICAL: Every Conversation Health Check

**Run these checks at the start of every conversation, even if the user doesn't ask:**

```bash
just generate           # Regenerate all code from token-registry.toml
just validate-registry  # Validate token-registry.toml
just validate-tokens    # Validate magetypes safety + summon() checks
just soundness          # Static intrinsic soundness verification
```

If any fail, fix them before starting other work. Report failures to the user.

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
| `#[rite]` / `#[rite(v3)]` | `#[token_target_features]` | Adds `#[target_feature]` + `#[inline]` directly (internal helper). Tier-based: `#[rite(v3)]` — no token param needed |
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

**Tokens exist everywhere.** `X64V3Token`, `Arm64`, etc. compile on all platforms—`summon()` just returns `None` on unsupported architectures. `#[arcane]`/`#[rite]` cfg-gate their output to the matching architecture automatically, so you don't need `#[cfg(target_arch)]` on function definitions. `incant!` also handles cfg-gating at call sites.

### CRITICAL: How the Macros Choose Features

`#[arcane]` and `#[rite]` determine features in three ways:

1. **Token-based** (default): Parse the token type from the function signature. `X64V3Token` → `#[target_feature(enable = "avx2,fma,...")]`.
2. **Tier-based** (`#[rite(v3)]`): The tier name specifies the features directly. No token parameter needed.
3. **Multi-tier** (`#[rite(v3, v4, neon)]`): Generates a suffixed variant for each tier (`fn_v3`, `fn_v4`, `fn_neon`), each with its own `#[target_feature]` and `#[cfg(target_arch)]`.

Single-tier and token-based produce identical `#[target_feature]` attributes. Multi-tier produces multiple functions — one per tier, each compiled with different features. The token form can be easier to remember if you already have the token in scope.

`#[arcane]` generates a wrapper: an outer function that calls an inner `#[target_feature]` function via `unsafe`. This wrapper is how you cross into SIMD code without writing `unsafe` yourself — but it also creates an LLVM optimization boundary. `#[rite]` applies `#[target_feature]` + `#[inline]` directly, with no wrapper and no boundary.

**`#[rite]` should be the default.** Use `#[arcane(import_intrinsics)]` only at the entry point. For internal helpers, use `#[rite(v3, import_intrinsics)]` (tier-based, no token parameter) or `#[rite(import_intrinsics)]` (token-based). For multi-tier auto-vectorization, use `#[rite(v3, v4, neon)]`. `import_intrinsics` auto-imports `archmage::intrinsics::{arch}::*` — a combined module where `safe_unaligned_simd` memory ops shadow `core::arch` pointer-based ones. No ambiguity, no qualification needed.

Multi-tier variants are safe to call from matching `#[arcane]` or `#[rite]` contexts — since Rust 1.85, `#[target_feature]` functions can safely call other `#[target_feature]` functions when the caller has matching or superset features.

### CRITICAL: Target-Feature Boundaries (4x Performance Impact)

**Enter `#[arcane(import_intrinsics)]` once at the top, use `#[rite(import_intrinsics)]` for everything inside.**

LLVM cannot inline across mismatched `#[target_feature]` attributes. Each `#[arcane]` call from non-SIMD code creates an optimization boundary — LLVM can't hoist loads, sink stores, or vectorize across it. This costs 4-6x depending on workload (see `benches/asm_inspection.rs` and `docs/PERFORMANCE.md`). Token hoisting doesn't help — even with the token pre-summoned, calling `#[arcane]` per iteration still hits the boundary.

```rust
// WRONG: #[arcane] boundary every iteration (4x slower)
#[arcane(import_intrinsics)]
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
// RIGHT: one #[arcane] entry, #[rite(v3)] functions inline freely
fn process_all(points: &[[f32; 8]]) {
    if let Some(token) = X64V3Token::summon() {
        process_all_simd(token, points);
    } else {
        process_all_scalar(points);
    }
}

#[arcane(import_intrinsics)]
fn process_all_simd(_token: X64V3Token, points: &[[f32; 8]]) {
    for i in 0..points.len() {
        for j in i+1..points.len() {
            dist_simd(&points[i], &points[j]); // #[rite(v3)] — no token
        }
    }
}

#[rite(v3, import_intrinsics)]
fn dist_simd(a: &[f32; 8], b: &[f32; 8]) -> f32 {
    // Inlines into process_all_simd — same LLVM optimization region
    ...
}
```

**The rule:** `#[arcane(import_intrinsics)]` at the entry point, `#[rite(v3, import_intrinsics)]` for everything called from SIMD code (or `#[rite(import_intrinsics)]` with a token when using magetypes).

### CRITICAL: Generic Bounds Are Optimization Barriers

**Generic passthrough with trait bounds breaks inlining.** The compiler cannot inline across generic boundaries — each trait-bounded call is a potential indirect call.

```rust
// BAD: Generic bound prevents inlining into caller
#[arcane(import_intrinsics)]
fn process_generic<T: HasX64V2>(token: T, data: &[f32]) -> f32 {
    inner_work(token, data)  // Can't inline — T could be any type
}

#[rite(import_intrinsics)]
fn inner_work<T: HasX64V2>(token: T, data: &[f32]) -> f32 {
    // Even with #[inline(always)], this may not inline through generic
    ...
}
```

```rust
// GOOD: Concrete token enables full inlining
#[arcane(import_intrinsics)]
fn process_concrete(token: X64V3Token, data: &[f32]) -> f32 {
    inner_work(token, data)  // Fully inlinable — concrete type
}

#[rite(import_intrinsics)]
fn inner_work(token: X64V3Token, data: &[f32]) -> f32 {
    // Inlines into caller, single #[target_feature] region
    ...
}
```

**Why this matters:**
- `#[target_feature]` functions inline to share the feature-enabled region
- Generic bounds break this chain — each function is a separate compilation unit
- Even `#[inline(always)]` can't force inlining across trait object boundaries

**Exception: magetypes backend generics are zero-cost inside `#[arcane]` — if they inline.**
`f32x8::<T>` where `T: F32x8Backend` produces **identical assembly** to concrete `f32x8::<x64v3>` — but only when the generic function inlines into the `#[arcane]` caller. The generic function has no `#[target_feature]` of its own; it inherits the caller's features through inlining. **Mark generic SIMD helpers `#[inline(always)]`** to guarantee this. With `#[inline(never)]`, the same generic code is 18x slower even inside `#[arcane]` — intrinsics become function calls because the non-inlined function body compiles without target features. See `benches/generic_vs_concrete.rs`.

**Downcasting is free:** Pass a higher token to a function expecting a lower one. Nested `#[arcane]` with downcasting preserves the inlining chain:

```rust
#[arcane(import_intrinsics)]
fn v4_kernel(token: X64V4Token, data: &mut [f32]) {
    // Can call V3 functions — V4 is a superset
    let partial = v3_impl(token, &data[..8]);  // Downcasts, still inlines
    // ... AVX-512 specific work ...
}

#[rite(import_intrinsics)]
fn v3_impl(token: X64V3Token, chunk: &[f32]) -> f32 {
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

### `#[arcane]`: Expansion Modes

**Sibling (default):** Two functions at same scope — `__arcane_fn` (unsafe, target_feature) + safe wrapper.
`self`/`Self` work naturally. Use for free functions and inherent impl methods.

```rust
impl MyType {
    #[arcane(import_intrinsics)]
    fn compute(&self, token: X64V3Token) -> f32 {
        self.data.iter().sum()  // self/Self just work!
    }
}
```

**Nested** (`nested` or `_self = Type`): Inner function inside outer. Required for trait impls
(sibling would add methods not in trait). `_self = Type` implies nested.

```rust
impl SimdOps for Processor {
    #[arcane(_self = Processor)]
    fn process(&self, token: X64V3Token) -> f32 {
        _self.threshold  // Use _self, not self
    }
}
```

### `#[arcane]`/`#[rite]`: Cross-Arch Behavior

**Default (cfg-out):** On wrong architecture, no function is emitted. Less dead code.
Direct *call sites* referencing the function by name must use `#[cfg]` guards, `stub`, or `incant!`. No `#[cfg]` is needed on the function *definitions* — the macros handle that.

**With `stub`:** Generates unreachable stub on wrong architecture.
Use when cross-arch dispatch references the function without cfg guards.

```rust
#[arcane(stub)]  // unreachable stub on non-x86
fn process_avx2(token: X64V3Token, data: &[f32]) -> f32 { ... }

#[arcane(stub)]  // unreachable stub on non-ARM
fn process_neon(token: NeonToken, data: &[f32]) -> f32 { ... }

// Both referenced without cfg guards — stubs make this compile everywhere
fn dispatch(data: &[f32]) -> f32 {
    if let Some(t) = X64V3Token::summon() { process_avx2(t, data) }
    else if let Some(t) = NeonToken::summon() { process_neon(t, data) }
    else { data.iter().sum() }
}
```

`incant!` is unaffected — it already cfg-gates dispatch calls.

`#[rite(stub)]` works the same way for `#[rite]` functions.

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
    incant!(sum(data), [v1, v3, neon, scalar])
}
// Requires: sum_v1, sum_v3, sum_neon, sum_scalar
```

Tier names accept the `_` prefix — `_v3` is identical to `v3`, matching the suffix pattern on generated function names.

**Feature-gated tiers** — canonical form `tier(cfg(feature))`, shorthand `tier(feature)`:

```rust
incant!(sum(data), [v4(cfg(avx512)), v3, neon, scalar])
// v4 dispatch arm wrapped in #[cfg(feature = "avx512")]
```

**Tier list modifiers** — add/remove from the default list instead of restating it:

```rust
incant!(sum(data), [+arm_v2])           // add arm_v2 to defaults
incant!(sum(data), [-neon, -wasm128])   // remove tiers
incant!(sum(data), [+v4])               // make v4 unconditional (no avx512 gate)
incant!(sum(data), [+default])          // replace scalar with tokenless default
incant!(sum(data), [-neon, +v1])        // combine freely
```

All entries must be `+`/`-` (modifier mode) or none (override mode) — mixing is a compile error.
`default` and `scalar` are interchangeable fallback slots (`+default` replaces `scalar`).

Known tiers: `v1`, `v2`, `x64_crypto`, `v3`, `v3_crypto`, `v4`, `v4x`, `arm_v2`, `arm_v3`,
`neon`, `neon_aes`, `neon_sha3`, `neon_crc`, `wasm128`, `wasm128_relaxed`, `scalar`.
Always include `scalar` in explicit tier lists. Currently auto-appended if omitted; will become a compile error in v1.0.

**Passthrough mode** (already have token):

```rust
fn inner<T: IntoConcreteToken>(token: T, data: &[f32]) -> f32 {
    incant!(process(data) with token)
}

// With explicit tiers:
fn inner<T: IntoConcreteToken>(token: T, data: &[f32]) -> f32 {
    incant!(process(data) with token, [v3, neon, scalar])
}
```

### `#[autoversion]`: Auto-Vectorized Dispatch

Write plain scalar code with a `SimdToken` placeholder. `#[autoversion]` generates per-tier
variants (each compiled with `#[target_feature]` via `#[arcane]`) plus a runtime dispatcher.
The compiler auto-vectorizes each variant — no intrinsics needed.

```rust
use archmage::prelude::*;

#[autoversion]
fn sum_of_squares(_token: SimdToken, data: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for &x in data {
        sum += x * x;
    }
    sum
}

// Call directly — no token, no unsafe:
let result = sum_of_squares(&my_data);
```

**What gets generated:** `sum_of_squares_v4`, `_v3`, `_neon`, `_wasm128`, `_scalar` variants
plus a `sum_of_squares(data)` dispatcher (token param removed).

**Explicit tiers:** `#[autoversion(v3, neon)]`. `scalar` is always implicit. Tier names accept
the `_` prefix — `_v3` is identical to `v3`.

**Feature-gated tiers:** `#[autoversion(v4(cfg(avx512)), v3, neon)]`. Shorthand `v4(avx512)` also works.

**Tier list modifiers:** `#[autoversion(+arm_v2)]` adds to defaults, `#[autoversion(-wasm128)]`
removes. All entries must be `+`/`-` or none — mixing is a compile error.

**Self receivers:** `#[autoversion(_self = MyType)]` — use `_self` in body.

**Trait methods:** Can't expand to siblings. Delegate to an autoversioned inherent method.

**vs `#[magetypes]` + `incant!`:**
- `SimdToken` is replaced only in the signature (fast compile). `#[magetypes]` does token-level replacement for `Token`, text substitution for other placeholders.
- `#[autoversion]` generates its own dispatcher. `#[magetypes]` requires separate `incant!`.
- Use `#[autoversion]` for scalar auto-vectorization. Use `#[magetypes]` + `incant!` for hand-written intrinsics.

Known tiers: `v1`, `v2`, `v3`, `v3_crypto`, `v4`, `v4x`, `neon`, `neon_aes`, `neon_sha3`,
`neon_crc`, `arm_v2`, `arm_v3`, `wasm128`, `wasm128_relaxed`, `x64_crypto`, `scalar`.

### `ScalarToken`

Always-available fallback. Used for:
- `incant!()` and `#[autoversion]` convention (`_scalar` suffix)
- Consistent API shape in dispatch

### Fixed-Size Types with Polyfills

**Pick a concrete size. Use polyfills for portability.**

```rust
use magetypes::simd::f32x8;  // Always 8 lanes, polyfilled on ARM/WASM

#[arcane(import_intrinsics)]
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
| **v3** | + AVX, AVX2, FMA, BMI1, BMI2, F16C | `X64V3Token` | Use token directly |
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

### `#[rite]` is the default, `#[arcane]` only at entry points

**`#[rite]` should be the default.** It adds `#[target_feature]` + `#[inline]` — LLVM inlines it into any caller with matching features.

**Three modes:**
- `#[rite(v3, import_intrinsics)]` — tier-based, no token needed
- `#[rite(import_intrinsics)]` — token-based, reads token from params
- `#[rite(v3, v4, neon, import_intrinsics)]` — multi-tier, generates suffixed variants (`fn_v3`, `fn_v4`, `fn_neon`)

Single-tier and token-based produce identical code. The token form can be easier to remember if you already have it in scope. Multi-tier generates one copy per tier, each compiled with different features.

Use `#[arcane(import_intrinsics)]` only when the function is called from non-SIMD code:
- After `summon()` in a public API
- From tests
- From non-`#[target_feature]` contexts

```rust
// Entry point (called after summon) - use #[arcane]
#[arcane(import_intrinsics)]
pub fn process(token: X64V3Token, data: &mut [f32]) {
    for chunk in data.chunks_exact_mut(8) {
        process_chunk(chunk);  // Calls #[rite(v3)] — inlines, no token
    }
}

// Called from SIMD code - use #[rite] with tier name
#[rite(v3, import_intrinsics)]
fn process_chunk(chunk: &mut [f32; 8]) {
    // ...
}

// Or with token (when you need to pass it to magetypes)
#[rite(import_intrinsics)]
fn process_chunk_with_token(_: X64V3Token, chunk: &mut [f32; 8]) {
    // ...
}

// Multi-tier: generates process_v3() and process_v4() from one body
#[rite(v3, v4, import_intrinsics)]
fn process_multi(data: &[f32; 4]) -> f32 {
    // Each variant compiled with different #[target_feature]
    data.iter().sum()
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

// CORRECT - use #[arcane] (generates #[target_feature] + auto-imports intrinsics)
#[arcane(import_intrinsics)]
fn process(token: X64V3Token, data: &[f32]) -> f32 {
    // This function body is compiled with #[target_feature(enable = "avx2,fma")]
    // Intrinsics auto-imported and inline properly into single SIMD instructions
    ...
}
```

### `import_intrinsics` handles safe memory ops

**`import_intrinsics` imports `archmage::intrinsics::{arch}::*`** — a combined module where safe memory ops shadow pointer-based ones. Memory ops like `_mm256_loadu_ps` resolve to the safe reference-based version automatically, with no ambiguity.

```rust
// WRONG - raw pointers need unsafe
let v = unsafe { _mm256_loadu_ps(data.as_ptr()) };

// CORRECT - import_intrinsics provides safe memory ops automatically
#[arcane(import_intrinsics)]
fn example(_token: X64V3Token, data: &[f32; 8]) -> __m256 {
    _mm256_loadu_ps(data)  // Takes &[f32; 8], not *const f32
}
```

## CRITICAL: Doc Example Testing (CI-Enforced)

Every code example in `docs/site/content/` MUST have a corresponding test in
`magetypes/tests/doc_examples.rs` (for magetypes docs) or `tests/doc_examples.rs`
(for archmage docs). Code that appears in documentation MUST compile and pass tests.

- When modifying doc pages, update the corresponding test
- When adding new doc pages with code examples, add tests first
- `cargo test -p magetypes --test doc_examples` must pass before pushing doc changes
- Magetypes examples MUST use the generic pattern: `f32x8::<T>`, not flat aliases
- Flat aliases (`use magetypes::simd::f32x8`) are BANNED in documentation

### Correct doc import pattern

```rust
use magetypes::simd::{
    generic::f32x8,
    backends::{F32x8Backend, x64v3, neon, scalar},
};
```

### Correct generic function pattern (primary in all docs)

Generic SIMD helpers MUST be `#[inline(always)]` — they have no `#[target_feature]` of their own and rely on inlining into the `#[arcane]` caller to get AVX2/NEON features. Without inlining, intrinsics become function calls (18x slower).

```rust
#[inline(always)]
fn sum<T: F32x8Backend>(token: T, data: &[f32]) -> f32 {
    let mut acc = f32x8::<T>::zero(token);
    for chunk in data.chunks_exact(8) {
        let v = f32x8::<T>::load(token, chunk.try_into().unwrap());
        acc = acc + v;
    }
    acc.reduce_add()
}
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
5. Intrinsic soundness verification
6. `cargo clippy --features "std avx512"` — zero warnings
7. `cargo clippy -p magetypes` (default features) — zero warnings
8. `cargo test --features "std avx512"` — all tests pass
9. **no_std compilation + tests** — archmage and magetypes build and test without `std`
10. **No-features integration test** — `archmage-no-features-test` crate (zero features, `deny(warnings)`)
11. `cargo fmt --check` — code is formatted
12. `cargo doc --features "std avx512" --no-deps` with `RUSTDOCFLAGS=-Dwarnings` — no broken doc links
13. Miri UB detection (skipped if not installed)
14. **ARM64 cross-compilation + tests** (requires `cross` + Docker)
15. **WASM cross-compilation + tests** (requires `wasmtime` + `wasm32-wasip1` target)
16. **ARM64 clippy** (requires `cross` + Docker)

**Note:** Parity check reports 0 issues. All W128 types have identical APIs across x86/ARM/WASM.

**Note:** Steps 12-14 require cross-compilation tooling. If `cross`/Docker/`wasmtime` are unavailable, they are skipped with warnings. For full coverage before publish, install the tooling:
```bash
cargo install cross --git https://github.com/cross-rs/cross  # ARM64 testing
curl https://wasmtime.dev/install.sh -sSf | bash              # WASM testing
rustup target add wasm32-wasip1                                # WASM target
```

If ANY check fails:
- Do NOT push
- Do NOT publish
- Fix the issue first
- Re-run `just ci` until it passes

**NEVER run `cargo publish` locally.** All releases go through GitHub Actions via tagged releases only. This ensures every published version has passed the full CI suite (including cross-platform tests, SDE emulation, Miri, etc.) before reaching crates.io. Local `cargo publish` bypasses CI and is banned.

**Release process:**
1. Update the version in `[workspace.package]` in the root `Cargo.toml` (single source of truth — all crates inherit via `version.workspace = true`)
2. Update internal dep version strings (`archmage-macros = { version = "X.Y.Z", ... }` in root, `archmage = { version = "X.Y.Z", ... }` in magetypes)
3. Commit the version bump, push to main
4. Wait for CI to pass on main
5. Create a GitHub release with matching tags — the release workflow handles `cargo publish`

**Git tags are MANDATORY for every publish.** Tags MUST match published versions:

```
v{version}                        # archmage
archmage-macros-v{version}        # archmage-macros
magetypes-v{version}              # magetypes
```

Publish order (respect dependency chain): `archmage-macros` → `archmage` → `magetypes`.

**Every published crate MUST have an `include` list in Cargo.toml.** Without one, cargo ships the entire repo — docs, CI workflows, intrinsics databases, browser apps, everything. Run `cargo package --list` before publishing and verify only source code, tests, benches, examples, and essential metadata are included. The archmage crate ballooned to 36MB (503 rejected by crates.io) before this was caught. Target: under 300KB compressed.

## MANDATORY: Cross-Platform Token Testing

Every token's feature claims MUST be verified by exercising real intrinsics on the target architecture. The test files:

| Test File | Architecture | What it tests |
|-----------|-------------|---------------|
| `tests/x86_crypto_intrinsics.rs` | x86_64 | X64CryptoToken (PCLMULQDQ, AES-NI), X64V3CryptoToken (VPCLMULQDQ, VAES) |
| `tests/avx512_intrinsics_exercise.rs` | x86_64 | X64V4Token, X64V4xToken (AVX-512 F/BW/CD/DQ/VL + extensions) |
| `tests/avx512fp16_intrinsics.rs` | x86_64 | Avx512Fp16Token (hierarchy only — all 935 intrinsics are nightly) |
| `tests/arm_feature_intrinsics.rs` | aarch64 | Arm64V2Token (RDM, DotProd, SHA2), NeonAesToken, NeonCrcToken, NeonSha3Token |
| `tests/wasm_intrinsics_exercise.rs` | wasm32 | Wasm128Token (SIMD128 — ~100 intrinsics) |
| `tests/feature_consistency.rs` | all | Token hierarchy, cross-arch None checks, feature detection consistency |

**Before ANY publish:**
1. `just ci` must pass (includes ARM64 + WASM when tooling available)
2. Every new token MUST have a corresponding test in the appropriate `*_intrinsics.rs` file
3. Every feature listed in token-registry.toml MUST be exercised by at least one intrinsic test
4. If a feature has NO stable intrinsics in Rust, document this explicitly

### Stable Intrinsic Coverage by Feature

| Feature | Stable Count | Status | Notes |
|---------|-------------|--------|-------|
| neon | ~1540 | Full | Baseline ARM64 |
| rdm | 36 | Full | All tested in arm_feature_intrinsics.rs |
| neon,sha3 | 22 | Full | All tested in arm_feature_intrinsics.rs |
| neon,aes (rounds + p64) | ~37 | Full | Tested in arm_feature_intrinsics.rs |
| sha2 | ~10 | Full | Tested in arm_feature_intrinsics.rs |
| crc | 8 | Full | Tested in arm_feature_intrinsics.rs |
| dotprod | ALL | Nightly | ALL dotprod intrinsics require `stdarch_neon_dotprod` (unstable) |
| neon,fp16 | 95/210 | Partial | 95 stable (conversion, div, FMA), 115 unstable |
| fcma | 34 | Nightly | All unstable |
| i8mm | 4 | Nightly | All unstable |
| fhm | 0 | None | No Rust intrinsics in stdarch |
| bf16 | 0 | None | No Rust intrinsics in stdarch |
| avx512fp16 | 438/440 | Stable | 438 stable, 2 unstable (per intrinsics CSV) |
| pclmulqdq + aes (128-bit) | ~10 | Full | Tested in x86_crypto_intrinsics.rs |
| vpclmulqdq + vaes (256-bit) | ~8 | Full | Tested in x86_crypto_intrinsics.rs |
| simd128 (wasm) | ~100+ | Full | Tested in wasm_intrinsics_exercise.rs |

**Features with zero stable intrinsics** (fhm, bf16, avx512fp16) are documented but cannot have exercise tests on stable Rust. When these stabilize, add tests immediately.

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
fn example() {
    let a = _mm256_setzero_ps();           // SAFE!
    let b = _mm256_add_ps(a, a);           // SAFE!
    let c = _mm256_fmadd_ps(a, a, a);      // SAFE!

    // Only memory ops remain unsafe (raw pointers)
    let v = unsafe { _mm256_loadu_ps(ptr) };  // Still needs unsafe
}
```

Note: Rust 2024 edition allows safe `#[target_feature]` functions — they don't need `unsafe fn`.
Calling them from non-matching contexts still requires `unsafe`.

This means we **don't need to wrap** arithmetic, shuffle, compare, bitwise, or other value-based intrinsics. Only:
1. **Tokens** - Prove CPU features are available
2. **`#[arcane]` macro** - Enable `#[target_feature]` via token proof
3. **`safe_unaligned_simd`** - Reference-based memory operations (user adds as dependency)

**`#![forbid(unsafe_code)]` compatible**: Downstream crates can use `#![forbid(unsafe_code)]` when combining archmage tokens + `#[arcane]`/`#[rite]` macros + `safe_unaligned_simd` for memory operations. The sibling `#[target_feature]` function is declared safe (Rust 2024 edition allows this), and the `unsafe` block in the wrapper is proc-macro-generated — `#![forbid(unsafe_code)]` allows `unsafe` from proc macros the same way it allows `unsafe` inside external crate code.

**CRITICAL: Never generate `unsafe fn` in sibling mode.** The sibling `#[target_feature]` function MUST be `fn`, not `unsafe fn`. `#![forbid(unsafe_code)]` rejects `unsafe fn` declarations even from proc macros, but allows `unsafe` blocks from proc macros. Only the wrapper's call to the sibling needs `unsafe { ... }`.

## How `#[arcane]` Works

**Sibling mode (default):** Generates a sibling `#[target_feature]` function at the same scope:

```rust
// You write:
#[arcane(import_intrinsics)]
fn my_kernel(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    let v = _mm256_setzero_ps();  // Safe! In scope from import_intrinsics.
    // ...
}

// Macro generates (x86_64 only):
#[cfg(target_arch = "x86_64")]
#[doc(hidden)]
#[target_feature(enable = "avx2,fma,...")]
fn __arcane_my_kernel(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    use archmage::intrinsics::x86_64::*;
    let v = _mm256_setzero_ps();
    // ...
}

#[cfg(target_arch = "x86_64")]
fn my_kernel(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    unsafe { __arcane_my_kernel(token, data) }
}
```

**Nested mode** (`nested` or `_self = Type`): Inner function inside original. Used for trait impls.

| Option | Sibling | Nested | Cfg-out | Stub |
|--------|---------|--------|---------|------|
| `#[arcane]` | Default | - | Default | - |
| `#[arcane(nested)]` | - | Yes | Default | - |
| `#[arcane(stub)]` | Default | - | - | Yes |
| `#[arcane(_self = T)]` | - | Implied | Default | - |
| `#[arcane(nested, stub)]` | - | Yes | - | Yes |

## Friendly Aliases

| Alias | Token | What it means |
|-------|-------|---------------|
| `Sse2Token` | `X64V1Token` | SSE + SSE2 (x86_64 baseline — always available) |
| `Server64` | `X64V4Token` | + AVX-512 (Xeon 2017+, Zen 4+) |
| `Arm64` | `NeonToken` | NEON (all 64-bit ARM) |

```rust
use archmage::{X64V3Token, SimdToken, arcane};

#[arcane(import_intrinsics)]
fn process(token: X64V3Token, data: &mut [f32; 8]) {
    // AVX2 + FMA intrinsics in scope from import_intrinsics
}

if let Some(token) = X64V3Token::summon() {
    process(token, &mut data);
}
```

## Directory Structure

```
token-registry.toml          # THE source of truth for all token/trait/feature data
docs/spec.md                 # Architecture spec and safety model documentation
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
├── site/                   # Zola documentation site (Goyo theme)
│   ├── config.toml         # Zola configuration
│   ├── content/
│   │   ├── archmage/       # Archmage docs (stable)
│   │   └── magetypes/      # Magetypes docs (experimental)
│   └── themes/goyo/        # Git submodule
├── generated/              # Auto-generated reference docs
│   ├── x86-intrinsics-by-token.md
│   ├── aarch64-intrinsics-by-token.md
│   └── memory-ops-reference.md
└── intrinsics-browser/     # Static HTML/JS intrinsics search
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
- `X64V3Token` - AVX2 + FMA + BMI2 (Haswell 2013+, Zen 1+)
- `X64V3CryptoToken` - V3 + VPCLMULQDQ + VAES (Zen 3+ 2020, Alder Lake 2021+)
- `X64V4Token` / `Avx512Token` - + AVX-512 F/BW/CD/DQ/VL + AES-NI (Skylake-X 2017+, Zen 4+)
- `X64V4xToken` - + modern extensions + VAES/VPCLMULQDQ/GFNI (Ice Lake 2019+, Zen 4+)
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

#[arcane(import_intrinsics)]
fn process_simd(token: X64V3Token, data: &[f32; 8]) -> f32 {
    let a = f32x8::load(token, data);
    let b = f32x8::splat(token, 2.0);
    let c = a * b;
    c.reduce_add()
}
```

On ARM/WASM, `f32x8` is polyfilled with two `f32x4` operations. Pick the size that fits your algorithm.

## Safe Memory Operations

Safe memory ops are available automatically inside `#[arcane]`/`#[rite]` functions with `import_intrinsics`:

```rust
use archmage::{X64V3Token, SimdToken, arcane};

#[arcane(import_intrinsics)]
fn process(_token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
    // import_intrinsics brings archmage::intrinsics::{arch}::* into scope
    // Safe memory ops shadow pointer-based ones automatically
    let v = _mm256_loadu_ps(data);  // Takes &[f32; 8], not *const f32
    let squared = _mm256_mul_ps(v, v);
    let mut out = [0.0f32; 8];
    _mm256_storeu_ps(&mut out, squared);  // Takes &mut [f32; 8], not *mut f32
    out
}
```

## Known Bugs

Found by macro expansion snapshot compilation tests (`tests/expand/*.expanded.rs`):

1. **`#[autoversion]` on `unsafe fn`: dispatcher drops `unsafe`** — The generated dispatcher function is `fn` instead of `unsafe fn`. Calls to unsafe variant functions (`unsafe_sum_v4(...)`) lack `unsafe` blocks. Soundness bug. (`tests/expand/autoversion_unsafe_fn.expanded.rs`)

2. **`#[rite]` on trait impl method: `#[target_feature]` on safe trait method is invalid** — Rust rejects `#[target_feature(..)]` on safe trait methods. The macro applies it directly, which works as macro output but the expanded code is invalid standalone Rust. (`tests/expand/rite_trait_impl.expanded.rs`)

3. **`#[autoversion]` on trait impl method: variants placed inside trait impl block** — Generated variant methods (`process_v3`, `process_v4`, `process_scalar`) are emitted inside `impl Trait for Type {}`, but they aren't members of the trait. Compile error E0407. (`tests/expand/autoversion_trait_impl.expanded.rs`)

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

### Known Platform Detection Issues

**macOS 15.x + `-Ctarget-cpu=native`** (fixed in 0.8.8+):
- LLVM's target-cpu=native drops sha2/sha3 (and other features) from compile-time on Apple aarch64
- `std::arch::is_aarch64_feature_detected!("sha2")` returns false on macOS 15.x (works on 14.x)
- This caused Arm64V2Token::summon() to return None when using target-cpu=native
- **Fix**: Apple vendor fallback in `__impl_aarch64_apple_or_runtime_check!` — features in the
  Apple target spec baseline (neon, crc, rdm, dotprod, fp16, aes, sha2, sha3, fhm, fcma) return
  true unconditionally on `target_vendor = "apple"` when compile-time detection fails
- Upstream bugs: LLVM native CPU probe + Rust std_detect on macOS 15.x

**Windows ARM64 — limited runtime detection** (not fixable in archmage):
- Windows `IsProcessorFeaturePresent` API only exposes: neon, crc, dotprod, aes, sha2
- Features NOT detectable at runtime on Windows: rdm, fp16, fhm, fcma, sha3, i8mm, bf16
- This means Arm64V2Token::summon() returns None on Windows ARM64 (rdm and fp16 missing)
- Snapdragon X definitely has these features but Windows doesn't expose them
- Possible future fix: registry-based detection or undocumented Windows APIs
- Tracked as a known Rust std_detect limitation

### ~~avx512 Feature Gating in Dispatch Macros~~ — Fixed

**All macros now handle avx512 correctly:**

- **`#[autoversion]`**: Always generates v4/v4x variants (scalar code + `#[target_feature]`, no safe memory ops needed). Has its own default tier list that always includes v4.
- **`incant!`/`#[magetypes]`**: Default tier list excludes v4 when archmage lacks avx512 feature. Explicit tier lists work unconditionally — no `#[cfg(feature)]` in output.
- **`#[arcane(import_intrinsics)]`/`#[rite(import_intrinsics)]` with V4 token**: Clear `compile_error!` when avx512 feature not enabled, telling user exactly what to add to Cargo.toml.
- **`#[arcane]`/`#[rite]` without `import_intrinsics`**: Always works with any token — value intrinsics need no cargo feature.

**Implementation:** `avx512` feature propagated from archmage → archmage-macros. Macros check `cfg!(feature = "avx512")` at expansion time. No `#[cfg(feature)]` ever emitted in output (was checking calling crate's features — always wrong for downstream crates).

**Test crates in `tests/avx512-cfg-tests/`** verify all scenarios including every token alias, trait bounds (`impl HasX64V4`), and generics (`T: HasX64V4`).

### Long-Term

- **v1.0: Require `default` or `scalar` in explicit tier lists (prefer `default`).** Explicit tier lists must include a fallback tier. `default` (tokenless) is the recommended form; `scalar` (takes `ScalarToken`) remains supported for `incant!` nesting. The auto-append of `scalar` stays until this becomes an error. Flip `REQUIRE_EXPLICIT_SCALAR` to `true` in `archmage-macros/src/lib.rs`. Re-enable the `scalar_not_in_tier_list` compile-fail test in `tests/compile_fail.rs`. Update the deprecation warning to say "include `default` or `scalar`" and suggest `default` as the preferred form.

- **v1.0: Require explicit `tier(cfg(feature))` syntax — remove implicit `cfg_feature` auto-gating.** Currently `TierDescriptor.cfg_feature` auto-applies `(avx512)` to v4/v4x in all tier lists (both default and explicit). This is backwards-compatible magic but confusing: plain `v4` in `[v4, v3, default]` silently gets a cfg gate. In v1.0, remove `cfg_feature` from `TierDescriptor` and `default_feature_gates` from `resolve_tiers`. Users must write `v4(cfg(avx512))` explicitly. Plain `v4` means unconditional dispatch — the function MUST exist. Default tier names drop v4 entirely; users add it with the syntax they want. `+v4` in modifier lists adds unconditional v4; `+v4(cfg(avx512))` adds gated v4. This makes the behavior visible and predictable. Migration: search for `[v4,` and `[v4x,` in tier lists, add `(cfg(avx512))`.

- **v1.0: Remove width traits** (`Has128BitSimd`, `Has256BitSimd`, `Has512BitSimd`). `Has256BitSimd` only enables AVX (not AVX2 or FMA) — actively misleading and causes suboptimal codegen. Use concrete tokens (`X64V3Token`) or tier traits (`HasX64V2`, `HasX64V4`).

- **v1.0: Remove `guaranteed()`**. Deprecated since 0.6.0, replaced by `compiled_with()`. Same semantics, better name.

- **v1.0: Remove `SimdToken` parameter support from `#[autoversion]`.** Deprecated since 0.9.11. Users should use tokenless (recommended) or `ScalarToken` for incant! nesting. `SimdToken` is a trait, not a type — it was always a macro-only placeholder.

- **v1.0: Remove `_self = Type` from `#[autoversion]`.** Plain `self` works in sibling mode since the beginning. Only trait delegation needs nested mode, and autoversion can't do trait impls anyway. Saves ~40 lines.

- **v1.0: Deprecate incant! passthrough mode (`with token`).** Zero uses in zen or any downstream crate. The use case (re-dispatch an existing token) is better served by `#[rite]` multi-tier or `IntoConcreteToken` directly. `default` tier solves the nesting case without passthrough.

- **NOT planned for removal:** `#[simd_fn]`, `simd_route!`, `try_new()`, `forge_token_dangerously()`. These are discouraged migration aliases but remain supported — they don't cause confusion or bugs, they just have better-named equivalents.

- **Generator test fixtures**: Add example input/expected output pairs to each xtask generator (SIMD types, width dispatch, tokens, macro registry). These serve as both documentation of expected output and cross-platform regression tests — run on x86, ARM, and WASM to catch codegen divergence.

- ~~**Target-feature boundary overhead benchmark**~~: Done. See `benches/asm_inspection.rs` and `docs/PERFORMANCE.md`. Key results:
  - Simple vector add (1000 x 8-float): `#[rite]` in `#[arcane]` 547 ns, `#[arcane]` per iteration 2209 ns (4x), bare `#[target_feature]` 2222 ns (4x, identical)
  - DCT-8 (100 rows x 8 dot products): `#[rite]` in `#[arcane]` 61 ns, `#[arcane]` per row 376 ns (6.2x), bare `#[target_feature]` 374 ns (6.2x, identical)
  - Cross-token nesting: downgrade (V4→V3, V3→V2) is free, upgrade (V2→V3, V3→V4) costs 4x, all patterns match bare `#[target_feature]`

  Key insight: the overhead is from the `#[target_feature]` optimization boundary, NOT from wrappers or archmage abstractions. The cost scales with computational density (4x simple add, 6.2x DCT-8). Feature direction matters: downgrades are free (superset enables inlining), upgrades hit the boundary.

- ~~**summon() caching**~~: **Implemented!** See `benches/summon_overhead.rs`. Results after adding atomic caching:
  - `X64V3Token::summon()` (cached): ~1.3 ns (was 2.6 ns — **2x faster**)
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
