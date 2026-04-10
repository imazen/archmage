//! Proc-macros for archmage SIMD capability tokens.
//!
//! Provides `#[arcane]`, `#[rite]`, `#[autoversion]`, `incant!`, and `#[magetypes]`.

mod arcane;
mod autoversion;
mod common;
mod generated;
mod incant;
mod magetypes;
mod rewrite;
mod rite;
mod tiers;
mod token_discovery;

use proc_macro::TokenStream;
use syn::parse_macro_input;

use arcane::*;
use autoversion::*;
use common::*;
use incant::*;
use magetypes::*;
use rite::*;
use tiers::*;

// Re-export items used by the test module (via `use super::*`).
#[cfg(test)]
use generated::{token_to_features, trait_to_features};
#[cfg(test)]
use quote::{ToTokens, format_ident};
#[cfg(test)]
use syn::{FnArg, PatType, Type};
#[cfg(test)]
use token_discovery::*;

// LightFn, filter_inline_attrs, is_lint_attr, filter_lint_attrs, gen_cfg_guard,
// build_turbofish, replace_self_in_tokens, suffix_path → moved to common.rs
// ArcaneArgs, SelfReceiver, arcane_impl, arcane_impl_* → moved to arcane.rs
// generate_imports → moved to common.rs

/// Mark a function as an arcane SIMD function.
///
/// This macro generates a safe wrapper around a `#[target_feature]` function.
/// The token parameter type determines which CPU features are enabled.
///
/// # Expansion Modes
///
/// ## Sibling (default)
///
/// Generates two functions at the same scope: a safe `#[target_feature]` sibling
/// and a safe wrapper. `self`/`Self` work naturally since both functions share scope.
/// Compatible with `#![forbid(unsafe_code)]`.
///
/// ```ignore
/// #[arcane]
/// fn process(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] { /* body */ }
/// // Expands to (x86_64 only):
/// #[cfg(target_arch = "x86_64")]
/// #[doc(hidden)]
/// #[target_feature(enable = "avx2,fma,...")]
/// fn __arcane_process(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] { /* body */ }
///
/// #[cfg(target_arch = "x86_64")]
/// fn process(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
///     unsafe { __arcane_process(token, data) }
/// }
/// ```
///
/// Methods work naturally:
///
/// ```ignore
/// impl MyType {
///     #[arcane]
///     fn compute(&self, token: X64V3Token) -> f32 {
///         self.data.iter().sum()  // self/Self just work!
///     }
/// }
/// ```
///
/// ## Nested (`nested` or `_self = Type`)
///
/// Generates a nested inner function inside the original. Required for trait impls
/// (where sibling functions would fail) and when `_self = Type` is used.
///
/// ```ignore
/// impl SimdOps for MyType {
///     #[arcane(_self = MyType)]
///     fn compute(&self, token: X64V3Token) -> Self {
///         // Use _self instead of self, Self replaced with MyType
///         _self.data.iter().sum()
///     }
/// }
/// ```
///
/// # Cross-Architecture Behavior
///
/// **Default (cfg-out):** On the wrong architecture, the function is not emitted
/// at all — no stub, no dead code. Code that references it must be cfg-gated.
///
/// **With `stub`:** Generates an `unreachable!()` stub on wrong architectures.
/// Use when cross-arch dispatch references the function without cfg guards.
///
/// ```ignore
/// #[arcane(stub)]  // generates stub on wrong arch
/// fn process_neon(token: NeonToken, data: &[f32]) -> f32 { ... }
/// ```
///
/// `incant!` is unaffected — it already cfg-gates dispatch calls by architecture.
///
/// # Token Parameter Forms
///
/// ```ignore
/// // Concrete token
/// #[arcane]
/// fn process(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] { ... }
///
/// // impl Trait bound
/// #[arcane]
/// fn process(token: impl HasX64V2, data: &[f32; 8]) -> [f32; 8] { ... }
///
/// // Generic with inline or where-clause bounds
/// #[arcane]
/// fn process<T: HasX64V2>(token: T, data: &[f32; 8]) -> [f32; 8] { ... }
///
/// // Wildcard
/// #[arcane]
/// fn process(_: X64V3Token, data: &[f32; 8]) -> [f32; 8] { ... }
/// ```
///
/// # Options
///
/// | Option | Effect |
/// |--------|--------|
/// | `stub` | Generate `unreachable!()` stub on wrong architecture |
/// | `nested` | Use nested inner function instead of sibling |
/// | `_self = Type` | Implies `nested`, transforms self receiver, replaces Self |
/// | `inline_always` | Use `#[inline(always)]` (requires nightly) |
/// | `import_intrinsics` | Auto-import `archmage::intrinsics::{arch}::*` (includes safe memory ops) |
/// | `import_magetypes` | Auto-import `magetypes::simd::{ns}::*` and `magetypes::simd::backends::*` |
///
/// ## Auto-Imports
///
/// `import_intrinsics` and `import_magetypes` inject `use` statements into the
/// function body, eliminating boilerplate. The macro derives the architecture and
/// namespace from the token type:
///
/// ```ignore
/// // Without auto-imports — lots of boilerplate:
/// use std::arch::x86_64::*;
/// use magetypes::simd::v3::*;
///
/// #[arcane]
/// fn process(token: X64V3Token, data: &[f32; 8]) -> f32 {
///     let v = f32x8::load(token, data);
///     let zero = _mm256_setzero_ps();
///     // ...
/// }
///
/// // With auto-imports — clean:
/// #[arcane(import_intrinsics, import_magetypes)]
/// fn process(token: X64V3Token, data: &[f32; 8]) -> f32 {
///     let v = f32x8::load(token, data);
///     let zero = _mm256_setzero_ps();
///     // ...
/// }
/// ```
///
/// The namespace mapping is token-driven:
///
/// | Token | `import_intrinsics` | `import_magetypes` |
/// |-------|--------------------|--------------------|
/// | `X64V1..V3Token` | `archmage::intrinsics::x86_64::*` | `magetypes::simd::v3::*` |
/// | `X64V4Token` | `archmage::intrinsics::x86_64::*` | `magetypes::simd::v4::*` |
/// | `X64V4xToken` | `archmage::intrinsics::x86_64::*` | `magetypes::simd::v4x::*` |
/// | `NeonToken` / ARM | `archmage::intrinsics::aarch64::*` | `magetypes::simd::neon::*` |
/// | `Wasm128Token` | `archmage::intrinsics::wasm32::*` | `magetypes::simd::wasm128::*` |
///
/// Works with concrete tokens, `impl Trait` bounds, and generic parameters.
///
/// # Supported Tokens
///
/// - **x86_64**: `X64V2Token`, `X64V3Token`/`Desktop64`, `X64V4Token`/`Avx512Token`/`Server64`,
///   `X64V4xToken`, `Avx512Fp16Token`, `X64CryptoToken`, `X64V3CryptoToken`
/// - **ARM**: `NeonToken`/`Arm64`, `Arm64V2Token`, `Arm64V3Token`,
///   `NeonAesToken`, `NeonSha3Token`, `NeonCrcToken`
/// - **WASM**: `Wasm128Token`
///
/// # Supported Trait Bounds
///
/// `HasX64V2`, `HasX64V4`, `HasNeon`, `HasNeonAes`, `HasNeonSha3`, `HasArm64V2`, `HasArm64V3`
///
/// ```ignore
/// #![feature(target_feature_inline_always)]
///
/// #[arcane(inline_always)]
/// fn fast_kernel(token: Avx2Token, data: &mut [f32]) {
///     // Inner function will use #[inline(always)]
/// }
/// ```
#[proc_macro_attribute]
pub fn arcane(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as ArcaneArgs);
    let input_fn = parse_macro_input!(item as LightFn);
    arcane_impl(input_fn, "arcane", args)
}

/// Legacy alias for [`arcane`].
///
/// **Deprecated:** Use `#[arcane]` instead. This alias exists only for migration.
#[proc_macro_attribute]
#[doc(hidden)]
pub fn simd_fn(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as ArcaneArgs);
    let input_fn = parse_macro_input!(item as LightFn);
    arcane_impl(input_fn, "simd_fn", args)
}

/// Descriptive alias for [`arcane`].
///
/// Generates a safe wrapper around a `#[target_feature]` inner function.
/// The token type in your signature determines which CPU features are enabled.
/// Creates an LLVM optimization boundary — use [`token_target_features`]
/// (alias for [`rite`]) for inner helpers to avoid this.
///
/// Since Rust 1.87, value-based SIMD intrinsics are safe inside
/// `#[target_feature]` functions. This macro generates the `#[target_feature]`
/// wrapper so you never need to write `unsafe` for SIMD code.
///
/// See [`arcane`] for full documentation and examples.
#[proc_macro_attribute]
pub fn token_target_features_boundary(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as ArcaneArgs);
    let input_fn = parse_macro_input!(item as LightFn);
    arcane_impl(input_fn, "token_target_features_boundary", args)
}

// ============================================================================
// Rite macro for inner SIMD functions (inlines into matching #[target_feature] callers)
// ============================================================================

/// Annotate inner SIMD helpers called from `#[arcane]` functions.
///
/// Unlike `#[arcane]`, which creates an inner `#[target_feature]` function behind
/// a safe boundary, `#[rite]` adds `#[target_feature]` and `#[inline]` directly.
/// LLVM inlines it into any caller with matching features — no boundary crossing.
///
/// # Three Modes
///
/// **Token-based:** Reads the token type from the function signature.
/// ```ignore
/// #[rite]
/// fn helper(_: X64V3Token, v: __m256) -> __m256 { _mm256_add_ps(v, v) }
/// ```
///
/// **Tier-based:** Specify the tier name directly, no token parameter needed.
/// ```ignore
/// #[rite(v3)]
/// fn helper(v: __m256) -> __m256 { _mm256_add_ps(v, v) }
/// ```
///
/// Both produce identical code. The token form can be easier to remember if
/// you already have the token in scope.
///
/// **Multi-tier:** Specify multiple tiers to generate suffixed variants.
/// ```ignore
/// #[rite(v3, v4)]
/// fn process(data: &[f32; 4]) -> f32 { data.iter().sum() }
/// // Generates: process_v3() and process_v4()
/// ```
///
/// Each variant gets its own `#[target_feature]` and `#[cfg(target_arch)]`.
/// Since Rust 1.86, calling these from a matching `#[arcane]` or `#[rite]`
/// context is safe — no `unsafe` needed when the caller has matching or
/// superset features.
///
/// # Safety
///
/// `#[rite]` functions can only be safely called from contexts where the
/// required CPU features are enabled:
/// - From within `#[arcane]` functions with matching/superset tokens
/// - From within other `#[rite]` functions with matching/superset tokens
/// - From code compiled with `-Ctarget-cpu` that enables the features
///
/// Calling from other contexts requires `unsafe` and the caller must ensure
/// the CPU supports the required features.
///
/// # Cross-Architecture Behavior
///
/// Like `#[arcane]`, defaults to cfg-out (no function on wrong arch).
/// Use `#[rite(stub)]` to generate an unreachable stub instead.
///
/// # Options
///
/// | Option | Effect |
/// |--------|--------|
/// | tier name(s) | `v3`, `neon`, etc. One = single function; multiple = suffixed variants |
/// | `stub` | Generate `unreachable!()` stub on wrong architecture |
/// | `import_intrinsics` | Auto-import `archmage::intrinsics::{arch}::*` (includes safe memory ops) |
/// | `import_magetypes` | Auto-import `magetypes::simd::{ns}::*` and `magetypes::simd::backends::*` |
///
/// See `#[arcane]` docs for the full namespace mapping table.
///
/// # Comparison with #[arcane]
///
/// | Aspect | `#[arcane]` | `#[rite]` |
/// |--------|-------------|-----------|
/// | Creates wrapper | Yes | No |
/// | Entry point | Yes | No |
/// | Inlines into caller | No (barrier) | Yes |
/// | Safe to call anywhere | Yes (with token) | Only from feature-enabled context |
/// | Multi-tier variants | No | Yes (`#[rite(v3, v4, neon)]`) |
/// | `stub` param | Yes | Yes |
/// | `import_intrinsics` | Yes | Yes |
/// | `import_magetypes` | Yes | Yes |
#[proc_macro_attribute]
pub fn rite(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as RiteArgs);
    let input_fn = parse_macro_input!(item as LightFn);
    rite_impl(input_fn, args)
}

/// Descriptive alias for [`rite`].
///
/// Applies `#[target_feature]` + `#[inline]` based on the token type in your
/// function signature. No wrapper, no optimization boundary. Use for functions
/// called from within `#[arcane]`/`#[token_target_features_boundary]` code.
///
/// Since Rust 1.86, calling a `#[target_feature]` function from another function
/// with matching features is safe — no `unsafe` needed.
///
/// See [`rite`] for full documentation and examples.
#[proc_macro_attribute]
pub fn token_target_features(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as RiteArgs);
    let input_fn = parse_macro_input!(item as LightFn);
    rite_impl(input_fn, args)
}

// RiteArgs, rite_impl, rite_single_impl, rite_multi_tier_impl → moved to rite.rs

// =============================================================================
// magetypes! macro - generate platform variants from generic function
// =============================================================================

/// Generate platform-specific variants from a function by replacing `Token`.
///
/// Use `Token` as a placeholder for the token type. The macro generates
/// suffixed variants with `Token` replaced by the concrete token type, and
/// each variant wrapped in the appropriate `#[cfg(target_arch = ...)]` guard.
///
/// # Default tiers
///
/// Without arguments, generates `_v3`, `_v4`, `_neon`, `_wasm128`, `_scalar`:
///
/// ```rust,ignore
/// #[magetypes]
/// fn process(token: Token, data: &[f32]) -> f32 {
///     inner_simd_work(token, data)
/// }
/// ```
///
/// # Explicit tiers
///
/// Specify which tiers to generate:
///
/// ```rust,ignore
/// #[magetypes(v1, v3, neon)]
/// fn process(token: Token, data: &[f32]) -> f32 {
///     inner_simd_work(token, data)
/// }
/// // Generates: process_v1, process_v3, process_neon, process_scalar
/// ```
///
/// `scalar` is always included implicitly.
///
/// Known tiers: `v1`, `v2`, `v3`, `v4`, `v4x`, `neon`, `neon_aes`,
/// `neon_sha3`, `neon_crc`, `wasm128`, `wasm128_relaxed`, `scalar`.
///
/// # What gets replaced
///
/// **Only `Token`** is replaced — with the concrete token type for each variant
/// (e.g., `archmage::X64V3Token`, `archmage::ScalarToken`). SIMD types like
/// `f32x8` and constants like `LANES` are **not** replaced by this macro.
///
/// # Usage with incant!
///
/// The generated variants work with `incant!` for dispatch:
///
/// ```rust,ignore
/// pub fn process_api(data: &[f32]) -> f32 {
///     incant!(process(data))
/// }
///
/// // Or with matching explicit tiers:
/// pub fn process_api(data: &[f32]) -> f32 {
///     incant!(process(data), [v1, v3, neon, scalar])
/// }
/// ```
#[proc_macro_attribute]
pub fn magetypes(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as LightFn);

    // Parse optional tier list from attribute args: tier1, tier2(feature), ...
    let tier_names: Vec<String> = if attr.is_empty() {
        DEFAULT_TIER_NAMES.iter().map(|s| s.to_string()).collect()
    } else {
        match syn::parse::Parser::parse(parse_tier_names, attr) {
            Ok(names) => names,
            Err(e) => return e.to_compile_error().into(),
        }
    };

    // default_optional: tiers with cfg_feature are optional by default
    let tiers = match resolve_tiers(
        &tier_names,
        input_fn.sig.ident.span(),
        true, // magetypes always uses default_optional for cfg_feature tiers
    ) {
        Ok(t) => t,
        Err(e) => return e.to_compile_error().into(),
    };

    magetypes_impl(input_fn, &tiers)
}

// =============================================================================
// incant! macro - dispatch to platform-specific variants
// =============================================================================
// incant! macro - dispatch to platform-specific variants
// =============================================================================

/// Dispatch to platform-specific SIMD variants.
///
/// # Entry Point Mode (no token yet)
///
/// Summons tokens and dispatches to the best available variant:
///
/// ```rust,ignore
/// pub fn public_api(data: &[f32]) -> f32 {
///     incant!(dot(Token, data))
/// }
/// ```
///
/// Expands to runtime feature detection + dispatch to `dot_v3`, `dot_v4`,
/// `dot_neon`, `dot_wasm128`, or `dot_scalar`. The `Token` marker is
/// replaced with the summoned token. Token can appear at any position
/// to match the callee's signature:
///
/// ```rust,ignore
/// incant!(process(Token, data), [v3, scalar])  // token-first
/// incant!(process(data, Token), [v3, scalar])  // token-last
/// ```
///
/// If `Token` is omitted, the token is prepended (backward compatible).
///
/// # Explicit Tiers
///
/// Specify which tiers to dispatch to:
///
/// ```rust,ignore
/// pub fn api(data: &[f32]) -> f32 {
///     incant!(process(Token, data), [v1, v3, neon, scalar])
/// }
/// ```
///
/// Always include `scalar` in explicit tier lists. Currently auto-appended
/// if omitted; will become a compile error in v1.0. Tiers are automatically
/// sorted by dispatch priority (highest first).
///
/// Known tiers: `v1`, `v2`, `v3`, `v4`, `v4x`, `neon`, `neon_aes`,
/// `neon_sha3`, `neon_crc`, `wasm128`, `wasm128_relaxed`, `scalar`.
///
/// # Automatic Rewriting (inside tier macros)
///
/// When `incant!` appears inside an `#[arcane]`, `#[rite]`, or
/// `#[autoversion]` function body, the outer macro **rewrites** it to
/// a direct call at compile time — bypassing the runtime dispatcher:
///
/// ```rust,ignore
/// #[arcane]
/// fn outer(token: X64V3Token, data: &[f32]) -> f32 {
///     // Rewritten to: inner_v3(token, data) — zero overhead
///     incant!(inner(token, data), [v3, scalar])
/// }
/// ```
///
/// The rewriter recognizes the caller's token variable by name and
/// handles downcasting (V4 caller → V3 callee), upgrade attempts
/// (summon a higher tier), and feature-gated tiers automatically.
///
/// Use `Token` or the caller's token variable name in the args to
/// control token position:
///
/// ```rust,ignore
/// #[arcane]
/// fn outer(my_token: X64V3Token, data: &[f32]) -> f32 {
///     // my_token recognized, placed where it appears in args
///     incant!(inner(data, my_token), [v3, scalar])
/// }
/// ```
///
/// # Passthrough Mode (generic token dispatch)
///
/// For functions generic over token types, use `with token` for
/// compile-time dispatch via `IntoConcreteToken`:
///
/// ```rust,ignore
/// fn dispatch<T: IntoConcreteToken>(token: T, data: &[f32]) -> f32 {
///     incant!(process(data) with token, [v3, neon, scalar])
/// }
/// ```
///
/// The compiler monomorphizes the dispatch — when `T = X64V3Token`,
/// only the V3 branch survives. No runtime summon, no overhead.
///
/// This is different from the rewriter: passthrough works on generic
/// `IntoConcreteToken` bounds where the concrete tier isn't known at
/// macro time. The rewriter works when the concrete tier IS known
/// (inside `#[arcane]`/`#[rite]`/`#[autoversion]` bodies).
///
/// # Variant Naming
///
/// Functions must have suffixed variants matching the selected tiers:
/// - `_v1` for `X64V1Token`
/// - `_v2` for `X64V2Token`
/// - `_v3` for `X64V3Token`
/// - `_v4` for `X64V4Token` (requires `avx512` feature)
/// - `_v4x` for `X64V4xToken` (requires `avx512` feature)
/// - `_neon` for `NeonToken`
/// - `_neon_aes` for `NeonAesToken`
/// - `_neon_sha3` for `NeonSha3Token`
/// - `_neon_crc` for `NeonCrcToken`
/// - `_wasm128` for `Wasm128Token`
/// - `_scalar` for `ScalarToken`
#[proc_macro]
pub fn incant(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as IncantInput);
    incant_impl(input)
}

/// Legacy alias for [`incant!`].
#[proc_macro]
pub fn simd_route(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as IncantInput);
    incant_impl(input)
}

/// Descriptive alias for [`incant!`].
///
/// Dispatches to architecture-specific function variants at runtime.
/// Looks for suffixed functions (`_v3`, `_v4`, `_neon`, `_wasm128`, `_scalar`)
/// and calls the best one the CPU supports.
///
/// See [`incant!`] for full documentation and examples.
#[proc_macro]
pub fn dispatch_variant(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as IncantInput);
    incant_impl(input)
}

// =============================================================================

/// Let the compiler auto-vectorize scalar code for each architecture.
///
/// Write a plain scalar function and let `#[autoversion]` generate
/// architecture-specific copies — each compiled with different
/// `#[target_feature]` flags via `#[arcane]` — plus a runtime dispatcher
/// that calls the best one the CPU supports.
///
/// # Quick start
///
/// ```rust,ignore
/// use archmage::autoversion;
///
/// #[autoversion]
/// fn sum_of_squares(data: &[f32]) -> f32 {
///     let mut sum = 0.0f32;
///     for &x in data {
///         sum += x * x;
///     }
///     sum
/// }
///
/// // Call directly — no token, no unsafe:
/// let result = sum_of_squares(&my_data);
/// ```
///
/// Each variant gets `#[arcane]` → `#[target_feature(enable = "avx2,fma,...")]`,
/// which unlocks the compiler's auto-vectorizer for that feature set.
/// On x86-64, that loop compiles to `vfmadd231ps`. On aarch64, `fmla`.
/// The `_scalar` fallback compiles without SIMD target features.
///
/// # SimdToken — optional placeholder
///
/// You can optionally write `_token: SimdToken` as a parameter. The macro
/// recognizes it and strips it from the dispatcher — both forms produce
/// identical output. Prefer the tokenless form for new code.
///
/// ```rust,ignore
/// #[autoversion]
/// fn normalize(_token: SimdToken, data: &mut [f32], scale: f32) {
///     for x in data.iter_mut() { *x = (*x - 128.0) * scale; }
/// }
/// // Dispatcher is: fn normalize(data: &mut [f32], scale: f32)
/// ```
///
/// # What gets generated
///
/// `#[autoversion] fn process(data: &[f32]) -> f32` expands to:
///
/// - `process_v4(token: X64V4Token, ...)` — AVX-512
/// - `process_v3(token: X64V3Token, ...)` — AVX2+FMA
/// - `process_neon(token: NeonToken, ...)` — aarch64 NEON
/// - `process_wasm128(token: Wasm128Token, ...)` — WASM SIMD
/// - `process_scalar(token: ScalarToken, ...)` — no SIMD, always available
/// - `process(data: &[f32]) -> f32` — **dispatcher**
///
/// Variants are private. The dispatcher gets the original function's visibility.
/// Within the same module, call variants directly for testing or benchmarking.
///
/// # Explicit tiers
///
/// ```rust,ignore
/// #[autoversion(v3, v4, neon, arm_v2, wasm128)]
/// fn process(data: &[f32]) -> f32 { ... }
/// ```
///
/// `scalar` is always included implicitly.
///
/// Default tiers: `v4`, `v3`, `neon`, `wasm128`, `scalar`.
///
/// Known tiers: `v1`, `v2`, `v3`, `v3_crypto`, `v4`, `v4x`, `neon`,
/// `neon_aes`, `neon_sha3`, `neon_crc`, `arm_v2`, `arm_v3`, `wasm128`,
/// `wasm128_relaxed`, `x64_crypto`, `scalar`.
///
/// # Methods
///
/// For inherent methods, `self` works naturally:
///
/// ```rust,ignore
/// impl ImageBuffer {
///     #[autoversion]
///     fn normalize(&mut self, gamma: f32) {
///         for pixel in &mut self.data {
///             *pixel = (*pixel / 255.0).powf(gamma);
///         }
///     }
/// }
/// buffer.normalize(2.2);
/// ```
///
/// For trait method delegation, use `_self = Type` (nested mode):
///
/// ```rust,ignore
/// impl MyType {
///     #[autoversion(_self = MyType)]
///     fn compute_impl(&self, data: &[f32]) -> f32 {
///         _self.weights.iter().zip(data).map(|(w, d)| w * d).sum()
///     }
/// }
/// ```
///
/// # Nesting with `incant!`
///
/// Hand-written SIMD for specific tiers, autoversion for the rest:
///
/// ```rust,ignore
/// pub fn process(data: &[f32]) -> f32 {
///     incant!(process(data), [v4, scalar])
/// }
///
/// #[arcane(import_intrinsics)]
/// fn process_v4(_t: X64V4Token, data: &[f32]) -> f32 { /* AVX-512 */ }
///
/// // Bridge: incant! passes ScalarToken, autoversion doesn't need one
/// fn process_scalar(_: ScalarToken, data: &[f32]) -> f32 {
///     process_auto(data)
/// }
///
/// #[autoversion(v3, neon)]
/// fn process_auto(data: &[f32]) -> f32 { data.iter().sum() }
/// ```
///
/// # Comparison with `#[magetypes]` + `incant!`
///
/// | | `#[autoversion]` | `#[magetypes]` + `incant!` |
/// |---|---|---|
/// | Generates variants + dispatcher | Yes | Variants only (+ separate `incant!`) |
/// | Body touched | No (signature only) | Yes (text substitution) |
/// | Best for | Scalar auto-vectorization | Hand-written SIMD types |
#[proc_macro_attribute]
pub fn autoversion(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as AutoversionArgs);
    let input_fn = parse_macro_input!(item as LightFn);
    autoversion_impl(input_fn, args)
}

// =============================================================================
// Unit tests for token/trait recognition maps
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    use super::generated::{ALL_CONCRETE_TOKENS, ALL_TRAIT_NAMES};
    use syn::{ItemFn, ReturnType};

    #[test]
    fn every_concrete_token_is_in_token_to_features() {
        for &name in ALL_CONCRETE_TOKENS {
            assert!(
                token_to_features(name).is_some(),
                "Token `{}` exists in runtime crate but is NOT recognized by \
                 token_to_features() in the proc macro. Add it!",
                name
            );
        }
    }

    #[test]
    fn every_trait_is_in_trait_to_features() {
        for &name in ALL_TRAIT_NAMES {
            assert!(
                trait_to_features(name).is_some(),
                "Trait `{}` exists in runtime crate but is NOT recognized by \
                 trait_to_features() in the proc macro. Add it!",
                name
            );
        }
    }

    #[test]
    fn token_aliases_map_to_same_features() {
        // Desktop64 = X64V3Token
        assert_eq!(
            token_to_features("Desktop64"),
            token_to_features("X64V3Token"),
            "Desktop64 and X64V3Token should map to identical features"
        );

        // Server64 = X64V4Token = Avx512Token
        assert_eq!(
            token_to_features("Server64"),
            token_to_features("X64V4Token"),
            "Server64 and X64V4Token should map to identical features"
        );
        assert_eq!(
            token_to_features("X64V4Token"),
            token_to_features("Avx512Token"),
            "X64V4Token and Avx512Token should map to identical features"
        );

        // Arm64 = NeonToken
        assert_eq!(
            token_to_features("Arm64"),
            token_to_features("NeonToken"),
            "Arm64 and NeonToken should map to identical features"
        );
    }

    #[test]
    fn trait_to_features_includes_tokens_as_bounds() {
        // Tier tokens should also work as trait bounds
        // (for `impl X64V3Token` patterns, even though Rust won't allow it,
        // the macro processes AST before type checking)
        let tier_tokens = [
            "X64V2Token",
            "X64CryptoToken",
            "X64V3Token",
            "Desktop64",
            "Avx2FmaToken",
            "X64V4Token",
            "Avx512Token",
            "Server64",
            "X64V4xToken",
            "Avx512Fp16Token",
            "NeonToken",
            "Arm64",
            "NeonAesToken",
            "NeonSha3Token",
            "NeonCrcToken",
            "Arm64V2Token",
            "Arm64V3Token",
        ];

        for &name in &tier_tokens {
            assert!(
                trait_to_features(name).is_some(),
                "Tier token `{}` should also be recognized in trait_to_features() \
                 for use as a generic bound. Add it!",
                name
            );
        }
    }

    #[test]
    fn trait_features_are_cumulative() {
        // HasX64V4 should include all HasX64V2 features plus more
        let v2_features = trait_to_features("HasX64V2").unwrap();
        let v4_features = trait_to_features("HasX64V4").unwrap();

        for &f in v2_features {
            assert!(
                v4_features.contains(&f),
                "HasX64V4 should include v2 feature `{}` but doesn't",
                f
            );
        }

        // v4 should have more features than v2
        assert!(
            v4_features.len() > v2_features.len(),
            "HasX64V4 should have more features than HasX64V2"
        );
    }

    #[test]
    fn x64v3_trait_features_include_v2() {
        // X64V3Token as trait bound should include v2 features
        let v2 = trait_to_features("HasX64V2").unwrap();
        let v3 = trait_to_features("X64V3Token").unwrap();

        for &f in v2 {
            assert!(
                v3.contains(&f),
                "X64V3Token trait features should include v2 feature `{}` but don't",
                f
            );
        }
    }

    #[test]
    fn has_neon_aes_includes_neon() {
        let neon = trait_to_features("HasNeon").unwrap();
        let neon_aes = trait_to_features("HasNeonAes").unwrap();

        for &f in neon {
            assert!(
                neon_aes.contains(&f),
                "HasNeonAes should include NEON feature `{}`",
                f
            );
        }
    }

    #[test]
    fn no_removed_traits_are_recognized() {
        // These traits were removed in 0.3.0 and should NOT be recognized
        let removed = [
            "HasSse",
            "HasSse2",
            "HasSse41",
            "HasSse42",
            "HasAvx",
            "HasAvx2",
            "HasFma",
            "HasAvx512f",
            "HasAvx512bw",
            "HasAvx512vl",
            "HasAvx512vbmi2",
            "HasSve",
            "HasSve2",
        ];

        for &name in &removed {
            assert!(
                trait_to_features(name).is_none(),
                "Removed trait `{}` should NOT be in trait_to_features(). \
                 It was removed in 0.3.0 — users should migrate to tier traits.",
                name
            );
        }
    }

    #[test]
    fn no_nonexistent_tokens_are_recognized() {
        // These tokens don't exist and should NOT be recognized
        let fake = [
            "SveToken",
            "Sve2Token",
            "Avx512VnniToken",
            "X64V4ModernToken",
            "NeonFp16Token",
        ];

        for &name in &fake {
            assert!(
                token_to_features(name).is_none(),
                "Non-existent token `{}` should NOT be in token_to_features()",
                name
            );
        }
    }

    #[test]
    fn featureless_traits_are_not_in_registries() {
        // SimdToken and IntoConcreteToken should NOT be in any feature registry
        // because they don't map to CPU features
        for &name in FEATURELESS_TRAIT_NAMES {
            assert!(
                token_to_features(name).is_none(),
                "`{}` should NOT be in token_to_features() — it has no CPU features",
                name
            );
            assert!(
                trait_to_features(name).is_none(),
                "`{}` should NOT be in trait_to_features() — it has no CPU features",
                name
            );
        }
    }

    #[test]
    fn find_featureless_trait_detects_simdtoken() {
        let names = vec!["SimdToken".to_string()];
        assert_eq!(find_featureless_trait(&names), Some("SimdToken"));

        let names = vec!["IntoConcreteToken".to_string()];
        assert_eq!(find_featureless_trait(&names), Some("IntoConcreteToken"));

        // Feature-bearing traits should NOT be detected
        let names = vec!["HasX64V2".to_string()];
        assert_eq!(find_featureless_trait(&names), None);

        let names = vec!["HasNeon".to_string()];
        assert_eq!(find_featureless_trait(&names), None);

        // Mixed: if SimdToken is among real traits, still detected
        let names = vec!["SimdToken".to_string(), "HasX64V2".to_string()];
        assert_eq!(find_featureless_trait(&names), Some("SimdToken"));
    }

    #[test]
    fn arm64_v2_v3_traits_are_cumulative() {
        let v2_features = trait_to_features("HasArm64V2").unwrap();
        let v3_features = trait_to_features("HasArm64V3").unwrap();

        for &f in v2_features {
            assert!(
                v3_features.contains(&f),
                "HasArm64V3 should include v2 feature `{}` but doesn't",
                f
            );
        }

        assert!(
            v3_features.len() > v2_features.len(),
            "HasArm64V3 should have more features than HasArm64V2"
        );
    }

    // =========================================================================
    // resolve_tiers — additive / subtractive / override
    // =========================================================================

    fn resolve_tier_names(names: &[&str], default_gates: bool) -> Vec<String> {
        let names: Vec<String> = names.iter().map(|s| s.to_string()).collect();
        resolve_tiers(&names, proc_macro2::Span::call_site(), default_gates)
            .unwrap()
            .iter()
            .map(|rt| {
                if let Some(ref gate) = rt.feature_gate {
                    format!("{}({})", rt.name, gate)
                } else {
                    rt.name.to_string()
                }
            })
            .collect()
    }

    #[test]
    fn resolve_defaults() {
        let tiers = resolve_tier_names(&["v4", "v3", "neon", "wasm128", "scalar"], true);
        assert!(tiers.contains(&"v3".to_string()));
        assert!(tiers.contains(&"scalar".to_string()));
        // v4 gets auto-gated when default_feature_gates=true
        assert!(tiers.contains(&"v4(avx512)".to_string()));
    }

    #[test]
    fn resolve_additive_appends() {
        let tiers = resolve_tier_names(&["+v1"], true);
        assert!(tiers.contains(&"v1".to_string()));
        assert!(tiers.contains(&"v3".to_string())); // from defaults
        assert!(tiers.contains(&"scalar".to_string())); // from defaults
    }

    #[test]
    fn resolve_additive_v4_overrides_gate() {
        // +v4 should replace v4(avx512) with plain v4 (no gate)
        let tiers = resolve_tier_names(&["+v4"], true);
        assert!(tiers.contains(&"v4".to_string())); // no gate
        assert!(!tiers.iter().any(|t| t == "v4(avx512)")); // gated version gone
    }

    #[test]
    fn resolve_additive_default_replaces_scalar() {
        let tiers = resolve_tier_names(&["+default"], true);
        assert!(tiers.contains(&"default".to_string()));
        assert!(!tiers.iter().any(|t| t == "scalar")); // scalar replaced
    }

    #[test]
    fn resolve_subtractive_removes() {
        let tiers = resolve_tier_names(&["-neon", "-wasm128"], true);
        assert!(!tiers.iter().any(|t| t == "neon"));
        assert!(!tiers.iter().any(|t| t == "wasm128"));
        assert!(tiers.contains(&"v3".to_string())); // others remain
    }

    #[test]
    fn resolve_mixed_add_remove() {
        let tiers = resolve_tier_names(&["-neon", "-wasm128", "+v1"], true);
        assert!(tiers.contains(&"v1".to_string()));
        assert!(!tiers.iter().any(|t| t == "neon"));
        assert!(!tiers.iter().any(|t| t == "wasm128"));
        assert!(tiers.contains(&"v3".to_string()));
        assert!(tiers.contains(&"scalar".to_string()));
    }

    #[test]
    fn resolve_additive_duplicate_is_noop() {
        // +v3 when v3 is already in defaults — no duplicate
        let tiers = resolve_tier_names(&["+v3"], true);
        let v3_count = tiers.iter().filter(|t| t.as_str() == "v3").count();
        assert_eq!(v3_count, 1);
    }

    #[test]
    fn resolve_mixing_plus_and_plain_is_error() {
        let names: Vec<String> = vec!["+v1".into(), "v3".into()];
        let result = resolve_tiers(&names, proc_macro2::Span::call_site(), true);
        assert!(result.is_err());
    }

    #[test]
    fn resolve_underscore_tier_name() {
        let tiers = resolve_tier_names(&["_v3", "_neon", "_scalar"], false);
        assert!(tiers.contains(&"v3".to_string()));
        assert!(tiers.contains(&"neon".to_string()));
        assert!(tiers.contains(&"scalar".to_string()));
    }

    // =========================================================================
    // autoversion — argument parsing
    // =========================================================================

    #[test]
    fn autoversion_args_empty() {
        let args: AutoversionArgs = syn::parse_str("").unwrap();
        assert!(args.self_type.is_none());
        assert!(args.tiers.is_none());
    }

    #[test]
    fn autoversion_args_single_tier() {
        let args: AutoversionArgs = syn::parse_str("v3").unwrap();
        assert!(args.self_type.is_none());
        assert_eq!(args.tiers.as_ref().unwrap(), &["v3"]);
    }

    #[test]
    fn autoversion_args_tiers_only() {
        let args: AutoversionArgs = syn::parse_str("v3, v4, neon").unwrap();
        assert!(args.self_type.is_none());
        let tiers = args.tiers.unwrap();
        assert_eq!(tiers, vec!["v3", "v4", "neon"]);
    }

    #[test]
    fn autoversion_args_many_tiers() {
        let args: AutoversionArgs =
            syn::parse_str("v1, v2, v3, v4, v4x, neon, arm_v2, wasm128").unwrap();
        assert_eq!(
            args.tiers.unwrap(),
            vec!["v1", "v2", "v3", "v4", "v4x", "neon", "arm_v2", "wasm128"]
        );
    }

    #[test]
    fn autoversion_args_trailing_comma() {
        let args: AutoversionArgs = syn::parse_str("v3, v4,").unwrap();
        assert_eq!(args.tiers.as_ref().unwrap(), &["v3", "v4"]);
    }

    #[test]
    fn autoversion_args_self_only() {
        let args: AutoversionArgs = syn::parse_str("_self = MyType").unwrap();
        assert!(args.self_type.is_some());
        assert!(args.tiers.is_none());
    }

    #[test]
    fn autoversion_args_self_and_tiers() {
        let args: AutoversionArgs = syn::parse_str("_self = MyType, v3, neon").unwrap();
        assert!(args.self_type.is_some());
        let tiers = args.tiers.unwrap();
        assert_eq!(tiers, vec!["v3", "neon"]);
    }

    #[test]
    fn autoversion_args_tiers_then_self() {
        // _self can appear after tier names
        let args: AutoversionArgs = syn::parse_str("v3, neon, _self = MyType").unwrap();
        assert!(args.self_type.is_some());
        let tiers = args.tiers.unwrap();
        assert_eq!(tiers, vec!["v3", "neon"]);
    }

    #[test]
    fn autoversion_args_self_with_path_type() {
        let args: AutoversionArgs = syn::parse_str("_self = crate::MyType").unwrap();
        assert!(args.self_type.is_some());
        assert!(args.tiers.is_none());
    }

    #[test]
    fn autoversion_args_self_with_generic_type() {
        let args: AutoversionArgs = syn::parse_str("_self = Vec<u8>").unwrap();
        assert!(args.self_type.is_some());
        let ty_str = args.self_type.unwrap().to_token_stream().to_string();
        assert!(ty_str.contains("Vec"), "Expected Vec<u8>, got: {}", ty_str);
    }

    #[test]
    fn autoversion_args_self_trailing_comma() {
        let args: AutoversionArgs = syn::parse_str("_self = MyType,").unwrap();
        assert!(args.self_type.is_some());
        assert!(args.tiers.is_none());
    }

    // =========================================================================
    // autoversion — find_autoversion_token_param
    // =========================================================================

    #[test]
    fn find_autoversion_token_param_simdtoken_first() {
        let f: ItemFn =
            syn::parse_str("fn process(token: SimdToken, data: &[f32]) -> f32 {}").unwrap();
        let param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
        assert_eq!(param.index, 0);
        assert_eq!(param.ident, "token");
        assert_eq!(param.kind, AutoversionTokenKind::SimdToken);
    }

    #[test]
    fn find_autoversion_token_param_simdtoken_second() {
        let f: ItemFn =
            syn::parse_str("fn process(data: &[f32], token: SimdToken) -> f32 {}").unwrap();
        let param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
        assert_eq!(param.index, 1);
        assert_eq!(param.kind, AutoversionTokenKind::SimdToken);
    }

    #[test]
    fn find_autoversion_token_param_underscore_prefix() {
        let f: ItemFn =
            syn::parse_str("fn process(_token: SimdToken, data: &[f32]) -> f32 {}").unwrap();
        let param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
        assert_eq!(param.index, 0);
        assert_eq!(param.ident, "_token");
    }

    #[test]
    fn find_autoversion_token_param_wildcard() {
        let f: ItemFn = syn::parse_str("fn process(_: SimdToken, data: &[f32]) -> f32 {}").unwrap();
        let param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
        assert_eq!(param.index, 0);
        assert_eq!(param.ident, "__autoversion_token");
    }

    #[test]
    fn find_autoversion_token_param_scalar_token() {
        let f: ItemFn =
            syn::parse_str("fn process_scalar(_: ScalarToken, data: &[f32]) -> f32 {}").unwrap();
        let param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
        assert_eq!(param.index, 0);
        assert_eq!(param.kind, AutoversionTokenKind::ScalarToken);
    }

    #[test]
    fn find_autoversion_token_param_not_found() {
        let f: ItemFn = syn::parse_str("fn process(data: &[f32]) -> f32 {}").unwrap();
        assert!(find_autoversion_token_param(&f.sig).unwrap().is_none());
    }

    #[test]
    fn find_autoversion_token_param_no_params() {
        let f: ItemFn = syn::parse_str("fn process() {}").unwrap();
        assert!(find_autoversion_token_param(&f.sig).unwrap().is_none());
    }

    #[test]
    fn find_autoversion_token_param_concrete_token_errors() {
        let f: ItemFn =
            syn::parse_str("fn process(token: X64V3Token, data: &[f32]) -> f32 {}").unwrap();
        let err = find_autoversion_token_param(&f.sig).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("concrete token"),
            "error should mention concrete token: {msg}"
        );
        assert!(
            msg.contains("#[arcane]"),
            "error should suggest #[arcane]: {msg}"
        );
    }

    #[test]
    fn find_autoversion_token_param_neon_token_errors() {
        let f: ItemFn =
            syn::parse_str("fn process(token: NeonToken, data: &[f32]) -> f32 {}").unwrap();
        assert!(find_autoversion_token_param(&f.sig).is_err());
    }

    #[test]
    fn find_autoversion_token_param_unknown_type_ignored() {
        // Random types are just regular params, not token params
        let f: ItemFn = syn::parse_str("fn process(data: &[f32], scale: f32) -> f32 {}").unwrap();
        assert!(find_autoversion_token_param(&f.sig).unwrap().is_none());
    }

    #[test]
    fn find_autoversion_token_param_among_many() {
        let f: ItemFn = syn::parse_str(
            "fn process(a: i32, b: f64, token: SimdToken, c: &str, d: bool) -> f32 {}",
        )
        .unwrap();
        let param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
        assert_eq!(param.index, 2);
        assert_eq!(param.ident, "token");
    }

    #[test]
    fn find_autoversion_token_param_with_generics() {
        let f: ItemFn =
            syn::parse_str("fn process<T: Clone>(token: SimdToken, data: &[T]) -> T {}").unwrap();
        let param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
        assert_eq!(param.index, 0);
    }

    #[test]
    fn find_autoversion_token_param_with_where_clause() {
        let f: ItemFn = syn::parse_str(
            "fn process<T>(token: SimdToken, data: &[T]) -> T where T: Copy + Default {}",
        )
        .unwrap();
        let param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
        assert_eq!(param.index, 0);
    }

    #[test]
    fn find_autoversion_token_param_with_lifetime() {
        let f: ItemFn =
            syn::parse_str("fn process<'a>(token: SimdToken, data: &'a [f32]) -> &'a f32 {}")
                .unwrap();
        let param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
        assert_eq!(param.index, 0);
    }

    // =========================================================================
    // autoversion — tier resolution
    // =========================================================================

    #[test]
    fn autoversion_default_tiers_all_resolve() {
        let names: Vec<String> = DEFAULT_TIER_NAMES.iter().map(|s| s.to_string()).collect();
        let tiers = resolve_tiers(&names, proc_macro2::Span::call_site(), false).unwrap();
        assert!(!tiers.is_empty());
        // scalar should be present
        assert!(tiers.iter().any(|t| t.name == "scalar"));
    }

    #[test]
    fn autoversion_scalar_always_appended() {
        let names = vec!["v3".to_string(), "neon".to_string()];
        let tiers = resolve_tiers(&names, proc_macro2::Span::call_site(), false).unwrap();
        assert!(
            tiers.iter().any(|t| t.name == "scalar"),
            "scalar must be auto-appended"
        );
    }

    #[test]
    fn autoversion_scalar_not_duplicated() {
        let names = vec!["v3".to_string(), "scalar".to_string()];
        let tiers = resolve_tiers(&names, proc_macro2::Span::call_site(), false).unwrap();
        let scalar_count = tiers.iter().filter(|t| t.name == "scalar").count();
        assert_eq!(scalar_count, 1, "scalar must not be duplicated");
    }

    #[test]
    fn autoversion_tiers_sorted_by_priority() {
        let names = vec!["neon".to_string(), "v4".to_string(), "v3".to_string()];
        let tiers = resolve_tiers(&names, proc_macro2::Span::call_site(), false).unwrap();
        // v4 (priority 40) > v3 (30) > neon (20) > scalar (0)
        let priorities: Vec<u32> = tiers.iter().map(|t| t.priority).collect();
        for window in priorities.windows(2) {
            assert!(
                window[0] >= window[1],
                "Tiers not sorted by priority: {:?}",
                priorities
            );
        }
    }

    #[test]
    fn autoversion_unknown_tier_errors() {
        let names = vec!["v3".to_string(), "avx9000".to_string()];
        let result = resolve_tiers(&names, proc_macro2::Span::call_site(), false);
        match result {
            Ok(_) => panic!("Expected error for unknown tier 'avx9000'"),
            Err(e) => {
                let err_msg = e.to_string();
                assert!(
                    err_msg.contains("avx9000"),
                    "Error should mention unknown tier: {}",
                    err_msg
                );
            }
        }
    }

    #[test]
    fn autoversion_all_known_tiers_resolve() {
        // Every tier in ALL_TIERS should be findable
        for tier in ALL_TIERS {
            assert!(
                find_tier(tier.name).is_some(),
                "Tier '{}' should be findable by name",
                tier.name
            );
        }
    }

    #[test]
    fn autoversion_default_tier_list_is_sensible() {
        // Defaults should cover x86, ARM, WASM, and scalar
        let names: Vec<String> = DEFAULT_TIER_NAMES.iter().map(|s| s.to_string()).collect();
        let tiers = resolve_tiers(&names, proc_macro2::Span::call_site(), false).unwrap();

        let has_x86 = tiers.iter().any(|t| t.target_arch == Some("x86_64"));
        let has_arm = tiers.iter().any(|t| t.target_arch == Some("aarch64"));
        let has_wasm = tiers.iter().any(|t| t.target_arch == Some("wasm32"));
        let has_scalar = tiers.iter().any(|t| t.name == "scalar");

        assert!(has_x86, "Default tiers should include an x86_64 tier");
        assert!(has_arm, "Default tiers should include an aarch64 tier");
        assert!(has_wasm, "Default tiers should include a wasm32 tier");
        assert!(has_scalar, "Default tiers should include scalar");
    }

    // =========================================================================
    // autoversion — variant replacement (AST manipulation)
    // =========================================================================

    /// Mirrors what `autoversion_impl` does for a single variant: parse an
    /// ItemFn (for test convenience), rename it, swap the SimdToken param
    /// type, optionally inject the `_self` preamble for scalar+self.
    fn do_variant_replacement(func: &str, tier_name: &str, has_self: bool) -> ItemFn {
        let mut f: ItemFn = syn::parse_str(func).unwrap();
        let fn_name = f.sig.ident.to_string();

        let tier = find_tier(tier_name).unwrap();

        // Rename
        f.sig.ident = format_ident!("{}_{}", fn_name, tier.suffix);

        // Find and replace SimdToken param type (skip for "default" — tokenless)
        let token_idx = find_autoversion_token_param(&f.sig)
            .expect("should not error on SimdToken")
            .unwrap_or_else(|| panic!("No SimdToken param in: {}", func))
            .index;
        if tier_name == "default" {
            // Remove the token param for default tier
            let stmts = f.block.stmts.clone();
            let mut inputs: Vec<FnArg> = f.sig.inputs.iter().cloned().collect();
            inputs.remove(token_idx);
            f.sig.inputs = inputs.into_iter().collect();
            f.block.stmts = stmts;
        } else {
            let concrete_type: Type = syn::parse_str(tier.token_path).unwrap();
            if let FnArg::Typed(pt) = &mut f.sig.inputs[token_idx] {
                *pt.ty = concrete_type;
            }
        }

        // Fallback (scalar/default) + self: inject preamble
        if (tier_name == "scalar" || tier_name == "default") && has_self {
            let preamble: syn::Stmt = syn::parse_quote!(let _self = self;);
            f.block.stmts.insert(0, preamble);
        }

        f
    }

    #[test]
    fn variant_replacement_v3_renames_function() {
        let f = do_variant_replacement(
            "fn process(token: SimdToken, data: &[f32]) -> f32 { 0.0 }",
            "v3",
            false,
        );
        assert_eq!(f.sig.ident, "process_v3");
    }

    #[test]
    fn variant_replacement_v3_replaces_token_type() {
        let f = do_variant_replacement(
            "fn process(token: SimdToken, data: &[f32]) -> f32 { 0.0 }",
            "v3",
            false,
        );
        let first_param_ty = match &f.sig.inputs[0] {
            FnArg::Typed(pt) => pt.ty.to_token_stream().to_string(),
            _ => panic!("Expected typed param"),
        };
        assert!(
            first_param_ty.contains("X64V3Token"),
            "Expected X64V3Token, got: {}",
            first_param_ty
        );
    }

    #[test]
    fn variant_replacement_neon_produces_valid_fn() {
        let f = do_variant_replacement(
            "fn compute(token: SimdToken, data: &[f32]) -> f32 { 0.0 }",
            "neon",
            false,
        );
        assert_eq!(f.sig.ident, "compute_neon");
        let first_param_ty = match &f.sig.inputs[0] {
            FnArg::Typed(pt) => pt.ty.to_token_stream().to_string(),
            _ => panic!("Expected typed param"),
        };
        assert!(
            first_param_ty.contains("NeonToken"),
            "Expected NeonToken, got: {}",
            first_param_ty
        );
    }

    #[test]
    fn variant_replacement_wasm128_produces_valid_fn() {
        let f = do_variant_replacement(
            "fn compute(_t: SimdToken, data: &[f32]) -> f32 { 0.0 }",
            "wasm128",
            false,
        );
        assert_eq!(f.sig.ident, "compute_wasm128");
    }

    #[test]
    fn variant_replacement_scalar_produces_valid_fn() {
        let f = do_variant_replacement(
            "fn compute(token: SimdToken, data: &[f32]) -> f32 { 0.0 }",
            "scalar",
            false,
        );
        assert_eq!(f.sig.ident, "compute_scalar");
        let first_param_ty = match &f.sig.inputs[0] {
            FnArg::Typed(pt) => pt.ty.to_token_stream().to_string(),
            _ => panic!("Expected typed param"),
        };
        assert!(
            first_param_ty.contains("ScalarToken"),
            "Expected ScalarToken, got: {}",
            first_param_ty
        );
    }

    #[test]
    fn variant_replacement_v4_produces_valid_fn() {
        let f = do_variant_replacement(
            "fn transform(token: SimdToken, data: &mut [f32]) { }",
            "v4",
            false,
        );
        assert_eq!(f.sig.ident, "transform_v4");
        let first_param_ty = match &f.sig.inputs[0] {
            FnArg::Typed(pt) => pt.ty.to_token_stream().to_string(),
            _ => panic!("Expected typed param"),
        };
        assert!(
            first_param_ty.contains("X64V4Token"),
            "Expected X64V4Token, got: {}",
            first_param_ty
        );
    }

    #[test]
    fn variant_replacement_v4x_produces_valid_fn() {
        let f = do_variant_replacement(
            "fn transform(token: SimdToken, data: &mut [f32]) { }",
            "v4x",
            false,
        );
        assert_eq!(f.sig.ident, "transform_v4x");
    }

    #[test]
    fn variant_replacement_arm_v2_produces_valid_fn() {
        let f = do_variant_replacement(
            "fn transform(token: SimdToken, data: &mut [f32]) { }",
            "arm_v2",
            false,
        );
        assert_eq!(f.sig.ident, "transform_arm_v2");
    }

    #[test]
    fn variant_replacement_preserves_generics() {
        let f = do_variant_replacement(
            "fn process<T: Copy + Default>(token: SimdToken, data: &[T]) -> T { T::default() }",
            "v3",
            false,
        );
        assert_eq!(f.sig.ident, "process_v3");
        // Generic params should still be present
        assert!(
            !f.sig.generics.params.is_empty(),
            "Generics should be preserved"
        );
    }

    #[test]
    fn variant_replacement_preserves_where_clause() {
        let f = do_variant_replacement(
            "fn process<T>(token: SimdToken, data: &[T]) -> T where T: Copy + Default { T::default() }",
            "v3",
            false,
        );
        assert!(
            f.sig.generics.where_clause.is_some(),
            "Where clause should be preserved"
        );
    }

    #[test]
    fn variant_replacement_preserves_return_type() {
        let f = do_variant_replacement(
            "fn process(token: SimdToken, data: &[f32]) -> Vec<f32> { vec![] }",
            "neon",
            false,
        );
        let ret = f.sig.output.to_token_stream().to_string();
        assert!(
            ret.contains("Vec"),
            "Return type should be preserved, got: {}",
            ret
        );
    }

    #[test]
    fn variant_replacement_preserves_multiple_params() {
        let f = do_variant_replacement(
            "fn process(token: SimdToken, a: &[f32], b: &[f32], scale: f32) -> f32 { 0.0 }",
            "v3",
            false,
        );
        // SimdToken → X64V3Token, plus the 3 other params
        assert_eq!(f.sig.inputs.len(), 4);
    }

    #[test]
    fn variant_replacement_preserves_no_return_type() {
        let f = do_variant_replacement(
            "fn transform(token: SimdToken, data: &mut [f32]) { }",
            "v3",
            false,
        );
        assert!(
            matches!(f.sig.output, ReturnType::Default),
            "No return type should remain as Default"
        );
    }

    #[test]
    fn variant_replacement_preserves_lifetime_params() {
        let f = do_variant_replacement(
            "fn process<'a>(token: SimdToken, data: &'a [f32]) -> &'a [f32] { data }",
            "v3",
            false,
        );
        assert!(!f.sig.generics.params.is_empty());
    }

    #[test]
    fn variant_replacement_scalar_self_injects_preamble() {
        let f = do_variant_replacement(
            "fn method(token: SimdToken, data: &[f32]) -> f32 { 0.0 }",
            "scalar",
            true, // has_self
        );
        assert_eq!(f.sig.ident, "method_scalar");

        // First statement should be `let _self = self;`
        let body_str = f.block.to_token_stream().to_string();
        assert!(
            body_str.contains("let _self = self"),
            "Scalar+self variant should have _self preamble, got: {}",
            body_str
        );
    }

    #[test]
    fn variant_replacement_all_default_tiers_produce_valid_fns() {
        let names: Vec<String> = DEFAULT_TIER_NAMES.iter().map(|s| s.to_string()).collect();
        let tiers = resolve_tiers(&names, proc_macro2::Span::call_site(), false).unwrap();

        for tier in &tiers {
            let f = do_variant_replacement(
                "fn process(token: SimdToken, data: &[f32]) -> f32 { 0.0 }",
                tier.name,
                false,
            );
            let expected_name = format!("process_{}", tier.suffix);
            assert_eq!(
                f.sig.ident.to_string(),
                expected_name,
                "Tier '{}' should produce function '{}'",
                tier.name,
                expected_name
            );
        }
    }

    #[test]
    fn variant_replacement_all_known_tiers_produce_valid_fns() {
        for tier in ALL_TIERS {
            let f = do_variant_replacement(
                "fn compute(token: SimdToken, data: &[f32]) -> f32 { 0.0 }",
                tier.name,
                false,
            );
            let expected_name = format!("compute_{}", tier.suffix);
            assert_eq!(
                f.sig.ident.to_string(),
                expected_name,
                "Tier '{}' should produce function '{}'",
                tier.name,
                expected_name
            );
        }
    }

    #[test]
    fn variant_replacement_no_simdtoken_remains() {
        for tier in ALL_TIERS {
            let f = do_variant_replacement(
                "fn compute(token: SimdToken, data: &[f32]) -> f32 { 0.0 }",
                tier.name,
                false,
            );
            let full_str = f.to_token_stream().to_string();
            assert!(
                !full_str.contains("SimdToken"),
                "Tier '{}' variant still contains 'SimdToken': {}",
                tier.name,
                full_str
            );
        }
    }

    // =========================================================================
    // autoversion — cfg guard and tier descriptor properties
    // =========================================================================

    #[test]
    fn tier_v3_targets_x86_64() {
        let tier = find_tier("v3").unwrap();
        assert_eq!(tier.target_arch, Some("x86_64"));
    }

    #[test]
    fn tier_v4_targets_x86_64() {
        let tier = find_tier("v4").unwrap();
        assert_eq!(tier.target_arch, Some("x86_64"));
    }

    #[test]
    fn tier_v4x_targets_x86_64() {
        let tier = find_tier("v4x").unwrap();
        assert_eq!(tier.target_arch, Some("x86_64"));
    }

    #[test]
    fn tier_neon_targets_aarch64() {
        let tier = find_tier("neon").unwrap();
        assert_eq!(tier.target_arch, Some("aarch64"));
    }

    #[test]
    fn tier_wasm128_targets_wasm32() {
        let tier = find_tier("wasm128").unwrap();
        assert_eq!(tier.target_arch, Some("wasm32"));
    }

    #[test]
    fn tier_scalar_has_no_guards() {
        let tier = find_tier("scalar").unwrap();
        assert_eq!(tier.target_arch, None);
        assert_eq!(tier.priority, 0);
    }

    #[test]
    fn tier_priorities_are_consistent() {
        // Higher-capability tiers within the same arch should have higher priority
        let v2 = find_tier("v2").unwrap();
        let v3 = find_tier("v3").unwrap();
        let v4 = find_tier("v4").unwrap();
        assert!(v4.priority > v3.priority);
        assert!(v3.priority > v2.priority);

        let neon = find_tier("neon").unwrap();
        let arm_v2 = find_tier("arm_v2").unwrap();
        let arm_v3 = find_tier("arm_v3").unwrap();
        assert!(arm_v3.priority > arm_v2.priority);
        assert!(arm_v2.priority > neon.priority);

        // scalar is lowest
        let scalar = find_tier("scalar").unwrap();
        assert!(neon.priority > scalar.priority);
        assert!(v2.priority > scalar.priority);
    }

    // =========================================================================
    // autoversion — dispatcher structure
    // =========================================================================

    #[test]
    fn dispatcher_param_removal_free_fn() {
        // Simulate what autoversion_impl does: remove the SimdToken param
        let f: ItemFn =
            syn::parse_str("fn process(token: SimdToken, data: &[f32], scale: f32) -> f32 { 0.0 }")
                .unwrap();

        let token_param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
        // SimdToken → strip from dispatcher
        assert_eq!(token_param.kind, AutoversionTokenKind::SimdToken);
        let mut dispatcher_inputs: Vec<FnArg> = f.sig.inputs.iter().cloned().collect();
        dispatcher_inputs.remove(token_param.index);
        assert_eq!(dispatcher_inputs.len(), 2);
    }

    #[test]
    fn dispatcher_param_removal_token_only() {
        let f: ItemFn = syn::parse_str("fn process(token: SimdToken) -> f32 { 0.0 }").unwrap();
        let token_param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
        let mut dispatcher_inputs: Vec<FnArg> = f.sig.inputs.iter().cloned().collect();
        dispatcher_inputs.remove(token_param.index);
        assert_eq!(dispatcher_inputs.len(), 0);
    }

    #[test]
    fn dispatcher_param_removal_token_last() {
        let f: ItemFn =
            syn::parse_str("fn process(data: &[f32], scale: f32, token: SimdToken) -> f32 { 0.0 }")
                .unwrap();
        let token_param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
        assert_eq!(token_param.index, 2);
        let mut dispatcher_inputs: Vec<FnArg> = f.sig.inputs.iter().cloned().collect();
        dispatcher_inputs.remove(token_param.index);
        assert_eq!(dispatcher_inputs.len(), 2);
    }

    #[test]
    fn dispatcher_scalar_token_kept() {
        // ScalarToken is a real type — kept in dispatcher for incant! compatibility
        let f: ItemFn =
            syn::parse_str("fn process_scalar(_: ScalarToken, data: &[f32]) -> f32 { 0.0 }")
                .unwrap();
        let token_param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
        assert_eq!(token_param.kind, AutoversionTokenKind::ScalarToken);
        // Should NOT be removed — dispatcher keeps it
        assert_eq!(f.sig.inputs.len(), 2);
    }

    #[test]
    fn dispatcher_dispatch_args_extraction() {
        // Test that we correctly extract idents for the dispatch call
        let f: ItemFn =
            syn::parse_str("fn process(data: &[f32], scale: f32) -> f32 { 0.0 }").unwrap();

        let dispatch_args: Vec<String> = f
            .sig
            .inputs
            .iter()
            .filter_map(|arg| {
                if let FnArg::Typed(PatType { pat, .. }) = arg {
                    if let syn::Pat::Ident(pi) = pat.as_ref() {
                        return Some(pi.ident.to_string());
                    }
                }
                None
            })
            .collect();

        assert_eq!(dispatch_args, vec!["data", "scale"]);
    }

    #[test]
    fn dispatcher_wildcard_params_get_renamed() {
        let f: ItemFn = syn::parse_str("fn process(_: &[f32], _: f32) -> f32 { 0.0 }").unwrap();

        let mut dispatcher_inputs: Vec<FnArg> = f.sig.inputs.iter().cloned().collect();

        let mut wild_counter = 0u32;
        for arg in &mut dispatcher_inputs {
            if let FnArg::Typed(pat_type) = arg {
                if matches!(pat_type.pat.as_ref(), syn::Pat::Wild(_)) {
                    let ident = format_ident!("__autoversion_wild_{}", wild_counter);
                    wild_counter += 1;
                    *pat_type.pat = syn::Pat::Ident(syn::PatIdent {
                        attrs: vec![],
                        by_ref: None,
                        mutability: None,
                        ident,
                        subpat: None,
                    });
                }
            }
        }

        // Both wildcards should be renamed
        assert_eq!(wild_counter, 2);

        let names: Vec<String> = dispatcher_inputs
            .iter()
            .filter_map(|arg| {
                if let FnArg::Typed(PatType { pat, .. }) = arg {
                    if let syn::Pat::Ident(pi) = pat.as_ref() {
                        return Some(pi.ident.to_string());
                    }
                }
                None
            })
            .collect();

        assert_eq!(names, vec!["__autoversion_wild_0", "__autoversion_wild_1"]);
    }

    // =========================================================================
    // autoversion — suffix_path (reused in dispatch)
    // =========================================================================

    #[test]
    fn suffix_path_simple() {
        let path: syn::Path = syn::parse_str("process").unwrap();
        let suffixed = suffix_path(&path, "v3");
        assert_eq!(suffixed.to_token_stream().to_string(), "process_v3");
    }

    #[test]
    fn suffix_path_qualified() {
        let path: syn::Path = syn::parse_str("module::process").unwrap();
        let suffixed = suffix_path(&path, "neon");
        let s = suffixed.to_token_stream().to_string();
        assert!(
            s.contains("process_neon"),
            "Expected process_neon, got: {}",
            s
        );
    }
}
