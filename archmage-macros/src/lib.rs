//! Proc-macros for archmage SIMD capability tokens.
//!
//! [Official guide and examples](https://imazen.github.io/archmage/) ·
//! [Intrinsics browser](https://imazen.github.io/archmage/intrinsics/) ·
//! [Archmage API](https://docs.rs/archmage/latest/archmage/) ·
//! [Magetypes vector API](https://docs.rs/magetypes/latest/magetypes/)
//!
//! Applications should import these macros through `archmage`, which pins the
//! compatible macro release. `#[magetypes]` generates functions; the separate
//! magetypes crate supplies their vector types. See the
//! [complete generic call chains](https://imazen.github.io/archmage/magetypes/dispatch/types-and-dispatch/).
//!
//! Provides `#[arcane]`, `#[rite]`, `#[autoversion]`, `incant!`, and `#[magetypes]`.

#[cfg(test)]
mod expansion_tests;

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
/// `stub` has been removed. Use `incant!` or explicit call-site cfg guards.
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
///     let v = f32x8::load_t(token, data);
///     let zero = _mm256_setzero_ps();
///     // ...
/// }
///
/// // With auto-imports — clean:
/// #[arcane(import_intrinsics, import_magetypes)]
/// fn process(token: X64V3Token, data: &[f32; 8]) -> f32 {
///     let v = f32x8::load_t(token, data);
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
///   `X64V4xToken`, `Avx512Fp16Token`, `X64CryptoToken`, `X64V3CryptoToken`,
///   `X64V3GfniCryptoToken`
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
///
/// Concrete tokens are checked through a shared, tier-specific associated constant.
/// This rejects accidental token-name aliases without reevaluating a tag comparison
/// in every expansion. The matching archmage release pins this macro crate exactly.
/// Getting past the check takes deliberately shadowing archmage's type names and
/// copying their hidden constants, which gives undefined behavior on CPUs without
/// the features; see the [safety model](https://imazen.github.io/archmage/archmage/concepts/safety/).
///
/// `#[arcane(suppress_const_test)]` omits this accidental-misuse check for trusted
/// generators. The caller must ensure the actual token proves the tier selected
/// by its name. Intrinsic target-feature checking remains enabled, but does not
/// authenticate that token: it checks instructions against the generated feature
/// context. Ordinary callers should keep the default check enabled. This option
/// is intended for generators whose registry already establishes that match.
#[proc_macro_attribute]
pub fn arcane(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as ArcaneArgs);
    let input_fn = parse_macro_input!(item as LightFn);
    arcane_impl(input_fn, "arcane", args).into()
}

/// Legacy alias for [`arcane`].
///
/// **Deprecated:** Use `#[arcane]` instead. This alias exists only for migration.
#[proc_macro_attribute]
#[doc(hidden)]
pub fn simd_fn(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as ArcaneArgs);
    let input_fn = parse_macro_input!(item as LightFn);
    arcane_impl(input_fn, "simd_fn", args).into()
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
    arcane_impl(input_fn, "token_target_features_boundary", args).into()
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
/// `stub` has been removed; use `incant!` or explicit call-site cfg guards.
///
/// # Options
///
/// | Option | Effect |
/// |--------|--------|
/// | tier name(s) | `v3`, `neon`, etc. One = single function; multiple = suffixed variants |
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
/// | Inlines into caller | When feature context permits | When feature context permits |
/// | Safe to call anywhere | Yes (with token) | Only from feature-enabled context |
/// | Multi-tier variants | No | Yes (`#[rite(v3, v4, neon)]`) |
/// | `import_intrinsics` | Yes | Yes |
/// | `import_magetypes` | Yes | Yes |
#[proc_macro_attribute]
pub fn rite(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as RiteArgs);
    let input_fn = parse_macro_input!(item as LightFn);
    rite_impl(input_fn, args).into()
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
    rite_impl(input_fn, args).into()
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

    // Parse attribute args: [rite,] [define(type, ...),] tier1, tier2(feature), ...
    //
    // Special keywords:
    //   `rite`: flag that changes per-tier variants to use
    //           `#[archmage::rite(import_intrinsics)]` (direct
    //           `#[target_feature]` + `#[inline]`) instead of
    //           `#[archmage::arcane]` (safe wrapper + inner trampoline).
    //
    //   `define(name1, name2, ...)`: list of magetypes type names to inject
    //           as local type aliases at the top of each variant body
    //           (e.g., `type f32x8 = ::magetypes::simd::generic::f32x8<Token>;`).
    //           `Token` in the alias RHS is substituted per tier.
    //
    // Assumption: neither `rite` nor `define` is or will become a tier name.
    // `token-registry.toml` must not declare `short_name = "rite"` or
    // `short_name = "define"`.
    let (rite_flag, defines, tier_names) =
        match syn::parse::Parser::parse(parse_magetypes_attr, attr) {
            Ok(parsed) => parsed,
            Err(e) => return e.to_compile_error().into(),
        };

    let tiers = if tier_names.is_empty() {
        default_tiers(true)
    } else {
        match resolve_tiers(&tier_names, input_fn.sig.ident.span(), true) {
            Ok(t) => t,
            Err(e) => return e.to_compile_error().into(),
        }
    };

    magetypes_impl(input_fn, &tiers, rite_flag, &defines).into()
}

/// Parse `#[magetypes]` attributes: `rite` flag, `define(list)`, and tier names.
///
/// Returns `(rite_flag, defines, tier_names)`. Tier names preserve the
/// `+`/`-` modifier prefixes and `(cfg(feat))` gates for the tier resolver.
fn parse_magetypes_attr(
    input: syn::parse::ParseStream,
) -> syn::Result<(bool, Vec<String>, Vec<String>)> {
    use syn::Token;
    let mut rite_flag = false;
    let mut defines = Vec::new();
    let mut tier_names = Vec::new();

    while !input.is_empty() {
        // Peek the leading ident without consuming — `rite` and `define` are
        // special, anything else is a tier name (possibly prefixed with +/-).
        let peek_rite = input.peek(syn::Ident) && {
            let fork = input.fork();
            fork.parse::<syn::Ident>()
                .is_ok_and(|i| i == "rite" && !fork.peek(syn::token::Paren))
        };
        let peek_define = input.peek(syn::Ident) && {
            let fork = input.fork();
            fork.parse::<syn::Ident>()
                .is_ok_and(|i| i == "define" && fork.peek(syn::token::Paren))
        };

        if peek_rite {
            let _: syn::Ident = input.parse()?;
            rite_flag = true;
        } else if peek_define {
            let _: syn::Ident = input.parse()?;
            let content;
            syn::parenthesized!(content in input);
            while !content.is_empty() {
                let ty: syn::Ident = content.parse()?;
                defines.push(ty.to_string());
                if content.peek(Token![,]) {
                    let _: Token![,] = content.parse()?;
                }
            }
        } else {
            // Fall through to tier-name parsing (preserves +/- prefix and cfg gates).
            tier_names.push(parse_one_tier(input)?);
        }

        if input.peek(Token![,]) {
            let _: Token![,] = input.parse()?;
        }
    }

    Ok((rite_flag, defines, tier_names))
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
    incant_impl(input).into()
}

/// Legacy alias for [`incant!`].
#[proc_macro]
pub fn simd_route(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as IncantInput);
    incant_impl(input).into()
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
    incant_impl(input).into()
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
/// Known tiers: `v1`, `v2`, `v3`, `v3_crypto`, `v3_gfni_crypto`, `v4`, `v4x`,
/// `neon`, `neon_aes`, `neon_sha3`, `neon_crc`, `arm_v2`, `arm_v3`, `wasm128`,
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
    autoversion_impl(input_fn, args).into()
}

#[cfg(test)]
mod tests;
