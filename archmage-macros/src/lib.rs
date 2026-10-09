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
mod attune;
mod autoversion;
mod common;
mod engine;
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
use quote::ToTokens;
#[cfg(test)]
use syn::{FnArg, PatType};
#[cfg(test)]
use token_discovery::*;

// LightFn, filter_inline_attrs, is_lint_attr, filter_lint_attrs, gen_cfg_guard,
// build_turbofish, replace_self_in_tokens, suffix_path → moved to common.rs
// ArcaneArgs, arcane_impl, arcane_impl_* → moved to arcane.rs
// generate_imports → moved to common.rs

#[doc = include_str!("docs/arcane.md")]
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
///
/// A globally enabled feature (`-Ctarget-cpu`, `-Ctarget-feature`) does not
/// make the call safe: rustc requires the caller's own `#[target_feature]`
/// to cover the callee's, whatever the build enables (E0133 otherwise).
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
/// `#[rite]` has no trait-method mode: it applies `#[target_feature]` directly,
/// which rustc rejects on a safe trait method on x86-64 and AArch64 (wasm32's
/// `simd128` functions are safe to call anywhere, so there it compiles).
/// `#[rite(in_trait)]` says so and points at `#[arcane(in_trait, _self = Type)]`;
/// a plain `#[rite]` on such a method leaves rustc's error on it. Inside an
/// inherent impl no flag is needed, so `in_impl` is rejected too.
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
/// # Options
///
/// | Option | Effect |
/// |--------|--------|
/// | `rite` | Variants use `#[rite(import_intrinsics)]` (no wrapper) instead of `#[arcane]` |
/// | `define(f32x8, ...)` | Alias each named generic vector to its `Token` instantiation inside the body |
/// | `in_impl` | Receiver-less associated function in an inherent impl: variants get `#[arcane(in_impl)]` |
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
    let (rite_flag, in_impl, defines, tier_names) =
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

    magetypes_impl(input_fn, &tiers, rite_flag, in_impl, &defines).into()
}

/// Parse `#[magetypes]` attributes: `rite` and `in_impl` flags, `define(list)`,
/// and tier names.
///
/// Returns `(rite_flag, in_impl, defines, tier_names)`. Tier names preserve the
/// `+`/`-` modifier prefixes and `(cfg(feat))` gates for the tier resolver.
fn parse_magetypes_attr(
    input: syn::parse::ParseStream,
) -> syn::Result<(bool, bool, Vec<String>, Vec<String>)> {
    use syn::Token;
    let mut rite_flag = false;
    let mut in_impl = false;
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
        let peek_in_impl = input.peek(syn::Ident) && {
            let fork = input.fork();
            fork.parse::<syn::Ident>().is_ok_and(|i| i == "in_impl")
        };

        if peek_rite {
            let _: syn::Ident = input.parse()?;
            rite_flag = true;
        } else if peek_in_impl {
            // A receiver-less associated function in an inherent impl: the
            // generated #[arcane] variants must call their siblings as `Self::`.
            let _: syn::Ident = input.parse()?;
            in_impl = true;
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

    Ok((rite_flag, in_impl, defines, tier_names))
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
/// An associated function without a receiver in an inherent impl needs
/// `in_impl`, so the dispatcher calls the variants as `Self::process_v3(...)`:
///
/// ```rust,ignore
/// impl Table {
///     #[autoversion(in_impl)]
///     fn build(n: usize) -> Self { Self::with_capacity(n) }
/// }
/// ```
///
/// A trait impl cannot take the variants as extra items, so `in_trait` (alias
/// `nested`) places them inside the dispatcher's body. A method with a receiver
/// also needs `_self = Type`: each variant takes `_self: &Type` in place of
/// the receiver, `Self` becomes the named type, and `self` in the body becomes
/// `_self`:
///
/// ```rust,ignore
/// impl Work for MyType {
///     #[autoversion(v3, neon, scalar, in_trait, _self = MyType)]
///     fn run(&self, data: &[f32]) -> f32 {
///         self.weights.iter().zip(data).map(|(w, d)| w * d).sum()
///     }
/// }
/// ```
///
/// `_self = Type` without `in_trait` keeps the variants beside the dispatcher,
/// which only an inherent impl accepts; plain `self` already works there. The
/// variants of an `in_trait` function are local to its body and cannot be
/// called directly.
///
/// `#[autoversion]` cannot dispatch an `impl Trait` return type: every variant
/// would return a distinct opaque type and the dispatcher can return only one.
/// Return a concrete type or `Box<dyn Trait>`.
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

/// Establish a feature context or generate explicit direct/proof/dispatch outputs.
///
/// `#[attune(v3)]` keeps the function name. `#[attune]` infers a registered
/// terminal tier suffix. `make(_v3, _v3_t, _)` requests a direct function,
/// a proof-taking wrapper, and a central dispatcher with a private scalar fallback.
///
/// `inline(default)` selects body hints from the operation's visibility before
/// lowering to private helpers: unrestricted `pub` gets `#[inline]`; restricted
/// or private operations get no attribute. Direct output visibility overrides
/// participate; wrapper visibility overrides apply independently.
/// `inline(none)`, `inline(hint)`, and `inline(never)` select explicit body policies.
/// Per-output `make(inline(...) _v3)` overrides the definition-level policy.
/// Definition-level policy does not alter proof-wrapper or dispatcher defaults.
#[proc_macro_attribute]
pub fn attune(attr: TokenStream, item: TokenStream) -> TokenStream {
    attune::expand(attr.into(), item.into())
        .unwrap_or_else(|error| error.to_compile_error())
        .into()
}

/// Invoke a family using the enclosing attune feature context, or runtime proof
/// entries outside one. Explicit tier lists never acquire an implicit fallback.
#[proc_macro]
pub fn attuned(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as attune::call::Call);
    input.expand(None, false).into()
}

/// Explicitly reconsider stronger tiers at runtime, retaining a guaranteed
/// fallback covered by the caller's feature context (or scalar outside one).
#[proc_macro]
pub fn reattune(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as attune::call::Call);
    input.expand(None, true).into()
}

#[cfg(test)]
mod variant_tests;
