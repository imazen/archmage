//! `#[magetypes]` — generate per-tier function variants via text substitution.

use proc_macro::TokenStream;
use quote::{ToTokens, quote};

use crate::common::*;
use crate::tiers::*;

/// Generate per-tier variants of the input function.
///
/// When `rite_flag` is false (default), non-fallback variants are wrapped
/// with `#[archmage::arcane]` (safe outer wrapper + `#[target_feature]`
/// inner via trampoline). When true, variants are annotated with
/// `#[archmage::rite(import_intrinsics)]` — direct `#[target_feature]` +
/// `#[inline]`, no trampoline, no optimization boundary. The rite form is
/// only safe to call from matching-feature contexts (another `#[arcane]`,
/// `#[rite]`, or `#[magetypes]`-generated variant, or via `incant!`
/// rewriting inside such a context). Standalone `incant!` dispatch at a
/// public boundary is NOT supported for rite-flavored magetypes because the
/// non-tier dispatcher can't safely call a bare `#[target_feature]` fn.
///
/// `defines` is a list of magetypes type names (e.g. `["f32x8", "u16x16"]`)
/// to inject as local type aliases at the top of each variant's body:
///
///     type f32x8 = ::magetypes::simd::generic::f32x8<Token>;
///
/// The alias's `Token` is substituted to the concrete token type for each
/// tier (same as the rest of the body). This eliminates the boilerplate
/// `type f32x8 = GenericF32x8<Token>;` line users would otherwise write
/// inside every `#[magetypes]` function body.
pub(crate) fn magetypes_impl(
    mut input_fn: LightFn,
    tiers: &[ResolvedTier],
    rite_flag: bool,
    defines: &[String],
) -> TokenStream {
    // Strip user-provided #[arcane] / #[rite] to prevent double-wrapping
    // (magetypes auto-adds one of them on non-scalar variants)
    input_fn
        .attrs
        .retain(|attr| !attr.path().is_ident("arcane") && !attr.path().is_ident("rite"));

    let fn_name = &input_fn.sig.ident;

    // Attrs to propagate to each variant: doc comments, #[allow], #[inline], etc.
    // Exclude #[magetypes] (consumed) and #[arcane]/#[rite] (already stripped above).
    let propagated_attrs: Vec<_> = input_fn
        .attrs
        .iter()
        .filter(|a| !a.path().is_ident("magetypes"))
        .cloned()
        .collect();

    // Build the `define(...)` type-alias preamble once. Each alias RHS still
    // references `Token` — the per-tier substitution below rewrites it to the
    // concrete token type (`X64V3Token`, `ScalarToken`, etc.).
    let define_preamble: proc_macro2::TokenStream = {
        let aliases = defines.iter().map(|name| {
            let ident = quote::format_ident!("{name}");
            quote! {
                #[allow(non_camel_case_types, dead_code)]
                type #ident = ::magetypes::simd::generic::#ident<Token>;
            }
        });
        quote! { #(#aliases)* }
    };

    let mut variants = Vec::new();

    for tier in tiers {
        // Clone and rename at the AST level (no string surgery)
        let mut variant_fn = input_fn.clone();
        variant_fn.sig.ident = quote::format_ident!("{}_{}", fn_name, tier.suffix);
        // Propagate doc comments, #[allow], etc. to each variant
        variant_fn.attrs = propagated_attrs.clone();

        // Prepend the `define(...)` type aliases to the body. They appear
        // inside the function scope, shadowing any outer `f32x8`/etc. for
        // this body only.
        if !defines.is_empty() {
            let original_body = &variant_fn.body;
            variant_fn.body = quote! {
                #define_preamble
                #original_body
            };
        }

        // Replace `Token` ident with the concrete token path at the token level.
        // This is safe: each identifier is a discrete token tree, so `ScalarToken`,
        // `IntoConcreteToken`, etc. are single Ident nodes that do NOT match "Token".
        let variant_tokens = if tier.token_path.is_empty() {
            // `default` tier has no token type — just emit the fn without replacement
            variant_fn.to_token_stream()
        } else {
            let concrete_tokens: proc_macro2::TokenStream = tier
                .token_path
                .parse()
                .expect("tier token_path must be valid tokens");
            replace_ident_in_tokens(variant_fn.to_token_stream(), "Token", &concrete_tokens)
        };

        // Add cfg guard: arch + optional feature gate
        let allow_attr = if tier.allow_unexpected_cfg {
            quote! { #[allow(unexpected_cfgs)] }
        } else {
            quote! {}
        };
        let cfg_guard = match (tier.target_arch, &tier.feature_gate) {
            (Some(arch), Some(feat)) => quote! {
                #[cfg(target_arch = #arch)]
                #allow_attr
                #[cfg(feature = #feat)]
            },
            (Some(arch), None) => quote! { #[cfg(target_arch = #arch)] },
            (None, Some(feat)) => quote! {
                #allow_attr
                #[cfg(feature = #feat)]
            },
            (None, None) => quote! {},
        };

        variants.push(if tier.name != "scalar" && tier.name != "default" {
            // Non-fallback variants carry target_feature. The `rite` flag
            // chooses which macro applies it:
            //   - #[archmage::arcane]: safe wrapper + #[target_feature] inner
            //     (trampoline pattern; callable from any context)
            //   - #[archmage::rite(import_intrinsics)]: direct #[target_feature]
            //     + #[inline], no wrapper (only callable from matching-feature
            //     contexts, e.g. via `incant!` rewriting from another tier body)
            let wrapper = if rite_flag {
                quote! { #[archmage::rite(import_intrinsics)] }
            } else {
                quote! { #[archmage::arcane] }
            };
            quote! {
                #cfg_guard
                #wrapper
                #variant_tokens
            }
        } else {
            quote! {
                #cfg_guard
                #variant_tokens
            }
        });
    }

    let output = quote! {
        #(#variants)*
    };

    output.into()
}
