//! `#[magetypes]` — generate per-tier function variants via text substitution.

use proc_macro::TokenStream;
use quote::{ToTokens, quote};

use crate::common::*;
use crate::tiers::*;

pub(crate) fn magetypes_impl(mut input_fn: LightFn, tiers: &[ResolvedTier]) -> TokenStream {
    // Strip user-provided #[arcane] / #[rite] to prevent double-wrapping
    // (magetypes auto-adds #[arcane] on non-scalar variants)
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

    let mut variants = Vec::new();

    for tier in tiers {
        // Clone and rename at the AST level (no string surgery)
        let mut variant_fn = input_fn.clone();
        variant_fn.sig.ident = quote::format_ident!("{}_{}", fn_name, tier.suffix);
        // Propagate doc comments, #[allow], etc. to each variant
        variant_fn.attrs = propagated_attrs.clone();

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
            // Non-fallback variants get #[arcane] so target_feature is applied
            quote! {
                #cfg_guard
                #[archmage::arcane]
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
