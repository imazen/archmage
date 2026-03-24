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
    let fn_attrs = &input_fn.attrs;

    // Convert function to string for text substitution
    let fn_str = input_fn.to_token_stream().to_string();

    let mut variants = Vec::new();

    for tier in tiers {
        // Create suffixed function name
        let suffixed_name = format!("{}_{}", fn_name, tier.suffix);

        // Do text substitution
        let mut variant_str = fn_str.clone();

        // Replace function name
        variant_str = variant_str.replacen(&fn_name.to_string(), &suffixed_name, 1);

        // Replace Token type with concrete token
        variant_str = variant_str.replace("Token", tier.token_path);

        // Parse back to tokens
        let variant_tokens: proc_macro2::TokenStream = match variant_str.parse() {
            Ok(t) => t,
            Err(e) => {
                return syn::Error::new_spanned(
                    &input_fn,
                    format!(
                        "Failed to parse generated variant `{}`: {}",
                        suffixed_name, e
                    ),
                )
                .to_compile_error()
                .into();
            }
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

    // Remove attributes from the list that should not be duplicated
    let filtered_attrs: Vec<_> = fn_attrs
        .iter()
        .filter(|a| !a.path().is_ident("magetypes"))
        .collect();

    let output = quote! {
        #(#filtered_attrs)*
        #(#variants)*
    };

    output.into()
}
