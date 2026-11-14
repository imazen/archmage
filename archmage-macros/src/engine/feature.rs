//! A feature-enabled function, independent of attribute spelling.
use super::inline::InlinePolicy;
use crate::common::*;
use crate::generated::{
    canonical_token_to_tier_suffix, token_to_arch, token_to_features, token_to_magetypes_namespace,
};
use crate::rite::DEFAULT_TIER_SENTINEL;
use crate::token_discovery::TokenParamInfo;
use proc_macro2::TokenStream;
use quote::quote;
use syn::Ident;

/// Where a `#[rite]` variant gets its features from.
pub(crate) struct FeatureContext {
    /// Canonical token name (`X64V3Token`), or `None` for the tokenless `default` tier.
    pub(crate) token: Option<&'static str>,
    /// The tier suffix (`v3`, `scalar`, `default`), when the tier is one the
    /// registry knows; it selects the nested `incant!` rewrite.
    pub(crate) suffix: Option<&'static str>,
    pub(crate) features: std::borrow::Cow<'static, [&'static str]>,
    pub(crate) target_arch: Option<&'static str>,
    pub(crate) magetypes_namespace: Option<&'static str>,
    /// The token parameter when the features came from the signature.
    pub(crate) token_ident: Option<Ident>,
    /// Tier traits to re-state through `::archmage::` (trait or generic bound).
    pub(crate) tier_traits: Vec<&'static str>,
}

impl FeatureContext {
    /// A tier named in the attribute: `#[rite(v3)]`, or `default`.
    pub(crate) fn from_tier_token(tier_token: &str) -> Option<Self> {
        if tier_token == DEFAULT_TIER_SENTINEL {
            return Some(FeatureContext {
                token: None,
                suffix: Some("default"),
                features: std::borrow::Cow::Borrowed(&[]),
                target_arch: None,
                magetypes_namespace: None,
                token_ident: None,
                tier_traits: Vec::new(),
            });
        }
        let token = crate::generated::known_token_str(tier_token)?;
        Some(FeatureContext {
            token: Some(token),
            suffix: canonical_token_to_tier_suffix(token),
            features: std::borrow::Cow::Borrowed(token_to_features(token)?),
            target_arch: token_to_arch(token),
            magetypes_namespace: token_to_magetypes_namespace(token),
            token_ident: None,
            tier_traits: Vec::new(),
        })
    }

    pub(crate) fn from_token_param(info: TokenParamInfo) -> Self {
        FeatureContext {
            suffix: info
                .token_type_name
                .and_then(canonical_token_to_tier_suffix),
            token: info.token_type_name,
            features: info.features,
            target_arch: info.target_arch,
            magetypes_namespace: info.magetypes_namespace,
            token_ident: Some(info.ident),
            tier_traits: info.tier_traits,
        }
    }
}

/// Emit one `#[target_feature]` + `#[inline]` function for `tier`, with
/// imports, nested `incant!` rewriting, the bound assertion and the cfg guard.
/// Both the single- and multi-tier forms go through here.
pub(crate) fn emit(
    mut variant_fn: LightFn,
    options: &SharedOptions,
    inline: Option<InlinePolicy>,
    preserve_inline: bool,
    tier: FeatureContext,
) -> Result<TokenStream, TokenStream> {
    if let Some(err) = reject_manual_target_feature(&variant_fn.attrs) {
        return Err(err.to_compile_error());
    }
    let token_desc = tier.token.unwrap_or("an AVX-512 token");
    if let Some(err) = avx512_import_error(
        &variant_fn.sig,
        options.import_intrinsics,
        &tier.features,
        token_desc,
    ) {
        return Err(err);
    }

    // Rewrite incant!() calls in the body to direct tier calls.
    // - With a token param: full rewrite (token-first direct calls + `without token`).
    // - Tokenless tier form (`#[rite(v3)]`): no token to thread, so plain incant!
    //   constructs proof with from_context() for covered tiers; `without token`
    //   calls tokenless helpers; `with token` stays explicit.
    if let Some(tier_suffix) = tier.suffix
        && let Some(resolved) = crate::tiers::find_tier(tier_suffix)
    {
        let ctx = match &tier.token_ident {
            Some(ident) => crate::rewrite::CallerContext {
                tier_suffix,
                target_arch: resolved.target_arch,
                token_ident: ident.clone(),
                has_token: true,
                derive_token: false,
            },
            None => crate::rewrite::CallerContext {
                tier_suffix,
                target_arch: resolved.target_arch,
                token_ident: Ident::new("_", proc_macro2::Span::call_site()),
                has_token: false,
                derive_token: true,
            },
        };
        variant_fn.body = crate::rewrite::rewrite_incant_in_function(
            variant_fn.body,
            &ctx,
            Some(&variant_fn.sig),
        );
    }

    if tier.suffix.is_none() {
        variant_fn.body = crate::attune::call::rewrite_context(
            variant_fn.body,
            crate::attune::call::Context {
                features: &tier.features,
                target_arch: tier.target_arch,
            },
            Some(&variant_fn.sig),
        );
    }

    // Build target_feature + inline tokens directly without `parse_quote!` round-trips.
    // Scalar and default tiers have no features — emit only `#[inline]` without
    // `#[target_feature]` (enable="" is an error).
    let target_feature_tokens = if !tier.features.is_empty() {
        let features_csv = crate::token_discovery::features_csv(tier.token, &tier.features);
        quote!(#[target_feature(enable = #features_csv)])
    } else {
        TokenStream::new()
    };
    // Explicit policy replaces source inline attributes, including when it
    // deliberately emits no attribute. Legacy callers keep their old defaults.
    let inline_tokens = if let Some(policy) = inline {
        policy.tokens(&variant_fn.vis)
    } else if !preserve_inline || !variant_fn.attrs.iter().any(|a| a.path().is_ident("inline")) {
        quote!(#[inline])
    } else {
        TokenStream::new()
    };
    if !preserve_inline || inline.is_some() {
        variant_fn
            .attrs
            .retain(|attr| !attr.path().is_ident("inline"));
    }

    let body_imports = generate_imports(
        tier.target_arch,
        tier.magetypes_namespace,
        options.import_intrinsics,
        options.import_magetypes,
    );
    // `#[rite]` has no wrapper — it puts `#[target_feature]` on the function
    // itself — so a trait/generic bound is authenticated at the top of the body.
    let tier_trait_assertion = match &tier.token_ident {
        Some(ident) => gen_tier_trait_assertion(&tier.tier_traits, ident),
        None => quote! {},
    };
    prepend_to_body(
        &mut variant_fn.body,
        quote! { #body_imports #tier_trait_assertion },
    );

    // Emit the function behind its cfg guard (empty for trait bounds and the
    // default tier, which have no architecture).
    let cfg_guard = drop_attrs_equal_to(
        &mut variant_fn.attrs,
        tier.target_arch,
        options.cfg_feature.as_deref(),
    );
    mark_emitted_fn_header(&mut variant_fn.vis, &mut variant_fn.sig);
    let vis = &variant_fn.vis;
    let sig = &variant_fn.sig;
    let attrs = &variant_fn.attrs;
    let body = braced_body(&variant_fn.brace_token, &variant_fn.body);
    Ok(quote! {
        #cfg_guard
        #target_feature_tokens
        #inline_tokens
        #(#attrs)*
        #vis #sig #body
    })
}
