//! A feature-enabled function, independent of attribute spelling.
use crate::common::*;
use crate::generated::{
    canonical_token_to_tier_suffix, token_to_arch, token_to_features, token_to_magetypes_namespace,
};
use crate::rite::DEFAULT_TIER_SENTINEL;
use crate::token_discovery::TokenParamInfo;
use proc_macro2::TokenStream;
use quote::quote;
use syn::{Attribute, Ident, parse_quote};

/// Where a `#[rite]` variant gets its features from.
pub(crate) struct FeatureContext {
    /// Canonical token name (`X64V3Token`), or `None` for the tokenless `default` tier.
    pub(crate) token: Option<String>,
    /// The tier suffix (`v3`, `scalar`, `default`), when the tier is one the
    /// registry knows; it selects the nested `incant!` rewrite.
    pub(crate) suffix: Option<&'static str>,
    pub(crate) features: std::borrow::Cow<'static, [&'static str]>,
    pub(crate) target_arch: Option<&'static str>,
    pub(crate) magetypes_namespace: Option<&'static str>,
    /// The token parameter when the features came from the signature.
    pub(crate) token_ident: Option<Ident>,
    /// Tier traits to re-state through `::archmage::` (trait or generic bound).
    pub(crate) tier_traits: Vec<String>,
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
        Some(FeatureContext {
            token: Some(tier_token.to_string()),
            suffix: canonical_token_to_tier_suffix(tier_token),
            features: std::borrow::Cow::Borrowed(token_to_features(tier_token)?),
            target_arch: token_to_arch(tier_token),
            magetypes_namespace: token_to_magetypes_namespace(tier_token),
            token_ident: None,
            tier_traits: Vec::new(),
        })
    }

    pub(crate) fn from_token_param(info: TokenParamInfo) -> Self {
        FeatureContext {
            suffix: info
                .token_type_name
                .as_deref()
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
    inline: Option<Attribute>,
    preserve_inline: bool,
    tier: FeatureContext,
) -> Result<TokenStream, TokenStream> {
    let token_desc = tier
        .token
        .clone()
        .unwrap_or_else(|| "an AVX-512 token".to_string());
    if let Some(err) = avx512_import_error(
        &variant_fn.sig,
        options.import_intrinsics,
        &tier.features,
        &token_desc,
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
                tier_suffix: tier_suffix.to_string(),
                target_arch: resolved.target_arch,
                token_ident: ident.clone(),
                has_token: true,
                derive_token: false,
            },
            None => crate::rewrite::CallerContext {
                tier_suffix: tier_suffix.to_string(),
                target_arch: resolved.target_arch,
                token_ident: quote::format_ident!("_"),
                has_token: false,
                derive_token: true,
            },
        };
        variant_fn.body = crate::rewrite::rewrite_incant_in_body(variant_fn.body, &ctx);
        variant_fn.body = crate::attune::call::rewrite(variant_fn.body, resolved);
    }

    // Build the attribute list. Scalar and default tiers have no features —
    // emit only `#[inline]` without `#[target_feature]` (enable="" is an error).
    let mut new_attrs: Vec<Attribute> = Vec::new();
    if !tier.features.is_empty() {
        let features_csv =
            crate::token_discovery::features_csv(tier.token.as_deref(), &tier.features);
        new_attrs.push(parse_quote!(#[target_feature(enable = #features_csv)]));
    }
    // Always use #[inline] - #[inline(always)] + #[target_feature] requires nightly
    if let Some(inline) = inline {
        new_attrs.push(inline);
    } else if !preserve_inline || !variant_fn.attrs.iter().any(|a| a.path().is_ident("inline")) {
        new_attrs.push(parse_quote!(#[inline]));
    }
    for attr in variant_fn
        .attrs
        .iter()
        .filter(|attr| preserve_inline || !attr.path().is_ident("inline"))
    {
        new_attrs.push(attr.clone());
    }
    variant_fn.attrs = new_attrs;

    // `#[rite]` has no wrapper — it puts `#[target_feature]` on the function
    // itself — so a trait/generic bound is authenticated at the top of the body.
    // Concrete tokens keep their tier-tag const in `#[arcane]`'s wrapper; a
    // tokenless tier form (`#[rite(v3)]`) has no bound to authenticate.
    let body_imports = generate_imports(
        tier.target_arch,
        tier.magetypes_namespace,
        options.import_intrinsics,
        options.import_magetypes,
    );
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
    let cfg_guard = gen_cfg_guard(tier.target_arch, options.cfg_feature.as_deref());
    let vis = &variant_fn.vis;
    let sig = &variant_fn.sig;
    let attrs = &variant_fn.attrs;
    let body = &variant_fn.body;
    Ok(quote! {
        #cfg_guard
        #(#attrs)*
        #vis #sig {
            #body
        }
    })
}
