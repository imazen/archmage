//! `#[rite]` — adds `#[target_feature]` + `#[inline]` directly.
//!
//! Single-tier and multi-tier helpers.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Attribute, Ident, Token,
    parse::{Parse, ParseStream},
    parse_quote,
};

use crate::common::*;
use crate::generated::{
    canonical_token_to_tier_suffix, tier_to_canonical_token, token_to_arch, token_to_features,
    token_to_magetypes_namespace,
};
use crate::token_discovery::*;

#[derive(Default)]
pub(crate) struct RiteArgs {
    /// Options spelled the same way in `#[arcane]`: imports and `cfg(feature)`.
    pub(crate) shared: SharedOptions,
    /// Tiers specified directly (e.g., `#[rite(v3)]` or `#[rite(v3, v4, neon)]`).
    /// Stored as canonical token names (e.g., "X64V3Token"), or the sentinel
    /// "" for the `default` tier (tokenless fallback — no `#[target_feature]`,
    /// no cfg-gating, no ScalarToken parameter).
    /// Single tier: generates one function (no suffix, no token parameter needed).
    /// Multiple tiers: generates suffixed variants (e.g., `fn_v3`, `fn_v4`, `fn_neon`).
    tier_tokens: Vec<String>,
}

/// The sentinel used in `tier_tokens` for the `default` tier.
///
/// `default` (and `_default`) is a tokenless fallback — it generates a
/// `fn_default(...)` variant with no `#[target_feature]`, no cfg-gating,
/// and no ScalarToken parameter. Distinct from `scalar` which is tokenful
/// (takes `ScalarToken` and participates in `incant!` token passing).
pub(crate) const DEFAULT_TIER_SENTINEL: &str = "";

impl Parse for RiteArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut args = RiteArgs::default();

        // Tier list, assembled with `+`/`-` modifier support (issue #48).
        // `#[rite]` has no dispatch-default set (it only emits the variants you
        // ask for), so the modifiers operate on the *explicit* list: plain and
        // `+tier` add a tier, `-tier` removes a previously-listed tier. This
        // keeps the grammar uniform with `#[magetypes]` / `incant!` so muscle
        // memory like `#[rite(v3, -scalar)]` parses (and means "just v3" here).
        let mut additions: Vec<String> = Vec::new();
        let mut removals: Vec<String> = Vec::new();

        // Map a tier ident to its `tier_tokens` entry (canonical token name, or
        // the empty sentinel for `default`/`_default`). `None` ⇒ not a tier.
        let resolve_tier = |name: &str| -> Option<String> {
            if name == "default" || name == "_default" {
                Some(String::from(DEFAULT_TIER_SENTINEL))
            } else {
                tier_to_canonical_token(name).map(String::from)
            }
        };

        while !input.is_empty() {
            // A `+`/`-` prefix is only valid before a tier name (never a keyword).
            if input.peek(Token![+]) || input.peek(Token![-]) {
                let is_removal = input.peek(Token![-]);
                if is_removal {
                    let _: Token![-] = input.parse()?;
                } else {
                    let _: Token![+] = input.parse()?;
                }
                let ident: Ident = input.parse()?;
                let name = ident.to_string();
                match resolve_tier(&name) {
                    Some(entry) => {
                        if is_removal {
                            removals.push(entry);
                        } else {
                            additions.push(entry);
                        }
                    }
                    None => {
                        return Err(syn::Error::new(
                            ident.span(),
                            format!(
                                "`{name}` after `{}` is not a tier name. \
                                 `+`/`-` modifiers apply only to tiers \
                                 (v1, v2, v3, v4, neon, arm_v2, wasm128, scalar, default, ...).",
                                if is_removal { "-" } else { "+" }
                            ),
                        ));
                    }
                }
            } else {
                let ident: Ident = input.parse()?;
                if parse_shared_option(&ident, input, &mut args.shared)? {
                    if input.peek(Token![,]) {
                        let _: Token![,] = input.parse()?;
                    }
                    continue;
                }
                match ident.to_string().as_str() {
                    // `#[arcane]` placement flags. A trait method can never
                    // carry `#[target_feature]` directly, and `#[rite]` has no
                    // wrapper to call `Self::` from, so neither flag has a
                    // meaning here; say what to use instead.
                    "in_trait" | "nested" => {
                        return Err(syn::Error::new(
                            ident.span(),
                            "`#[rite]` has no trait-method mode: it applies `#[target_feature]` \
                             directly, which rustc rejects on a safe trait method on x86-64 \
                             and AArch64 (wasm32's simd128 is the exception). Use \
                             `#[arcane(in_trait, _self = Type)]` on the trait method, or have \
                             it call a `#[rite]` free function.",
                        ));
                    }
                    "in_impl" => {
                        return Err(syn::Error::new(
                            ident.span(),
                            "`in_impl` is an `#[arcane]` option. `#[rite]` has no wrapper, \
                             so an associated function in an inherent impl needs no flag.",
                        ));
                    }
                    "default" | "_default" => {
                        // Tokenless fallback tier. No ScalarToken parameter, no
                        // target_feature, no cfg-gating — just an `#[inline]`
                        // variant named `_default` that slots into `incant!`'s
                        // suffix convention for fully portable fallbacks.
                        additions.push(String::from(DEFAULT_TIER_SENTINEL));
                    }
                    other => {
                        if let Some(canonical) = tier_to_canonical_token(other) {
                            additions.push(String::from(canonical));
                        } else {
                            return Err(syn::Error::new(
                                ident.span(),
                                format!(
                                    "unknown rite argument: `{}`. Supported: tier names \
                                     (v1, v2, v3, v4, neon, arm_v2, wasm128, scalar, default, ...), \
                                     optional `+`/`-` tier modifiers, \
                                     `import_intrinsics`, `import_magetypes`, `cfg(feature)`.",
                                    other
                                ),
                            ));
                        }
                    }
                }
            }
            if input.peek(Token![,]) {
                let _: Token![,] = input.parse()?;
            }
        }

        // Resolve additions minus removals, preserving first-seen order and
        // de-duplicating (so `#[rite(v3, v3)]` is one variant, `#[rite(v3, -v3)]`
        // is none).
        for entry in additions {
            if removals.contains(&entry) {
                continue;
            }
            if !args.tier_tokens.contains(&entry) {
                args.tier_tokens.push(entry);
            }
        }

        Ok(args)
    }
}

/// Implementation for the `#[rite]` macro.
pub(crate) fn rite_impl(input_fn: LightFn, args: RiteArgs) -> TokenStream {
    // Multi-tier mode: generate suffixed variants for each tier
    if args.tier_tokens.len() > 1 {
        return rite_multi_tier_impl(input_fn, &args);
    }
    // Single-tier or token-param mode
    rite_single_impl(input_fn, args)
}

/// Where a `#[rite]` variant gets its features from.
struct RiteTier {
    /// Canonical token name (`X64V3Token`), or `None` for the tokenless `default` tier.
    token: Option<String>,
    /// The tier suffix (`v3`, `scalar`, `default`), when the tier is one the
    /// registry knows; it selects the nested `incant!` rewrite.
    suffix: Option<&'static str>,
    features: std::borrow::Cow<'static, [&'static str]>,
    target_arch: Option<&'static str>,
    magetypes_namespace: Option<&'static str>,
    /// The token parameter when the features came from the signature.
    token_ident: Option<Ident>,
    /// Tier traits to re-state through `::archmage::` (trait or generic bound).
    tier_traits: Vec<String>,
}

impl RiteTier {
    /// A tier named in the attribute: `#[rite(v3)]`, or `default`.
    fn from_tier_token(tier_token: &str) -> Option<Self> {
        if tier_token == DEFAULT_TIER_SENTINEL {
            return Some(RiteTier {
                token: None,
                suffix: Some("default"),
                features: std::borrow::Cow::Borrowed(&[]),
                target_arch: None,
                magetypes_namespace: None,
                token_ident: None,
                tier_traits: Vec::new(),
            });
        }
        Some(RiteTier {
            token: Some(tier_token.to_string()),
            suffix: canonical_token_to_tier_suffix(tier_token),
            features: std::borrow::Cow::Borrowed(token_to_features(tier_token)?),
            target_arch: token_to_arch(tier_token),
            magetypes_namespace: token_to_magetypes_namespace(tier_token),
            token_ident: None,
            tier_traits: Vec::new(),
        })
    }

    fn from_token_param(info: TokenParamInfo) -> Self {
        RiteTier {
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

/// Generate a single `#[rite]` function (single tier or token-param mode).
pub(crate) fn rite_single_impl(mut input_fn: LightFn, args: RiteArgs) -> TokenStream {
    // Resolve features: either from tier name or from token parameter
    let tier = if let Some(tier_token) = args.tier_tokens.first() {
        // Tier specified directly (e.g., #[rite(v3)]) — no token param needed.
        RiteTier::from_tier_token(tier_token)
            .expect("tier_to_canonical_token returned invalid token name")
    } else {
        // A wildcard token gets a name so the tier assertion and the nested
        // dispatch rewrite can refer to it; every other pattern stays as
        // written (there is no wrapper that would need to forward it).
        rename_wildcard_token(&mut input_fn.sig);
        match find_token_param(&input_fn.sig) {
            Some(info) => RiteTier::from_token_param(info),
            None => {
                return missing_token_error(
                    &input_fn.sig,
                    "rite",
                    " or a tier name. Supported forms:\n\
                     - Tier name: `#[rite(v3)]`, `#[rite(neon)]`\n\
                     - Multi-tier: `#[rite(v3, v4, neon)]` (generates suffixed variants)\n\
                     - Concrete: `token: X64V3Token`\n\
                     - impl Trait: `token: impl HasX64V2`\n\
                     - Generic: `fn foo<T: HasX64V2>(token: T, ...)`",
                );
            }
        }
    };
    // In token-param mode one token decides the features (see #122).
    if tier.token_ident.is_some() {
        let token_params = token_param_idents(&input_fn.sig);
        if token_params.len() > 1 {
            return multiple_tokens_error(&input_fn.sig, "rite", &token_params);
        }
    }
    match emit_rite_variant(input_fn, &args, tier) {
        Ok(tokens) => tokens,
        Err(err) => err,
    }
}

/// Generate multiple suffixed `#[rite]` variants for multi-tier mode.
///
/// `#[rite(v3, v4, neon)]` on `fn process(...)` generates:
/// - `fn process_v3(...)` with `#[target_feature(enable = "avx2,fma,...")]`
/// - `fn process_v4(...)` with `#[target_feature(enable = "avx512f,...")]`
/// - `fn process_neon(...)` with `#[target_feature(enable = "neon")]`
///
/// Each variant is cfg-gated to its architecture and gets `#[inline]`.
pub(crate) fn rite_multi_tier_impl(input_fn: LightFn, args: &RiteArgs) -> TokenStream {
    let fn_name = &input_fn.sig.ident;
    let mut variants = proc_macro2::TokenStream::new();

    for tier_token in &args.tier_tokens {
        let Some(tier) = RiteTier::from_tier_token(tier_token) else {
            return syn::Error::new_spanned(
                &input_fn.sig,
                format!("unknown token `{tier_token}` in multi-tier #[rite]"),
            )
            .to_compile_error();
        };
        let suffix = tier.suffix.expect("every attribute tier has a suffix");

        // Clone and rename the function: process → process_v3
        let mut variant_fn = input_fn.clone();
        variant_fn.sig.ident = format_ident!("{}_{}", fn_name, suffix);

        // A multi-tier variant may still carry a token parameter; the rewrite
        // threads it through nested incant! calls when it does.
        let tier = match crate::token_discovery::find_token_param(&variant_fn.sig) {
            Some(info) => RiteTier {
                token_ident: Some(info.ident),
                ..tier
            },
            None => tier,
        };
        match emit_rite_variant(variant_fn, args, tier) {
            Ok(tokens) => variants.extend(tokens),
            Err(err) => return err,
        }
    }

    variants
}

/// Emit one `#[target_feature]` + `#[inline]` function for `tier`, with
/// imports, nested `incant!` rewriting, the bound assertion and the cfg guard.
/// Both the single- and multi-tier forms go through here.
fn emit_rite_variant(
    mut variant_fn: LightFn,
    args: &RiteArgs,
    tier: RiteTier,
) -> Result<TokenStream, TokenStream> {
    let token_desc = tier
        .token
        .clone()
        .unwrap_or_else(|| "an AVX-512 token".to_string());
    if let Some(err) = avx512_import_error(
        &variant_fn.sig,
        args.shared.import_intrinsics,
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
    new_attrs.push(parse_quote!(#[inline]));
    for attr in filter_inline_attrs(&variant_fn.attrs) {
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
        args.shared.import_intrinsics,
        args.shared.import_magetypes,
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
    let cfg_guard = gen_cfg_guard(tier.target_arch, args.shared.cfg_feature.as_deref());
    drop_attrs_equal_to(&mut variant_fn.attrs, &cfg_guard);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &str) -> RiteArgs {
        syn::parse_str::<RiteArgs>(args).expect("RiteArgs should parse")
    }

    #[test]
    fn single_tier() {
        assert_eq!(parse("v3").tier_tokens, vec!["X64V3Token"]);
    }

    #[test]
    fn multi_tier() {
        assert_eq!(
            parse("v3, v4").tier_tokens,
            vec!["X64V3Token", "X64V4Token"]
        );
    }

    #[test]
    fn minus_scalar_is_noop_when_absent() {
        // #48: `#[rite(v3, -scalar)]` parses and emits just the v3 variant.
        assert_eq!(parse("v3, -scalar").tier_tokens, vec!["X64V3Token"]);
    }

    #[test]
    fn minus_removes_listed_tier() {
        // scalar added then removed → only v3 remains.
        assert_eq!(parse("v3, scalar, -scalar").tier_tokens, vec!["X64V3Token"]);
    }

    #[test]
    fn plus_prefix_adds() {
        assert_eq!(
            parse("+v3, +neon").tier_tokens,
            vec!["X64V3Token", "NeonToken"]
        );
    }

    #[test]
    fn plain_and_plus_mix() {
        assert_eq!(
            parse("v3, +v4, -scalar").tier_tokens,
            vec!["X64V3Token", "X64V4Token"]
        );
    }

    #[test]
    fn minus_cancels_plain() {
        // `-v3` removes the `v3` addition → no tiers.
        assert!(parse("v3, -v3").tier_tokens.is_empty());
    }

    #[test]
    fn dedup() {
        assert_eq!(parse("v3, v3").tier_tokens, vec!["X64V3Token"]);
    }

    #[test]
    fn default_sentinel() {
        assert_eq!(parse("default").tier_tokens, vec![DEFAULT_TIER_SENTINEL]);
        assert_eq!(parse("v3, -default").tier_tokens, vec!["X64V3Token"]);
    }

    #[test]
    fn tiers_coexist_with_keywords() {
        let a = parse("v3, import_intrinsics");
        assert_eq!(a.tier_tokens, vec!["X64V3Token"]);
        assert!(a.shared.import_intrinsics);
    }

    #[test]
    fn minus_before_keyword_errors() {
        assert!(syn::parse_str::<RiteArgs>("+import_intrinsics").is_err());
        assert!(syn::parse_str::<RiteArgs>("-cfg").is_err());
    }

    #[test]
    fn underscore_prefix_accepted() {
        assert_eq!(parse("_v3, -_scalar").tier_tokens, vec!["X64V3Token"]);
    }
}
