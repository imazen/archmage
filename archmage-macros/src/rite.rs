//! `#[rite]` — adds `#[target_feature]` + `#[inline]` directly.
//!
//! Single-tier, multi-tier, and stub modes.

use proc_macro::TokenStream;
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
    /// Generate an `unreachable!()` stub on the wrong architecture.
    /// Default is false (cfg-out: no function emitted on wrong arch).
    pub(crate) stub: bool,
    /// Inject `use archmage::intrinsics::{arch}::*;` (includes safe memory ops).
    pub(crate) import_intrinsics: bool,
    /// Inject `use magetypes::simd::{ns}::*;`, `use magetypes::simd::generic::*;`,
    /// and `use magetypes::simd::backends::*;`.
    import_magetypes: bool,
    /// Tiers specified directly (e.g., `#[rite(v3)]` or `#[rite(v3, v4, neon)]`).
    /// Stored as canonical token names (e.g., "X64V3Token"), or the sentinel
    /// "" for the `default` tier (tokenless fallback — no `#[target_feature]`,
    /// no cfg-gating, no ScalarToken parameter).
    /// Single tier: generates one function (no suffix, no token parameter needed).
    /// Multiple tiers: generates suffixed variants (e.g., `fn_v3`, `fn_v4`, `fn_neon`).
    tier_tokens: Vec<String>,
    /// Additional cargo feature gate (same as arcane's cfg_feature).
    pub(crate) cfg_feature: Option<String>,
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
                match ident.to_string().as_str() {
                    "stub" => {
                        return Err(syn::Error::new(
                            ident.span(),
                            "`stub` has been removed. Use `incant!` for cross-arch dispatch instead.",
                        ));
                    }
                    "import_intrinsics" => args.import_intrinsics = true,
                    "import_magetypes" => args.import_magetypes = true,
                    "cfg" => {
                        let content;
                        syn::parenthesized!(content in input);
                        let feat: Ident = content.parse()?;
                        args.cfg_feature = Some(feat.to_string());
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
                                     `stub`, `import_intrinsics`, `import_magetypes`, `cfg(feature)`.",
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

/// Generate a single `#[rite]` function (single tier or token-param mode).
pub(crate) fn rite_single_impl(mut input_fn: LightFn, args: RiteArgs) -> TokenStream {
    // Resolve features: either from tier name or from token parameter
    let TokenParamInfo {
        ident: token_ident,
        features,
        target_arch,
        token_type_name: _token_type_name,
        magetypes_namespace,
        token_type: _,
    } = if let Some(tier_token) = args.tier_tokens.first() {
        // Tier specified directly (e.g., #[rite(v3)]) — no token param needed.
        // `default` tier (DEFAULT_TIER_SENTINEL) is tokenless: no features,
        // no arch, no target_feature attribute.
        let is_default = tier_token == DEFAULT_TIER_SENTINEL;
        let features: Vec<&'static str> = if is_default {
            Vec::new()
        } else {
            token_to_features(tier_token)
                .expect("tier_to_canonical_token returned invalid token name")
                .to_vec()
        };
        let target_arch = if is_default {
            None
        } else {
            token_to_arch(tier_token)
        };
        let magetypes_namespace = if is_default {
            None
        } else {
            token_to_magetypes_namespace(tier_token)
        };
        TokenParamInfo {
            ident: Ident::new("_", proc_macro2::Span::call_site()),
            features,
            target_arch,
            token_type_name: if is_default {
                None
            } else {
                Some(tier_token.clone())
            },
            magetypes_namespace,
            token_type: None,
        }
    } else {
        match find_token_param(&input_fn.sig) {
            Some(result) => result,
            None => {
                // Check for specific misuse: featureless traits like SimdToken
                if let Some(trait_name) = diagnose_featureless_token(&input_fn.sig) {
                    let msg = format!(
                        "`{trait_name}` cannot be used as a token bound in #[rite] \
                         because it doesn't specify any CPU features.\n\
                         \n\
                         #[rite] needs concrete features to generate #[target_feature]. \
                         Use a concrete token, a feature trait, or a tier name:\n\
                         \n\
                         Concrete tokens: X64V3Token, Desktop64, NeonToken, Arm64V2Token, ...\n\
                         Feature traits:  impl HasX64V2, impl HasNeon, impl HasArm64V3, ...\n\
                         Tier names:      #[rite(v3)], #[rite(neon)], #[rite(v4)], ..."
                    );
                    return syn::Error::new_spanned(&input_fn.sig, msg)
                        .to_compile_error()
                        .into();
                }
                let msg = "rite requires a token parameter or a tier name. Supported forms:\n\
                     - Tier name: `#[rite(v3)]`, `#[rite(neon)]`\n\
                     - Multi-tier: `#[rite(v3, v4, neon)]` (generates suffixed variants)\n\
                     - Concrete: `token: X64V3Token`\n\
                     - impl Trait: `token: impl HasX64V2`\n\
                     - Generic: `fn foo<T: HasX64V2>(token: T, ...)`";
                return syn::Error::new_spanned(&input_fn.sig, msg)
                    .to_compile_error()
                    .into();
            }
        }
    };

    // Check: import_intrinsics with AVX-512 features requires the avx512 cargo feature.
    // Check resolved features (not token name) for uniform handling of concrete/trait/generic.
    #[cfg(not(feature = "avx512"))]
    if args.import_intrinsics && features.iter().any(|f| f.starts_with("avx512")) {
        let token_desc = _token_type_name.as_deref().unwrap_or("an AVX-512 token");
        let msg = format!(
            "Using {token_desc} with `import_intrinsics` requires the `avx512` feature.\n\
             \n\
             Add to your Cargo.toml:\n\
             \x20 archmage = {{ version = \"...\", features = [\"avx512\"] }}\n\
             \n\
             Without it, 512-bit safe memory ops (_mm512_loadu_ps etc.) are not available.\n\
             If you only need value intrinsics (no memory ops), remove `import_intrinsics`."
        );
        return syn::Error::new_spanned(&input_fn.sig, msg)
            .to_compile_error()
            .into();
    }

    // Rewrite incant!() calls in the body to direct tier calls.
    // Only when a real token parameter exists — tokenless rite (e.g., #[rite(v3)])
    // has no token to pass to callees.
    if token_ident != "_"
        && let Some(ref type_name) = _token_type_name
        && let Some(tier_suffix) = crate::generated::canonical_token_to_tier_suffix(type_name)
        && let Some(tier) = crate::tiers::find_tier(tier_suffix)
    {
        let ctx = crate::rewrite::CallerContext {
            tier_suffix: tier_suffix.to_string(),
            target_arch: tier.target_arch,
            token_ident: token_ident.clone(),
        };
        input_fn.body = crate::rewrite::rewrite_incant_in_body(input_fn.body.clone(), &ctx);
    }

    // Build the attribute list. Scalar tier has no features — emit only
    // `#[inline]` without `#[target_feature]` (enable="" is a compile error).
    let mut new_attrs: Vec<Attribute> = Vec::new();
    if !features.is_empty() {
        let features_csv = features.join(",");
        new_attrs.push(parse_quote!(#[target_feature(enable = #features_csv)]));
    }
    // Always use #[inline] - #[inline(always)] + #[target_feature] requires nightly
    new_attrs.push(parse_quote!(#[inline]));
    for attr in filter_inline_attrs(&input_fn.attrs) {
        new_attrs.push(attr.clone());
    }
    input_fn.attrs = new_attrs;

    // Prepend import statements to body if requested
    let body_imports = generate_imports(
        target_arch,
        magetypes_namespace,
        args.import_intrinsics,
        args.import_magetypes,
    );
    if !body_imports.is_empty() {
        let original_body = &input_fn.body;
        input_fn.body = quote! {
            #body_imports
            #original_body
        };
    }

    // If we know the target arch, generate cfg-gated impl (+ optional stub)
    let cfg_guard = gen_cfg_guard(target_arch, args.cfg_feature.as_deref());
    if target_arch.is_some() {
        let vis = &input_fn.vis;
        let sig = &input_fn.sig;
        let attrs = &input_fn.attrs;
        let body = &input_fn.body;

        let stub = if args.stub {
            let not_cfg = match (target_arch, args.cfg_feature.as_deref()) {
                (Some(arch), Some(feat)) => {
                    quote! { #[cfg(not(all(target_arch = #arch, feature = #feat)))] }
                }
                (Some(arch), None) => quote! { #[cfg(not(target_arch = #arch))] },
                _ => quote! {},
            };
            quote! {
                #not_cfg
                #vis #sig {
                    unreachable!("This function requires a specific architecture and feature set")
                }
            }
        } else {
            quote! {}
        };

        quote! {
            #cfg_guard
            #(#attrs)*
            #vis #sig {
                #body
            }

            #stub
        }
        .into()
    } else {
        // No specific arch (trait bounds) - just emit the annotated function
        quote!(#input_fn).into()
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
        // `default` tier uses DEFAULT_TIER_SENTINEL (empty string) — tokenless,
        // no features, no arch, suffix "_default", no imports.
        let is_default = tier_token == DEFAULT_TIER_SENTINEL;
        let features: &[&'static str] = if is_default {
            &[]
        } else {
            match token_to_features(tier_token) {
                Some(f) => f,
                None => {
                    return syn::Error::new_spanned(
                        &input_fn.sig,
                        format!("unknown token `{tier_token}` in multi-tier #[rite]"),
                    )
                    .to_compile_error()
                    .into();
                }
            }
        };
        let target_arch = if is_default {
            None
        } else {
            token_to_arch(tier_token)
        };
        let magetypes_namespace = if is_default {
            None
        } else {
            token_to_magetypes_namespace(tier_token)
        };

        // Check: import_intrinsics with AVX-512 features requires the avx512 cargo feature.
        #[cfg(not(feature = "avx512"))]
        if args.import_intrinsics && features.iter().any(|f| f.starts_with("avx512")) {
            let msg = format!(
                "Using {tier_token} with `import_intrinsics` requires the `avx512` feature.\n\
                 \n\
                 Add to your Cargo.toml:\n\
                 \x20 archmage = {{ version = \"...\", features = [\"avx512\"] }}\n\
                 \n\
                 Without it, 512-bit safe memory ops (_mm512_loadu_ps etc.) are not available.\n\
                 If you only need value intrinsics (no memory ops), remove `import_intrinsics`."
            );
            return syn::Error::new_spanned(&input_fn.sig, msg)
                .to_compile_error()
                .into();
        }

        let suffix = if is_default {
            "default"
        } else {
            canonical_token_to_tier_suffix(tier_token)
                .expect("canonical token must have a tier suffix")
        };

        // Build suffixed function name
        let suffixed_ident = format_ident!("{}_{}", fn_name, suffix);

        // Clone and rename the function
        let mut variant_fn = input_fn.clone();
        variant_fn.sig.ident = suffixed_ident;

        // Rewrite incant!() calls in the variant body.
        // Only for rite functions with a token param — tokenless rite can't pass tokens.
        if let Some(tier) = crate::tiers::find_tier(suffix)
            && let Some(token_info) = crate::token_discovery::find_token_param(&variant_fn.sig)
        {
            let ctx = crate::rewrite::CallerContext {
                tier_suffix: suffix.to_string(),
                target_arch: tier.target_arch,
                token_ident: token_info.ident,
            };
            variant_fn.body = crate::rewrite::rewrite_incant_in_body(variant_fn.body.clone(), &ctx);
        }

        // Build the attribute list. Scalar tier has no features — emit only
        // `#[inline]` without `#[target_feature]` (enable="" is a compile error).
        let mut new_attrs: Vec<Attribute> = Vec::new();
        if !features.is_empty() {
            let features_csv = features.join(",");
            new_attrs.push(parse_quote!(#[target_feature(enable = #features_csv)]));
        }
        new_attrs.push(parse_quote!(#[inline]));
        for attr in filter_inline_attrs(&variant_fn.attrs) {
            new_attrs.push(attr.clone());
        }
        variant_fn.attrs = new_attrs;

        // Prepend import statements if requested
        let body_imports = generate_imports(
            target_arch,
            magetypes_namespace,
            args.import_intrinsics,
            args.import_magetypes,
        );
        if !body_imports.is_empty() {
            let original_body = &variant_fn.body;
            variant_fn.body = quote! {
                #body_imports
                #original_body
            };
        }

        // Emit cfg-gated variant
        let variant_cfg = gen_cfg_guard(target_arch, args.cfg_feature.as_deref());
        if target_arch.is_some() {
            let vis = &variant_fn.vis;
            let sig = &variant_fn.sig;
            let attrs = &variant_fn.attrs;
            let body = &variant_fn.body;

            variants.extend(quote! {
                #variant_cfg
                #(#attrs)*
                #vis #sig {
                    #body
                }
            });

            if args.stub {
                let not_cfg = match (target_arch, args.cfg_feature.as_deref()) {
                    (Some(arch), Some(feat)) => {
                        quote! { #[cfg(not(all(target_arch = #arch, feature = #feat)))] }
                    }
                    (Some(arch), None) => quote! { #[cfg(not(target_arch = #arch))] },
                    _ => quote! {},
                };
                let arch_str = target_arch.unwrap_or("unknown");
                variants.extend(quote! {
                    #not_cfg
                    #vis #sig {
                        unreachable!(concat!(
                            "This function requires ",
                            #arch_str,
                            " architecture"
                        ))
                    }
                });
            }
        } else {
            // No specific arch — just emit the annotated function
            variants.extend(quote!(#variant_fn));
        }
    }

    variants.into()
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
        assert!(a.import_intrinsics);
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
