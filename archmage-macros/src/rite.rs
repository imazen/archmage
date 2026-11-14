//! `#[rite]` — adds `#[target_feature]` + `#[inline]` directly.
//!
//! Single-tier and multi-tier helpers.

use proc_macro2::TokenStream;
use quote::format_ident;
use syn::{
    Ident, Token,
    parse::{Parse, ParseStream},
};

use crate::common::*;
use crate::engine::feature::FeatureContext;
use crate::generated::tier_to_canonical_token;
use crate::token_discovery::*;

#[derive(Default)]
pub(crate) struct RiteArgs {
    /// Options spelled the same way in `#[arcane]`: imports and `cfg(feature)`.
    pub(crate) shared: SharedOptions,
    /// Tiers specified directly (e.g., `#[rite(v3)]` or `#[rite(v3, v4, neon)]`).
    /// Stored as canonical `&'static str` token names, or `""` for `default`.
    tier_tokens: Vec<&'static str>,
}

/// The sentinel used in `tier_tokens` for the `default` tier.
pub(crate) const DEFAULT_TIER_SENTINEL: &str = "";

impl Parse for RiteArgs {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut args = RiteArgs::default();

        let mut additions: SmallVec<&'static str, 6> = SmallVec::new();
        let mut removals: SmallVec<&'static str, 6> = SmallVec::new();

        let resolve_tier = |name: &str| -> Option<&'static str> {
            if name == "default" || name == "_default" {
                Some(DEFAULT_TIER_SENTINEL)
            } else {
                tier_to_canonical_token(name)
            }
        };

        while !input.is_empty() {
            if input.peek(Token![+]) || input.peek(Token![-]) {
                let is_removal = input.peek(Token![-]);
                if is_removal {
                    let _: Token![-] = input.parse()?;
                } else {
                    let _: Token![+] = input.parse()?;
                }
                let ident: Ident = input.parse()?;
                with_ident_str(&ident, |name| match resolve_tier(name) {
                    Some(entry) => {
                        if is_removal {
                            removals.push(entry);
                        } else {
                            additions.push(entry);
                        }
                        Ok(())
                    }
                    None => Err(syn::Error::new(
                        ident.span(),
                        format!(
                            "`{name}` after `{}` is not a tier name. \
                             `+`/`-` modifiers apply only to tiers \
                             (v1, v2, v3, v4, neon, arm_v2, wasm128, scalar, default, ...).",
                            if is_removal { "-" } else { "+" }
                        ),
                    )),
                })?;
            } else {
                let ident: Ident = input.parse()?;
                if parse_shared_option(&ident, input, &mut args.shared)? {
                    if input.peek(Token![,]) {
                        let _: Token![,] = input.parse()?;
                    }
                    continue;
                }
                with_ident_str(&ident, |s| match s {
                    "in_trait" | "nested" => Err(syn::Error::new(
                        ident.span(),
                        "`#[rite]` has no trait-method mode: it applies `#[target_feature]` \
                         directly, which rustc rejects on a safe trait method on x86-64 \
                         and AArch64 (wasm32's simd128 is the exception). Use \
                         `#[arcane(in_trait, _self = Type)]` on the trait method, or have \
                         it call a `#[rite]` free function.",
                    )),
                    "in_impl" => Err(syn::Error::new(
                        ident.span(),
                        "`in_impl` is an `#[arcane]` option. `#[rite]` has no wrapper, \
                         so an associated function in an inherent impl needs no flag.",
                    )),
                    "default" | "_default" => {
                        additions.push(DEFAULT_TIER_SENTINEL);
                        Ok(())
                    }
                    other => {
                        if let Some(canonical) = tier_to_canonical_token(other) {
                            additions.push(canonical);
                            Ok(())
                        } else {
                            Err(syn::Error::new(
                                ident.span(),
                                format!(
                                    "unknown rite argument: `{}`. Supported: tier names \
                                     (v1, v2, v3, v4, neon, arm_v2, wasm128, scalar, default, ...), \
                                     optional `+`/`-` tier modifiers, \
                                     `import_intrinsics`, `import_magetypes`, `cfg(feature)`.",
                                    other
                                ),
                            ))
                        }
                    }
                })?;
            }
            if input.peek(Token![,]) {
                let _: Token![,] = input.parse()?;
            }
        }

        for &entry in &additions {
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
    if args.tier_tokens.len() > 1 {
        return rite_multi_tier_impl(input_fn, &args);
    }
    rite_single_impl(input_fn, args)
}

/// Generate a single `#[rite]` function (single tier or token-param mode).
pub(crate) fn rite_single_impl(mut input_fn: LightFn, args: RiteArgs) -> TokenStream {
    let tier = if let Some(&tier_token) = args.tier_tokens.first() {
        FeatureContext::from_tier_token(tier_token)
            .expect("tier_to_canonical_token returned invalid token name")
    } else {
        rename_wildcard_token(&mut input_fn.sig);
        match find_token_param(&input_fn.sig) {
            Some(info) => FeatureContext::from_token_param(info),
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
    if tier.token_ident.is_some()
        && let Some(err) = check_single_token_param(&input_fn.sig, "rite")
    {
        return err;
    }
    match crate::engine::feature::emit(input_fn, &args.shared, None, false, tier) {
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
    let mut variants = TokenStream::new();

    for &tier_token in &args.tier_tokens {
        let Some(tier) = FeatureContext::from_tier_token(tier_token) else {
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
        let tier = match find_token_param(&variant_fn.sig) {
            Some(info) => FeatureContext {
                token_ident: Some(info.ident),
                ..tier
            },
            None => tier,
        };
        match crate::engine::feature::emit(variant_fn, &args.shared, None, false, tier) {
            Ok(tokens) => variants.extend(tokens),
            Err(err) => return err,
        }
    }

    variants
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

    #[test]
    fn rite_rejects_safe_tokenless_wasm128_relaxed_fma() {
        let single_args: RiteArgs = syn::parse_str("wasm128_relaxed_fma").unwrap();
        let single_fn: LightFn = syn::parse_quote!(
            fn kernel(x: f32) -> f32 {
                x
            }
        );
        let single_out = rite_impl(single_fn, single_args).to_string();
        assert!(
            single_out.contains("requires a runtime fusion probe"),
            "{single_out}"
        );

        let multi_args: RiteArgs = syn::parse_str("wasm128, wasm128_relaxed_fma").unwrap();
        let multi_fn: LightFn = syn::parse_quote!(
            fn kernel(x: f32) -> f32 {
                x
            }
        );
        let multi_out = rite_multi_tier_impl(multi_fn, &multi_args).to_string();
        assert!(
            multi_out.contains("requires a runtime fusion probe"),
            "{multi_out}"
        );

        let token_args: RiteArgs = syn::parse_str("wasm128_relaxed_fma").unwrap();
        let with_token: LightFn = syn::parse_quote!(
            fn kernel(_token: Wasm128RelaxedFmaToken, x: f32) -> f32 {
                x
            }
        );
        let ok_out = rite_impl(with_token, token_args).to_string();
        assert!(
            !ok_out.contains("compile_error")
                && !ok_out.contains("requires a runtime fusion probe"),
            "{ok_out}"
        );

        let unsafe_args: RiteArgs = syn::parse_str("wasm128_relaxed_fma").unwrap();
        let unsafe_fn: LightFn = syn::parse_quote!(
            unsafe fn kernel(x: f32) -> f32 {
                x
            }
        );
        let unsafe_ok = rite_impl(unsafe_fn, unsafe_args).to_string();
        assert!(
            !unsafe_ok.contains("compile_error")
                && !unsafe_ok.contains("requires a runtime fusion probe"),
            "{unsafe_ok}"
        );
    }
}
