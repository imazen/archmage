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

/// Generate a single `#[rite]` function (single tier or token-param mode).
pub(crate) fn rite_single_impl(mut input_fn: LightFn, args: RiteArgs) -> TokenStream {
    // Resolve features: either from tier name or from token parameter
    let tier = if let Some(tier_token) = args.tier_tokens.first() {
        // Tier specified directly (e.g., #[rite(v3)]) — no token param needed.
        FeatureContext::from_tier_token(tier_token)
            .expect("tier_to_canonical_token returned invalid token name")
    } else {
        // Only wildcard proof parameters need names for assertions and calls.
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
    // In token-param mode one token decides the features (see #122).
    if tier.token_ident.is_some() {
        let token_params = token_param_idents(&input_fn.sig);
        if token_params.len() > 1 {
            return multiple_tokens_error(&input_fn.sig, "rite", &token_params);
        }
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
    let mut variants = proc_macro2::TokenStream::new();

    for tier_token in &args.tier_tokens {
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
        let tier = match crate::token_discovery::find_token_param(&variant_fn.sig) {
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
}
