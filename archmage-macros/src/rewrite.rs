//! Body rewriter: rewrites `incant!()` calls inside tier-specialized function bodies.
//!
//! When `#[autoversion]`, `#[arcane]`, or `#[rite]` processes a function body,
//! this module scans for `incant!(fn(args), [tiers])` invocations and rewrites
//! them to direct calls — bypassing the runtime dispatcher.
//!
//! The rewriting produces:
//! - **Upgrade attempts** (callee tier > caller tier): `if let Some(t) = summon() { call }`
//! - **Direct calls** (callee tier ≤ caller tier): `fn_tier(token.tier(), args)`
//! - **Scalar fallback**: `fn_scalar(ScalarToken, args)`

use proc_macro2::{Delimiter, Ident, Spacing, TokenStream, TokenTree};
use quote::{format_ident, quote};

use crate::common::suffix_path;
use crate::incant::IncantInput;
use crate::tiers::{self, DEFAULT_TIER_NAMES, ResolvedTier};

/// Context about the caller's tier, used to decide how to rewrite each incant! call.
#[derive(Clone)]
pub(crate) struct CallerContext {
    /// The caller's tier suffix (e.g., "v3", "v4", "neon")
    pub tier_suffix: String,
    /// The caller's tier priority (higher = more features)
    pub tier_priority: u32,
    /// The caller's target arch (e.g., Some("x86_64"))
    pub target_arch: Option<&'static str>,
    /// The token ident available in the caller's scope (e.g., `token`, `__token`, `_token`)
    pub token_ident: Ident,
}

/// Rewrite `incant!()` calls in a function body for a specific tier context.
///
/// Walks the token stream looking for `incant ! ( ... )` patterns.
/// Skips inner `fn` items (they don't inherit `#[target_feature]`).
/// Descends into closures and other expressions.
///
/// Returns a new TokenStream with incant! calls replaced by direct tier calls.
pub(crate) fn rewrite_incant_in_body(body: TokenStream, ctx: &CallerContext) -> TokenStream {
    let tokens: Vec<TokenTree> = body.into_iter().collect();
    let mut result = Vec::new();
    let mut i = 0;

    while i < tokens.len() {
        // Check for `fn` keyword — skip the following function body
        if is_ident(&tokens[i], "fn") {
            // Emit the `fn` token and everything up to and including the body block
            result.push(tokens[i].clone());
            i += 1;
            // Skip to the body block (find the next `{...}` group)
            while i < tokens.len() {
                let is_body =
                    matches!(&tokens[i], TokenTree::Group(g) if g.delimiter() == Delimiter::Brace);
                result.push(tokens[i].clone());
                i += 1;
                if is_body {
                    break;
                }
            }
            continue;
        }

        // Check for `incant ! ( ... )` pattern
        if is_ident(&tokens[i], "incant") && i + 2 < tokens.len() && is_punct(&tokens[i + 1], '!') {
            if let Some(TokenTree::Group(group)) = tokens.get(i + 2) {
                if group.delimiter() == Delimiter::Parenthesis {
                    // Try to parse the incant arguments
                    let inner = group.stream();
                    if let Ok(input) = syn::parse2::<IncantInput>(inner) {
                        // Rewrite this incant! call
                        let rewritten = rewrite_single_incant(&input, ctx);
                        result.extend(rewritten.into_iter());
                        i += 3; // skip `incant`, `!`, `(...)`
                        continue;
                    }
                }
            }
        }

        // For groups (blocks, parens, brackets), recurse into them
        if let TokenTree::Group(group) = &tokens[i] {
            let inner = rewrite_incant_in_body(group.stream(), ctx);
            let mut new_group = proc_macro2::Group::new(group.delimiter(), inner);
            new_group.set_span(group.span());
            result.push(TokenTree::Group(new_group));
            i += 1;
            continue;
        }

        // Pass through everything else
        result.push(tokens[i].clone());
        i += 1;
    }

    result.into_iter().collect()
}

/// Rewrite a single parsed `incant!()` invocation.
fn rewrite_single_incant(input: &IncantInput, ctx: &CallerContext) -> TokenStream {
    // If passthrough mode (`with token`), don't rewrite — leave for incant! to handle
    if input.with_token.is_some() {
        // Reconstruct the original incant! call
        return reconstruct_incant(input);
    }

    let func_path = &input.func_path;
    let args = &input.args;

    // Resolve callee tiers
    let tier_names: Vec<String> = match &input.tiers {
        Some((names, _)) => names.clone(),
        None => DEFAULT_TIER_NAMES.iter().map(|s| s.to_string()).collect(),
    };
    let error_span = proc_macro2::Span::call_site();
    let tiers = match tiers::resolve_tiers(&tier_names, error_span, true) {
        Ok(t) => t,
        Err(_) => return reconstruct_incant(input), // parse error — let incant! handle it
    };

    // Partition callee tiers into:
    // 1. Same-arch tiers ABOVE the caller (need summon — upgrade attempts)
    // 2. Same-arch tier matching or below the caller (direct call — guaranteed)
    // 3. Scalar/default fallback
    let mut upgrade_tiers: Vec<&ResolvedTier> = Vec::new();
    let mut direct_tier: Option<&ResolvedTier> = None;

    for rt in &tiers {
        if rt.name == "scalar" || rt.name == "default" {
            continue; // handled as fallback
        }
        // Only consider tiers on the same architecture
        if rt.target_arch != ctx.target_arch {
            continue;
        }
        if rt.priority > ctx.tier_priority {
            // Callee tier is above caller — needs upgrade (summon)
            upgrade_tiers.push(rt);
        } else if direct_tier.is_none() {
            // First tier at or below caller — direct call (best match)
            direct_tier = Some(rt);
        }
    }

    // Sort upgrade tiers by priority descending (try highest first)
    upgrade_tiers.sort_by(|a, b| b.priority.cmp(&a.priority));

    let token_ident = &ctx.token_ident;

    // Build the upgrade attempt arms (need labeled block + break)
    let mut upgrade_arms = Vec::new();
    for rt in &upgrade_tiers {
        let fn_suffixed = suffix_path(func_path, rt.suffix);
        let token_path: syn::Path = syn::parse_str(rt.token_path).unwrap();

        let check = quote! {
            if let Some(__t) = #token_path::summon() {
                break '__incant_rewrite #fn_suffixed(__t, #(#args),*);
            }
        };

        if let Some(feat) = &rt.feature_gate {
            let allow_attr = if rt.allow_unexpected_cfg {
                quote! { #[allow(unexpected_cfgs)] }
            } else {
                quote! {}
            };
            upgrade_arms.push(quote! {
                #allow_attr
                #[cfg(feature = #feat)]
                { #check }
            });
        } else {
            upgrade_arms.push(check);
        }
    }

    // Build the direct call (guaranteed hit — no summon)
    let fallback_call = if let Some(rt) = direct_tier {
        let fn_suffixed = suffix_path(func_path, rt.suffix);
        if rt.suffix == ctx.tier_suffix {
            // Exact match — pass token directly
            quote! { #fn_suffixed(#token_ident, #(#args),*) }
        } else {
            // Downgrade — call the tier method on the token
            let downgrade_method = format_ident!("{}", rt.suffix);
            quote! { #fn_suffixed(#token_ident.#downgrade_method(), #(#args),*) }
        }
    } else {
        // No same-arch tier at or below caller — fall through to scalar
        let has_default = tiers.iter().any(|t| t.name == "default");
        if has_default {
            let fn_default = suffix_path(func_path, "default");
            quote! { #fn_default(#(#args),*) }
        } else {
            let fn_scalar = suffix_path(func_path, "scalar");
            quote! { #fn_scalar(archmage::ScalarToken, #(#args),*) }
        }
    };

    if upgrade_arms.is_empty() {
        // No upgrades to try — just the direct call, no labeled block needed
        fallback_call
    } else {
        // Upgrade attempts + guaranteed fallback
        quote! {
            '__incant_rewrite: {
                use archmage::SimdToken;
                #(#upgrade_arms)*
                #fallback_call
            }
        }
    }
}

/// Reconstruct the original `incant!(...)` call (for cases we don't rewrite).
fn reconstruct_incant(input: &IncantInput) -> TokenStream {
    let func_path = &input.func_path;
    let args = &input.args;

    let with_part = input.with_token.as_ref().map(|t| quote! { with #t });
    let tier_part = input.tiers.as_ref().map(|(names, _)| {
        let tier_strs: Vec<_> = names
            .iter()
            .map(|n| {
                let ident = format_ident!("{}", n);
                quote! { #ident }
            })
            .collect();
        quote! { , [#(#tier_strs),*] }
    });

    quote! {
        archmage::incant!(#func_path(#(#args),*) #with_part #tier_part)
    }
}

fn is_ident(tt: &TokenTree, name: &str) -> bool {
    matches!(tt, TokenTree::Ident(id) if *id == name)
}

fn is_punct(tt: &TokenTree, ch: char) -> bool {
    matches!(tt, TokenTree::Punct(p) if p.as_char() == ch && p.spacing() == Spacing::Alone)
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;

    fn make_ctx(tier: &str, priority: u32, arch: Option<&'static str>) -> CallerContext {
        CallerContext {
            tier_suffix: tier.to_string(),
            tier_priority: priority,
            target_arch: arch,
            token_ident: format_ident!("__token"),
        }
    }

    #[test]
    fn skips_inner_fn_items() {
        let body = quote! {
            fn inner() {
                incant!(process(data))
            }
            incant!(outer_call(data))
        };
        let ctx = make_ctx("v3", 30, Some("x86_64"));
        let result = rewrite_incant_in_body(body, &ctx).to_string();
        // inner's incant! should NOT be rewritten (inner fn doesn't inherit target_feature)
        assert!(
            result.contains("incant ! (process (data))"),
            "inner fn incant! should be preserved, got: {result}"
        );
        // outer incant! SHOULD be rewritten
        assert!(
            result.contains("outer_call_v3"),
            "outer incant! should be rewritten, got: {result}"
        );
    }

    #[test]
    fn exact_tier_match_no_summon() {
        let body = quote! {
            let x = incant!(process(data), [v3, scalar]);
        };
        let ctx = make_ctx("v3", 30, Some("x86_64"));
        let result = rewrite_incant_in_body(body, &ctx).to_string();
        // Should be a direct call — no summon, no labeled block
        assert!(
            result.contains("process_v3"),
            "should call process_v3, got: {result}"
        );
        assert!(
            result.contains("__token"),
            "should pass token, got: {result}"
        );
        assert!(
            !result.contains("summon"),
            "should not summon for exact match, got: {result}"
        );
    }

    #[test]
    fn upgrade_attempt_with_summon() {
        let body = quote! {
            let x = incant!(process(data), [v4, v3, scalar]);
        };
        let ctx = make_ctx("v3", 30, Some("x86_64"));
        let result = rewrite_incant_in_body(body, &ctx).to_string();
        // Should attempt v4 upgrade with summon, fall back to v3 direct
        assert!(
            result.contains("summon"),
            "should summon for v4 upgrade, got: {result}"
        );
        assert!(
            result.contains("process_v4"),
            "should try process_v4, got: {result}"
        );
        assert!(
            result.contains("process_v3"),
            "should fall back to process_v3, got: {result}"
        );
        assert!(
            result.contains("__token"),
            "should pass token for v3, got: {result}"
        );
    }

    #[test]
    fn upgrade_with_feature_gate() {
        let body = quote! {
            let x = incant!(process(data), [v4(cfg(avx512)), v3, scalar]);
        };
        let ctx = make_ctx("v3", 30, Some("x86_64"));
        let result = rewrite_incant_in_body(body, &ctx).to_string();
        // V4 upgrade should be feature-gated
        assert!(
            result.contains("avx512"),
            "v4 upgrade should be gated on avx512, got: {result}"
        );
        assert!(
            result.contains("summon"),
            "should summon for v4 upgrade, got: {result}"
        );
        assert!(
            result.contains("process_v3"),
            "should fall back to process_v3, got: {result}"
        );
    }

    #[test]
    fn scalar_fallback_when_no_matching_tier() {
        let body = quote! {
            let x = incant!(process(data), [neon, scalar]);
        };
        // x86 caller — neon is wrong arch, only scalar available
        let ctx = make_ctx("v3", 30, Some("x86_64"));
        let result = rewrite_incant_in_body(body, &ctx).to_string();
        assert!(
            result.contains("process_scalar"),
            "should fall through to scalar, got: {result}"
        );
        assert!(
            result.contains("ScalarToken"),
            "should use ScalarToken, got: {result}"
        );
    }

    #[test]
    fn passthrough_not_rewritten() {
        let body = quote! {
            let x = incant!(process(data) with token);
        };
        let ctx = make_ctx("v3", 30, Some("x86_64"));
        let result = rewrite_incant_in_body(body, &ctx).to_string();
        // Passthrough should be reconstructed as incant!, not rewritten
        assert!(
            result.contains("incant"),
            "passthrough should be preserved, got: {result}"
        );
    }

    #[test]
    fn downgrade_uses_method() {
        let body = quote! {
            let x = incant!(process(data), [v3, scalar]);
        };
        // Caller is v4, callee only has v3 — downgrade
        let ctx = make_ctx("v4", 40, Some("x86_64"));
        let result = rewrite_incant_in_body(body, &ctx).to_string();
        assert!(
            result.contains("process_v3"),
            "should call process_v3, got: {result}"
        );
        assert!(
            result.contains("__token . v3 ()"),
            "should downgrade token, got: {result}"
        );
        assert!(
            !result.contains("summon"),
            "should not summon for downgrade, got: {result}"
        );
    }
}
