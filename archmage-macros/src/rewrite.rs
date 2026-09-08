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
use crate::tiers::{self, ResolvedTier};

/// Context about the caller's tier, used to decide how to rewrite each incant! call.
#[derive(Clone)]
pub(crate) struct CallerContext {
    /// The caller's tier suffix (e.g., "v3", "v4", "neon")
    pub tier_suffix: String,
    /// The caller's target arch (e.g., Some("x86_64"))
    pub target_arch: Option<&'static str>,
    /// The token ident available in the caller's scope (e.g., `token`, `__token`, `_token`)
    pub token_ident: Ident,
    /// Whether a real token is in scope. `false` for tokenless tier bodies
    /// (tier-based `#[rite(v3, …)]`). Tokenless callers can construct the
    /// selected callee's proof with `from_context()` for a covered tier.
    pub has_token: bool,
    /// Only tokenless rite contexts opt into feature-proved token construction.
    pub derive_token: bool,
}

/// Rewrite `incant!()` calls in a function body for a specific tier context.
///
/// Walks the token stream looking for `incant ! ( ... )` patterns.
/// Skips inner `fn` items (they don't inherit `#[target_feature]`).
/// Descends into closures and other expressions.
///
/// Returns a new TokenStream with incant! calls replaced by direct tier calls.
pub(crate) fn rewrite_incant_in_body(body: TokenStream, ctx: &CallerContext) -> TokenStream {
    // Most kernels have no dispatch inside them. Keep their original groups,
    // spans, and token storage instead of allocating two vectors at every depth.
    if !crate::common::tokens_contain_ident(&body, &["incant", "dispatch_variant"]) {
        return body;
    }
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
        if (is_ident(&tokens[i], "incant") || is_ident(&tokens[i], "dispatch_variant"))
            && i + 2 < tokens.len()
            && is_punct(&tokens[i + 1], '!')
            && let Some(TokenTree::Group(group)) = tokens.get(i + 2)
            && group.delimiter() == Delimiter::Parenthesis
        {
            // Try to parse the incant arguments
            let inner = group.stream();
            if let Ok(input) = syn::parse2::<IncantInput>(inner) {
                match rewrite_single_incant(&input, ctx) {
                    // Rewrote this incant! call to a direct tier call.
                    Some(rewritten) => result.extend(rewritten),
                    // Not rewritten (passthrough / tokenless body / parse issue):
                    // emit the original `incant ! ( ... )` tokens verbatim so the
                    // standalone `incant!` macro expands them unchanged (byte-identical).
                    None => {
                        result.push(tokens[i].clone());
                        result.push(tokens[i + 1].clone());
                        result.push(tokens[i + 2].clone());
                    }
                }
                i += 3; // skip `incant`, `!`, `(...)`
                continue;
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

/// Rewrite a single parsed `incant!()` invocation. Returns `None` when the call
/// should be left as-is (the caller then emits the original tokens verbatim).
fn rewrite_single_incant(input: &IncantInput, ctx: &CallerContext) -> Option<TokenStream> {
    // `without token`: tokenless direct call to the caller's exact-tier variant.
    // `f_<caller_tier>(args)` — no token threaded, no summon. The caller is
    // already inside the matching `#[target_feature]` region, so this is a safe
    // matching-feature call; a missing `f_<tier>` or a feature mismatch is a
    // compile error. Works whether or not the caller itself holds a token.
    if input.without_token {
        let fn_suffixed = suffix_path(&input.func_path, &ctx.tier_suffix);
        let args = &input.args;
        return Some(quote! { #fn_suffixed(#(#args),*) });
    }

    // Passthrough mode (`with token`): leave for the standalone incant! to handle.
    if input.with_token.is_some() {
        return None;
    }

    // A tokenless feature context already proves covered tiers. Do not repeat
    // detection or attempt an implicit upgrade while composing its helpers.
    if !ctx.has_token {
        return ctx
            .derive_token
            .then(|| rewrite_tokenless_incant(input, ctx))
            .flatten();
    }

    let func_path = &input.func_path;
    let args = &input.args;

    let tiers = match &input.tiers {
        None => tiers::default_tiers(true),
        Some((names, _)) => {
            tiers::resolve_tiers(names, proc_macro2::Span::call_site(), true).ok()?
        }
    };

    // Partition callee tiers into:
    // 1. Exact match (caller tier == callee tier) — direct call, no method
    // 2. Downgrade (caller can reach callee via ancestor chain) — token.method()
    // 3. Upgrade (callee above caller, or unreachable branch) — needs summon
    // 4. Scalar/default fallback
    //
    // Uses can_downgrade_tier() which checks the actual parent DAG, not just
    // numeric priority. This correctly handles cross-branch cases (e.g., V4
    // cannot downgrade to V3_crypto even though V4 has higher priority).
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
        if rt.suffix == ctx.tier_suffix {
            // Exact match — direct call, no downgrade method
            if direct_tier.is_none() {
                direct_tier = Some(rt);
            }
        } else if crate::generated::can_downgrade_tier(&ctx.tier_suffix, rt.suffix) {
            // Caller can downgrade to this tier — direct call with method
            if direct_tier.is_none() {
                direct_tier = Some(rt);
            }
        } else {
            // Can't downgrade — needs upgrade summon
            upgrade_tiers.push(rt);
        }
    }

    // Filtering preserves resolve_tiers' descending priority order.

    let token_ident = &ctx.token_ident;

    // Build the upgrade attempt arms (need labeled block + break)
    let mut upgrade_arms = Vec::new();
    for rt in &upgrade_tiers {
        let fn_suffixed = suffix_path(func_path, rt.suffix);
        let token_path: syn::Path = syn::parse_str(rt.token_path).unwrap();

        let token_expr = quote! { __t };
        let caller_ident = token_ident.to_string();
        let call_args =
            crate::common::build_call_args_with_ident(args, &token_expr, Some(&caller_ident));
        let check = quote! {
            if let Some(__t) = #token_path::summon() {
                break '__incant_rewrite #fn_suffixed(#call_args);
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
        let token_expr = if rt.suffix == ctx.tier_suffix {
            // Exact match — pass token directly
            quote! { #token_ident }
        } else {
            // Downgrade — call the tier method on the token
            let downgrade_method = format_ident!("{}", rt.suffix);
            quote! { #token_ident.#downgrade_method() }
        };
        let caller_ident = token_ident.to_string();
        let call_args =
            crate::common::build_call_args_with_ident(args, &token_expr, Some(&caller_ident));
        quote! { #fn_suffixed(#call_args) }
    } else {
        // No same-arch tier at or below caller — fall through to scalar
        let has_default = tiers.iter().any(|t| t.name == "default");
        if has_default {
            let fn_default = suffix_path(func_path, "default");
            // Strip token markers from args for tokenless default call
            let caller_ident = token_ident.to_string();
            let default_args: Vec<&syn::Expr> = args
                .iter()
                .filter(|a| {
                    !crate::common::is_bare_ident_pub(a, "Token")
                        && !crate::common::is_bare_ident_pub(a, &caller_ident)
                })
                .collect();
            quote! { #fn_default(#(#default_args),*) }
        } else {
            let fn_scalar = suffix_path(func_path, "scalar");
            let scalar_args = crate::common::build_scalar_call_args(args);
            quote! { #fn_scalar(#scalar_args) }
        }
    };

    Some(if upgrade_arms.is_empty() {
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
    })
}

/// Select only callees covered by the caller's feature context. The registry
/// DAG determines eligibility, and rustc independently checks the emitted
/// `from_context()` call against the actual caller attributes. No unsafe code,
/// runtime detection, hidden token binding, or evaluation of discarded args.
fn rewrite_tokenless_incant(input: &IncantInput, ctx: &CallerContext) -> Option<TokenStream> {
    let tiers = match &input.tiers {
        None => tiers::default_tiers(true),
        Some((names, _)) => {
            tiers::resolve_tiers(names, proc_macro2::Span::call_site(), true).ok()?
        }
    };
    let eligible = tiers.iter().filter(|tier| {
        tier.name == "scalar"
            || tier.name == "default"
            || (tier.target_arch == ctx.target_arch
                && (tier.suffix == ctx.tier_suffix
                    || crate::generated::can_downgrade_tier(&ctx.tier_suffix, tier.suffix)))
    });

    // Build from the fallback up: cfg-gating a preferred tier must expose the
    // next covered tier, never leave a reference to an omitted function.
    let mut result = quote! {
        compile_error!("incant!: no callee tier is covered by this tokenless context; include a covered tier or scalar/default fallback")
    };
    for tier in eligible.rev() {
        let function = suffix_path(&input.func_path, tier.suffix);
        let call = if tier.name == "default" {
            let args: Vec<_> = input
                .args
                .iter()
                .filter(|arg| !crate::common::is_bare_ident_pub(arg, "Token"))
                .collect();
            quote! { #function(#(#args),*) }
        } else {
            let token = if tier.name == "scalar" {
                quote! { archmage::ScalarToken }
            } else {
                let token_path: syn::Path = syn::parse_str(tier.token_path).ok()?;
                quote! { #token_path::from_context() }
            };
            let args = crate::common::build_call_args(&input.args, &token);
            quote! { #function(#args) }
        };
        result = if let Some(feature) = &tier.feature_gate {
            let allow = tier
                .allow_unexpected_cfg
                .then(|| quote! { #[allow(unexpected_cfgs)] });
            quote! {{
                #allow #[cfg(feature = #feature)] { #call }
                #allow #[cfg(not(feature = #feature))] { #result }
            }}
        } else {
            call
        };
    }
    Some(result)
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

    fn make_ctx(tier: &str, _priority: u32, arch: Option<&'static str>) -> CallerContext {
        CallerContext {
            tier_suffix: tier.to_string(),
            target_arch: arch,
            token_ident: format_ident!("__token"),
            has_token: true,
            derive_token: false,
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

    #[test]
    fn tokenless_context_never_probes_a_stronger_or_unrelated_tier() {
        let mut ctx = make_ctx("v3", 30, Some("x86_64"));
        ctx.has_token = false;
        ctx.derive_token = true;
        let result = rewrite_incant_in_body(
            quote! { incant!(work(value, Token), [v4, v3_crypto, neon, v2, -scalar]) },
            &ctx,
        )
        .to_string();
        assert!(result.contains("work_v2"), "{result}");
        assert!(result.contains("X64V2Token :: from_context"), "{result}");
        for absent in [
            "summon",
            "work_v4",
            "work_v3_crypto",
            "work_neon",
            "ScalarToken",
        ] {
            assert!(!result.contains(absent), "{result}");
        }
    }

    #[test]
    fn tokenless_rewrite_is_opt_in_and_respects_item_boundaries() {
        let mut ctx = make_ctx("v3", 30, Some("x86_64"));
        ctx.has_token = false;
        let body = quote! {
            fn nested() { incant!(work(x), [v3, scalar]); }
            dispatch_variant!(work(x), [v3, scalar]);
        };
        assert_eq!(
            rewrite_incant_in_body(body.clone(), &ctx).to_string(),
            body.to_string()
        );
        ctx.derive_token = true;
        let result = rewrite_incant_in_body(body, &ctx).to_string();
        assert!(result.contains("fn nested () { incant !"), "{result}");
        assert!(result.contains("work_v3"), "{result}");
        assert!(!result.contains("dispatch_variant"), "{result}");
    }
}
