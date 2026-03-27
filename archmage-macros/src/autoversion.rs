//! `#[autoversion]` — combined variant generation + dispatch.
//!
//! Generates architecture-specific function variants and a runtime
//! dispatcher from a single annotated function.

use proc_macro::TokenStream;
use quote::{format_ident, quote, quote_spanned};
use syn::{
    Attribute, FnArg, Ident, PatType, Signature, Token, Type,
    parse::{Parse, ParseStream},
    parse_quote,
};

use crate::common::*;
use crate::generated::token_to_features;
use crate::tiers::*;

/// Arguments to the `#[autoversion]` macro.
pub(crate) struct AutoversionArgs {
    /// The concrete type to use for `self` receiver (inherent methods only).
    pub(crate) self_type: Option<Type>,
    /// Explicit tier names (None = default tiers).
    pub(crate) tiers: Option<Vec<String>>,
    /// When set, emit full autoversion under `#[cfg(feature = "...")]` and a
    /// plain scalar fallback under `#[cfg(not(feature = "..."))]`. Solves the
    /// hygiene issue with `macro_rules!` wrappers.
    pub(crate) cfg_feature: Option<String>,
}

impl Parse for AutoversionArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut self_type = None;
        let mut tier_names = Vec::new();
        let mut cfg_feature = None;

        while !input.is_empty() {
            // Check for +tier/-tier (modify defaults) before consuming ident
            if input.peek(Token![+]) || input.peek(Token![-]) {
                tier_names.push(crate::tiers::parse_one_tier(input)?);
            } else {
                let ident: Ident = input.parse()?;
                if ident == "_self" {
                    let _: Token![=] = input.parse()?;
                    self_type = Some(input.parse()?);
                } else if ident == "cfg" {
                    let content;
                    syn::parenthesized!(content in input);
                    let feat: Ident = content.parse()?;
                    cfg_feature = Some(feat.to_string());
                } else {
                    // Treat as tier name, optionally with cfg gate
                    tier_names.push(crate::tiers::parse_tier_name_with_gate(&ident, input)?);
                }
            }
            if input.peek(Token![,]) {
                let _: Token![,] = input.parse()?;
            }
        }

        Ok(AutoversionArgs {
            self_type,
            tiers: if tier_names.is_empty() {
                None
            } else {
                Some(tier_names)
            },
            cfg_feature,
        })
    }
}

/// What kind of token parameter was found in the autoversion function signature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AutoversionTokenKind {
    /// `SimdToken` — legacy placeholder, stripped from dispatcher (deprecated).
    SimdToken,
    /// `ScalarToken` — real type, kept in dispatcher for incant! compatibility.
    ScalarToken,
    /// No token found — auto-injected internally, stripped from dispatcher.
    AutoInjected,
}

/// Information about the token parameter in an autoversion function signature.
#[derive(Debug)]
pub(crate) struct AutoversionTokenParam {
    /// Index of the parameter in `sig.inputs`
    pub(crate) index: usize,
    /// The parameter identifier
    #[allow(dead_code)]
    pub(crate) ident: Ident,
    /// What kind of token was found
    pub(crate) kind: AutoversionTokenKind,
}

/// Find a token parameter (`SimdToken` or `ScalarToken`) in a function signature
/// for `#[autoversion]`.
///
/// Returns Ok(Some) for recognized tokens, Ok(None) for no token, or Err for
/// concrete SIMD tokens (X64V3Token etc.) which should use `#[arcane]` instead.
pub(crate) fn find_autoversion_token_param(
    sig: &Signature,
) -> Result<Option<AutoversionTokenParam>, syn::Error> {
    for (i, arg) in sig.inputs.iter().enumerate() {
        if let FnArg::Typed(PatType { pat, ty, .. }) = arg
            && let Type::Path(type_path) = ty.as_ref()
            && let Some(seg) = type_path.path.segments.last()
        {
            let name = seg.ident.to_string();

            // Recognized autoversion tokens
            let kind = if name == "SimdToken" {
                AutoversionTokenKind::SimdToken
            } else if name == "ScalarToken" {
                AutoversionTokenKind::ScalarToken
            } else if token_to_features(&name).is_some() {
                // It's a concrete SIMD token (X64V3Token, NeonToken, etc.)
                return Err(syn::Error::new_spanned(
                    ty,
                    format!(
                        "#[autoversion] generates multi-tier dispatch — it can't take a \
                         concrete token like `{name}`.\n\
                         Use #[arcane] or #[rite] for single-token functions.\n\
                         Use #[autoversion] with no token parameter (recommended) or \
                         ScalarToken for incant! nesting."
                    ),
                ));
            } else {
                continue;
            };

            let ident = match pat.as_ref() {
                syn::Pat::Ident(pi) => pi.ident.clone(),
                syn::Pat::Wild(w) => Ident::new("__autoversion_token", w.underscore_token.span),
                _ => continue,
            };
            return Ok(Some(AutoversionTokenParam {
                index: i,
                ident,
                kind,
            }));
        }
    }
    Ok(None)
}

/// Core implementation for `#[autoversion]`.
///
/// Generates suffixed SIMD variants (like `#[magetypes]`) and a runtime
/// dispatcher function (like `incant!`) from a single annotated function.
pub(crate) fn autoversion_impl(mut input_fn: LightFn, args: AutoversionArgs) -> TokenStream {
    // Check for self receiver
    let has_self = input_fn
        .sig
        .inputs
        .first()
        .is_some_and(|arg| matches!(arg, FnArg::Receiver(_)));

    // _self = Type is only needed for trait impls (nested mode in #[arcane]).
    // For inherent methods, self/Self work naturally in sibling mode.

    // Find token parameter (SimdToken or ScalarToken), or auto-inject one.
    //
    // Three modes:
    // - ScalarToken: kept in dispatcher (real type, compiles, incant!-compatible)
    // - SimdToken: stripped from dispatcher (legacy, deprecated)
    // - None: auto-inject internally, strip from dispatcher (tokenless)
    let token_param = match find_autoversion_token_param(&input_fn.sig) {
        Err(e) => return e.to_compile_error().into(),
        Ok(Some(p)) => p,
        Ok(None) => {
            let insert_pos = if has_self { 1 } else { 0 };
            let token_arg: FnArg = parse_quote!(_token: SimdToken);
            input_fn.sig.inputs.insert(insert_pos, token_arg);
            AutoversionTokenParam {
                index: insert_pos,
                ident: Ident::new("_token", input_fn.sig.ident.span()),
                kind: AutoversionTokenKind::AutoInjected,
            }
        }
    };

    // Deprecation warning for SimdToken. We emit a function-local deprecation
    // by referencing a deprecated item inside the dispatcher body.
    let simdtoken_deprecation_in_body = if token_param.kind == AutoversionTokenKind::SimdToken {
        let msg = "SimdToken parameter in #[autoversion] is deprecated — \
                   remove it (tokenless) or use ScalarToken for incant! nesting";
        Some(quote! {
            {
                #[deprecated(note = #msg)]
                #[allow(dead_code)]
                const SIMDTOKEN_DEPRECATED: () = ();
                let _ = SIMDTOKEN_DEPRECATED;
            }
        })
    } else {
        None
    };

    // Whether to keep the token param in the dispatcher.
    // ScalarToken is a real type → keep it (incant! compatibility).
    // SimdToken and AutoInjected → strip (can't compile / internal).
    let keep_token_in_dispatcher = token_param.kind == AutoversionTokenKind::ScalarToken;

    // Resolve tiers — autoversion always includes v4 in its defaults because it
    // generates scalar code compiled with #[target_feature], not import_intrinsics.
    let tier_names: Vec<String> = match &args.tiers {
        Some(names) => names.clone(),
        None => DEFAULT_TIER_NAMES.iter().map(|s| s.to_string()).collect(),
    };
    // autoversion never skips avx512 — it generates scalar code with #[target_feature]
    let tiers = match resolve_tiers(&tier_names, input_fn.sig.ident.span(), false) {
        Ok(t) => t,
        Err(e) => return e.to_compile_error().into(),
    };

    // Strip #[arcane] / #[rite] to prevent double-wrapping
    input_fn
        .attrs
        .retain(|attr| !attr.path().is_ident("arcane") && !attr.path().is_ident("rite"));

    let fn_name = &input_fn.sig.ident;
    let vis = input_fn.vis.clone();

    // Move attrs to dispatcher only; variants get no user attrs
    let fn_attrs: Vec<Attribute> = input_fn.attrs.drain(..).collect();

    // =========================================================================
    // Generate suffixed variants
    // =========================================================================
    //
    // AST manipulation only — we clone the parsed LightFn and swap the token
    // param's type annotation. No serialize/reparse round-trip. The body is
    // never touched unless _self = Type requires a `let _self = self;`
    // preamble on the scalar variant.

    let mut variants = Vec::new();

    for tier in &tiers {
        let mut variant_fn = input_fn.clone();

        // Variants are always private — only the dispatcher is public.
        variant_fn.vis = syn::Visibility::Inherited;

        // Rename: process → process_v3
        variant_fn.sig.ident = format_ident!("{}_{}", fn_name, tier.suffix);

        // Replace token param type with concrete token type.
        // For "default" tier: remove the token param entirely (tokenless variant).
        if tier.name == "default" {
            let mut inputs: Vec<FnArg> = variant_fn.sig.inputs.iter().cloned().collect();
            inputs.remove(token_param.index);
            variant_fn.sig.inputs = inputs.into_iter().collect();
        } else {
            let concrete_type: Type = syn::parse_str(tier.token_path).unwrap();
            if let FnArg::Typed(pt) = &mut variant_fn.sig.inputs[token_param.index] {
                *pt.ty = concrete_type;
            }
        }

        // Fallback (scalar/default) with _self = Type: inject `let _self = self;` preamble
        // so body's _self references resolve (non-fallback variants get this from
        // #[arcane(_self = Type)])
        if (tier.name == "scalar" || tier.name == "default") && has_self && args.self_type.is_some()
        {
            let original_body = variant_fn.body.clone();
            variant_fn.body = quote!(let _self = self; #original_body);
        }

        // cfg guard: arch + optional feature gate from tier(feature) syntax
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

        // All variants are private implementation details of the dispatcher.
        // Suppress dead_code: if the dispatcher is unused, rustc warns on IT
        // (via quote_spanned! with the user's span). Warning on individual
        // variants would be confusing — the user didn't write _scalar or _v3.
        if tier.name != "scalar" && tier.name != "default" {
            let arcane_attr = if let Some(ref self_type) = args.self_type {
                quote! { #[archmage::arcane(_self = #self_type)] }
            } else {
                quote! { #[archmage::arcane] }
            };
            variants.push(quote! {
                #cfg_guard
                #[allow(dead_code)]
                #arcane_attr
                #variant_fn
            });
        } else {
            variants.push(quote! {
                #cfg_guard
                #[allow(dead_code)]
                #variant_fn
            });
        }
    }

    // =========================================================================
    // Generate dispatcher (adapted from gen_incant_entry)
    // =========================================================================

    // Build dispatcher inputs.
    //
    // ScalarToken is kept (real type, incant!-compatible).
    // SimdToken and AutoInjected are stripped.
    let mut dispatcher_inputs: Vec<FnArg> = input_fn.sig.inputs.iter().cloned().collect();
    if !keep_token_in_dispatcher {
        dispatcher_inputs.remove(token_param.index);
    }

    // Rename wildcard params so we can pass them as arguments.
    // Skip the kept ScalarToken param if it's a wildcard — the dispatcher
    // ignores it (does its own summon()), no need to name it.
    let mut wild_counter = 0u32;
    for (i, arg) in dispatcher_inputs.iter_mut().enumerate() {
        if keep_token_in_dispatcher && i == token_param.index {
            continue; // Don't rename the kept token's pattern
        }
        if let FnArg::Typed(pat_type) = arg
            && matches!(pat_type.pat.as_ref(), syn::Pat::Wild(_))
        {
            let ident = format_ident!("__autoversion_wild_{}", wild_counter);
            wild_counter += 1;
            *pat_type.pat = syn::Pat::Ident(syn::PatIdent {
                attrs: vec![],
                by_ref: None,
                mutability: None,
                ident,
                subpat: None,
            });
        }
    }

    // Collect argument idents for dispatch calls (exclude self receiver
    // AND the kept ScalarToken param — variants get their own token from
    // summon(), not from the dispatcher's ScalarToken parameter).
    let dispatch_args: Vec<Ident> = dispatcher_inputs
        .iter()
        .enumerate()
        .filter_map(|(i, arg)| {
            if keep_token_in_dispatcher && i == token_param.index {
                return None; // Skip the kept token param
            }
            if let FnArg::Typed(PatType { pat, .. }) = arg
                && let syn::Pat::Ident(pi) = pat.as_ref()
            {
                return Some(pi.ident.clone());
            }
            None
        })
        .collect();

    // Build turbofish for forwarding type/const generics to variant calls
    let turbofish = build_turbofish(&input_fn.sig.generics);

    // Group non-fallback tiers by target_arch for cfg blocks
    let mut arch_groups: Vec<(Option<&str>, Vec<&ResolvedTier>)> = Vec::new();
    for tier in &tiers {
        if tier.name == "scalar" || tier.name == "default" {
            continue;
        }
        if let Some(group) = arch_groups.iter_mut().find(|(a, _)| *a == tier.target_arch) {
            group.1.push(tier);
        } else {
            arch_groups.push((tier.target_arch, vec![tier]));
        }
    }

    // If the original function is `unsafe fn`, the dispatcher must also be `unsafe fn`
    // and variant calls must be wrapped in `unsafe {}`.
    let is_unsafe = input_fn.sig.unsafety.is_some();

    let mut dispatch_arms = Vec::new();
    for (target_arch, group_tiers) in &arch_groups {
        let mut tier_checks = Vec::new();
        for rt in group_tiers {
            let suffixed = format_ident!("{}_{}", fn_name, rt.suffix);
            let token_path: syn::Path = syn::parse_str(rt.token_path).unwrap();

            let raw_call = if has_self {
                quote! { self.#suffixed #turbofish(__t, #(#dispatch_args),*) }
            } else {
                quote! { #suffixed #turbofish(__t, #(#dispatch_args),*) }
            };

            // Wrap call in unsafe if the original function (and thus variants) is unsafe
            let call = if is_unsafe {
                quote! { unsafe { #raw_call } }
            } else {
                raw_call
            };

            let check = quote! {
                if let Some(__t) = #token_path::summon() {
                    return #call;
                }
            };

            if let Some(feat) = &rt.feature_gate {
                let allow_attr = if rt.allow_unexpected_cfg {
                    quote! { #[allow(unexpected_cfgs)] }
                } else {
                    quote! {}
                };
                tier_checks.push(quote! {
                    #allow_attr
                    #[cfg(feature = #feat)]
                    { #check }
                });
            } else {
                tier_checks.push(check);
            }
        }

        let inner = quote! { #(#tier_checks)* };

        if let Some(arch) = target_arch {
            dispatch_arms.push(quote! {
                #[cfg(target_arch = #arch)]
                { #inner }
            });
        } else {
            dispatch_arms.push(inner);
        }
    }

    // Fallback call (always available, no summon needed)
    let has_default_tier = tiers.iter().any(|t| t.name == "default");
    let fallback_suffix = if has_default_tier {
        "default"
    } else {
        "scalar"
    };
    let fallback_name = format_ident!("{}_{}", fn_name, fallback_suffix);
    let raw_fallback = if has_default_tier {
        // default: tokenless call
        if has_self {
            quote! { self.#fallback_name #turbofish(#(#dispatch_args),*) }
        } else {
            quote! { #fallback_name #turbofish(#(#dispatch_args),*) }
        }
    } else {
        // scalar: call with ScalarToken
        if has_self {
            quote! { self.#fallback_name #turbofish(archmage::ScalarToken, #(#dispatch_args),*) }
        } else {
            quote! { #fallback_name #turbofish(archmage::ScalarToken, #(#dispatch_args),*) }
        }
    };
    let fallback_call = if is_unsafe {
        quote! { unsafe { #raw_fallback } }
    } else {
        raw_fallback
    };

    // Build dispatcher function
    let dispatcher_inputs_punct: syn::punctuated::Punctuated<FnArg, Token![,]> =
        dispatcher_inputs.into_iter().collect();
    let output = &input_fn.sig.output;
    let generics = &input_fn.sig.generics;
    let where_clause = &generics.where_clause;
    let unsafety = &input_fn.sig.unsafety;

    // Use the user's span for the dispatcher so dead_code lint fires on the
    // function the user actually wrote, not on invisible generated variants.
    let user_span = fn_name.span();

    // autoversion uses `return` instead of `break '__dispatch` — no labeled block
    // needed. This avoids label hygiene issues when #[autoversion] is applied inside
    // macro_rules! (labels from proc macros can't be seen from macro_rules! contexts).
    let dispatcher = if let Some(ref feat) = args.cfg_feature {
        // cfg(feature): full dispatch when on, scalar-only when off
        quote_spanned! { user_span =>
            #[cfg(feature = #feat)]
            #(#fn_attrs)*
            #vis #unsafety fn #fn_name #generics (#dispatcher_inputs_punct) #output #where_clause {
                #simdtoken_deprecation_in_body
                use archmage::SimdToken;
                #(#dispatch_arms)*
                #fallback_call
            }

            #[cfg(not(feature = #feat))]
            #(#fn_attrs)*
            #vis #unsafety fn #fn_name #generics (#dispatcher_inputs_punct) #output #where_clause {
                #simdtoken_deprecation_in_body
                #fallback_call
            }
        }
    } else {
        quote_spanned! { user_span =>
            #(#fn_attrs)*
            #vis #unsafety fn #fn_name #generics (#dispatcher_inputs_punct) #output #where_clause {
                #simdtoken_deprecation_in_body
                use archmage::SimdToken;
                #(#dispatch_arms)*
                #fallback_call
            }
        }
    };

    let expanded = quote! {
        #dispatcher
        #(#variants)*
    };

    expanded.into()
}
