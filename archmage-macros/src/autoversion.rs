//! `#[autoversion]` — combined variant generation + dispatch.
//!
//! Generates architecture-specific function variants and a runtime
//! dispatcher from a single annotated function.

use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident, quote, quote_spanned};
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
    /// The function is a receiver-less associated function in an inherent
    /// impl: variants are called as `Self::name_v3`.
    pub(crate) in_impl: bool,
    /// The function is a trait impl method: variants nest inside the
    /// dispatcher, since a trait impl cannot take extra items. A receiver
    /// needs `_self = Type` so the variants can take it as a parameter.
    pub(crate) in_trait: bool,
}

impl Parse for AutoversionArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut self_type = None;
        let mut tier_names = Vec::new();
        let mut cfg_feature = None;
        let mut in_impl = false;
        let mut in_trait = false;

        while !input.is_empty() {
            // Check for +tier/-tier (modify defaults) before consuming ident
            if input.peek(Token![+]) || input.peek(Token![-]) {
                tier_names.push(crate::tiers::parse_one_tier(input)?);
            } else {
                let ident: Ident = input.parse()?;
                if ident == "_self" {
                    let _: Token![=] = input.parse()?;
                    self_type = Some(input.parse()?);
                } else if ident == "in_impl" {
                    in_impl = true;
                } else if ident == "in_trait" || ident == "nested" {
                    in_trait = true;
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
            in_impl,
            in_trait,
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

    if args.in_impl && args.in_trait {
        return syn::Error::new_spanned(
            &input_fn.sig,
            "#[autoversion]: `in_impl` and `in_trait` describe different places; use one.",
        )
        .to_compile_error();
    }
    if args.in_trait && has_self && args.self_type.is_none() {
        return syn::Error::new_spanned(
            &input_fn.sig,
            "#[autoversion(in_trait)] on a method needs `_self = Type`: the variants nest \
             inside the dispatcher and take the receiver as a `_self` parameter, so the \
             macro must know its type. Example: #[autoversion(v3, scalar, in_trait, _self = MyType)]",
        )
        .to_compile_error();
    }
    // Each variant would return its own opaque type, and one dispatcher cannot
    // return both. Say so instead of leaving rustc's E0308 on generated code.
    if let syn::ReturnType::Type(_, ty) = &input_fn.sig.output
        && matches!(**ty, Type::ImplTrait(_))
    {
        return syn::Error::new_spanned(
            &input_fn.sig.output,
            "#[autoversion] cannot dispatch an `impl Trait` return type: every tier variant \
             returns a distinct opaque type and the dispatcher can return only one. Return a \
             named type, or `Box<dyn Trait>`.",
        )
        .to_compile_error();
    }

    // Find token parameter (SimdToken or ScalarToken), or auto-inject one.
    //
    // Three modes:
    // - ScalarToken: kept in dispatcher (real type, compiles, incant!-compatible)
    // - SimdToken: stripped from dispatcher (legacy, deprecated)
    // - None: auto-inject internally, strip from dispatcher (tokenless)
    let token_param = match find_autoversion_token_param(&input_fn.sig) {
        Err(e) => return e.to_compile_error(),
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
    let tiers = match &args.tiers {
        None => default_tiers(false),
        Some(names) => match resolve_tiers(names, input_fn.sig.ident.span(), false) {
            Ok(t) => t,
            Err(e) => return e.to_compile_error(),
        },
    };

    // Strip #[arcane] / #[rite] to prevent double-wrapping
    input_fn
        .attrs
        .retain(|attr| !attr.path().is_ident("arcane") && !attr.path().is_ident("rite"));

    let fn_name = &input_fn.sig.ident;
    let vis = input_fn.vis.clone();

    // Move attrs to dispatcher only; variants get no user attrs
    let fn_attrs: Vec<Attribute> = core::mem::take(&mut input_fn.attrs);

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

        if args.in_trait {
            // The variant nests inside the dispatcher, where it cannot be a
            // method. Its receiver becomes a `_self` parameter of the named
            // type, `Self` becomes that type, and `self` in the body becomes
            // `_self`.
            if let Some(self_ty) = &args.self_type
                && has_self
            {
                let receiver_param: FnArg = match &variant_fn.sig.inputs[0] {
                    FnArg::Receiver(receiver) => match &receiver.kind {
                        syn::ReceiverKind::Reference(_, _, Some(_)) => {
                            parse_quote!(_self: &mut #self_ty)
                        }
                        syn::ReceiverKind::Reference(_, _, None) => parse_quote!(_self: &#self_ty),
                        _ => parse_quote!(_self: #self_ty),
                    },
                    FnArg::Typed(_) => unreachable!("has_self checked the first parameter"),
                };
                variant_fn.sig.inputs[0] = receiver_param;
                let self_ident = format_ident!("_self");
                variant_fn.body = replace_self_value_in_tokens(
                    replace_self_in_tokens(variant_fn.body.clone(), self_ty),
                    &self_ident,
                );
                variant_fn.sig.output = syn::parse2(replace_self_in_tokens(
                    variant_fn.sig.output.to_token_stream(),
                    self_ty,
                ))
                .expect("replacing Self keeps the return type parseable");
            }
        } else if (tier.name == "scalar" || tier.name == "default")
            && has_self
            && args.self_type.is_some()
        {
            // Fallback (scalar/default) with _self = Type: inject `let _self = self;`
            // preamble so body's _self references resolve (non-fallback variants
            // get this from #[arcane(_self = Type)])
            let original_body = variant_fn.body.clone();
            variant_fn.body = quote!(let _self = self; #original_body);
        }

        // Rewrite incant!() calls in the variant body to direct tier calls.
        // scalar/default have no token to thread, so they only rewrite
        // `incant!(.. without token)` (the tokenless variant call); plain
        // `incant!` there is left for standalone expansion, as before.
        let has_token = tier.name != "scalar" && tier.name != "default";
        let token_ident = if has_token {
            token_param.ident.clone()
        } else {
            quote::format_ident!("_")
        };
        let ctx = crate::rewrite::CallerContext {
            tier_suffix: tier.suffix.to_string(),
            target_arch: tier.target_arch,
            token_ident,
            has_token,
            derive_token: false,
        };
        variant_fn.body = crate::rewrite::rewrite_incant_in_body(variant_fn.body, &ctx);

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
            let arcane_attr = if args.in_trait {
                // The receiver is already a plain `_self` parameter.
                quote! { #[archmage::arcane] }
            } else if let Some(ref self_type) = args.self_type {
                quote! { #[archmage::arcane(_self = #self_type)] }
            } else if args.in_impl {
                quote! { #[archmage::arcane(in_impl)] }
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

    // Name wildcard and tuple patterns so the dispatcher can forward them.
    // The dispatcher only forwards, so the rebinds are dropped; the variants
    // keep the user's patterns and `#[arcane]` rebinds them there. A kept
    // ScalarToken wildcard gets a name too, which the dispatcher ignores.
    let mut dispatcher_sig = input_fn.sig.clone();
    dispatcher_sig.inputs = dispatcher_inputs.into_iter().collect();
    let _ = rename_non_ident_params(&mut dispatcher_sig);
    let dispatcher_inputs: Vec<FnArg> = dispatcher_sig.inputs.into_iter().collect();

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
    // syn 3 replaced `Signature::unsafety: Option<Token![unsafe]>` with
    // `Signature::safety: Safety`. A free `fn` can only parse as `Unsafe` or
    // `Default` (the `safe` qualifier is only accepted inside `unsafe extern`
    // blocks), so this matches the old `unsafety.is_some()` exactly.
    let is_unsafe = matches!(input_fn.sig.safety, syn::Safety::Unsafe(_));

    let mut dispatch_arms = Vec::new();
    for (target_arch, group_tiers) in &arch_groups {
        let mut tier_checks = Vec::new();
        for rt in group_tiers {
            let suffixed = format_ident!("{}_{}", fn_name, rt.suffix);
            let token_path: syn::Path = syn::parse_str(rt.token_path).unwrap();

            let raw_call = if has_self && args.in_trait {
                quote! { #suffixed #turbofish(self, __t, #(#dispatch_args),*) }
            } else if has_self {
                quote! { self.#suffixed #turbofish(__t, #(#dispatch_args),*) }
            } else if args.in_impl {
                quote! { Self::#suffixed #turbofish(__t, #(#dispatch_args),*) }
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
    // default: tokenless call; scalar: call with ScalarToken
    let fallback_token = if has_default_tier {
        quote! {}
    } else {
        quote! { archmage::ScalarToken, }
    };
    let raw_fallback = if has_self && args.in_trait {
        quote! { #fallback_name #turbofish(self, #fallback_token #(#dispatch_args),*) }
    } else if has_self {
        quote! { self.#fallback_name #turbofish(#fallback_token #(#dispatch_args),*) }
    } else if args.in_impl {
        quote! { Self::#fallback_name #turbofish(#fallback_token #(#dispatch_args),*) }
    } else {
        quote! { #fallback_name #turbofish(#fallback_token #(#dispatch_args),*) }
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
    // `Safety`'s ToTokens emits `unsafe` for `Unsafe` and nothing for `Default`,
    // so interpolating it below produces the same tokens syn 2's
    // `Option<Token![unsafe]>` did.
    let unsafety = &input_fn.sig.safety;

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
                // Suppress unused_imports on archs where every dispatch arm is
                // cfg'd out (e.g., 32-bit x86 with [v3, neon, wasm128] tiers).
                // The `use` carries the user's span, so a downstream warning
                // would point at the user's fn declaration. See issue #34.
                #[allow(unused_imports)]
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
                // Suppress unused_imports on archs where every dispatch arm is
                // cfg'd out (e.g., 32-bit x86 with [v3, neon, wasm128] tiers).
                // The `use` carries the user's span, so a downstream warning
                // would point at the user's fn declaration. See issue #34.
                #[allow(unused_imports)]
                use archmage::SimdToken;
                #(#dispatch_arms)*
                #fallback_call
            }
        }
    };

    if args.in_trait {
        // A trait impl cannot take extra items: the variants live inside the
        // dispatcher's body, where block items are visible throughout.
        return place_variants_inside(dispatcher, quote! { #(#variants)* });
    }
    quote! {
        #dispatcher
        #(#variants)*
    }
}

/// Insert `items` at the start of every function body in `dispatcher` (one
/// body normally, two under `cfg(feature)`). The bodies are the top-level
/// brace groups; attribute arguments never use braces here.
fn place_variants_inside(dispatcher: TokenStream, items: TokenStream) -> TokenStream {
    let mut out = TokenStream::new();
    for tt in dispatcher {
        match tt {
            proc_macro2::TokenTree::Group(group)
                if group.delimiter() == proc_macro2::Delimiter::Brace =>
            {
                let body = group.stream();
                let mut new_group =
                    proc_macro2::Group::new(group.delimiter(), quote! { #items #body });
                new_group.set_span(group.span());
                out.extend([proc_macro2::TokenTree::Group(new_group)]);
            }
            other => out.extend([other]),
        }
    }
    out
}
