//! Feature-proof boundaries shared by all attribute frontends.
//! Parsing syntax is kept out of this module; the proof is authenticated before
//! emitting the single unsafe call. Body tokens are never parsed as expressions.
use super::inline::InlinePolicy;
use crate::common::*;
use crate::token_discovery::*;
use proc_macro2::{Ident, Span, TokenStream};
use quote::{ToTokens, format_ident, quote};
use syn::{FnArg, Type};

/// Validate a concrete token using the tier-specific constant supplied by archmage.
///
/// The signature's full type path preserves renamed dependencies and re-exports.
/// A weaker tier lacks the expected constant, so accidental aliasing is rejected.
/// The initializer is checked once in archmage; each expansion only references it.
/// This public constant is an accidental-misuse check, not an anti-forgery boundary.
/// The exact archmage-macros dependency pin keeps this internal protocol in sync.
fn gen_token_assertion(
    token_type_name: Option<&str>,
    token_type: &Option<Type>,
    suppress_const_test: bool,
) -> TokenStream {
    if suppress_const_test {
        return quote! {};
    }
    if let (Some(name), Some(ty)) = (token_type_name, token_type)
        && let Some(expected_tag) = crate::generated::expected_tier_tag(name)
    {
        let assertion = format_ident!("__ARCHMAGE_ASSERT_TIER_{:08X}", expected_tag);
        return quote! { let _: () = <#ty>::#assertion; };
    }
    quote! {}
}

#[derive(Default)]
pub(crate) struct BoundaryOptions {
    /// Attune body policy; legacy frontends leave this unset.
    pub(crate) body_inline: Option<InlinePolicy>,
    /// Options spelled the same way in `#[rite]`: imports and `cfg(feature)`.
    pub(crate) shared: SharedOptions,
    /// Trusted generators may omit the accidental token-name mismatch check.
    /// Intrinsic feature checking remains enabled.
    pub(crate) suppress_const_test: bool,
    /// Use `#[inline(always)]` instead of `#[inline]` for the inner function.
    /// Requires nightly Rust with `#![feature(target_feature_inline_always)]`.
    pub(crate) inline_always: bool,
    /// The concrete type to use for `self` receiver.
    /// When specified, `self`/`&self`/`&mut self` is transformed to `_self: Type`/`&Type`/`&mut Type`.
    /// Implies `nested = true`.
    pub(crate) self_type: Option<Type>,
    /// Use a nested inner function instead of a sibling function. Spelled
    /// `nested` or `in_trait`; implied by `_self = Type`. Required in trait
    /// impls, which cannot take the extra sibling item.
    pub(crate) nested: bool,
    /// The function is a receiver-less associated function in an inherent
    /// impl, so the wrapper calls its sibling as `Self::__arcane_fn`. An
    /// attribute macro cannot see the enclosing impl, so this is spelled out.
    pub(crate) in_impl: bool,
    /// Attribute tokens for the forwarding wrapper only. In sibling mode the
    /// wrapper holds no author code, so lint levels here never reach the
    /// author's body; on wasm32 (one function) and in nested mode they do.
    pub(crate) wrapper_attrs: TokenStream,
}

/// Shared implementation for arcane/attune(wrap) boundary macros.
pub(crate) fn expand(
    mut input_fn: LightFn,
    macro_name: &str,
    args: BoundaryOptions,
) -> TokenStream {
    if let Some(err) = reject_manual_target_feature(&input_fn.attrs) {
        return err.to_compile_error();
    }
    let has_self_receiver = input_fn
        .sig
        .inputs
        .first()
        .is_some_and(|arg| matches!(arg, FnArg::Receiver(_)));

    if has_self_receiver && args.nested && args.self_type.is_none() {
        let msg = format!(
            "{} with self receiver in nested mode requires `_self = Type` argument.\n\
             Example: #[{}(in_trait, _self = MyType)]\n\
             The body may keep using `self`; it is renamed to `_self` for the inner function.\n\
             \n\
             Alternatively, remove `nested`/`in_trait` to use sibling expansion (default), \
             which handles self/Self naturally in inherent impls.",
            macro_name, macro_name
        );
        return syn::Error::new_spanned(&input_fn.sig, msg).to_compile_error();
    }

    let TokenParamInfo {
        ident: mut _token_ident,
        param_index: token_param_index,
        features,
        target_arch,
        token_type_name,
        magetypes_namespace,
        token_type,
        tier_traits,
    } = match find_token_param(&input_fn.sig).or_else(|| {
        // Receiver-only proof discovery is sound only in nested mode, where
        // `nested_self_param` binds `_self: [&]Type` and the wrapper calls
        // `__simd_inner(self)`, enforcing `Self == Type`. Explicitly typed
        // receivers (`self: &Other`, `self: Box<Other>`) are rejected
        // conservatively because `self: &Other` bypasses the `Self == Type`
        // check; require an explicit token parameter for those.
        if !args.nested {
            return None;
        }
        let self_ty = args.self_type.as_ref()?;
        let FnArg::Receiver(receiver) = input_fn.sig.inputs.first()? else {
            return None;
        };
        if !matches!(
            receiver.kind,
            syn::ReceiverKind::Value | syn::ReceiverKind::Reference(..)
        ) {
            return None;
        }
        find_token_in_self_type(&input_fn.sig, self_ty)
    }) {
        Some(result) => result,
        None => {
            return missing_token_error(
                &input_fn.sig,
                macro_name,
                &format!(
                    ". Supported forms:\n\
                     - Concrete: `token: X64V3Token`\n\
                     - impl Trait: `token: impl HasX64V2`\n\
                     - Generic: `fn foo<T: HasX64V2>(token: T, ...)`\n\
                     - With self: `#[{macro_name}(_self = Type)] fn method(&self, token: impl HasNeon, ...)`"
                ),
            );
        }
    };

    // One token decides the features. Two would have been resolved by position,
    // which is the surprising rule issue #122 reports.
    if let Some(err) = check_single_token_param(&input_fn.sig, macro_name) {
        return err;
    }

    // import_intrinsics with AVX-512 features needs archmage's avx512 feature
    // (propagated to archmage-macros) for the 512-bit safe memory wrappers.
    if let Some(err) = avx512_import_error(
        &input_fn.sig,
        args.shared.import_intrinsics,
        &features,
        token_type_name.unwrap_or("an AVX-512 token"),
    ) {
        return err;
    }

    prepend_to_body(
        &mut input_fn.body,
        generate_imports(
            target_arch,
            magetypes_namespace,
            args.shared.import_intrinsics,
            args.shared.import_magetypes,
        ),
    );

    let wrapper_sig = {
        let mut sig = input_fn.sig.clone();
        let _ = rename_non_ident_params(&mut sig);
        crate::common::rename_underscore_params(&mut sig);
        sig
    };
    if let Some(ident) = rename_wildcard_token(&mut input_fn.sig) {
        _token_ident = ident;
    }
    let wrapper_token_ident = match token_param_index {
        Some(idx) => match &wrapper_sig.inputs[idx] {
            FnArg::Typed(pat_type) => match pat_type.pat.as_ref() {
                syn::Pat::Ident(pat) => pat.ident.clone(),
                _ => unreachable!("wrapper_sig normalizes all typed params to Pat::Ident"),
            },
            FnArg::Receiver(_) => unreachable!("token param at idx was FnArg::Typed"),
        },
        None => _token_ident.clone(),
    };

    if let Some(type_name) = token_type_name
        && let Some(tier_suffix) = crate::generated::canonical_token_to_tier_suffix(type_name)
        && let Some(tier) = crate::tiers::find_tier(tier_suffix)
    {
        let ctx = crate::rewrite::CallerContext {
            tier_suffix,
            target_arch: tier.target_arch,
            token_ident: _token_ident.clone(),
            has_token: true,
            derive_token: false,
        };
        input_fn.body =
            crate::rewrite::rewrite_incant_in_function(input_fn.body, &ctx, Some(&input_fn.sig));
    }

    if token_type_name.is_none() {
        input_fn.body = crate::attune::call::rewrite_context(
            input_fn.body,
            crate::attune::call::Context {
                features: &features,
                target_arch,
            },
            Some(&input_fn.sig),
        );
    }

    let features_csv = features_csv(token_type_name, &features);
    if args.body_inline == Some(InlinePolicy::Always) && !features.is_empty() {
        return syn::Error::new_spanned(
            &input_fn.sig.ident,
            "inline(always) on target-feature bodies requires nightly; use inline(hint) or inline(never)",
        )
        .to_compile_error();
    }
    let inline_tokens = args
        .body_inline
        .unwrap_or(if args.inline_always {
            InlinePolicy::Always
        } else {
            InlinePolicy::Hint
        })
        .tokens(&input_fn.vis);

    if features_csv.is_empty() {
        mark_emitted_fn_header(&mut input_fn.vis, &mut input_fn.sig);
        let vis = &input_fn.vis;
        let sig = &input_fn.sig;
        let attrs = filter_inline_attrs(&input_fn.attrs);
        let body = &input_fn.body;
        let self_binding = args
            .self_type
            .as_ref()
            .map(|_| quote! { let _self = self; });
        let cfg = gen_cfg_guard(None, args.shared.cfg_feature.as_deref());
        let body = braced_body(&input_fn.brace_token, quote! { #self_binding #body });
        return quote! { #cfg #(#attrs)* #inline_tokens #vis #sig #body };
    }
    let target_feature_tokens = quote!(#[target_feature(enable = #features_csv)]);

    // On wasm32, #[target_feature(enable = "simd128")] functions are safe (Rust 1.54+).
    // The wasm validation model guarantees unsupported instructions trap deterministically,
    // so there's no UB from feature mismatch. Skip the unsafe wrapper entirely.
    if target_arch == Some("wasm32") {
        return arcane_impl_wasm_safe(input_fn, &args, target_feature_tokens, inline_tokens);
    }

    let cfg_guard = drop_attrs_equal_to(
        &mut input_fn.attrs,
        target_arch,
        args.shared.cfg_feature.as_deref(),
    );
    let callee = if args.nested {
        format_ident!("__simd_inner_{}", input_fn.sig.ident)
    } else {
        format_ident!("__arcane_{}", input_fn.sig.ident)
    };
    for sig in [&input_fn.sig, &wrapper_sig] {
        if let Some(err) = reserved_param_error_ident(sig, &callee, macro_name) {
            return err;
        }
    }
    let parts = BoundaryParts {
        callee,
        cfg_guard,
        target_feature_tokens,
        inline_tokens,
        token_assertion: gen_token_assertion(
            token_type_name,
            &token_type,
            args.suppress_const_test,
        ),
        tier_trait_assertion: gen_tier_trait_assertion(&tier_traits, &wrapper_token_ident),
        wrapper_sig,
    };
    emit_boundary(input_fn, &args, parts)
}

/// Reject a parameter or const generic named like the generated callee, so
/// nothing in the value namespace can shadow the wrapper's `unsafe` call.
pub(crate) fn reserved_param_error_ident(
    sig: &syn::Signature,
    callee: &Ident,
    macro_name: &str,
) -> Option<TokenStream> {
    with_ident_str(callee, |raw_callee| {
        let target = raw_callee.strip_prefix("r#").unwrap_or(raw_callee);
        for param in &sig.generics.params {
            if let syn::GenericParam::Const(c) = param
                && with_ident_str(&c.ident, |s| s.strip_prefix("r#").unwrap_or(s) == target)
            {
                return Some(
                    syn::Error::new_spanned(
                        &c.ident,
                        format!(
                            "const generic `{target}` shadows the function #[{macro_name}] generates for `{}`; rename the parameter",
                            sig.ident
                        ),
                    )
                    .to_compile_error(),
                );
            }
        }
        sig.inputs.iter().find_map(|arg| match arg {
            FnArg::Typed(pat_type) => match pat_type.pat.as_ref() {
                syn::Pat::Ident(pat)
                    if with_ident_str(&pat.ident, |s| {
                        s.strip_prefix("r#").unwrap_or(s) == target
                    }) =>
                {
                    Some(
                        syn::Error::new_spanned(
                            &pat.ident,
                            format!(
                                "parameter `{target}` shadows the function #[{macro_name}] generates for \
                                 `{}` and would receive its `unsafe` call; rename the parameter",
                                sig.ident
                            ),
                        )
                        .to_compile_error(),
                    )
                }
                _ => None,
            },
            FnArg::Receiver(_) => None,
        })
    })
}

/// The pieces every boundary expansion emits: the cfg guard on both halves,
/// the attributes of the feature-enabled half, and the two assertions that
/// authenticate the token before the one `unsafe` call.
struct BoundaryParts {
    callee: Ident,
    cfg_guard: TokenStream,
    target_feature_tokens: TokenStream,
    inline_tokens: TokenStream,
    token_assertion: TokenStream,
    tier_trait_assertion: TokenStream,
    wrapper_sig: syn::Signature,
}

/// WASM-safe expansion: emits rite-style output (no unsafe wrapper).
///
/// On wasm32, `#[target_feature(enable = "simd128")]` is safe — the wasm validation
/// model traps deterministically on unsupported instructions, so there's no UB.
/// We emit the function directly with `#[target_feature]` + `#[inline]`, like `#[rite]`.
///
/// If `_self = Type` is set, we inject `let _self = self;` at the top of the body
/// (the function stays in impl scope, so `Self` resolves naturally — no replacement needed).
pub(crate) fn arcane_impl_wasm_safe(
    mut input_fn: LightFn,
    args: &BoundaryOptions,
    target_feature_tokens: TokenStream,
    inline_tokens: TokenStream,
) -> TokenStream {
    let cfg_guard = drop_attrs_equal_to(
        &mut input_fn.attrs,
        Some("wasm32"),
        args.shared.cfg_feature.as_deref(),
    );
    mark_emitted_fn_header(&mut input_fn.vis, &mut input_fn.sig);
    let vis = &input_fn.vis;
    let sig = &input_fn.sig;
    let filtered_attrs = filter_inline_attrs(&input_fn.attrs);
    let wrapper_attrs = &args.wrapper_attrs;

    let body = if args.self_type.is_some() {
        let original_body = &input_fn.body;
        quote! {
            let _self = self;
            #original_body
        }
    } else {
        input_fn.body
    };

    let body = braced_body(&input_fn.brace_token, body);
    quote! {
        #cfg_guard
        #target_feature_tokens
        #inline_tokens
        #wrapper_attrs
        #(#filtered_attrs)*
        #vis #sig #body
    }
}

/// One boundary emitter for sibling and nested placement. Placement changes
/// name resolution and the receiver representation, never the proof obligation.
fn emit_boundary(input: LightFn, options: &BoundaryOptions, parts: BoundaryParts) -> TokenStream {
    let BoundaryParts {
        callee: name,
        cfg_guard,
        target_feature_tokens,
        inline_tokens,
        token_assertion,
        tier_trait_assertion,
        wrapper_sig,
    } = parts;
    let sig = &input.sig;
    let generics = &sig.generics;
    let vis = &input.vis;
    let mut unsafety = sig.safety.clone();
    if let syn::Safety::Unsafe(tok) = &mut unsafety {
        tok.span = proc_macro2::Span::call_site();
    }
    let attrs = if options.nested {
        filter_inline_attrs(&input.attrs)
            .into_iter()
            .cloned()
            .collect()
    } else {
        expect_as_allow(filter_inline_attrs(&input.attrs))
    };
    let body_attrs = body_attrs(&input.attrs, !options.nested);
    let turbofish = build_turbofish(generics);
    let has_self = sig
        .inputs
        .first()
        .is_some_and(|arg| matches!(arg, FnArg::Receiver(_)));
    let self_ident = Ident::new("self", Span::call_site());
    let forwarded = wrapper_sig.inputs.iter().filter_map(|arg| match arg {
        FnArg::Receiver(_) => options.nested.then_some(&self_ident),
        FnArg::Typed(param) => match param.pat.as_ref() {
            syn::Pat::Ident(pat) => Some(&pat.ident),
            _ => None,
        },
    });
    let params = if options.nested {
        let params = sig.inputs.iter().map(|arg| match arg {
            FnArg::Receiver(receiver) => {
                let ty = options
                    .self_type
                    .as_ref()
                    .expect("nested receiver validated before emission");
                nested_self_param(receiver, ty).to_token_stream()
            }
            FnArg::Typed(param) => {
                let ts = param.to_token_stream();
                match &options.self_type {
                    Some(ty) if tokens_contain_ident(&ts, &["Self"]) => {
                        replace_self_in_tokens(ts, ty)
                    }
                    _ => ts,
                }
            }
        });
        quote!(#(#params),*)
    } else {
        sig.inputs.to_token_stream()
    };
    let mut output = sig.output.to_token_stream();
    let mut where_clause = generics.where_clause.to_token_stream();
    let mut body = input.body;
    if options.nested
        && let Some(ty) = &options.self_type
    {
        if !output.is_empty() && tokens_contain_ident(&output, &["Self"]) {
            output = replace_self_in_tokens(output, ty);
        }
        if !where_clause.is_empty() && tokens_contain_ident(&where_clause, &["Self"]) {
            where_clause = replace_self_in_tokens(where_clause, ty);
        }
        body = replace_self_value_in_tokens(
            replace_self_in_tokens(body, ty),
            &Ident::new("_self", Span::call_site()),
        );
    }
    let call = if options.nested {
        quote!(#name #turbofish(#(#forwarded),*))
    } else if has_self {
        quote!(self.#name #turbofish(#(#forwarded),*))
    } else if options.in_impl {
        quote!(Self::#name #turbofish(#(#forwarded),*))
    } else {
        quote!(#name #turbofish(#(#forwarded),*))
    };
    // Keep legacy attribute order, including lint expectations. The body is
    // emitted once; only its lexical placement differs between these policies.
    let before = if options.nested {
        quote!()
    } else {
        quote!(#[doc(hidden)] #(#body_attrs)*)
    };
    let after = if options.nested {
        quote!(#(#body_attrs)*)
    } else {
        quote!()
    };
    let body = braced_body(&input.brace_token, body);
    let inner = quote! {
        #before #target_feature_tokens #inline_tokens #after
        #unsafety fn #name #generics (#params) #output #where_clause #body
    };
    let proof_call = proof_call(call);
    let sibling = (!options.nested).then_some(&inner);
    let sibling_cfg = (!options.nested).then_some(&cfg_guard);
    let nested = options.nested.then_some(&inner);
    // The wrapper's parameters carry reserved names for any `_name` the author
    // wrote (`rename_underscore_params`), so forwarding them trips no lint; in
    // nested mode the author's body (the inner fn) keeps its own names. Clippy
    // does not report `inline_always` on this macro-spanned wrapper, so it
    // carries no lint levels of its own (measured 2026-10-24).
    let wrapper_attrs = &options.wrapper_attrs;
    quote! {
        #sibling_cfg #sibling
        #cfg_guard #(#attrs)* #wrapper_attrs #[inline(always)]
        #vis #wrapper_sig {
            #nested #token_assertion #tier_trait_assertion #proof_call
        }
    }
}

/// The only feature-boundary call emitted for native proof entries. Callers
/// must authenticate the supplied proof and resolve a body generated in the
/// same scope before reaching this function.
pub(crate) fn proof_call(call: TokenStream) -> TokenStream {
    // SAFETY: the enclosing proof entry establishes the callee's feature set.
    quote!(unsafe { #call })
}

#[cfg(test)]
mod assertion_tests {
    use super::*;
    use crate::arcane::ArcaneArgs;
    use syn::parse_quote;

    #[test]
    fn concrete_checks_are_a_single_shared_lookup() {
        let ty = Some(parse_quote!(renamed::X64V3Token));
        let check = gen_token_assertion(Some("X64V3Token"), &ty, false);
        assert_eq!(
            check.to_string(),
            quote!(let _: () = <renamed::X64V3Token>::__ARCHMAGE_ASSERT_TIER_F38B284B;).to_string()
        );
        assert!(gen_token_assertion(None, &None, false).is_empty());
        assert!(gen_token_assertion(Some("X64V3Token"), &ty, true).is_empty());
    }

    #[test]
    fn no_shared_opt_in() {
        assert!(syn::parse_str::<ArcaneArgs>("shared").is_err());
    }

    #[test]
    fn in_impl_and_nested_are_exclusive() {
        assert!(syn::parse_str::<ArcaneArgs>("in_impl, nested").is_err());
        assert!(syn::parse_str::<ArcaneArgs>("in_impl").unwrap().in_impl);
        assert!(syn::parse_str::<ArcaneArgs>("in_trait").unwrap().nested);
    }
}
