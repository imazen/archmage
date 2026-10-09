//! Feature-proof boundaries shared by all attribute frontends.
//! Parsing syntax is kept out of this module; the proof is authenticated before
//! emitting the single unsafe call. Body tokens are never parsed as expressions.
use super::inline::InlinePolicy;
use crate::common::*;
use crate::token_discovery::*;
use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::{Attribute, FnArg, Type, parse_quote};

/// Validate a concrete token using the tier-specific constant supplied by archmage.
///
/// The signature's full type path preserves renamed dependencies and re-exports.
/// A weaker tier lacks the expected constant, so accidental aliasing is rejected.
/// The initializer is checked once in archmage; each expansion only references it.
/// This public constant is an accidental-misuse check, not an anti-forgery boundary.
/// The exact archmage-macros dependency pin keeps this internal protocol in sync.
fn gen_token_assertion(
    token_type_name: &Option<String>,
    token_type: &Option<Type>,
    suppress_const_test: bool,
) -> proc_macro2::TokenStream {
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
}

/// Shared implementation for arcane/arcane macros.
pub(crate) fn expand(
    mut input_fn: LightFn,
    macro_name: &str,
    args: BoundaryOptions,
) -> TokenStream {
    // Check for self receiver
    let has_self_receiver = input_fn
        .sig
        .inputs
        .first()
        .map(|arg| matches!(arg, FnArg::Receiver(_)))
        .unwrap_or(false);

    // Nested mode is required when _self = Type is used (for Self replacement in nested fn).
    // In sibling mode, self/Self work naturally since both fns live in the same impl scope.
    // However, if there's a self receiver in nested mode, we still need _self = Type.
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

    // Find the token parameter, its features, target arch, and token type name
    let TokenParamInfo {
        index: token_index,
        ident: mut _token_ident,
        features,
        target_arch,
        token_type_name,
        magetypes_namespace,
        token_type,
        tier_traits,
    } = match find_token_param(&input_fn.sig).or_else(|| {
        // A concrete token receiver can itself prove the required features.
        // Reuse normal discovery so token validation and feature lookup agree.
        let self_ty = args.self_type.as_ref()?;
        if !has_self_receiver {
            return None;
        }
        let mut receiver_sig = input_fn.sig.clone();
        receiver_sig.inputs.clear();
        receiver_sig.inputs.push(syn::parse_quote!(_self: #self_ty));
        find_token_param(&receiver_sig)
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
    let token_params = token_param_idents(&input_fn.sig);
    if token_params.len() > 1 {
        return multiple_tokens_error(&input_fn.sig, macro_name, &token_params);
    }

    // import_intrinsics with AVX-512 features needs archmage's avx512 feature
    // (propagated to archmage-macros) for the 512-bit safe memory wrappers.
    if let Some(err) = avx512_import_error(
        &input_fn.sig,
        args.shared.import_intrinsics,
        &features,
        token_type_name.as_deref().unwrap_or("an AVX-512 token"),
    ) {
        return err;
    }

    // Prepend import statements to body if requested
    prepend_to_body(
        &mut input_fn.body,
        generate_imports(
            target_arch,
            magetypes_namespace,
            args.shared.import_intrinsics,
            args.shared.import_magetypes,
        ),
    );

    // Rename non-ident patterns to named params so the wrapper → sibling call works.
    // The original patterns are re-bound at the top of the inner body.
    let rebinds = rename_non_ident_params(&mut input_fn.sig);
    prepend_to_body(&mut input_fn.body, quote! { #(#rebinds)* });
    // Pattern normalization preserves positions. Reuse discovery instead of
    // resolving the token's type and generic bounds a second time. A receiver
    // proof keeps the synthetic `_self` name used by the nested function.
    if let FnArg::Typed(param) = &input_fn.sig.inputs[token_index]
        && let syn::Pat::Ident(pat) = param.pat.as_ref()
    {
        _token_ident = pat.ident.clone();
    }

    // Rewrite incant!() calls in the body to direct tier calls.
    // Only for concrete tokens where we can determine the tier suffix.
    if let Some(ref type_name) = token_type_name
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
        input_fn.body = crate::rewrite::rewrite_incant_in_body(input_fn.body, &ctx);
    }

    if token_type_name.is_none() {
        input_fn.body = crate::attune::call::rewrite_context(
            input_fn.body,
            crate::attune::call::Context {
                features: &features,
                target_arch,
            },
        );
    }

    // Build a single target_feature attribute with all features comma-joined
    let features_csv = crate::token_discovery::features_csv(token_type_name.as_deref(), &features);
    if args.body_inline == Some(InlinePolicy::Always) && !features.is_empty() {
        return syn::Error::new_spanned(
            &input_fn.sig.ident,
            "inline(always) on target-feature bodies requires nightly; use inline(hint) or inline(never)",
        ).to_compile_error();
    }
    // Native boundary bodies are private even when their proof wrapper is pub.
    // Scalar and wasm lower directly, so their body retains the input visibility.
    let body_vis = if features.is_empty() || target_arch == Some("wasm32") {
        &input_fn.vis
    } else {
        &syn::Visibility::Inherited
    };
    let inline_attr = args
        .body_inline
        .unwrap_or(if args.inline_always {
            InlinePolicy::Always
        } else {
            InlinePolicy::Hint
        })
        .attribute(body_vis);

    // Scalar has no instruction-set boundary. Preserve its signature and body
    // without emitting the invalid #[target_feature(enable = "")].
    if features_csv.is_empty() {
        let vis = &input_fn.vis;
        let sig = &input_fn.sig;
        let attrs = filter_inline_attrs(&input_fn.attrs);
        let body = &input_fn.body;
        let self_binding = args
            .self_type
            .as_ref()
            .map(|_| quote! { let _self = self; });
        let cfg = gen_cfg_guard(None, args.shared.cfg_feature.as_deref());
        return quote! { #cfg #(#attrs)* #inline_attr #vis #sig { #self_binding #body } };
    }
    let target_feature_attrs: Vec<Attribute> =
        vec![parse_quote!(#[target_feature(enable = #features_csv)])];

    // On wasm32, #[target_feature(enable = "simd128")] functions are safe (Rust 1.54+).
    // The wasm validation model guarantees unsupported instructions trap deterministically,
    // so there's no UB from feature mismatch. Skip the unsafe wrapper entirely.
    if target_arch == Some("wasm32") {
        return arcane_impl_wasm_safe(input_fn, &args, target_feature_attrs, inline_attr);
    }

    let parts = BoundaryParts {
        cfg_guard: gen_cfg_guard(target_arch, args.shared.cfg_feature.as_deref()),
        target_feature_attrs,
        inline_attr,
        token_assertion: gen_token_assertion(
            &token_type_name,
            &token_type,
            args.suppress_const_test,
        ),
        tier_trait_assertion: gen_tier_trait_assertion(&tier_traits, &_token_ident),
    };
    let callee = if args.nested {
        format!("__simd_inner_{}", input_fn.sig.ident)
    } else {
        format!("__arcane_{}", input_fn.sig.ident)
    };
    if let Some(err) = reserved_param_error(&input_fn.sig, &callee, macro_name) {
        return err;
    }
    emit_boundary(input_fn, &args, parts)
}

/// Reject a parameter named like the generated callee, so nothing a caller
/// writes can stand between the wrapper's `unsafe` call and that function.
pub(crate) fn reserved_param_error(
    sig: &syn::Signature,
    callee: &str,
    macro_name: &str,
) -> Option<TokenStream> {
    sig.inputs.iter().find_map(|arg| match arg {
        FnArg::Typed(pat_type) => match pat_type.pat.as_ref() {
            syn::Pat::Ident(pat) if pat.ident == callee => Some(
                syn::Error::new_spanned(
                    &pat.ident,
                    format!(
                        "parameter `{callee}` shadows the function #[{macro_name}] generates for \
                         `{}` and would receive its `unsafe` call; rename the parameter",
                        sig.ident
                    ),
                )
                .to_compile_error(),
            ),
            _ => None,
        },
        FnArg::Receiver(_) => None,
    })
}

/// The pieces every boundary expansion emits: the cfg guard on both halves,
/// the attributes of the feature-enabled half, and the two assertions that
/// authenticate the token before the one `unsafe` call.
struct BoundaryParts {
    cfg_guard: TokenStream,
    target_feature_attrs: Vec<Attribute>,
    inline_attr: Option<Attribute>,
    token_assertion: TokenStream,
    tier_trait_assertion: TokenStream,
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
    input_fn: LightFn,
    args: &BoundaryOptions,
    target_feature_attrs: Vec<Attribute>,
    inline_attr: Option<Attribute>,
) -> TokenStream {
    let vis = &input_fn.vis;
    let sig = &input_fn.sig;
    let attrs = &input_fn.attrs;

    // If _self = Type is set, inject `let _self = self;` at top of body so user code
    // referencing `_self` works. The function remains in impl scope, so `Self` resolves
    // naturally — no Self replacement needed (unlike nested mode's inner fn).
    let body = if args.self_type.is_some() {
        let original_body = &input_fn.body;
        quote! {
            let _self = self;
            #original_body
        }
    } else {
        input_fn.body.clone()
    };

    // Prepend target_feature + inline attrs, filtering user #[inline] to avoid duplicates
    let mut new_attrs = target_feature_attrs;
    new_attrs.extend(inline_attr);
    for attr in filter_inline_attrs(attrs) {
        new_attrs.push(attr.clone());
    }

    let cfg_guard = gen_cfg_guard(Some("wasm32"), args.shared.cfg_feature.as_deref());
    quote! {
        #cfg_guard
        #(#new_attrs)*
        #vis #sig {
            #body
        }
    }
}

/// One boundary emitter for sibling and nested placement. Placement changes
/// name resolution and the receiver representation, never the proof obligation.
fn emit_boundary(input: LightFn, options: &BoundaryOptions, parts: BoundaryParts) -> TokenStream {
    let BoundaryParts {
        cfg_guard,
        target_feature_attrs,
        inline_attr,
        token_assertion,
        tier_trait_assertion,
    } = parts;
    let sig = &input.sig;
    let generics = &sig.generics;
    let vis = &input.vis;
    let name = if options.nested {
        format_ident!("__simd_inner_{}", sig.ident)
    } else {
        format_ident!("__arcane_{}", sig.ident)
    };
    let attrs = filter_inline_attrs(&input.attrs);
    let lints = filter_lint_attrs(&input.attrs);
    let turbofish = build_turbofish(generics);
    let has_self = sig
        .inputs
        .iter()
        .any(|arg| matches!(arg, FnArg::Receiver(_)));
    let self_ident = format_ident!("self");
    let mut forwarded = Vec::with_capacity(sig.inputs.len());
    let params = if options.nested {
        let params: Vec<_> = sig
            .inputs
            .iter()
            .map(|arg| match arg {
                FnArg::Receiver(receiver) => {
                    forwarded.push(&self_ident);
                    let ty = options
                        .self_type
                        .as_ref()
                        .expect("nested receiver validated before emission");
                    nested_self_param(receiver, ty).to_token_stream()
                }
                FnArg::Typed(param) => {
                    if let syn::Pat::Ident(pat) = param.pat.as_ref() {
                        let ident = &pat.ident;
                        forwarded.push(ident);
                    }
                    options.self_type.as_ref().map_or_else(
                        || quote!(#param),
                        |ty| replace_self_in_tokens(quote!(#param), ty),
                    )
                }
            })
            .collect();
        quote!(#(#params),*)
    } else {
        for arg in &sig.inputs {
            if let FnArg::Typed(param) = arg
                && let syn::Pat::Ident(pat) = param.pat.as_ref()
            {
                let ident = &pat.ident;
                forwarded.push(ident);
            }
        }
        sig.inputs.to_token_stream()
    };
    let mut output = sig.output.to_token_stream();
    let mut where_clause = generics.where_clause.to_token_stream();
    let mut body = input.body;
    if options.nested
        && let Some(ty) = &options.self_type
    {
        output = replace_self_in_tokens(output, ty);
        where_clause = replace_self_in_tokens(where_clause, ty);
        body =
            replace_self_value_in_tokens(replace_self_in_tokens(body, ty), &format_ident!("_self"));
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
        quote!(#[doc(hidden)] #(#lints)*)
    };
    let after = if options.nested {
        quote!(#(#lints)*)
    } else {
        quote!()
    };
    let inner = quote! {
        #before #(#target_feature_attrs)* #inline_attr #after
        fn #name #generics (#params) #output #where_clause { #body }
    };
    let proof_call = proof_call(call);
    let sibling = (!options.nested).then_some(&inner);
    let sibling_cfg = (!options.nested).then_some(&cfg_guard);
    let nested = options.nested.then_some(&inner);
    quote! {
        #sibling_cfg #sibling
        #cfg_guard #(#attrs)* #[inline(always)] #vis #sig {
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

    #[test]
    fn concrete_checks_are_a_single_shared_lookup() {
        let ty = Some(parse_quote!(renamed::X64V3Token));
        let check = gen_token_assertion(&Some("X64V3Token".into()), &ty, false);
        assert_eq!(
            check.to_string(),
            quote!(let _: () = <renamed::X64V3Token>::__ARCHMAGE_ASSERT_TIER_F38B284B;).to_string()
        );
        // No fresh const initializer, comparison, indexing, or branch per call.
        assert!(gen_token_assertion(&None, &None, false).is_empty());
        assert!(gen_token_assertion(&Some("X64V3Token".into()), &ty, true).is_empty());
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
