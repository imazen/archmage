//! Feature-proof boundaries shared by all attribute frontends.
//! Parsing syntax is kept out of this module; the proof is authenticated before
//! emitting the single unsafe call. Body tokens are never parsed as expressions.
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

/// Represents the kind of self receiver and the transformed parameter.
pub(crate) enum SelfReceiver {
    /// `self` (by value/move)
    Owned,
    /// `&self` (shared reference)
    Ref,
    /// `&mut self` (mutable reference)
    RefMut,
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

    // Rewrite incant!() calls in the body to direct tier calls.
    // Only for concrete tokens where we can determine the tier suffix.
    if let Some(ref type_name) = token_type_name
        && let Some(tier_suffix) = crate::generated::canonical_token_to_tier_suffix(type_name)
        && let Some(tier) = crate::tiers::find_tier(tier_suffix)
    {
        let ctx = crate::rewrite::CallerContext {
            tier_suffix: tier_suffix.to_string(),
            target_arch: tier.target_arch,
            token_ident: _token_ident.clone(),
            has_token: true,
            derive_token: false,
        };
        input_fn.body = crate::rewrite::rewrite_incant_in_body(input_fn.body, &ctx);
    }

    if let Some(tier) = crate::tiers::ALL_TIERS
        .iter()
        .filter(|tier| tier.name != "default")
        .find(|tier| {
            crate::generated::tier_to_canonical_token(tier.name)
                .and_then(crate::generated::token_to_features)
                .is_some_and(|candidate| {
                    candidate.len() == features.len()
                        && candidate.iter().all(|feature| features.contains(feature))
                })
        })
    {
        input_fn.body = crate::attune::call::rewrite(input_fn.body, tier);
    }

    // Build a single target_feature attribute with all features comma-joined
    let features_csv = crate::token_discovery::features_csv(token_type_name.as_deref(), &features);
    let inline_attr: Attribute = if args.inline_always {
        parse_quote!(#[inline(always)])
    } else {
        parse_quote!(#[inline])
    };

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

    // Rename non-ident patterns to named params so the wrapper → sibling call works.
    // The original patterns are re-bound at the top of the inner body.
    let rebinds = rename_non_ident_params(&mut input_fn.sig);
    prepend_to_body(&mut input_fn.body, quote! { #(#rebinds)* });
    // Renaming may have changed the token parameter's ident (a wildcard
    // `_: X64V3Token` becomes `__archmage_arg_0: X64V3Token` and binds
    // nothing, so it leaves no rebind behind). Re-discover it.
    if let Some(info) = find_token_param(&input_fn.sig) {
        _token_ident = info.ident;
    }

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
    if args.nested {
        arcane_impl_nested(input_fn, &args, parts)
    } else {
        arcane_impl_sibling(input_fn, &args, parts)
    }
}

/// The pieces every boundary expansion emits: the cfg guard on both halves,
/// the attributes of the feature-enabled half, and the two assertions that
/// authenticate the token before the one `unsafe` call.
struct BoundaryParts {
    cfg_guard: TokenStream,
    target_feature_attrs: Vec<Attribute>,
    inline_attr: Attribute,
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
    inline_attr: Attribute,
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
    new_attrs.push(inline_attr);
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

/// Sibling expansion (default): generates two functions at the same scope level.
///
/// The sibling function is safe (Rust 2024 edition allows safe `#[target_feature]`
/// functions). Only the call from the wrapper needs `unsafe` because the wrapper
/// lacks matching target features. Compatible with `#![forbid(unsafe_code)]`.
///
/// Self/self work naturally since both functions live in the same impl scope.
fn arcane_impl_sibling(
    input_fn: LightFn,
    args: &BoundaryOptions,
    parts: BoundaryParts,
) -> TokenStream {
    let vis = &input_fn.vis;
    let sig = &input_fn.sig;
    let fn_name = &sig.ident;
    let generics = &sig.generics;
    let where_clause = &generics.where_clause;
    let inputs = &sig.inputs;
    let output = &sig.output;
    let body = &input_fn.body;
    // Filter out user #[inline] attrs to avoid duplicates (will become a hard error).
    // The wrapper gets #[inline(always)] unconditionally — it's a trivial unsafe { sibling() }.
    let attrs = filter_inline_attrs(&input_fn.attrs);
    // Lint-control attrs (#[allow(...)], #[expect(...)], etc.) must also go on the sibling,
    // because the sibling has the same parameters and clippy lints it independently.
    let lint_attrs = filter_lint_attrs(&input_fn.attrs);
    let BoundaryParts {
        cfg_guard,
        target_feature_attrs,
        inline_attr,
        token_assertion,
        tier_trait_assertion,
    } = parts;

    let sibling_name = format_ident!("__arcane_{}", fn_name);

    // Detect self receiver
    let has_self_receiver = inputs
        .first()
        .map(|arg| matches!(arg, FnArg::Receiver(_)))
        .unwrap_or(false);

    // Build turbofish for forwarding type/const generic params to sibling
    let turbofish = build_turbofish(generics);

    // Every parameter is an identifier by now (see rename_non_ident_params).
    let forwarded_args: Vec<proc_macro2::TokenStream> = inputs
        .iter()
        .filter_map(|arg| match arg {
            FnArg::Typed(pat_type) => match pat_type.pat.as_ref() {
                syn::Pat::Ident(pat_ident) => {
                    let ident = &pat_ident.ident;
                    Some(quote!(#ident))
                }
                _ => None,
            },
            FnArg::Receiver(_) => None,
        })
        .collect();

    // Build the call from wrapper to sibling. A method calls through `self`;
    // an associated function in an impl needs `Self::`, which the macro cannot
    // infer, so `in_impl` says so; a free function calls the sibling by name.
    let sibling_call = if has_self_receiver {
        quote! { self.#sibling_name #turbofish(#(#forwarded_args),*) }
    } else if args.in_impl {
        quote! { Self::#sibling_name #turbofish(#(#forwarded_args),*) }
    } else {
        quote! { #sibling_name #turbofish(#(#forwarded_args),*) }
    };

    // Sibling function: #[doc(hidden)] #[target_feature] fn __arcane_fn(...)
    // Always private — only the wrapper is user-visible.
    // Safe declaration — Rust 2024 allows safe #[target_feature] functions.
    quote! {
        #cfg_guard
        #[doc(hidden)]
        #(#lint_attrs)*
        #(#target_feature_attrs)*
        #inline_attr
        fn #sibling_name #generics (#inputs) #output #where_clause {
            #body
        }

        #cfg_guard
        #(#attrs)*
        #[inline(always)]
        #vis #sig {
            #token_assertion
            #tier_trait_assertion
            // SAFETY: The token parameter proves the required CPU features are available.
            // Calling a #[target_feature] function from a non-matching context requires
            // unsafe because the CPU may not support those instructions. The token's
            // existence proves summon() succeeded, so the features are available.
            unsafe { #sibling_call }
        }
    }
}

/// Nested inner function expansion (opt-in via `nested`, `in_trait` or `_self = Type`).
///
/// Generates a nested inner function inside the original function. Required in
/// trait impls, which cannot take a sibling item, and when `_self = Type` is
/// used because `Self` must be replaced in the nested function (where it's not
/// in scope).
fn arcane_impl_nested(
    input_fn: LightFn,
    args: &BoundaryOptions,
    parts: BoundaryParts,
) -> TokenStream {
    let vis = &input_fn.vis;
    let sig = &input_fn.sig;
    let fn_name = &sig.ident;
    let generics = &sig.generics;
    let where_clause = &generics.where_clause;
    let inputs = &sig.inputs;
    let output = &sig.output;
    let body = &input_fn.body;
    // Filter out user #[inline] attrs to avoid duplicates (will become a hard error).
    let attrs = filter_inline_attrs(&input_fn.attrs);
    // Propagate lint attrs to inner function (same issue as sibling mode — #17)
    let lint_attrs = filter_lint_attrs(&input_fn.attrs);
    let BoundaryParts {
        cfg_guard,
        target_feature_attrs,
        inline_attr,
        token_assertion,
        tier_trait_assertion,
    } = parts;

    // Determine self receiver type if present
    let self_receiver_kind: Option<SelfReceiver> = inputs.first().and_then(|arg| match arg {
        FnArg::Receiver(receiver) => {
            // syn 3 moved the by-reference shape into `Receiver::kind`. Owned
            // (`self`/`mut self`) and typed (`self: Box<Self>`) receivers both
            // map to `Owned`.
            match &receiver.kind {
                syn::ReceiverKind::Reference(_, _, mutability) => {
                    if mutability.is_some() {
                        Some(SelfReceiver::RefMut)
                    } else {
                        Some(SelfReceiver::Ref)
                    }
                }
                _ => Some(SelfReceiver::Owned),
            }
        }
        _ => None,
    });

    // Build inner function parameters, transforming self if needed.
    // Also replace Self in non-self parameter types when _self = Type is set,
    // since the inner function is a nested fn where Self from the impl is not in scope.
    let inner_params: Vec<proc_macro2::TokenStream> = inputs
        .iter()
        .map(|arg| match arg {
            FnArg::Receiver(_) => {
                // Transform self receiver to _self parameter
                let self_ty = args.self_type.as_ref().unwrap();
                match self_receiver_kind.as_ref().unwrap() {
                    SelfReceiver::Owned => quote!(_self: #self_ty),
                    SelfReceiver::Ref => quote!(_self: &#self_ty),
                    SelfReceiver::RefMut => quote!(_self: &mut #self_ty),
                }
            }
            FnArg::Typed(pat_type) => {
                if let Some(ref self_ty) = args.self_type {
                    replace_self_in_tokens(quote!(#pat_type), self_ty)
                } else {
                    quote!(#pat_type)
                }
            }
        })
        .collect();

    // Build inner function call arguments
    let inner_args: Vec<proc_macro2::TokenStream> = inputs
        .iter()
        .filter_map(|arg| match arg {
            FnArg::Typed(pat_type) => {
                if let syn::Pat::Ident(pat_ident) = pat_type.pat.as_ref() {
                    let ident = &pat_ident.ident;
                    Some(quote!(#ident))
                } else {
                    None
                }
            }
            FnArg::Receiver(_) => Some(quote!(self)), // Pass self to inner as _self
        })
        .collect();

    let inner_fn_name = format_ident!("__simd_inner_{}", fn_name);

    // Build turbofish for forwarding type/const generic params to inner function
    let turbofish = build_turbofish(generics);

    // The inner function cannot see the impl's `Self` or its `self` value.
    // With `_self = Type`, replace the type `Self` by `Type` in the output,
    // where clause and body, and rename the value `self` to `_self`, which
    // the inner function takes as a parameter. `self::` paths are left alone.
    let (inner_output, inner_body, inner_where_clause): (
        proc_macro2::TokenStream,
        proc_macro2::TokenStream,
        proc_macro2::TokenStream,
    ) = if let Some(ref self_ty) = args.self_type {
        let self_ident = format_ident!("_self");
        let transformed_output = replace_self_in_tokens(output.to_token_stream(), self_ty);
        let transformed_body = replace_self_value_in_tokens(
            replace_self_in_tokens(body.clone(), self_ty),
            &self_ident,
        );
        let transformed_where = where_clause
            .as_ref()
            .map(|wc| replace_self_in_tokens(wc.to_token_stream(), self_ty))
            .unwrap_or_default();
        (transformed_output, transformed_body, transformed_where)
    } else {
        (
            output.to_token_stream(),
            body.clone(),
            where_clause
                .as_ref()
                .map(|wc| wc.to_token_stream())
                .unwrap_or_default(),
        )
    };

    quote! {
        #cfg_guard
        #(#attrs)*
        #[inline(always)]
        #vis #sig {
            #(#target_feature_attrs)*
            #inline_attr
            #(#lint_attrs)*
            fn #inner_fn_name #generics (#(#inner_params),*) #inner_output #inner_where_clause {
                #inner_body
            }
            #token_assertion
            #tier_trait_assertion
            // SAFETY: The token parameter proves the required CPU features are available.
            unsafe { #inner_fn_name #turbofish(#(#inner_args),*) }
        }
    }
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
