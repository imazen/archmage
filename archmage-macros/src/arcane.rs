//! `#[arcane]` — generates safe `#[target_feature]` wrappers.
//!
//! Sibling mode (default), nested mode, and WASM-safe mode.

use proc_macro2::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::{
    Attribute, FnArg, Ident, Token, Type,
    parse::{Parse, ParseStream},
    parse_quote,
};

use crate::common::*;
use crate::token_discovery::*;

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
pub(crate) struct ArcaneArgs {
    /// Options spelled the same way in `#[rite]`: imports and `cfg(feature)`.
    pub(crate) shared: SharedOptions,
    /// Trusted generators may omit the accidental token-name mismatch check.
    /// Intrinsic feature checking remains enabled.
    suppress_const_test: bool,
    /// Use `#[inline(always)]` instead of `#[inline]` for the inner function.
    /// Requires nightly Rust with `#![feature(target_feature_inline_always)]`.
    inline_always: bool,
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

impl Parse for ArcaneArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut args = ArcaneArgs::default();

        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            if !parse_shared_option(&ident, input, &mut args.shared)? {
                match ident.to_string().as_str() {
                    "suppress_const_test" => args.suppress_const_test = true,
                    "inline_always" => args.inline_always = true,
                    "nested" | "in_trait" => args.nested = true,
                    "in_impl" => args.in_impl = true,
                    "_self" => {
                        let _: Token![=] = input.parse()?;
                        args.self_type = Some(input.parse()?);
                    }
                    other => {
                        return Err(syn::Error::new(
                            ident.span(),
                            format!(
                                "unknown arcane argument: `{other}`. Supported: `in_impl`, \
                                 `in_trait` (or `nested`), `_self = Type`, `import_intrinsics`, \
                                 `import_magetypes`, `cfg(feature)`, `suppress_const_test`."
                            ),
                        ));
                    }
                }
            }
            // Consume optional comma
            if input.peek(Token![,]) {
                let _: Token![,] = input.parse()?;
            }
        }

        // _self = Type implies nested (inner fn needed for Self replacement)
        if args.self_type.is_some() {
            args.nested = true;
        }
        if args.in_impl && args.nested {
            return Err(syn::Error::new(
                input.span(),
                "`in_impl` is for a receiver-less function in an inherent impl, where the \
                 sibling expansion applies; `in_trait`/`nested`/`_self` already avoid the \
                 sibling. Use one of them.",
            ));
        }

        Ok(args)
    }
}

/// Shared implementation for arcane/arcane macros.
pub(crate) fn arcane_impl(
    mut input_fn: LightFn,
    macro_name: &str,
    args: ArcaneArgs,
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

    // The wrapper forwards every argument by name, so its signature names
    // every non-identifier pattern (`_`, `(a, b)`) as `__archmage_arg_N`. The
    // feature-enabled function keeps the user's patterns; only a wildcard
    // token gets a name there, because the nested-dispatch rewrite below
    // threads the token by name. The wrapper's token name is what the tier
    // assertions use.
    let wrapper_sig = {
        let mut sig = input_fn.sig.clone();
        let _ = rename_non_ident_params(&mut sig);
        sig
    };
    let wrapper_token_ident = find_token_param(&wrapper_sig)
        .map(|info| info.ident)
        .unwrap_or_else(|| _token_ident.clone());
    if let Some(ident) = rename_wildcard_token(&mut input_fn.sig) {
        _token_ident = ident;
    }

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

    // On wasm32, #[target_feature(enable = "simd128")] functions are safe (Rust 1.54+).
    // The wasm validation model guarantees unsupported instructions trap deterministically,
    // so there's no UB from feature mismatch. Skip the unsafe wrapper entirely.
    if target_arch == Some("wasm32") {
        return arcane_impl_wasm_safe(input_fn, &args, target_feature_attrs, inline_attr);
    }

    let cfg_guard = gen_cfg_guard(target_arch, args.shared.cfg_feature.as_deref());
    drop_attrs_equal_to(&mut input_fn.attrs, &cfg_guard);
    let parts = BoundaryParts {
        cfg_guard,
        target_feature_attrs,
        inline_attr,
        token_assertion: gen_token_assertion(
            &token_type_name,
            &token_type,
            args.suppress_const_test,
        ),
        tier_trait_assertion: gen_tier_trait_assertion(&tier_traits, &wrapper_token_ident),
        wrapper_sig,
    };
    // The wrapper's one `unsafe` call names the generated function, and the
    // only bindings in scope at that call are the parameters: a parameter
    // with that name would shadow the item and receive the call (a `Deref`
    // to an `unsafe fn` would then run arbitrary code under the token's
    // proof). Module-level collisions are rustc errors; this one is ours.
    let callee = if args.nested {
        format!("__simd_inner_{}", input_fn.sig.ident)
    } else {
        format!("__arcane_{}", input_fn.sig.ident)
    };
    if let Some(err) = reserved_param_error(&input_fn.sig, &callee) {
        return err;
    }
    if args.nested {
        arcane_impl_nested(input_fn, &args, parts)
    } else {
        arcane_impl_sibling(input_fn, &args, parts)
    }
}

/// Reject a parameter named like the generated callee, so nothing a caller
/// writes can stand between the wrapper's `unsafe` call and that function.
fn reserved_param_error(sig: &syn::Signature, callee: &str) -> Option<TokenStream> {
    sig.inputs.iter().find_map(|arg| match arg {
        FnArg::Typed(pat_type) => match pat_type.pat.as_ref() {
            syn::Pat::Ident(pat) if pat.ident == callee => Some(
                syn::Error::new_spanned(
                    &pat.ident,
                    format!(
                        "parameter `{callee}` shadows the function #[arcane] generates for \
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
    inline_attr: Attribute,
    token_assertion: TokenStream,
    tier_trait_assertion: TokenStream,
    /// The user's signature with every non-identifier pattern named, for the
    /// wrapper, which forwards its arguments by name.
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
    args: &ArcaneArgs,
    target_feature_attrs: Vec<Attribute>,
    inline_attr: Attribute,
) -> TokenStream {
    drop_attrs_equal_to(
        &mut input_fn.attrs,
        &gen_cfg_guard(Some("wasm32"), args.shared.cfg_feature.as_deref()),
    );
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
/// The sibling function is safe when the input is (Rust 2024 edition allows
/// safe `#[target_feature]` functions). Only the call from the wrapper needs
/// `unsafe` because the wrapper lacks matching target features. Compatible
/// with `#![forbid(unsafe_code)]`. An `unsafe fn` input keeps `unsafe` on
/// both halves: the sibling is callable without `unsafe` from a matching
/// feature context, so a safe sibling would discard its preconditions.
///
/// Self/self work naturally since both functions live in the same impl scope.
fn arcane_impl_sibling(input_fn: LightFn, args: &ArcaneArgs, parts: BoundaryParts) -> TokenStream {
    let vis = &input_fn.vis;
    let sig = &input_fn.sig;
    let fn_name = &sig.ident;
    let generics = &sig.generics;
    let where_clause = &generics.where_clause;
    let inputs = &sig.inputs;
    let output = &sig.output;
    let body = &input_fn.body;
    // The sibling keeps the input's `unsafe`: it is callable without `unsafe`
    // from any matching feature context, so a safe sibling would discard the
    // preconditions the user declared.
    let unsafety = &sig.safety;
    // The wrapper gets #[inline(always)] unconditionally — it's a trivial
    // unsafe { sibling() } — and every other attribute; the sibling gets the
    // ones that belong with the body (`#[expect]` only there).
    let attrs = expect_as_allow(filter_inline_attrs(&input_fn.attrs));
    let sibling_attrs = body_attrs(&input_fn.attrs, true);
    let BoundaryParts {
        cfg_guard,
        target_feature_attrs,
        inline_attr,
        token_assertion,
        tier_trait_assertion,
        wrapper_sig,
    } = parts;

    let sibling_name = format_ident!("__arcane_{}", fn_name);

    // Detect self receiver
    let has_self_receiver = inputs
        .first()
        .map(|arg| matches!(arg, FnArg::Receiver(_)))
        .unwrap_or(false);

    // Build turbofish for forwarding type/const generic params to sibling
    let turbofish = build_turbofish(generics);

    // The wrapper's parameters are all identifiers (see rename_non_ident_params).
    let forwarded_args: Vec<proc_macro2::TokenStream> = wrapper_sig
        .inputs
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
    // Declared safe when the input is — Rust 2024 allows safe #[target_feature]
    // functions — and `unsafe` when the input is.
    quote! {
        #cfg_guard
        #[doc(hidden)]
        #(#sibling_attrs)*
        #(#target_feature_attrs)*
        #inline_attr
        #unsafety fn #sibling_name #generics (#inputs) #output #where_clause {
            #body
        }

        #cfg_guard
        #(#attrs)*
        #[inline(always)]
        #vis #wrapper_sig {
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
fn arcane_impl_nested(input_fn: LightFn, args: &ArcaneArgs, parts: BoundaryParts) -> TokenStream {
    let vis = &input_fn.vis;
    let sig = &input_fn.sig;
    let fn_name = &sig.ident;
    let generics = &sig.generics;
    let where_clause = &generics.where_clause;
    let inputs = &sig.inputs;
    let output = &sig.output;
    let body = &input_fn.body;
    // The inner fn keeps the input's `unsafe` (see the sibling expansion).
    let unsafety = &sig.safety;
    // The wrapper's lint levels cover the inner fn nested in its body, so the
    // wrapper keeps `#[expect]` (fulfilled by the body inside it) and the
    // inner fn gets no lint levels of its own (a second `#[expect]` would
    // leave the wrapper's unfulfilled); `#[track_caller]` goes on both.
    let attrs = filter_inline_attrs(&input_fn.attrs);
    let inner_attrs = body_attrs(&input_fn.attrs, false);
    let BoundaryParts {
        cfg_guard,
        target_feature_attrs,
        inline_attr,
        token_assertion,
        tier_trait_assertion,
        wrapper_sig,
    } = parts;

    // Build inner function parameters, transforming self if needed.
    // Also replace Self in non-self parameter types when _self = Type is set,
    // since the inner function is a nested fn where Self from the impl is not in scope.
    let inner_params: Vec<proc_macro2::TokenStream> = inputs
        .iter()
        .map(|arg| match arg {
            FnArg::Receiver(receiver) => {
                // The receiver becomes `_self`, keeping its reference, lifetime
                // and mutability, or its explicit type (`self: Box<Self>`).
                let self_ty = args.self_type.as_ref().unwrap();
                nested_self_param(receiver, self_ty).to_token_stream()
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

    // Build inner function call arguments from the outer (wrapper) signature,
    // whose parameters are all identifiers.
    let inner_args: Vec<proc_macro2::TokenStream> = wrapper_sig
        .inputs
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
        #vis #wrapper_sig {
            #(#target_feature_attrs)*
            #inline_attr
            #(#inner_attrs)*
            #unsafety fn #inner_fn_name #generics (#(#inner_params),*) #inner_output #inner_where_clause {
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
