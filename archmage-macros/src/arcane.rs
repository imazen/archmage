//! `#[arcane]` — generates safe `#[target_feature]` wrappers.
//!
//! Sibling mode (default), nested mode, and WASM-safe mode.

use proc_macro::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::{
    Attribute, FnArg, Ident, Token, Type,
    parse::{Parse, ParseStream},
    parse_quote,
};

use crate::common::*;
use crate::token_discovery::*;

#[derive(Default)]
pub(crate) struct ArcaneArgs {
    /// Use `#[inline(always)]` instead of `#[inline]` for the inner function.
    /// Requires nightly Rust with `#![feature(target_feature_inline_always)]`.
    inline_always: bool,
    /// The concrete type to use for `self` receiver.
    /// When specified, `self`/`&self`/`&mut self` is transformed to `_self: Type`/`&Type`/`&mut Type`.
    /// Implies `nested = true`.
    pub(crate) self_type: Option<Type>,
    /// Generate an `unreachable!()` stub on the wrong architecture.
    /// Default is false (cfg-out: no function emitted on wrong arch).
    pub(crate) stub: bool,
    /// Use nested inner function instead of sibling function.
    /// Implied by `_self = Type`. Required for associated functions in impl blocks
    /// that have no `self` receiver (the macro can't distinguish them from free functions).
    pub(crate) nested: bool,
    /// Inject `use archmage::intrinsics::{arch}::*;` (includes safe memory ops).
    pub(crate) import_intrinsics: bool,
    /// Inject `use magetypes::simd::{ns}::*;`, `use magetypes::simd::generic::*;`,
    /// and `use magetypes::simd::backends::*;`.
    pub(crate) import_magetypes: bool,
    /// Additional cargo feature gate. When set, the generated `#[cfg(target_arch)]`
    /// becomes `#[cfg(all(target_arch = "...", feature = "..."))]`.
    /// Example: `#[arcane(cfg(avx512))]` → `#[cfg(all(target_arch = "x86_64", feature = "avx512"))]`
    pub(crate) cfg_feature: Option<String>,
}

impl Parse for ArcaneArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut args = ArcaneArgs::default();

        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            match ident.to_string().as_str() {
                "inline_always" => args.inline_always = true,
                "stub" => {
                    return Err(syn::Error::new(
                        ident.span(),
                        "`stub` has been removed. Use `incant!` for cross-arch dispatch \
                         instead — it cfg-gates each architecture automatically.\n\
                         \n\
                         Before: #[arcane(stub)] fn process(token: X64V3Token, ...) { ... }\n\
                         After:  #[arcane] fn process_v3(token: X64V3Token, ...) { ... }\n\
                         \x20       fn dispatch(...) { incant!(process(...)) }",
                    ));
                }
                "nested" => args.nested = true,
                "import_intrinsics" => args.import_intrinsics = true,
                "import_magetypes" => args.import_magetypes = true,
                "cfg" => {
                    let content;
                    syn::parenthesized!(content in input);
                    let feat: Ident = content.parse()?;
                    args.cfg_feature = Some(feat.to_string());
                }
                "_self" => {
                    let _: Token![=] = input.parse()?;
                    args.self_type = Some(input.parse()?);
                }
                other => {
                    return Err(syn::Error::new(
                        ident.span(),
                        format!("unknown arcane argument: `{}`", other),
                    ));
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

        Ok(args)
    }
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
             Example: #[{}(nested, _self = MyType)]\n\
             Use `_self` (not `self`) in the function body to refer to self.\n\
             \n\
             Alternatively, remove `nested` to use sibling expansion (default), \
             which handles self/Self naturally.",
            macro_name, macro_name
        );
        return syn::Error::new_spanned(&input_fn.sig, msg)
            .to_compile_error()
            .into();
    }

    // Find the token parameter, its features, target arch, and token type name
    let TokenParamInfo {
        ident: _token_ident,
        features,
        target_arch,
        token_type_name,
        magetypes_namespace,
    } = match find_token_param(&input_fn.sig) {
        Some(result) => result,
        None => {
            // Check for specific misuse: featureless traits like SimdToken
            if let Some(trait_name) = diagnose_featureless_token(&input_fn.sig) {
                let msg = format!(
                    "`{trait_name}` cannot be used as a token bound in #[{macro_name}] \
                     because it doesn't specify any CPU features.\n\
                     \n\
                     #[{macro_name}] needs concrete features to generate #[target_feature]. \
                     Use a concrete token or a feature trait:\n\
                     \n\
                     Concrete tokens: X64V3Token, Desktop64, NeonToken, Arm64V2Token, ...\n\
                     Feature traits:  impl HasX64V2, impl HasNeon, impl HasArm64V3, ..."
                );
                return syn::Error::new_spanned(&input_fn.sig, msg)
                    .to_compile_error()
                    .into();
            }
            let msg = format!(
                "{} requires a token parameter. Supported forms:\n\
                 - Concrete: `token: X64V3Token`\n\
                 - impl Trait: `token: impl HasX64V2`\n\
                 - Generic: `fn foo<T: HasX64V2>(token: T, ...)`\n\
                 - With self: `#[{}(_self = Type)] fn method(&self, token: impl HasNeon, ...)`",
                macro_name, macro_name
            );
            return syn::Error::new_spanned(&input_fn.sig, msg)
                .to_compile_error()
                .into();
        }
    };

    // Check: import_intrinsics with AVX-512 features requires the avx512 cargo feature
    // on archmage (propagated to archmage-macros). Without it, 512-bit safe memory ops
    // from safe_unaligned_simd are not available, and _mm512_loadu_ps etc. would resolve
    // to the unsafe core::arch versions (taking raw pointers instead of references).
    //
    // We check the resolved features (not the token name) so this works uniformly for
    // concrete tokens (X64V4Token), trait bounds (impl HasX64V4), and generics (T: HasX64V4).
    #[cfg(not(feature = "avx512"))]
    if args.import_intrinsics && features.iter().any(|f| f.starts_with("avx512")) {
        let token_desc = token_type_name.as_deref().unwrap_or("an AVX-512 token");
        let msg = format!(
            "Using {token_desc} with `import_intrinsics` requires the `avx512` feature.\n\
             \n\
             Add to your Cargo.toml:\n\
             \x20 archmage = {{ version = \"...\", features = [\"avx512\"] }}\n\
             \n\
             Without it, 512-bit safe memory ops (_mm512_loadu_ps etc.) are not available.\n\
             If you only need value intrinsics (no memory ops), remove `import_intrinsics`."
        );
        return syn::Error::new_spanned(&input_fn.sig, msg)
            .to_compile_error()
            .into();
    }

    // Prepend import statements to body if requested
    let body_imports = generate_imports(
        target_arch,
        magetypes_namespace,
        args.import_intrinsics,
        args.import_magetypes,
    );
    if !body_imports.is_empty() {
        let original_body = &input_fn.body;
        input_fn.body = quote! {
            #body_imports
            #original_body
        };
    }

    // Build a single target_feature attribute with all features comma-joined
    let features_csv = features.join(",");
    let target_feature_attrs: Vec<Attribute> =
        vec![parse_quote!(#[target_feature(enable = #features_csv)])];

    // Rename wildcard patterns (`_: Type`) to named params so the inner/sibling call works
    let mut wild_rename_counter = 0u32;
    for arg in &mut input_fn.sig.inputs {
        if let FnArg::Typed(pat_type) = arg
            && matches!(pat_type.pat.as_ref(), syn::Pat::Wild(_))
        {
            let ident = format_ident!("__archmage_wild_{}", wild_rename_counter);
            wild_rename_counter += 1;
            *pat_type.pat = syn::Pat::Ident(syn::PatIdent {
                attrs: vec![],
                by_ref: None,
                mutability: None,
                ident,
                subpat: None,
            });
        }
    }

    // Choose inline attribute based on args
    let inline_attr: Attribute = if args.inline_always {
        parse_quote!(#[inline(always)])
    } else {
        parse_quote!(#[inline])
    };

    // On wasm32, #[target_feature(enable = "simd128")] functions are safe (Rust 1.54+).
    // The wasm validation model guarantees unsupported instructions trap deterministically,
    // so there's no UB from feature mismatch. Skip the unsafe wrapper entirely.
    if target_arch == Some("wasm32") {
        return arcane_impl_wasm_safe(
            input_fn,
            &args,
            token_type_name,
            target_feature_attrs,
            inline_attr,
        );
    }

    if args.nested {
        arcane_impl_nested(
            input_fn,
            &args,
            target_arch,
            token_type_name,
            target_feature_attrs,
            inline_attr,
        )
    } else {
        arcane_impl_sibling(
            input_fn,
            &args,
            target_arch,
            token_type_name,
            target_feature_attrs,
            inline_attr,
        )
    }
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
    args: &ArcaneArgs,
    token_type_name: Option<String>,
    target_feature_attrs: Vec<Attribute>,
    inline_attr: Attribute,
) -> TokenStream {
    let vis = &input_fn.vis;
    let sig = &input_fn.sig;
    let fn_name = &sig.ident;
    let attrs = &input_fn.attrs;

    let token_type_str = token_type_name.as_deref().unwrap_or("UnknownToken");

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

    let stub = if args.stub {
        // Build stub args for suppressing unused-variable warnings
        let stub_args: Vec<proc_macro2::TokenStream> = sig
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
                FnArg::Receiver(_) => None,
            })
            .collect();

        quote! {
            #[cfg(not(target_arch = "wasm32"))]
            #vis #sig {
                let _ = (#(#stub_args),*);
                unreachable!(
                    "BUG: {}() was called but requires {} (target_arch = \"wasm32\"). \
                     {}::summon() returns None on this architecture, so this function \
                     is unreachable in safe code. If you used forge_token_dangerously(), \
                     that is the bug.",
                    stringify!(#fn_name),
                    #token_type_str,
                    #token_type_str,
                )
            }
        }
    } else {
        quote! {}
    };

    let expanded = quote! {
        #[cfg(target_arch = "wasm32")]
        #(#new_attrs)*
        #vis #sig {
            #body
        }

        #stub
    };

    expanded.into()
}

/// Sibling expansion (default): generates two functions at the same scope level.
///
/// ```ignore
/// // #[arcane] fn process(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] { body }
/// // expands to:
/// #[cfg(target_arch = "x86_64")]
/// #[doc(hidden)]
/// #[target_feature(enable = "avx2,fma,...")]
/// #[inline]
/// fn __arcane_process(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] { body }
///
/// #[cfg(target_arch = "x86_64")]
/// fn process(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
///     unsafe { __arcane_process(token, data) }
/// }
/// ```
///
/// The sibling function is safe (Rust 2024 edition allows safe `#[target_feature]`
/// functions). Only the call from the wrapper needs `unsafe` because the wrapper
/// lacks matching target features. Compatible with `#![forbid(unsafe_code)]`.
///
/// Self/self work naturally since both functions live in the same impl scope.
pub(crate) fn arcane_impl_sibling(
    input_fn: LightFn,
    args: &ArcaneArgs,
    target_arch: Option<&str>,
    token_type_name: Option<String>,
    target_feature_attrs: Vec<Attribute>,
    inline_attr: Attribute,
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

    let sibling_name = format_ident!("__arcane_{}", fn_name);

    // Detect self receiver
    let has_self_receiver = inputs
        .first()
        .map(|arg| matches!(arg, FnArg::Receiver(_)))
        .unwrap_or(false);

    // Build sibling signature: same as original but with sibling name, #[doc(hidden)]
    // NOT unsafe — Rust 2024 edition allows safe #[target_feature] functions.
    // Only the call from non-matching context (the wrapper) needs unsafe.
    let sibling_sig_inputs = inputs;

    // Build turbofish for forwarding type/const generic params to sibling
    let turbofish = build_turbofish(generics);

    // Build the call from wrapper to sibling
    let sibling_call = if has_self_receiver {
        // Method: self.__arcane_fn::<T, N>(other_args...)
        let other_args: Vec<proc_macro2::TokenStream> = inputs
            .iter()
            .skip(1) // skip self receiver
            .filter_map(|arg| {
                if let FnArg::Typed(pat_type) = arg
                    && let syn::Pat::Ident(pat_ident) = pat_type.pat.as_ref()
                {
                    let ident = &pat_ident.ident;
                    Some(quote!(#ident))
                } else {
                    None
                }
            })
            .collect();
        quote! { self.#sibling_name #turbofish(#(#other_args),*) }
    } else {
        // Free function: __arcane_fn::<T, N>(all_args...)
        let all_args: Vec<proc_macro2::TokenStream> = inputs
            .iter()
            .filter_map(|arg| {
                if let FnArg::Typed(pat_type) = arg
                    && let syn::Pat::Ident(pat_ident) = pat_type.pat.as_ref()
                {
                    let ident = &pat_ident.ident;
                    Some(quote!(#ident))
                } else {
                    None
                }
            })
            .collect();
        quote! { #sibling_name #turbofish(#(#all_args),*) }
    };

    // Build stub args for suppressing unused warnings
    let stub_args: Vec<proc_macro2::TokenStream> = inputs
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
            FnArg::Receiver(_) => None, // self doesn't need _ = suppression
        })
        .collect();

    let token_type_str = token_type_name.as_deref().unwrap_or("UnknownToken");

    let cfg_guard = gen_cfg_guard(target_arch, args.cfg_feature.as_deref());

    let expanded = if target_arch.is_some() {
        // Sibling function: #[doc(hidden)] #[target_feature] fn __arcane_fn(...)
        // Always private — only the wrapper is user-visible.
        // Safe declaration — Rust 2024 allows safe #[target_feature] functions.
        let sibling_fn = quote! {
            #cfg_guard
            #[doc(hidden)]
            #(#lint_attrs)*
            #(#target_feature_attrs)*
            #inline_attr
            fn #sibling_name #generics (#sibling_sig_inputs) #output #where_clause {
                #body
            }
        };

        // Wrapper function: fn original_name(...) { unsafe { sibling_call } }
        // The unsafe block is needed because the sibling has #[target_feature] and
        // the wrapper doesn't — calling across this boundary requires unsafe.
        let wrapper_fn = quote! {
            #cfg_guard
            #(#attrs)*
            #[inline(always)]
            #vis #sig {
                // SAFETY: The token parameter proves the required CPU features are available.
                // Calling a #[target_feature] function from a non-matching context requires
                // unsafe because the CPU may not support those instructions. The token's
                // existence proves summon() succeeded, so the features are available.
                unsafe { #sibling_call }
            }
        };

        // Optional stub for other architectures / missing feature
        let stub = if args.stub {
            let arch_str = target_arch.unwrap_or("unknown");
            // Negate the cfg guard used for the real implementation
            let not_cfg = match (target_arch, args.cfg_feature.as_deref()) {
                (Some(arch), Some(feat)) => {
                    quote! { #[cfg(not(all(target_arch = #arch, feature = #feat)))] }
                }
                (Some(arch), None) => quote! { #[cfg(not(target_arch = #arch))] },
                _ => quote! {},
            };
            quote! {
                #not_cfg
                #(#attrs)*
                #vis #sig {
                    let _ = (#(#stub_args),*);
                    unreachable!(
                        "BUG: {}() was called but requires {} (target_arch = \"{}\"). \
                         {}::summon() returns None on this architecture, so this function \
                         is unreachable in safe code. If you used forge_token_dangerously(), \
                         that is the bug.",
                        stringify!(#fn_name),
                        #token_type_str,
                        #arch_str,
                        #token_type_str,
                    )
                }
            }
        } else {
            quote! {}
        };

        quote! {
            #sibling_fn
            #wrapper_fn
            #stub
        }
    } else {
        // No specific arch (trait bounds or generic) - no cfg guards, no stub needed.
        // Still use sibling pattern for consistency. Sibling is always private.
        let sibling_fn = quote! {
            #[doc(hidden)]
            #(#lint_attrs)*
            #(#target_feature_attrs)*
            #inline_attr
            fn #sibling_name #generics (#sibling_sig_inputs) #output #where_clause {
                #body
            }
        };

        let wrapper_fn = quote! {
            #(#attrs)*
            #[inline(always)]
            #vis #sig {
                // SAFETY: The token proves the required CPU features are available.
                unsafe { #sibling_call }
            }
        };

        quote! {
            #sibling_fn
            #wrapper_fn
        }
    };

    expanded.into()
}

/// Nested inner function expansion (opt-in via `nested` or `_self = Type`).
///
/// This is the original approach: generates a nested inner function inside the
/// original function. Required when `_self = Type` is used because Self must be
/// replaced in the nested function (where it's not in scope).
pub(crate) fn arcane_impl_nested(
    input_fn: LightFn,
    args: &ArcaneArgs,
    target_arch: Option<&str>,
    token_type_name: Option<String>,
    target_feature_attrs: Vec<Attribute>,
    inline_attr: Attribute,
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

    // Determine self receiver type if present
    let self_receiver_kind: Option<SelfReceiver> = inputs.first().and_then(|arg| match arg {
        FnArg::Receiver(receiver) => {
            if receiver.reference.is_none() {
                Some(SelfReceiver::Owned)
            } else if receiver.mutability.is_some() {
                Some(SelfReceiver::RefMut)
            } else {
                Some(SelfReceiver::Ref)
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

    // Transform output, body, and where clause to replace Self with concrete type if needed.
    let (inner_output, inner_body, inner_where_clause): (
        proc_macro2::TokenStream,
        proc_macro2::TokenStream,
        proc_macro2::TokenStream,
    ) = if let Some(ref self_ty) = args.self_type {
        let transformed_output = replace_self_in_tokens(output.to_token_stream(), self_ty);
        let transformed_body = replace_self_in_tokens(body.clone(), self_ty);
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

    let token_type_str = token_type_name.as_deref().unwrap_or("UnknownToken");
    let cfg_guard = gen_cfg_guard(target_arch, args.cfg_feature.as_deref());

    let expanded = if target_arch.is_some() {
        let stub = if args.stub {
            let arch_str = target_arch.unwrap_or("unknown");
            let not_cfg = match (target_arch, args.cfg_feature.as_deref()) {
                (Some(arch), Some(feat)) => {
                    quote! { #[cfg(not(all(target_arch = #arch, feature = #feat)))] }
                }
                (Some(arch), None) => quote! { #[cfg(not(target_arch = #arch))] },
                _ => quote! {},
            };
            quote! {
                #not_cfg
                #(#attrs)*
                #vis #sig {
                    let _ = (#(#inner_args),*);
                    unreachable!(
                        "BUG: {}() was called but requires {} (target_arch = \"{}\"). \
                         {}::summon() returns None on this architecture, so this function \
                         is unreachable in safe code. If you used forge_token_dangerously(), \
                         that is the bug.",
                        stringify!(#fn_name),
                        #token_type_str,
                        #arch_str,
                        #token_type_str,
                    )
                }
            }
        } else {
            quote! {}
        };

        quote! {
            // Real implementation for the correct architecture
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

                // SAFETY: The token parameter proves the required CPU features are available.
                unsafe { #inner_fn_name #turbofish(#(#inner_args),*) }
            }

            #stub
        }
    } else {
        // No specific arch (trait bounds or generic) - generate without cfg guards
        quote! {
            #(#attrs)*
            #[inline(always)]
            #vis #sig {
                #(#target_feature_attrs)*
                #inline_attr
                #(#lint_attrs)*
                fn #inner_fn_name #generics (#(#inner_params),*) #inner_output #inner_where_clause {
                    #inner_body
                }

                // SAFETY: The token proves the required CPU features are available.
                unsafe { #inner_fn_name #turbofish(#(#inner_args),*) }
            }
        }
    };

    expanded.into()
}
