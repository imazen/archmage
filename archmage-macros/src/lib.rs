//! Proc-macros for archmage SIMD capability tokens.
//!
//! Provides `#[arcane]` attribute (with `#[arcane]` alias) to make raw intrinsics
//! safe via token proof.

use proc_macro::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::{
    Attribute, FnArg, GenericParam, Ident, PatType, Signature, Token, Type, TypeParamBound,
    parse::{Parse, ParseStream},
    parse_macro_input, parse_quote, token,
};

/// A function parsed with the body left as an opaque TokenStream.
///
/// Only the signature is fully parsed into an AST — the body tokens are collected
/// without building any AST nodes (no expressions, statements, or patterns parsed).
/// This saves ~2ms per function invocation at 100 lines of code.
struct LightFn {
    attrs: Vec<Attribute>,
    vis: syn::Visibility,
    sig: Signature,
    brace_token: token::Brace,
    body: proc_macro2::TokenStream,
}

impl Parse for LightFn {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let vis: syn::Visibility = input.parse()?;
        let sig: Signature = input.parse()?;
        let content;
        let brace_token = syn::braced!(content in input);
        let body: proc_macro2::TokenStream = content.parse()?;
        Ok(LightFn {
            attrs,
            vis,
            sig,
            brace_token,
            body,
        })
    }
}

impl ToTokens for LightFn {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        for attr in &self.attrs {
            attr.to_tokens(tokens);
        }
        self.vis.to_tokens(tokens);
        self.sig.to_tokens(tokens);
        self.brace_token.surround(tokens, |tokens| {
            self.body.to_tokens(tokens);
        });
    }
}

/// Replace all `Self` identifier tokens with a concrete type in a token stream.
///
/// Recurses into groups (braces, parens, brackets). Used for `#[arcane(_self = Type)]`
/// to replace `Self` in both the return type and body without needing to parse the body.
fn replace_self_in_tokens(
    tokens: proc_macro2::TokenStream,
    replacement: &Type,
) -> proc_macro2::TokenStream {
    let mut result = proc_macro2::TokenStream::new();
    for tt in tokens {
        match tt {
            proc_macro2::TokenTree::Ident(ref ident) if ident == "Self" => {
                result.extend(replacement.to_token_stream());
            }
            proc_macro2::TokenTree::Group(group) => {
                let new_stream = replace_self_in_tokens(group.stream(), replacement);
                let mut new_group = proc_macro2::Group::new(group.delimiter(), new_stream);
                new_group.set_span(group.span());
                result.extend(std::iter::once(proc_macro2::TokenTree::Group(new_group)));
            }
            other => {
                result.extend(std::iter::once(other));
            }
        }
    }
    result
}

/// Arguments to the `#[arcane]` macro.
#[derive(Default)]
struct ArcaneArgs {
    /// Use `#[inline(always)]` instead of `#[inline]` for the inner function.
    /// Requires nightly Rust with `#![feature(target_feature_inline_always)]`.
    inline_always: bool,
    /// The concrete type to use for `self` receiver.
    /// When specified, `self`/`&self`/`&mut self` is transformed to `_self: Type`/`&Type`/`&mut Type`.
    /// Implies `nested = true`.
    self_type: Option<Type>,
    /// Generate an `unreachable!()` stub on the wrong architecture.
    /// Default is false (cfg-out: no function emitted on wrong arch).
    stub: bool,
    /// Use nested inner function instead of sibling function.
    /// Implied by `_self = Type`. Required for associated functions in impl blocks
    /// that have no `self` receiver (the macro can't distinguish them from free functions).
    nested: bool,
}

impl Parse for ArcaneArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut args = ArcaneArgs::default();

        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            match ident.to_string().as_str() {
                "inline_always" => args.inline_always = true,
                "stub" => args.stub = true,
                "nested" => args.nested = true,
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

// Token-to-features and trait-to-features mappings are generated from
// token-registry.toml by xtask. Regenerate with: cargo run -p xtask -- generate
mod generated;
use generated::{token_to_arch, token_to_features, trait_to_features};

/// Result of extracting token info from a type.
enum TokenTypeInfo {
    /// Concrete token type (e.g., `Avx2Token`)
    Concrete(String),
    /// impl Trait with the trait names (e.g., `impl HasX64V2`)
    ImplTrait(Vec<String>),
    /// Generic type parameter name (e.g., `T`)
    Generic(String),
}

/// Extract token type information from a type.
fn extract_token_type_info(ty: &Type) -> Option<TokenTypeInfo> {
    match ty {
        Type::Path(type_path) => {
            // Get the last segment of the path (e.g., "Avx2Token" from "archmage::Avx2Token")
            type_path.path.segments.last().map(|seg| {
                let name = seg.ident.to_string();
                // Check if it's a known concrete token type
                if token_to_features(&name).is_some() {
                    TokenTypeInfo::Concrete(name)
                } else {
                    // Might be a generic type parameter like `T`
                    TokenTypeInfo::Generic(name)
                }
            })
        }
        Type::Reference(type_ref) => {
            // Handle &Token or &mut Token
            extract_token_type_info(&type_ref.elem)
        }
        Type::ImplTrait(impl_trait) => {
            // Handle `impl HasX64V2` or `impl HasX64V2 + HasNeon`
            let traits: Vec<String> = extract_trait_names_from_bounds(&impl_trait.bounds);
            if traits.is_empty() {
                None
            } else {
                Some(TokenTypeInfo::ImplTrait(traits))
            }
        }
        _ => None,
    }
}

/// Extract trait names from type param bounds.
fn extract_trait_names_from_bounds(
    bounds: &syn::punctuated::Punctuated<TypeParamBound, Token![+]>,
) -> Vec<String> {
    bounds
        .iter()
        .filter_map(|bound| {
            if let TypeParamBound::Trait(trait_bound) = bound {
                trait_bound
                    .path
                    .segments
                    .last()
                    .map(|seg| seg.ident.to_string())
            } else {
                None
            }
        })
        .collect()
}

/// Look up a generic type parameter in the function's generics.
fn find_generic_bounds(sig: &Signature, type_name: &str) -> Option<Vec<String>> {
    // Check inline bounds first (e.g., `fn foo<T: HasX64V2>(token: T)`)
    for param in &sig.generics.params {
        if let GenericParam::Type(type_param) = param
            && type_param.ident == type_name
        {
            let traits = extract_trait_names_from_bounds(&type_param.bounds);
            if !traits.is_empty() {
                return Some(traits);
            }
        }
    }

    // Check where clause (e.g., `fn foo<T>(token: T) where T: HasX64V2`)
    if let Some(where_clause) = &sig.generics.where_clause {
        for predicate in &where_clause.predicates {
            if let syn::WherePredicate::Type(pred_type) = predicate
                && let Type::Path(type_path) = &pred_type.bounded_ty
                && let Some(seg) = type_path.path.segments.last()
                && seg.ident == type_name
            {
                let traits = extract_trait_names_from_bounds(&pred_type.bounds);
                if !traits.is_empty() {
                    return Some(traits);
                }
            }
        }
    }

    None
}

/// Convert trait names to features, collecting all features from all traits.
fn traits_to_features(trait_names: &[String]) -> Option<Vec<&'static str>> {
    let mut all_features = Vec::new();

    for trait_name in trait_names {
        if let Some(features) = trait_to_features(trait_name) {
            for &feature in features {
                if !all_features.contains(&feature) {
                    all_features.push(feature);
                }
            }
        }
    }

    if all_features.is_empty() {
        None
    } else {
        Some(all_features)
    }
}

/// Trait names that don't map to any CPU features. These are valid in the type
/// system but cannot be used as token bounds in `#[arcane]`/`#[rite]` because
/// the macros need concrete features to generate `#[target_feature]` attributes.
const FEATURELESS_TRAIT_NAMES: &[&str] = &["SimdToken", "IntoConcreteToken"];

/// Check if any trait names are featureless (no CPU feature mapping).
/// Returns the first featureless trait name found.
fn find_featureless_trait(trait_names: &[String]) -> Option<&'static str> {
    for name in trait_names {
        for &featureless in FEATURELESS_TRAIT_NAMES {
            if name == featureless {
                return Some(featureless);
            }
        }
    }
    None
}

/// Diagnose why `find_token_param` failed. Returns the name of a featureless
/// trait if the signature has a parameter bounded by one (e.g., `SimdToken`).
fn diagnose_featureless_token(sig: &Signature) -> Option<&'static str> {
    for arg in &sig.inputs {
        if let FnArg::Typed(PatType { ty, .. }) = arg
            && let Some(info) = extract_token_type_info(ty)
        {
            match &info {
                TokenTypeInfo::ImplTrait(names) => {
                    if let Some(name) = find_featureless_trait(names) {
                        return Some(name);
                    }
                }
                TokenTypeInfo::Generic(type_name) => {
                    // Check if the type name itself is a featureless trait
                    // (e.g., `token: SimdToken` used as a bare path)
                    let as_vec = vec![type_name.clone()];
                    if let Some(name) = find_featureless_trait(&as_vec) {
                        return Some(name);
                    }
                    // Check generic bounds (e.g., `T: SimdToken`)
                    if let Some(bounds) = find_generic_bounds(sig, type_name)
                        && let Some(name) = find_featureless_trait(&bounds)
                    {
                        return Some(name);
                    }
                }
                TokenTypeInfo::Concrete(_) => {}
            }
        }
    }
    None
}

/// Result of finding a token parameter in a function signature.
struct TokenParamInfo {
    /// The parameter identifier (e.g., `token`)
    ident: Ident,
    /// Target features to enable (e.g., `["avx2", "fma"]`)
    features: Vec<&'static str>,
    /// Target architecture (Some for concrete tokens, None for traits/generics)
    target_arch: Option<&'static str>,
    /// Concrete token type name (Some for concrete tokens, None for traits/generics)
    token_type_name: Option<String>,
}

/// Find the first token parameter in a function signature.
fn find_token_param(sig: &Signature) -> Option<TokenParamInfo> {
    for arg in &sig.inputs {
        match arg {
            FnArg::Receiver(_) => {
                // Self receivers (self, &self, &mut self) are not yet supported.
                // The macro creates an inner function, and Rust's inner functions
                // cannot have `self` parameters. Supporting this would require
                // AST rewriting to replace `self` with a regular parameter.
                // See the module docs for the workaround.
                continue;
            }
            FnArg::Typed(PatType { pat, ty, .. }) => {
                if let Some(info) = extract_token_type_info(ty) {
                    let (features, arch, token_name) = match info {
                        TokenTypeInfo::Concrete(ref name) => {
                            let features = token_to_features(name).map(|f| f.to_vec());
                            let arch = token_to_arch(name);
                            (features, arch, Some(name.clone()))
                        }
                        TokenTypeInfo::ImplTrait(trait_names) => {
                            (traits_to_features(&trait_names), None, None)
                        }
                        TokenTypeInfo::Generic(type_name) => {
                            // Look up the generic parameter's bounds
                            let features = find_generic_bounds(sig, &type_name)
                                .and_then(|traits| traits_to_features(&traits));
                            (features, None, None)
                        }
                    };

                    if let Some(features) = features {
                        // Extract parameter name (or synthesize one for wildcard `_`)
                        let ident = match pat.as_ref() {
                            syn::Pat::Ident(pat_ident) => Some(pat_ident.ident.clone()),
                            syn::Pat::Wild(w) => {
                                Some(Ident::new("__archmage_token", w.underscore_token.span))
                            }
                            _ => None,
                        };
                        if let Some(ident) = ident {
                            return Some(TokenParamInfo {
                                ident,
                                features,
                                target_arch: arch,
                                token_type_name: token_name,
                            });
                        }
                    }
                }
            }
        }
    }
    None
}

/// Represents the kind of self receiver and the transformed parameter.
enum SelfReceiver {
    /// `self` (by value/move)
    Owned,
    /// `&self` (shared reference)
    Ref,
    /// `&mut self` (mutable reference)
    RefMut,
}

/// Shared implementation for arcane/arcane macros.
fn arcane_impl(mut input_fn: LightFn, macro_name: &str, args: ArcaneArgs) -> TokenStream {
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

    // Build target_feature attributes
    let target_feature_attrs: Vec<Attribute> = features
        .iter()
        .map(|feature| parse_quote!(#[target_feature(enable = #feature)]))
        .collect();

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
fn arcane_impl_wasm_safe(
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

    // Prepend target_feature + inline attrs
    let mut new_attrs = target_feature_attrs;
    new_attrs.push(inline_attr);
    for attr in attrs {
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
/// unsafe fn __arcane_process(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] { body }
///
/// #[cfg(target_arch = "x86_64")]
/// fn process(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
///     unsafe { __arcane_process(token, data) }
/// }
/// ```
///
/// Self/self work naturally since both functions live in the same impl scope.
fn arcane_impl_sibling(
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
    let attrs = &input_fn.attrs;

    let sibling_name = format_ident!("__arcane_{}", fn_name);

    // Detect self receiver
    let has_self_receiver = inputs
        .first()
        .map(|arg| matches!(arg, FnArg::Receiver(_)))
        .unwrap_or(false);

    // Build sibling signature: same as original but with sibling name, unsafe, #[doc(hidden)]
    // The sibling has the exact same parameters (including self receiver if present).
    let sibling_sig_inputs = inputs;

    // Build the call from wrapper to sibling
    let sibling_call = if has_self_receiver {
        // Method: self.__arcane_fn(other_args...)
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
        quote! { self.#sibling_name(#(#other_args),*) }
    } else {
        // Free function: __arcane_fn(all_args...)
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
        quote! { #sibling_name(#(#all_args),*) }
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

    let expanded = if let Some(arch) = target_arch {
        // Sibling function: #[doc(hidden)] #[target_feature] pub(?) unsafe fn __arcane_fn(...)
        // Note: visibility must come before `unsafe` in Rust syntax
        let sibling_fn = quote! {
            #[cfg(target_arch = #arch)]
            #[doc(hidden)]
            #(#target_feature_attrs)*
            #inline_attr
            #vis unsafe fn #sibling_name #generics (#sibling_sig_inputs) #output #where_clause {
                #body
            }
        };

        // Wrapper function: fn original_name(...) { unsafe { sibling_call } }
        let wrapper_fn = quote! {
            #[cfg(target_arch = #arch)]
            #(#attrs)*
            #vis #sig {
                // SAFETY: The token parameter proves the required CPU features are available.
                // Calling a #[target_feature] function from a non-matching context requires
                // unsafe because the CPU may not support those instructions. The token's
                // existence proves summon() succeeded, so the features are available.
                unsafe { #sibling_call }
            }
        };

        // Optional stub for other architectures
        let stub = if args.stub {
            quote! {
                #[cfg(not(target_arch = #arch))]
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
                        #arch,
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
        // Still use sibling pattern for consistency.
        let sibling_fn = quote! {
            #[doc(hidden)]
            #(#target_feature_attrs)*
            #inline_attr
            #vis unsafe fn #sibling_name #generics (#sibling_sig_inputs) #output #where_clause {
                #body
            }
        };

        let wrapper_fn = quote! {
            #(#attrs)*
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
fn arcane_impl_nested(
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
    let attrs = &input_fn.attrs;

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
    let expanded = if let Some(arch) = target_arch {
        let stub = if args.stub {
            quote! {
                // Stub for other architectures - the token cannot be obtained
                #[cfg(not(target_arch = #arch))]
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
                        #arch,
                        #token_type_str,
                    )
                }
            }
        } else {
            quote! {}
        };

        quote! {
            // Real implementation for the correct architecture
            #[cfg(target_arch = #arch)]
            #(#attrs)*
            #vis #sig {
                #(#target_feature_attrs)*
                #inline_attr
                fn #inner_fn_name #generics (#(#inner_params),*) #inner_output #inner_where_clause {
                    #inner_body
                }

                // SAFETY: The token parameter proves the required CPU features are available.
                unsafe { #inner_fn_name(#(#inner_args),*) }
            }

            #stub
        }
    } else {
        // No specific arch (trait bounds or generic) - generate without cfg guards
        quote! {
            #(#attrs)*
            #vis #sig {
                #(#target_feature_attrs)*
                #inline_attr
                fn #inner_fn_name #generics (#(#inner_params),*) #inner_output #inner_where_clause {
                    #inner_body
                }

                // SAFETY: The token proves the required CPU features are available.
                unsafe { #inner_fn_name(#(#inner_args),*) }
            }
        }
    };

    expanded.into()
}

/// Mark a function as an arcane SIMD function.
///
/// This macro generates a safe wrapper around a `#[target_feature]` function.
/// The token parameter type determines which CPU features are enabled.
///
/// # Expansion Modes
///
/// ## Sibling (default)
///
/// Generates two functions at the same scope: a `#[target_feature]` unsafe sibling
/// and a safe wrapper. `self`/`Self` work naturally since both functions share scope.
///
/// ```ignore
/// #[arcane]
/// fn process(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] { /* body */ }
/// // Expands to (x86_64 only):
/// #[cfg(target_arch = "x86_64")]
/// #[doc(hidden)]
/// #[target_feature(enable = "avx2,fma,...")]
/// unsafe fn __arcane_process(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] { /* body */ }
///
/// #[cfg(target_arch = "x86_64")]
/// fn process(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] {
///     unsafe { __arcane_process(token, data) }
/// }
/// ```
///
/// Methods work naturally:
///
/// ```ignore
/// impl MyType {
///     #[arcane]
///     fn compute(&self, token: X64V3Token) -> f32 {
///         self.data.iter().sum()  // self/Self just work!
///     }
/// }
/// ```
///
/// ## Nested (`nested` or `_self = Type`)
///
/// Generates a nested inner function inside the original. Required for trait impls
/// (where sibling functions would fail) and when `_self = Type` is used.
///
/// ```ignore
/// impl SimdOps for MyType {
///     #[arcane(_self = MyType)]
///     fn compute(&self, token: X64V3Token) -> Self {
///         // Use _self instead of self, Self replaced with MyType
///         _self.data.iter().sum()
///     }
/// }
/// ```
///
/// # Cross-Architecture Behavior
///
/// **Default (cfg-out):** On the wrong architecture, the function is not emitted
/// at all — no stub, no dead code. Code that references it must be cfg-gated.
///
/// **With `stub`:** Generates an `unreachable!()` stub on wrong architectures.
/// Use when cross-arch dispatch references the function without cfg guards.
///
/// ```ignore
/// #[arcane(stub)]  // generates stub on wrong arch
/// fn process_neon(token: NeonToken, data: &[f32]) -> f32 { ... }
/// ```
///
/// `incant!` is unaffected — it already cfg-gates dispatch calls by architecture.
///
/// # Token Parameter Forms
///
/// ```ignore
/// // Concrete token
/// #[arcane]
/// fn process(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] { ... }
///
/// // impl Trait bound
/// #[arcane]
/// fn process(token: impl HasX64V2, data: &[f32; 8]) -> [f32; 8] { ... }
///
/// // Generic with inline or where-clause bounds
/// #[arcane]
/// fn process<T: HasX64V2>(token: T, data: &[f32; 8]) -> [f32; 8] { ... }
///
/// // Wildcard
/// #[arcane]
/// fn process(_: X64V3Token, data: &[f32; 8]) -> [f32; 8] { ... }
/// ```
///
/// # Options
///
/// | Option | Effect |
/// |--------|--------|
/// | `stub` | Generate `unreachable!()` stub on wrong architecture |
/// | `nested` | Use nested inner function instead of sibling |
/// | `_self = Type` | Implies `nested`, transforms self receiver, replaces Self |
/// | `inline_always` | Use `#[inline(always)]` (requires nightly) |
///
/// # Supported Tokens
///
/// - **x86_64**: `X64V2Token`, `X64V3Token`/`Desktop64`, `X64V4Token`/`Avx512Token`/`Server64`,
///   `X64V4xToken`, `Avx512Fp16Token`, `X64CryptoToken`, `X64V3CryptoToken`
/// - **ARM**: `NeonToken`/`Arm64`, `Arm64V2Token`, `Arm64V3Token`,
///   `NeonAesToken`, `NeonSha3Token`, `NeonCrcToken`
/// - **WASM**: `Wasm128Token`
///
/// # Supported Trait Bounds
///
/// `HasX64V2`, `HasX64V4`, `HasNeon`, `HasNeonAes`, `HasNeonSha3`, `HasArm64V2`, `HasArm64V3`
///
/// ```ignore
/// #![feature(target_feature_inline_always)]
///
/// #[arcane(inline_always)]
/// fn fast_kernel(token: Avx2Token, data: &mut [f32]) {
///     // Inner function will use #[inline(always)]
/// }
/// ```
#[proc_macro_attribute]
pub fn arcane(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as ArcaneArgs);
    let input_fn = parse_macro_input!(item as LightFn);
    arcane_impl(input_fn, "arcane", args)
}

/// Legacy alias for [`arcane`].
///
/// **Deprecated:** Use `#[arcane]` instead. This alias exists only for migration.
#[proc_macro_attribute]
#[doc(hidden)]
pub fn simd_fn(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as ArcaneArgs);
    let input_fn = parse_macro_input!(item as LightFn);
    arcane_impl(input_fn, "simd_fn", args)
}

/// Descriptive alias for [`arcane`].
///
/// Generates a safe wrapper around a `#[target_feature]` inner function.
/// The token type in your signature determines which CPU features are enabled.
/// Creates an LLVM optimization boundary — use [`token_target_features`]
/// (alias for [`rite`]) for inner helpers to avoid this.
///
/// Since Rust 1.85, value-based SIMD intrinsics are safe inside
/// `#[target_feature]` functions. This macro generates the `#[target_feature]`
/// wrapper so you never need to write `unsafe` for SIMD code.
///
/// See [`arcane`] for full documentation and examples.
#[proc_macro_attribute]
pub fn token_target_features_boundary(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as ArcaneArgs);
    let input_fn = parse_macro_input!(item as LightFn);
    arcane_impl(input_fn, "token_target_features_boundary", args)
}

// ============================================================================
// Rite macro for inner SIMD functions (inlines into matching #[target_feature] callers)
// ============================================================================

/// Annotate inner SIMD helpers called from `#[arcane]` functions.
///
/// Unlike `#[arcane]`, which creates an inner `#[target_feature]` function behind
/// a safe boundary, `#[rite]` adds `#[target_feature]` and `#[inline]` directly.
/// LLVM inlines it into any caller with matching features — no boundary crossing.
///
/// # When to Use
///
/// Use `#[rite]` for helper functions that are **only** called from within
/// `#[arcane]` functions with matching or superset token types:
///
/// ```ignore
/// use archmage::{arcane, rite, X64V3Token};
///
/// #[arcane]
/// fn outer(token: X64V3Token, data: &[f32; 8]) -> f32 {
///     // helper inlines — same target features, no boundary
///     helper(token, data) * 2.0
/// }
///
/// #[rite]
/// fn helper(token: X64V3Token, data: &[f32; 8]) -> f32 {
///     // Just has #[target_feature(enable = "avx2,fma,...")]
///     // Called from #[arcane] context, so features are guaranteed
///     let v = f32x8::from_array(token, *data);
///     v.reduce_add()
/// }
/// ```
///
/// # Safety
///
/// `#[rite]` functions can only be safely called from contexts where the
/// required CPU features are enabled:
/// - From within `#[arcane]` functions with matching/superset tokens
/// - From within other `#[rite]` functions with matching/superset tokens
/// - From code compiled with `-Ctarget-cpu` that enables the features
///
/// Calling from other contexts requires `unsafe` and the caller must ensure
/// the CPU supports the required features.
///
/// # Cross-Architecture Behavior
///
/// Like `#[arcane]`, defaults to cfg-out (no function on wrong arch).
/// Use `#[rite(stub)]` to generate an unreachable stub instead.
///
/// # Comparison with #[arcane]
///
/// | Aspect | `#[arcane]` | `#[rite]` |
/// |--------|-------------|-----------|
/// | Creates wrapper | Yes | No |
/// | Entry point | Yes | No |
/// | Inlines into caller | No (barrier) | Yes |
/// | Safe to call anywhere | Yes (with token) | Only from feature-enabled context |
/// | `stub` param | Yes | Yes |
#[proc_macro_attribute]
pub fn rite(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as RiteArgs);
    let input_fn = parse_macro_input!(item as LightFn);
    rite_impl(input_fn, args)
}

/// Descriptive alias for [`rite`].
///
/// Applies `#[target_feature]` + `#[inline]` based on the token type in your
/// function signature. No wrapper, no optimization boundary. Use for functions
/// called from within `#[arcane]`/`#[token_target_features_boundary]` code.
///
/// Since Rust 1.85, calling a `#[target_feature]` function from another function
/// with matching features is safe — no `unsafe` needed.
///
/// See [`rite`] for full documentation and examples.
#[proc_macro_attribute]
pub fn token_target_features(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as RiteArgs);
    let input_fn = parse_macro_input!(item as LightFn);
    rite_impl(input_fn, args)
}

/// Arguments for the `#[rite]` macro.
#[derive(Default)]
struct RiteArgs {
    /// Generate an `unreachable!()` stub on the wrong architecture.
    /// Default is false (cfg-out: no function emitted on wrong arch).
    stub: bool,
}

impl Parse for RiteArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut args = RiteArgs::default();

        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            match ident.to_string().as_str() {
                "stub" => args.stub = true,
                other => {
                    return Err(syn::Error::new(
                        ident.span(),
                        format!(
                            "unknown rite argument: `{}`. Supported: `stub`.\n\
                             Note: inline_always is not supported because \
                             #[inline(always)] + #[target_feature] requires nightly Rust.",
                            other
                        ),
                    ));
                }
            }
            if input.peek(Token![,]) {
                let _: Token![,] = input.parse()?;
            }
        }

        Ok(args)
    }
}

/// Implementation for the `#[rite]` macro.
fn rite_impl(mut input_fn: LightFn, args: RiteArgs) -> TokenStream {
    // Find the token parameter and its features
    let TokenParamInfo {
        features,
        target_arch,
        ..
    } = match find_token_param(&input_fn.sig) {
        Some(result) => result,
        None => {
            // Check for specific misuse: featureless traits like SimdToken
            if let Some(trait_name) = diagnose_featureless_token(&input_fn.sig) {
                let msg = format!(
                    "`{trait_name}` cannot be used as a token bound in #[rite] \
                     because it doesn't specify any CPU features.\n\
                     \n\
                     #[rite] needs concrete features to generate #[target_feature]. \
                     Use a concrete token or a feature trait:\n\
                     \n\
                     Concrete tokens: X64V3Token, Desktop64, NeonToken, Arm64V2Token, ...\n\
                     Feature traits:  impl HasX64V2, impl HasNeon, impl HasArm64V3, ..."
                );
                return syn::Error::new_spanned(&input_fn.sig, msg)
                    .to_compile_error()
                    .into();
            }
            let msg = "rite requires a token parameter. Supported forms:\n\
                 - Concrete: `token: X64V3Token`\n\
                 - impl Trait: `token: impl HasX64V2`\n\
                 - Generic: `fn foo<T: HasX64V2>(token: T, ...)`";
            return syn::Error::new_spanned(&input_fn.sig, msg)
                .to_compile_error()
                .into();
        }
    };

    // Build target_feature attributes
    let target_feature_attrs: Vec<Attribute> = features
        .iter()
        .map(|feature| parse_quote!(#[target_feature(enable = #feature)]))
        .collect();

    // Always use #[inline] - #[inline(always)] + #[target_feature] requires nightly
    let inline_attr: Attribute = parse_quote!(#[inline]);

    // Prepend attributes to the function
    let mut new_attrs = target_feature_attrs;
    new_attrs.push(inline_attr);
    new_attrs.append(&mut input_fn.attrs);
    input_fn.attrs = new_attrs;

    // If we know the target arch, generate cfg-gated impl (+ optional stub)
    if let Some(arch) = target_arch {
        let vis = &input_fn.vis;
        let sig = &input_fn.sig;
        let attrs = &input_fn.attrs;
        let body = &input_fn.body;

        let stub = if args.stub {
            quote! {
                #[cfg(not(target_arch = #arch))]
                #vis #sig {
                    unreachable!(concat!(
                        "This function requires ",
                        #arch,
                        " architecture"
                    ))
                }
            }
        } else {
            quote! {}
        };

        quote! {
            #[cfg(target_arch = #arch)]
            #(#attrs)*
            #vis #sig {
                #body
            }

            #stub
        }
        .into()
    } else {
        // No specific arch (trait bounds) - just emit the annotated function
        quote!(#input_fn).into()
    }
}

// =============================================================================
// magetypes! macro - generate platform variants from generic function
// =============================================================================

/// Generate platform-specific variants from a function by replacing `Token`.
///
/// Use `Token` as a placeholder for the token type. The macro generates
/// suffixed variants with `Token` replaced by the concrete token type, and
/// each variant wrapped in the appropriate `#[cfg(target_arch = ...)]` guard.
///
/// # Default tiers
///
/// Without arguments, generates `_v3`, `_v4`, `_neon`, `_wasm128`, `_scalar`:
///
/// ```rust,ignore
/// #[magetypes]
/// fn process(token: Token, data: &[f32]) -> f32 {
///     inner_simd_work(token, data)
/// }
/// ```
///
/// # Explicit tiers
///
/// Specify which tiers to generate:
///
/// ```rust,ignore
/// #[magetypes(v1, v3, neon)]
/// fn process(token: Token, data: &[f32]) -> f32 {
///     inner_simd_work(token, data)
/// }
/// // Generates: process_v1, process_v3, process_neon, process_scalar
/// ```
///
/// `scalar` is always included implicitly.
///
/// Known tiers: `v1`, `v2`, `v3`, `v4`, `v4x`, `neon`, `neon_aes`,
/// `neon_sha3`, `neon_crc`, `wasm128`, `wasm128_relaxed`, `scalar`.
///
/// # What gets replaced
///
/// **Only `Token`** is replaced — with the concrete token type for each variant
/// (e.g., `archmage::X64V3Token`, `archmage::ScalarToken`). SIMD types like
/// `f32x8` and constants like `LANES` are **not** replaced by this macro.
///
/// # Usage with incant!
///
/// The generated variants work with `incant!` for dispatch:
///
/// ```rust,ignore
/// pub fn process_api(data: &[f32]) -> f32 {
///     incant!(process(data))
/// }
///
/// // Or with matching explicit tiers:
/// pub fn process_api(data: &[f32]) -> f32 {
///     incant!(process(data), [v1, v3, neon])
/// }
/// ```
#[proc_macro_attribute]
pub fn magetypes(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as LightFn);

    // Parse optional tier list from attribute args
    let tier_names: Vec<String> = if attr.is_empty() {
        DEFAULT_TIER_NAMES.iter().map(|s| s.to_string()).collect()
    } else {
        let parser = |input: ParseStream| input.parse_terminated(Ident::parse, Token![,]);
        let idents = match syn::parse::Parser::parse(parser, attr) {
            Ok(p) => p,
            Err(e) => return e.to_compile_error().into(),
        };
        idents.iter().map(|i| i.to_string()).collect()
    };

    let tiers = match resolve_tiers(&tier_names, input_fn.sig.ident.span()) {
        Ok(t) => t,
        Err(e) => return e.to_compile_error().into(),
    };

    magetypes_impl(input_fn, &tiers)
}

fn magetypes_impl(mut input_fn: LightFn, tiers: &[&TierDescriptor]) -> TokenStream {
    // Strip user-provided #[arcane] / #[rite] to prevent double-wrapping
    // (magetypes auto-adds #[arcane] on non-scalar variants)
    input_fn
        .attrs
        .retain(|attr| !attr.path().is_ident("arcane") && !attr.path().is_ident("rite"));

    let fn_name = &input_fn.sig.ident;
    let fn_attrs = &input_fn.attrs;

    // Convert function to string for text substitution
    let fn_str = input_fn.to_token_stream().to_string();

    let mut variants = Vec::new();

    for tier in tiers {
        // Create suffixed function name
        let suffixed_name = format!("{}_{}", fn_name, tier.suffix);

        // Do text substitution
        let mut variant_str = fn_str.clone();

        // Replace function name
        variant_str = variant_str.replacen(&fn_name.to_string(), &suffixed_name, 1);

        // Replace Token type with concrete token
        variant_str = variant_str.replace("Token", tier.token_path);

        // Parse back to tokens
        let variant_tokens: proc_macro2::TokenStream = match variant_str.parse() {
            Ok(t) => t,
            Err(e) => {
                return syn::Error::new_spanned(
                    &input_fn,
                    format!(
                        "Failed to parse generated variant `{}`: {}",
                        suffixed_name, e
                    ),
                )
                .to_compile_error()
                .into();
            }
        };

        // Add cfg guards
        let cfg_guard = match (tier.target_arch, tier.cargo_feature) {
            (Some(arch), Some(feature)) => {
                quote! { #[cfg(all(target_arch = #arch, feature = #feature))] }
            }
            (Some(arch), None) => {
                quote! { #[cfg(target_arch = #arch)] }
            }
            (None, Some(feature)) => {
                quote! { #[cfg(feature = #feature)] }
            }
            (None, None) => {
                quote! {} // No guard needed (scalar)
            }
        };

        variants.push(if tier.name != "scalar" {
            // Non-scalar variants get #[arcane] so target_feature is applied
            quote! {
                #cfg_guard
                #[archmage::arcane]
                #variant_tokens
            }
        } else {
            quote! {
                #cfg_guard
                #variant_tokens
            }
        });
    }

    // Remove attributes from the list that should not be duplicated
    let filtered_attrs: Vec<_> = fn_attrs
        .iter()
        .filter(|a| !a.path().is_ident("magetypes"))
        .collect();

    let output = quote! {
        #(#filtered_attrs)*
        #(#variants)*
    };

    output.into()
}

// =============================================================================
// incant! macro - dispatch to platform-specific variants
// =============================================================================

// =============================================================================
// Tier descriptors for incant! and #[magetypes]
// =============================================================================

/// Describes a dispatch tier for incant! and #[magetypes].
struct TierDescriptor {
    /// Tier name as written in user code (e.g., "v3", "neon")
    name: &'static str,
    /// Function suffix (e.g., "v3", "neon", "scalar")
    suffix: &'static str,
    /// Token type path (e.g., "archmage::X64V3Token")
    token_path: &'static str,
    /// IntoConcreteToken method name (e.g., "as_x64v3")
    as_method: &'static str,
    /// Target architecture for cfg guard (None = no guard)
    target_arch: Option<&'static str>,
    /// Required cargo feature (None = no feature guard)
    cargo_feature: Option<&'static str>,
    /// Dispatch priority (higher = tried first within same arch)
    priority: u32,
}

/// All known tiers in dispatch-priority order (highest first within arch).
const ALL_TIERS: &[TierDescriptor] = &[
    // x86: highest to lowest
    TierDescriptor {
        name: "v4x",
        suffix: "v4x",
        token_path: "archmage::X64V4xToken",
        as_method: "as_x64v4x",
        target_arch: Some("x86_64"),
        cargo_feature: Some("avx512"),
        priority: 50,
    },
    TierDescriptor {
        name: "v4",
        suffix: "v4",
        token_path: "archmage::X64V4Token",
        as_method: "as_x64v4",
        target_arch: Some("x86_64"),
        cargo_feature: Some("avx512"),
        priority: 40,
    },
    TierDescriptor {
        name: "v3_crypto",
        suffix: "v3_crypto",
        token_path: "archmage::X64V3CryptoToken",
        as_method: "as_x64v3_crypto",
        target_arch: Some("x86_64"),
        cargo_feature: None,
        priority: 35,
    },
    TierDescriptor {
        name: "v3",
        suffix: "v3",
        token_path: "archmage::X64V3Token",
        as_method: "as_x64v3",
        target_arch: Some("x86_64"),
        cargo_feature: None,
        priority: 30,
    },
    TierDescriptor {
        name: "x64_crypto",
        suffix: "x64_crypto",
        token_path: "archmage::X64CryptoToken",
        as_method: "as_x64_crypto",
        target_arch: Some("x86_64"),
        cargo_feature: None,
        priority: 25,
    },
    TierDescriptor {
        name: "v2",
        suffix: "v2",
        token_path: "archmage::X64V2Token",
        as_method: "as_x64v2",
        target_arch: Some("x86_64"),
        cargo_feature: None,
        priority: 20,
    },
    TierDescriptor {
        name: "v1",
        suffix: "v1",
        token_path: "archmage::X64V1Token",
        as_method: "as_x64v1",
        target_arch: Some("x86_64"),
        cargo_feature: None,
        priority: 10,
    },
    // ARM: highest to lowest
    TierDescriptor {
        name: "arm_v3",
        suffix: "arm_v3",
        token_path: "archmage::Arm64V3Token",
        as_method: "as_arm_v3",
        target_arch: Some("aarch64"),
        cargo_feature: None,
        priority: 50,
    },
    TierDescriptor {
        name: "arm_v2",
        suffix: "arm_v2",
        token_path: "archmage::Arm64V2Token",
        as_method: "as_arm_v2",
        target_arch: Some("aarch64"),
        cargo_feature: None,
        priority: 40,
    },
    TierDescriptor {
        name: "neon_aes",
        suffix: "neon_aes",
        token_path: "archmage::NeonAesToken",
        as_method: "as_neon_aes",
        target_arch: Some("aarch64"),
        cargo_feature: None,
        priority: 30,
    },
    TierDescriptor {
        name: "neon_sha3",
        suffix: "neon_sha3",
        token_path: "archmage::NeonSha3Token",
        as_method: "as_neon_sha3",
        target_arch: Some("aarch64"),
        cargo_feature: None,
        priority: 30,
    },
    TierDescriptor {
        name: "neon_crc",
        suffix: "neon_crc",
        token_path: "archmage::NeonCrcToken",
        as_method: "as_neon_crc",
        target_arch: Some("aarch64"),
        cargo_feature: None,
        priority: 30,
    },
    TierDescriptor {
        name: "neon",
        suffix: "neon",
        token_path: "archmage::NeonToken",
        as_method: "as_neon",
        target_arch: Some("aarch64"),
        cargo_feature: None,
        priority: 20,
    },
    // WASM
    TierDescriptor {
        name: "wasm128_relaxed",
        suffix: "wasm128_relaxed",
        token_path: "archmage::Wasm128RelaxedToken",
        as_method: "as_wasm128_relaxed",
        target_arch: Some("wasm32"),
        cargo_feature: None,
        priority: 21,
    },
    TierDescriptor {
        name: "wasm128",
        suffix: "wasm128",
        token_path: "archmage::Wasm128Token",
        as_method: "as_wasm128",
        target_arch: Some("wasm32"),
        cargo_feature: None,
        priority: 20,
    },
    // Scalar (always last)
    TierDescriptor {
        name: "scalar",
        suffix: "scalar",
        token_path: "archmage::ScalarToken",
        as_method: "as_scalar",
        target_arch: None,
        cargo_feature: None,
        priority: 0,
    },
];

/// Default tiers (backwards-compatible with pre-explicit behavior).
const DEFAULT_TIER_NAMES: &[&str] = &["v4", "v3", "neon", "wasm128", "scalar"];

/// Look up a tier by name, returning an error on unknown names.
fn find_tier(name: &str) -> Option<&'static TierDescriptor> {
    ALL_TIERS.iter().find(|t| t.name == name)
}

/// Resolve tier names to descriptors, sorted by dispatch priority (highest first).
/// Always appends "scalar" if not already present.
fn resolve_tiers(
    tier_names: &[String],
    error_span: proc_macro2::Span,
) -> syn::Result<Vec<&'static TierDescriptor>> {
    let mut tiers = Vec::new();
    for name in tier_names {
        match find_tier(name) {
            Some(tier) => tiers.push(tier),
            None => {
                let known: Vec<&str> = ALL_TIERS.iter().map(|t| t.name).collect();
                return Err(syn::Error::new(
                    error_span,
                    format!("unknown tier `{}`. Known tiers: {}", name, known.join(", ")),
                ));
            }
        }
    }

    // Always include scalar fallback
    if !tiers.iter().any(|t| t.name == "scalar") {
        tiers.push(find_tier("scalar").unwrap());
    }

    // Sort by priority (highest first) for correct dispatch order
    tiers.sort_by(|a, b| b.priority.cmp(&a.priority));

    Ok(tiers)
}

// =============================================================================
// incant! macro - dispatch to platform-specific variants
// =============================================================================

/// Input for the incant! macro
struct IncantInput {
    /// Function path to call (e.g. `func` or `module::func`)
    func_path: syn::Path,
    /// Arguments to pass
    args: Vec<syn::Expr>,
    /// Optional token variable for passthrough mode
    with_token: Option<syn::Expr>,
    /// Optional explicit tier list (None = default tiers)
    tiers: Option<(Vec<String>, proc_macro2::Span)>,
}

/// Create a suffixed version of a function path.
/// e.g. `module::func` + `"v3"` → `module::func_v3`
fn suffix_path(path: &syn::Path, suffix: &str) -> syn::Path {
    let mut suffixed = path.clone();
    if let Some(last) = suffixed.segments.last_mut() {
        last.ident = format_ident!("{}_{}", last.ident, suffix);
    }
    suffixed
}

impl Parse for IncantInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // Parse: function_path(arg1, arg2, ...) [with token_expr] [, [tier1, tier2, ...]]
        let func_path: syn::Path = input.parse()?;

        // Parse parenthesized arguments
        let content;
        syn::parenthesized!(content in input);
        let args = content
            .parse_terminated(syn::Expr::parse, Token![,])?
            .into_iter()
            .collect();

        // Check for optional "with token"
        let with_token = if input.peek(Ident) {
            let kw: Ident = input.parse()?;
            if kw != "with" {
                return Err(syn::Error::new_spanned(kw, "expected `with` keyword"));
            }
            Some(input.parse()?)
        } else {
            None
        };

        // Check for optional tier list: , [tier1, tier2, ...]
        let tiers = if input.peek(Token![,]) {
            let _: Token![,] = input.parse()?;
            let bracket_content;
            let bracket = syn::bracketed!(bracket_content in input);
            let tier_idents = bracket_content.parse_terminated(Ident::parse, Token![,])?;
            let tier_names: Vec<String> = tier_idents.iter().map(|i| i.to_string()).collect();
            Some((tier_names, bracket.span.join()))
        } else {
            None
        };

        Ok(IncantInput {
            func_path,
            args,
            with_token,
            tiers,
        })
    }
}

/// Dispatch to platform-specific SIMD variants.
///
/// # Entry Point Mode (no token yet)
///
/// Summons tokens and dispatches to the best available variant:
///
/// ```rust,ignore
/// pub fn public_api(data: &[f32]) -> f32 {
///     incant!(dot(data))
/// }
/// ```
///
/// Expands to runtime feature detection + dispatch to `dot_v3`, `dot_v4`,
/// `dot_neon`, `dot_wasm128`, or `dot_scalar`.
///
/// # Explicit Tiers
///
/// Specify which tiers to dispatch to:
///
/// ```rust,ignore
/// // Only dispatch to v1, v3, neon, and scalar
/// pub fn api(data: &[f32]) -> f32 {
///     incant!(process(data), [v1, v3, neon])
/// }
/// ```
///
/// `scalar` is always included implicitly. Unknown tier names cause a
/// compile error. Tiers are automatically sorted into correct dispatch
/// order (highest priority first).
///
/// Known tiers: `v1`, `v2`, `v3`, `v4`, `v4x`, `neon`, `neon_aes`,
/// `neon_sha3`, `neon_crc`, `wasm128`, `wasm128_relaxed`, `scalar`.
///
/// # Passthrough Mode (already have token)
///
/// Uses compile-time dispatch via `IntoConcreteToken`:
///
/// ```rust,ignore
/// #[arcane]
/// fn outer(token: X64V3Token, data: &[f32]) -> f32 {
///     incant!(inner(data) with token)
/// }
/// ```
///
/// Also supports explicit tiers:
///
/// ```rust,ignore
/// fn inner<T: IntoConcreteToken>(token: T, data: &[f32]) -> f32 {
///     incant!(process(data) with token, [v3, neon])
/// }
/// ```
///
/// The compiler monomorphizes the dispatch, eliminating non-matching branches.
///
/// # Variant Naming
///
/// Functions must have suffixed variants matching the selected tiers:
/// - `_v1` for `X64V1Token`
/// - `_v2` for `X64V2Token`
/// - `_v3` for `X64V3Token`
/// - `_v4` for `X64V4Token` (requires `avx512` feature)
/// - `_v4x` for `X64V4xToken` (requires `avx512` feature)
/// - `_neon` for `NeonToken`
/// - `_neon_aes` for `NeonAesToken`
/// - `_neon_sha3` for `NeonSha3Token`
/// - `_neon_crc` for `NeonCrcToken`
/// - `_wasm128` for `Wasm128Token`
/// - `_scalar` for `ScalarToken`
#[proc_macro]
pub fn incant(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as IncantInput);
    incant_impl(input)
}

/// Legacy alias for [`incant!`].
#[proc_macro]
pub fn simd_route(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as IncantInput);
    incant_impl(input)
}

/// Descriptive alias for [`incant!`].
///
/// Dispatches to architecture-specific function variants at runtime.
/// Looks for suffixed functions (`_v3`, `_v4`, `_neon`, `_wasm128`, `_scalar`)
/// and calls the best one the CPU supports.
///
/// See [`incant!`] for full documentation and examples.
#[proc_macro]
pub fn dispatch_variant(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as IncantInput);
    incant_impl(input)
}

fn incant_impl(input: IncantInput) -> TokenStream {
    let func_path = &input.func_path;
    let args = &input.args;

    // Resolve tiers
    let tier_names: Vec<String> = match &input.tiers {
        Some((names, _)) => names.clone(),
        None => DEFAULT_TIER_NAMES.iter().map(|s| s.to_string()).collect(),
    };
    let last_segment_span = func_path
        .segments
        .last()
        .map(|s| s.ident.span())
        .unwrap_or_else(proc_macro2::Span::call_site);
    let error_span = input
        .tiers
        .as_ref()
        .map(|(_, span)| *span)
        .unwrap_or(last_segment_span);

    let tiers = match resolve_tiers(&tier_names, error_span) {
        Ok(t) => t,
        Err(e) => return e.to_compile_error().into(),
    };

    // Group tiers by architecture for cfg-guarded blocks
    // Within each arch, tiers are already sorted by priority (highest first)
    if let Some(token_expr) = &input.with_token {
        gen_incant_passthrough(func_path, args, token_expr, &tiers)
    } else {
        gen_incant_entry(func_path, args, &tiers)
    }
}

/// Generate incant! passthrough mode (already have a token).
fn gen_incant_passthrough(
    func_path: &syn::Path,
    args: &[syn::Expr],
    token_expr: &syn::Expr,
    tiers: &[&TierDescriptor],
) -> TokenStream {
    let mut dispatch_arms = Vec::new();

    // Group non-scalar tiers by (target_arch, cargo_feature) for nested cfg blocks
    let mut arch_groups: Vec<(Option<&str>, Option<&str>, Vec<&TierDescriptor>)> = Vec::new();
    for tier in tiers {
        if tier.name == "scalar" {
            continue; // Handle scalar separately at the end
        }
        let key = (tier.target_arch, tier.cargo_feature);
        if let Some(group) = arch_groups.iter_mut().find(|(a, f, _)| (*a, *f) == key) {
            group.2.push(tier);
        } else {
            arch_groups.push((tier.target_arch, tier.cargo_feature, vec![tier]));
        }
    }

    for (target_arch, cargo_feature, group_tiers) in &arch_groups {
        let mut tier_checks = Vec::new();
        for tier in group_tiers {
            let fn_suffixed = suffix_path(func_path, tier.suffix);
            let as_method = format_ident!("{}", tier.as_method);
            tier_checks.push(quote! {
                if let Some(__t) = __incant_token.#as_method() {
                    break '__incant #fn_suffixed(__t, #(#args),*);
                }
            });
        }

        let inner = quote! { #(#tier_checks)* };

        let guarded = match (target_arch, cargo_feature) {
            (Some(arch), Some(feat)) => quote! {
                #[cfg(target_arch = #arch)]
                {
                    #[cfg(feature = #feat)]
                    { #inner }
                }
            },
            (Some(arch), None) => quote! {
                #[cfg(target_arch = #arch)]
                { #inner }
            },
            (None, Some(feat)) => quote! {
                #[cfg(feature = #feat)]
                { #inner }
            },
            (None, None) => inner,
        };

        dispatch_arms.push(guarded);
    }

    // Scalar fallback (always last)
    let fn_scalar = suffix_path(func_path, "scalar");
    let scalar_arm = if tiers.iter().any(|t| t.name == "scalar") {
        quote! {
            if let Some(__t) = __incant_token.as_scalar() {
                break '__incant #fn_scalar(__t, #(#args),*);
            }
            unreachable!("Token did not match any known variant")
        }
    } else {
        quote! { unreachable!("Token did not match any known variant") }
    };

    let expanded = quote! {
        '__incant: {
            use archmage::IntoConcreteToken;
            let __incant_token = #token_expr;
            #(#dispatch_arms)*
            #scalar_arm
        }
    };
    expanded.into()
}

/// Generate incant! entry point mode (summon tokens).
fn gen_incant_entry(
    func_path: &syn::Path,
    args: &[syn::Expr],
    tiers: &[&TierDescriptor],
) -> TokenStream {
    let mut dispatch_arms = Vec::new();

    // Group non-scalar tiers by target_arch for cfg blocks.
    // Within each arch group, further split by cargo_feature.
    let mut arch_groups: Vec<(Option<&str>, Vec<&TierDescriptor>)> = Vec::new();
    for tier in tiers {
        if tier.name == "scalar" {
            continue;
        }
        if let Some(group) = arch_groups.iter_mut().find(|(a, _)| *a == tier.target_arch) {
            group.1.push(tier);
        } else {
            arch_groups.push((tier.target_arch, vec![tier]));
        }
    }

    for (target_arch, group_tiers) in &arch_groups {
        let mut tier_checks = Vec::new();
        for tier in group_tiers {
            let fn_suffixed = suffix_path(func_path, tier.suffix);
            let token_path: syn::Path = syn::parse_str(tier.token_path).unwrap();

            let check = quote! {
                if let Some(__t) = #token_path::summon() {
                    break '__incant #fn_suffixed(__t, #(#args),*);
                }
            };

            if let Some(feat) = tier.cargo_feature {
                tier_checks.push(quote! {
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

    // Scalar fallback
    let fn_scalar = suffix_path(func_path, "scalar");

    let expanded = quote! {
        '__incant: {
            use archmage::SimdToken;
            #(#dispatch_arms)*
            #fn_scalar(archmage::ScalarToken, #(#args),*)
        }
    };
    expanded.into()
}

// =============================================================================
// Unit tests for token/trait recognition maps
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    use super::generated::{ALL_CONCRETE_TOKENS, ALL_TRAIT_NAMES};

    #[test]
    fn every_concrete_token_is_in_token_to_features() {
        for &name in ALL_CONCRETE_TOKENS {
            assert!(
                token_to_features(name).is_some(),
                "Token `{}` exists in runtime crate but is NOT recognized by \
                 token_to_features() in the proc macro. Add it!",
                name
            );
        }
    }

    #[test]
    fn every_trait_is_in_trait_to_features() {
        for &name in ALL_TRAIT_NAMES {
            assert!(
                trait_to_features(name).is_some(),
                "Trait `{}` exists in runtime crate but is NOT recognized by \
                 trait_to_features() in the proc macro. Add it!",
                name
            );
        }
    }

    #[test]
    fn token_aliases_map_to_same_features() {
        // Desktop64 = X64V3Token
        assert_eq!(
            token_to_features("Desktop64"),
            token_to_features("X64V3Token"),
            "Desktop64 and X64V3Token should map to identical features"
        );

        // Server64 = X64V4Token = Avx512Token
        assert_eq!(
            token_to_features("Server64"),
            token_to_features("X64V4Token"),
            "Server64 and X64V4Token should map to identical features"
        );
        assert_eq!(
            token_to_features("X64V4Token"),
            token_to_features("Avx512Token"),
            "X64V4Token and Avx512Token should map to identical features"
        );

        // Arm64 = NeonToken
        assert_eq!(
            token_to_features("Arm64"),
            token_to_features("NeonToken"),
            "Arm64 and NeonToken should map to identical features"
        );
    }

    #[test]
    fn trait_to_features_includes_tokens_as_bounds() {
        // Tier tokens should also work as trait bounds
        // (for `impl X64V3Token` patterns, even though Rust won't allow it,
        // the macro processes AST before type checking)
        let tier_tokens = [
            "X64V2Token",
            "X64CryptoToken",
            "X64V3Token",
            "Desktop64",
            "Avx2FmaToken",
            "X64V4Token",
            "Avx512Token",
            "Server64",
            "X64V4xToken",
            "Avx512Fp16Token",
            "NeonToken",
            "Arm64",
            "NeonAesToken",
            "NeonSha3Token",
            "NeonCrcToken",
            "Arm64V2Token",
            "Arm64V3Token",
        ];

        for &name in &tier_tokens {
            assert!(
                trait_to_features(name).is_some(),
                "Tier token `{}` should also be recognized in trait_to_features() \
                 for use as a generic bound. Add it!",
                name
            );
        }
    }

    #[test]
    fn trait_features_are_cumulative() {
        // HasX64V4 should include all HasX64V2 features plus more
        let v2_features = trait_to_features("HasX64V2").unwrap();
        let v4_features = trait_to_features("HasX64V4").unwrap();

        for &f in v2_features {
            assert!(
                v4_features.contains(&f),
                "HasX64V4 should include v2 feature `{}` but doesn't",
                f
            );
        }

        // v4 should have more features than v2
        assert!(
            v4_features.len() > v2_features.len(),
            "HasX64V4 should have more features than HasX64V2"
        );
    }

    #[test]
    fn x64v3_trait_features_include_v2() {
        // X64V3Token as trait bound should include v2 features
        let v2 = trait_to_features("HasX64V2").unwrap();
        let v3 = trait_to_features("X64V3Token").unwrap();

        for &f in v2 {
            assert!(
                v3.contains(&f),
                "X64V3Token trait features should include v2 feature `{}` but don't",
                f
            );
        }
    }

    #[test]
    fn has_neon_aes_includes_neon() {
        let neon = trait_to_features("HasNeon").unwrap();
        let neon_aes = trait_to_features("HasNeonAes").unwrap();

        for &f in neon {
            assert!(
                neon_aes.contains(&f),
                "HasNeonAes should include NEON feature `{}`",
                f
            );
        }
    }

    #[test]
    fn no_removed_traits_are_recognized() {
        // These traits were removed in 0.3.0 and should NOT be recognized
        let removed = [
            "HasSse",
            "HasSse2",
            "HasSse41",
            "HasSse42",
            "HasAvx",
            "HasAvx2",
            "HasFma",
            "HasAvx512f",
            "HasAvx512bw",
            "HasAvx512vl",
            "HasAvx512vbmi2",
            "HasSve",
            "HasSve2",
        ];

        for &name in &removed {
            assert!(
                trait_to_features(name).is_none(),
                "Removed trait `{}` should NOT be in trait_to_features(). \
                 It was removed in 0.3.0 — users should migrate to tier traits.",
                name
            );
        }
    }

    #[test]
    fn no_nonexistent_tokens_are_recognized() {
        // These tokens don't exist and should NOT be recognized
        let fake = [
            "SveToken",
            "Sve2Token",
            "Avx512VnniToken",
            "X64V4ModernToken",
            "NeonFp16Token",
        ];

        for &name in &fake {
            assert!(
                token_to_features(name).is_none(),
                "Non-existent token `{}` should NOT be in token_to_features()",
                name
            );
        }
    }

    #[test]
    fn featureless_traits_are_not_in_registries() {
        // SimdToken and IntoConcreteToken should NOT be in any feature registry
        // because they don't map to CPU features
        for &name in FEATURELESS_TRAIT_NAMES {
            assert!(
                token_to_features(name).is_none(),
                "`{}` should NOT be in token_to_features() — it has no CPU features",
                name
            );
            assert!(
                trait_to_features(name).is_none(),
                "`{}` should NOT be in trait_to_features() — it has no CPU features",
                name
            );
        }
    }

    #[test]
    fn find_featureless_trait_detects_simdtoken() {
        let names = vec!["SimdToken".to_string()];
        assert_eq!(find_featureless_trait(&names), Some("SimdToken"));

        let names = vec!["IntoConcreteToken".to_string()];
        assert_eq!(find_featureless_trait(&names), Some("IntoConcreteToken"));

        // Feature-bearing traits should NOT be detected
        let names = vec!["HasX64V2".to_string()];
        assert_eq!(find_featureless_trait(&names), None);

        let names = vec!["HasNeon".to_string()];
        assert_eq!(find_featureless_trait(&names), None);

        // Mixed: if SimdToken is among real traits, still detected
        let names = vec!["SimdToken".to_string(), "HasX64V2".to_string()];
        assert_eq!(find_featureless_trait(&names), Some("SimdToken"));
    }

    #[test]
    fn arm64_v2_v3_traits_are_cumulative() {
        let v2_features = trait_to_features("HasArm64V2").unwrap();
        let v3_features = trait_to_features("HasArm64V3").unwrap();

        for &f in v2_features {
            assert!(
                v3_features.contains(&f),
                "HasArm64V3 should include v2 feature `{}` but doesn't",
                f
            );
        }

        assert!(
            v3_features.len() > v2_features.len(),
            "HasArm64V3 should have more features than HasArm64V2"
        );
    }
}
