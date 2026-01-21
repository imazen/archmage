//! Proc-macros for archmage SIMD capability tokens.
//!
//! Provides `#[arcane]` attribute (with `#[simd_fn]` alias) to make raw intrinsics
//! safe via token proof.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    fold::Fold,
    parse::{Parse, ParseStream},
    parse_macro_input, parse_quote, Attribute, FnArg, GenericParam, Ident, ItemFn, PatType,
    ReturnType, Signature, Token, Type, TypeParamBound,
};

/// A Fold implementation that replaces `Self` with a concrete type.
struct ReplaceSelf<'a> {
    replacement: &'a Type,
}

impl Fold for ReplaceSelf<'_> {
    fn fold_type(&mut self, ty: Type) -> Type {
        match ty {
            Type::Path(ref type_path) if type_path.qself.is_none() => {
                // Check if it's just `Self`
                if type_path.path.is_ident("Self") {
                    return self.replacement.clone();
                }
                // Otherwise continue folding
                syn::fold::fold_type(self, ty)
            }
            _ => syn::fold::fold_type(self, ty),
        }
    }
}

/// Arguments to the `#[arcane]` macro.
#[derive(Default)]
struct ArcaneArgs {
    /// Use `#[inline(always)]` instead of `#[inline]` for the inner function.
    /// Requires nightly Rust with `#![feature(target_feature_inline_always)]`.
    inline_always: bool,
    /// The concrete type to use for `self` receiver.
    /// When specified, `self`/`&self`/`&mut self` is transformed to `_self: Type`/`&Type`/`&mut Type`.
    self_type: Option<Type>,
}

impl Parse for ArcaneArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut args = ArcaneArgs::default();

        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            match ident.to_string().as_str() {
                "inline_always" => args.inline_always = true,
                "_self" => {
                    let _: Token![=] = input.parse()?;
                    args.self_type = Some(input.parse()?);
                }
                other => {
                    return Err(syn::Error::new(
                        ident.span(),
                        format!("unknown arcane argument: `{}`", other),
                    ))
                }
            }
            // Consume optional comma
            if input.peek(Token![,]) {
                let _: Token![,] = input.parse()?;
            }
        }

        Ok(args)
    }
}

/// Maps a token type name to its required target features.
fn token_to_features(token_name: &str) -> Option<&'static [&'static str]> {
    match token_name {
        // x86_64 feature tokens (SSE4.2 is baseline)
        "Sse42Token" => Some(&["sse4.2"]),
        "AvxToken" => Some(&["avx"]),
        "Avx2Token" => Some(&["avx2"]),
        "Avx2FmaToken" => Some(&["avx2", "fma"]),
        "Avx512Token" => Some(&["avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"]),
        "Avx512ModernToken" => Some(&["avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl", "avx512vbmi2", "avx512vnni"]),
        "Avx512Fp16Token" => Some(&["avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl", "avx512fp16"]),

        // x86_64 profile tokens
        "X64V3Token" | "Desktop64" => Some(&["avx2", "fma", "bmi1", "bmi2"]),
        "X64V4Token" | "Server64" => {
            Some(&["avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"])
        }

        // ARM tokens
        "NeonToken" | "Arm64" => Some(&["neon"]),
        "NeonAesToken" => Some(&["neon", "aes"]),
        "NeonSha3Token" => Some(&["neon", "sha3"]),
        "NeonFp16Token" => Some(&["neon", "fp16"]),

        // WASM tokens
        "Simd128Token" => Some(&["simd128"]),

        _ => None,
    }
}

/// Maps a trait bound name to its required target features.
/// Used for `impl HasAvx2` and `T: HasAvx2` style parameters.
fn trait_to_features(trait_name: &str) -> Option<&'static [&'static str]> {
    match trait_name {
        // x86 feature marker traits
        "HasSse42" => Some(&["sse4.2"]),
        "HasAvx" => Some(&["avx"]),
        "HasAvx2" => Some(&["avx2"]),
        "HasAvx2Fma" => Some(&["avx2", "fma"]),
        "HasX64V3" | "HasDesktop64" => Some(&["avx2", "fma", "bmi2"]),
        "HasAvx512" | "HasX64V4" | "HasServer64" => Some(&["avx512f", "avx512cd", "avx512vl", "avx512dq", "avx512bw"]),
        "HasModernAvx512" => Some(&["avx512f", "avx512cd", "avx512vl", "avx512dq", "avx512bw", "avx512vbmi2", "avx512vnni"]),

        // Width marker traits
        "Has128BitSimd" => Some(&["sse2"]),
        "Has256BitSimd" => Some(&["avx"]),
        "Has512BitSimd" => Some(&["avx512f"]),

        // ARM feature marker traits
        "HasNeon" => Some(&["neon"]),
        "HasSve" => Some(&["sve"]),
        "HasSve2" => Some(&["sve2"]),
        "HasArm64" => Some(&["neon"]),
        "HasArmAes" => Some(&["neon", "aes"]),
        "HasArmSha3" => Some(&["neon", "sha3"]),
        "HasArmFp16" => Some(&["neon", "fp16"]),

        _ => None,
    }
}

/// Result of extracting token info from a type.
enum TokenTypeInfo {
    /// Concrete token type (e.g., `Avx2Token`)
    Concrete(String),
    /// impl Trait with the trait names (e.g., `impl HasAvx2`)
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
            // Handle `impl HasAvx2` or `impl HasAvx2Fma`
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
    // Check inline bounds first (e.g., `fn foo<T: HasAvx2>(token: T)`)
    for param in &sig.generics.params {
        if let GenericParam::Type(type_param) = param {
            if type_param.ident == type_name {
                let traits = extract_trait_names_from_bounds(&type_param.bounds);
                if !traits.is_empty() {
                    return Some(traits);
                }
            }
        }
    }

    // Check where clause (e.g., `fn foo<T>(token: T) where T: HasAvx2`)
    if let Some(where_clause) = &sig.generics.where_clause {
        for predicate in &where_clause.predicates {
            if let syn::WherePredicate::Type(pred_type) = predicate {
                if let Type::Path(type_path) = &pred_type.bounded_ty {
                    if let Some(seg) = type_path.path.segments.last() {
                        if seg.ident == type_name {
                            let traits = extract_trait_names_from_bounds(&pred_type.bounds);
                            if !traits.is_empty() {
                                return Some(traits);
                            }
                        }
                    }
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

/// Find the first token parameter and return its name and features.
fn find_token_param(sig: &Signature) -> Option<(Ident, Vec<&'static str>)> {
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
                    let features = match info {
                        TokenTypeInfo::Concrete(name) => {
                            token_to_features(&name).map(|f| f.to_vec())
                        }
                        TokenTypeInfo::ImplTrait(trait_names) => traits_to_features(&trait_names),
                        TokenTypeInfo::Generic(type_name) => {
                            // Look up the generic parameter's bounds
                            find_generic_bounds(sig, &type_name)
                                .and_then(|traits| traits_to_features(&traits))
                        }
                    };

                    if let Some(features) = features {
                        // Extract parameter name
                        if let syn::Pat::Ident(pat_ident) = pat.as_ref() {
                            return Some((pat_ident.ident.clone(), features));
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

/// Shared implementation for arcane/simd_fn macros.
fn arcane_impl(input_fn: ItemFn, macro_name: &str, args: ArcaneArgs) -> TokenStream {
    // Check for self receiver
    let has_self_receiver = input_fn
        .sig
        .inputs
        .first()
        .map(|arg| matches!(arg, FnArg::Receiver(_)))
        .unwrap_or(false);

    // If there's a self receiver, we need _self = Type
    if has_self_receiver && args.self_type.is_none() {
        let msg = format!(
            "{} with self receiver requires `_self = Type` argument.\n\
             Example: #[{}(_self = MyType)]\n\
             Use `_self` (not `self`) in the function body to refer to self.",
            macro_name, macro_name
        );
        return syn::Error::new_spanned(&input_fn.sig, msg)
            .to_compile_error()
            .into();
    }

    // Find the token parameter and its features
    let (_token_ident, features) = match find_token_param(&input_fn.sig) {
        Some(result) => result,
        None => {
            let msg = format!(
                "{} requires a token parameter. Supported forms:\n\
                 - Concrete: `token: Avx2Token`\n\
                 - impl Trait: `token: impl HasAvx2`\n\
                 - Generic: `fn foo<T: HasAvx2>(token: T, ...)`\n\
                 - With self: `#[{}(_self = Type)] fn method(&self, token: impl HasAvx2, ...)`",
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

    // Extract function components
    let vis = &input_fn.vis;
    let sig = &input_fn.sig;
    let fn_name = &sig.ident;
    let generics = &sig.generics;
    let where_clause = &generics.where_clause;
    let inputs = &sig.inputs;
    let output = &sig.output;
    let body = &input_fn.block;
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

    // Build inner function parameters, transforming self if needed
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
            FnArg::Typed(pat_type) => quote!(#pat_type),
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

    // Choose inline attribute based on args
    // Note: #[inline(always)] + #[target_feature] requires nightly with
    // #![feature(target_feature_inline_always)]
    let inline_attr: Attribute = if args.inline_always {
        parse_quote!(#[inline(always)])
    } else {
        parse_quote!(#[inline])
    };

    // Transform output and body to replace Self with concrete type if needed
    let (inner_output, inner_body): (ReturnType, syn::Block) =
        if let Some(ref self_ty) = args.self_type {
            let mut replacer = ReplaceSelf {
                replacement: self_ty,
            };
            let transformed_output = replacer.fold_return_type(output.clone());
            let transformed_body = replacer.fold_block((**body).clone());
            (transformed_output, transformed_body)
        } else {
            (output.clone(), (**body).clone())
        };

    // Generate the expanded function
    let expanded = quote! {
        #(#attrs)*
        #vis #sig {
            #(#target_feature_attrs)*
            #inline_attr
            unsafe fn #inner_fn_name #generics (#(#inner_params),*) #inner_output #where_clause
            #inner_body

            // SAFETY: The token parameter proves the required CPU features are available.
            // Tokens can only be constructed when features are verified (via try_new()
            // runtime check or forge_token_dangerously() in a context where features are guaranteed).
            unsafe { #inner_fn_name(#(#inner_args),*) }
        }
    };

    expanded.into()
}

/// Mark a function as an arcane SIMD function.
///
/// This macro enables safe use of SIMD intrinsics by generating an inner function
/// with the appropriate `#[target_feature(enable = "...")]` attributes based on
/// the token parameter type. The outer function calls the inner function unsafely,
/// which is justified because the token parameter proves the features are available.
///
/// **The token is passed through to the inner function**, so you can call other
/// token-taking functions from inside `#[arcane]`.
///
/// # Token Parameter Forms
///
/// The macro supports four forms of token parameters:
///
/// ## Concrete Token Types
///
/// ```ignore
/// #[arcane]
/// fn process(token: Avx2Token, data: &[f32; 8]) -> [f32; 8] {
///     // AVX2 intrinsics safe here
/// }
/// ```
///
/// ## impl Trait Bounds
///
/// ```ignore
/// #[arcane]
/// fn process(token: impl HasAvx2, data: &[f32; 8]) -> [f32; 8] {
///     // Accepts any token that provides AVX2
/// }
/// ```
///
/// ## Generic Type Parameters
///
/// ```ignore
/// #[arcane]
/// fn process<T: HasAvx2>(token: T, data: &[f32; 8]) -> [f32; 8] {
///     // Generic over any AVX2-capable token
/// }
///
/// // Also works with where clauses:
/// #[arcane]
/// fn process<T>(token: T, data: &[f32; 8]) -> [f32; 8]
/// where
///     T: HasAvx2
/// {
///     // ...
/// }
/// ```
///
/// ## Methods with Self Receivers
///
/// Methods with `self`, `&self`, `&mut self` receivers are supported via the
/// `_self = Type` argument. Use `_self` in the function body instead of `self`:
///
/// ```ignore
/// use archmage::{HasAvx2, arcane};
/// use wide::f32x8;
///
/// trait Avx2Ops {
///     fn double(&self, token: impl HasAvx2) -> Self;
///     fn square(self, token: impl HasAvx2) -> Self;
///     fn scale(&mut self, token: impl HasAvx2, factor: f32);
/// }
///
/// impl Avx2Ops for f32x8 {
///     #[arcane(_self = f32x8)]
///     fn double(&self, _token: impl HasAvx2) -> Self {
///         // Use _self instead of self in the body
///         *_self + *_self
///     }
///
///     #[arcane(_self = f32x8)]
///     fn square(self, _token: impl HasAvx2) -> Self {
///         _self * _self
///     }
///
///     #[arcane(_self = f32x8)]
///     fn scale(&mut self, _token: impl HasAvx2, factor: f32) {
///         *_self = *_self * f32x8::splat(factor);
///     }
/// }
/// ```
///
/// **Why `_self`?** The macro generates an inner function where `self` becomes
/// a regular parameter named `_self`. Using `_self` in your code reminds you
/// that you're not using the normal `self` keyword.
///
/// **All receiver types are supported:**
/// - `self` (by value/move) → `_self: Type`
/// - `&self` (shared reference) → `_self: &Type`
/// - `&mut self` (mutable reference) → `_self: &mut Type`
///
/// # FMA Trait
///
/// `HasAvx2Fma` implies `HasAvx2`, so you only need to specify one:
///
/// ```ignore
/// #[arcane]
/// fn fma_kernel(token: impl HasAvx2Fma, data: &[f32; 8]) -> [f32; 8] {
///     // Both AVX2 and FMA intrinsics are safe here
/// }
/// ```
///
/// # Expansion
///
/// The macro expands to approximately:
///
/// ```ignore
/// fn process(token: Avx2Token, data: &[f32; 8]) -> [f32; 8] {
///     #[target_feature(enable = "avx2")]
///     #[inline]
///     unsafe fn __simd_inner_process(token: Avx2Token, data: &[f32; 8]) -> [f32; 8] {
///         let v = unsafe { _mm256_loadu_ps(data.as_ptr()) };
///         let doubled = _mm256_add_ps(v, v);
///         let mut out = [0.0f32; 8];
///         unsafe { _mm256_storeu_ps(out.as_mut_ptr(), doubled) };
///         out
///     }
///     // SAFETY: Token proves the required features are available
///     unsafe { __simd_inner_process(token, data) }
/// }
/// ```
///
/// # Profile Tokens
///
/// Profile tokens automatically enable all required features:
///
/// ```ignore
/// #[arcane]
/// fn kernel(token: X64V3Token, data: &mut [f32]) {
///     // AVX2 + FMA + BMI1 + BMI2 intrinsics all safe here!
/// }
/// ```
///
/// # Supported Tokens
///
/// - **x86_64**: `Sse42Token`, `AvxToken`, `Avx2Token`, `Avx2FmaToken`,
///   `Avx512Token`, `Avx512ModernToken`, `Avx512Fp16Token`
/// - **x86_64 profiles**: `X64V3Token` (Desktop64), `X64V4Token` (Server64)
/// - **ARM**: `NeonToken`, `NeonAesToken`, `NeonSha3Token`, `NeonFp16Token`
/// - **WASM**: `Simd128Token`
///
/// # Supported Trait Bounds
///
/// - **x86_64**: `HasSse42`, `HasAvx`, `HasAvx2`, `HasAvx2Fma`,
///   `HasX64V3`, `HasDesktop64`, `HasAvx512`, `HasX64V4`, `HasServer64`, `HasModernAvx512`
/// - **ARM**: `HasNeon`, `HasArm64`, `HasArmAes`, `HasArmSha3`, `HasArmFp16`
/// - **Generic**: `Has128BitSimd`, `Has256BitSimd`, `Has512BitSimd`
///
/// # Options
///
/// ## `inline_always`
///
/// Use `#[inline(always)]` instead of `#[inline]` for the inner function.
/// This can improve performance by ensuring aggressive inlining, but requires
/// nightly Rust with `#![feature(target_feature_inline_always)]` enabled in
/// the crate using the macro.
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
    let input_fn = parse_macro_input!(item as ItemFn);
    arcane_impl(input_fn, "arcane", args)
}

/// Alias for [`arcane`] - mark a function as an arcane SIMD function.
///
/// See [`arcane`] for full documentation.
#[proc_macro_attribute]
pub fn simd_fn(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as ArcaneArgs);
    let input_fn = parse_macro_input!(item as ItemFn);
    arcane_impl(input_fn, "simd_fn", args)
}
