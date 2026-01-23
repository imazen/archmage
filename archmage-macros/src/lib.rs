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
///
/// Based on LLVM x86-64 microarchitecture levels (psABI).
fn token_to_features(token_name: &str) -> Option<&'static [&'static str]> {
    match token_name {
        // x86_64 feature tokens (kept for backwards compatibility)
        "Sse41Token" => Some(&["sse4.1"]),
        "Sse42Token" => Some(&["sse4.2"]),
        "AvxToken" => Some(&["avx"]),
        "Avx2Token" => Some(&["avx2"]),
        "FmaToken" => Some(&["fma"]),
        "Avx2FmaToken" => Some(&["avx2", "fma"]),
        "Avx512fToken" => Some(&["avx512f"]),
        "Avx512bwToken" => Some(&["avx512bw"]),

        // x86_64 tier tokens
        "X64V2Token" => Some(&["sse4.2", "popcnt"]),
        "X64V3Token" | "Desktop64" => Some(&["avx2", "fma", "bmi1", "bmi2"]),
        "X64V4Token" | "Avx512Token" | "Server64" => {
            Some(&["avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"])
        }
        "Avx512ModernToken" => Some(&[
            "avx512f",
            "avx512bw",
            "avx512cd",
            "avx512dq",
            "avx512vl",
            "avx512vpopcntdq",
            "avx512ifma",
            "avx512vbmi",
            "avx512vbmi2",
            "avx512bitalg",
            "avx512vnni",
            "avx512bf16",
            "vpclmulqdq",
            "gfni",
            "vaes",
        ]),
        "Avx512Fp16Token" => Some(&[
            "avx512f",
            "avx512bw",
            "avx512cd",
            "avx512dq",
            "avx512vl",
            "avx512fp16",
        ]),

        // AArch64 tokens
        "NeonToken" | "Arm64" => Some(&["neon"]),
        "NeonAesToken" => Some(&["neon", "aes"]),
        "NeonSha3Token" => Some(&["neon", "sha3"]),
        "ArmCryptoToken" => Some(&["neon", "aes", "sha2"]),
        "ArmCrypto3Token" => Some(&["neon", "aes", "sha2", "sha3"]),

        // WASM tokens
        "Simd128Token" => Some(&["simd128"]),

        _ => None,
    }
}

/// Maps a trait bound name to its required target features.
///
/// IMPORTANT: Each entry must include ALL implied features, not just the defining ones.
/// The compiler needs explicit #[target_feature] for each feature used.
///
/// Based on LLVM x86-64 microarchitecture levels (psABI).
fn trait_to_features(trait_name: &str) -> Option<&'static [&'static str]> {
    match trait_name {
        // x86 tier traits - each includes all features from lower tiers
        "HasX64V2" => Some(&["sse3", "ssse3", "sse4.1", "sse4.2", "popcnt"]),
        "HasX64V4" => Some(&[
            // v2 features
            "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", // v3 features
            "avx", "avx2", "fma", "bmi1", "bmi2", "f16c", "lzcnt", // v4 features
            "avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl",
        ]),

        // x86 token types - when used directly as bounds
        "X64V2Token" => Some(&["sse3", "ssse3", "sse4.1", "sse4.2", "popcnt"]),
        "X64V3Token" | "Desktop64" | "Avx2FmaToken" => Some(&[
            "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", "avx", "avx2", "fma", "bmi1", "bmi2",
            "f16c", "lzcnt",
        ]),
        "X64V4Token" | "Avx512Token" | "Server64" => Some(&[
            "sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", "avx", "avx2", "fma", "bmi1", "bmi2",
            "f16c", "lzcnt", "avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl",
        ]),
        "Avx512ModernToken" => Some(&[
            "sse3",
            "ssse3",
            "sse4.1",
            "sse4.2",
            "popcnt",
            "avx",
            "avx2",
            "fma",
            "bmi1",
            "bmi2",
            "f16c",
            "lzcnt",
            "avx512f",
            "avx512bw",
            "avx512cd",
            "avx512dq",
            "avx512vl",
            "avx512vpopcntdq",
            "avx512ifma",
            "avx512vbmi",
            "avx512vbmi2",
            "avx512bitalg",
            "avx512vnni",
            "avx512bf16",
            "vpclmulqdq",
            "gfni",
            "vaes",
        ]),
        "Avx512Fp16Token" => Some(&[
            "sse3",
            "ssse3",
            "sse4.1",
            "sse4.2",
            "popcnt",
            "avx",
            "avx2",
            "fma",
            "bmi1",
            "bmi2",
            "f16c",
            "lzcnt",
            "avx512f",
            "avx512bw",
            "avx512cd",
            "avx512dq",
            "avx512vl",
            "avx512fp16",
        ]),

        // Width traits - minimal features to satisfy width
        "Has128BitSimd" => Some(&["sse2"]),
        "Has256BitSimd" => Some(&["avx"]),
        "Has512BitSimd" => Some(&["avx512f"]),

        // AArch64 traits
        "HasNeon" => Some(&["neon"]),
        "HasNeonAes" => Some(&["neon", "aes"]),
        "HasNeonSha3" => Some(&["neon", "sha3"]),

        // AArch64 token types
        "NeonToken" | "Arm64" => Some(&["neon"]),
        "NeonAesToken" => Some(&["neon", "aes"]),
        "NeonSha3Token" => Some(&["neon", "sha3"]),
        "ArmCryptoToken" => Some(&["neon", "aes", "sha2"]),
        "ArmCrypto3Token" => Some(&["neon", "aes", "sha2", "sha3"]),

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
            // Handle `impl HasAvx2` or `impl HasAvx2 + HasFma`
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
/// # Multiple Trait Bounds
///
/// When using `impl Trait` or generic bounds with multiple traits,
/// all required features are enabled:
///
/// ```ignore
/// #[arcane]
/// fn fma_kernel(token: impl HasAvx2 + HasFma, data: &[f32; 8]) -> [f32; 8] {
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
/// - **x86_64**: `Sse2Token`, `Sse41Token`, `Sse42Token`, `AvxToken`, `Avx2Token`,
///   `FmaToken`, `Avx2FmaToken`, `Avx512fToken`, `Avx512bwToken`
/// - **x86_64 profiles**: `X64V2Token`, `X64V3Token`, `X64V4Token`
/// - **ARM**: `NeonToken`, `SveToken`, `Sve2Token`
/// - **WASM**: `Simd128Token`
///
/// # Supported Trait Bounds
///
/// - **x86_64**: `HasSse`, `HasSse2`, `HasSse41`, `HasSse42`, `HasAvx`, `HasAvx2`,
///   `HasAvx512f`, `HasAvx512vl`, `HasAvx512bw`, `HasAvx512vbmi2`, `HasFma`
/// - **ARM**: `HasNeon`, `HasSve`, `HasSve2`
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

// ============================================================================
// Multiwidth macro for width-agnostic SIMD code
// ============================================================================

use syn::ItemMod;

/// Arguments to the `#[multiwidth]` macro.
struct MultiwidthArgs {
    /// Include SSE (128-bit) specialization
    sse: bool,
    /// Include AVX2 (256-bit) specialization
    avx2: bool,
    /// Include AVX-512 (512-bit) specialization
    avx512: bool,
}

impl Default for MultiwidthArgs {
    fn default() -> Self {
        Self {
            sse: true,
            avx2: true,
            avx512: true,
        }
    }
}

impl Parse for MultiwidthArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut args = MultiwidthArgs {
            sse: false,
            avx2: false,
            avx512: false,
        };

        // If no args provided, enable all
        if input.is_empty() {
            return Ok(MultiwidthArgs::default());
        }

        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            match ident.to_string().as_str() {
                "sse" => args.sse = true,
                "avx2" => args.avx2 = true,
                "avx512" => args.avx512 = true,
                other => {
                    return Err(syn::Error::new(
                        ident.span(),
                        format!(
                            "unknown multiwidth target: `{}`. Expected: sse, avx2, avx512",
                            other
                        ),
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

/// Width configuration for specialization.
struct WidthConfig {
    /// Module name suffix (e.g., "sse", "avx2", "avx512")
    name: &'static str,
    /// The namespace import path
    namespace: &'static str,
    /// Token type name
    token: &'static str,
    /// Whether this requires a feature flag
    feature: Option<&'static str>,
    /// Target features to enable for this width
    target_features: &'static [&'static str],
}

const WIDTH_CONFIGS: &[WidthConfig] = &[
    WidthConfig {
        name: "sse",
        namespace: "archmage::simd::sse",
        token: "archmage::Sse41Token",
        feature: None,
        target_features: &["sse4.1"],
    },
    WidthConfig {
        name: "avx2",
        namespace: "archmage::simd::avx2",
        token: "archmage::Avx2FmaToken",
        feature: None,
        target_features: &["avx2", "fma"],
    },
    WidthConfig {
        name: "avx512",
        namespace: "archmage::simd::avx512",
        token: "archmage::X64V4Token",
        feature: Some("avx512"),
        target_features: &["avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"],
    },
];

/// Generate width-specialized SIMD code.
///
/// This macro takes a module containing width-agnostic SIMD code and generates
/// specialized versions for each target width (SSE, AVX2, AVX-512).
///
/// # Usage
///
/// ```ignore
/// use archmage::multiwidth;
///
/// #[multiwidth]
/// mod kernels {
///     // Inside this module, these types are available:
///     // - f32xN, i32xN, etc. (width-appropriate SIMD types)
///     // - Token (the token type: Sse41Token, Avx2FmaToken, or X64V4Token)
///     // - LANES_F32, LANES_32, etc. (lane count constants)
///
///     use archmage::simd::*;
///
///     pub fn normalize(token: Token, data: &mut [f32]) {
///         for chunk in data.chunks_exact_mut(LANES_F32) {
///             let v = f32xN::load(token, chunk.try_into().unwrap());
///             let result = v * f32xN::splat(token, 1.0 / 255.0);
///             result.store(chunk.try_into().unwrap());
///         }
///     }
/// }
///
/// // Generated modules:
/// // - kernels::sse::normalize(token: Sse41Token, data: &mut [f32])
/// // - kernels::avx2::normalize(token: Avx2FmaToken, data: &mut [f32])
/// // - kernels::avx512::normalize(token: X64V4Token, data: &mut [f32])  // if avx512 feature
/// // - kernels::normalize(data: &mut [f32])  // runtime dispatcher
/// ```
///
/// # Selective Targets
///
/// You can specify which targets to generate:
///
/// ```ignore
/// #[multiwidth(avx2, avx512)]  // Only AVX2 and AVX-512, no SSE
/// mod fast_kernels { ... }
/// ```
///
/// # How It Works
///
/// 1. The macro duplicates the module content for each width target
/// 2. Each copy imports from the appropriate namespace (`archmage::simd::sse`, etc.)
/// 3. The `use archmage::simd::*` statement is rewritten to the width-specific import
/// 4. A dispatcher function is generated that picks the best available at runtime
///
/// # Requirements
///
/// - Functions should use `Token` as their token parameter type
/// - Use `f32xN`, `i32xN`, etc. for SIMD types (not concrete types like `f32x8`)
/// - Use `LANES_F32`, `LANES_32`, etc. for lane counts
#[proc_macro_attribute]
pub fn multiwidth(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as MultiwidthArgs);
    let input_mod = parse_macro_input!(item as ItemMod);

    multiwidth_impl(input_mod, args)
}

fn multiwidth_impl(input_mod: ItemMod, args: MultiwidthArgs) -> TokenStream {
    let mod_name = &input_mod.ident;
    let mod_vis = &input_mod.vis;
    let mod_attrs = &input_mod.attrs;

    // Get module content
    let content = match &input_mod.content {
        Some((_, items)) => items,
        None => {
            return syn::Error::new_spanned(
                &input_mod,
                "multiwidth requires an inline module (mod name { ... }), not a file module",
            )
            .to_compile_error()
            .into();
        }
    };

    // Build specialized modules
    let mut specialized_mods = Vec::new();
    let mut enabled_configs = Vec::new();

    for config in WIDTH_CONFIGS {
        let enabled = match config.name {
            "sse" => args.sse,
            "avx2" => args.avx2,
            "avx512" => args.avx512,
            _ => false,
        };

        if !enabled {
            continue;
        }

        enabled_configs.push(config);

        let width_mod_name = format_ident!("{}", config.name);
        let namespace: syn::Path = syn::parse_str(config.namespace).unwrap();

        // Transform the content: replace `use archmage::simd::*` with width-specific import
        // and add target_feature attributes for optimization
        let transformed_items: Vec<syn::Item> = content
            .iter()
            .map(|item| transform_item_for_width(item.clone(), &namespace, config))
            .collect();

        let feature_attr = config.feature.map(|f| {
            let f_lit = syn::LitStr::new(f, proc_macro2::Span::call_site());
            quote!(#[cfg(feature = #f_lit)])
        });

        specialized_mods.push(quote! {
            #feature_attr
            pub mod #width_mod_name {
                #(#transformed_items)*
            }
        });
    }

    // Generate dispatcher functions for each public function in the module
    let dispatchers = generate_dispatchers(content, &enabled_configs);

    let expanded = quote! {
        #(#mod_attrs)*
        #mod_vis mod #mod_name {
            #(#specialized_mods)*

            #dispatchers
        }
    };

    expanded.into()
}

/// Transform a single item for a specific width namespace.
fn transform_item_for_width(
    item: syn::Item,
    namespace: &syn::Path,
    config: &WidthConfig,
) -> syn::Item {
    match item {
        syn::Item::Use(mut use_item) => {
            // Check if this is `use archmage::simd::*` or similar
            if is_simd_wildcard_use(&use_item) {
                // Replace with width-specific import
                use_item.tree = syn::UseTree::Path(syn::UsePath {
                    ident: format_ident!("{}", namespace.segments.first().unwrap().ident),
                    colon2_token: Default::default(),
                    tree: Box::new(build_use_tree_from_path(namespace, 1)),
                });
            }
            syn::Item::Use(use_item)
        }
        syn::Item::Fn(func) => {
            // Transform function to use inner function pattern with target_feature
            // This is the same pattern as #[arcane], enabling SIMD optimization
            // without requiring -C target-cpu=native
            transform_fn_with_target_feature(func, config)
        }
        other => other,
    }
}

/// Transform a function to use the inner function pattern with target_feature.
/// This generates:
/// ```ignore
/// pub fn example(token: Token, data: &[f32]) -> f32 {
///     #[target_feature(enable = "avx2", enable = "fma")]
///     #[inline]
///     unsafe fn inner(token: Token, data: &[f32]) -> f32 {
///         // original body
///     }
///     // SAFETY: Token proves CPU support
///     unsafe { inner(token, data) }
/// }
/// ```
fn transform_fn_with_target_feature(func: syn::ItemFn, config: &WidthConfig) -> syn::Item {
    let vis = &func.vis;
    let sig = &func.sig;
    let fn_name = &sig.ident;
    let generics = &sig.generics;
    let where_clause = &generics.where_clause;
    let inputs = &sig.inputs;
    let output = &sig.output;
    let body = &func.block;
    let attrs = &func.attrs;

    // Build target_feature attributes
    let target_feature_attrs: Vec<syn::Attribute> = config
        .target_features
        .iter()
        .map(|feature| parse_quote!(#[target_feature(enable = #feature)]))
        .collect();

    // Build parameter list for inner function
    let inner_params: Vec<proc_macro2::TokenStream> =
        inputs.iter().map(|arg| quote!(#arg)).collect();

    // Build argument list for calling inner function
    let call_args: Vec<proc_macro2::TokenStream> = inputs
        .iter()
        .filter_map(|arg| match arg {
            syn::FnArg::Typed(pat_type) => {
                if let syn::Pat::Ident(pat_ident) = pat_type.pat.as_ref() {
                    let ident = &pat_ident.ident;
                    Some(quote!(#ident))
                } else {
                    None
                }
            }
            syn::FnArg::Receiver(_) => Some(quote!(self)),
        })
        .collect();

    let inner_fn_name = format_ident!("__multiwidth_inner_{}", fn_name);

    let expanded = quote! {
        #(#attrs)*
        #vis #sig {
            #(#target_feature_attrs)*
            #[inline]
            unsafe fn #inner_fn_name #generics (#(#inner_params),*) #output #where_clause
            #body

            // SAFETY: The Token parameter proves the required CPU features are available.
            // Tokens can only be constructed via try_new() which checks CPU support.
            unsafe { #inner_fn_name(#(#call_args),*) }
        }
    };

    syn::parse2(expanded).expect("Failed to parse transformed function")
}

/// Check if a use item is `use archmage::simd::*` or `use crate::simd::*`.
fn is_simd_wildcard_use(use_item: &syn::ItemUse) -> bool {
    fn check_tree(tree: &syn::UseTree) -> bool {
        match tree {
            syn::UseTree::Path(path) => {
                let ident = path.ident.to_string();
                if ident == "archmage" || ident == "crate" {
                    check_tree_for_simd(&path.tree)
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    fn check_tree_for_simd(tree: &syn::UseTree) -> bool {
        match tree {
            syn::UseTree::Path(path) => {
                if path.ident == "simd" {
                    matches!(path.tree.as_ref(), syn::UseTree::Glob(_))
                } else {
                    check_tree_for_simd(&path.tree)
                }
            }
            _ => false,
        }
    }

    check_tree(&use_item.tree)
}

/// Build a UseTree from a path, starting at a given segment index.
fn build_use_tree_from_path(path: &syn::Path, start_idx: usize) -> syn::UseTree {
    let segments: Vec<_> = path.segments.iter().skip(start_idx).collect();

    if segments.is_empty() {
        syn::UseTree::Glob(syn::UseGlob {
            star_token: Default::default(),
        })
    } else if segments.len() == 1 {
        syn::UseTree::Path(syn::UsePath {
            ident: segments[0].ident.clone(),
            colon2_token: Default::default(),
            tree: Box::new(syn::UseTree::Glob(syn::UseGlob {
                star_token: Default::default(),
            })),
        })
    } else {
        let first = &segments[0];
        let rest_path = syn::Path {
            leading_colon: None,
            segments: path.segments.iter().skip(start_idx + 1).cloned().collect(),
        };
        syn::UseTree::Path(syn::UsePath {
            ident: first.ident.clone(),
            colon2_token: Default::default(),
            tree: Box::new(build_use_tree_from_path(&rest_path, 0)),
        })
    }
}

/// Width-specific type names that can't be used in dispatcher signatures.
const WIDTH_SPECIFIC_TYPES: &[&str] = &[
    "f32xN", "f64xN", "i8xN", "i16xN", "i32xN", "i64xN", "u8xN", "u16xN", "u32xN", "u64xN", "Token",
];

/// Check if a type string contains width-specific types.
fn contains_width_specific_type(ty_str: &str) -> bool {
    WIDTH_SPECIFIC_TYPES.iter().any(|t| ty_str.contains(t))
}

/// Check if a function signature uses width-specific types (can't have a dispatcher).
fn uses_width_specific_types(func: &syn::ItemFn) -> bool {
    // Check return type
    if let syn::ReturnType::Type(_, ty) = &func.sig.output {
        let ty_str = quote!(#ty).to_string();
        if contains_width_specific_type(&ty_str) {
            return true;
        }
    }

    // Check parameters (excluding Token which we filter out anyway)
    for arg in &func.sig.inputs {
        if let syn::FnArg::Typed(pat_type) = arg {
            let ty = &pat_type.ty;
            let ty_str = quote!(#ty).to_string();
            // Skip Token parameters - they're filtered out for dispatchers
            if ty_str.contains("Token") {
                continue;
            }
            if contains_width_specific_type(&ty_str) {
                return true;
            }
        }
    }

    false
}

/// Generate runtime dispatcher functions for public functions.
///
/// Note: Dispatchers are only generated for functions that don't use width-specific
/// types (f32xN, Token, etc.) in their signature. Functions that take/return
/// width-specific types can only be called via the width-specific submodules.
fn generate_dispatchers(
    content: &[syn::Item],
    configs: &[&WidthConfig],
) -> proc_macro2::TokenStream {
    let mut dispatchers = Vec::new();

    for item in content {
        if let syn::Item::Fn(func) = item {
            // Only generate dispatchers for public functions
            if !matches!(func.vis, syn::Visibility::Public(_)) {
                continue;
            }

            // Skip functions that use width-specific types - they can't have dispatchers
            if uses_width_specific_types(func) {
                continue;
            }

            let fn_name = &func.sig.ident;
            let fn_generics = &func.sig.generics;
            let fn_output = &func.sig.output;
            let fn_attrs: Vec<_> = func
                .attrs
                .iter()
                .filter(|a| !a.path().is_ident("arcane") && !a.path().is_ident("simd_fn"))
                .collect();

            // Filter out the token parameter from the dispatcher signature
            let non_token_params: Vec<_> = func
                .sig
                .inputs
                .iter()
                .filter(|arg| {
                    match arg {
                        syn::FnArg::Typed(pat_type) => {
                            // Check if type contains "Token"
                            let ty = &pat_type.ty;
                            let ty_str = quote!(#ty).to_string();
                            !ty_str.contains("Token")
                        }
                        _ => true,
                    }
                })
                .collect();

            // Extract just the parameter names for passing to specialized functions
            let param_names: Vec<_> = non_token_params
                .iter()
                .filter_map(|arg| match arg {
                    syn::FnArg::Typed(pat_type) => {
                        if let syn::Pat::Ident(pat_ident) = pat_type.pat.as_ref() {
                            Some(&pat_ident.ident)
                        } else {
                            None
                        }
                    }
                    _ => None,
                })
                .collect();

            // Generate dispatch branches (highest capability first)
            let mut branches = Vec::new();

            for config in configs.iter().rev() {
                let mod_name = format_ident!("{}", config.name);
                let token_path: syn::Path = syn::parse_str(config.token).unwrap();

                let feature_check = config.feature.map(|f| {
                    let f_lit = syn::LitStr::new(f, proc_macro2::Span::call_site());
                    quote!(#[cfg(feature = #f_lit)])
                });

                branches.push(quote! {
                    #feature_check
                    if let Some(token) = #token_path::try_new() {
                        return #mod_name::#fn_name(token, #(#param_names),*);
                    }
                });
            }

            // Generate dispatcher
            dispatchers.push(quote! {
                #(#fn_attrs)*
                /// Runtime dispatcher - automatically selects the best available SIMD implementation.
                pub fn #fn_name #fn_generics (#(#non_token_params),*) #fn_output {
                    use archmage::SimdToken;

                    #(#branches)*

                    // Fallback: panic if no SIMD available
                    // TODO: Allow user-provided scalar fallback
                    panic!("No SIMD support available for {}", stringify!(#fn_name));
                }
            });
        }
    }

    quote! { #(#dispatchers)* }
}
