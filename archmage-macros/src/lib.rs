//! Proc-macros for archmage SIMD capability tokens.
//!
//! Provides `#[arcane]` attribute (with `#[simd_fn]` alias) to make raw intrinsics
//! safe via token proof.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, parse_quote, Attribute, FnArg, GenericParam, Ident, ItemFn, PatType,
    Signature, Token, Type, TypeParamBound,
};

/// Arguments to the `#[arcane]` macro.
#[derive(Default)]
struct ArcaneArgs {
    /// Use `#[inline(always)]` instead of `#[inline]` for the inner function.
    /// Requires nightly Rust with `#![feature(target_feature_inline_always)]`.
    inline_always: bool,
}

impl Parse for ArcaneArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut args = ArcaneArgs::default();

        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            match ident.to_string().as_str() {
                "inline_always" => args.inline_always = true,
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
        // x86_64 granular tokens
        "Sse2Token" => Some(&["sse2"]),
        "Sse41Token" => Some(&["sse4.1"]),
        "Sse42Token" => Some(&["sse4.2"]),
        "AvxToken" => Some(&["avx"]),
        "Avx2Token" => Some(&["avx2"]),
        "FmaToken" => Some(&["fma"]),
        "Avx2FmaToken" => Some(&["avx2", "fma"]),
        "Avx512fToken" => Some(&["avx512f"]),
        "Avx512bwToken" => Some(&["avx512bw"]),

        // x86_64 profile tokens
        "X64V2Token" => Some(&["sse4.2", "popcnt"]),
        "X64V3Token" | "Desktop64" => Some(&["avx2", "fma", "bmi1", "bmi2"]),
        "X64V4Token" | "Server64" => {
            Some(&["avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"])
        }

        // ARM tokens
        "NeonToken" | "Arm64" => Some(&["neon"]),
        "SveToken" => Some(&["sve"]),
        "Sve2Token" => Some(&["sve2"]),

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
        "HasSse" => Some(&["sse"]),
        "HasSse2" => Some(&["sse2"]),
        "HasSse41" => Some(&["sse4.1"]),
        "HasSse42" => Some(&["sse4.2"]),
        "HasAvx" => Some(&["avx"]),
        "HasAvx2" => Some(&["avx2"]),
        "HasAvx512f" => Some(&["avx512f"]),
        "HasAvx512vl" => Some(&["avx512f", "avx512vl"]),
        "HasAvx512bw" => Some(&["avx512bw"]),
        "HasAvx512vbmi2" => Some(&["avx512vbmi2"]),

        // Capability marker traits - use most specific features that satisfy them
        "HasFma" => Some(&["fma"]),
        "Has128BitSimd" => Some(&["sse2"]),
        "Has256BitSimd" => Some(&["avx"]),
        "Has512BitSimd" => Some(&["avx512f"]),

        // ARM feature marker traits
        "HasNeon" => Some(&["neon"]),
        "HasSve" => Some(&["sve"]),
        "HasSve2" => Some(&["sve2"]),

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

/// Shared implementation for arcane/simd_fn macros.
fn arcane_impl(input_fn: ItemFn, macro_name: &str, args: ArcaneArgs) -> TokenStream {
    // Find the token parameter and its features
    let (_token_ident, features) = match find_token_param(&input_fn.sig) {
        Some(result) => result,
        None => {
            let msg = format!(
                "{} requires a token parameter. Supported forms:\n\
                 - Concrete: `token: Avx2Token`\n\
                 - impl Trait: `token: impl HasAvx2`\n\
                 - Generic: `fn foo<T: HasAvx2>(token: T, ...)`\n\
                 Note: self receivers (&self, &mut self) are not yet supported.",
                macro_name
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

    // Build inner function parameters (ALL parameters including token)
    let inner_params: Vec<_> = inputs.iter().cloned().collect();

    // Build inner function call arguments (ALL arguments including token)
    let inner_args: Vec<_> = inputs
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
            FnArg::Receiver(_) => Some(quote!(self)),
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

    // Generate the expanded function
    let expanded = quote! {
        #(#attrs)*
        #vis #sig {
            #(#target_feature_attrs)*
            #inline_attr
            unsafe fn #inner_fn_name #generics (#(#inner_params),*) #output #where_clause
            #body

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
/// ## Methods with Self Receivers (NOT YET SUPPORTED)
///
/// Methods with `self`, `&self`, `&mut self` receivers are **not currently supported**.
///
/// **Why:** The macro works by creating an inner function with `#[target_feature]`.
/// Rust's inner functions cannot have `self` parameters—`self` only works in
/// associated functions. Supporting this would require rewriting the function body
/// to replace `self` with a regular parameter, which adds significant complexity.
///
/// **Workaround:** Use a free function with the token as an explicit parameter:
///
/// ```ignore
/// impl MyProcessor {
///     fn process(&mut self, data: &[f32; 8]) -> [f32; 8] {
///         // Delegate to a free function
///         process_impl(self.token, data)
///     }
/// }
///
/// #[arcane]
/// fn process_impl(token: impl HasAvx2, data: &[f32; 8]) -> [f32; 8] {
///     // SIMD intrinsics safe here
/// }
/// ```
///
/// **Future work:** Supporting `self` receivers would require:
/// 1. Adding a type parameter `__Self` to the inner function
/// 2. Converting the receiver to a regular parameter (`&self` → `__self: &__Self`)
/// 3. Walking the AST to replace all `self` with `__self` and `Self` with `__Self`
/// 4. Copying where clauses with the type substitution
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
