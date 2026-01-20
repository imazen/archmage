//! Proc-macros for archmage SIMD capability tokens.
//!
//! Provides `#[arcane]` attribute (with `#[simd_fn]` alias) to make raw intrinsics
//! safe via token proof.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse::{Parse, ParseStream},
    parse_macro_input, parse_quote, Attribute, FnArg, Ident, ItemFn, PatType, Signature, Token,
    Type,
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
        "X64V3Token" => Some(&["avx2", "fma", "bmi1", "bmi2"]),
        "X64V4Token" => Some(&["avx512f", "avx512bw", "avx512cd", "avx512dq", "avx512vl"]),

        // ARM tokens
        "NeonToken" => Some(&["neon"]),
        "SveToken" => Some(&["sve"]),
        "Sve2Token" => Some(&["sve2"]),

        // WASM tokens
        "Simd128Token" => Some(&["simd128"]),

        _ => None,
    }
}

/// Extract the token type name from a type.
fn extract_token_type_name(ty: &Type) -> Option<String> {
    match ty {
        Type::Path(type_path) => {
            // Get the last segment of the path (e.g., "Avx2Token" from "archmage::Avx2Token")
            type_path
                .path
                .segments
                .last()
                .map(|seg| seg.ident.to_string())
        }
        Type::Reference(type_ref) => {
            // Handle &Token or &mut Token
            extract_token_type_name(&type_ref.elem)
        }
        _ => None,
    }
}

/// Find the first token parameter and return its name and features.
fn find_token_param(sig: &Signature) -> Option<(Ident, Vec<&'static str>)> {
    for arg in &sig.inputs {
        if let FnArg::Typed(PatType { pat, ty, .. }) = arg {
            if let Some(token_name) = extract_token_type_name(ty) {
                if let Some(features) = token_to_features(&token_name) {
                    // Extract parameter name
                    if let syn::Pat::Ident(pat_ident) = pat.as_ref() {
                        return Some((pat_ident.ident.clone(), features.to_vec()));
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
                "{} requires a token parameter (e.g., `token: Avx2Token`)",
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
/// # Example
///
/// ```ignore
/// use archmage::{Avx2Token, arcane};
/// use std::arch::x86_64::*;
///
/// #[arcane]
/// fn process(token: Avx2Token, data: &[f32; 8]) -> [f32; 8] {
///     // Raw intrinsics are safe here - token proves AVX2 is available!
///     let v = unsafe { _mm256_loadu_ps(data.as_ptr()) };
///     let doubled = _mm256_add_ps(v, v);  // SAFE - value operation
///     let mut out = [0.0f32; 8];
///     unsafe { _mm256_storeu_ps(out.as_mut_ptr(), doubled) };
///     out
/// }
/// ```
///
/// This expands to approximately:
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
