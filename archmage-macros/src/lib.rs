//! Proc-macros for archmage SIMD capability tokens.
//!
//! Provides `#[arcane]` attribute (with `#[arcane]` alias) to make raw intrinsics
//! safe via token proof.

use proc_macro::TokenStream;
use quote::{format_ident, quote, ToTokens};
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
        if let GenericParam::Type(type_param) = param {
            if type_param.ident == type_name {
                let traits = extract_trait_names_from_bounds(&type_param.bounds);
                if !traits.is_empty() {
                    return Some(traits);
                }
            }
        }
    }

    // Check where clause (e.g., `fn foo<T>(token: T) where T: HasX64V2`)
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
                        // Extract parameter name
                        if let syn::Pat::Ident(pat_ident) = pat.as_ref() {
                            return Some(TokenParamInfo {
                                ident: pat_ident.ident.clone(),
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

    // Find the token parameter, its features, target arch, and token type name
    let TokenParamInfo {
        ident: _token_ident,
        features,
        target_arch,
        token_type_name,
    } = match find_token_param(&input_fn.sig) {
        Some(result) => result,
        None => {
            let msg = format!(
                "{} requires a token parameter. Supported forms:\n\
                 - Concrete: `token: X64V3Token`\n\
                 - impl Trait: `token: impl Has256BitSimd`\n\
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
    // If we know the target arch (concrete token), generate cfg-gated real impl + stub
    let token_type_str = token_type_name.as_deref().unwrap_or("UnknownToken");
    let expanded = if let Some(arch) = target_arch {
        quote! {
            // Real implementation for the correct architecture
            #[cfg(target_arch = #arch)]
            #(#attrs)*
            #vis #sig {
                #(#target_feature_attrs)*
                #inline_attr
                fn #inner_fn_name #generics (#(#inner_params),*) #inner_output #where_clause
                #inner_body

                // SAFETY: The token parameter proves the required CPU features are available.
                // Calling a #[target_feature] function from a non-matching context requires
                // unsafe because the CPU may not support those instructions. The token's
                // existence proves summon() succeeded, so the features are available.
                unsafe { #inner_fn_name(#(#inner_args),*) }
            }

            // Stub for other architectures - the token cannot be obtained, so this is unreachable
            #[cfg(not(target_arch = #arch))]
            #(#attrs)*
            #vis #sig {
                // This token type cannot be summoned on this architecture.
                // If you're seeing this at runtime, there's a bug in dispatch logic
                // or forge_token_dangerously() was used incorrectly.
                let _ = (#(#inner_args),*); // suppress unused warnings
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
        // No specific arch (trait bounds or generic) - generate without cfg guards
        quote! {
            #(#attrs)*
            #vis #sig {
                #(#target_feature_attrs)*
                #inline_attr
                fn #inner_fn_name #generics (#(#inner_params),*) #inner_output #where_clause
                #inner_body

                // SAFETY: Calling a #[target_feature] function from a non-matching context
                // requires unsafe. The token proves the required CPU features are available.
                unsafe { #inner_fn_name(#(#inner_args),*) }
            }
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
/// fn process(token: impl HasX64V2, data: &[f32; 8]) -> [f32; 8] {
///     // Accepts any token with x86-64-v2 features (SSE4.2+)
/// }
/// ```
///
/// ## Generic Type Parameters
///
/// ```ignore
/// #[arcane]
/// fn process<T: HasX64V2>(token: T, data: &[f32; 8]) -> [f32; 8] {
///     // Generic over any v2-capable token
/// }
///
/// // Also works with where clauses:
/// #[arcane]
/// fn process<T>(token: T, data: &[f32; 8]) -> [f32; 8]
/// where
///     T: HasX64V2
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
/// use archmage::{X64V3Token, arcane};
/// use wide::f32x8;
///
/// trait SimdOps {
///     fn double(&self, token: X64V3Token) -> Self;
///     fn square(self, token: X64V3Token) -> Self;
///     fn scale(&mut self, token: X64V3Token, factor: f32);
/// }
///
/// impl SimdOps for f32x8 {
///     #[arcane(_self = f32x8)]
///     fn double(&self, _token: X64V3Token) -> Self {
///         // Use _self instead of self in the body
///         *_self + *_self
///     }
///
///     #[arcane(_self = f32x8)]
///     fn square(self, _token: X64V3Token) -> Self {
///         _self * _self
///     }
///
///     #[arcane(_self = f32x8)]
///     fn scale(&mut self, _token: X64V3Token, factor: f32) {
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
/// fn fma_kernel(token: impl HasX64V2 + HasNeon, data: &[f32; 8]) -> [f32; 8] {
///     // Cross-platform: SSE4.2 on x86, NEON on ARM
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
///     fn __simd_inner_process(token: Avx2Token, data: &[f32; 8]) -> [f32; 8] {
///         let v = unsafe { _mm256_loadu_ps(data.as_ptr()) };
///         let doubled = _mm256_add_ps(v, v);
///         let mut out = [0.0f32; 8];
///         unsafe { _mm256_storeu_ps(out.as_mut_ptr(), doubled) };
///         out
///     }
///     // SAFETY: Calling #[target_feature] fn from non-matching context.
///     // Token proves the required features are available.
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
/// - **x86_64 tiers**: `X64V2Token`, `X64V3Token` / `Desktop64` / `Avx2FmaToken`,
///   `X64V4Token` / `Avx512Token` / `Server64`, `X64V4xToken`, `Avx512Fp16Token`
/// - **ARM**: `NeonToken` / `Arm64`, `NeonAesToken`, `NeonSha3Token`, `NeonCrcToken`
/// - **WASM**: `Wasm128Token`
///
/// # Supported Trait Bounds
///
/// - **x86_64 tiers**: `HasX64V2`, `HasX64V4`
/// - **ARM**: `HasNeon`, `HasNeonAes`, `HasNeonSha3`
///
/// **Preferred:** Use concrete tokens (`X64V3Token`, `Desktop64`, `NeonToken`) directly.
/// Concrete token types also work as trait bounds (e.g., `impl X64V3Token`).
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

/// Legacy alias for [`arcane`].
///
/// **Deprecated:** Use `#[arcane]` instead. This alias exists only for migration.
#[proc_macro_attribute]
#[doc(hidden)]
pub fn simd_fn(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as ArcaneArgs);
    let input_fn = parse_macro_input!(item as ItemFn);
    arcane_impl(input_fn, "simd_fn", args)
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
/// # Comparison with #[arcane]
///
/// | Aspect | `#[arcane]` | `#[rite]` |
/// |--------|-------------|-----------|
/// | Creates wrapper | Yes | No |
/// | Entry point | Yes | No |
/// | Inlines into caller | No (barrier) | Yes |
/// | Safe to call anywhere | Yes (with token) | Only from feature-enabled context |
#[proc_macro_attribute]
pub fn rite(attr: TokenStream, item: TokenStream) -> TokenStream {
    // Parse optional arguments (currently just inline_always)
    let args = parse_macro_input!(attr as RiteArgs);
    let input_fn = parse_macro_input!(item as ItemFn);
    rite_impl(input_fn, args)
}

/// Arguments for the `#[rite]` macro.
///
/// Currently empty - `#[inline(always)]` is not supported because
/// `#[inline(always)]` + `#[target_feature]` requires nightly Rust.
/// The regular `#[inline]` hint is sufficient when called from
/// matching `#[target_feature]` contexts.
#[derive(Default)]
struct RiteArgs {
    // No options currently - inline_always doesn't work on stable
}

impl Parse for RiteArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        if !input.is_empty() {
            let ident: Ident = input.parse()?;
            return Err(syn::Error::new(
                ident.span(),
                "#[rite] takes no arguments. Note: inline_always is not supported \
                 because #[inline(always)] + #[target_feature] requires nightly Rust.",
            ));
        }
        Ok(RiteArgs::default())
    }
}

/// Implementation for the `#[rite]` macro.
fn rite_impl(mut input_fn: ItemFn, args: RiteArgs) -> TokenStream {
    // Find the token parameter and its features
    let TokenParamInfo {
        features,
        target_arch,
        ..
    } = match find_token_param(&input_fn.sig) {
        Some(result) => result,
        None => {
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
    let _ = args; // RiteArgs is currently empty but kept for future extensibility
    let inline_attr: Attribute = parse_quote!(#[inline]);

    // Prepend attributes to the function
    let mut new_attrs = target_feature_attrs;
    new_attrs.push(inline_attr);
    new_attrs.append(&mut input_fn.attrs);
    input_fn.attrs = new_attrs;

    // If we know the target arch, generate cfg-gated impl + stub
    if let Some(arch) = target_arch {
        let vis = &input_fn.vis;
        let sig = &input_fn.sig;
        let attrs = &input_fn.attrs;
        let block = &input_fn.block;

        quote! {
            #[cfg(target_arch = #arch)]
            #(#attrs)*
            #vis #sig
            #block

            #[cfg(not(target_arch = #arch))]
            #vis #sig {
                unreachable!(concat!(
                    "This function requires ",
                    #arch,
                    " architecture"
                ))
            }
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
/// `neon_sha3`, `neon_crc`, `wasm128`, `scalar`.
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
    let input_fn = parse_macro_input!(item as ItemFn);

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

fn magetypes_impl(mut input_fn: ItemFn, tiers: &[&TierDescriptor]) -> TokenStream {
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
        name: "v3",
        suffix: "v3",
        token_path: "archmage::X64V3Token",
        as_method: "as_x64v3",
        target_arch: Some("x86_64"),
        cargo_feature: None,
        priority: 30,
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
    /// Function name to call
    func_name: Ident,
    /// Arguments to pass
    args: Vec<syn::Expr>,
    /// Optional token variable for passthrough mode
    with_token: Option<syn::Expr>,
    /// Optional explicit tier list (None = default tiers)
    tiers: Option<(Vec<String>, proc_macro2::Span)>,
}

impl Parse for IncantInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // Parse: function_name(arg1, arg2, ...) [with token_expr] [, [tier1, tier2, ...]]
        let func_name: Ident = input.parse()?;

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
            func_name,
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
/// `neon_sha3`, `neon_crc`, `wasm128`, `scalar`.
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

fn incant_impl(input: IncantInput) -> TokenStream {
    let func_name = &input.func_name;
    let args = &input.args;

    // Resolve tiers
    let tier_names: Vec<String> = match &input.tiers {
        Some((names, _)) => names.clone(),
        None => DEFAULT_TIER_NAMES.iter().map(|s| s.to_string()).collect(),
    };
    let error_span = input
        .tiers
        .as_ref()
        .map(|(_, span)| *span)
        .unwrap_or_else(|| func_name.span());

    let tiers = match resolve_tiers(&tier_names, error_span) {
        Ok(t) => t,
        Err(e) => return e.to_compile_error().into(),
    };

    // Group tiers by architecture for cfg-guarded blocks
    // Within each arch, tiers are already sorted by priority (highest first)
    if let Some(token_expr) = &input.with_token {
        gen_incant_passthrough(func_name, args, token_expr, &tiers)
    } else {
        gen_incant_entry(func_name, args, &tiers)
    }
}

/// Generate incant! passthrough mode (already have a token).
fn gen_incant_passthrough(
    func_name: &Ident,
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
            let fn_suffixed = format_ident!("{}_{}", func_name, tier.suffix);
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
    let fn_scalar = format_ident!("{}_scalar", func_name);
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
    func_name: &Ident,
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
            let fn_suffixed = format_ident!("{}_{}", func_name, tier.suffix);
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
    let fn_scalar = format_ident!("{}_scalar", func_name);

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
}
