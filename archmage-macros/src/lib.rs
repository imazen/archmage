//! Proc-macros for archmage SIMD capability tokens.
//!
//! Provides `#[arcane]`, `#[rite]`, `#[autoversion]`, `incant!`, and `#[magetypes]`.

mod common;
mod generated;
mod tiers;

use proc_macro::TokenStream;
use quote::{ToTokens, format_ident, quote, quote_spanned};
use syn::{
    Attribute, FnArg, GenericParam, Ident, PatType, Signature, Token, Type, TypeParamBound,
    parse::{Parse, ParseStream},
    parse_macro_input, parse_quote,
};

use common::*;
use tiers::*;

// LightFn, filter_inline_attrs, is_lint_attr, filter_lint_attrs, gen_cfg_guard,
// build_turbofish, replace_self_in_tokens, suffix_path → moved to common.rs

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
    /// Inject `use archmage::intrinsics::{arch}::*;` (includes safe memory ops).
    import_intrinsics: bool,
    /// Inject `use magetypes::simd::{ns}::*;`, `use magetypes::simd::generic::*;`,
    /// and `use magetypes::simd::backends::*;`.
    import_magetypes: bool,
    /// Additional cargo feature gate. When set, the generated `#[cfg(target_arch)]`
    /// becomes `#[cfg(all(target_arch = "...", feature = "..."))]`.
    /// Example: `#[arcane(cfg(avx512))]` → `#[cfg(all(target_arch = "x86_64", feature = "avx512"))]`
    cfg_feature: Option<String>,
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

// Token-to-features and trait-to-features mappings are generated from
// token-registry.toml by xtask. Regenerate with: cargo run -p xtask -- generate
use generated::{
    canonical_token_to_tier_suffix, tier_to_canonical_token, token_to_arch, token_to_features,
    token_to_magetypes_namespace, trait_to_arch, trait_to_features, trait_to_magetypes_namespace,
};

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
    /// Magetypes width namespace (e.g., "v3", "neon", "wasm128")
    magetypes_namespace: Option<&'static str>,
}

/// Resolve magetypes namespace from a list of trait names.
/// Returns the first matching namespace found.
fn traits_to_magetypes_namespace(trait_names: &[String]) -> Option<&'static str> {
    for name in trait_names {
        if let Some(ns) = trait_to_magetypes_namespace(name) {
            return Some(ns);
        }
    }
    None
}

/// Given trait bound names, return the first matching target architecture.
fn traits_to_arch(trait_names: &[String]) -> Option<&'static str> {
    for name in trait_names {
        if let Some(arch) = trait_to_arch(name) {
            return Some(arch);
        }
    }
    None
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
                    let (features, arch, token_name, mage_ns) = match info {
                        TokenTypeInfo::Concrete(ref name) => {
                            let features = token_to_features(name).map(|f| f.to_vec());
                            let arch = token_to_arch(name);
                            let ns = token_to_magetypes_namespace(name);
                            (features, arch, Some(name.clone()), ns)
                        }
                        TokenTypeInfo::ImplTrait(ref trait_names) => {
                            let ns = traits_to_magetypes_namespace(trait_names);
                            let arch = traits_to_arch(trait_names);
                            (traits_to_features(trait_names), arch, None, ns)
                        }
                        TokenTypeInfo::Generic(type_name) => {
                            // Look up the generic parameter's bounds
                            let bounds = find_generic_bounds(sig, &type_name);
                            let features = bounds.as_ref().and_then(|t| traits_to_features(t));
                            let ns = bounds
                                .as_ref()
                                .and_then(|t| traits_to_magetypes_namespace(t));
                            let arch = bounds.as_ref().and_then(|t| traits_to_arch(t));
                            (features, arch, None, ns)
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
                                magetypes_namespace: mage_ns,
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

/// Generate import statements to prepend to a function body.
///
/// Returns a `TokenStream` of `use` statements based on the import flags,
/// target architecture, and magetypes namespace.
fn generate_imports(
    target_arch: Option<&str>,
    magetypes_namespace: Option<&str>,
    import_intrinsics: bool,
    import_magetypes: bool,
) -> proc_macro2::TokenStream {
    let mut imports = proc_macro2::TokenStream::new();

    if import_intrinsics && let Some(arch) = target_arch {
        let arch_ident = format_ident!("{}", arch);
        imports.extend(quote! {
            #[allow(unused_imports)]
            use archmage::intrinsics::#arch_ident::*;
        });
        // ScalarToken or unknown arch: import_intrinsics is a no-op
    }

    if import_magetypes && let Some(ns) = magetypes_namespace {
        let ns_ident = format_ident!("{}", ns);
        imports.extend(quote! {
            #[allow(unused_imports)]
            use magetypes::simd::#ns_ident::*;
            #[allow(unused_imports)]
            use magetypes::simd::backends::*;
        });
    }

    imports
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

/// Mark a function as an arcane SIMD function.
///
/// This macro generates a safe wrapper around a `#[target_feature]` function.
/// The token parameter type determines which CPU features are enabled.
///
/// # Expansion Modes
///
/// ## Sibling (default)
///
/// Generates two functions at the same scope: a safe `#[target_feature]` sibling
/// and a safe wrapper. `self`/`Self` work naturally since both functions share scope.
/// Compatible with `#![forbid(unsafe_code)]`.
///
/// ```ignore
/// #[arcane]
/// fn process(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] { /* body */ }
/// // Expands to (x86_64 only):
/// #[cfg(target_arch = "x86_64")]
/// #[doc(hidden)]
/// #[target_feature(enable = "avx2,fma,...")]
/// fn __arcane_process(token: X64V3Token, data: &[f32; 8]) -> [f32; 8] { /* body */ }
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
/// | `import_intrinsics` | Auto-import `archmage::intrinsics::{arch}::*` (includes safe memory ops) |
/// | `import_magetypes` | Auto-import `magetypes::simd::{ns}::*` and `magetypes::simd::backends::*` |
///
/// ## Auto-Imports
///
/// `import_intrinsics` and `import_magetypes` inject `use` statements into the
/// function body, eliminating boilerplate. The macro derives the architecture and
/// namespace from the token type:
///
/// ```ignore
/// // Without auto-imports — lots of boilerplate:
/// use std::arch::x86_64::*;
/// use magetypes::simd::v3::*;
///
/// #[arcane]
/// fn process(token: X64V3Token, data: &[f32; 8]) -> f32 {
///     let v = f32x8::load(token, data);
///     let zero = _mm256_setzero_ps();
///     // ...
/// }
///
/// // With auto-imports — clean:
/// #[arcane(import_intrinsics, import_magetypes)]
/// fn process(token: X64V3Token, data: &[f32; 8]) -> f32 {
///     let v = f32x8::load(token, data);
///     let zero = _mm256_setzero_ps();
///     // ...
/// }
/// ```
///
/// The namespace mapping is token-driven:
///
/// | Token | `import_intrinsics` | `import_magetypes` |
/// |-------|--------------------|--------------------|
/// | `X64V1..V3Token` | `archmage::intrinsics::x86_64::*` | `magetypes::simd::v3::*` |
/// | `X64V4Token` | `archmage::intrinsics::x86_64::*` | `magetypes::simd::v4::*` |
/// | `X64V4xToken` | `archmage::intrinsics::x86_64::*` | `magetypes::simd::v4x::*` |
/// | `NeonToken` / ARM | `archmage::intrinsics::aarch64::*` | `magetypes::simd::neon::*` |
/// | `Wasm128Token` | `archmage::intrinsics::wasm32::*` | `magetypes::simd::wasm128::*` |
///
/// Works with concrete tokens, `impl Trait` bounds, and generic parameters.
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
/// # Three Modes
///
/// **Token-based:** Reads the token type from the function signature.
/// ```ignore
/// #[rite]
/// fn helper(_: X64V3Token, v: __m256) -> __m256 { _mm256_add_ps(v, v) }
/// ```
///
/// **Tier-based:** Specify the tier name directly, no token parameter needed.
/// ```ignore
/// #[rite(v3)]
/// fn helper(v: __m256) -> __m256 { _mm256_add_ps(v, v) }
/// ```
///
/// Both produce identical code. The token form can be easier to remember if
/// you already have the token in scope.
///
/// **Multi-tier:** Specify multiple tiers to generate suffixed variants.
/// ```ignore
/// #[rite(v3, v4)]
/// fn process(data: &[f32; 4]) -> f32 { data.iter().sum() }
/// // Generates: process_v3() and process_v4()
/// ```
///
/// Each variant gets its own `#[target_feature]` and `#[cfg(target_arch)]`.
/// Since Rust 1.85, calling these from a matching `#[arcane]` or `#[rite]`
/// context is safe — no `unsafe` needed when the caller has matching or
/// superset features.
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
/// # Options
///
/// | Option | Effect |
/// |--------|--------|
/// | tier name(s) | `v3`, `neon`, etc. One = single function; multiple = suffixed variants |
/// | `stub` | Generate `unreachable!()` stub on wrong architecture |
/// | `import_intrinsics` | Auto-import `archmage::intrinsics::{arch}::*` (includes safe memory ops) |
/// | `import_magetypes` | Auto-import `magetypes::simd::{ns}::*` and `magetypes::simd::backends::*` |
///
/// See `#[arcane]` docs for the full namespace mapping table.
///
/// # Comparison with #[arcane]
///
/// | Aspect | `#[arcane]` | `#[rite]` |
/// |--------|-------------|-----------|
/// | Creates wrapper | Yes | No |
/// | Entry point | Yes | No |
/// | Inlines into caller | No (barrier) | Yes |
/// | Safe to call anywhere | Yes (with token) | Only from feature-enabled context |
/// | Multi-tier variants | No | Yes (`#[rite(v3, v4, neon)]`) |
/// | `stub` param | Yes | Yes |
/// | `import_intrinsics` | Yes | Yes |
/// | `import_magetypes` | Yes | Yes |
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
    /// Inject `use archmage::intrinsics::{arch}::*;` (includes safe memory ops).
    import_intrinsics: bool,
    /// Inject `use magetypes::simd::{ns}::*;`, `use magetypes::simd::generic::*;`,
    /// and `use magetypes::simd::backends::*;`.
    import_magetypes: bool,
    /// Tiers specified directly (e.g., `#[rite(v3)]` or `#[rite(v3, v4, neon)]`).
    /// Stored as canonical token names (e.g., "X64V3Token").
    /// Single tier: generates one function (no suffix, no token parameter needed).
    /// Multiple tiers: generates suffixed variants (e.g., `fn_v3`, `fn_v4`, `fn_neon`).
    tier_tokens: Vec<String>,
    /// Additional cargo feature gate (same as arcane's cfg_feature).
    cfg_feature: Option<String>,
}

impl Parse for RiteArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut args = RiteArgs::default();

        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            match ident.to_string().as_str() {
                "stub" => args.stub = true,
                "import_intrinsics" => args.import_intrinsics = true,
                "import_magetypes" => args.import_magetypes = true,
                "cfg" => {
                    let content;
                    syn::parenthesized!(content in input);
                    let feat: Ident = content.parse()?;
                    args.cfg_feature = Some(feat.to_string());
                }
                other => {
                    if let Some(canonical) = tier_to_canonical_token(other) {
                        args.tier_tokens.push(String::from(canonical));
                    } else {
                        return Err(syn::Error::new(
                            ident.span(),
                            format!(
                                "unknown rite argument: `{}`. Supported: tier names \
                                 (v1, v2, v3, v4, neon, arm_v2, wasm128, ...), \
                                 `stub`, `import_intrinsics`, `import_magetypes`, `cfg(feature)`.",
                                other
                            ),
                        ));
                    }
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
fn rite_impl(input_fn: LightFn, args: RiteArgs) -> TokenStream {
    // Multi-tier mode: generate suffixed variants for each tier
    if args.tier_tokens.len() > 1 {
        return rite_multi_tier_impl(input_fn, &args);
    }

    // Single-tier or token-param mode
    rite_single_impl(input_fn, args)
}

/// Generate a single `#[rite]` function (single tier or token-param mode).
fn rite_single_impl(mut input_fn: LightFn, args: RiteArgs) -> TokenStream {
    // Resolve features: either from tier name or from token parameter
    let TokenParamInfo {
        features,
        target_arch,
        token_type_name: _token_type_name,
        magetypes_namespace,
        ..
    } = if let Some(tier_token) = args.tier_tokens.first() {
        // Tier specified directly (e.g., #[rite(v3)]) — no token param needed
        let features = token_to_features(tier_token)
            .expect("tier_to_canonical_token returned invalid token name")
            .to_vec();
        let target_arch = token_to_arch(tier_token);
        let magetypes_namespace = token_to_magetypes_namespace(tier_token);
        TokenParamInfo {
            ident: Ident::new("_", proc_macro2::Span::call_site()),
            features,
            target_arch,
            token_type_name: Some(tier_token.clone()),
            magetypes_namespace,
        }
    } else {
        match find_token_param(&input_fn.sig) {
            Some(result) => result,
            None => {
                // Check for specific misuse: featureless traits like SimdToken
                if let Some(trait_name) = diagnose_featureless_token(&input_fn.sig) {
                    let msg = format!(
                        "`{trait_name}` cannot be used as a token bound in #[rite] \
                         because it doesn't specify any CPU features.\n\
                         \n\
                         #[rite] needs concrete features to generate #[target_feature]. \
                         Use a concrete token, a feature trait, or a tier name:\n\
                         \n\
                         Concrete tokens: X64V3Token, Desktop64, NeonToken, Arm64V2Token, ...\n\
                         Feature traits:  impl HasX64V2, impl HasNeon, impl HasArm64V3, ...\n\
                         Tier names:      #[rite(v3)], #[rite(neon)], #[rite(v4)], ..."
                    );
                    return syn::Error::new_spanned(&input_fn.sig, msg)
                        .to_compile_error()
                        .into();
                }
                let msg = "rite requires a token parameter or a tier name. Supported forms:\n\
                     - Tier name: `#[rite(v3)]`, `#[rite(neon)]`\n\
                     - Multi-tier: `#[rite(v3, v4, neon)]` (generates suffixed variants)\n\
                     - Concrete: `token: X64V3Token`\n\
                     - impl Trait: `token: impl HasX64V2`\n\
                     - Generic: `fn foo<T: HasX64V2>(token: T, ...)`";
                return syn::Error::new_spanned(&input_fn.sig, msg)
                    .to_compile_error()
                    .into();
            }
        }
    };

    // Check: import_intrinsics with AVX-512 features requires the avx512 cargo feature.
    // Check resolved features (not token name) for uniform handling of concrete/trait/generic.
    #[cfg(not(feature = "avx512"))]
    if args.import_intrinsics && features.iter().any(|f| f.starts_with("avx512")) {
        let token_desc = _token_type_name.as_deref().unwrap_or("an AVX-512 token");
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

    // Build target_feature attributes
    let target_feature_attrs: Vec<Attribute> = features
        .iter()
        .map(|feature| parse_quote!(#[target_feature(enable = #feature)]))
        .collect();

    // Always use #[inline] - #[inline(always)] + #[target_feature] requires nightly
    let inline_attr: Attribute = parse_quote!(#[inline]);

    // Prepend attributes to the function, filtering user #[inline] to avoid duplicates
    let mut new_attrs = target_feature_attrs;
    new_attrs.push(inline_attr);
    for attr in filter_inline_attrs(&input_fn.attrs) {
        new_attrs.push(attr.clone());
    }
    input_fn.attrs = new_attrs;

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

    // If we know the target arch, generate cfg-gated impl (+ optional stub)
    let cfg_guard = gen_cfg_guard(target_arch, args.cfg_feature.as_deref());
    if target_arch.is_some() {
        let vis = &input_fn.vis;
        let sig = &input_fn.sig;
        let attrs = &input_fn.attrs;
        let body = &input_fn.body;

        let stub = if args.stub {
            let not_cfg = match (target_arch, args.cfg_feature.as_deref()) {
                (Some(arch), Some(feat)) => {
                    quote! { #[cfg(not(all(target_arch = #arch, feature = #feat)))] }
                }
                (Some(arch), None) => quote! { #[cfg(not(target_arch = #arch))] },
                _ => quote! {},
            };
            quote! {
                #not_cfg
                #vis #sig {
                    unreachable!("This function requires a specific architecture and feature set")
                }
            }
        } else {
            quote! {}
        };

        quote! {
            #cfg_guard
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

/// Generate multiple suffixed `#[rite]` variants for multi-tier mode.
///
/// `#[rite(v3, v4, neon)]` on `fn process(...)` generates:
/// - `fn process_v3(...)` with `#[target_feature(enable = "avx2,fma,...")]`
/// - `fn process_v4(...)` with `#[target_feature(enable = "avx512f,...")]`
/// - `fn process_neon(...)` with `#[target_feature(enable = "neon")]`
///
/// Each variant is cfg-gated to its architecture and gets `#[inline]`.
fn rite_multi_tier_impl(input_fn: LightFn, args: &RiteArgs) -> TokenStream {
    let fn_name = &input_fn.sig.ident;
    let mut variants = proc_macro2::TokenStream::new();

    for tier_token in &args.tier_tokens {
        let features = match token_to_features(tier_token) {
            Some(f) => f,
            None => {
                return syn::Error::new_spanned(
                    &input_fn.sig,
                    format!("unknown token `{tier_token}` in multi-tier #[rite]"),
                )
                .to_compile_error()
                .into();
            }
        };
        let target_arch = token_to_arch(tier_token);
        let magetypes_namespace = token_to_magetypes_namespace(tier_token);

        // Check: import_intrinsics with AVX-512 features requires the avx512 cargo feature.
        #[cfg(not(feature = "avx512"))]
        if args.import_intrinsics && features.iter().any(|f| f.starts_with("avx512")) {
            let msg = format!(
                "Using {tier_token} with `import_intrinsics` requires the `avx512` feature.\n\
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

        let suffix = canonical_token_to_tier_suffix(tier_token)
            .expect("canonical token must have a tier suffix");

        // Build suffixed function name
        let suffixed_ident = format_ident!("{}_{}", fn_name, suffix);

        // Clone and rename the function
        let mut variant_fn = input_fn.clone();
        variant_fn.sig.ident = suffixed_ident;

        // Build target_feature attributes
        let target_feature_attrs: Vec<Attribute> = features
            .iter()
            .map(|feature| parse_quote!(#[target_feature(enable = #feature)]))
            .collect();
        let inline_attr: Attribute = parse_quote!(#[inline]);

        let mut new_attrs = target_feature_attrs;
        new_attrs.push(inline_attr);
        for attr in filter_inline_attrs(&variant_fn.attrs) {
            new_attrs.push(attr.clone());
        }
        variant_fn.attrs = new_attrs;

        // Prepend import statements if requested
        let body_imports = generate_imports(
            target_arch,
            magetypes_namespace,
            args.import_intrinsics,
            args.import_magetypes,
        );
        if !body_imports.is_empty() {
            let original_body = &variant_fn.body;
            variant_fn.body = quote! {
                #body_imports
                #original_body
            };
        }

        // Emit cfg-gated variant
        let variant_cfg = gen_cfg_guard(target_arch, args.cfg_feature.as_deref());
        if target_arch.is_some() {
            let vis = &variant_fn.vis;
            let sig = &variant_fn.sig;
            let attrs = &variant_fn.attrs;
            let body = &variant_fn.body;

            variants.extend(quote! {
                #variant_cfg
                #(#attrs)*
                #vis #sig {
                    #body
                }
            });

            if args.stub {
                let not_cfg = match (target_arch, args.cfg_feature.as_deref()) {
                    (Some(arch), Some(feat)) => {
                        quote! { #[cfg(not(all(target_arch = #arch, feature = #feat)))] }
                    }
                    (Some(arch), None) => quote! { #[cfg(not(target_arch = #arch))] },
                    _ => quote! {},
                };
                let arch_str = target_arch.unwrap_or("unknown");
                variants.extend(quote! {
                    #not_cfg
                    #vis #sig {
                        unreachable!(concat!(
                            "This function requires ",
                            #arch_str,
                            " architecture"
                        ))
                    }
                });
            }
        } else {
            // No specific arch — just emit the annotated function
            variants.extend(quote!(#variant_fn));
        }
    }

    variants.into()
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
///     incant!(process(data), [v1, v3, neon, scalar])
/// }
/// ```
#[proc_macro_attribute]
pub fn magetypes(attr: TokenStream, item: TokenStream) -> TokenStream {
    let input_fn = parse_macro_input!(item as LightFn);

    // Parse optional tier list from attribute args: tier1, tier2(feature), ...
    let tier_names: Vec<String> = if attr.is_empty() {
        DEFAULT_TIER_NAMES.iter().map(|s| s.to_string()).collect()
    } else {
        match syn::parse::Parser::parse(parse_tier_names, attr) {
            Ok(names) => names,
            Err(e) => return e.to_compile_error().into(),
        }
    };

    // default_optional: tiers with cfg_feature are optional by default
    let tiers = match resolve_tiers(
        &tier_names,
        input_fn.sig.ident.span(),
        true, // magetypes always uses default_optional for cfg_feature tiers
    ) {
        Ok(t) => t,
        Err(e) => return e.to_compile_error().into(),
    };

    magetypes_impl(input_fn, &tiers)
}

fn magetypes_impl(mut input_fn: LightFn, tiers: &[ResolvedTier]) -> TokenStream {
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

        // Add cfg guard: arch + optional feature gate
        let allow_attr = if tier.allow_unexpected_cfg {
            quote! { #[allow(unexpected_cfgs)] }
        } else {
            quote! {}
        };
        let cfg_guard = match (tier.target_arch, &tier.feature_gate) {
            (Some(arch), Some(feat)) => quote! {
                #[cfg(target_arch = #arch)]
                #allow_attr
                #[cfg(feature = #feat)]
            },
            (Some(arch), None) => quote! { #[cfg(target_arch = #arch)] },
            (None, Some(feat)) => quote! {
                #allow_attr
                #[cfg(feature = #feat)]
            },
            (None, None) => quote! {},
        };

        variants.push(if tier.name != "scalar" && tier.name != "default" {
            // Non-fallback variants get #[arcane] so target_feature is applied
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

// suffix_path → moved to common.rs

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

        // Check for optional tier list: , [tier1, tier2(feature), ...]
        // tier(feature) wraps dispatch in #[cfg(feature = "feature")].
        // Example: [v4(avx512), v3, neon(simd), scalar]
        let tiers = if input.peek(Token![,]) {
            let _: Token![,] = input.parse()?;
            let bracket_content;
            let bracket = syn::bracketed!(bracket_content in input);
            let mut tier_names = Vec::new();
            while !bracket_content.is_empty() {
                let ident: Ident = bracket_content.parse()?;
                let name = if bracket_content.peek(syn::token::Paren) {
                    // Parse tier(feature) — feature gate syntax
                    let paren_content;
                    syn::parenthesized!(paren_content in bracket_content);
                    let feat: Ident = paren_content.parse()?;
                    format!("{}({})", ident, feat)
                } else {
                    ident.to_string()
                };
                tier_names.push(name);
                if bracket_content.peek(Token![,]) {
                    let _: Token![,] = bracket_content.parse()?;
                }
            }
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
///     incant!(process(data), [v1, v3, neon, scalar])
/// }
/// ```
///
/// Always include `scalar` in explicit tier lists — `incant!` always
/// emits a `fn_scalar()` call as the final fallback, and listing it
/// documents this dependency. Currently auto-appended if omitted;
/// will become a compile error in v1.0. Unknown tier names cause a
/// compile error. Tiers are automatically sorted into correct
/// dispatch order (highest priority first).
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
///     incant!(process(data) with token, [v3, neon, scalar])
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

    // When the user specifies explicit tiers without `scalar` or `default` in
    // incant!, emit a deprecation warning. In v1.0 this will become a compile error.
    // A fallback tier is always auto-appended, but not listing it explicitly
    // hides the fact that a _scalar/_default function is required.
    let scalar_warning = if let Some((names, _span)) = &input.tiers {
        if !names.iter().any(|n| {
            let base = n.split('(').next().unwrap_or(n);
            base == "scalar" || base == "default"
        }) {
            quote! {
                #[deprecated(since = "0.9.9", note = "\
                    explicit incant! tier lists should include `scalar`. \
                    incant! always calls fn_scalar() as the final fallback. \
                    This will become a compile error in v1.0. \
                    Example: incant!(foo(x), [v3, neon, scalar])")]
                #[allow(non_upper_case_globals)]
                const __incant_missing_scalar_tier: () = ();
                let _ = __incant_missing_scalar_tier;
            }
        } else {
            quote! {}
        }
    } else {
        quote! {}
    };

    // Apply default feature gates: tiers with cfg_feature (v4→avx512) auto-get
    // the gate unless the user explicitly wrote tier(feature). This is true for
    // BOTH default and explicit tier lists — backwards compatible with published
    // crates using [v4, v3, neon] where _v4 is behind #[cfg(feature = "avx512")].
    // Users with unconditional _v4 functions use v4(!) or just don't cfg-gate them.
    let tiers = match resolve_tiers(&tier_names, error_span, true) {
        Ok(t) => t,
        Err(e) => return e.to_compile_error().into(),
    };

    // Group tiers by architecture for cfg-guarded blocks
    // Within each arch, tiers are already sorted by priority (highest first)
    let dispatch: TokenStream = if let Some(token_expr) = &input.with_token {
        gen_incant_passthrough(func_path, args, token_expr, &tiers)
    } else {
        gen_incant_entry(func_path, args, &tiers)
    };

    if scalar_warning.is_empty() {
        dispatch
    } else {
        // Wrap dispatch in a block that includes the deprecation warning
        let dispatch2: proc_macro2::TokenStream = dispatch.into();
        quote! {
            {
                #scalar_warning
                #dispatch2
            }
        }
        .into()
    }
}

/// Generate incant! passthrough mode (already have a token).
fn gen_incant_passthrough(
    func_path: &syn::Path,
    args: &[syn::Expr],
    token_expr: &syn::Expr,
    tiers: &[ResolvedTier],
) -> TokenStream {
    let mut dispatch_arms = Vec::new();

    // Group non-fallback tiers by target_arch for cfg blocks
    let mut arch_groups: Vec<(Option<&str>, Vec<&ResolvedTier>)> = Vec::new();
    for rt in tiers {
        if rt.name == "scalar" || rt.name == "default" {
            continue; // Handle fallback separately at the end
        }
        if let Some(group) = arch_groups.iter_mut().find(|(a, _)| *a == rt.target_arch) {
            group.1.push(rt);
        } else {
            arch_groups.push((rt.target_arch, vec![rt]));
        }
    }

    for (target_arch, group_tiers) in &arch_groups {
        let mut tier_checks = Vec::new();
        for rt in group_tiers {
            let fn_suffixed = suffix_path(func_path, rt.suffix);
            let as_method = format_ident!("{}", rt.as_method);

            let check = quote! {
                if let Some(__t) = __incant_token.#as_method() {
                    break '__incant #fn_suffixed(__t, #(#args),*);
                }
            };

            if let Some(feat) = &rt.feature_gate {
                let allow_attr = if rt.allow_unexpected_cfg {
                    quote! { #[allow(unexpected_cfgs)] }
                } else {
                    quote! {}
                };
                tier_checks.push(quote! {
                    #allow_attr
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

    // Fallback (always last): scalar (with token) or default (tokenless)
    let has_default = tiers.iter().any(|t| t.name == "default");
    let fallback_arm = if has_default {
        let fn_default = suffix_path(func_path, "default");
        quote! {
            break '__incant #fn_default(#(#args),*);
        }
    } else if tiers.iter().any(|t| t.name == "scalar") {
        let fn_scalar = suffix_path(func_path, "scalar");
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
            #fallback_arm
        }
    };
    expanded.into()
}

/// Generate incant! entry point mode (summon tokens).
fn gen_incant_entry(
    func_path: &syn::Path,
    args: &[syn::Expr],
    tiers: &[ResolvedTier],
) -> TokenStream {
    let mut dispatch_arms = Vec::new();

    // Group non-fallback tiers by target_arch for cfg blocks.
    let mut arch_groups: Vec<(Option<&str>, Vec<&ResolvedTier>)> = Vec::new();
    for rt in tiers {
        if rt.name == "scalar" || rt.name == "default" {
            continue;
        }
        if let Some(group) = arch_groups.iter_mut().find(|(a, _)| *a == rt.target_arch) {
            group.1.push(rt);
        } else {
            arch_groups.push((rt.target_arch, vec![rt]));
        }
    }

    for (target_arch, group_tiers) in &arch_groups {
        let mut tier_checks = Vec::new();
        for rt in group_tiers {
            let fn_suffixed = suffix_path(func_path, rt.suffix);
            let token_path: syn::Path = syn::parse_str(rt.token_path).unwrap();

            let check = quote! {
                if let Some(__t) = #token_path::summon() {
                    break '__incant #fn_suffixed(__t, #(#args),*);
                }
            };

            if let Some(feat) = &rt.feature_gate {
                let allow_attr = if rt.allow_unexpected_cfg {
                    quote! { #[allow(unexpected_cfgs)] }
                } else {
                    quote! {}
                };
                tier_checks.push(quote! {
                    #allow_attr
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

    // Fallback: scalar (with ScalarToken) or default (tokenless)
    let has_default = tiers.iter().any(|rt| rt.name == "default");
    let fallback_call = if has_default {
        let fn_default = suffix_path(func_path, "default");
        quote! { #fn_default(#(#args),*) }
    } else {
        let fn_scalar = suffix_path(func_path, "scalar");
        quote! { #fn_scalar(archmage::ScalarToken, #(#args),*) }
    };

    let expanded = quote! {
        '__incant: {
            use archmage::SimdToken;
            #(#dispatch_arms)*
            #fallback_call
        }
    };
    expanded.into()
}

// =============================================================================
// autoversion - combined variant generation + dispatch
// =============================================================================

/// Arguments to the `#[autoversion]` macro.
struct AutoversionArgs {
    /// The concrete type to use for `self` receiver (inherent methods only).
    self_type: Option<Type>,
    /// Explicit tier names (None = default tiers).
    tiers: Option<Vec<String>>,
    /// When set, emit full autoversion under `#[cfg(feature = "...")]` and a
    /// plain scalar fallback under `#[cfg(not(feature = "..."))]`. Solves the
    /// hygiene issue with `macro_rules!` wrappers.
    cfg_feature: Option<String>,
}

impl Parse for AutoversionArgs {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut self_type = None;
        let mut tier_names = Vec::new();
        let mut cfg_feature = None;

        while !input.is_empty() {
            let ident: Ident = input.parse()?;
            if ident == "_self" {
                let _: Token![=] = input.parse()?;
                self_type = Some(input.parse()?);
            } else if ident == "cfg" {
                let content;
                syn::parenthesized!(content in input);
                let feat: Ident = content.parse()?;
                cfg_feature = Some(feat.to_string());
            } else {
                // Treat as tier name, optionally with (feature) gate
                let name = if input.peek(syn::token::Paren) {
                    let paren_content;
                    syn::parenthesized!(paren_content in input);
                    let feat: Ident = paren_content.parse()?;
                    format!("{}({})", ident, feat)
                } else {
                    ident.to_string()
                };
                tier_names.push(name);
            }
            if input.peek(Token![,]) {
                let _: Token![,] = input.parse()?;
            }
        }

        Ok(AutoversionArgs {
            self_type,
            tiers: if tier_names.is_empty() {
                None
            } else {
                Some(tier_names)
            },
            cfg_feature,
        })
    }
}

/// What kind of token parameter was found in the autoversion function signature.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AutoversionTokenKind {
    /// `SimdToken` — legacy placeholder, stripped from dispatcher (deprecated).
    SimdToken,
    /// `ScalarToken` — real type, kept in dispatcher for incant! compatibility.
    ScalarToken,
    /// No token found — auto-injected internally, stripped from dispatcher.
    AutoInjected,
}

/// Information about the token parameter in an autoversion function signature.
#[derive(Debug)]
struct AutoversionTokenParam {
    /// Index of the parameter in `sig.inputs`
    index: usize,
    /// The parameter identifier
    #[allow(dead_code)]
    ident: Ident,
    /// What kind of token was found
    kind: AutoversionTokenKind,
}

/// Find a token parameter (`SimdToken` or `ScalarToken`) in a function signature
/// for `#[autoversion]`.
///
/// Returns Ok(Some) for recognized tokens, Ok(None) for no token, or Err for
/// concrete SIMD tokens (X64V3Token etc.) which should use `#[arcane]` instead.
fn find_autoversion_token_param(
    sig: &Signature,
) -> Result<Option<AutoversionTokenParam>, syn::Error> {
    for (i, arg) in sig.inputs.iter().enumerate() {
        if let FnArg::Typed(PatType { pat, ty, .. }) = arg
            && let Type::Path(type_path) = ty.as_ref()
            && let Some(seg) = type_path.path.segments.last()
        {
            let name = seg.ident.to_string();

            // Recognized autoversion tokens
            let kind = if name == "SimdToken" {
                AutoversionTokenKind::SimdToken
            } else if name == "ScalarToken" {
                AutoversionTokenKind::ScalarToken
            } else if token_to_features(&name).is_some() {
                // It's a concrete SIMD token (X64V3Token, NeonToken, etc.)
                return Err(syn::Error::new_spanned(
                    ty,
                    format!(
                        "#[autoversion] generates multi-tier dispatch — it can't take a \
                         concrete token like `{name}`.\n\
                         Use #[arcane] or #[rite] for single-token functions.\n\
                         Use #[autoversion] with no token parameter (recommended) or \
                         ScalarToken for incant! nesting."
                    ),
                ));
            } else {
                continue;
            };

            let ident = match pat.as_ref() {
                syn::Pat::Ident(pi) => pi.ident.clone(),
                syn::Pat::Wild(w) => Ident::new("__autoversion_token", w.underscore_token.span),
                _ => continue,
            };
            return Ok(Some(AutoversionTokenParam {
                index: i,
                ident,
                kind,
            }));
        }
    }
    Ok(None)
}

/// Core implementation for `#[autoversion]`.
///
/// Generates suffixed SIMD variants (like `#[magetypes]`) and a runtime
/// dispatcher function (like `incant!`) from a single annotated function.
fn autoversion_impl(mut input_fn: LightFn, args: AutoversionArgs) -> TokenStream {
    // Check for self receiver
    let has_self = input_fn
        .sig
        .inputs
        .first()
        .is_some_and(|arg| matches!(arg, FnArg::Receiver(_)));

    // _self = Type is only needed for trait impls (nested mode in #[arcane]).
    // For inherent methods, self/Self work naturally in sibling mode.

    // Find token parameter (SimdToken or ScalarToken), or auto-inject one.
    //
    // Three modes:
    // - ScalarToken: kept in dispatcher (real type, compiles, incant!-compatible)
    // - SimdToken: stripped from dispatcher (legacy, deprecated)
    // - None: auto-inject internally, strip from dispatcher (tokenless)
    let token_param = match find_autoversion_token_param(&input_fn.sig) {
        Err(e) => return e.to_compile_error().into(),
        Ok(Some(p)) => p,
        Ok(None) => {
            let insert_pos = if has_self { 1 } else { 0 };
            let token_arg: FnArg = parse_quote!(_token: SimdToken);
            input_fn.sig.inputs.insert(insert_pos, token_arg);
            AutoversionTokenParam {
                index: insert_pos,
                ident: Ident::new("_token", input_fn.sig.ident.span()),
                kind: AutoversionTokenKind::AutoInjected,
            }
        }
    };

    // Deprecation warning for SimdToken. We emit a function-local deprecation
    // by referencing a deprecated item inside the dispatcher body.
    let simdtoken_deprecation_in_body = if token_param.kind == AutoversionTokenKind::SimdToken {
        let msg = "SimdToken parameter in #[autoversion] is deprecated — \
                   remove it (tokenless) or use ScalarToken for incant! nesting";
        Some(quote! {
            {
                #[deprecated(note = #msg)]
                #[allow(dead_code)]
                const SIMDTOKEN_DEPRECATED: () = ();
                let _ = SIMDTOKEN_DEPRECATED;
            }
        })
    } else {
        None
    };

    // Whether to keep the token param in the dispatcher.
    // ScalarToken is a real type → keep it (incant! compatibility).
    // SimdToken and AutoInjected → strip (can't compile / internal).
    let keep_token_in_dispatcher = token_param.kind == AutoversionTokenKind::ScalarToken;

    // Resolve tiers — autoversion always includes v4 in its defaults because it
    // generates scalar code compiled with #[target_feature], not import_intrinsics.
    let tier_names: Vec<String> = match &args.tiers {
        Some(names) => names.clone(),
        None => DEFAULT_TIER_NAMES.iter().map(|s| s.to_string()).collect(),
    };
    // autoversion never skips avx512 — it generates scalar code with #[target_feature]
    let tiers = match resolve_tiers(&tier_names, input_fn.sig.ident.span(), false) {
        Ok(t) => t,
        Err(e) => return e.to_compile_error().into(),
    };

    // Strip #[arcane] / #[rite] to prevent double-wrapping
    input_fn
        .attrs
        .retain(|attr| !attr.path().is_ident("arcane") && !attr.path().is_ident("rite"));

    let fn_name = &input_fn.sig.ident;
    let vis = input_fn.vis.clone();

    // Move attrs to dispatcher only; variants get no user attrs
    let fn_attrs: Vec<Attribute> = input_fn.attrs.drain(..).collect();

    // =========================================================================
    // Generate suffixed variants
    // =========================================================================
    //
    // AST manipulation only — we clone the parsed LightFn and swap the token
    // param's type annotation. No serialize/reparse round-trip. The body is
    // never touched unless _self = Type requires a `let _self = self;`
    // preamble on the scalar variant.

    let mut variants = Vec::new();

    for tier in &tiers {
        let mut variant_fn = input_fn.clone();

        // Variants are always private — only the dispatcher is public.
        variant_fn.vis = syn::Visibility::Inherited;

        // Rename: process → process_v3
        variant_fn.sig.ident = format_ident!("{}_{}", fn_name, tier.suffix);

        // Replace token param type with concrete token type.
        // For "default" tier: remove the token param entirely (tokenless variant).
        if tier.name == "default" {
            let mut inputs: Vec<FnArg> = variant_fn.sig.inputs.iter().cloned().collect();
            inputs.remove(token_param.index);
            variant_fn.sig.inputs = inputs.into_iter().collect();
        } else {
            let concrete_type: Type = syn::parse_str(tier.token_path).unwrap();
            if let FnArg::Typed(pt) = &mut variant_fn.sig.inputs[token_param.index] {
                *pt.ty = concrete_type;
            }
        }

        // Fallback (scalar/default) with _self = Type: inject `let _self = self;` preamble
        // so body's _self references resolve (non-fallback variants get this from
        // #[arcane(_self = Type)])
        if (tier.name == "scalar" || tier.name == "default") && has_self && args.self_type.is_some()
        {
            let original_body = variant_fn.body.clone();
            variant_fn.body = quote!(let _self = self; #original_body);
        }

        // cfg guard: arch + optional feature gate from tier(feature) syntax
        let allow_attr = if tier.allow_unexpected_cfg {
            quote! { #[allow(unexpected_cfgs)] }
        } else {
            quote! {}
        };
        let cfg_guard = match (tier.target_arch, &tier.feature_gate) {
            (Some(arch), Some(feat)) => quote! {
                #[cfg(target_arch = #arch)]
                #allow_attr
                #[cfg(feature = #feat)]
            },
            (Some(arch), None) => quote! { #[cfg(target_arch = #arch)] },
            (None, Some(feat)) => quote! {
                #allow_attr
                #[cfg(feature = #feat)]
            },
            (None, None) => quote! {},
        };

        // All variants are private implementation details of the dispatcher.
        // Suppress dead_code: if the dispatcher is unused, rustc warns on IT
        // (via quote_spanned! with the user's span). Warning on individual
        // variants would be confusing — the user didn't write _scalar or _v3.
        if tier.name != "scalar" && tier.name != "default" {
            let arcane_attr = if let Some(ref self_type) = args.self_type {
                quote! { #[archmage::arcane(_self = #self_type)] }
            } else {
                quote! { #[archmage::arcane] }
            };
            variants.push(quote! {
                #cfg_guard
                #[allow(dead_code)]
                #arcane_attr
                #variant_fn
            });
        } else {
            variants.push(quote! {
                #cfg_guard
                #[allow(dead_code)]
                #variant_fn
            });
        }
    }

    // =========================================================================
    // Generate dispatcher (adapted from gen_incant_entry)
    // =========================================================================

    // Build dispatcher inputs.
    //
    // ScalarToken is kept (real type, incant!-compatible).
    // SimdToken and AutoInjected are stripped.
    let mut dispatcher_inputs: Vec<FnArg> = input_fn.sig.inputs.iter().cloned().collect();
    if !keep_token_in_dispatcher {
        dispatcher_inputs.remove(token_param.index);
    }

    // Rename wildcard params so we can pass them as arguments.
    // Skip the kept ScalarToken param if it's a wildcard — the dispatcher
    // ignores it (does its own summon()), no need to name it.
    let mut wild_counter = 0u32;
    for (i, arg) in dispatcher_inputs.iter_mut().enumerate() {
        if keep_token_in_dispatcher && i == token_param.index {
            continue; // Don't rename the kept token's pattern
        }
        if let FnArg::Typed(pat_type) = arg
            && matches!(pat_type.pat.as_ref(), syn::Pat::Wild(_))
        {
            let ident = format_ident!("__autoversion_wild_{}", wild_counter);
            wild_counter += 1;
            *pat_type.pat = syn::Pat::Ident(syn::PatIdent {
                attrs: vec![],
                by_ref: None,
                mutability: None,
                ident,
                subpat: None,
            });
        }
    }

    // Collect argument idents for dispatch calls (exclude self receiver
    // AND the kept ScalarToken param — variants get their own token from
    // summon(), not from the dispatcher's ScalarToken parameter).
    let dispatch_args: Vec<Ident> = dispatcher_inputs
        .iter()
        .enumerate()
        .filter_map(|(i, arg)| {
            if keep_token_in_dispatcher && i == token_param.index {
                return None; // Skip the kept token param
            }
            if let FnArg::Typed(PatType { pat, .. }) = arg
                && let syn::Pat::Ident(pi) = pat.as_ref()
            {
                return Some(pi.ident.clone());
            }
            None
        })
        .collect();

    // Build turbofish for forwarding type/const generics to variant calls
    let turbofish = build_turbofish(&input_fn.sig.generics);

    // Group non-fallback tiers by target_arch for cfg blocks
    let mut arch_groups: Vec<(Option<&str>, Vec<&ResolvedTier>)> = Vec::new();
    for tier in &tiers {
        if tier.name == "scalar" || tier.name == "default" {
            continue;
        }
        if let Some(group) = arch_groups.iter_mut().find(|(a, _)| *a == tier.target_arch) {
            group.1.push(tier);
        } else {
            arch_groups.push((tier.target_arch, vec![tier]));
        }
    }

    let mut dispatch_arms = Vec::new();
    for (target_arch, group_tiers) in &arch_groups {
        let mut tier_checks = Vec::new();
        for rt in group_tiers {
            let suffixed = format_ident!("{}_{}", fn_name, rt.suffix);
            let token_path: syn::Path = syn::parse_str(rt.token_path).unwrap();

            let call = if has_self {
                quote! { self.#suffixed #turbofish(__t, #(#dispatch_args),*) }
            } else {
                quote! { #suffixed #turbofish(__t, #(#dispatch_args),*) }
            };

            let check = quote! {
                if let Some(__t) = #token_path::summon() {
                    return #call;
                }
            };

            if let Some(feat) = &rt.feature_gate {
                let allow_attr = if rt.allow_unexpected_cfg {
                    quote! { #[allow(unexpected_cfgs)] }
                } else {
                    quote! {}
                };
                tier_checks.push(quote! {
                    #allow_attr
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

    // Fallback call (always available, no summon needed)
    let has_default_tier = tiers.iter().any(|t| t.name == "default");
    let fallback_suffix = if has_default_tier {
        "default"
    } else {
        "scalar"
    };
    let fallback_name = format_ident!("{}_{}", fn_name, fallback_suffix);
    let fallback_call = if has_default_tier {
        // default: tokenless call
        if has_self {
            quote! { self.#fallback_name #turbofish(#(#dispatch_args),*) }
        } else {
            quote! { #fallback_name #turbofish(#(#dispatch_args),*) }
        }
    } else {
        // scalar: call with ScalarToken
        if has_self {
            quote! { self.#fallback_name #turbofish(archmage::ScalarToken, #(#dispatch_args),*) }
        } else {
            quote! { #fallback_name #turbofish(archmage::ScalarToken, #(#dispatch_args),*) }
        }
    };

    // Build dispatcher function
    let dispatcher_inputs_punct: syn::punctuated::Punctuated<FnArg, Token![,]> =
        dispatcher_inputs.into_iter().collect();
    let output = &input_fn.sig.output;
    let generics = &input_fn.sig.generics;
    let where_clause = &generics.where_clause;

    // Use the user's span for the dispatcher so dead_code lint fires on the
    // function the user actually wrote, not on invisible generated variants.
    let user_span = fn_name.span();

    // autoversion uses `return` instead of `break '__dispatch` — no labeled block
    // needed. This avoids label hygiene issues when #[autoversion] is applied inside
    // macro_rules! (labels from proc macros can't be seen from macro_rules! contexts).
    let dispatcher = if let Some(ref feat) = args.cfg_feature {
        // cfg(feature): full dispatch when on, scalar-only when off
        quote_spanned! { user_span =>
            #[cfg(feature = #feat)]
            #(#fn_attrs)*
            #vis fn #fn_name #generics (#dispatcher_inputs_punct) #output #where_clause {
                #simdtoken_deprecation_in_body
                use archmage::SimdToken;
                #(#dispatch_arms)*
                #fallback_call
            }

            #[cfg(not(feature = #feat))]
            #(#fn_attrs)*
            #vis fn #fn_name #generics (#dispatcher_inputs_punct) #output #where_clause {
                #simdtoken_deprecation_in_body
                #fallback_call
            }
        }
    } else {
        quote_spanned! { user_span =>
            #(#fn_attrs)*
            #vis fn #fn_name #generics (#dispatcher_inputs_punct) #output #where_clause {
                #simdtoken_deprecation_in_body
                use archmage::SimdToken;
                #(#dispatch_arms)*
                #fallback_call
            }
        }
    };

    let expanded = quote! {
        #dispatcher
        #(#variants)*
    };

    expanded.into()
}

/// Let the compiler auto-vectorize scalar code for each architecture.
///
/// Write a plain scalar function and let `#[autoversion]` generate
/// architecture-specific copies — each compiled with different
/// `#[target_feature]` flags via `#[arcane]` — plus a runtime dispatcher
/// that calls the best one the CPU supports.
///
/// # Quick start
///
/// ```rust,ignore
/// use archmage::autoversion;
///
/// #[autoversion]
/// fn sum_of_squares(data: &[f32]) -> f32 {
///     let mut sum = 0.0f32;
///     for &x in data {
///         sum += x * x;
///     }
///     sum
/// }
///
/// // Call directly — no token, no unsafe:
/// let result = sum_of_squares(&my_data);
/// ```
///
/// Each variant gets `#[arcane]` → `#[target_feature(enable = "avx2,fma,...")]`,
/// which unlocks the compiler's auto-vectorizer for that feature set.
/// On x86-64, that loop compiles to `vfmadd231ps`. On aarch64, `fmla`.
/// The `_scalar` fallback compiles without SIMD target features.
///
/// # SimdToken — optional placeholder
///
/// You can optionally write `_token: SimdToken` as a parameter. The macro
/// recognizes it and strips it from the dispatcher — both forms produce
/// identical output. Prefer the tokenless form for new code.
///
/// ```rust,ignore
/// #[autoversion]
/// fn normalize(_token: SimdToken, data: &mut [f32], scale: f32) {
///     for x in data.iter_mut() { *x = (*x - 128.0) * scale; }
/// }
/// // Dispatcher is: fn normalize(data: &mut [f32], scale: f32)
/// ```
///
/// # What gets generated
///
/// `#[autoversion] fn process(data: &[f32]) -> f32` expands to:
///
/// - `process_v4(token: X64V4Token, ...)` — AVX-512
/// - `process_v3(token: X64V3Token, ...)` — AVX2+FMA
/// - `process_neon(token: NeonToken, ...)` — aarch64 NEON
/// - `process_wasm128(token: Wasm128Token, ...)` — WASM SIMD
/// - `process_scalar(token: ScalarToken, ...)` — no SIMD, always available
/// - `process(data: &[f32]) -> f32` — **dispatcher**
///
/// Variants are private. The dispatcher gets the original function's visibility.
/// Within the same module, call variants directly for testing or benchmarking.
///
/// # Explicit tiers
///
/// ```rust,ignore
/// #[autoversion(v3, v4, neon, arm_v2, wasm128)]
/// fn process(data: &[f32]) -> f32 { ... }
/// ```
///
/// `scalar` is always included implicitly.
///
/// Default tiers: `v4`, `v3`, `neon`, `wasm128`, `scalar`.
///
/// Known tiers: `v1`, `v2`, `v3`, `v3_crypto`, `v4`, `v4x`, `neon`,
/// `neon_aes`, `neon_sha3`, `neon_crc`, `arm_v2`, `arm_v3`, `wasm128`,
/// `wasm128_relaxed`, `x64_crypto`, `scalar`.
///
/// # Methods
///
/// For inherent methods, `self` works naturally:
///
/// ```rust,ignore
/// impl ImageBuffer {
///     #[autoversion]
///     fn normalize(&mut self, gamma: f32) {
///         for pixel in &mut self.data {
///             *pixel = (*pixel / 255.0).powf(gamma);
///         }
///     }
/// }
/// buffer.normalize(2.2);
/// ```
///
/// For trait method delegation, use `_self = Type` (nested mode):
///
/// ```rust,ignore
/// impl MyType {
///     #[autoversion(_self = MyType)]
///     fn compute_impl(&self, data: &[f32]) -> f32 {
///         _self.weights.iter().zip(data).map(|(w, d)| w * d).sum()
///     }
/// }
/// ```
///
/// # Nesting with `incant!`
///
/// Hand-written SIMD for specific tiers, autoversion for the rest:
///
/// ```rust,ignore
/// pub fn process(data: &[f32]) -> f32 {
///     incant!(process(data), [v4, scalar])
/// }
///
/// #[arcane(import_intrinsics)]
/// fn process_v4(_t: X64V4Token, data: &[f32]) -> f32 { /* AVX-512 */ }
///
/// // Bridge: incant! passes ScalarToken, autoversion doesn't need one
/// fn process_scalar(_: ScalarToken, data: &[f32]) -> f32 {
///     process_auto(data)
/// }
///
/// #[autoversion(v3, neon)]
/// fn process_auto(data: &[f32]) -> f32 { data.iter().sum() }
/// ```
///
/// # Comparison with `#[magetypes]` + `incant!`
///
/// | | `#[autoversion]` | `#[magetypes]` + `incant!` |
/// |---|---|---|
/// | Generates variants + dispatcher | Yes | Variants only (+ separate `incant!`) |
/// | Body touched | No (signature only) | Yes (text substitution) |
/// | Best for | Scalar auto-vectorization | Hand-written SIMD types |
#[proc_macro_attribute]
pub fn autoversion(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as AutoversionArgs);
    let input_fn = parse_macro_input!(item as LightFn);
    autoversion_impl(input_fn, args)
}

// =============================================================================
// Unit tests for token/trait recognition maps
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    use super::generated::{ALL_CONCRETE_TOKENS, ALL_TRAIT_NAMES};
    use syn::{ItemFn, ReturnType};

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

    // =========================================================================
    // autoversion — argument parsing
    // =========================================================================

    #[test]
    fn autoversion_args_empty() {
        let args: AutoversionArgs = syn::parse_str("").unwrap();
        assert!(args.self_type.is_none());
        assert!(args.tiers.is_none());
    }

    #[test]
    fn autoversion_args_single_tier() {
        let args: AutoversionArgs = syn::parse_str("v3").unwrap();
        assert!(args.self_type.is_none());
        assert_eq!(args.tiers.as_ref().unwrap(), &["v3"]);
    }

    #[test]
    fn autoversion_args_tiers_only() {
        let args: AutoversionArgs = syn::parse_str("v3, v4, neon").unwrap();
        assert!(args.self_type.is_none());
        let tiers = args.tiers.unwrap();
        assert_eq!(tiers, vec!["v3", "v4", "neon"]);
    }

    #[test]
    fn autoversion_args_many_tiers() {
        let args: AutoversionArgs =
            syn::parse_str("v1, v2, v3, v4, v4x, neon, arm_v2, wasm128").unwrap();
        assert_eq!(
            args.tiers.unwrap(),
            vec!["v1", "v2", "v3", "v4", "v4x", "neon", "arm_v2", "wasm128"]
        );
    }

    #[test]
    fn autoversion_args_trailing_comma() {
        let args: AutoversionArgs = syn::parse_str("v3, v4,").unwrap();
        assert_eq!(args.tiers.as_ref().unwrap(), &["v3", "v4"]);
    }

    #[test]
    fn autoversion_args_self_only() {
        let args: AutoversionArgs = syn::parse_str("_self = MyType").unwrap();
        assert!(args.self_type.is_some());
        assert!(args.tiers.is_none());
    }

    #[test]
    fn autoversion_args_self_and_tiers() {
        let args: AutoversionArgs = syn::parse_str("_self = MyType, v3, neon").unwrap();
        assert!(args.self_type.is_some());
        let tiers = args.tiers.unwrap();
        assert_eq!(tiers, vec!["v3", "neon"]);
    }

    #[test]
    fn autoversion_args_tiers_then_self() {
        // _self can appear after tier names
        let args: AutoversionArgs = syn::parse_str("v3, neon, _self = MyType").unwrap();
        assert!(args.self_type.is_some());
        let tiers = args.tiers.unwrap();
        assert_eq!(tiers, vec!["v3", "neon"]);
    }

    #[test]
    fn autoversion_args_self_with_path_type() {
        let args: AutoversionArgs = syn::parse_str("_self = crate::MyType").unwrap();
        assert!(args.self_type.is_some());
        assert!(args.tiers.is_none());
    }

    #[test]
    fn autoversion_args_self_with_generic_type() {
        let args: AutoversionArgs = syn::parse_str("_self = Vec<u8>").unwrap();
        assert!(args.self_type.is_some());
        let ty_str = args.self_type.unwrap().to_token_stream().to_string();
        assert!(ty_str.contains("Vec"), "Expected Vec<u8>, got: {}", ty_str);
    }

    #[test]
    fn autoversion_args_self_trailing_comma() {
        let args: AutoversionArgs = syn::parse_str("_self = MyType,").unwrap();
        assert!(args.self_type.is_some());
        assert!(args.tiers.is_none());
    }

    // =========================================================================
    // autoversion — find_autoversion_token_param
    // =========================================================================

    #[test]
    fn find_autoversion_token_param_simdtoken_first() {
        let f: ItemFn =
            syn::parse_str("fn process(token: SimdToken, data: &[f32]) -> f32 {}").unwrap();
        let param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
        assert_eq!(param.index, 0);
        assert_eq!(param.ident, "token");
        assert_eq!(param.kind, AutoversionTokenKind::SimdToken);
    }

    #[test]
    fn find_autoversion_token_param_simdtoken_second() {
        let f: ItemFn =
            syn::parse_str("fn process(data: &[f32], token: SimdToken) -> f32 {}").unwrap();
        let param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
        assert_eq!(param.index, 1);
        assert_eq!(param.kind, AutoversionTokenKind::SimdToken);
    }

    #[test]
    fn find_autoversion_token_param_underscore_prefix() {
        let f: ItemFn =
            syn::parse_str("fn process(_token: SimdToken, data: &[f32]) -> f32 {}").unwrap();
        let param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
        assert_eq!(param.index, 0);
        assert_eq!(param.ident, "_token");
    }

    #[test]
    fn find_autoversion_token_param_wildcard() {
        let f: ItemFn = syn::parse_str("fn process(_: SimdToken, data: &[f32]) -> f32 {}").unwrap();
        let param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
        assert_eq!(param.index, 0);
        assert_eq!(param.ident, "__autoversion_token");
    }

    #[test]
    fn find_autoversion_token_param_scalar_token() {
        let f: ItemFn =
            syn::parse_str("fn process_scalar(_: ScalarToken, data: &[f32]) -> f32 {}").unwrap();
        let param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
        assert_eq!(param.index, 0);
        assert_eq!(param.kind, AutoversionTokenKind::ScalarToken);
    }

    #[test]
    fn find_autoversion_token_param_not_found() {
        let f: ItemFn = syn::parse_str("fn process(data: &[f32]) -> f32 {}").unwrap();
        assert!(find_autoversion_token_param(&f.sig).unwrap().is_none());
    }

    #[test]
    fn find_autoversion_token_param_no_params() {
        let f: ItemFn = syn::parse_str("fn process() {}").unwrap();
        assert!(find_autoversion_token_param(&f.sig).unwrap().is_none());
    }

    #[test]
    fn find_autoversion_token_param_concrete_token_errors() {
        let f: ItemFn =
            syn::parse_str("fn process(token: X64V3Token, data: &[f32]) -> f32 {}").unwrap();
        let err = find_autoversion_token_param(&f.sig).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("concrete token"),
            "error should mention concrete token: {msg}"
        );
        assert!(
            msg.contains("#[arcane]"),
            "error should suggest #[arcane]: {msg}"
        );
    }

    #[test]
    fn find_autoversion_token_param_neon_token_errors() {
        let f: ItemFn =
            syn::parse_str("fn process(token: NeonToken, data: &[f32]) -> f32 {}").unwrap();
        assert!(find_autoversion_token_param(&f.sig).is_err());
    }

    #[test]
    fn find_autoversion_token_param_unknown_type_ignored() {
        // Random types are just regular params, not token params
        let f: ItemFn = syn::parse_str("fn process(data: &[f32], scale: f32) -> f32 {}").unwrap();
        assert!(find_autoversion_token_param(&f.sig).unwrap().is_none());
    }

    #[test]
    fn find_autoversion_token_param_among_many() {
        let f: ItemFn = syn::parse_str(
            "fn process(a: i32, b: f64, token: SimdToken, c: &str, d: bool) -> f32 {}",
        )
        .unwrap();
        let param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
        assert_eq!(param.index, 2);
        assert_eq!(param.ident, "token");
    }

    #[test]
    fn find_autoversion_token_param_with_generics() {
        let f: ItemFn =
            syn::parse_str("fn process<T: Clone>(token: SimdToken, data: &[T]) -> T {}").unwrap();
        let param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
        assert_eq!(param.index, 0);
    }

    #[test]
    fn find_autoversion_token_param_with_where_clause() {
        let f: ItemFn = syn::parse_str(
            "fn process<T>(token: SimdToken, data: &[T]) -> T where T: Copy + Default {}",
        )
        .unwrap();
        let param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
        assert_eq!(param.index, 0);
    }

    #[test]
    fn find_autoversion_token_param_with_lifetime() {
        let f: ItemFn =
            syn::parse_str("fn process<'a>(token: SimdToken, data: &'a [f32]) -> &'a f32 {}")
                .unwrap();
        let param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
        assert_eq!(param.index, 0);
    }

    // =========================================================================
    // autoversion — tier resolution
    // =========================================================================

    #[test]
    fn autoversion_default_tiers_all_resolve() {
        let names: Vec<String> = DEFAULT_TIER_NAMES.iter().map(|s| s.to_string()).collect();
        let tiers = resolve_tiers(&names, proc_macro2::Span::call_site(), false).unwrap();
        assert!(!tiers.is_empty());
        // scalar should be present
        assert!(tiers.iter().any(|t| t.name == "scalar"));
    }

    #[test]
    fn autoversion_scalar_always_appended() {
        let names = vec!["v3".to_string(), "neon".to_string()];
        let tiers = resolve_tiers(&names, proc_macro2::Span::call_site(), false).unwrap();
        assert!(
            tiers.iter().any(|t| t.name == "scalar"),
            "scalar must be auto-appended"
        );
    }

    #[test]
    fn autoversion_scalar_not_duplicated() {
        let names = vec!["v3".to_string(), "scalar".to_string()];
        let tiers = resolve_tiers(&names, proc_macro2::Span::call_site(), false).unwrap();
        let scalar_count = tiers.iter().filter(|t| t.name == "scalar").count();
        assert_eq!(scalar_count, 1, "scalar must not be duplicated");
    }

    #[test]
    fn autoversion_tiers_sorted_by_priority() {
        let names = vec!["neon".to_string(), "v4".to_string(), "v3".to_string()];
        let tiers = resolve_tiers(&names, proc_macro2::Span::call_site(), false).unwrap();
        // v4 (priority 40) > v3 (30) > neon (20) > scalar (0)
        let priorities: Vec<u32> = tiers.iter().map(|t| t.priority).collect();
        for window in priorities.windows(2) {
            assert!(
                window[0] >= window[1],
                "Tiers not sorted by priority: {:?}",
                priorities
            );
        }
    }

    #[test]
    fn autoversion_unknown_tier_errors() {
        let names = vec!["v3".to_string(), "avx9000".to_string()];
        let result = resolve_tiers(&names, proc_macro2::Span::call_site(), false);
        match result {
            Ok(_) => panic!("Expected error for unknown tier 'avx9000'"),
            Err(e) => {
                let err_msg = e.to_string();
                assert!(
                    err_msg.contains("avx9000"),
                    "Error should mention unknown tier: {}",
                    err_msg
                );
            }
        }
    }

    #[test]
    fn autoversion_all_known_tiers_resolve() {
        // Every tier in ALL_TIERS should be findable
        for tier in ALL_TIERS {
            assert!(
                find_tier(tier.name).is_some(),
                "Tier '{}' should be findable by name",
                tier.name
            );
        }
    }

    #[test]
    fn autoversion_default_tier_list_is_sensible() {
        // Defaults should cover x86, ARM, WASM, and scalar
        let names: Vec<String> = DEFAULT_TIER_NAMES.iter().map(|s| s.to_string()).collect();
        let tiers = resolve_tiers(&names, proc_macro2::Span::call_site(), false).unwrap();

        let has_x86 = tiers.iter().any(|t| t.target_arch == Some("x86_64"));
        let has_arm = tiers.iter().any(|t| t.target_arch == Some("aarch64"));
        let has_wasm = tiers.iter().any(|t| t.target_arch == Some("wasm32"));
        let has_scalar = tiers.iter().any(|t| t.name == "scalar");

        assert!(has_x86, "Default tiers should include an x86_64 tier");
        assert!(has_arm, "Default tiers should include an aarch64 tier");
        assert!(has_wasm, "Default tiers should include a wasm32 tier");
        assert!(has_scalar, "Default tiers should include scalar");
    }

    // =========================================================================
    // autoversion — variant replacement (AST manipulation)
    // =========================================================================

    /// Mirrors what `autoversion_impl` does for a single variant: parse an
    /// ItemFn (for test convenience), rename it, swap the SimdToken param
    /// type, optionally inject the `_self` preamble for scalar+self.
    fn do_variant_replacement(func: &str, tier_name: &str, has_self: bool) -> ItemFn {
        let mut f: ItemFn = syn::parse_str(func).unwrap();
        let fn_name = f.sig.ident.to_string();

        let tier = find_tier(tier_name).unwrap();

        // Rename
        f.sig.ident = format_ident!("{}_{}", fn_name, tier.suffix);

        // Find and replace SimdToken param type (skip for "default" — tokenless)
        let token_idx = find_autoversion_token_param(&f.sig)
            .expect("should not error on SimdToken")
            .unwrap_or_else(|| panic!("No SimdToken param in: {}", func))
            .index;
        if tier_name == "default" {
            // Remove the token param for default tier
            let stmts = f.block.stmts.clone();
            let mut inputs: Vec<FnArg> = f.sig.inputs.iter().cloned().collect();
            inputs.remove(token_idx);
            f.sig.inputs = inputs.into_iter().collect();
            f.block.stmts = stmts;
        } else {
            let concrete_type: Type = syn::parse_str(tier.token_path).unwrap();
            if let FnArg::Typed(pt) = &mut f.sig.inputs[token_idx] {
                *pt.ty = concrete_type;
            }
        }

        // Fallback (scalar/default) + self: inject preamble
        if (tier_name == "scalar" || tier_name == "default") && has_self {
            let preamble: syn::Stmt = syn::parse_quote!(let _self = self;);
            f.block.stmts.insert(0, preamble);
        }

        f
    }

    #[test]
    fn variant_replacement_v3_renames_function() {
        let f = do_variant_replacement(
            "fn process(token: SimdToken, data: &[f32]) -> f32 { 0.0 }",
            "v3",
            false,
        );
        assert_eq!(f.sig.ident, "process_v3");
    }

    #[test]
    fn variant_replacement_v3_replaces_token_type() {
        let f = do_variant_replacement(
            "fn process(token: SimdToken, data: &[f32]) -> f32 { 0.0 }",
            "v3",
            false,
        );
        let first_param_ty = match &f.sig.inputs[0] {
            FnArg::Typed(pt) => pt.ty.to_token_stream().to_string(),
            _ => panic!("Expected typed param"),
        };
        assert!(
            first_param_ty.contains("X64V3Token"),
            "Expected X64V3Token, got: {}",
            first_param_ty
        );
    }

    #[test]
    fn variant_replacement_neon_produces_valid_fn() {
        let f = do_variant_replacement(
            "fn compute(token: SimdToken, data: &[f32]) -> f32 { 0.0 }",
            "neon",
            false,
        );
        assert_eq!(f.sig.ident, "compute_neon");
        let first_param_ty = match &f.sig.inputs[0] {
            FnArg::Typed(pt) => pt.ty.to_token_stream().to_string(),
            _ => panic!("Expected typed param"),
        };
        assert!(
            first_param_ty.contains("NeonToken"),
            "Expected NeonToken, got: {}",
            first_param_ty
        );
    }

    #[test]
    fn variant_replacement_wasm128_produces_valid_fn() {
        let f = do_variant_replacement(
            "fn compute(_t: SimdToken, data: &[f32]) -> f32 { 0.0 }",
            "wasm128",
            false,
        );
        assert_eq!(f.sig.ident, "compute_wasm128");
    }

    #[test]
    fn variant_replacement_scalar_produces_valid_fn() {
        let f = do_variant_replacement(
            "fn compute(token: SimdToken, data: &[f32]) -> f32 { 0.0 }",
            "scalar",
            false,
        );
        assert_eq!(f.sig.ident, "compute_scalar");
        let first_param_ty = match &f.sig.inputs[0] {
            FnArg::Typed(pt) => pt.ty.to_token_stream().to_string(),
            _ => panic!("Expected typed param"),
        };
        assert!(
            first_param_ty.contains("ScalarToken"),
            "Expected ScalarToken, got: {}",
            first_param_ty
        );
    }

    #[test]
    fn variant_replacement_v4_produces_valid_fn() {
        let f = do_variant_replacement(
            "fn transform(token: SimdToken, data: &mut [f32]) { }",
            "v4",
            false,
        );
        assert_eq!(f.sig.ident, "transform_v4");
        let first_param_ty = match &f.sig.inputs[0] {
            FnArg::Typed(pt) => pt.ty.to_token_stream().to_string(),
            _ => panic!("Expected typed param"),
        };
        assert!(
            first_param_ty.contains("X64V4Token"),
            "Expected X64V4Token, got: {}",
            first_param_ty
        );
    }

    #[test]
    fn variant_replacement_v4x_produces_valid_fn() {
        let f = do_variant_replacement(
            "fn transform(token: SimdToken, data: &mut [f32]) { }",
            "v4x",
            false,
        );
        assert_eq!(f.sig.ident, "transform_v4x");
    }

    #[test]
    fn variant_replacement_arm_v2_produces_valid_fn() {
        let f = do_variant_replacement(
            "fn transform(token: SimdToken, data: &mut [f32]) { }",
            "arm_v2",
            false,
        );
        assert_eq!(f.sig.ident, "transform_arm_v2");
    }

    #[test]
    fn variant_replacement_preserves_generics() {
        let f = do_variant_replacement(
            "fn process<T: Copy + Default>(token: SimdToken, data: &[T]) -> T { T::default() }",
            "v3",
            false,
        );
        assert_eq!(f.sig.ident, "process_v3");
        // Generic params should still be present
        assert!(
            !f.sig.generics.params.is_empty(),
            "Generics should be preserved"
        );
    }

    #[test]
    fn variant_replacement_preserves_where_clause() {
        let f = do_variant_replacement(
            "fn process<T>(token: SimdToken, data: &[T]) -> T where T: Copy + Default { T::default() }",
            "v3",
            false,
        );
        assert!(
            f.sig.generics.where_clause.is_some(),
            "Where clause should be preserved"
        );
    }

    #[test]
    fn variant_replacement_preserves_return_type() {
        let f = do_variant_replacement(
            "fn process(token: SimdToken, data: &[f32]) -> Vec<f32> { vec![] }",
            "neon",
            false,
        );
        let ret = f.sig.output.to_token_stream().to_string();
        assert!(
            ret.contains("Vec"),
            "Return type should be preserved, got: {}",
            ret
        );
    }

    #[test]
    fn variant_replacement_preserves_multiple_params() {
        let f = do_variant_replacement(
            "fn process(token: SimdToken, a: &[f32], b: &[f32], scale: f32) -> f32 { 0.0 }",
            "v3",
            false,
        );
        // SimdToken → X64V3Token, plus the 3 other params
        assert_eq!(f.sig.inputs.len(), 4);
    }

    #[test]
    fn variant_replacement_preserves_no_return_type() {
        let f = do_variant_replacement(
            "fn transform(token: SimdToken, data: &mut [f32]) { }",
            "v3",
            false,
        );
        assert!(
            matches!(f.sig.output, ReturnType::Default),
            "No return type should remain as Default"
        );
    }

    #[test]
    fn variant_replacement_preserves_lifetime_params() {
        let f = do_variant_replacement(
            "fn process<'a>(token: SimdToken, data: &'a [f32]) -> &'a [f32] { data }",
            "v3",
            false,
        );
        assert!(!f.sig.generics.params.is_empty());
    }

    #[test]
    fn variant_replacement_scalar_self_injects_preamble() {
        let f = do_variant_replacement(
            "fn method(token: SimdToken, data: &[f32]) -> f32 { 0.0 }",
            "scalar",
            true, // has_self
        );
        assert_eq!(f.sig.ident, "method_scalar");

        // First statement should be `let _self = self;`
        let body_str = f.block.to_token_stream().to_string();
        assert!(
            body_str.contains("let _self = self"),
            "Scalar+self variant should have _self preamble, got: {}",
            body_str
        );
    }

    #[test]
    fn variant_replacement_all_default_tiers_produce_valid_fns() {
        let names: Vec<String> = DEFAULT_TIER_NAMES.iter().map(|s| s.to_string()).collect();
        let tiers = resolve_tiers(&names, proc_macro2::Span::call_site(), false).unwrap();

        for tier in &tiers {
            let f = do_variant_replacement(
                "fn process(token: SimdToken, data: &[f32]) -> f32 { 0.0 }",
                tier.name,
                false,
            );
            let expected_name = format!("process_{}", tier.suffix);
            assert_eq!(
                f.sig.ident.to_string(),
                expected_name,
                "Tier '{}' should produce function '{}'",
                tier.name,
                expected_name
            );
        }
    }

    #[test]
    fn variant_replacement_all_known_tiers_produce_valid_fns() {
        for tier in ALL_TIERS {
            let f = do_variant_replacement(
                "fn compute(token: SimdToken, data: &[f32]) -> f32 { 0.0 }",
                tier.name,
                false,
            );
            let expected_name = format!("compute_{}", tier.suffix);
            assert_eq!(
                f.sig.ident.to_string(),
                expected_name,
                "Tier '{}' should produce function '{}'",
                tier.name,
                expected_name
            );
        }
    }

    #[test]
    fn variant_replacement_no_simdtoken_remains() {
        for tier in ALL_TIERS {
            let f = do_variant_replacement(
                "fn compute(token: SimdToken, data: &[f32]) -> f32 { 0.0 }",
                tier.name,
                false,
            );
            let full_str = f.to_token_stream().to_string();
            assert!(
                !full_str.contains("SimdToken"),
                "Tier '{}' variant still contains 'SimdToken': {}",
                tier.name,
                full_str
            );
        }
    }

    // =========================================================================
    // autoversion — cfg guard and tier descriptor properties
    // =========================================================================

    #[test]
    fn tier_v3_targets_x86_64() {
        let tier = find_tier("v3").unwrap();
        assert_eq!(tier.target_arch, Some("x86_64"));
    }

    #[test]
    fn tier_v4_targets_x86_64() {
        let tier = find_tier("v4").unwrap();
        assert_eq!(tier.target_arch, Some("x86_64"));
    }

    #[test]
    fn tier_v4x_targets_x86_64() {
        let tier = find_tier("v4x").unwrap();
        assert_eq!(tier.target_arch, Some("x86_64"));
    }

    #[test]
    fn tier_neon_targets_aarch64() {
        let tier = find_tier("neon").unwrap();
        assert_eq!(tier.target_arch, Some("aarch64"));
    }

    #[test]
    fn tier_wasm128_targets_wasm32() {
        let tier = find_tier("wasm128").unwrap();
        assert_eq!(tier.target_arch, Some("wasm32"));
    }

    #[test]
    fn tier_scalar_has_no_guards() {
        let tier = find_tier("scalar").unwrap();
        assert_eq!(tier.target_arch, None);
        assert_eq!(tier.priority, 0);
    }

    #[test]
    fn tier_priorities_are_consistent() {
        // Higher-capability tiers within the same arch should have higher priority
        let v2 = find_tier("v2").unwrap();
        let v3 = find_tier("v3").unwrap();
        let v4 = find_tier("v4").unwrap();
        assert!(v4.priority > v3.priority);
        assert!(v3.priority > v2.priority);

        let neon = find_tier("neon").unwrap();
        let arm_v2 = find_tier("arm_v2").unwrap();
        let arm_v3 = find_tier("arm_v3").unwrap();
        assert!(arm_v3.priority > arm_v2.priority);
        assert!(arm_v2.priority > neon.priority);

        // scalar is lowest
        let scalar = find_tier("scalar").unwrap();
        assert!(neon.priority > scalar.priority);
        assert!(v2.priority > scalar.priority);
    }

    // =========================================================================
    // autoversion — dispatcher structure
    // =========================================================================

    #[test]
    fn dispatcher_param_removal_free_fn() {
        // Simulate what autoversion_impl does: remove the SimdToken param
        let f: ItemFn =
            syn::parse_str("fn process(token: SimdToken, data: &[f32], scale: f32) -> f32 { 0.0 }")
                .unwrap();

        let token_param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
        // SimdToken → strip from dispatcher
        assert_eq!(token_param.kind, AutoversionTokenKind::SimdToken);
        let mut dispatcher_inputs: Vec<FnArg> = f.sig.inputs.iter().cloned().collect();
        dispatcher_inputs.remove(token_param.index);
        assert_eq!(dispatcher_inputs.len(), 2);
    }

    #[test]
    fn dispatcher_param_removal_token_only() {
        let f: ItemFn = syn::parse_str("fn process(token: SimdToken) -> f32 { 0.0 }").unwrap();
        let token_param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
        let mut dispatcher_inputs: Vec<FnArg> = f.sig.inputs.iter().cloned().collect();
        dispatcher_inputs.remove(token_param.index);
        assert_eq!(dispatcher_inputs.len(), 0);
    }

    #[test]
    fn dispatcher_param_removal_token_last() {
        let f: ItemFn =
            syn::parse_str("fn process(data: &[f32], scale: f32, token: SimdToken) -> f32 { 0.0 }")
                .unwrap();
        let token_param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
        assert_eq!(token_param.index, 2);
        let mut dispatcher_inputs: Vec<FnArg> = f.sig.inputs.iter().cloned().collect();
        dispatcher_inputs.remove(token_param.index);
        assert_eq!(dispatcher_inputs.len(), 2);
    }

    #[test]
    fn dispatcher_scalar_token_kept() {
        // ScalarToken is a real type — kept in dispatcher for incant! compatibility
        let f: ItemFn =
            syn::parse_str("fn process_scalar(_: ScalarToken, data: &[f32]) -> f32 { 0.0 }")
                .unwrap();
        let token_param = find_autoversion_token_param(&f.sig).unwrap().unwrap();
        assert_eq!(token_param.kind, AutoversionTokenKind::ScalarToken);
        // Should NOT be removed — dispatcher keeps it
        assert_eq!(f.sig.inputs.len(), 2);
    }

    #[test]
    fn dispatcher_dispatch_args_extraction() {
        // Test that we correctly extract idents for the dispatch call
        let f: ItemFn =
            syn::parse_str("fn process(data: &[f32], scale: f32) -> f32 { 0.0 }").unwrap();

        let dispatch_args: Vec<String> = f
            .sig
            .inputs
            .iter()
            .filter_map(|arg| {
                if let FnArg::Typed(PatType { pat, .. }) = arg {
                    if let syn::Pat::Ident(pi) = pat.as_ref() {
                        return Some(pi.ident.to_string());
                    }
                }
                None
            })
            .collect();

        assert_eq!(dispatch_args, vec!["data", "scale"]);
    }

    #[test]
    fn dispatcher_wildcard_params_get_renamed() {
        let f: ItemFn = syn::parse_str("fn process(_: &[f32], _: f32) -> f32 { 0.0 }").unwrap();

        let mut dispatcher_inputs: Vec<FnArg> = f.sig.inputs.iter().cloned().collect();

        let mut wild_counter = 0u32;
        for arg in &mut dispatcher_inputs {
            if let FnArg::Typed(pat_type) = arg {
                if matches!(pat_type.pat.as_ref(), syn::Pat::Wild(_)) {
                    let ident = format_ident!("__autoversion_wild_{}", wild_counter);
                    wild_counter += 1;
                    *pat_type.pat = syn::Pat::Ident(syn::PatIdent {
                        attrs: vec![],
                        by_ref: None,
                        mutability: None,
                        ident,
                        subpat: None,
                    });
                }
            }
        }

        // Both wildcards should be renamed
        assert_eq!(wild_counter, 2);

        let names: Vec<String> = dispatcher_inputs
            .iter()
            .filter_map(|arg| {
                if let FnArg::Typed(PatType { pat, .. }) = arg {
                    if let syn::Pat::Ident(pi) = pat.as_ref() {
                        return Some(pi.ident.to_string());
                    }
                }
                None
            })
            .collect();

        assert_eq!(names, vec!["__autoversion_wild_0", "__autoversion_wild_1"]);
    }

    // =========================================================================
    // autoversion — suffix_path (reused in dispatch)
    // =========================================================================

    #[test]
    fn suffix_path_simple() {
        let path: syn::Path = syn::parse_str("process").unwrap();
        let suffixed = suffix_path(&path, "v3");
        assert_eq!(suffixed.to_token_stream().to_string(), "process_v3");
    }

    #[test]
    fn suffix_path_qualified() {
        let path: syn::Path = syn::parse_str("module::process").unwrap();
        let suffixed = suffix_path(&path, "neon");
        let s = suffixed.to_token_stream().to_string();
        assert!(
            s.contains("process_neon"),
            "Expected process_neon, got: {}",
            s
        );
    }
}
