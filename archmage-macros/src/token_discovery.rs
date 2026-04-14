//! Token parameter discovery for #[arcane] and #[rite].
//!
//! Finds token parameters in function signatures, extracts feature sets,
//! and diagnoses invalid token types.

use syn::{FnArg, GenericParam, Ident, PatType, Signature, Token, Type, TypeParamBound};

use crate::generated::{
    token_to_arch, token_to_features, token_to_magetypes_namespace, trait_to_arch,
    trait_to_features, trait_to_magetypes_namespace,
};

pub(crate) enum TokenTypeInfo {
    /// Concrete token type (e.g., `Avx2Token`)
    Concrete(String),
    /// impl Trait with the trait names (e.g., `impl HasX64V2`)
    ImplTrait(Vec<String>),
    /// Generic type parameter name (e.g., `T`)
    Generic(String),
}

/// Extract token type information from a type.
pub(crate) fn extract_token_type_info(ty: &Type) -> Option<TokenTypeInfo> {
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
pub(crate) fn extract_trait_names_from_bounds(
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
pub(crate) fn find_generic_bounds(sig: &Signature, type_name: &str) -> Option<Vec<String>> {
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
pub(crate) fn traits_to_features(trait_names: &[String]) -> Option<Vec<&'static str>> {
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
pub(crate) const FEATURELESS_TRAIT_NAMES: &[&str] = &["SimdToken", "IntoConcreteToken"];

/// Check if any trait names are featureless (no CPU feature mapping).
/// Returns the first featureless trait name found.
pub(crate) fn find_featureless_trait(trait_names: &[String]) -> Option<&'static str> {
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
pub(crate) fn diagnose_featureless_token(sig: &Signature) -> Option<&'static str> {
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
pub(crate) struct TokenParamInfo {
    /// The parameter identifier (e.g., `token`)
    pub ident: Ident,
    /// Target features to enable (e.g., `["avx2", "fma"]`)
    pub features: Vec<&'static str>,
    /// Target architecture (Some for concrete tokens, None for traits/generics)
    pub target_arch: Option<&'static str>,
    /// Concrete token type name (Some for concrete tokens, None for traits/generics)
    pub token_type_name: Option<String>,
    /// Magetypes width namespace (e.g., "v3", "neon", "wasm128")
    pub magetypes_namespace: Option<&'static str>,
    /// Full type from the function signature (for const tier tag assertion).
    /// Set for concrete token types, None for trait/generic bounds.
    pub token_type: Option<Type>,
}

/// Resolve magetypes namespace from a list of trait names.
/// Returns the first matching namespace found.
pub(crate) fn traits_to_magetypes_namespace(trait_names: &[String]) -> Option<&'static str> {
    for name in trait_names {
        if let Some(ns) = trait_to_magetypes_namespace(name) {
            return Some(ns);
        }
    }
    None
}

/// Given trait bound names, return the first matching target architecture.
pub(crate) fn traits_to_arch(trait_names: &[String]) -> Option<&'static str> {
    for name in trait_names {
        if let Some(arch) = trait_to_arch(name) {
            return Some(arch);
        }
    }
    None
}

/// Find the first token parameter in a function signature.
pub(crate) fn find_token_param(sig: &Signature) -> Option<TokenParamInfo> {
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
                    let (features, arch, token_name, mage_ns, full_type) = match info {
                        TokenTypeInfo::Concrete(ref name) => {
                            let features = token_to_features(name).map(|f| f.to_vec());
                            let arch = token_to_arch(name);
                            let ns = token_to_magetypes_namespace(name);
                            // Clone the full Type for const tier tag assertion.
                            // This preserves any path prefix (e.g., `my_crate::X64V3Token`)
                            // so the assertion resolves through re-exports.
                            let full_type = Some(ty.as_ref().clone());
                            (features, arch, Some(name.clone()), ns, full_type)
                        }
                        TokenTypeInfo::ImplTrait(ref trait_names) => {
                            let ns = traits_to_magetypes_namespace(trait_names);
                            let arch = traits_to_arch(trait_names);
                            (traits_to_features(trait_names), arch, None, ns, None)
                        }
                        TokenTypeInfo::Generic(type_name) => {
                            // Look up the generic parameter's bounds
                            let bounds = find_generic_bounds(sig, &type_name);
                            let features = bounds.as_ref().and_then(|t| traits_to_features(t));
                            let ns = bounds
                                .as_ref()
                                .and_then(|t| traits_to_magetypes_namespace(t));
                            let arch = bounds.as_ref().and_then(|t| traits_to_arch(t));
                            (features, arch, None, ns, None)
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
                                token_type: full_type,
                            });
                        }
                    }
                }
            }
        }
    }
    None
}
