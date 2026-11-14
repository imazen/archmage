//! Token parameter discovery for #[arcane] and #[rite].
//!
//! Finds token parameters in function signatures, extracts feature sets,
//! and diagnoses invalid token types without per-parameter heap allocations.

use std::borrow::Cow;

use proc_macro2::Span;
use syn::{GenericParam, Ident, Token, Type, TypeParamBound};

use crate::common::{FnArg, Pat, PatType, Signature, SmallVec, with_ident_str};
use crate::generated::{
    token_to_arch, token_to_features, token_to_magetypes_namespace, trait_to_arch,
    trait_to_features, trait_to_magetypes_namespace,
};

/// Map any recognized concrete token spelling (including aliases and `ScalarToken`)
/// to a `'static` string slice with zero heap allocation.
pub(crate) fn known_token_str(name: &str) -> Option<&'static str> {
    match name {
        "ScalarToken" => Some("ScalarToken"),
        "X64V1Token" => Some("X64V1Token"),
        "Sse2Token" => Some("Sse2Token"),
        "X64V2Token" => Some("X64V2Token"),
        "X64CryptoToken" => Some("X64CryptoToken"),
        "X64V3Token" => Some("X64V3Token"),
        "Desktop64" => Some("Desktop64"),
        "Avx2FmaToken" => Some("Avx2FmaToken"),
        "X64V3CryptoToken" => Some("X64V3CryptoToken"),
        "X64V3GfniCryptoToken" => Some("X64V3GfniCryptoToken"),
        "X64V3VnniToken" => Some("X64V3VnniToken"),
        "X64V4Token" => Some("X64V4Token"),
        "Avx512Token" => Some("Avx512Token"),
        "Server64" => Some("Server64"),
        "X64V4xToken" => Some("X64V4xToken"),
        "Avx512ModernToken" => Some("Avx512ModernToken"),
        "Avx512Fp16Token" => Some("Avx512Fp16Token"),
        "NeonToken" => Some("NeonToken"),
        "Arm64" => Some("Arm64"),
        "NeonAesToken" => Some("NeonAesToken"),
        "NeonSha3Token" => Some("NeonSha3Token"),
        "NeonCrcToken" => Some("NeonCrcToken"),
        "Arm64V2Token" => Some("Arm64V2Token"),
        "Arm64V3Token" => Some("Arm64V3Token"),
        "Wasm128Token" => Some("Wasm128Token"),
        "Wasm128RelaxedToken" => Some("Wasm128RelaxedToken"),
        "Wasm128RelaxedFmaToken" => Some("Wasm128RelaxedFmaToken"),
        _ => None,
    }
}

/// Map any recognized feature-bearing trait/bound spelling to a `'static` string slice.
pub(crate) fn known_trait_str(name: &str) -> Option<&'static str> {
    match name {
        "Has128BitSimd" => Some("Has128BitSimd"),
        "Has256BitSimd" => Some("Has256BitSimd"),
        "Has512BitSimd" => Some("Has512BitSimd"),
        "HasX64V2" => Some("HasX64V2"),
        "HasX64V4" => Some("HasX64V4"),
        "HasNeon" => Some("HasNeon"),
        "HasNeonAes" => Some("HasNeonAes"),
        "HasNeonSha3" => Some("HasNeonSha3"),
        "HasArm64V2" => Some("HasArm64V2"),
        "HasArm64V3" => Some("HasArm64V3"),
        other => known_token_str(other).filter(|t| trait_to_features(t).is_some()),
    }
}

/// Abstraction over borrowed `&Ident` (zero-allocation production path) and
/// `String` (unit tests).
pub(crate) trait TraitNameRef {
    fn with_str<R>(&self, f: impl FnOnce(&str) -> R) -> R;
}

impl TraitNameRef for String {
    #[inline]
    fn with_str<R>(&self, f: impl FnOnce(&str) -> R) -> R {
        f(self.as_str())
    }
}

impl TraitNameRef for Ident {
    #[inline]
    fn with_str<R>(&self, f: impl FnOnce(&str) -> R) -> R {
        with_ident_str(self, f)
    }
}

impl TraitNameRef for &Ident {
    #[inline]
    fn with_str<R>(&self, f: impl FnOnce(&str) -> R) -> R {
        with_ident_str(self, f)
    }
}

pub(crate) enum TokenTypeInfo<'a> {
    /// Concrete token type (e.g., `Avx2FmaToken`, `X64V3Token`)
    Concrete(&'static str),
    /// `impl Trait` with borrowed trait identifiers (e.g., `impl HasX64V2`)
    ImplTrait(SmallVec<&'a Ident, 4>),
    /// Generic type parameter or unknown path identifier (e.g., `T`)
    Generic(&'a Ident),
}

/// Extract token type information from a type without heap allocations.
pub(crate) fn extract_token_type_info(ty: &Type) -> Option<TokenTypeInfo<'_>> {
    match ty {
        Type::Path(type_path) => type_path.path.segments.last().map(|seg| {
            if let Some(known) = with_ident_str(&seg.ident, known_token_str) {
                TokenTypeInfo::Concrete(known)
            } else {
                TokenTypeInfo::Generic(&seg.ident)
            }
        }),
        Type::Reference(type_ref) => extract_token_type_info(&type_ref.elem),
        Type::ImplTrait(impl_trait) => {
            let traits = extract_trait_idents_from_bounds(&impl_trait.bounds);
            if traits.is_empty() {
                None
            } else {
                Some(TokenTypeInfo::ImplTrait(traits))
            }
        }
        _ => None,
    }
}

/// Extract trait bound identifiers into a stack `SmallVec`.
pub(crate) fn extract_trait_idents_from_bounds(
    bounds: &syn::punctuated::Punctuated<TypeParamBound, Token![+]>,
) -> SmallVec<&Ident, 4> {
    let mut out = SmallVec::new();
    for bound in bounds {
        if let TypeParamBound::Trait(trait_bound) = bound
            && let Some(seg) = trait_bound.path.segments.last()
        {
            out.push(&seg.ident);
        }
    }
    out
}

/// Every trait bound on a generic type parameter, from its inline bounds and
/// from every `where` predicate that names it, in source order.
pub(crate) fn find_generic_bounds<'a>(
    sig: &'a Signature,
    type_ident: &Ident,
) -> Option<SmallVec<&'a Ident, 4>> {
    let mut traits = SmallVec::new();
    for param in &sig.generics.params {
        if let GenericParam::Type(type_param) = param
            && type_param.ident == *type_ident
        {
            traits.extend(
                extract_trait_idents_from_bounds(&type_param.bounds)
                    .iter()
                    .copied(),
            );
        }
    }
    if let Some(where_clause) = &sig.generics.where_clause {
        for predicate in &where_clause.predicates {
            if let syn::WherePredicate::Type(pred_type) = predicate
                && let Type::Path(type_path) = &pred_type.bounded_ty
                && type_path.path.segments.len() == 1
                && let Some(seg) = type_path.path.segments.last()
                && seg.ident == *type_ident
            {
                traits.extend(
                    extract_trait_idents_from_bounds(&pred_type.bounds)
                        .iter()
                        .copied(),
                );
            }
        }
    }
    if traits.is_empty() {
        None
    } else {
        Some(traits)
    }
}

/// Convert trait names to features, collecting all features from all traits.
pub(crate) fn traits_to_features_any<'a, I, S>(
    trait_names: I,
) -> Option<Cow<'static, [&'static str]>>
where
    I: IntoIterator<Item = &'a S>,
    S: TraitNameRef + 'a,
{
    let mut all_features: Cow<'static, [&'static str]> = Cow::Borrowed(&[]);

    for trait_name in trait_names {
        if let Some(features) = trait_name.with_str(trait_to_features) {
            if all_features.is_empty() {
                all_features = Cow::Borrowed(features);
                continue;
            }
            for &feature in features {
                if !all_features.contains(&feature) {
                    all_features.to_mut().push(feature);
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

#[cfg(test)]
pub(crate) fn traits_to_features(trait_names: &[String]) -> Option<Cow<'static, [&'static str]>> {
    traits_to_features_any(trait_names)
}

/// Concrete tokens use generated strings; generic bounds may combine feature
/// sets in arbitrary order, so retain the ordinary join for those.
pub(crate) fn features_csv(token_name: Option<&str>, features: &[&str]) -> Cow<'static, str> {
    token_name
        .and_then(crate::generated::token_to_features_csv)
        .map(Cow::Borrowed)
        .unwrap_or_else(|| Cow::Owned(features.join(",")))
}

/// The archmage tier traits among `trait_names` — i.e. the ones that actually
/// contributed target features, stored inline on the stack.
pub(crate) fn traits_to_tier_traits<'a, I, S>(trait_names: I) -> SmallVec<&'static str, 4>
where
    I: IntoIterator<Item = &'a S>,
    S: TraitNameRef + 'a,
{
    let mut tiers = SmallVec::new();
    for trait_name in trait_names {
        if let Some(known) = trait_name.with_str(known_trait_str)
            && !tiers.contains(&known)
        {
            tiers.push(known);
        }
    }
    tiers
}

/// Trait names that don't map to any CPU features.
pub(crate) const FEATURELESS_TRAIT_NAMES: &[&str] = &["SimdToken", "IntoConcreteToken"];

/// Check if any trait names are featureless (no CPU feature mapping).
pub(crate) fn find_featureless_trait<'a, I, S>(trait_names: I) -> Option<&'static str>
where
    I: IntoIterator<Item = &'a S>,
    S: TraitNameRef + 'a,
{
    for name in trait_names {
        let found = name.with_str(|s| FEATURELESS_TRAIT_NAMES.iter().copied().find(|&fl| fl == s));
        if found.is_some() {
            return found;
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
                TokenTypeInfo::Generic(type_ident) => {
                    if let Some(name) = find_featureless_trait(std::slice::from_ref(type_ident)) {
                        return Some(name);
                    }
                    if let Some(bounds) = find_generic_bounds(sig, type_ident)
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
    pub features: Cow<'static, [&'static str]>,
    /// Target architecture (Some for concrete tokens, None for traits/generics)
    pub target_arch: Option<&'static str>,
    /// Concrete token type name (Some for concrete tokens, None for traits/generics)
    pub token_type_name: Option<&'static str>,
    /// Magetypes width namespace (e.g., "v3", "neon", "wasm128")
    pub magetypes_namespace: Option<&'static str>,
    /// Full type from the function signature (for const tier tag assertion).
    pub token_type: Option<Type>,
    /// The archmage tier traits the features were derived from.
    pub tier_traits: SmallVec<&'static str, 4>,
}

/// Resolve magetypes namespace from a list of trait names.
pub(crate) fn traits_to_magetypes_namespace<'a, I, S>(trait_names: I) -> Option<&'static str>
where
    I: IntoIterator<Item = &'a S>,
    S: TraitNameRef + 'a,
{
    for name in trait_names {
        if let Some(ns) = name.with_str(trait_to_magetypes_namespace) {
            return Some(ns);
        }
    }
    None
}

/// Given trait bound names, return the first matching target architecture.
pub(crate) fn traits_to_arch<'a, I, S>(trait_names: I) -> Option<&'static str>
where
    I: IntoIterator<Item = &'a S>,
    S: TraitNameRef + 'a,
{
    for name in trait_names {
        if let Some(arch) = name.with_str(trait_to_arch) {
            return Some(arch);
        }
    }
    None
}

/// Check whether `ty` resolves to a feature-bearing token in `sig` without cloning ASTs.
pub(crate) fn is_token_type_in_sig(sig: &Signature, ty: &Type) -> bool {
    let Some(info) = extract_token_type_info(ty) else {
        return false;
    };
    match &info {
        TokenTypeInfo::Concrete(name) => token_to_features(name).is_some(),
        TokenTypeInfo::ImplTrait(names) => traits_to_features_any(names).is_some(),
        TokenTypeInfo::Generic(type_ident) => {
            (!sig.generics.params.is_empty() || sig.generics.where_clause.is_some())
                && find_generic_bounds(sig, type_ident)
                    .and_then(|b| traits_to_features_any(&b))
                    .is_some()
        }
    }
}

struct ResolvedTokenType {
    features: Cow<'static, [&'static str]>,
    target_arch: Option<&'static str>,
    token_type_name: Option<&'static str>,
    magetypes_namespace: Option<&'static str>,
    token_type: Option<Type>,
    tier_traits: SmallVec<&'static str, 4>,
}

impl ResolvedTokenType {
    #[inline]
    fn into_param_info(self, ident: Ident) -> TokenParamInfo {
        TokenParamInfo {
            ident,
            features: self.features,
            target_arch: self.target_arch,
            token_type_name: self.token_type_name,
            magetypes_namespace: self.magetypes_namespace,
            token_type: self.token_type,
            tier_traits: self.tier_traits,
        }
    }
}

fn resolve_type_features(sig: &Signature, ty: &Type) -> Option<ResolvedTokenType> {
    let info = extract_token_type_info(ty)?;
    match info {
        TokenTypeInfo::Concrete(name) => {
            let features = token_to_features(name).map(Cow::Borrowed)?;
            let arch = token_to_arch(name);
            let ns = token_to_magetypes_namespace(name);
            Some(ResolvedTokenType {
                features,
                target_arch: arch,
                token_type_name: Some(name),
                magetypes_namespace: ns,
                token_type: Some(ty.clone()),
                tier_traits: SmallVec::new(),
            })
        }
        TokenTypeInfo::ImplTrait(ref trait_names) => {
            let features = traits_to_features_any(trait_names)?;
            let ns = traits_to_magetypes_namespace(trait_names);
            let arch = traits_to_arch(trait_names);
            let tiers = traits_to_tier_traits(trait_names);
            Some(ResolvedTokenType {
                features,
                target_arch: arch,
                token_type_name: None,
                magetypes_namespace: ns,
                token_type: None,
                tier_traits: tiers,
            })
        }
        TokenTypeInfo::Generic(type_ident) => {
            if sig.generics.params.is_empty() && sig.generics.where_clause.is_none() {
                return None;
            }
            let bounds = find_generic_bounds(sig, type_ident)?;
            let features = traits_to_features_any(&bounds)?;
            let ns = traits_to_magetypes_namespace(&bounds);
            let arch = traits_to_arch(&bounds);
            let tiers = traits_to_tier_traits(&bounds);
            Some(ResolvedTokenType {
                features,
                target_arch: arch,
                token_type_name: None,
                magetypes_namespace: ns,
                token_type: None,
                tier_traits: tiers,
            })
        }
    }
}

/// Inspect `_self = Type` receiver type directly without cloning `Signature` or running `parse_quote!`.
pub(crate) fn find_token_in_self_type(sig: &Signature, self_ty: &Type) -> Option<TokenParamInfo> {
    Some(
        resolve_type_features(sig, self_ty)?
            .into_param_info(Ident::new("_self", Span::call_site())),
    )
}

/// Find the first token parameter in a function signature.
pub(crate) fn find_token_param(sig: &Signature) -> Option<TokenParamInfo> {
    for arg in &sig.inputs {
        let FnArg::Typed(PatType { pat, ty, .. }) = arg else {
            continue;
        };
        if let Some(resolved) = resolve_type_features(sig, ty) {
            let ident = match pat.as_ref() {
                Pat::Ident(pat_ident) => Some(pat_ident.ident.clone()),
                Pat::Wild(w) => Some(Ident::new("__archmage_token", w.underscore_token.span)),
                _ => None,
            };
            if let Some(ident) = ident {
                return Some(resolved.into_param_info(ident));
            }
        }
    }
    None
}
