//! Token parameter discovery for #[arcane] and #[rite].
//!
//! Finds token parameters in function signatures, extracts feature sets,
//! and diagnoses invalid token types without per-parameter heap allocations.

use std::borrow::Cow;

use proc_macro2::Span;
use syn::{FnArg, GenericParam, Ident, PatType, Signature, Token, Type, TypeParamBound};

use crate::common::with_ident_str;
use crate::generated::{
    known_token_str, known_trait_str, token_to_arch, token_to_features,
    token_to_magetypes_namespace, trait_to_arch, trait_to_features, trait_to_magetypes_namespace,
};

pub(crate) enum TokenTypeInfo<'a> {
    /// Concrete token type (e.g., `Avx2FmaToken`, `X64V3Token`)
    Concrete(&'static str),
    /// `impl Trait` with borrowed trait identifiers (e.g., `impl HasX64V2`)
    ImplTrait(Vec<&'a Ident>),
    /// Generic type parameter or unknown path identifier (e.g., `T`)
    Generic(&'a Ident),
}

/// Extract token type information from a type.
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

/// Push trait bound identifiers from a punctuated bound list into `out`.
fn push_trait_idents_from_bounds<'a>(
    out: &mut Vec<&'a Ident>,
    bounds: &'a syn::punctuated::Punctuated<TypeParamBound, Token![+]>,
) {
    for bound in bounds {
        if let TypeParamBound::Trait(trait_bound) = bound
            && let Some(seg) = trait_bound.path.segments.last()
        {
            out.push(&seg.ident);
        }
    }
}

/// Extract trait bound identifiers from a punctuated bound list.
pub(crate) fn extract_trait_idents_from_bounds(
    bounds: &syn::punctuated::Punctuated<TypeParamBound, Token![+]>,
) -> Vec<&Ident> {
    let mut out = Vec::new();
    push_trait_idents_from_bounds(&mut out, bounds);
    out
}

/// Every trait bound on a generic type parameter, from its inline bounds and
/// from every `where` predicate that names it, in source order.
///
/// Inline bounds and `where` predicates are unioned rather than short-circuited
/// so a token bound split across both positions (e.g. `T: Copy` inline plus
/// `where T: HasX64V2`, issue #122) is still discovered.
pub(crate) fn find_generic_bounds<'a>(
    sig: &'a Signature,
    type_ident: &Ident,
) -> Option<Vec<&'a Ident>> {
    with_ident_str(type_ident, |target| {
        let mut traits = Vec::new();
        for param in &sig.generics.params {
            if let GenericParam::Type(type_param) = param
                && with_ident_str(&type_param.ident, |s| s == target)
            {
                push_trait_idents_from_bounds(&mut traits, &type_param.bounds);
            }
        }
        if let Some(where_clause) = &sig.generics.where_clause {
            for predicate in &where_clause.predicates {
                if let syn::WherePredicate::Type(pred_type) = predicate
                    && let Type::Path(type_path) = &pred_type.bounded_ty
                    && type_path.path.segments.len() == 1
                    && let Some(seg) = type_path.path.segments.last()
                    && with_ident_str(&seg.ident, |s| s == target)
                {
                    push_trait_idents_from_bounds(&mut traits, &pred_type.bounds);
                }
            }
        }
        if traits.is_empty() {
            None
        } else {
            Some(traits)
        }
    })
}

/// Short-circuit check for whether any bound in `bounds` maps to a non-empty
/// feature list, without allocating a `Vec` or constructing a feature union.
fn bounds_have_any_trait_features(
    bounds: &syn::punctuated::Punctuated<TypeParamBound, Token![+]>,
) -> bool {
    bounds.iter().any(|bound| {
        if let TypeParamBound::Trait(trait_bound) = bound
            && let Some(seg) = trait_bound.path.segments.last()
        {
            with_ident_str(&seg.ident, |name| {
                trait_to_features(name).is_some_and(|feats| !feats.is_empty())
            })
        } else {
            false
        }
    })
}

/// Short-circuit check for whether generic `type_ident` has any feature-bearing
/// inline or `where` trait bound in `sig`, without allocating.
fn generic_has_any_trait_features(sig: &Signature, type_ident: &Ident) -> bool {
    if sig.generics.params.is_empty() && sig.generics.where_clause.is_none() {
        return false;
    }
    with_ident_str(type_ident, |target| {
        for param in &sig.generics.params {
            if let GenericParam::Type(type_param) = param
                && with_ident_str(&type_param.ident, |s| s == target)
                && bounds_have_any_trait_features(&type_param.bounds)
            {
                return true;
            }
        }
        if let Some(where_clause) = &sig.generics.where_clause {
            for predicate in &where_clause.predicates {
                if let syn::WherePredicate::Type(pred_type) = predicate
                    && let Type::Path(type_path) = &pred_type.bounded_ty
                    && type_path.path.segments.len() == 1
                    && let Some(seg) = type_path.path.segments.last()
                    && with_ident_str(&seg.ident, |s| s == target)
                    && bounds_have_any_trait_features(&pred_type.bounds)
                {
                    return true;
                }
            }
        }
        false
    })
}

/// Convert trait bound identifiers to features, collecting all features from all traits.
pub(crate) fn traits_to_features(trait_names: &[&Ident]) -> Option<Cow<'static, [&'static str]>> {
    // Registry lists are unique (tested); only a union that adds features
    // needs owned storage. Preserve first-seen order for generated attributes.
    let mut all_features: Cow<'static, [&'static str]> = Cow::Borrowed(&[]);

    for trait_name in trait_names {
        if let Some(features) = with_ident_str(trait_name, trait_to_features) {
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

/// Concrete tokens use generated strings; generic bounds may combine feature
/// sets in arbitrary order, so retain the ordinary join for those.
pub(crate) fn features_csv(token_name: Option<&str>, features: &[&str]) -> Cow<'static, str> {
    token_name
        .and_then(crate::generated::token_to_features_csv)
        .map(Cow::Borrowed)
        .unwrap_or_else(|| Cow::Owned(features.join(",")))
}

/// The archmage tier traits among `trait_names` — i.e. the ones that actually
/// contributed target features.
///
/// `#[arcane]`/`#[rite]` resolve a trait bound **by name**, so on its own a bound
/// like `impl HasX64V2` proves nothing: a downstream crate can declare its own
/// trait of that name and receive the tier's `#[target_feature]` with no proof
/// at all. The emitted code re-states these names through an absolute
/// `::archmage::` path so rustc checks that the bound really is archmage's.
/// See `gen_tier_trait_assertion` in `common.rs`.
pub(crate) fn traits_to_tier_traits(trait_names: &[&Ident]) -> Vec<&'static str> {
    let mut tiers = Vec::new();
    for trait_name in trait_names {
        if let Some(known) = with_ident_str(trait_name, known_trait_str)
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
pub(crate) fn find_featureless_trait(trait_names: &[&Ident]) -> Option<&'static str> {
    for name in trait_names {
        let found = with_ident_str(name, |s| {
            FEATURELESS_TRAIT_NAMES.iter().copied().find(|&fl| fl == s)
        });
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
                    if let Some(name) = find_featureless_trait(&[*type_ident]) {
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
    /// Index of the parameter in `sig.inputs`, or `None` when discovered from `_self = Type`.
    pub param_index: Option<usize>,
    /// Target features to enable (e.g., `["avx2", "fma"]`)
    pub features: Cow<'static, [&'static str]>,
    /// Target architecture (Some for concrete tokens, None for traits/generics)
    pub target_arch: Option<&'static str>,
    /// Concrete token type name (Some for concrete tokens, None for traits/generics)
    pub token_type_name: Option<&'static str>,
    /// Magetypes width namespace (e.g., "v3", "neon", "wasm128")
    pub magetypes_namespace: Option<&'static str>,
    /// Full type from the function signature (for const tier tag assertion).
    /// Set for concrete token types, None for trait/generic bounds.
    pub token_type: Option<Type>,
    /// The archmage tier traits the features were derived from. Empty for a
    /// concrete token (which is checked by its tier-tag const instead); one or
    /// more names for a trait or generic bound, which the emitted code
    /// re-states through `::archmage::` so a same-named local trait cannot
    /// stand in for archmage's.
    pub tier_traits: Vec<&'static str>,
}

/// Resolve magetypes namespace from a list of trait names.
/// Returns the first matching namespace found.
pub(crate) fn traits_to_magetypes_namespace(trait_names: &[&Ident]) -> Option<&'static str> {
    for name in trait_names {
        if let Some(ns) = with_ident_str(name, trait_to_magetypes_namespace) {
            return Some(ns);
        }
    }
    None
}

/// Given trait bound names, return the first matching target architecture.
pub(crate) fn traits_to_arch(trait_names: &[&Ident]) -> Option<&'static str> {
    for name in trait_names {
        if let Some(arch) = with_ident_str(name, trait_to_arch) {
            return Some(arch);
        }
    }
    None
}

/// Check whether `ty` resolves to a feature-bearing token in `sig` without
/// cloning ASTs or heap-allocating bound lists or feature unions.
pub(crate) fn is_token_type_in_sig(sig: &Signature, ty: &Type) -> bool {
    match ty {
        Type::Path(type_path) => {
            let Some(seg) = type_path.path.segments.last() else {
                return false;
            };
            if let Some(known) = with_ident_str(&seg.ident, known_token_str) {
                token_to_features(known).is_some()
            } else {
                generic_has_any_trait_features(sig, &seg.ident)
            }
        }
        Type::Reference(type_ref) => is_token_type_in_sig(sig, &type_ref.elem),
        Type::ImplTrait(impl_trait) => bounds_have_any_trait_features(&impl_trait.bounds),
        _ => false,
    }
}

struct ResolvedTokenType {
    features: Cow<'static, [&'static str]>,
    target_arch: Option<&'static str>,
    token_type_name: Option<&'static str>,
    magetypes_namespace: Option<&'static str>,
    token_type: Option<Type>,
    tier_traits: Vec<&'static str>,
}

impl ResolvedTokenType {
    #[inline]
    fn into_param_info(self, ident: Ident, param_index: Option<usize>) -> TokenParamInfo {
        TokenParamInfo {
            ident,
            param_index,
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
                tier_traits: Vec::new(),
            })
        }
        TokenTypeInfo::ImplTrait(ref trait_names) => {
            let features = traits_to_features(trait_names)?;
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
            let features = traits_to_features(&bounds)?;
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
            .into_param_info(Ident::new("_self", Span::call_site()), None),
    )
}

/// Find the first token parameter in a function signature.
pub(crate) fn find_token_param(sig: &Signature) -> Option<TokenParamInfo> {
    for (idx, arg) in sig.inputs.iter().enumerate() {
        let FnArg::Typed(PatType { pat, ty, .. }) = arg else {
            // A receiver is never the token. Methods are supported: the
            // sibling expansion keeps `self`, and the nested expansion
            // lowers it to `_self` (see `common::nested_self_param`).
            continue;
        };
        if let Some(resolved) = resolve_type_features(sig, ty) {
            let ident = match pat.as_ref() {
                syn::Pat::Ident(pat_ident) => Some(pat_ident.ident.clone()),
                syn::Pat::Wild(w) => Some(Ident::new("__archmage_token", w.underscore_token.span)),
                _ => None,
            };
            if let Some(ident) = ident {
                return Some(resolved.into_param_info(ident, Some(idx)));
            }
        }
    }
    None
}
