//! Definition grammar -> validated output plan -> emission.
//!
//! The grammar retains selectors and spans without expanding tiers. Resolution
//! owns wildcard expansion and cross-option invariants. Emitters see only concrete
//! outputs. Legacy attribute parsers do not pass through this definition grammar.
mod grammar;
mod resolve;
#[cfg(test)]
mod tests;
use proc_macro2::Span;
use syn::{
    Ident, Path, Token, Visibility,
    ext::IdentExt,
    parse::{Parse, ParseStream},
};

pub(super) use crate::engine::inline::InlinePolicy as Inline;
use crate::tiers::{TierDescriptor, find_tier};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Form {
    /// Implementation selected only for a central dispatcher, never an exposed API.
    Hidden,
    Direct,
    Proof,
}

#[derive(Clone)]
pub(super) struct Selection {
    pub tier: &'static TierDescriptor,
    pub form: Form,
    pub gate: Option<String>,
    pub visibility: Option<Visibility>,
    pub inline: Option<Inline>,
    pub span: Span,
}

pub(super) struct Rename {
    pub tier: &'static TierDescriptor,
    pub form: Form,
    pub path: Path,
}

#[derive(Default)]
pub(super) struct Options {
    pub body_inline: Option<Inline>,
    pub tier: Option<&'static TierDescriptor>,
    pub wrap: bool,
    pub names: Vec<Rename>,
    pub in_impl: bool,
    pub in_trait: bool,
    pub self_type: Option<syn::Type>,
    pub imports: crate::common::SharedOptions,
    pub defines: Vec<Ident>,
}

/// Validated definition options plus concrete outputs; no wildcard interpretation
/// belongs in code generation.
pub(super) struct Args {
    pub options: Options,
    pub family: bool,
    pub selections: Vec<Selection>,
    pub dispatcher: Option<(Option<Visibility>, Option<Inline>)>,
}

pub(super) const DEFAULTS: &[&str] = &["v3", "neon", "wasm128", "scalar"];

pub(super) fn selector(name: &str, span: Span) -> syn::Result<(&'static TierDescriptor, Form)> {
    let name = name.trim_start_matches('_');
    let (name, form) = name
        .strip_suffix("_t")
        .map_or((name, Form::Direct), |n| (n, Form::Proof));
    let tier = find_tier(name)
        .filter(|t| t.name != "default")
        .ok_or_else(|| syn::Error::new(span, format!("unknown attune tier `{name}`")))?;
    Ok((tier, form))
}

pub(super) fn gate(input: ParseStream) -> syn::Result<Option<String>> {
    if !input.peek(syn::token::Paren) {
        return Ok(None);
    }
    let inner;
    syn::parenthesized!(inner in input);
    let name: Ident = inner.parse()?;
    let value = if name == "cfg" {
        let nested;
        syn::parenthesized!(nested in inner);
        let feature: Ident = nested.parse()?;
        if !nested.is_empty() {
            return Err(nested.error("expected one Cargo feature"));
        }
        feature.to_string()
    } else {
        name.to_string()
    };
    if !inner.is_empty() {
        return Err(inner.error("expected one Cargo feature"));
    }
    Ok(Some(value))
}

pub(super) fn parse_names(input: ParseStream) -> syn::Result<Vec<Rename>> {
    let inner;
    syn::parenthesized!(inner in input);
    let mut names: Vec<Rename> = Vec::new();
    while !inner.is_empty() {
        let name: Ident = inner.parse()?;
        let (tier, form) = selector(&name.to_string(), name.span())?;
        inner.parse::<Token![=]>()?;
        let path = inner.parse()?;
        if names
            .iter()
            .any(|n| n.tier.name == tier.name && n.form == form)
        {
            return Err(syn::Error::new(name.span(), "duplicate name mapping"));
        }
        names.push(Rename { tier, form, path });
        if !inner.is_empty() {
            inner.parse::<Token![,]>()?;
        }
    }
    Ok(names)
}

impl Parse for Args {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        grammar::Definition::parse(input)?.resolve()
    }
}

fn parse_inline(input: ParseStream) -> syn::Result<Inline> {
    let inner;
    syn::parenthesized!(inner in input);
    let value: Ident = inner.parse()?;
    let policy = match value.to_string().as_str() {
        "default" => Inline::Default,
        "none" => Inline::None,
        "hint" => Inline::Hint,
        "always" => Inline::Always,
        "never" => Inline::Never,
        _ => {
            return Err(syn::Error::new(
                value.span(),
                "expected default, none, hint, always, or never",
            ));
        }
    };
    if !inner.is_empty() {
        return Err(inner.error("expected one inline policy"));
    }
    Ok(policy)
}
