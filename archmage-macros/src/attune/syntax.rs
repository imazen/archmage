//! Syntax is lowered once to explicit outputs; emitters never interpret wildcards.
use proc_macro2::Span;
use syn::{
    Ident, Path, Token, Visibility,
    ext::IdentExt,
    parse::{Parse, ParseStream},
};

use crate::tiers::{TierDescriptor, find_tier};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Form {
    Direct,
    Proof,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Inline {
    Hint,
    Always,
    Never,
}

#[derive(Clone)]
pub(super) struct Selection {
    pub tier: &'static TierDescriptor,
    pub form: Form,
    pub gate: Option<String>,
    pub visibility: Option<Visibility>,
    pub inline: Option<Inline>,
}

pub(super) struct Rename {
    pub tier: &'static TierDescriptor,
    pub form: Form,
    pub path: Path,
}

#[derive(Default)]
pub(super) struct Args {
    pub tier: Option<&'static TierDescriptor>,
    pub wrap: bool,
    pub family: bool,
    pub selections: Vec<Selection>,
    pub dispatcher: Option<(Option<Visibility>, Option<Inline>)>,
    pub names: Vec<Rename>,
    pub in_impl: bool,
    pub in_trait: bool,
    pub self_type: Option<syn::Type>,
    pub imports: crate::common::SharedOptions,
    pub defines: Vec<Ident>,
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
        let mut args = Self::default();
        while !input.is_empty() {
            let name: Ident = input.parse()?;
            match name.to_string().as_str() {
                "make" => {
                    if args.family {
                        return Err(syn::Error::new(name.span(), "duplicate make(...)"));
                    }
                    args.family = true;
                    let inner;
                    syn::parenthesized!(inner in input);
                    args.parse_make(&inner)?;
                }
                "wrap" => args.wrap = true,
                "in_impl" => args.in_impl = true,
                "in_trait" | "nested" => args.in_trait = true,
                "_self" => {
                    input.parse::<Token![=]>()?;
                    args.self_type = Some(input.parse()?);
                }
                "names" => args.names = parse_names(input)?,
                "define" => {
                    let inner;
                    syn::parenthesized!(inner in input);
                    args.defines = inner
                        .parse_terminated(Ident::parse, Token![,])?
                        .into_iter()
                        .collect();
                }
                _ if crate::common::parse_shared_option(&name, input, &mut args.imports)? => {}
                _ => {
                    let (tier, form) = selector(&name.to_string(), name.span())?;
                    if args.tier.replace(tier).is_some() || form != Form::Direct {
                        return Err(syn::Error::new(
                            name.span(),
                            "one tier is allowed here; use make(...) for a family",
                        ));
                    }
                }
            }
            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }
        if args.family && (args.wrap || args.tier.is_some()) {
            return Err(input.error("make(...) cannot be combined with wrap or a single tier"));
        }
        if args.in_impl && (args.in_trait || args.self_type.is_some()) {
            return Err(input.error("in_impl and in_trait/_self describe different placements"));
        }
        Ok(args)
    }
}

impl Args {
    fn parse_make(&mut self, input: ParseStream) -> syn::Result<()> {
        let mut modifiers = Vec::new();
        let mut wildcard_forms = Vec::new();
        while !input.is_empty() {
            let visibility = if input.peek(Token![pub]) {
                Some(input.parse()?)
            } else {
                None
            };
            let mut inline = None;
            if input.peek(Ident) && input.fork().parse::<Ident>()? == "inline" {
                input.parse::<Ident>()?;
                let inner;
                syn::parenthesized!(inner in input);
                let value: Ident = inner.parse()?;
                inline = Some(match value.to_string().as_str() {
                    "hint" => Inline::Hint,
                    "always" => Inline::Always,
                    "never" => Inline::Never,
                    _ => {
                        return Err(syn::Error::new(
                            value.span(),
                            "expected hint, always, or never",
                        ));
                    }
                });
                if !inner.is_empty() {
                    return Err(inner.error("expected one inline policy"));
                }
            }
            let remove = input.peek(Token![-]);
            let add = input.peek(Token![+]);
            if remove {
                input.parse::<Token![-]>()?;
            }
            if add {
                input.parse::<Token![+]>()?;
            }
            let name = input.call(Ident::parse_any)?;
            let text = name.to_string();
            if text == "all" || (text == "_" && input.peek(Token![*])) {
                if remove || add {
                    return Err(syn::Error::new(
                        name.span(),
                        "modifiers require a named tier",
                    ));
                }
                let forms = if text == "all" {
                    self.dispatcher = Some((visibility.clone(), inline));
                    vec![Form::Direct, Form::Proof]
                } else {
                    input.parse::<Token![*]>()?;
                    if input.peek(Ident) {
                        let suffix: Ident = input.parse()?;
                        if suffix != "_t" {
                            return Err(syn::Error::new(suffix.span(), "expected _*_t"));
                        }
                        vec![Form::Proof]
                    } else {
                        vec![Form::Direct]
                    }
                };
                for form in forms {
                    wildcard_forms.push((form, visibility.clone(), inline));
                    for name in DEFAULTS {
                        self.selections.push(Selection {
                            tier: find_tier(name).unwrap(),
                            form,
                            gate: None,
                            visibility: visibility.clone(),
                            inline,
                        });
                    }
                }
            } else if text == "_" {
                if remove || add || self.dispatcher.is_some() {
                    return Err(syn::Error::new(
                        name.span(),
                        "duplicate or modified dispatcher selector",
                    ));
                }
                self.dispatcher = Some((visibility, inline));
            } else {
                let (tier, form) = selector(&text, name.span())?;
                let gate = gate(input)?;
                if remove || add {
                    modifiers.push((remove, tier, form, gate, name.span()));
                } else {
                    self.selections.push(Selection {
                        tier,
                        form,
                        gate,
                        visibility,
                        inline,
                    });
                }
            }
            if !input.is_empty() {
                input.parse::<Token![,]>()?;
            }
        }
        for (remove, tier, form, gate, span) in modifiers {
            if remove {
                self.selections.retain(|s| {
                    s.tier.name != tier.name || (form == Form::Proof && s.form != form)
                });
            } else {
                if wildcard_forms.is_empty() {
                    return Err(syn::Error::new(
                        span,
                        "+tier requires a wildcard output form",
                    ));
                }
                for (form, visibility, inline) in &wildcard_forms {
                    self.selections.push(Selection {
                        tier,
                        form: *form,
                        gate: gate.clone(),
                        visibility: visibility.clone(),
                        inline: *inline,
                    });
                }
            }
        }
        if self.selections.is_empty() && self.dispatcher.is_none() {
            return Err(input.error("make(...) selects no outputs"));
        }
        let mut unique: Vec<Selection> = Vec::new();
        for selection in self.selections.drain(..) {
            if let Some(old) = unique
                .iter()
                .find(|s| s.tier.name == selection.tier.name && s.form == selection.form)
            {
                let old_vis = &old.visibility;
                let new_vis = &selection.visibility;
                if old.gate != selection.gate
                    || old.inline != selection.inline
                    || quote::quote!(#old_vis).to_string() != quote::quote!(#new_vis).to_string()
                {
                    return Err(input.error("conflicting policies for the same output"));
                }
            } else {
                unique.push(selection);
            }
        }
        self.selections = unique;
        Ok(())
    }
}
