//! Syntax is lowered once to explicit outputs; emitters never interpret wildcards.
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
}

pub(super) struct Rename {
    pub tier: &'static TierDescriptor,
    pub form: Form,
    pub path: Path,
}

#[derive(Default)]
pub(super) struct Args {
    pub body_inline: Option<Inline>,
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
                "inline" => {
                    if args.body_inline.replace(parse_inline(input)?).is_some() {
                        return Err(syn::Error::new(name.span(), "duplicate body inline policy"));
                    }
                }
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
        if args.wrap && args.tier.is_some() {
            return Err(input.error(
                "wrap derives its features from the proof parameter; omit the explicit tier",
            ));
        }
        if !args.family && !args.names.is_empty() {
            return Err(input.error("names(...) on a definition requires make(...)"));
        }
        if args.family && args.imports.cfg_feature.is_some() {
            return Err(input.error("gate a whole family with an outer #[cfg(...)], or gate a selector with _v3(feature)"));
        }
        if args.wrap && !args.defines.is_empty() {
            return Err(input.error(
                "define(...) requires a concrete tier; use a direct tier body or a family",
            ));
        }
        if args.family && (args.wrap || args.tier.is_some()) {
            return Err(input.error("make(...) cannot be combined with wrap or a single tier"));
        }
        if args.in_impl && (args.in_trait || args.self_type.is_some()) {
            return Err(input.error("in_impl and in_trait/_self describe different placements"));
        }
        if (args.in_trait || args.self_type.is_some()) && args.body_inline == Some(Inline::Default)
        {
            return Err(input.error("inline(default) cannot infer trait visibility; choose inline(hint), inline(none), or inline(never)"));
        }
        Ok(args)
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
                inline = Some(parse_inline(input)?);
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
                    if self.dispatcher.is_some() {
                        return Err(input.error("duplicate dispatcher selector"));
                    }
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
            } else if text == "_" || text == "dispatch" {
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
        // Dispatcher-only families have the default tier set even when they
        // carry modifiers. Normalize it before applying additions/removals so
        // private implementation choices never become exposed direct outputs.
        if self.dispatcher.is_some() && self.selections.is_empty() {
            self.selections
                .extend(DEFAULTS.iter().map(|name| Selection {
                    tier: find_tier(name).unwrap(),
                    form: Form::Hidden,
                    gate: None,
                    visibility: None,
                    inline: None,
                }));
        }
        for (remove, tier, form, gate, span) in modifiers {
            if remove {
                if self.dispatcher.is_some() && tier.name == "scalar" && form == Form::Direct {
                    return Err(syn::Error::new(
                        span,
                        "a dispatcher requires its scalar fallback",
                    ));
                }
                self.selections.retain(|s| {
                    s.tier.name != tier.name || (form == Form::Proof && s.form != form)
                });
            } else {
                if wildcard_forms.is_empty() {
                    if self.dispatcher.is_none() {
                        return Err(syn::Error::new(
                            span,
                            "+tier requires a wildcard output form or a dispatcher",
                        ));
                    }
                    if form == Form::Proof {
                        return Err(syn::Error::new(
                            span,
                            "use +tier for a hidden dispatcher tier, or tier_t to expose a proof wrapper",
                        ));
                    }
                    self.selections.push(Selection {
                        tier,
                        form: Form::Hidden,
                        gate,
                        visibility: None,
                        inline: None,
                    });
                    continue;
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
        if self.dispatcher.is_some()
            && unique
                .iter()
                .any(|s| s.tier.name == "scalar" && s.gate.is_some())
        {
            return Err(input.error("a dispatcher requires an ungated scalar fallback"));
        }
        self.selections = unique;
        Ok(())
    }
}
