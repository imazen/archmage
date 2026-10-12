//! Read the written declaration. Do not expand wildcards or resolve conflicts
//! between outputs here: those decisions belong to `resolve`.
use super::*;

pub(super) struct Definition {
    pub options: Options,
    pub family: bool,
    pub outputs: Vec<Output>,
    pub span: Span,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Action {
    Select,
    Add,
    Remove,
}

#[derive(Clone, Copy)]
pub(super) enum Target {
    Tier(&'static TierDescriptor, Form),
    Wildcard(Form),
    All,
    Dispatcher,
}

#[derive(Default)]
pub(super) struct OutputOptions {
    pub gate: Option<String>,
    pub visibility: Option<Visibility>,
    pub inline: Option<Inline>,
}

pub(super) struct Output {
    pub target: Target,
    pub action: Action,
    pub options: OutputOptions,
    pub span: Span,
}

// Fixed-size duplicate tracking: no map or per-option allocation. Alias spellings
// share a key so they cannot accidentally overwrite one another.
#[derive(Clone, Copy)]
enum Key {
    Inline,
    Make,
    Wrap,
    Impl,
    Trait,
    SelfType,
    Names,
    Define,
    Intrinsics,
    Magetypes,
    Cfg,
    Tier,
}

#[derive(Default)]
struct Seen(u16);

impl Seen {
    fn insert(&mut self, key: Key, name: &Ident) -> syn::Result<()> {
        let bit = 1 << key as u16;
        if self.0 & bit != 0 {
            let message = match key {
                Key::Inline => "duplicate body inline policy".to_owned(),
                Key::Make => "duplicate make(...)".to_owned(),
                _ => format!("duplicate `{name}` option"),
            };
            return Err(syn::Error::new(name.span(), message));
        }
        self.0 |= bit;
        Ok(())
    }
}

impl Parse for Definition {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let mut definition = Self {
            options: Options::default(),
            family: false,
            outputs: Vec::new(),
            span: input.span(),
        };
        let mut seen = Seen::default();
        let mut grouped = false;
        let mut flat = false;
        while !input.is_empty() {
            // Modifier punctuation belongs to the output grammar, not options.
            if input.peek(Token![+]) || input.peek(Token![-]) {
                if grouped {
                    return Err(input.error("keep all outputs inside make(...), or omit make(...)"));
                }
                flat = true;
                definition.outputs.push(Output::parse(input, false)?);
            } else {
                let name = input.call(Ident::parse_any)?;
                let spelling = name.to_string();
                let key = match spelling.as_str() {
                    "inline" => Some(Key::Inline),
                    "make" => Some(Key::Make),
                    "wrap" => Some(Key::Wrap),
                    "in_impl" => Some(Key::Impl),
                    "in_trait" | "nested" => Some(Key::Trait),
                    "_self" => Some(Key::SelfType),
                    "names" => Some(Key::Names),
                    "define" => Some(Key::Define),
                    "import_intrinsics" => Some(Key::Intrinsics),
                    "import_magetypes" => Some(Key::Magetypes),
                    "cfg" => Some(Key::Cfg),
                    _ => None,
                };
                if let Some(key) = key {
                    seen.insert(key, &name)?;
                    let opts = &mut definition.options;
                    match key {
                        Key::Inline => opts.body_inline = Some(parse_inline(input)?),
                        Key::Make => {
                            if flat {
                                return Err(syn::Error::new(
                                    name.span(),
                                    "keep all outputs inside make(...), or omit make(...)",
                                ));
                            }
                            grouped = true;
                            let inner;
                            syn::parenthesized!(inner in input);
                            while !inner.is_empty() {
                                definition.outputs.push(Output::parse(&inner, true)?);
                                comma(&inner)?;
                            }
                        }
                        Key::Wrap => opts.wrap = true,
                        Key::Impl => opts.in_impl = true,
                        Key::Trait => opts.in_trait = true,
                        Key::SelfType => {
                            input.parse::<Token![=]>()?;
                            opts.self_type = Some(input.parse()?);
                        }
                        Key::Names => opts.names = parse_names(input)?,
                        Key::Define => {
                            let inner;
                            syn::parenthesized!(inner in input);
                            opts.defines = inner
                                .parse_terminated(Ident::parse, Token![,])?
                                .into_iter()
                                .collect();
                        }
                        Key::Intrinsics | Key::Magetypes | Key::Cfg => {
                            crate::common::parse_shared_option(&name, input, &mut opts.imports)?;
                        }
                        Key::Tier => unreachable!("tier is handled below"),
                    }
                } else if spelling.starts_with('_')
                    || matches!(spelling.as_str(), "dispatch" | "all")
                {
                    if grouped {
                        return Err(syn::Error::new(
                            name.span(),
                            "keep all outputs inside make(...), or omit make(...)",
                        ));
                    }
                    flat = true;
                    definition.outputs.push(Output::after_name(
                        input,
                        name,
                        Action::Select,
                        OutputOptions::default(),
                        false,
                    )?);
                } else {
                    let (tier, form) = selector(&spelling, name.span())?;
                    if form != Form::Direct {
                        return Err(syn::Error::new(
                            name.span(),
                            "use an underscored output selector, such as _v3_t",
                        ));
                    }
                    seen.insert(Key::Tier, &name)?;
                    definition.options.tier = Some(tier);
                }
            }
            comma(input)?;
        }
        definition.family = grouped || flat;
        Ok(definition)
    }
}

fn comma(input: ParseStream) -> syn::Result<()> {
    if !input.is_empty() {
        input.parse::<Token![,]>()?;
    }
    Ok(())
}

impl Output {
    fn parse(input: ParseStream, grouped: bool) -> syn::Result<Self> {
        let mut options = OutputOptions::default();
        // Compatibility for the existing draft's positional make(...) grammar.
        // Flat syntax only accepts selector-local options.
        if grouped {
            if input.peek(Token![pub]) {
                options.visibility = Some(input.parse()?);
            }
            if input.peek(Ident) && input.fork().parse::<Ident>()? == "inline" {
                input.parse::<Ident>()?;
                options.inline = Some(parse_inline(input)?);
            }
        }
        let action = if input.peek(Token![+]) {
            input.parse::<Token![+]>()?;
            Action::Add
        } else if input.peek(Token![-]) {
            input.parse::<Token![-]>()?;
            Action::Remove
        } else {
            Action::Select
        };
        let name = input.call(Ident::parse_any)?;
        Self::after_name(input, name, action, options, grouped)
    }

    fn after_name(
        input: ParseStream,
        name: Ident,
        action: Action,
        mut options: OutputOptions,
        grouped: bool,
    ) -> syn::Result<Self> {
        let text = name.to_string();
        if text.starts_with("__") {
            return Err(syn::Error::new(
                name.span(),
                "output selectors use one leading underscore",
            ));
        }
        let target = match text.as_str() {
            "_" if input.peek(Token![*]) => {
                input.parse::<Token![*]>()?;
                let form = if input.peek(Ident) {
                    let suffix: Ident = input.parse()?;
                    if suffix != "_t" {
                        return Err(syn::Error::new(suffix.span(), "expected _*_t"));
                    }
                    Form::Proof
                } else {
                    Form::Direct
                };
                Target::Wildcard(form)
            }
            "_" | "dispatch" => Target::Dispatcher,
            "all" => Target::All,
            _ => {
                let (tier, form) = selector(&text, name.span())?;
                Target::Tier(tier, form)
            }
        };
        if input.peek(syn::token::Paren) {
            let inner;
            syn::parenthesized!(inner in input);
            let mut first = true;
            while !inner.is_empty() {
                let span = inner.span();
                if inner.peek(Token![pub]) {
                    let visibility = inner.parse()?;
                    set(&mut options.visibility, visibility, span, "visibility")?;
                } else {
                    let key: Ident = inner.parse()?;
                    match key.to_string().as_str() {
                        "inline" => {
                            set(&mut options.inline, parse_inline(&inner)?, span, "inline")?
                        }
                        "cfg" => {
                            let feature;
                            syn::parenthesized!(feature in inner);
                            let name: Ident = feature.parse()?;
                            if !feature.is_empty() {
                                return Err(feature.error("expected one Cargo feature"));
                            }
                            set(&mut options.gate, name.to_string(), span, "cfg")?;
                        }
                        // Preserve the old make(_v3(feature)) shorthand only.
                        _ if grouped
                            && matches!(target, Target::Tier(..))
                            && inner.is_empty()
                            && first =>
                        {
                            options.gate = Some(key.to_string());
                        }
                        _ => {
                            return Err(syn::Error::new(
                                span,
                                "expected pub visibility, cfg(feature), or inline(policy)",
                            ));
                        }
                    }
                }
                first = false;
                comma(&inner)?;
            }
        }
        Ok(Self {
            target,
            action,
            options,
            span: name.span(),
        })
    }
}

fn set<T>(slot: &mut Option<T>, value: T, span: Span, name: &str) -> syn::Result<()> {
    if slot.replace(value).is_some() {
        return Err(syn::Error::new(span, format!("duplicate {name} option")));
    }
    Ok(())
}
