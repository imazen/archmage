//! Normalize selectors once and validate the complete declaration before emission.
use super::{grammar::*, *};

impl Definition {
    pub(super) fn resolve(self) -> syn::Result<Args> {
        self.validate_options()?;
        let mut args = Args {
            options: self.options,
            family: self.family,
            selections: Vec::new(),
            dispatcher: None,
        };
        for output in &self.outputs {
            output.validate()?;
            if output.action != Action::Select {
                continue;
            }
            match output.target {
                Target::Tier(tier, form) => push(&mut args, output, tier, form, None),
                Target::Wildcard(form) => defaults(&mut args, output, form),
                Target::All => {
                    defaults(&mut args, output, Form::Direct);
                    defaults(&mut args, output, Form::Proof);
                    dispatcher(&mut args, output)?;
                }
                Target::Dispatcher => dispatcher(&mut args, output)?,
            }
        }
        if args.dispatcher.is_some() && args.selections.is_empty() {
            // A dispatcher-only family starts from defaults. Modifiers never
            // create exposed functions unless an output form was requested.
            for name in DEFAULTS {
                args.selections.push(Selection {
                    tier: find_tier(name).unwrap(),
                    form: Form::Hidden,
                    gate: None,
                    visibility: None,
                    inline: None,
                    span: self.span,
                });
            }
        }
        for output in &self.outputs {
            let Target::Tier(tier, form) = output.target else {
                continue;
            };
            match output.action {
                Action::Select => {}
                Action::Remove => {
                    if args.dispatcher.is_some() && tier.name == "scalar" && form == Form::Direct {
                        return Err(syn::Error::new(
                            output.span,
                            "a dispatcher requires its scalar fallback",
                        ));
                    }
                    args.selections.retain(|s| {
                        s.tier.name != tier.name || (form == Form::Proof && s.form != form)
                    });
                }
                Action::Add => {
                    let mut found = false;
                    for wildcard in &self.outputs {
                        match wildcard.target {
                            Target::Wildcard(form) => {
                                push(&mut args, output, tier, form, Some(&wildcard.options));
                                found = true;
                            }
                            Target::All => {
                                push(
                                    &mut args,
                                    output,
                                    tier,
                                    Form::Direct,
                                    Some(&wildcard.options),
                                );
                                push(
                                    &mut args,
                                    output,
                                    tier,
                                    Form::Proof,
                                    Some(&wildcard.options),
                                );
                                found = true;
                            }
                            _ => {}
                        }
                    }
                    if !found {
                        if args.dispatcher.is_none() {
                            return Err(syn::Error::new(
                                output.span,
                                "+tier requires a wildcard output form or a dispatcher",
                            ));
                        }
                        if output.options.visibility.is_some() || output.options.inline.is_some() {
                            return Err(syn::Error::new(
                                output.span,
                                "hidden dispatcher additions accept cfg only; select an explicit output for visibility or inline policy",
                            ));
                        }
                        push(&mut args, output, tier, Form::Hidden, None);
                    }
                }
            }
        }
        validate_outputs(&mut args, self.span)?;
        Ok(args)
    }

    fn validate_options(&self) -> syn::Result<()> {
        let opts = &self.options;
        let error = |message| Err(syn::Error::new(self.span, message));
        if opts.wrap && opts.tier.is_some() {
            return error(
                "wrap derives its features from the proof parameter; omit the explicit tier",
            );
        }
        if !self.family && !opts.names.is_empty() {
            return error("names(...) on a definition requires generated outputs");
        }
        if self.family && opts.imports.cfg_feature.is_some() {
            return error(
                "gate a whole family with an outer #[cfg(...)], or gate a selector with _v3(cfg(feature))",
            );
        }
        if opts.wrap && !opts.defines.is_empty() {
            return error(
                "define(...) requires a concrete tier; use a direct tier body or a family",
            );
        }
        if self.family && (opts.wrap || opts.tier.is_some()) {
            return error(
                "generated outputs cannot be combined with wrap or a single context tier",
            );
        }
        if opts.in_impl && (opts.in_trait || opts.self_type.is_some()) {
            return error("in_impl and in_trait/_self describe different placements");
        }
        if (opts.in_trait || opts.self_type.is_some()) && opts.body_inline == Some(Inline::Default)
        {
            return error(
                "inline(default) cannot infer trait visibility; choose inline(hint), inline(none), or inline(never)",
            );
        }
        Ok(())
    }
}

impl Output {
    fn validate(&self) -> syn::Result<()> {
        let error = |message| Err(syn::Error::new(self.span, message));
        if self.action != Action::Select && !matches!(self.target, Target::Tier(..)) {
            return error("modifiers require a named tier");
        }
        if self.options.gate.is_some() && !matches!(self.target, Target::Tier(..)) {
            return error(
                "cfg belongs on a named tier; use outer #[cfg(...)] to gate the whole family",
            );
        }
        if self.action == Action::Remove
            && (self.options.gate.is_some()
                || self.options.visibility.is_some()
                || self.options.inline.is_some())
        {
            return error("removals do not accept options");
        }
        if self.action == Action::Add && matches!(self.target, Target::Tier(_, Form::Proof)) {
            return error(
                "use +tier to extend selected forms, or _tier_t to request a proof wrapper",
            );
        }
        Ok(())
    }
}

fn push(
    args: &mut Args,
    output: &Output,
    tier: &'static TierDescriptor,
    form: Form,
    inherited: Option<&OutputOptions>,
) {
    args.selections.push(Selection {
        tier,
        form,
        gate: output.options.gate.clone(),
        visibility: output
            .options
            .visibility
            .as_ref()
            .or_else(|| inherited.and_then(|o| o.visibility.as_ref()))
            .cloned(),
        inline: output
            .options
            .inline
            .or_else(|| inherited.and_then(|o| o.inline)),
        span: output.span,
    });
}

fn defaults(args: &mut Args, output: &Output, form: Form) {
    for name in DEFAULTS {
        push(args, output, find_tier(name).unwrap(), form, None);
    }
}

fn dispatcher(args: &mut Args, output: &Output) -> syn::Result<()> {
    if args
        .dispatcher
        .replace((output.options.visibility.clone(), output.options.inline))
        .is_some()
    {
        return Err(syn::Error::new(
            output.span,
            "duplicate dispatcher selector",
        ));
    }
    Ok(())
}

fn validate_outputs(args: &mut Args, span: Span) -> syn::Result<()> {
    if args.family && args.selections.is_empty() && args.dispatcher.is_none() {
        return Err(syn::Error::new(span, "declaration selects no outputs"));
    }
    // Coalesce identical selectors in place. Keep source order and avoid a
    // second allocation just to deduplicate a small tier set.
    let mut i = 0;
    while i < args.selections.len() {
        let selection = &args.selections[i];
        let prior = &args.selections[..i];
        if prior
            .iter()
            .any(|s| s.tier.name == selection.tier.name && s.gate != selection.gate)
        {
            return Err(syn::Error::new(
                selection.span,
                "a tier's direct, proof and dispatcher implementations must have the same feature gate",
            ));
        }
        if let Some(old) = prior
            .iter()
            .find(|s| s.tier.name == selection.tier.name && s.form == selection.form)
        {
            if old.inline != selection.inline
                || !same_visibility(&old.visibility, &selection.visibility)
            {
                return Err(syn::Error::new(
                    selection.span,
                    "conflicting policies for the same output",
                ));
            }
            args.selections.remove(i);
        } else {
            i += 1;
        }
    }
    if args.dispatcher.is_some()
        && args
            .selections
            .iter()
            .any(|s| s.tier.name == "scalar" && s.gate.is_some())
    {
        return Err(syn::Error::new(
            span,
            "a dispatcher requires an ungated scalar fallback",
        ));
    }
    Ok(())
}

fn same_visibility(a: &Option<Visibility>, b: &Option<Visibility>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(Visibility::Public(_)), Some(Visibility::Public(_))) => true,
        (Some(Visibility::Inherited), Some(Visibility::Inherited)) => true,
        (Some(Visibility::Restricted(a)), Some(Visibility::Restricted(b))) => {
            a.in_token.is_some() == b.in_token.is_some()
                && a.path.leading_colon.is_some() == b.path.leading_colon.is_some()
                && a.path.segments.len() == b.path.segments.len()
                && a.path
                    .segments
                    .pairs()
                    .zip(b.path.segments.pairs())
                    .all(|(a, b)| a.value().ident == b.value().ident)
        }
        _ => false,
    }
}
