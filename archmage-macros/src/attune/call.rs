//! Calls use the same tier/form descriptors as definitions. Runtime probing is
//! confined to ordinary callers and explicit reattune invocations.
use proc_macro2::{Delimiter, Group, TokenStream, TokenTree};
use quote::{ToTokens, format_ident, quote};
use syn::{
    Ident, Token,
    parse::{Parse, ParseStream},
};

use super::parent::{Access, Parent};
use super::syntax::{self, Form, Rename, Selection};
use crate::tiers::{TierDescriptor, find_tier};

pub(crate) struct Call {
    path: syn::Path,
    args: Vec<syn::Expr>,
    selections: Vec<Selection>,
    names: Vec<Rename>,
    proof: Option<syn::Expr>,
}

impl Parse for Call {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let path = input.parse()?;
        let inner;
        syn::parenthesized!(inner in input);
        let args = inner
            .parse_terminated(syn::Expr::parse, Token![,])?
            .into_iter()
            .collect();
        let mut selections = None;
        let mut names = Vec::new();
        let mut proof = None;
        while !input.is_empty() {
            input.parse::<Token![,]>()?;
            if input.is_empty() {
                break;
            }
            if input.peek(syn::token::Bracket) {
                if selections.is_some() {
                    return Err(input.error("duplicate callee list"));
                }
                let inner;
                syn::bracketed!(inner in input);
                let mut entries = Vec::new();
                while !inner.is_empty() {
                    let name: Ident = inner.parse()?;
                    let (tier, form) = syntax::selector(&name.to_string(), name.span())?;
                    let gate = syntax::gate(&inner)?;
                    entries.push(Selection {
                        tier,
                        form,
                        gate,
                        visibility: None,
                        inline: None,
                    });
                    if !inner.is_empty() {
                        inner.parse::<Token![,]>()?;
                    }
                }
                selections = Some(entries);
            } else {
                let name: Ident = input.parse()?;
                match name.to_string().as_str() {
                    "names" => names = syntax::parse_names(input)?,
                    "using" => {
                        if proof.is_some() {
                            return Err(input.error("duplicate using(...)"));
                        }
                        let inner;
                        syn::parenthesized!(inner in input);
                        proof = Some(inner.parse()?);
                        if !inner.is_empty() {
                            return Err(inner.error("expected one proof expression"));
                        }
                    }
                    _ => {
                        return Err(syn::Error::new(
                            name.span(),
                            "expected a callee list, names(...), or using(...)",
                        ));
                    }
                }
            }
        }
        Ok(Self {
            path,
            args,
            names,
            proof,
            selections: selections.unwrap_or_else(|| {
                syntax::DEFAULTS
                    .iter()
                    .map(|name| Selection {
                        tier: find_tier(name).unwrap(),
                        form: Form::Direct,
                        gate: None,
                        visibility: None,
                        inline: None,
                    })
                    .collect()
            }),
        })
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Context<'a> {
    pub features: &'a [&'static str],
    pub target_arch: Option<&'static str>,
}

impl Context<'_> {
    fn covers(self, callee: &TierDescriptor) -> bool {
        if callee.name == "scalar" {
            return true;
        }
        if self.target_arch != callee.target_arch {
            return false;
        }
        crate::generated::tier_to_canonical_token(callee.name)
            .and_then(crate::generated::token_to_features)
            .is_some_and(|features| {
                features
                    .iter()
                    .all(|feature| self.features.contains(feature))
            })
    }
}

impl Call {
    fn path(&self, tier: &TierDescriptor, form: Form) -> syn::Path {
        if let Some(rename) = self
            .names
            .iter()
            .find(|r| r.tier.name == tier.name && r.form == form)
        {
            let mut path = rename.path.clone();
            // A dictionary renames the item, not its generic arguments.
            if let (Some(source), Some(target)) =
                (self.path.segments.last(), path.segments.last_mut())
            {
                target.arguments = source.arguments.clone();
            }
            path
        } else {
            let suffix = if form == Form::Proof {
                format!("{}_t", tier.suffix)
            } else {
                tier.suffix.to_string()
            };
            crate::common::suffix_path(&self.path, &suffix)
        }
    }

    pub(crate) fn expand(&self, caller: Option<Context<'_>>, reselect: bool) -> TokenStream {
        self.expand_scoped(caller, reselect, None, &mut 0)
    }

    fn expand_scoped(
        &self,
        caller: Option<Context<'_>>,
        reselect: bool,
        parent: Option<&Parent<'_>>,
        serial: &mut usize,
    ) -> TokenStream {
        let label = syn::Lifetime::new(
            &format!("'__attune_call_{}", *serial),
            proc_macro2::Span::mixed_site(),
        );
        *serial += 1;
        let proof_ident = syn::Ident::new("__attune_proof", proc_macro2::Span::mixed_site());
        let supplied_ident = syn::Ident::new("__attune_supplied", proc_macro2::Span::mixed_site());
        let args: Vec<_> = self
            .args
            .iter()
            .map(|arg| match caller {
                Some(context) => rewrite_scoped(arg.to_token_stream(), context, parent, serial),
                None => arg.to_token_stream(),
            })
            .collect();
        let has_marker = self
            .args
            .iter()
            .any(|arg| crate::common::is_bare_ident(arg, "Token"));
        let mut candidates: Vec<_> = self.selections.iter().collect();
        candidates.sort_by_key(|s| std::cmp::Reverse(s.tier.priority));
        let mut branches = Vec::new();
        let mut guarantees = Vec::<TokenStream>::new();
        let mut unconditional = false;
        for selection in candidates {
            let tier = selection.tier;
            if caller.is_some_and(|context| {
                context.target_arch.is_some()
                    && tier.target_arch.is_some()
                    && context.target_arch != tier.target_arch
            }) {
                continue;
            }
            let covered = tier.name == "scalar" || caller.is_some_and(|c| c.covers(tier));
            let inherited = if caller.is_some() && !reselect && self.proof.is_none() && !covered {
                match parent.map(|parent| parent.access(tier)).transpose() {
                    Ok(Some(Some(access))) => Some(access),
                    Ok(_) => continue,
                    Err(diagnostic) => return diagnostic,
                }
            } else {
                None
            };
            let token: syn::Path =
                syn::parse_str(&format!("::{}", tier.token_path)).expect("registered token path");
            let runtime = !covered || self.proof.is_some();
            let form = if caller.is_none() || runtime {
                Form::Proof
            } else {
                selection.form
            };
            let path = self.path(tier, form);
            let proof = if tier.name == "scalar" {
                quote!(::archmage::ScalarToken)
            } else if let Some(Access::Guaranteed(value)) = &inherited {
                value.clone()
            } else if runtime {
                quote!(#proof_ident)
            } else {
                quote!(#token::from_context())
            };
            let call_args = if has_marker {
                let args = self.args.iter().zip(&args).map(|(original, tokens)| {
                    if crate::common::is_bare_ident(original, "Token") {
                        proof.clone()
                    } else {
                        tokens.clone()
                    }
                });
                quote!(#(#args),*)
            } else if form == Form::Proof {
                quote!(#proof, #(#args),*)
            } else {
                quote!(#(#args),*)
            };
            let invocation = quote!(#path(#call_args));
            let guaranteed = tier.name == "scalar"
                || !runtime
                || matches!(inherited, Some(Access::Guaranteed(_)));
            let branch = if guaranteed {
                quote!(break #label #invocation;)
            } else if let Some(Access::Conditional(value)) = &inherited {
                let method = format_ident!("{}", tier.as_method);
                quote!(if let Some(#proof_ident) = ::archmage::IntoConcreteToken::#method(#value) { break #label #invocation; })
            } else if self.proof.is_some() {
                let method = format_ident!("{}", tier.as_method);
                quote!(if let Some(#proof_ident) = ::archmage::IntoConcreteToken::#method(#supplied_ident) { break #label #invocation; })
            } else {
                quote!(if let Some(#proof_ident) = <#token as ::archmage::SimdToken>::summon() { break #label #invocation; })
            };
            let arch = if caller.is_some_and(|c| c.target_arch == tier.target_arch) {
                None
            } else {
                tier.target_arch
            };
            let condition = match (arch, selection.gate.as_deref()) {
                (Some(a), Some(f)) => Some(quote!(all(target_arch = #a, feature = #f))),
                (Some(a), None) => Some(quote!(target_arch = #a)),
                (None, Some(f)) => Some(quote!(feature = #f)),
                (None, None) => None,
            };
            let guard = match (&condition, guarantees.is_empty()) {
                (None, true) => quote!(),
                (Some(condition), true) => quote!(#[cfg(#condition)]),
                (None, false) => quote!(#[cfg(not(any(#(#guarantees),*)))]),
                (Some(condition), false) => {
                    quote!(#[cfg(all(#condition, not(any(#(#guarantees),*))))])
                }
            };
            branches.push(quote!(#guard { #branch }));
            if guaranteed {
                if let Some(condition) = condition {
                    guarantees.push(condition);
                } else {
                    unconditional = true;
                    break;
                }
            }
        }
        let failure = if unconditional {
            quote!()
        } else {
            quote!(#[cfg(not(any(#(#guarantees),*)))] compile_error!("attuned!/reattune!: no guaranteed fallback; include scalar or a tier covered by the caller's target features");)
        };
        let supplied = self.proof.as_ref().map(|proof| {
            let proof = match caller {
                Some(context) => rewrite_scoped(proof.to_token_stream(), context, parent, serial),
                None => proof.to_token_stream(),
            };
            quote!(let #supplied_ident = #proof;)
        });
        quote!({ #supplied #label: { #(#branches)* #failure } })
    }
}

/// Inspect only macro invocations. Nested items have their own feature context.
/// Qualified invocation paths are consumed along with the macro name.
pub(crate) fn rewrite(
    body: TokenStream,
    caller: &TierDescriptor,
    signature: Option<&syn::Signature>,
) -> TokenStream {
    let features = crate::generated::tier_to_canonical_token(caller.name)
        .and_then(crate::generated::token_to_features)
        .unwrap_or(&[]);
    rewrite_context(
        body,
        Context {
            features,
            target_arch: caller.target_arch,
        },
        signature,
    )
}

pub(crate) fn rewrite_context(
    body: TokenStream,
    caller: Context<'_>,
    signature: Option<&syn::Signature>,
) -> TokenStream {
    if !crate::common::tokens_contain_ident(&body, &["attuned", "reattune"]) {
        return body;
    }
    let parent = Parent::new(signature);
    let body = rewrite_scoped(body, caller, Some(&parent), &mut 0);
    let capture = parent.capture();
    quote!(#capture #body)
}

fn rewrite_scoped(
    body: TokenStream,
    caller: Context<'_>,
    parent: Option<&Parent<'_>>,
    serial: &mut usize,
) -> TokenStream {
    if !crate::common::tokens_contain_ident(&body, &["attuned", "reattune"]) {
        return body;
    }
    let tokens: Vec<_> = body.into_iter().collect();
    let mut out = TokenStream::new();
    let mut i = 0;
    while i < tokens.len() {
        if matches!(&tokens[i], TokenTree::Ident(id) if id == "fn" || id == "mod" || id == "impl" || id == "trait")
        {
            while i < tokens.len() {
                let done = matches!(&tokens[i], TokenTree::Group(g) if g.delimiter() == Delimiter::Brace)
                    || matches!(&tokens[i], TokenTree::Punct(p) if p.as_char() == ';');
                out.extend([tokens[i].clone()]);
                i += 1;
                if done {
                    break;
                }
            }
            continue;
        }
        let mut end = i;
        if matches!(tokens.get(end), Some(TokenTree::Punct(p)) if p.as_char() == ':')
            && matches!(tokens.get(end + 1), Some(TokenTree::Punct(p)) if p.as_char() == ':')
        {
            end += 2;
        }
        while end + 3 < tokens.len()
            && matches!(&tokens[end], TokenTree::Ident(_))
            && matches!(&tokens[end + 1], TokenTree::Punct(p) if p.as_char() == ':')
            && matches!(&tokens[end + 2], TokenTree::Punct(p) if p.as_char() == ':')
        {
            end += 3;
        }
        if let Some(TokenTree::Ident(name)) = tokens.get(end)
            && (name == "attuned" || name == "reattune")
            && matches!(tokens.get(end + 1), Some(TokenTree::Punct(p)) if p.as_char() == '!')
            && let Some(TokenTree::Group(group)) = tokens.get(end + 2)
        {
            let expansion = syn::parse2::<Call>(group.stream())
                .map(|call| call.expand_scoped(Some(caller), name == "reattune", parent, serial))
                .unwrap_or_else(|error| error.to_compile_error());
            out.extend(expansion);
            i = end + 3;
            continue;
        }
        if let TokenTree::Group(group) = &tokens[i] {
            let mut next = Group::new(
                group.delimiter(),
                rewrite_scoped(group.stream(), caller, parent, serial),
            );
            next.set_span(group.span());
            out.extend([TokenTree::Group(next)]);
        } else {
            out.extend([tokens[i].clone()]);
        }
        i += 1;
    }
    out
}
