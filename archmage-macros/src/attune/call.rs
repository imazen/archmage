//! Calls use the same tier/form descriptors as definitions. Runtime probing is
//! confined to ordinary callers and explicit reattune invocations.
use proc_macro2::{Delimiter, Group, TokenStream, TokenTree};
use quote::{format_ident, quote};
use syn::{
    Ident, Token,
    parse::{Parse, ParseStream},
};

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

pub(super) fn covers(caller: &TierDescriptor, callee: &TierDescriptor) -> bool {
    if callee.name == "scalar" {
        return true;
    }
    if caller.target_arch != callee.target_arch {
        return false;
    }
    let features = |tier: &TierDescriptor| {
        crate::generated::tier_to_canonical_token(tier.name)
            .and_then(crate::generated::token_to_features)
            .unwrap_or(&[])
    };
    let caller_features = features(caller);
    features(callee).iter().all(|f| caller_features.contains(f))
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

    pub(crate) fn expand(&self, caller: Option<&TierDescriptor>, reselect: bool) -> TokenStream {
        let mut candidates: Vec<_> = self.selections.iter().collect();
        candidates.sort_by_key(|s| std::cmp::Reverse(s.tier.priority));
        let args = &self.args;
        let mut result = quote! { compile_error!("attuned!/reattune!: no guaranteed fallback; include scalar or a tier covered by the caller's target features") };
        for selection in candidates.into_iter().rev() {
            let tier = selection.tier;
            let covered = tier.name == "scalar" || caller.is_some_and(|c| covers(c, tier));
            if caller.is_some() && !reselect && self.proof.is_none() && !covered {
                continue;
            }
            let token_path: syn::Path =
                syn::parse_str(tier.token_path).expect("registered token path");
            let runtime = !covered || self.proof.is_some();
            let form = if caller.is_none() || runtime {
                Form::Proof
            } else {
                selection.form
            };
            let path = self.path(tier, form);
            let proof = if tier.name == "scalar" {
                quote!(::archmage::ScalarToken)
            } else if runtime {
                quote!(__attune_proof)
            } else {
                quote!(#token_path::from_context())
            };
            let invocation = if form == Form::Proof {
                quote!(#path(#proof, #(#args),*))
            } else {
                quote!(#path(#(#args),*))
            };
            let branch = if tier.name == "scalar" || !runtime {
                invocation
            } else if self.proof.is_some() {
                let method = format_ident!("{}", tier.as_method);
                quote! { if let Some(__attune_proof) = ::archmage::IntoConcreteToken::#method(__attune_supplied) { #invocation } else { #result } }
            } else {
                quote! { if let Some(__attune_proof) = <#token_path as ::archmage::SimdToken>::summon() { #invocation } else { #result } }
            };
            let arch = if caller.is_some_and(|c| c.target_arch == tier.target_arch) {
                None
            } else {
                tier.target_arch
            };
            let feature = selection.gate.as_deref();
            let condition = match (arch, feature) {
                (Some(a), Some(f)) => Some(quote!(all(target_arch = #a, feature = #f))),
                (Some(a), None) => Some(quote!(target_arch = #a)),
                (None, Some(f)) => Some(quote!(feature = #f)),
                (None, None) => None,
            };
            result = if let Some(condition) = condition {
                quote! {{ #[cfg(#condition)] { #branch } #[cfg(not(#condition))] { #result } }}
            } else {
                branch
            };
        }
        if let Some(proof) = &self.proof {
            quote! {{ let __attune_supplied = #proof; #result }}
        } else {
            quote! {{ #result }}
        }
    }
}

/// Inspect only macro invocations. Nested items have their own feature context.
/// Qualified invocation paths are consumed along with the macro name.
pub(crate) fn rewrite(body: TokenStream, caller: &TierDescriptor) -> TokenStream {
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
                .map(|call| call.expand(Some(caller), name == "reattune"))
                .unwrap_or_else(|error| error.to_compile_error());
            out.extend(expansion);
            i = end + 3;
            continue;
        }
        if let TokenTree::Group(group) = &tokens[i] {
            let mut next = Group::new(group.delimiter(), rewrite(group.stream(), caller));
            next.set_span(group.span());
            out.extend([TokenTree::Group(next)]);
        } else {
            out.extend([tokens[i].clone()]);
        }
        i += 1;
    }
    out
}
