//! The entire trusted emitter: registry, proof construction, boundaries, dispatch.
#![forbid(unsafe_code)]
use proc_macro::TokenStream;
use proc_macro2::TokenStream as Tokens;
use quote::{format_ident, quote};
use syn::{
    Error, Expr, ExprCall, FnArg, GenericParam, Ident, ItemFn, Pat, Result, Token,
    parse::{Parse, ParseStream},
    parse_macro_input, parse_quote,
    punctuated::Punctuated,
    spanned::Spanned,
};
mod tiers;
use tiers::{TIERS, Tier};

fn lookup(name: &Ident) -> Result<Tier> {
    TIERS
        .iter()
        .find(|t| name == t.name)
        .copied()
        .ok_or_else(|| Error::new(name.span(), "unknown tier; see the complete registry"))
}
fn attrs(t: Tier) -> Tokens {
    if t.arch.is_empty() {
        return quote!();
    }
    let arch = t.arch;
    let features = t.features().join(",");
    quote!(#[cfg(target_arch = #arch)] #[target_feature(enable = #features)])
}

/// Internal emitter used by the runtime crate; counterfeit local types cannot
/// satisfy the canonical runtime crate's concrete proof parameters.
#[proc_macro]
pub fn tokens(_: TokenStream) -> TokenStream {
    let mut out = Tokens::new();
    for &t in TIERS {
        let name = format_ident!("{}", t.proof);
        let arch = t.arch;
        let features = t.features();
        let direct_attrs = attrs(t);
        let detect = match arch {
            "x86_64" => quote!(std::is_x86_feature_detected!),
            "aarch64" => quote!(std::arch::is_aarch64_feature_detected!),
            _ => quote!(cfg!), // WASM has no runtime detection: module validation is external.
        };
        let runtime = if matches!(arch, "x86_64" | "aarch64") {
            quote! {
                #[cfg(all(feature = "std", target_arch = #arch))]
                { if true #(&& #detect(#features))* { return Some(Self(())); } }
            }
        } else {
            quote!()
        };
        let compiled = if arch.is_empty() {
            quote!(true)
        } else {
            quote!(cfg!(all(target_arch = #arch, #(target_feature = #features),*)))
        };
        out.extend(quote! {
            #[derive(Clone, Copy, Debug)]
            pub struct #name(());
            impl #name {
                /// Detect once at a boundary. no_std uses only build-time guarantees.
                #[inline]
                pub fn summon() -> Option<Self> {
                    if #compiled { return Some(Self(())); }
                    #runtime
                    None
                }
                /// Construct in a context whose features rustc checks.
                #direct_attrs
                #[inline]
                pub fn from_context() -> Self { Self(()) }
            }
        });
        for &weak in TIERS {
            if weak.name != t.name
                && (weak.arch.is_empty() || weak.arch == arch)
                && weak.features().iter().all(|f| features.contains(f))
            {
                let target = format_ident!("{}", weak.proof);
                out.extend(quote! {
                    impl From<#name> for #target {
                        #[inline(always)]
                        fn from(_: #name) -> Self { Self(()) }
                    }
                });
            }
        }
    }
    out.into()
}

impl Parse for Tier {
    fn parse(input: ParseStream) -> Result<Self> {
        lookup(&input.parse()?)
    }
}

fn emit(tier: Tier, mut raw: ItemFn, bridge: bool) -> Result<Tokens> {
    let s = &raw.sig;
    if s.receiver().is_some() {
        return Err(Error::new(s.span(), "only free functions are supported"));
    }
    if !matches!(s.safety, syn::Safety::Default)
        || s.asyncness.is_some()
        || s.constness.is_some()
        || s.abi.is_some()
        || s.variadic.is_some()
    {
        return Err(Error::new(
            s.span(),
            "requires a safe, synchronous Rust function",
        ));
    }
    // Reject arbitrary transforming attributes, particularly extra target_feature
    // or cfg_attr(target_feature): a bridge's proof must cover its entire body.
    for a in &raw.attrs {
        let allowed = [
            "cfg",
            "doc",
            "inline",
            "track_caller",
            "allow",
            "deny",
            "warn",
            "expect",
            "forbid",
        ];
        if !allowed.iter().any(|n| a.path().is_ident(n)) {
            return Err(Error::new(
                a.span(),
                "unsupported attribute on a feature body",
            ));
        }
        if a.path().is_ident("inline")
            && a.meta
                .require_list()
                .is_ok_and(|m| m.tokens.to_string() == "always")
        {
            return Err(Error::new(
                a.span(),
                "feature bodies support inline or inline(never), not inline(always)",
            ));
        }
    }
    if !raw.attrs.iter().any(|a| a.path().is_ident("inline")) {
        raw.attrs.push(parse_quote!(#[inline]));
    }
    let enabled = attrs(tier);
    if !bridge {
        return Ok(quote!(const _: () = (); #enabled #raw));
    }
    let mut wrapper = raw.clone();
    let name = raw.sig.ident.clone();
    wrapper.sig.ident = format_ident!("{}_t", name);
    let proof = format_ident!("{}", tier.proof);
    let mut args = Vec::new();
    for arg in &mut wrapper.sig.inputs {
        if let FnArg::Typed(arg) = arg {
            let Pat::Ident(pat) = &mut *arg.pat else {
                return Err(Error::new(
                    arg.span(),
                    "bridge arguments must be identifiers",
                ));
            };
            if pat.by_ref.is_some()
                || pat.subpat.is_some()
                || pat.ident == name
                || pat.ident == "__proof"
            {
                return Err(Error::new(
                    pat.span(),
                    "unsupported or reserved bridge argument",
                ));
            }
            pat.mutability = None;
            args.push(pat.ident.clone());
        }
    }
    let generics: Vec<_> = raw
        .sig
        .generics
        .params
        .iter()
        .filter_map(|p| match p {
            GenericParam::Type(p) => {
                let ident = &p.ident;
                Some(quote!(#ident))
            }
            GenericParam::Const(p) => {
                let ident = &p.ident;
                Some(quote!(#ident))
            }
            GenericParam::Lifetime(_) => None,
        })
        .collect();
    let params = if generics.is_empty() {
        quote!()
    } else {
        quote!(::<#(#generics),*>)
    };
    wrapper
        .sig
        .inputs
        .insert(0, parse_quote!(__proof: ::artisanal_dispatch::#proof));
    // Only cfg/doc/lint routing is necessary on the wrapper. Expect belongs to
    // the user's body; using allow on the forwarding copy avoids unfulfilled lints.
    wrapper.attrs.retain(|a| !a.path().is_ident("inline"));
    for a in &mut wrapper.attrs {
        if a.path().is_ident("expect")
            && let syn::Meta::List(list) = &mut a.meta
        {
            list.path = parse_quote!(allow);
        }
    }
    wrapper.attrs.push(parse_quote!(#[inline(always)]));
    // THE ONLY UNSAFE EXPRESSION EMITTED: the concrete proof proves every feature
    // in `enabled`; we rejected unsafe inputs and additional feature attributes.
    let call = quote!(#name #params (#(#args),*));
    wrapper.block = syn::parse2(if tier.arch.is_empty() {
        quote!({ #call })
    } else {
        quote!({ unsafe { #call } })
    })?;
    let arch_gate = if tier.arch.is_empty() {
        quote!()
    } else {
        let arch = tier.arch;
        quote!(#[cfg(target_arch = #arch)])
    };
    // An anonymous const is legal in modules/blocks, but rejected in impls and
    // traits. Enforce free-function scope even for receiverless associated fns;
    // otherwise an unqualified call could resolve to an unrelated unsafe fn.
    Ok(quote!(const _: () = (); #enabled #raw #arch_gate #wrapper))
}
#[proc_macro_attribute]
pub fn tier(attr: TokenStream, item: TokenStream) -> TokenStream {
    emit(
        parse_macro_input!(attr as Tier),
        parse_macro_input!(item as ItemFn),
        false,
    )
    .unwrap_or_else(Error::into_compile_error)
    .into()
}
#[proc_macro_attribute]
pub fn bridge(attr: TokenStream, item: TokenStream) -> TokenStream {
    emit(
        parse_macro_input!(attr as Tier),
        parse_macro_input!(item as ItemFn),
        true,
    )
    .unwrap_or_else(Error::into_compile_error)
    .into()
}

struct Call {
    call: ExprCall,
    tiers: Punctuated<Ident, Token![,]>,
}
impl Parse for Call {
    fn parse(input: ParseStream) -> Result<Self> {
        let call = input.parse()?;
        input.parse::<Token![;]>()?;
        Ok(Self {
            call,
            tiers: Punctuated::parse_terminated(input)?,
        })
    }
}
fn select(mut input: Call) -> Result<Tokens> {
    let Expr::Path(path) = &mut *input.call.func else {
        return Err(Error::new(
            input.call.func.span(),
            "dispatch requires a function path",
        ));
    };
    let tiers: Vec<_> = input.tiers.iter().map(lookup).collect::<Result<_>>()?;
    if tiers.last().map(|t| t.name) != Some("scalar")
        || tiers[..tiers.len() - 1].iter().any(|t| t.name == "scalar")
    {
        return Err(Error::new(
            input.tiers.span(),
            "dispatch requires one final scalar fallback",
        ));
    }
    let original = path.path.segments.last().unwrap().ident.clone();
    let args: Vec<_> = input.call.args.iter().collect();
    let bindings: Vec<_> = (0..args.len())
        .map(|i| format_ident!("__arg{i}", span = proc_macro2::Span::mixed_site()))
        .collect();
    let label = syn::Lifetime::new("'__dispatch", proc_macro2::Span::mixed_site());
    let mut attempts = Tokens::new();
    for t in tiers {
        path.path.segments.last_mut().unwrap().ident = format_ident!("{}_{}_t", original, t.name);
        let proof = format_ident!("{}", t.proof);
        if t.arch.is_empty() {
            attempts.extend(quote!(break #label #path(::artisanal_dispatch::Scalar::from_context(), #(#bindings),*);));
        } else {
            let arch = t.arch;
            attempts.extend(quote! {
                #[cfg(target_arch = #arch)]
                if let Some(__proof) = ::artisanal_dispatch::#proof::summon() {
                    break #label #path(__proof, #(#bindings),*);
                }
            });
        }
    }
    // Evaluate arguments once, outside the generated label. Ordinary return/?/
    // break in argument expressions retain the caller's control-flow meaning.
    Ok(quote!(match (#(#args,)*) { (#(#bindings,)*) => #label: { #attempts } }))
}
#[proc_macro]
pub fn dispatch(input: TokenStream) -> TokenStream {
    select(parse_macro_input!(input as Call))
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

#[cfg(test)]
#[path = "../tests/audit.rs"]
mod audit;
