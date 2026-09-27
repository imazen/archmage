//! Body-local Context aliases. Selection does not grant CPU features: the
//! constructors retain their own target_feature requirements, checked by rustc.

use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::{Ident, Token, parse::ParseStream};

use crate::generated::{resolve_vector_name, token_to_vector_width};

pub(crate) fn parse_use(input: ParseStream, names: &mut Vec<String>) -> syn::Result<()> {
    input.parse::<Token![use]>()?;
    let content;
    syn::parenthesized!(content in input);
    while !content.is_empty() {
        let ident: Ident = content.parse()?;
        let name = ident.to_string();
        validate_name(&name, ident.span())?;
        if names.contains(&name) {
            return Err(syn::Error::new(ident.span(), "duplicate type alias in use"));
        }
        names.push(name);
        if !content.is_empty() {
            content.parse::<Token![,]>()?;
        }
    }
    Ok(())
}

pub(crate) fn validate_name(name: &str, span: Span) -> syn::Result<()> {
    if resolve_vector_name(name, 128).is_none() {
        return Err(syn::Error::new(
            span,
            format!(
                "unknown vector `{name}`; use a fixed shape such as f32x8 or an adaptive family such as f32xN"
            ),
        ));
    }
    Ok(())
}

pub(crate) fn aliases(
    names: &[String],
    token_name: Option<&str>,
    span: Span,
) -> syn::Result<TokenStream> {
    if names.is_empty() {
        return Ok(TokenStream::new());
    }
    let token_name = token_name.ok_or_else(|| syn::Error::new(
        span,
        "use(...) requires an explicit tier or a concrete token parameter; generic feature bounds do not select a vector backend",
    ))?;
    let token = format_ident!("{token_name}");
    let mut out = TokenStream::new();
    for name in names {
        validate_name(name, span)?;
        let width = if name.ends_with("xN") {
            token_to_vector_width(token_name).ok_or_else(|| syn::Error::new(
                span,
                format!("adaptive use({name}) has no vector backend for {token_name}; supported tiers: v3, v4, v4x, neon, wasm128, scalar/default"),
            ))?
        } else {
            128 // Fixed shapes ignore this parameter.
        };
        let shape = format_ident!(
            "{}",
            resolve_vector_name(name, width).expect("validated vector")
        );
        let alias = format_ident!("{name}");
        out.extend(quote! {
            #[allow(non_camel_case_types, dead_code)]
            type #alias = ::magetypes::simd::generic::local::#shape<archmage::#token>;
        });
    }
    Ok(out)
}

pub(crate) fn prepend(
    function: &mut crate::common::LightFn,
    names: &[String],
    token_name: Option<&str>,
) -> syn::Result<()> {
    let preamble = aliases(names, token_name, function.sig.ident.span())?;
    if !preamble.is_empty() {
        let body = &function.body;
        function.body = quote!(#preamble #body);
    }
    Ok(())
}

pub(crate) fn tier_token(tier: &crate::tiers::ResolvedTier) -> &str {
    tier.token_path
        .rsplit("::")
        .next()
        .filter(|s| !s.is_empty())
        .unwrap_or("ScalarToken")
}
