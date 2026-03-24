//! Shared utilities for all proc-macros.

use quote::{ToTokens, quote};
use syn::{Attribute, GenericParam, Signature, Type, parse::ParseStream, token};

/// A function parsed with the body left as an opaque TokenStream.
///
/// Only the signature is fully parsed into an AST — the body tokens are collected
/// without building any AST nodes (no expressions, statements, or patterns parsed).
/// This saves ~2ms per function invocation at 100 lines of code.
#[derive(Clone)]
pub(crate) struct LightFn {
    pub attrs: Vec<Attribute>,
    pub vis: syn::Visibility,
    pub sig: Signature,
    pub brace_token: token::Brace,
    pub body: proc_macro2::TokenStream,
}

impl syn::parse::Parse for LightFn {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let vis: syn::Visibility = input.parse()?;
        let sig: Signature = input.parse()?;
        let content;
        let brace_token = syn::braced!(content in input);
        let body: proc_macro2::TokenStream = content.parse()?;
        Ok(LightFn {
            attrs,
            vis,
            sig,
            brace_token,
            body,
        })
    }
}

impl ToTokens for LightFn {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        for attr in &self.attrs {
            attr.to_tokens(tokens);
        }
        self.vis.to_tokens(tokens);
        self.sig.to_tokens(tokens);
        self.brace_token.surround(tokens, |tokens| {
            self.body.to_tokens(tokens);
        });
    }
}

/// Filter out `#[inline]`, `#[inline(always)]`, `#[inline(never)]` from attributes.
pub(crate) fn filter_inline_attrs(attrs: &[Attribute]) -> Vec<&Attribute> {
    attrs
        .iter()
        .filter(|attr| !attr.path().is_ident("inline"))
        .collect()
}

/// Check if an attribute is a lint-control attribute.
pub(crate) fn is_lint_attr(attr: &Attribute) -> bool {
    let path = attr.path();
    path.is_ident("allow")
        || path.is_ident("expect")
        || path.is_ident("deny")
        || path.is_ident("warn")
        || path.is_ident("forbid")
}

/// Extract lint-control attributes from a list of attributes.
pub(crate) fn filter_lint_attrs(attrs: &[Attribute]) -> Vec<&Attribute> {
    attrs.iter().filter(|attr| is_lint_attr(attr)).collect()
}

/// Generate a cfg guard combining target_arch and an optional feature gate.
pub(crate) fn gen_cfg_guard(
    target_arch: Option<&str>,
    cfg_feature: Option<&str>,
) -> proc_macro2::TokenStream {
    match (target_arch, cfg_feature) {
        (Some(arch), Some(feat)) => {
            quote! { #[cfg(all(target_arch = #arch, feature = #feat))] }
        }
        (Some(arch), None) => quote! { #[cfg(target_arch = #arch)] },
        (None, Some(feat)) => quote! { #[cfg(feature = #feat)] },
        (None, None) => quote! {},
    }
}

/// Build a turbofish token stream from a function's generics.
pub(crate) fn build_turbofish(generics: &syn::Generics) -> proc_macro2::TokenStream {
    let params: Vec<proc_macro2::TokenStream> = generics
        .params
        .iter()
        .filter_map(|param| match param {
            GenericParam::Type(tp) => {
                let ident = &tp.ident;
                Some(quote! { #ident })
            }
            GenericParam::Const(cp) => {
                let ident = &cp.ident;
                Some(quote! { #ident })
            }
            GenericParam::Lifetime(_) => None,
        })
        .collect();
    if params.is_empty() {
        quote! {}
    } else {
        quote! { ::<#(#params),*> }
    }
}

/// Replace all `Self` identifier tokens with a concrete type in a token stream.
pub(crate) fn replace_self_in_tokens(
    tokens: proc_macro2::TokenStream,
    replacement: &Type,
) -> proc_macro2::TokenStream {
    let mut result = proc_macro2::TokenStream::new();
    for tt in tokens {
        match tt {
            proc_macro2::TokenTree::Ident(ref ident) if ident == "Self" => {
                result.extend(replacement.to_token_stream());
            }
            proc_macro2::TokenTree::Group(group) => {
                let new_stream = replace_self_in_tokens(group.stream(), replacement);
                let mut new_group = proc_macro2::Group::new(group.delimiter(), new_stream);
                new_group.set_span(group.span());
                result.extend(std::iter::once(proc_macro2::TokenTree::Group(new_group)));
            }
            other => {
                result.extend(std::iter::once(other));
            }
        }
    }
    result
}

/// Generate import statements for intrinsics and/or magetypes.
pub(crate) fn generate_imports(
    target_arch: Option<&str>,
    magetypes_namespace: Option<&str>,
    import_intrinsics: bool,
    import_magetypes: bool,
) -> proc_macro2::TokenStream {
    let mut imports = proc_macro2::TokenStream::new();

    if import_intrinsics && let Some(arch) = target_arch {
        let arch_ident = quote::format_ident!("{}", arch);
        imports.extend(quote! {
            #[allow(unused_imports)]
            use archmage::intrinsics::#arch_ident::*;
        });
    }

    if import_magetypes && let Some(ns) = magetypes_namespace {
        let ns_ident = quote::format_ident!("{}", ns);
        imports.extend(quote! {
            #[allow(unused_imports)]
            use magetypes::simd::#ns_ident::*;
            #[allow(unused_imports)]
            use magetypes::simd::backends::*;
        });
    }

    imports
}

/// Suffix the last segment of a path: `process` → `process_v3`.
pub(crate) fn suffix_path(path: &syn::Path, suffix: &str) -> syn::Path {
    let mut suffixed = path.clone();
    if let Some(last) = suffixed.segments.last_mut() {
        last.ident = quote::format_ident!("{}_{}", last.ident, suffix);
    }
    suffixed
}
