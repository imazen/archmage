//! Add migration spellings without changing the original token-first API.
//!
//! Inspect the generated signatures rather than maintaining a second method
//! roster. Handwritten scalar and cross-width methods use this same pass.

use quote::{ToTokens, format_ident, quote};
use syn::{FnArg, GenericParam, ImplItem, Item, Pat};

pub(super) fn generate(source: &str) -> String {
    let file = syn::parse_file(source).expect("vector source must parse");
    let mut output = String::new();
    for item in file.items {
        let Item::Impl(mut imp) = item else { continue };
        if imp.trait_.is_some() {
            continue;
        }
        let mut aliases = Vec::new();
        for item in &imp.items {
            let ImplItem::Fn(method) = item else { continue };
            if !matches!(method.vis, syn::Visibility::Public(_)) {
                continue;
            }
            let name = &method.sig.ident;
            // This uniform raw entry point is generated directly, since the
            // existing from_raw method has no token argument. Native NEON
            // names already end in `_t`; use from_raw_t, not an `_t_t` alias.
            if name.to_string().ends_with("_t") {
                continue;
            }
            let Some(FnArg::Typed(first)) = method.sig.inputs.first() else {
                continue;
            };
            let syn::Type::Path(first_type) = &*first.ty else {
                continue;
            };
            let token_type = &first_type.path.segments.last().unwrap().ident;
            if token_type != "T" && !token_type.to_string().ends_with("Token") {
                continue;
            }
            assert!(
                matches!(method.sig.safety, syn::Safety::Default),
                "unsafe constructor: {name}"
            );
            let mut alias = method.clone();
            alias.sig.ident = format_ident!("{name}_t");
            let mut arguments = Vec::new();
            for (index, arg) in alias.sig.inputs.iter_mut().enumerate() {
                let FnArg::Typed(arg) = arg else {
                    panic!("constructor with receiver")
                };
                if matches!(*arg.pat, Pat::Wild(_)) {
                    let ident = format_ident!("argument_{index}");
                    arg.pat = Box::new(syn::parse_quote!(#ident));
                }
                let Pat::Ident(pat) = &*arg.pat else {
                    panic!("constructor pattern: {name}")
                };
                arguments.push(pat.ident.clone());
            }
            let generics: Vec<_> = method
                .sig
                .generics
                .params
                .iter()
                .filter_map(|p| match p {
                    GenericParam::Lifetime(_) => None,
                    GenericParam::Type(p) => Some(p.ident.clone()),
                    GenericParam::Const(p) => Some(p.ident.clone()),
                })
                .collect();
            let call = if generics.is_empty() {
                quote!(Self::#name(#(#arguments),*))
            } else {
                quote!(Self::#name::<#(#generics),*>(#(#arguments),*))
            };
            alias.block = syn::parse_quote!({ #call });
            alias.attrs.retain(|a| !a.path().is_ident("doc"));
            let doc = format!(
                "Explicit-token alias of [`Self::{name}`], with identical arguments and behavior.\n\nThe `_t` spelling is intended for migration to magetypes 0.10.\nThe caller does not need a target-feature annotation."
            );
            alias.attrs.push(syn::parse_quote!(#[doc = #doc]));
            alias.attrs.push(syn::parse_quote!(#[forbid(unsafe_code)]));
            aliases.push(ImplItem::Fn(alias));
        }
        if !aliases.is_empty() {
            imp.items = aliases;
            output.push_str(&imp.to_token_stream().to_string());
            output.push('\n');
        }
    }
    if output.is_empty() {
        output
    } else {
        "// Generated explicit-token migration aliases. Do not edit.\n".to_owned() + &output
    }
}

/// Emit the exact inherent construction surface, grouped by vector type.
/// Both original methods and generated aliases are read from the same source.
pub(super) fn inventory(
    sources: impl Iterator<Item = String>,
) -> std::collections::BTreeMap<String, String> {
    use std::collections::{BTreeMap, BTreeSet};
    let mut groups: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for source in sources {
        let file = syn::parse_file(&source).expect("constructor inventory source");
        for item in file.items {
            let Item::Impl(imp) = item else { continue };
            if imp.trait_.is_some() {
                continue;
            }
            let syn::Type::Path(ty) = &*imp.self_ty else {
                continue;
            };
            let ty_name = ty.path.segments.last().unwrap().ident.to_string();
            let (generics, _, bounds) = imp.generics.split_for_impl();
            let self_ty = &imp.self_ty;
            let gates: Vec<_> = imp
                .attrs
                .iter()
                .filter(|a| a.path().is_ident("cfg"))
                .collect();
            for item in &imp.items {
                let ImplItem::Fn(method) = item else { continue };
                if !matches!(method.vis, syn::Visibility::Public(_)) {
                    continue;
                }
                let token = match method.sig.inputs.first() {
                    Some(FnArg::Typed(arg)) => match &*arg.ty {
                        syn::Type::Path(p) => {
                            let name = p.path.segments.last().unwrap().ident.to_string();
                            name == "T" || name.ends_with("Token")
                        }
                        _ => false,
                    },
                    _ => false,
                };
                if !token && method.sig.ident != "from_raw" {
                    continue;
                }
                let attrs: Vec<_> = method
                    .attrs
                    .iter()
                    .filter(|a| a.path().is_ident("cfg") || a.path().is_ident("target_feature"))
                    .collect();
                let sig = &method.sig;
                let code =
                    quote!(#(#gates)* impl #generics #self_ty #bounds { #(#attrs)* pub #sig; })
                        .to_string()
                        .replace(" :: ", "::")
                        .replace(" < ", "<")
                        .replace(" >", ">")
                        .replace(" ,", ",")
                        .replace(" ;", ";")
                        .replace("{ ", "{\n    ")
                        .replace(" }", "\n}");
                groups.entry(ty_name.clone()).or_default().insert(code);
            }
        }
    }
    groups.into_iter().map(|(name, methods)| {
        let mut doc = format!("# {name} construction signatures\n\nGenerated by `cargo run -p xtask -- generate`; do not edit.\n\nSee [the inventory index](README.md) for naming, module paths, and width gates.\n\n");
        for method in methods { doc.push_str(&format!("```rust\n{method}\n```\n\n")); }
        (name, doc)
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_names_use_uniform_raw_spelling() {
        let source = r#"
            impl Vector<NeonToken> {
                pub fn from_float32x4_t(token: NeonToken, raw: float32x4_t) -> Self { todo!() }
                pub fn from_raw_t(token: NeonToken, raw: float32x4_t) -> Self { todo!() }
            }
        "#;
        assert!(generate(source).is_empty());
    }

    #[test]
    fn preserves_gates_bounds_lifetimes_and_const_arguments() {
        let source = r#"
            #[cfg(feature = "w512")]
            impl<T: Backend> Vector<T> where T::Repr: Copy {
                #[cfg(target_arch = "wasm32")]
                #[inline(always)]
                pub fn partition<'a, const N: usize>(_: T, values: &'a mut [u8])
                    -> &'a mut [[u8; N]] where T: Other { todo!() }
                pub fn value(self) -> i32 { 0 }
                pub(crate) fn internal(token: T) -> Self { todo!() }
            }
        "#;
        let parsed = syn::parse_file(&generate(source)).unwrap();
        let Item::Impl(imp) = &parsed.items[0] else {
            panic!()
        };
        assert_eq!(imp.items.len(), 1);
        assert_eq!(imp.attrs.len(), 1);
        assert!(imp.generics.where_clause.is_some());
        let ImplItem::Fn(method) = &imp.items[0] else {
            panic!()
        };
        assert_eq!(method.sig.ident, "partition_t");
        assert!(method.sig.generics.where_clause.is_some());
        assert!(method.attrs.iter().any(|a| a.path().is_ident("cfg")));
        assert_eq!(
            method.block.to_token_stream().to_string(),
            "{ Self :: partition :: < N > (argument_0 , values) }"
        );
    }
}
