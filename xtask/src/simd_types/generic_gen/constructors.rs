//! Derive both constructor APIs from the token-taking implementation signatures.
//!
//! A constructor body is written once in the ordinary generators. This pass
//! retains it as a crate-private helper, then emits explicit-token and
//! feature-context entry points. It does not duplicate arithmetic kernels.
use quote::ToTokens;
use regex::Regex;
use syn::{FnArg, GenericParam, ImplItem, Item, Pat};

fn attributes(attrs: &[syn::Attribute]) -> String {
    attrs
        .iter()
        .map(|a| {
            if a.path().is_ident("doc") {
                if let syn::Meta::NameValue(nv) = &a.meta {
                    if let syn::Expr::Lit(lit) = &nv.value {
                        if let syn::Lit::Str(text) = &lit.lit {
                            return text
                                .value()
                                .lines()
                                .map(|line| format!("///{line}\n"))
                                .collect::<String>();
                        }
                    }
                }
            }
            a.to_token_stream().to_string() + "\n"
        })
        .collect()
}

fn word(text: &str, from: &str, to: &str) -> String {
    Regex::new(&format!(r"\b{from}\b"))
        .unwrap()
        .replace_all(text, to)
        .into_owned()
}

pub(super) fn names(source: &str) -> Vec<String> {
    let file = syn::parse_file(source).expect("generated core must parse");
    file.items
        .into_iter()
        .filter_map(|i| if let Item::Impl(i) = i { Some(i) } else { None })
        .flat_map(|i| i.items)
        .filter_map(|i| {
            let ImplItem::Fn(f) = i else { return None };
            if !matches!(f.vis, syn::Visibility::Public(_)) {
                return None;
            }
            let Some(FnArg::Typed(arg)) = f.sig.inputs.first() else {
                return None;
            };
            let Pat::Ident(pat) = &*arg.pat else {
                return None;
            };
            matches!(pat.ident.to_string().as_str(), "token" | "_token")
                .then(|| f.sig.ident.to_string())
        })
        .collect()
}

pub(super) fn generate(source: &str, constructor_names: &[String]) -> String {
    let file = syn::parse_file(source).expect("generated core must parse");
    let mut wrappers = std::collections::BTreeMap::<String, String>::new();
    let mut names = Vec::new();
    for item in file.items {
        let Item::Impl(imp) = item else { continue };
        if imp.trait_.is_some() {
            continue;
        }
        for item in &imp.items {
            let ImplItem::Fn(method) = item else { continue };
            if !matches!(method.vis, syn::Visibility::Public(_)) {
                continue;
            }
            let Some(FnArg::Typed(first)) = method.sig.inputs.first() else {
                continue;
            };
            let Pat::Ident(first_name) = &*first.pat else {
                continue;
            };
            if !matches!(first_name.ident.to_string().as_str(), "token" | "_token") {
                continue;
            }
            let name = method.sig.ident.to_string();
            names.push(name.clone());
            let helper = format!("{name}_with_token");
            let args: Vec<_> = method
                .sig
                .inputs
                .iter()
                .map(|arg| {
                    let FnArg::Typed(arg) = arg else {
                        panic!("constructor with receiver")
                    };
                    let Pat::Ident(pat) = &*arg.pat else {
                        panic!("constructor with pattern")
                    };
                    pat.ident.to_string()
                })
                .collect();
            let attrs = attributes(&method.attrs);
            let cfg = attributes(&imp.attrs);
            let mut generics = imp.generics.clone();
            generics.params = generics
                .params
                .into_iter()
                .filter(|p| !matches!(p, GenericParam::Type(t) if t.ident == "M"))
                .collect();
            let explicit = "crate::simd::generic::Explicit";
            let self_ty = word(&imp.self_ty.to_token_stream().to_string(), "M", explicit);
            let sig = word(&method.sig.to_token_stream().to_string(), "M", explicit);
            let generic_text = generics.to_token_stream().to_string();
            wrappers
                .entry(format!("{cfg}impl {generic_text} {self_ty}"))
                .or_default()
                .push_str(&format!(
                    "{attrs}pub {sig} {{ Self::{helper}({}) }}\n",
                    args.join(",")
                ));

            let generic_token = generics
                .params
                .iter()
                .any(|p| matches!(p, GenericParam::Type(t) if t.ident == "T"));
            let original_type = imp.self_ty.to_token_stream().to_string();
            let shape = original_type.split('<').next().unwrap().trim();
            let width512 = matches!(
                shape,
                "f32x16"
                    | "f64x8"
                    | "i8x64"
                    | "u8x64"
                    | "i16x32"
                    | "u16x32"
                    | "i32x16"
                    | "u32x16"
                    | "i64x8"
                    | "u64x8"
            );
            // These are the backend families implemented by magetypes, not all
            // archmage tokens. V4's narrow delegation intentionally covers f32
            // arithmetic only; its narrower integer conversion traits are absent.
            let tiers = [
                ("ScalarToken", "", ""),
                ("X64V3Token", "v3", "target_arch = \"x86_64\""),
                ("NeonToken", "neon", "target_arch = \"aarch64\""),
                ("Wasm128Token", "wasm128", "target_arch = \"wasm32\""),
                (
                    "X64V4Token",
                    "v4",
                    "all(target_arch = \"x86_64\", feature = \"avx512\")",
                ),
                (
                    "X64V4xToken",
                    "v4x",
                    "all(target_arch = \"x86_64\", feature = \"avx512\")",
                ),
                (
                    "Avx512Fp16Token",
                    "fp16",
                    "all(target_arch = \"x86_64\", feature = \"avx512\")",
                ),
            ];
            for (token, tier, guard) in tiers {
                if generic_token {
                    if matches!(tier, "v4" | "v4x" | "fp16") {
                        if !(width512 && tier != "fp16" || matches!(shape, "f32x4" | "f32x8")) {
                            continue;
                        }
                        if !width512 && generic_text.contains("Convert") {
                            continue;
                        }
                    }
                } else if !original_type.contains(token) {
                    continue;
                }
                let token_path = format!("archmage::{token}");
                let specialize = |s: &str| {
                    word(
                        &word(s, "M", "crate::simd::generic::Context"),
                        "T",
                        &token_path,
                    )
                };
                let mut sig = method.sig.clone();
                sig.inputs = sig.inputs.into_iter().skip(1).collect();
                let mut sig = specialize(&sig.to_token_stream().to_string());
                if sig.contains(&format!("{token_path} :: Repr")) {
                    let backend = format!("{}Backend", shape[..1].to_uppercase() + &shape[1..]);
                    sig = sig.replace(
                        &format!("{token_path} :: Repr"),
                        &format!("<{token_path} as crate::simd::backends::{backend}>::Repr"),
                    );
                }
                let self_ty = specialize(&original_type);
                let guard = if guard.is_empty() {
                    String::new()
                } else {
                    format!("#[cfg({guard})]\n")
                };
                let feature = if tier.is_empty() {
                    "#[inline(always)]".to_owned()
                } else {
                    format!("#[archmage::rite({tier})]")
                };
                let docs: Vec<_> = method
                    .attrs
                    .iter()
                    .filter(|a| a.path().is_ident("doc"))
                    .cloned()
                    .collect();
                let docs = attributes(&docs)
                    .replace("(token-gated)", "(requires matching target features)");
                let proof = if tier.is_empty() {
                    token_path.clone()
                } else {
                    format!("{token_path}::from_context()")
                };
                wrappers.entry(format!("{cfg}{guard}impl {self_ty}")).or_default().push_str(&format!("{docs}#[forbid(unsafe_code)]\n{feature}\npub {sig} {{ Self::{helper}({proof}{comma}{args}) }}\n", comma=if args.len()>1 {","} else {""},args=args[1..].join(",")));
            }
        }
    }
    let mut core = source.to_owned();
    names.sort();
    names.dedup();
    for name in names {
        core = core.replace(
            &format!("pub fn {name}("),
            &format!("pub(crate) fn {name}_with_token("),
        );
    }
    // Shared methods use the helper, whose token comes from an existing vector.
    // Backend calls (T::...) are deliberately outside this pattern.
    let calls = Regex::new(&format!(
        r"((?:Self|[fiu][0-9]+x[0-9]+)(?:::[<][^>]+[>])?::)({})\(",
        constructor_names.join("|")
    ))
    .unwrap();
    core = calls
        .replace_all(&core, "${1}${2}_with_token(")
        .into_owned();
    for (header, methods) in wrappers {
        core.push_str(&format!("\n{header} {{\n{methods}\n}}\n"));
    }
    core
}
