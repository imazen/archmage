//! Derive both constructor APIs from the token-taking implementation signatures.
//!
//! A constructor body is written once in the ordinary generators. This pass
//! retains it as a crate-private helper, then emits explicit-token and
//! feature-context entry points. Context entry points flatten simple value
//! construction; multi-step bodies and memory views keep the shared helper.
//! Arithmetic kernels are not duplicated.
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

pub(super) fn generate(
    source: &str,
    constructor_names: &[String],
    registry: &crate::registry::Registry,
) -> String {
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
                    feature_attributes(registry, token)
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
                let body = match flat_value_body(method, &imp.generics, &token_path) {
                    Some(body) => format!("let {} = {proof}; {}", args[0], specialize(&body)),
                    None => format!(
                        "Self::{helper}({proof}{comma}{args})",
                        comma = if args.len() > 1 { "," } else { "" },
                        args = args[1..].join(",")
                    ),
                };
                wrappers
                    .entry(format!("{cfg}{guard}impl {self_ty}"))
                    .or_default()
                    .push_str(&format!(
                        "{docs}#[forbid(unsafe_code)]\n{feature}\npub {sig} {{ {body} }}\n"
                    ));
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
    // Older raw constructors use the same concrete tier attributes. Lower only
    // complete attribute lines, never documentation examples.
    let rite = Regex::new(r"(?m)^[ \t]*#\[archmage::rite\((\w+)\)\]$").unwrap();
    rite.replace_all(&core, |caps: &regex::Captures<'_>| {
        feature_attributes(registry, &caps[1])
    })
    .into_owned()
}

fn feature_attributes(registry: &crate::registry::Registry, token: &str) -> String {
    let token = registry
        .find_token(token)
        .or_else(|| {
            registry
                .token
                .iter()
                .find(|t| t.short_name.as_deref() == Some(token))
        })
        .expect("generated token is registered");
    assert!(
        !token.features.is_empty(),
        "SIMD entry point needs features"
    );
    format!(
        "/// # Safety\n/// The CPU must support the enabled target features. Safe calls require a\n/// matching or stronger target-feature context, which Rust checks.\n#[target_feature(enable = {:?})]\n#[inline]",
        token.features.join(",")
    )
}

/// Only flatten single-expression value construction, not blocks or multi-step views.
/// Keep `new_repr` as the single definition of the representation/token layout.
/// Explicit constructors and shared operations still use the original helper.
fn flat_value_body(
    method: &syn::ImplItemFn,
    generics: &syn::Generics,
    token: &str,
) -> Option<String> {
    let [syn::Stmt::Expr(expr, None)] = method.block.stmts.as_slice() else {
        return None;
    };
    let syn::Expr::Call(call) = expr else {
        return None;
    };
    let syn::Expr::Path(path) = &*call.func else {
        return None;
    };
    if path.path.to_token_stream().to_string() != "Self :: new_repr" || !simple_value(expr) {
        return None;
    }
    let mut body = expr.to_token_stream().to_string();
    // A concrete token implements many backend traits. Qualify the original
    // generic bound rather than relying on ambiguous concrete method lookup.
    if body.contains("T ::") {
        let param = generics.type_params().find(|p| p.ident == "T")?;
        let [syn::TypeParamBound::Trait(bound)] =
            param.bounds.iter().collect::<Vec<_>>().as_slice()
        else {
            return None;
        };
        body = body.replace(
            "T ::",
            &format!("<{token} as {}>::", bound.path.to_token_stream()),
        );
    }
    Some(body)
}

fn simple_value(expr: &syn::Expr) -> bool {
    match expr {
        syn::Expr::Path(_) | syn::Expr::Lit(_) => true,
        syn::Expr::Call(c) => simple_value(&c.func) && c.args.iter().all(simple_value),
        syn::Expr::MethodCall(c) => simple_value(&c.receiver) && c.args.iter().all(simple_value),
        syn::Expr::Field(f) => simple_value(&f.base),
        syn::Expr::Reference(r) => simple_value(&r.expr),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> crate::registry::Registry {
        crate::registry::Registry::load(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../token-registry.toml"),
        )
        .unwrap()
    }

    #[test]
    fn features_come_from_the_complete_registry_set() {
        let reg = registry();
        for token in &reg.token {
            if token.features.is_empty() {
                continue;
            }
            let attrs = feature_attributes(&reg, &token.name);
            let parse = syn::Attribute::parse_outer;
            let attrs = syn::parse::Parser::parse_str(parse, &attrs).unwrap();
            let feature = attrs
                .iter()
                .find(|a| a.path().is_ident("target_feature"))
                .unwrap();
            feature
                .parse_nested_meta(|meta| {
                    assert!(meta.path.is_ident("enable"));
                    let value: syn::LitStr = meta.value()?.parse()?;
                    assert_eq!(value.value(), token.features.join(","));
                    Ok(())
                })
                .unwrap();
        }
    }

    #[test]
    fn flat_construction_keeps_proof_local_and_qualifies_backend() {
        let source = r#"
            impl<M: ConstructorMode, T: F32x8Backend> f32x8<T, M> {
                pub fn splat(token: T, value: f32) -> Self {
                    Self::new_repr(T::splat(token, value), token)
                }
                pub fn from_slice(token: T, slice: &[f32]) -> Self {
                    let arr = slice[..8].try_into().unwrap();
                    Self::new_repr(T::from_array(token, arr), token)
                }
            }
        "#;
        let generated = generate(source, &names(source), &registry());
        let parsed = syn::parse_file(&generated).unwrap();
        let local = parsed
            .items
            .iter()
            .filter_map(|i| match i {
                Item::Impl(i)
                    if i.self_ty
                        .to_token_stream()
                        .to_string()
                        .contains("X64V3Token") =>
                {
                    Some(i)
                }
                _ => None,
            })
            .next()
            .unwrap();
        let functions: Vec<_> = local
            .items
            .iter()
            .filter_map(|i| match i {
                ImplItem::Fn(f) => Some(f),
                _ => None,
            })
            .collect();
        let splat = functions.iter().find(|f| f.sig.ident == "splat").unwrap();
        assert!(!matches!(splat.sig.safety, syn::Safety::Unsafe(_)));
        assert!(
            splat
                .attrs
                .iter()
                .any(|a| a.path().is_ident("target_feature"))
        );
        assert!(splat.attrs.iter().any(|a| {
            a.to_token_stream()
                .to_string()
                .contains("forbid (unsafe_code)")
        }));
        let body = splat.block.to_token_stream().to_string();
        assert!(
            body.contains("let token = archmage :: X64V3Token :: from_context ()"),
            "{body}"
        );
        assert!(
            body.contains("< archmage :: X64V3Token as F32x8Backend > :: splat"),
            "{body}"
        );
        assert!(!body.contains("splat_with_token"));
        let slice = functions
            .iter()
            .find(|f| f.sig.ident == "from_slice")
            .unwrap();
        assert!(
            slice
                .block
                .to_token_stream()
                .to_string()
                .contains("from_slice_with_token")
        );
        assert!(!generated.contains("#[archmage::rite("));
    }

    #[test]
    fn flattening_rejects_blocks_and_multiple_trait_bounds() {
        let generics = syn::parse_str::<syn::Generics>("<T: Backend>").unwrap();
        let with_block = syn::parse_str::<syn::ImplItemFn>(
            "fn f(token: T) -> Self { Self::new_repr(unsafe { get() }, token) }",
        )
        .unwrap();
        assert!(flat_value_body(&with_block, &generics, "Token").is_none());
        let simple = syn::parse_str::<syn::ImplItemFn>(
            "fn f(token: T) -> Self { Self::new_repr(T::zero(token), token) }",
        )
        .unwrap();
        let ambiguous = syn::parse_str::<syn::Generics>("<T: First + Second>").unwrap();
        assert!(flat_value_body(&simple, &ambiguous, "Token").is_none());
    }
}
