//! Check context-created tokens against function attributes, not runtime proofs.
use std::collections::HashSet;
use syn::{
    Token,
    buffer::Cursor,
    ext::IdentExt as _,
    visit::{self, Visit},
};

use crate::registry::Registry;

pub(super) fn verify(reg: &Registry, rel: &str, text: &str, errors: &mut Vec<String>) {
    let in_scope = rel.starts_with("magetypes/src")
        || rel.starts_with("magemath/src")
        || ((rel == "src" || rel.starts_with("src/")) && !rel.starts_with("src/tokens/generated/"));
    if !in_scope || !text.contains("from_context") {
        return;
    }
    let file = match syn::parse_file(text) {
        Ok(file) => file,
        Err(error) => {
            errors.push(format!("{rel}: cannot audit from_context: {error}"));
            return;
        }
    };
    let mut visitor = ContextVisitor {
        reg,
        rel,
        errors,
        frame: Frame::default(),
        has_relative_from_context: false,
        has_glob_import: false,
    };
    visitor.visit_file(&file);
    if visitor.has_relative_from_context && visitor.has_glob_import {
        visitor.errors.push(format!(
            "{rel}: glob import `use ...::*` makes relative `archmage::...::from_context` resolution ambiguous; use `::archmage::...::from_context`",
        ));
    }
}

#[derive(Default)]
struct Frame {
    features: HashSet<String>,
    name: String,
    unsafe_code: bool,
    constructions: usize,
}

struct ContextVisitor<'a> {
    reg: &'a Registry,
    rel: &'a str,
    errors: &'a mut Vec<String>,
    frame: Frame,
    has_relative_from_context: bool,
    has_glob_import: bool,
}

impl ContextVisitor<'_> {
    fn function(&mut self, attrs: &[syn::Attribute], sig: &syn::Signature, body: &syn::Block) {
        let outer = std::mem::take(&mut self.frame);
        self.frame.name = sig.ident.to_string();
        self.frame.unsafe_code = matches!(sig.safety, syn::Safety::Unsafe(_));
        for attr in attrs {
            let segs: Vec<String> = attr
                .path()
                .segments
                .iter()
                .map(|s| s.ident.unraw().to_string())
                .collect();
            if attr.path().leading_colon.is_none() && segs.as_slice() == ["target_feature"] {
                let result = attr.parse_nested_meta(|meta| {
                    if meta.path.leading_colon.is_none()
                        && meta.path.segments.len() == 1
                        && meta.path.segments[0].ident.unraw() == "enable"
                    {
                        let value: syn::LitStr = meta.value()?.parse()?;
                        self.frame
                            .features
                            .extend(value.value().split(',').map(|s| s.trim().to_owned()));
                        Ok(())
                    } else {
                        Err(meta.error("unsupported target_feature argument"))
                    }
                });
                if result.is_err() {
                    self.frame.features.clear();
                }
            } else {
                let macro_kind = match segs.as_slice() {
                    [name] if attr.path().leading_colon.is_none() => Some(name.as_str()),
                    [pkg, name] if pkg == "archmage" => Some(name.as_str()),
                    _ => None,
                };
                if matches!(
                    macro_kind,
                    Some("rite" | "token_target_features" | "attune")
                ) && let syn::Meta::List(list) = &attr.meta
                    && let Some(proved) = parse_macro_tier_features(self.reg, list)
                {
                    self.frame.features.extend(proved);
                }
            }
        }
        self.visit_block(body);
        if self.frame.constructions > 0 && self.frame.unsafe_code {
            self.errors.push(format!(
                "{}: from_context in `{}` must not use unsafe code",
                self.rel, self.frame.name
            ));
        }
        self.frame = outer;
    }
}

impl<'ast> Visit<'ast> for ContextVisitor<'_> {
    fn visit_item_fn(&mut self, item: &'ast syn::ItemFn) {
        // Nested functions do not inherit their parent's target features.
        self.function(&item.attrs, &item.sig, &item.block);
    }

    fn visit_impl_item_fn(&mut self, item: &'ast syn::ImplItemFn) {
        self.function(&item.attrs, &item.sig, &item.block);
    }

    fn visit_trait_item_fn(&mut self, item: &'ast syn::TraitItemFn) {
        if let Some(body) = &item.default {
            self.function(&item.attrs, &item.sig, body);
        }
    }

    fn visit_expr_unsafe(&mut self, item: &'ast syn::ExprUnsafe) {
        self.frame.unsafe_code = true;
        visit::visit_expr_unsafe(self, item);
    }

    fn visit_expr_path(&mut self, item: &'ast syn::ExprPath) {
        if item
            .path
            .segments
            .last()
            .is_some_and(|s| s.ident.unraw() == "from_context")
        {
            self.frame.constructions += 1;
            let parts: Vec<_> = item
                .path
                .segments
                .iter()
                .map(|s| s.ident.unraw().to_string())
                .collect();
            // Syntactic path audit paired with `rustc`: accept `::archmage::Token::from_context`
            // (unambiguous extern-prelude path) or relative `archmage::Token::from_context` when
            // no local `mod archmage`, `extern crate ... as archmage`, `use ...::archmage`,
            // `use ...::archmage::{self, ...}`, `as archmage`, or `use ...::*` glob is present.
            if item.path.leading_colon.is_none() {
                self.has_relative_from_context = true;
            }
            let token = if parts.len() == 3 && parts[0] == "archmage" && item.qself.is_none() {
                self.reg.find_token(&parts[1])
            } else {
                None
            };
            let valid = token.is_some_and(|t| {
                !self.frame.name.is_empty()
                    && t.features.iter().all(|f| self.frame.features.contains(f))
            });
            if !valid {
                self.errors.push(format!("{}: from_context in `{}` requires an explicit archmage token and matching or superset function target features", self.rel, self.frame.name));
            }
        }
        visit::visit_expr_path(self, item);
    }

    fn visit_item_mod(&mut self, item: &'ast syn::ItemMod) {
        if item.ident.unraw() == "archmage" {
            self.errors.push(format!(
                "{}: local `mod archmage` shadows the explicit crate path required for from_context",
                self.rel
            ));
        }
        visit::visit_item_mod(self, item);
    }

    fn visit_item_extern_crate(&mut self, item: &'ast syn::ItemExternCrate) {
        let binds_archmage = match &item.rename {
            Some((_, rename)) => rename.unraw() == "archmage",
            None => false,
        };
        if binds_archmage {
            self.errors.push(format!(
                "{}: `extern crate ... as archmage` shadows the external crate path required for from_context",
                self.rel
            ));
        }
        visit::visit_item_extern_crate(self, item);
    }

    fn visit_use_path(&mut self, item: &'ast syn::UsePath) {
        if item.ident.unraw() == "archmage" && use_tree_binds_self(&item.tree) {
            self.errors.push(format!(
                "{}: `self` import of `archmage` shadows the external crate path required for from_context",
                self.rel
            ));
        }
        visit::visit_use_path(self, item);
    }

    fn visit_use_glob(&mut self, item: &'ast syn::UseGlob) {
        self.has_glob_import = true;
        visit::visit_use_glob(self, item);
    }

    fn visit_use_name(&mut self, item: &'ast syn::UseName) {
        if item.ident.unraw() == "from_context" {
            self.errors.push(format!(
                "{}: import of from_context hides the explicit token path",
                self.rel
            ));
        }
        if item.ident.unraw() == "archmage" {
            self.errors.push(format!(
                "{}: import of `archmage` shadows the external crate path required for from_context",
                self.rel
            ));
        }
    }

    fn visit_use_rename(&mut self, item: &'ast syn::UseRename) {
        if item.ident.unraw() == "from_context" || item.rename.unraw() == "from_context" {
            self.errors.push(format!(
                "{}: alias of from_context hides the explicit token path",
                self.rel
            ));
        }
        if item.rename.unraw() == "archmage" {
            self.errors.push(format!(
                "{}: alias `as archmage` shadows the explicit crate path required for from_context",
                self.rel
            ));
        }
    }

    fn visit_expr_const(&mut self, item: &'ast syn::ExprConst) {
        let outer = std::mem::take(&mut self.frame);
        visit::visit_expr_const(self, item);
        self.frame = outer;
    }

    fn visit_item_const(&mut self, item: &'ast syn::ItemConst) {
        let outer = std::mem::take(&mut self.frame);
        visit::visit_item_const(self, item);
        self.frame = outer;
    }

    fn visit_item_static(&mut self, item: &'ast syn::ItemStatic) {
        let outer = std::mem::take(&mut self.frame);
        visit::visit_item_static(self, item);
        self.frame = outer;
    }

    fn visit_macro(&mut self, item: &'ast syn::Macro) {
        // Unexpanded macros must not hide a construction or an unsafe block.
        let mut saw_unsafe = false;
        let mut saw_from_context = false;
        let _ = item.parse_body_with(|input: syn::parse::ParseStream<'_>| {
            saw_unsafe = cursor_has_ident(input.cursor(), "unsafe");
            saw_from_context = cursor_has_ident(input.cursor(), "from_context");
            input.step(|step| {
                let mut c: Cursor<'_> = *step;
                while let Some((_, next)) = c.token_tree() {
                    c = next;
                }
                Ok(((), c))
            })
        });
        if saw_unsafe {
            self.frame.unsafe_code = true;
        }
        if saw_from_context {
            self.errors.push(format!(
                "{}: from_context inside an unexpanded macro cannot be audited",
                self.rel
            ));
        }
    }
}

fn cursor_has_ident(mut cursor: Cursor<'_>, target: &str) -> bool {
    while !cursor.eof() {
        if let Some((ident, next)) = cursor.ident() {
            if ident.unraw() == target {
                return true;
            }
            cursor = next;
            continue;
        }
        if let Some((inner, _, _, next)) = cursor.any_group() {
            if cursor_has_ident(inner, target) {
                return true;
            }
            cursor = next;
            continue;
        }
        if let Some((_, next)) = cursor.token_tree() {
            cursor = next;
        } else {
            break;
        }
    }
    false
}

fn use_tree_binds_self(tree: &syn::UseTree) -> bool {
    match tree {
        syn::UseTree::Name(n) => n.ident == "self",
        syn::UseTree::Rename(r) => r.ident == "self",
        syn::UseTree::Group(g) => g.items.iter().any(use_tree_binds_self),
        _ => false,
    }
}

/// Parse `#[rite(...)]` / `#[attune(...)]` arguments and return the intersection
/// of features guaranteed across every generated variant. Unknown directives
/// (such as `dispatch`, `wrap`, or wildcard `_*_t`) fail closed (`None`).
fn parse_macro_tier_features(reg: &Registry, list: &syn::MetaList) -> Option<HashSet<String>> {
    let resolve = |raw: &str| -> Option<HashSet<String>> {
        let stripped = raw.strip_prefix('_').unwrap_or(raw);
        if stripped == "scalar" || stripped == "default" {
            return Some(HashSet::new());
        }
        let token = reg.token.iter().find(|t| {
            t.short_name.as_deref() == Some(stripped)
                || t.extraction_aliases.iter().any(|a| a == stripped)
        })?;
        Some(token.features.iter().cloned().collect())
    };

    list.parse_args_with(
        |input: syn::parse::ParseStream<'_>| -> syn::Result<Option<HashSet<String>>> {
            let mut additions: Vec<(String, HashSet<String>)> = Vec::new();
            let mut removals: HashSet<String> = HashSet::new();

            while !input.is_empty() {
                if input.peek(Token![+]) || input.peek(Token![-]) {
                    let is_remove = input.peek(Token![-]);
                    if is_remove {
                        let _: Token![-] = input.parse()?;
                    } else {
                        let _: Token![+] = input.parse()?;
                    }
                    let ident: syn::Ident = input.parse()?;
                    let name = ident.unraw().to_string();
                    let key = name.strip_prefix('_').unwrap_or(&name).to_owned();
                    let Some(feats) = resolve(&name) else {
                        return Ok(None);
                    };
                    if is_remove {
                        removals.insert(key);
                    } else {
                        additions.push((key, feats));
                    }
                } else {
                    let ident: syn::Ident = input.parse()?;
                    let name = ident.unraw().to_string();
                    match name.as_str() {
                        "import_intrinsics" | "import_magetypes" | "suppress_const_test" => {}
                        "cfg" | "inline" | "define" => {
                            let content;
                            syn::parenthesized!(content in input);
                            content.step(|step| {
                                let mut c: Cursor<'_> = *step;
                                while let Some((_, next)) = c.token_tree() {
                                    c = next;
                                }
                                Ok(((), c))
                            })?;
                        }
                        _ => {
                            if name.ends_with("_t") {
                                return Ok(None);
                            }
                            let key = name.strip_prefix('_').unwrap_or(&name).to_owned();
                            let Some(feats) = resolve(&name) else {
                                return Ok(None);
                            };
                            additions.push((key, feats));
                        }
                    }
                }
                if input.peek(Token![,]) {
                    let _: Token![,] = input.parse()?;
                } else if !input.is_empty() {
                    return Ok(None);
                }
            }

            let kept: Vec<HashSet<String>> = additions
                .into_iter()
                .filter(|(k, _)| !removals.contains(k))
                .map(|(_, f)| f)
                .collect();
            let Some((first, rest)) = kept.split_first() else {
                return Ok(None);
            };
            let mut intersection = first.clone();
            for set in rest {
                intersection.retain(|f| set.contains(f));
            }
            Ok(Some(intersection))
        },
    )
    .ok()
    .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_at(rel: &str, source: &str) -> Vec<String> {
        let reg = Registry::load(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("token-registry.toml"),
        )
        .unwrap();
        let mut errors = Vec::new();
        verify(&reg, rel, source, &mut errors);
        errors
    }

    fn check(source: &str) -> Vec<String> {
        check_at("magetypes/src/test.rs", source)
    }

    #[test]
    fn matching_and_superset_contexts() {
        for attribute in [
            "archmage::rite(v3)",
            "archmage::rite(v4)",
            "archmage::rite(v3, import_intrinsics)",
            "archmage::rite(v3, v4)",
            "archmage::token_target_features(v3)",
            "archmage::attune(v3)",
            "archmage::attune(v4)",
            "archmage::attune(v3, inline(hint))",
            "::archmage::attune(v3, inline(never))",
            "attune(v3, import_magetypes)",
        ] {
            let source =
                format!("#[{attribute}] fn wrap() {{ archmage::X64V3Token::from_context(); }}");
            assert!(check(&source).is_empty(), "{source}: {:?}", check(&source));
        }
        assert!(check("#[target_feature(enable = \"neon\")] fn wrap() { ::archmage::NeonToken::from_context(); }").is_empty());
    }

    #[test]
    fn audits_src_and_magemath_while_exempting_generated_tokens() {
        let bad = "fn wrap() { archmage::X64V3Token::from_context(); }";
        assert!(!check_at("src/helper.rs", bad).is_empty());
        assert!(!check_at("magemath/src/helper.rs", bad).is_empty());
        assert!(check_at("src/tokens/generated/x86.rs", bad).is_empty());
    }

    #[test]
    fn rejects_missing_weaker_nested_and_runtime_proofs() {
        for source in [
            "fn wrap() { archmage::X64V3Token::from_context(); }",
            "use archmage::X64V3Token::from_context as make; fn wrap() { make(); }",
            "use archmage::X64V3Token::r#from_context; fn wrap() { from_context(); }",
            "mod archmage { pub struct X64V3Token; } #[archmage::rite(v3)] fn wrap() { archmage::X64V3Token::from_context(); }",
            "use local as archmage; #[archmage::rite(v3)] fn wrap() { archmage::X64V3Token::from_context(); }",
            "#[other::attune(v3)] fn wrap() { archmage::X64V3Token::from_context(); }",
            "#[archmage::rite(v3, v4)] fn wrap() { archmage::X64V4Token::from_context(); }",
            "#[archmage::attune(dispatch, _*_t)] fn wrap() { archmage::X64V3Token::from_context(); }",
            "#[archmage::rite(v3)] fn wrap() { const { archmage::X64V3Token::from_context(); } }",
            "#[archmage::rite(v2)] fn wrap() { archmage::X64V3Token::from_context(); }",
            "#[archmage::rite(v3)] fn wrap() { fn nested() { archmage::X64V3Token::from_context(); } }",
            "impl Backend for archmage::X64V3Token { fn wrap(self) { archmage::X64V3Token::from_context(); } }",
            "fn wrap(t: archmage::X64V3Token) { archmage::X64V3Token::from_context(); }",
            "#[archmage::rite(v3)] fn wrap() { Alias::from_context(); }",
            "#[archmage::rite(neon)] fn wrap() { archmage::X64V3Token::from_context(); }",
            "macro_rules! wrap { () => { archmage::X64V3Token::from_context() } }",
            "#[archmage::rite(v3)] fn wrap() { archmage::X64V3Token::r#from_context(); r#unsafe { unrelated(); } }",
        ] {
            assert!(!check(source).is_empty(), "accepted {source}");
        }

        for (source, expected_substr) in [
            (
                "use other::archmage; #[archmage::rite(v3)] fn wrap() { archmage::X64V3Token::from_context(); }",
                "shadows",
            ),
            (
                "use other::archmage::{self}; #[archmage::rite(v3)] fn wrap() { archmage::X64V3Token::from_context(); }",
                "shadows",
            ),
            (
                "extern crate other as archmage; #[archmage::rite(v3)] fn wrap() { archmage::X64V3Token::from_context(); }",
                "shadows",
            ),
            (
                "use other::*; #[archmage::rite(v3)] fn wrap() { archmage::X64V3Token::from_context(); }",
                "ambiguous",
            ),
        ] {
            let errs = check(source);
            assert!(
                errs.iter().any(|e| e.contains(expected_substr)),
                "expected `{expected_substr}` in {errs:?} for `{source}`"
            );
        }
        // Leading `::archmage` resolves unambiguously to extern prelude even with a glob import.
        assert!(
            check("use other::*; #[archmage::rite(v3)] fn wrap() { ::archmage::X64V3Token::from_context(); }")
                .is_empty()
        );
    }

    #[test]
    fn rejects_unsafe_even_in_matching_context() {
        for source in [
            "#[archmage::rite(v3)] unsafe fn wrap() { archmage::X64V3Token::from_context(); }",
            "#[archmage::rite(v3)] fn wrap() { unsafe { archmage::X64V3Token::from_context(); } }",
            "#[archmage::rite(v3)] fn wrap() { archmage::X64V3Token::from_context(); unsafe { unrelated(); } }",
        ] {
            assert!(
                check(source)
                    .iter()
                    .any(|e| e.contains("must not use unsafe")),
                "accepted {source}"
            );
        }
    }
}
