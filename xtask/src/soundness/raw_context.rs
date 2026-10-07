//! Check context-created tokens against function attributes, not runtime proofs.
use std::collections::HashSet;
use syn::visit::{self, Visit};

use crate::registry::Registry;

pub(super) fn verify(reg: &Registry, rel: &str, text: &str, errors: &mut Vec<String>) {
    if !rel.starts_with("magetypes/src") || !text.contains("from_context") {
        return;
    }
    let file = match syn::parse_file(text) {
        Ok(file) => file,
        Err(error) => {
            errors.push(format!("{rel}: cannot audit from_context: {error}"));
            return;
        }
    };
    ContextVisitor {
        reg,
        rel,
        errors,
        frame: Frame::default(),
    }
    .visit_file(&file);
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
}

impl ContextVisitor<'_> {
    fn function(&mut self, attrs: &[syn::Attribute], sig: &syn::Signature, body: &syn::Block) {
        let outer = std::mem::take(&mut self.frame);
        self.frame.name = sig.ident.to_string();
        self.frame.unsafe_code = matches!(sig.safety, syn::Safety::Unsafe(_));
        for attr in attrs {
            if attr.path().is_ident("target_feature") {
                let result = attr.parse_nested_meta(|meta| {
                    if meta.path.is_ident("enable") {
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
            } else if matches!(
                attr.path()
                    .segments
                    .last()
                    .map(|s| s.ident.to_string())
                    .as_deref(),
                Some("rite" | "token_target_features")
            ) {
                // Explicit single-tier form only. Unknown/compound forms fail closed.
                if let Ok(tier) = attr.parse_args::<syn::Ident>() {
                    let tier = tier.to_string();
                    let tier = tier.trim_start_matches('_');
                    if let Some(token) = self.reg.token.iter().find(|t| {
                        t.short_name.as_deref() == Some(tier)
                            || t.extraction_aliases.iter().any(|a| a == tier)
                    }) {
                        self.frame.features.extend(token.features.iter().cloned());
                    }
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
            .is_some_and(|s| s.ident == "from_context")
        {
            self.frame.constructions += 1;
            let parts: Vec<_> = item
                .path
                .segments
                .iter()
                .map(|s| s.ident.to_string())
                .collect();
            // Require the explicit archmage path so aliases cannot change the proof.
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

    fn visit_use_name(&mut self, item: &'ast syn::UseName) {
        if item.ident == "from_context" {
            self.errors.push(format!(
                "{}: import of from_context hides the explicit token path",
                self.rel
            ));
        }
    }

    fn visit_use_rename(&mut self, item: &'ast syn::UseRename) {
        if item.ident == "from_context" {
            self.errors.push(format!(
                "{}: alias of from_context hides the explicit token path",
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
        let tokens = item.tokens.to_string();
        if tokens
            .split(|c: char| !c.is_alphanumeric() && c != '_')
            .any(|t| t == "unsafe")
        {
            self.frame.unsafe_code = true;
        }
        if tokens.contains("from_context") {
            self.errors.push(format!(
                "{}: from_context inside an unexpanded macro cannot be audited",
                self.rel
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(source: &str) -> Vec<String> {
        let reg = Registry::load(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .unwrap()
                .join("token-registry.toml"),
        )
        .unwrap();
        let mut errors = Vec::new();
        verify(&reg, "magetypes/src/test.rs", source, &mut errors);
        errors
    }

    #[test]
    fn matching_and_superset_contexts() {
        for attribute in [
            "archmage::rite(v3)",
            "archmage::rite(v4)",
            "archmage::token_target_features(v3)",
        ] {
            let source =
                format!("#[{attribute}] fn wrap() {{ archmage::X64V3Token::from_context(); }}");
            assert!(check(&source).is_empty(), "{source}: {:?}", check(&source));
        }
        assert!(check("#[target_feature(enable = \"neon\")] fn wrap() { archmage::NeonToken::from_context(); }").is_empty());
    }

    #[test]
    fn rejects_missing_weaker_nested_and_runtime_proofs() {
        for source in [
            "fn wrap() { archmage::X64V3Token::from_context(); }",
            "use archmage::X64V3Token::from_context as make; fn wrap() { make(); }",
            "#[archmage::rite(v3)] fn wrap() { const { archmage::X64V3Token::from_context(); } }",
            "#[archmage::rite(v2)] fn wrap() { archmage::X64V3Token::from_context(); }",
            "#[archmage::rite(v3)] fn wrap() { fn nested() { archmage::X64V3Token::from_context(); } }",
            "impl Backend for archmage::X64V3Token { fn wrap(self) { archmage::X64V3Token::from_context(); } }",
            "fn wrap(t: archmage::X64V3Token) { archmage::X64V3Token::from_context(); }",
            "#[archmage::rite(v3)] fn wrap() { Alias::from_context(); }",
            "#[archmage::rite(neon)] fn wrap() { archmage::X64V3Token::from_context(); }",
            "macro_rules! wrap { () => { archmage::X64V3Token::from_context() } }",
        ] {
            assert!(!check(source).is_empty(), "accepted {source}");
        }
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
