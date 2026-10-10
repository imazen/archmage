//! Raw expansion snapshots: the macros' own output, pretty-printed, before
//! rustc evaluates any `cfg`.
//!
//! The `.expanded.rs` snapshots under `tests/expand/` come from `cargo expand`,
//! which prints the crate *after* cfg evaluation: an item behind a false
//! `#[cfg]` is gone, and a true `#[cfg]` is stripped from the item. So those
//! snapshots cannot show a feature gate, and on an x86-64 host every NEON and
//! WASM variant is missing from them. The files under `tests/expand-raw/`
//! mirror the same inputs through the macro implementations directly: every
//! architecture's variant with its `#[cfg]`, every feature gate, and every
//! diagnostic a rejected input produces. Macro output is expanded again until
//! no attribute of ours is left (a `#[magetypes]` variant's `#[archmage::arcane]`
//! expands too), and `incant!` in bodies is expanded in place, so the file is
//! the complete expansion as rustc would see it before cfg evaluation.
//!
//! `ARCHMAGE_RAW_SNAPSHOTS=overwrite cargo test -p archmage-macros raw_snapshots`
//! rewrites the files; without the variable the test fails on any difference,
//! naming the files, like macrotest does.

use super::expansion_tests::expand;
use proc_macro2::TokenStream;
use quote::ToTokens;
use std::path::{Path, PathBuf};
use syn::visit_mut::VisitMut;
use syn::{Attribute, ImplItem, Item, Meta};

const INPUT_ROOT: &str = "tests/expand";
const OUTPUT_ROOT: &str = "tests/expand-raw";

/// The attribute macros this crate exports, by the last path segment.
const ATTRIBUTE_MACROS: &[&str] = &[
    "arcane",
    "simd_fn",
    "token_target_features_boundary",
    "rite",
    "token_target_features",
    "autoversion",
    "magetypes",
];

/// The function-like macros, expanded in expression and statement position.
const FUNCTION_MACROS: &[&str] = &["incant", "dispatch_variant", "simd_route"];

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

/// The first of our attributes on `attrs`, removed from the list and returned
/// as (macro name, arguments).
fn take_macro_attr(attrs: &mut Vec<Attribute>) -> Option<(String, TokenStream)> {
    let idx = attrs.iter().position(|a| {
        a.path()
            .segments
            .last()
            .is_some_and(|s| ATTRIBUTE_MACROS.contains(&s.ident.to_string().as_str()))
    })?;
    let attr = attrs.remove(idx);
    let name = attr.path().segments.last().unwrap().ident.to_string();
    let args = match &attr.meta {
        Meta::List(list) => list.tokens.clone(),
        _ => TokenStream::new(),
    };
    Some((name, args))
}

/// Expand one item (a fn, or a method inside an impl) and parse the result as
/// the items that replace it. Errors become the `compile_error!` the macro
/// would emit, so rejected inputs snapshot their diagnostic.
fn expand_item(name: &str, args: TokenStream, item: TokenStream) -> TokenStream {
    match expand(name, args, item) {
        Ok(tokens) => tokens,
        Err(err) => err.to_compile_error(),
    }
}

struct Expander;

impl Expander {
    fn expand_impl_items(&mut self, items: Vec<ImplItem>) -> Vec<ImplItem> {
        let mut out = Vec::new();
        for mut member in items {
            if let ImplItem::Fn(f) = &mut member
                && let Some((name, args)) = take_macro_attr(&mut f.attrs)
            {
                let tokens = expand_item(&name, args, f.to_token_stream());
                let wrapped: syn::ItemImpl = syn::parse2(quote::quote! { impl __Raw { #tokens } })
                    .expect("macro output parses as impl items");
                out.extend(self.expand_impl_items(wrapped.items));
            } else {
                out.push(member);
            }
        }
        out
    }

    fn expand_items(&mut self, items: Vec<Item>) -> Vec<Item> {
        let mut out = Vec::new();
        for mut item in items {
            match &mut item {
                Item::Fn(f) => {
                    if let Some((name, args)) = take_macro_attr(&mut f.attrs) {
                        let tokens = expand_item(&name, args, f.to_token_stream());
                        let file: syn::File =
                            syn::parse2(tokens).expect("macro output parses as items");
                        out.extend(self.expand_items(file.items));
                        continue;
                    }
                }
                Item::Impl(imp) => {
                    imp.items = self.expand_impl_items(std::mem::take(&mut imp.items));
                }
                Item::Mod(m) => {
                    if let Some((brace, items)) = m.content.take() {
                        m.content = Some((brace, self.expand_items(items)));
                    }
                }
                _ => {}
            }
            // Function-like macros inside bodies.
            self.visit_item_mut(&mut item);
            out.push(item);
        }
        out
    }
}

fn function_macro_name(mac: &syn::Macro) -> Option<String> {
    let last = mac.path.segments.last()?.ident.to_string();
    FUNCTION_MACROS.contains(&last.as_str()).then_some(last)
}

fn expand_function_macro(mac: &syn::Macro) -> Option<TokenStream> {
    function_macro_name(mac)?;
    let input: super::IncantInput = match syn::parse2(mac.tokens.clone()) {
        Ok(input) => input,
        Err(err) => return Some(err.to_compile_error()),
    };
    Some(super::incant_impl(input))
}

impl VisitMut for Expander {
    fn visit_block_mut(&mut self, block: &mut syn::Block) {
        let mut statements = Vec::new();
        for mut statement in std::mem::take(&mut block.stmts) {
            if let syn::Stmt::Item(item) = statement {
                statements.extend(
                    self.expand_items(vec![item])
                        .into_iter()
                        .map(syn::Stmt::Item),
                );
            } else {
                self.visit_stmt_mut(&mut statement);
                statements.push(statement);
            }
        }
        block.stmts = statements;
    }

    fn visit_expr_mut(&mut self, expr: &mut syn::Expr) {
        if let syn::Expr::Macro(m) = expr
            && let Some(tokens) = expand_function_macro(&m.mac)
        {
            *expr = syn::parse2(tokens).expect("incant! output parses as an expression");
            return;
        }
        syn::visit_mut::visit_expr_mut(self, expr);
    }

    fn visit_stmt_mut(&mut self, stmt: &mut syn::Stmt) {
        if let syn::Stmt::Macro(m) = stmt
            && let Some(tokens) = expand_function_macro(&m.mac)
        {
            let expr: syn::Expr =
                syn::parse2(tokens).expect("incant! output parses as an expression");
            *stmt = syn::Stmt::Expr(expr, m.semi_token);
            return;
        }
        syn::visit_mut::visit_stmt_mut(self, stmt);
    }
}

fn raw_expand(source: &str) -> String {
    let mut file: syn::File = syn::parse_str(source).expect("snapshot input parses");
    file.items = Expander.expand_items(std::mem::take(&mut file.items));
    format!(
        "// @generated by archmage-macros/src/raw_snapshots.rs: the macro output before cfg evaluation.\n{}",
        prettyplease::unparse(&file)
    )
}

fn inputs() -> Vec<PathBuf> {
    let root = repo_root().join(INPUT_ROOT);
    let mut found = Vec::new();
    let mut stack = vec![root];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("read tests/expand") {
            let path = entry.unwrap().path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|e| e == "rs")
                && !path.to_string_lossy().ends_with(".expanded.rs")
            {
                found.push(path);
            }
        }
    }
    found.sort();
    found
}

#[test]
fn raw_snapshots_match() {
    let overwrite = std::env::var("ARCHMAGE_RAW_SNAPSHOTS").is_ok_and(|v| v == "overwrite");
    let actual_dir = std::env::var_os("ARCHMAGE_RAW_ACTUAL_DIR").map(PathBuf::from);
    let root = repo_root();
    let mut stale = Vec::new();
    let mut count = 0usize;
    for input in inputs() {
        let rel = input.strip_prefix(root.join(INPUT_ROOT)).unwrap();
        let output = root.join(OUTPUT_ROOT).join(rel);
        // Windows checkouts may carry CRLF (git autocrlf); compare as LF.
        let source = std::fs::read_to_string(&input)
            .unwrap()
            .replace("\r\n", "\n");
        let expanded = raw_expand(&source);
        count += 1;
        let current = std::fs::read_to_string(&output)
            .ok()
            .map(|c| c.replace("\r\n", "\n"));
        if current.as_deref() == Some(expanded.as_str()) {
            continue;
        }
        if overwrite {
            std::fs::create_dir_all(output.parent().unwrap()).unwrap();
            std::fs::write(&output, &expanded).unwrap();
            eprintln!(
                "{} - refreshed",
                output.strip_prefix(&root).unwrap().display()
            );
        } else {
            if let Some(directory) = &actual_dir {
                let actual = directory.join(rel);
                std::fs::create_dir_all(actual.parent().unwrap()).unwrap();
                std::fs::write(actual, &expanded).unwrap();
            }
            stale.push(output.strip_prefix(&root).unwrap().display().to_string());
        }
    }
    assert!(count > 100, "expected the expansion inputs, found {count}");
    assert!(
        stale.is_empty(),
        "{} raw snapshot(s) differ from the macro output; run\n  \
         ARCHMAGE_RAW_SNAPSHOTS=overwrite cargo test -p archmage-macros raw_snapshots\n{}",
        stale.len(),
        stale.join("\n")
    );
}

#[test]
fn raw_expansion_reaches_methods_and_block_items() {
    let expanded = raw_expand(
        r#"
        struct Kernel;
        impl Kernel {
            #[archmage::autoversion(v3, scalar)]
            fn work(&self, x: u32) -> u32 { x }
        }
        fn outer() {
            #[archmage::arcane]
            fn nested(t: archmage::X64V3Token) {}
        }
    "#,
    );
    assert!(!expanded.contains("#[archmage::"), "{expanded}");
    assert!(expanded.contains("fn __arcane_work_v3"), "{expanded}");
    assert!(expanded.contains("fn __arcane_nested"), "{expanded}");
}
