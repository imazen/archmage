//! Shorten fully qualified paths that a generated file already imports.
//!
//! Templates spell paths out (`archmage::X64V3Token`, `<Self as
//! crate::simd::backends::I16x8Backend>::f`) while the file header imports
//! the same names, so `-W unused_qualifications` (enabled by some downstream
//! crates through RUSTFLAGS, which reaches path dependencies) reported about
//! 1,700 sites in magetypes. This pass reads the file's top-level `use`
//! declarations and rewrites matching qualified paths to the imported name.
//! It stops at the first nested `mod`, whose scope does not inherit the
//! imports, never touches `use` lines, and leaves `macro_rules!` bodies alone:
//! their paths resolve where the macro is invoked, which need not share this
//! file's imports. It rewrites library source only; proc-macro output keeps
//! absolute `::archmage::` paths because the caller's scope is unknown.

/// Imports declared at the top level of `code`: explicit `(path, name)` pairs
/// and glob prefixes.
fn top_level_imports(code: &str) -> (Vec<(String, String)>, Vec<String>) {
    let mut explicit = Vec::new();
    let mut globs = Vec::new();
    let mut statement = String::new();
    let mut in_use = false;
    for line in code.lines() {
        if !in_use {
            if !(line.starts_with("use ") || line.starts_with("pub use ")) {
                continue;
            }
            in_use = true;
            statement.clear();
        }
        statement.push_str(line.trim());
        statement.push(' ');
        if !line.trim_end().ends_with(';') {
            continue;
        }
        in_use = false;
        let body = statement
            .trim()
            .trim_start_matches("pub ")
            .trim_start_matches("use ")
            .trim_end_matches(';')
            .trim()
            .to_string();
        let (prefix, items) = match body.find("::{") {
            Some(at) if body.ends_with('}') => (
                body[..at].to_string(),
                body[at + 3..body.len() - 1]
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>(),
            ),
            _ => match body.rfind("::") {
                Some(at) => (body[..at].to_string(), vec![body[at + 2..].to_string()]),
                None => continue,
            },
        };
        for item in items {
            if item == "*" {
                globs.push(prefix.clone());
            } else if item.contains(' ') || item.contains("::") || item == "self" {
                // `X as Y`, nested groups and `self` are left alone.
            } else {
                explicit.push((format!("{prefix}::{item}"), item));
            }
        }
    }
    (explicit, globs)
}

fn is_ident_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

/// Replace `path::Name` with `Name` where it stands alone: not preceded by an
/// identifier character or `:` (so `foo::path::Name` is untouched) and not
/// followed by an identifier character.
fn replace_path(text: &str, path: &str, name: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find(path) {
        let before = out.chars().last().or_else(|| rest[..at].chars().last());
        let prev = if at > 0 {
            rest[..at].chars().last()
        } else {
            before
        };
        let next = rest[at + path.len()..].chars().next();
        let standalone =
            !prev.is_some_and(|c| is_ident_char(c) || c == ':') && !next.is_some_and(is_ident_char);
        out.push_str(&rest[..at]);
        out.push_str(if standalone { name } else { path });
        rest = &rest[at + path.len()..];
    }
    out.push_str(rest);
    out
}

/// Names of items defined at the top level of `code` (column 0).
fn top_level_items(code: &str) -> Vec<String> {
    let mut names = Vec::new();
    for line in code.lines() {
        let rest = line
            .strip_prefix("pub(crate) ")
            .or_else(|| line.strip_prefix("pub "))
            .unwrap_or(line);
        for kw in [
            "trait ", "struct ", "enum ", "type ", "fn ", "const ", "static ",
        ] {
            if let Some(after) = rest.strip_prefix(kw) {
                let name: String = after.chars().take_while(|&c| is_ident_char(c)).collect();
                if !name.is_empty() {
                    names.push(name);
                }
            }
        }
    }
    names
}

/// Rewrite qualified paths that `code`'s top-level imports already bring into
/// scope, and `super::`/`self::` paths to items this file defines at its top
/// level (the parent re-exports them, so both name the same item).
pub fn shorten_imported_paths(code: &str) -> String {
    let (mut explicit, globs) = top_level_imports(code);
    for name in top_level_items(code) {
        explicit.push((format!("super::{name}"), name.clone()));
        explicit.push((format!("self::{name}"), name));
    }
    if explicit.is_empty() && globs.is_empty() {
        return code.to_string();
    }
    let explicit_names: std::collections::BTreeSet<&str> =
        explicit.iter().map(|(_, n)| n.as_str()).collect();
    let mut out = String::with_capacity(code.len());
    let mut in_nested_mod = false;
    // Inside a `macro_rules!` definition: its brace depth, and whether its
    // opening brace has been seen yet (it may sit on the next line).
    let mut in_macro = false;
    let mut macro_depth = 0isize;
    let mut macro_opened = false;
    for line in code.split_inclusive('\n') {
        let trimmed = line.trim_start();
        if !in_nested_mod
            && (trimmed.starts_with("mod ") || trimmed.starts_with("pub mod "))
            && line.trim_end().ends_with('{')
        {
            in_nested_mod = true;
        }
        if !in_macro && trimmed.starts_with("macro_rules!") {
            in_macro = true;
            macro_depth = 0;
            macro_opened = false;
        }
        if in_macro {
            let opened = line.matches('{').count() as isize;
            let closed = line.matches('}').count() as isize;
            macro_opened |= opened > 0;
            macro_depth += opened - closed;
            if macro_opened && macro_depth <= 0 {
                in_macro = false;
            }
            out.push_str(line);
            continue;
        }
        if in_nested_mod || trimmed.starts_with("use ") || trimmed.starts_with("pub use ") {
            out.push_str(line);
            continue;
        }
        let mut line = line.to_string();
        for (path, name) in &explicit {
            if line.contains(path.as_str()) {
                line = replace_path(&line, path, name);
            }
        }
        for prefix in &globs {
            let needle = format!("{prefix}::");
            while let Some(at) = line.find(&needle) {
                let tail = &line[at + needle.len()..];
                let name_len = tail.find(|c: char| !is_ident_char(c)).unwrap_or(tail.len());
                let name = &tail[..name_len];
                // A glob loses to an explicit import of a different item of
                // the same name; only shorten names nothing else claims.
                let prev = line[..at].chars().last();
                if name.is_empty()
                    || explicit_names.contains(name)
                    || prev.is_some_and(|c| is_ident_char(c) || c == ':')
                {
                    break;
                }
                line = format!("{}{}", &line[..at], &line[at + needle.len()..]);
            }
        }
        out.push_str(&line);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::shorten_imported_paths;

    #[test]
    fn shortens_explicit_and_glob_imports_only() {
        let code = "use archmage::{X64V3Token, arcane};\nuse crate::simd::backends::*;\n\n\
            impl F32x4Backend for archmage::X64V3Token {}\n\
            fn f(t: archmage::X64V1Token) {}\n\
            fn g() -> <Self as crate::simd::backends::I16x8Backend>::Repr {}\n\
            #[archmage::arcane]\nfn h() { other::archmage::X64V3Token; }\n";
        let out = shorten_imported_paths(code);
        assert!(out.contains("impl F32x4Backend for X64V3Token {}"));
        assert!(
            out.contains("fn f(t: archmage::X64V1Token)"),
            "not imported: kept"
        );
        assert!(out.contains("<Self as I16x8Backend>::Repr"));
        assert!(out.contains("#[arcane]"));
        assert!(
            out.contains("other::archmage::X64V3Token"),
            "longer path: kept"
        );
        assert!(out.starts_with("use archmage::{X64V3Token, arcane};"));
    }

    #[test]
    fn leaves_macro_rules_bodies_alone() {
        let code = "use archmage::{X64V3Token, arcane};\nmacro_rules! m {\n    () => {\n        #[archmage::arcane]\n        fn f(_: archmage::X64V3Token) {}\n    };\n}\nfn g(_: archmage::X64V3Token) {}\n";
        let out = shorten_imported_paths(code);
        assert!(out.contains("#[archmage::arcane]"));
        assert!(out.contains("fn f(_: archmage::X64V3Token)"));
        assert!(out.contains("fn g(_: X64V3Token)"));
    }

    #[test]
    fn shortens_super_paths_to_items_defined_here() {
        let code = "use archmage::SimdToken;\npub trait I16x8Backend: SimdToken {\n    fn f(a: <Self as super::I16x8Backend>::Repr) -> <Self as super::Other>::Repr;\n}\n";
        let out = shorten_imported_paths(code);
        assert!(out.contains("a: <Self as I16x8Backend>::Repr"));
        assert!(
            out.contains("<Self as super::Other>::Repr"),
            "not defined here: kept"
        );
    }

    #[test]
    fn leaves_nested_modules_alone() {
        let code = "use archmage::X64V3Token;\nfn a(_: archmage::X64V3Token) {}\nmod inner {\n    fn b(_: archmage::X64V3Token) {}\n}\n";
        let out = shorten_imported_paths(code);
        assert!(out.contains("fn a(_: X64V3Token)"));
        assert!(out.contains("fn b(_: archmage::X64V3Token)"));
    }
}
