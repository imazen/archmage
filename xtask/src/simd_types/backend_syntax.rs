//! Declarative boundary syntax emitted by native backend templates.
//!
//! Templates write `{arcane}` before each native trait method; [`rewrite_arcane_methods`]
//! lowers those marker attributes into `arcane_fn! { <Token> => fn ... }` invocations
//! backed by `archmage::__arcane_fn!`, avoiding ~1,700 proc-macro round-trips during
//! `magetypes` compilation while preserving `#[target_feature]` checking and `$crate`
//! concrete-token authentication.

pub(super) fn arcane(token: &str) -> String {
    format!("#[arcane(suppress_const_test, _self = {token})]")
}

/// SSE/SSE2 are baseline x86-64 features. Do not create an AVX boundary for
/// operations available to every caller. Rust checks the SSE2 inner body, so
/// accidentally introducing a stronger intrinsic is a compile error.
pub(super) fn sse2_or_arcane(token: &str, width_bits: usize) -> String {
    if width_bits == 128 {
        "sse2_baseline! {".into()
    } else {
        arcane(token)
    }
}

/// Baseline arithmetic in a declarative `X64V1Token` (SSE, SSE2) boundary.
/// The body is checked against exactly those features, and the region inlines
/// into any x86-64 caller, AVX or not. The boundary's `unsafe` is archmage's,
/// so the generated file contains none. Used only for value parameters;
/// raw-pointer memory operations do not belong here.
pub(super) const SSE2_BOUNDARY: &str = r#"
macro_rules! sse2_baseline {
    (fn $name:ident(self, $($arg:ident: $ty:ty),* $(,)?) -> $ret:ty $body:block) => {
        archmage::__arcane_fn!(@sse2 fn $name(self, $($arg: $ty),*) -> $ret $body);
    };
}
"#;

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

fn contains_ident(text: &str, ident: &str) -> bool {
    let bytes = text.as_bytes();
    let needle = ident.as_bytes();
    let mut start = 0;
    while let Some(pos) = text[start..].find(ident) {
        let abs = start + pos;
        let before_ok = abs == 0 || !is_ident_byte(bytes[abs - 1]);
        let after = abs + needle.len();
        let after_ok = after == bytes.len() || !is_ident_byte(bytes[after]);
        if before_ok && after_ok {
            return true;
        }
        start = abs + 1;
    }
    false
}

/// Lower template `#[arcane(suppress_const_test, _self = <Token>)]` method markers
/// into declarative `arcane_fn! { <Token> => fn ... }` invocations.
pub(super) fn rewrite_arcane_methods(source: String) -> String {
    let lines: Vec<&str> = source.lines().collect();
    let mut out = String::with_capacity(source.len());
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("#[arcane(suppress_const_test, _self = ")
            && let Some(token) = rest.strip_suffix(")]")
        {
            let indent = &line[..line.len() - line.trim_start().len()];
            let mut j = i + 1;
            let mut depth = 0isize;
            let mut started = false;
            while j < lines.len() {
                let code_part = lines[j].split("//").next().unwrap_or("");
                for ch in code_part.chars() {
                    if ch == '{' {
                        depth += 1;
                        started = true;
                    } else if ch == '}' {
                        depth -= 1;
                    }
                }
                if started && depth == 0 {
                    break;
                }
                j += 1;
            }
            assert!(
                started && depth == 0 && j < lines.len(),
                "unterminated #[arcane] method starting at line {}",
                i + 1
            );
            let method_block = lines[i + 1..=j].join("\n");
            let uses_self = contains_ident(&method_block, "_self");
            let rewritten_block = if contains_ident(&method_block, "Self") {
                method_block.replace("<Self as ", &format!("<{token} as "))
            } else {
                method_block
            };
            assert!(
                !contains_ident(&rewritten_block, "Self"),
                "unrewritten `Self` in declarative backend method:\n{rewritten_block}"
            );
            if uses_self {
                out.push_str(&format!("{indent}arcane_fn! {{ {token}, _self =>\n"));
            } else {
                out.push_str(&format!("{indent}arcane_fn! {{ {token} =>\n"));
            }
            out.push_str(&rewritten_block);
            out.push_str(" }\n");
            i = j + 1;
            continue;
        }
        out.push_str(line);
        out.push('\n');
        i += 1;
    }
    assert!(
        !out.contains("#[arcane") && !out.contains("#[archmage::arcane"),
        "leftover proc-macro #[arcane] attribute in generated backend impls"
    );
    out
}

#[cfg(test)]
mod tests {
    #[test]
    fn generic_storage_uses_checked_helpers_before_formatting() {
        let registry = crate::registry::Registry::load(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../token-registry.toml"),
        )
        .unwrap();
        for (file, source) in super::super::generic_gen::generate_generic_files(&registry) {
            for line in source
                .lines()
                .filter(|line| !line.trim_start().starts_with("//"))
            {
                assert!(!line.contains("unsafe {"), "{file}: {line}");
                assert!(!line.contains("core::ptr::"), "{file}: {line}");
                assert!(!line.contains("core::mem::transmute"), "{file}: {line}");
                assert!(!line.contains("from_raw_parts"), "{file}: {line}");
            }
        }
    }

    #[test]
    fn pure_native_methods_use_declarative_arcane_fn() {
        let files = super::super::backend_gen::generate_backend_files();
        for file in ["impls/x86_v3.rs", "impls/x86_v4.rs", "impls/arm_neon.rs"] {
            let source = &files[file];
            assert!(source.contains("arcane_fn! { "), "{file}");
            assert!(!source.contains("#[arcane"), "{file}");
            assert!(!source.contains("#[archmage::arcane"), "{file}");
            // A representative pure intrinsic must no longer need an unsafe block.
            assert!(!source.contains("unsafe { _mm_add_ps"), "{file}");
            assert!(!source.contains("unsafe { vaddq_f32"), "{file}");
        }
    }
}
