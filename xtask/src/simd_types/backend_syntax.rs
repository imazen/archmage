//! Attribute syntax shared by native backend method templates.
//!
//! Bind `let arcane = backend_syntax::arcane(token)` and interpolate `{arcane}`
//! immediately before each method. Bodies use `_self` for the receiver, and
//! emit safe intrinsics/storage helpers directly. No generated-text rewriting.

pub(super) fn arcane(token: &str) -> String {
    format!("#[arcane(_self = {token})]")
}

#[cfg(test)]
mod tests {
    #[test]
    fn native_backends_are_safe_before_formatting() {
        let files = super::super::backend_gen::generate_backend_files();
        for file in ["impls/x86_v3.rs", "impls/x86_v4.rs", "impls/arm_neon.rs"] {
            let source = &files[file];
            assert!(source.contains("#[arcane(_self = "), "{file}");
            for line in source
                .lines()
                .filter(|line| !line.trim_start().starts_with("//"))
            {
                assert!(!line.contains("unsafe"), "{file}: {line}");
                assert!(!line.contains("core::mem::transmute"), "{file}: {line}");
                assert!(!line.contains(".as_ptr()"), "{file}: {line}");
                assert!(!line.contains(".as_mut_ptr()"), "{file}: {line}");
            }
        }
    }
}
