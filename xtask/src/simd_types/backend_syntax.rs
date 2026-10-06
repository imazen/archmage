//! Attribute syntax emitted directly by native backend templates.
//! The generator selects the concrete token. Suppression removes only the
//! redundant name check; rustc still checks intrinsic feature requirements.
pub(super) fn arcane(token: &str) -> String {
    // Only registry-selected concrete tokens reach this generator. Suppression
    // removes a redundant name-mismatch guard, not intrinsic feature checking.
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

/// Baseline arithmetic in an `#[arcane]` region for `X64V1Token` (SSE, SSE2).
/// The body is checked against exactly those features, and the region inlines
/// into any x86-64 caller, AVX or not. The boundary's `unsafe` is archmage's,
/// so the generated file contains none. Used only for value parameters;
/// raw-pointer memory operations do not belong here.
pub(super) const SSE2_BOUNDARY: &str = r#"
macro_rules! sse2_baseline {
    (fn $name:ident(self, $($arg:ident: $ty:ty),* $(,)?) -> $ret:ty $body:block) => {
        #[inline(always)]
        fn $name(self, $($arg: $ty),*) -> $ret {
            #[archmage::arcane(suppress_const_test)]
            fn inner(_token: archmage::X64V1Token, $($arg: $ty),*) -> $ret $body
            inner(self.v1(), $($arg),*)
        }
    };
}
"#;

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
    fn pure_native_methods_use_arcane() {
        let files = super::super::backend_gen::generate_backend_files();
        for file in ["impls/x86_v3.rs", "impls/x86_v4.rs", "impls/arm_neon.rs"] {
            let source = &files[file];
            assert!(
                source.contains("#[arcane(suppress_const_test, _self = "),
                "{file}"
            );
            // A representative pure intrinsic must no longer need an unsafe block.
            assert!(!source.contains("unsafe { _mm_add_ps"), "{file}");
            assert!(!source.contains("unsafe { vaddq_f32"), "{file}");
        }
    }
}
