//! Attribute syntax emitted directly by native backend templates.
//! The generator selects the concrete token. Suppression removes only the
//! redundant name check; rustc still checks intrinsic feature requirements.
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

/// One auditable unsafe call for baseline arithmetic. Used only for value
/// parameters; raw-pointer memory operations do not belong here.
pub(super) const SSE2_BOUNDARY: &str = r#"
macro_rules! sse2_baseline {
    (fn $name:ident(self, $($arg:ident: $ty:ty),* $(,)?) -> $ret:ty $body:block) => {
        #[inline(always)]
        fn $name(self, $($arg: $ty),*) -> $ret {
            #[target_feature(enable = "sse2")]
            #[inline]
            fn inner($($arg: $ty),*) -> $ret $body
            // SAFETY: SSE2 is guaranteed by the x86-64 architecture. The inner
            // body is safe Rust checked with precisely SSE2, not the AVX tier.
            unsafe { inner($($arg),*) }
        }
    };
}
"#;

#[cfg(test)]
mod tests {
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
