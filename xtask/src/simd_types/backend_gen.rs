//! Backend trait + implementation orchestrator for the generic SIMD strategy pattern.
//!
//! Coordinates per-architecture backend emitters (all driven by `types::SimdType`):
//! - `backend_traits` — `backends/sealed.rs`, `backends/<type>.rs`, `popcnt.rs`, `convert*.rs`, `backends/mod.rs`
//! - `backend_x86_v3` — `impls/x86_v3.rs`
//! - `backend_x86_v4` — `impls/x86_v4.rs`
//! - `backend_neon`   — `impls/arm_neon.rs`
//! - `backend_wasm`   — `impls/wasm128.rs`
//! - `backend_scalar` — `impls/scalar.rs`, `impls/scalar_w512.rs`
//! - `v4_delegation_gen` — `impls/x86_v3_delegated.rs`, `impls/x86_v4_*_delegated.rs`

use std::collections::BTreeMap;

use indoc::{formatdoc, indoc};

/// Generate all backend trait definitions and implementations.
///
/// Returns a map of relative paths (under `magetypes/src/simd/`) to file contents.
pub fn generate_backend_files() -> BTreeMap<String, String> {
    use super::backend_neon::generate_neon_file;
    use super::backend_scalar::{generate_scalar_file, generate_scalar_w512_file};
    use super::backend_traits::{
        generate_additional_convert_traits, generate_backend_trait, generate_backends_mod,
        generate_convert_traits, generate_popcnt_backend_traits, generate_sealed,
        narrow_backend_types, w512_backend_types,
    };
    use super::backend_wasm::generate_wasm_file;
    use super::backend_x86_v3::generate_x86_v3_file;
    use super::backend_x86_v4::generate_x86_v4_file;

    let narrow_types = narrow_backend_types();
    let w512_types = w512_backend_types();
    let mut files = BTreeMap::new();

    // 1. sealed.rs
    files.insert("backends/sealed.rs".to_string(), generate_sealed());

    // 2. Backend trait definitions (all 30 SIMD types: W128, W256, W512)
    for ty in narrow_types.iter().chain(w512_types.iter()) {
        files.insert(
            format!("backends/{}.rs", ty.name()),
            generate_backend_trait(*ty),
        );
    }

    // 3. Extension backend traits (popcnt for Modern token)
    files.insert(
        "backends/popcnt.rs".to_string(),
        generate_popcnt_backend_traits(&w512_types),
    );

    // 4. Conversion & bitcast trait definitions
    files.insert("backends/convert.rs".to_string(), generate_convert_traits());
    files.insert(
        "backends/convert_int.rs".to_string(),
        generate_additional_convert_traits(),
    );

    // 5. backends/mod.rs
    files.insert(
        "backends/mod.rs".to_string(),
        generate_backends_mod(&narrow_types, &w512_types),
    );

    // 6. Platform implementation files (W512 polyfills live hand-written in `impls/polyfill.rs`)
    files.insert(
        "impls/x86_v3.rs".to_string(),
        generate_x86_v3_file(&narrow_types),
    );
    files.insert(
        "impls/scalar.rs".to_string(),
        generate_scalar_file(&narrow_types),
    );
    files.insert(
        "impls/scalar_w512.rs".to_string(),
        generate_scalar_w512_file(&w512_types),
    );
    files.insert(
        "impls/arm_neon.rs".to_string(),
        generate_neon_file(&narrow_types),
    );
    files.insert(
        "impls/wasm128.rs".to_string(),
        generate_wasm_file(&narrow_types),
    );
    files.insert(
        "impls/x86_v4.rs".to_string(),
        generate_x86_v4_file(&w512_types),
    );

    // 7. Token delegation files derived from the generated `backends/*.rs` traits
    let trait_index = super::v4_delegation_gen::TraitIndex::from_files(
        files
            .iter()
            .filter(|(path, _)| path.starts_with("backends/"))
            .map(|(_, src)| src.as_str()),
    );
    let backend_trait_name =
        |name: &str| format!("{}Backend", super::generic_gen::uppercase_first(name));

    // `X64V4xToken`'s 12 base W512 backends delegated to `X64V4Token` via `self.v4()`.
    {
        let mut w512_names: Vec<String> =
            vec!["F32x16Backend".to_string(), "F32x16Internal".to_string()];
        for name in [
            "f64x8", "i8x64", "u8x64", "i16x32", "u16x32", "i32x16", "u32x16", "i64x8", "u64x8",
        ] {
            w512_names.push(backend_trait_name(name));
        }
        w512_names.push("F32x16Convert".to_string());
        let w512_refs: Vec<&str> = w512_names.iter().map(String::as_str).collect();
        files.insert(
            "impls/x86_v4_w512_delegated.rs".to_string(),
            super::v4_delegation_gen::generate_v4_w512_delegated(&trait_index, &w512_refs),
        );
    }

    // The AVX-512 tokens' W128/W256 f32 backends, forwarded to V3's.
    files.insert(
        "impls/x86_v4_f32_delegated.rs".to_string(),
        super::v4_delegation_gen::generate(&trait_index),
    );

    // Every other W128/W256 backend — f64, i8/i16/i32/i64, u8/u16/u32/u64 —
    // plus the float↔int `*Convert` and `*Bitcast` traits, delegated to V3.
    {
        let mut narrow_names: Vec<String> = [
            "f64x2", "f64x4", "i8x16", "i8x32", "i16x8", "i16x16", "i32x4", "i32x8", "i64x2",
            "i64x4", "u8x16", "u8x32", "u16x8", "u16x16", "u32x4", "u32x8", "u64x2", "u64x4",
        ]
        .into_iter()
        .map(backend_trait_name)
        .collect();
        narrow_names.extend(
            [
                "F32x4Convert",
                "F32x8Convert",
                "U32x4Bitcast",
                "U32x8Bitcast",
                "I64x2Bitcast",
                "I64x4Bitcast",
                "I8x16Bitcast",
                "I8x32Bitcast",
                "I16x8Bitcast",
                "I16x16Bitcast",
                "U64x2Bitcast",
                "U64x4Bitcast",
            ]
            .into_iter()
            .map(String::from),
        );
        let narrow_refs: Vec<&str> = narrow_names.iter().map(String::as_str).collect();
        files.insert(
            "impls/x86_v3_delegated.rs".to_string(),
            super::v4_delegation_gen::generate_v3_delegated(&trait_index, &narrow_refs),
        );
    }

    // 8. impls/mod.rs
    files.insert("impls/mod.rs".to_string(), generate_impls_mod());

    files
}

fn generate_impls_mod() -> String {
    formatdoc! {r#"
        //! Backend trait implementations for each token type.
        //!
        //! Each file implements the backend traits (e.g., `F32x8Backend`) for one
        //! token, using that platform's native intrinsics.
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.

        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        mod x86_v3;

        // Pure-V3 W128 / W256 backend delegation: all 34 narrow traits for
        // `X64V3VnniToken` (independent of `feature = "avx512"`) plus the 30
        // non-f32 narrow traits for the AVX-512 tokens under `feature = "avx512"`.
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        mod x86_v3_delegated;

        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "sse2",
            feature = "avx512",
            feature = "w512"
        ))]
        mod x86_v4;

        // `X64V4xToken`'s base W512 backends, forwarded to `X64V4Token` via
        // `self.v4()` (V4x ⊃ V4, zero `unsafe`).
        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "sse2",
            feature = "avx512",
            feature = "w512"
        ))]
        mod x86_v4_w512_delegated;

        // The AVX-512 tokens' W128 / W256 f32 backends, forwarded to V3's
        // (generated from the trait definitions; V4 ⊃ V3, so delegating is
        // sound), plus the hand-written AVX-512VL pixel-packing overrides.
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
        mod x86_v4_f32_delegated;
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
        mod x86_v4_f32_overrides;

        #[cfg(target_arch = "aarch64")]
        mod arm_neon;

        #[cfg(target_arch = "wasm32")]
        mod wasm128;

        // Hand-written width polyfills (W256 for NEON/WASM128, W512 for V3/NEON/WASM128,
        // zero `unsafe`).
        #[cfg(any(
            all(target_arch = "x86_64", target_feature = "sse2", feature = "w512"),
            target_arch = "aarch64",
            target_arch = "wasm32"
        ))]
        mod polyfill;

        mod scalar;

        #[cfg(feature = "w512")]
        mod scalar_w512;
    "#}
}

/// The uniform safety contract emitted at the top of every generated
/// backend-impls file that calls intrinsics. Every `unsafe` block in these
/// files shares the same one-line justification, so it is stated once per
/// file instead of ~500 times per file; the block-level obligations that
/// differ (references for loads/stores, layout for transmutes) are named
/// here too. `cargo xtask soundness` mechanically re-verifies the
/// feature-subset claim on every run.
pub(super) fn impls_safety_contract(token_desc: &str) -> String {
    if token_desc.contains("Wasm") {
        return formatdoc! {r#"
            //! # Safety (audit contract — checked backend boundaries)
            //!
            //! Value intrinsics use WebAssembly's safe SIMD API with `{token_desc}`.
            //! Whole-array storage copies use `crate::simd_storage`, whose POD bounds
            //! and compile-time size checks cover both v128 and arrays of v128.
            //! No raw-pointer load/store operations are generated here.
        "#};
    }
    if token_desc.contains("Neon") {
        return indoc! {"
            //! # Safety (audit contract — checked backend boundaries)
            //!
            //! `#[attune(wrap)]` checks value intrinsics against the receiver token's features.
            //! Whole-array loads/stores and bit casts use `crate::simd_storage` helpers,
            //! which require POD storage and enforce equal sizes at compile time.
            //! Token-bearing wrappers never implement the storage POD trait.
        "}
        .to_string();
    }
    indoc! {"
        //! # Safety (audit contract — checked backend boundaries)
        //!
        //! `#[attune(wrap)]` checks value intrinsics against the receiver token's features.
        //! SSE2-only arithmetic runs in `#[attune(wrap)]` regions for `X64V1Token`, the
        //! x86-64 baseline, so it inlines into callers without AVX.
        //! Whole-array loads/stores and bit casts use `crate::simd_storage` helpers,
        //! which require POD storage and enforce equal sizes at compile time.
        //! Token-bearing wrappers never implement the storage POD trait.
    "}
    .to_string()
}
