//! Backend trait + implementation generation for the generic SIMD strategy pattern.
//!
//! Generates:
//! - Backend trait definitions (e.g., `F32x8Backend`) in `backends/`
//! - Sealed trait in `backends/sealed.rs`
//! - Backend implementations for each token × type in `impls/`
//! - Module routing files (`backends/mod.rs`, `impls/mod.rs`)
//!
//! Currently supports float types (f32, f64) at all widths,
//! signed 32-bit integer types (i32x4, i32x8) with conversion traits,
//! unsigned 32-bit integer types (u32x4, u32x8),
//! and signed 64-bit integer types (i64x2, i64x4).

use std::collections::BTreeMap;

use indoc::{formatdoc, indoc};

// ============================================================================
// Data Model
// ============================================================================

/// A float vector type for backend generation.
#[derive(Clone, Debug)]
struct FloatVecType {
    /// Element type: "f32" or "f64"
    elem: &'static str,
    /// Number of lanes
    lanes: usize,
    /// Width in bits (128, 256, 512)
    width_bits: usize,
}

impl FloatVecType {
    /// Trait name: "F32x8Backend", "F64x4Backend", etc.
    fn trait_name(&self) -> String {
        let upper_elem = match self.elem {
            "f32" => "F32",
            "f64" => "F64",
            _ => unreachable!(),
        };
        format!("{upper_elem}x{}Backend", self.lanes)
    }

    /// Array type for load/store: "[f32; 8]", "[f64; 4]", etc.
    fn array_type(&self) -> String {
        format!("[{}; {}]", self.elem, self.lanes)
    }

    /// x86 intrinsic prefix: "_mm", "_mm256"
    fn x86_prefix(&self) -> &'static str {
        match self.width_bits {
            128 => "_mm",
            256 => "_mm256",
            _ => unreachable!(),
        }
    }

    /// x86 intrinsic suffix: "ps" for f32, "pd" for f64
    fn x86_suffix(&self) -> &'static str {
        match self.elem {
            "f32" => "ps",
            "f64" => "pd",
            _ => unreachable!(),
        }
    }

    /// x86 inner type: "__m128", "__m256", "__m128d", "__m256d"
    fn x86_inner_type(&self) -> &'static str {
        match (self.elem, self.width_bits) {
            ("f32", 128) => "__m128",
            ("f32", 256) => "__m256",
            ("f64", 128) => "__m128d",
            ("f64", 256) => "__m256d",
            _ => unreachable!(),
        }
    }
}

/// All float vector types to generate backends for.
fn all_float_types() -> Vec<FloatVecType> {
    vec![
        FloatVecType {
            elem: "f32",
            lanes: 4,
            width_bits: 128,
        },
        FloatVecType {
            elem: "f32",
            lanes: 8,
            width_bits: 256,
        },
        FloatVecType {
            elem: "f64",
            lanes: 2,
            width_bits: 128,
        },
        FloatVecType {
            elem: "f64",
            lanes: 4,
            width_bits: 256,
        },
    ]
}

/// A signed 32-bit integer vector type for backend generation.
#[derive(Clone, Debug)]
struct I32VecType {
    /// Number of lanes
    lanes: usize,
    /// Width in bits (128, 256)
    width_bits: usize,
}

impl I32VecType {
    /// Type name: "i32x4", "i32x8"
    fn name(&self) -> String {
        format!("i32x{}", self.lanes)
    }

    /// Trait name: "I32x4Backend", "I32x8Backend"
    fn trait_name(&self) -> String {
        format!("I32x{}Backend", self.lanes)
    }

    /// Array type: "[i32; 4]", "[i32; 8]"
    fn array_type(&self) -> String {
        format!("[i32; {}]", self.lanes)
    }

    /// x86 intrinsic prefix: "_mm", "_mm256"
    fn x86_prefix(&self) -> &'static str {
        match self.width_bits {
            128 => "_mm",
            256 => "_mm256",
            _ => unreachable!(),
        }
    }

    /// x86 inner type: "__m128i", "__m256i"
    fn x86_inner_type(&self) -> &'static str {
        match self.width_bits {
            128 => "__m128i",
            256 => "__m256i",
            _ => unreachable!(),
        }
    }
}

/// All i32 vector types to generate backends for.
fn all_i32_types() -> Vec<I32VecType> {
    vec![
        I32VecType {
            lanes: 4,
            width_bits: 128,
        },
        I32VecType {
            lanes: 8,
            width_bits: 256,
        },
    ]
}

/// An unsigned 32-bit integer vector type for backend generation.
#[derive(Clone, Debug)]
struct U32VecType {
    /// Number of lanes
    lanes: usize,
    /// Width in bits (128, 256)
    width_bits: usize,
}

impl U32VecType {
    /// Trait name: "U32x4Backend", "U32x8Backend"
    fn trait_name(&self) -> String {
        format!("U32x{}Backend", self.lanes)
    }

    /// Array type: "[u32; 4]", "[u32; 8]"
    fn array_type(&self) -> String {
        format!("[u32; {}]", self.lanes)
    }

    /// x86 intrinsic prefix: "_mm", "_mm256"
    fn x86_prefix(&self) -> &'static str {
        match self.width_bits {
            128 => "_mm",
            256 => "_mm256",
            _ => unreachable!(),
        }
    }

    /// x86 inner type: "__m128i", "__m256i"
    fn x86_inner_type(&self) -> &'static str {
        match self.width_bits {
            128 => "__m128i",
            256 => "__m256i",
            _ => unreachable!(),
        }
    }
}

/// All u32 vector types to generate backends for.
fn all_u32_types() -> Vec<U32VecType> {
    vec![
        U32VecType {
            lanes: 4,
            width_bits: 128,
        },
        U32VecType {
            lanes: 8,
            width_bits: 256,
        },
    ]
}

// ============================================================================
// Public Entry Point
// ============================================================================

/// Generate all backend trait definitions and implementations.
///
/// Returns a map of relative paths (under `magetypes/src/simd/`) to file contents.
pub fn generate_backend_files() -> BTreeMap<String, String> {
    use super::backend_gen_i64::{all_i64_types, generate_x86_i64_impls};
    use super::backend_gen_remaining_int::{
        all_remaining_int_types, generate_x86_additional_convert_impls, generate_x86_int_impls,
    };
    use super::backend_gen_w512::all_w512_types;
    use super::backend_neon::generate_neon_file;
    use super::backend_scalar::{generate_scalar_file, generate_scalar_w512_file};
    use super::backend_traits::{
        generate_additional_convert_traits, generate_backend_trait, generate_backends_mod,
        generate_convert_traits, generate_popcnt_backend_traits, generate_sealed,
        narrow_backend_types, w512_backend_types,
    };
    use super::backend_wasm::generate_wasm_file;

    let types = all_float_types();
    let i32_types = all_i32_types();
    let u32_types = all_u32_types();
    let i64_types = all_i64_types();
    let remaining_int_types = all_remaining_int_types();
    let w512_types = all_w512_types();
    let narrow_types = narrow_backend_types();
    let w512_simd_types = w512_backend_types();
    let mut files = BTreeMap::new();

    // 1. sealed.rs
    files.insert("backends/sealed.rs".to_string(), generate_sealed());

    // 2. Backend trait definitions (all 30 SIMD types: W128, W256, W512)
    for ty in narrow_types.iter().chain(w512_simd_types.iter()) {
        files.insert(
            format!("backends/{}.rs", ty.name()),
            generate_backend_trait(*ty),
        );
    }

    // 3. Extension backend traits (popcnt for Modern token)
    files.insert(
        "backends/popcnt.rs".to_string(),
        generate_popcnt_backend_traits(&w512_simd_types),
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
        generate_backends_mod(&narrow_types, &w512_simd_types),
    );

    // 10. Implementation files (W512 polyfills live hand-written in `impls/polyfill.rs`)
    files.insert(
        "impls/x86_v3.rs".to_string(),
        generate_x86_impls(&types)
            + &generate_x86_i32_impls(&i32_types)
            + &generate_x86_u32_impls(&u32_types)
            + &generate_x86_i64_impls(&i64_types)
            + &generate_x86_int_impls(&remaining_int_types)
            + &generate_x86_convert_impls()
            + &generate_x86_additional_convert_impls()
            + &super::float_layout::x86_v3_f32x16_to_u16_helper(),
    );
    files.insert(
        "impls/scalar.rs".to_string(),
        generate_scalar_file(&narrow_types),
    );
    files.insert(
        "impls/scalar_w512.rs".to_string(),
        generate_scalar_w512_file(&w512_simd_types),
    );
    files.insert(
        "impls/arm_neon.rs".to_string(),
        generate_neon_file(&narrow_types),
    );
    files.insert(
        "impls/wasm128.rs".to_string(),
        generate_wasm_file(&narrow_types),
    );

    // 10b. x86 V4 native AVX-512 implementation (W512 types only; gated on
    // `avx512` + `w512` at the `mod x86_v4` declaration in `impls/mod.rs`).
    files.insert(
        "impls/x86_v4.rs".to_string(),
        generate_x86_v4_impls_file(&w512_types),
    );

    // Parse every generated `backends/*.rs` file once for all three x86
    // delegation generators below.
    let trait_index = super::v4_delegation_gen::TraitIndex::from_files(
        files
            .iter()
            .filter(|(path, _)| path.starts_with("backends/"))
            .map(|(_, src)| src.as_str()),
    );
    let backend_trait_name =
        |name: &str| format!("{}Backend", super::generic_gen::uppercase_first(name));

    // 10b2. `X64V4xToken`'s 12 base W512 backends delegated to `X64V4Token`
    // via `self.v4()`. `X64V4xToken` retains only its native VPOPCNTDQ/BITALG
    // popcnt extension impls in `x86_v4.rs`.
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

    // 10c. The AVX-512 tokens' W128/W256 f32 backends, forwarded to V3's,
    // derived from the trait definitions generated above so no method can be
    // left to its scalar default body.
    files.insert(
        "impls/x86_v4_f32_delegated.rs".to_string(),
        super::v4_delegation_gen::generate(&trait_index),
    );

    // 10d. Every other W128/W256 backend — f64, i8/i16/i32/i64, u8/u16/u32/u64
    // — plus the float↔int `*Convert` and `*Bitcast` traits, delegated to
    // V3 the same way. Without these, `T: F64x4Backend` / `T: F32x8Convert`
    // generic kernels cannot accept a V4 token (the iwssim `+v4` blocker).
    // Popcnt extension backends are excluded: they exist only natively on
    // V4x (AVX-512 VPOPCNTDQ) and have no V3 impls to forward to.
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

    // 11. impls/mod.rs
    files.insert("impls/mod.rs".to_string(), generate_impls_mod());

    files
}

// ============================================================================
// impls/mod.rs Generation
// ============================================================================

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

/// Generate the x86 V4 (native AVX-512) implementation file.
///
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

/// Contains native W512 backend impls for X64V4Token, plus Modern-specific
/// extension impls (popcnt) for X64V4xToken.
/// X64V4xToken's base W512 backends are delegated to X64V4Token in
/// `x86_v4_w512_delegated.rs`; W128 and W256 types delegate to X64V3Token.
fn generate_x86_v4_impls_file(w512_types: &[super::backend_gen_w512::W512Type]) -> String {
    use super::backend_gen_w512::{generate_popcnt_impls, generate_x86_v4_w512_impls};

    let safety = impls_safety_contract("X64V4Token / X64V4xToken");
    let mut code = formatdoc! {r#"
        //! Backend implementations for X64V4Token and X64V4xToken (native AVX-512).
        //!
        //! Implements the W512 backend traits using native 512-bit AVX-512 intrinsics
        //! for X64V4Token (base AVX-512), plus extension trait impls (popcnt) for
        //! X64V4xToken (+ VPOPCNTDQ, BITALG).
        //!
        //! X64V4xToken's base W512 backends forward to X64V4Token via `self.v4()` in
        //! `x86_v4_w512_delegated.rs`. W128 and W256 types delegate to X64V3Token.
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.
        //!
        {safety}
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        use core::arch::x86_64::*;

        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        use archmage::{{attune, X64V4Token, X64V4xToken}};

        use crate::simd::backends::*;

    "#};

    // Base backend impls for X64V4Token
    code.push_str(
        "// ============================================================================\n",
    );
    code.push_str("// X64V4Token — base AVX-512 (F/BW/CD/DQ/VL)\n");
    code.push_str(
        "// ============================================================================\n\n",
    );
    code.push_str(&generate_x86_v4_w512_impls(w512_types));

    // Extension impls: popcnt (X64V4xToken only)
    code.push_str(
        "\n// ============================================================================\n",
    );
    code.push_str("// X64V4xToken — extension: popcnt (VPOPCNTDQ + BITALG)\n");
    code.push_str(
        "// ============================================================================\n\n",
    );
    code.push_str(&generate_popcnt_impls(w512_types));

    // F32x16Convert impl for V4 (V4x delegates in `x86_v4_w512_delegated.rs`)
    code.push_str(
        "\n// ============================================================================\n",
    );
    code.push_str("// F32x16Convert — native AVX-512 conversions\n");
    code.push_str(
        "// ============================================================================\n\n",
    );
    code.push_str(&generate_x86_v4_f32x16_convert());

    code
}

fn generate_x86_v4_f32x16_convert() -> String {
    let arcane = super::backend_syntax::arcane("X64V4Token");
    formatdoc! {r#"
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "w512"))]
        impl F32x16Convert for archmage::X64V4Token {{
            {arcane}
            fn bitcast_f32_to_i32(self, a: __m512) -> __m512i {{
                _mm512_castps_si512(a)
            }}

            {arcane}
            fn bitcast_i32_to_f32(self, a: __m512i) -> __m512 {{
                _mm512_castsi512_ps(a)
            }}

            {arcane}
            fn convert_f32_to_i32(self, a: __m512) -> __m512i {{
                _mm512_cvttps_epi32(a)
            }}

            // Issue #80 fixup, mask-register form.
            {arcane}
            fn convert_f32_to_i32_saturating(self, a: __m512) -> __m512i {{
                let t = _mm512_cvttps_epi32(a);
                let big = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(a, _mm512_set1_ps(2_147_483_648.0));
                let t = _mm512_mask_mov_epi32(t, big, _mm512_set1_epi32(i32::MAX));
                let not_nan = _mm512_cmp_ps_mask::<_CMP_ORD_Q>(a, a);
                _mm512_maskz_mov_epi32(not_nan, t)
            }}

            {arcane}
            fn convert_f32_to_i32_round(self, a: __m512) -> __m512i {{
                _mm512_cvtps_epi32(a)
            }}

            {arcane}
            fn convert_i32_to_f32(self, a: __m512i) -> __m512 {{
                _mm512_cvtepi32_ps(a)
            }}
        }}

    "#}
}

// ============================================================================
// x86 Implementation Generation
// ============================================================================

fn generate_x86_impls(types: &[FloatVecType]) -> String {
    let sse2_boundary = super::backend_syntax::SSE2_BOUNDARY;
    let safety = impls_safety_contract("X64V3Token");
    let mut code = formatdoc! {r#"
        //! Backend implementations for X64V3Token (x86-64).
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.
        //!
        {safety}
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        use core::arch::x86_64::*;

        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        use archmage::{{attune, X64V3Token}};

        use crate::simd::backends::*;

        {sse2_boundary}

    "#};

    for ty in types {
        code.push_str("#[cfg(all(target_arch = \"x86_64\", target_feature = \"sse2\"))]\n");
        code.push_str(&generate_x86_float_impl(ty));
        code.push('\n');
    }

    code
}

fn generate_x86_float_impl(ty: &FloatVecType) -> String {
    let arcane = super::backend_syntax::arcane("X64V3Token");
    let baseline = super::backend_syntax::sse2_or_arcane("X64V3Token", ty.width_bits);
    let baseline_end = if ty.width_bits == 128 { "}" } else { "" };
    let trait_name = ty.trait_name();
    let inner = ty.x86_inner_type();
    let p = ty.x86_prefix();
    let s = ty.x86_suffix();
    let bits = ty.width_bits;
    let array = ty.array_type();
    let elem = ty.elem;

    // set1 intrinsic: _mm256_set1_ps, _mm_set1_pd, etc.
    let set1 = format!("{p}_set1_{s}");
    // setzero: _mm256_setzero_ps, etc.
    let setzero = format!("{p}_setzero_{s}");
    // Cast si to float: _mm256_castsi256_ps
    let cast_int_to_float = format!("{p}_castsi{bits}_{s}");
    let cast_float_to_int = format!("{p}_cast{s}_si{bits}");
    // Integer set1 for abs mask
    let set1_int = if elem == "f32" { "epi32" } else { "epi64x" };
    let abs_mask = if elem == "f32" {
        "0x7FFF_FFFFi32"
    } else {
        "0x7FFF_FFFF_FFFF_FFFFi64"
    };

    let floor_intr = format!("{p}_floor_{s}");
    let ceil_intr = format!("{p}_ceil_{s}");
    let round_intr =
        format!("{p}_round_{s}::<{{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }}>");

    // Reduction helpers
    let reduce_add_body = generate_x86_reduce_add(ty);
    let reduce_min_body = generate_x86_reduce_min(ty);
    let reduce_max_body = generate_x86_reduce_max(ty);

    // Comparison predicates
    let cmp = format!("{p}_cmp_{s}");

    // Approximation intrinsics (f32 only on AVX2)
    let (rcp_fn, rsqrt_fn) = if elem == "f32" {
        ("rcp", "rsqrt")
    } else {
        ("", "")
    };

    let approx_section = if !rcp_fn.is_empty() {
        formatdoc! {r#"
            {arcane}
            fn rcp_approx(self, a: {inner}) -> {inner} {{
                {p}_{rcp_fn}_{s}(a)
            }}

            {arcane}
            fn rsqrt_approx(self, a: {inner}) -> {inner} {{
                {p}_{rsqrt_fn}_{s}(a)
            }}

            // Estimate + one FMA-Newton step + a branchless rail rescue: the
            // fastest form meeting the bare-name contract (<= 4 ULP AND exact
            // IEEE rails at ±0/±inf/NaN). The Newton error term `e = 1 - a*r`
            // is NaN exactly on rail lanes (inf*0), so an unordered _self-
            // compare selects them and blendv rescues those lanes to the raw
            // estimate — which is rail-EXACT on x86 (rcpps(±0) = ±inf,
            // rcpps(±inf) = ±0, signed; likewise rsqrtps). Measured cost of
            // the rescue vs the plain Newton form on Zen 5: recip +2.6%,
            // rsqrt +20% — vs +21%/+232% for exact division
            // (benchmarks/recip_x86_zen5-9950x3d_2026-09-03.md). Subnormal
            // inputs remain unspecified at this tier (rcpps flushes them);
            // `recip_portable`/`rsqrt_portable` divide: 0 ULP + subnormals.
            {arcane}
            fn recip(self, a: {inner}) -> {inner} {{
                let one = {{ {p}_set1_{s}(1.0) }};
                let r = <Self as {trait_name}>::rcp_approx(_self, a);
                {{
                    let e = {p}_fnmadd_{s}(a, r, one);
                    let refined = {p}_fmadd_{s}(r, e, r);
                    let bad = {p}_cmp_{s}::<_CMP_UNORD_Q>(e, e);
                    {p}_blendv_{s}(refined, r, bad)
                }}
            }}

            {arcane}
            fn rsqrt(self, a: {inner}) -> {inner} {{
                let half = {{ {p}_set1_{s}(0.5) }};
                let three_halves = {{ {p}_set1_{s}(1.5) }};
                let y = <Self as {trait_name}>::rsqrt_approx(_self, a);
                {{
                    let t = {p}_mul_{s}(a, {p}_mul_{s}(y, y));
                    let refined = {p}_mul_{s}(y, {p}_fnmadd_{s}(half, t, three_halves));
                    let bad = {p}_cmp_{s}::<_CMP_UNORD_Q>(t, t);
                    {p}_blendv_{s}(refined, y, bad)
                }}
            }}
        "#}
    } else if elem == "f64" {
        // No hardware reciprocal estimate for f64 below AVX-512. Both the
        // "approx" and full methods use exact IEEE division / sqrt — full
        // precision and deterministic. Mirrors the WASM/scalar fallback, and
        // overrides the trait default (which is identity) so f64x2/f64x4 are
        // correct rather than returning the input unchanged.
        formatdoc! {r#"
            {arcane}
            fn rcp_approx(self, a: {inner}) -> {inner} {{
                let one = {{ {p}_set1_{s}(1.0) }};
                <Self as {trait_name}>::div(_self, one, a)
            }}

            {arcane}
            fn rsqrt_approx(self, a: {inner}) -> {inner} {{
                let one = {{ {p}_set1_{s}(1.0) }};
                <Self as {trait_name}>::div(_self, one, <Self as {trait_name}>::sqrt(_self, a))
            }}

            {arcane}
            fn recip(self, a: {inner}) -> {inner} {{
                let one = {{ {p}_set1_{s}(1.0) }};
                <Self as {trait_name}>::div(_self, one, a)
            }}

            {arcane}
            fn rsqrt(self, a: {inner}) -> {inner} {{
                let one = {{ {p}_set1_{s}(1.0) }};
                <Self as {trait_name}>::div(_self, one, <Self as {trait_name}>::sqrt(_self, a))
            }}
        "#}
    } else {
        String::new()
    };

    // Native pixel-pack override (f32 only) — round + saturating cvt/pack.
    // LLVM cannot recover this from the scalar `roundevenf` default (issue #60).
    let to_u8_x86 = if elem == "f32" && bits == 128 {
        formatdoc! {r#"

            {arcane}
            fn to_u8_bytes(self, a: {inner}) -> [u8; 4] {{
                // Clamp above at 255 first: cvtps turns +inf and anything past
                // i32::MAX into i32::MIN, which the packs would saturate to 0.
                // `min(255, a)` keeps NaN (minps returns its second operand),
                // and NaN still converts to 0, as in the scalar reference.
                let i32s = _mm_cvtps_epi32(_mm_min_ps(_mm_set1_ps(255.0), a));
                let i16s = _mm_packs_epi32(i32s, i32s);
                let u8s = _mm_packus_epi16(i16s, i16s);
                (_mm_cvtsi128_si32(u8s) as u32).to_ne_bytes()
            }}
        "#}
    } else if elem == "f32" && bits == 256 {
        formatdoc! {r#"

            {arcane}
            fn to_u8_bytes(self, a: {inner}) -> [u8; 8] {{
                // Clamp above at 255 before cvtps; see the 128-bit form.
                let i32s = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_set1_ps(255.0), a));
                let lo = _mm256_castsi256_si128(i32s);
                let hi = _mm256_extracti128_si256::<1>(i32s);
                let i16s = _mm_packs_epi32(lo, hi);
                let u8s = _mm_packus_epi16(i16s, i16s);
                (_mm_cvtsi128_si64(u8s) as u64).to_ne_bytes()
            }}
        "#}
    } else {
        String::new()
    };

    // Native pixel interleave (overrides the to_u8_bytes-based default, whose
    // generic byte interleave LLVM expands to ~8 pshufb). All 4 planes packed
    // together in-register, one pshufb to RGBA order. Round-to-even via cvtps
    // (MXCSR default); packs/packus saturate to [0,255] (= clamp). Each plane is
    // first clamped above at 255: cvtps maps +inf and values past i32::MAX to
    // i32::MIN, which would otherwise saturate to 0.
    let store_rgba_x86 = if elem == "f32" && bits == 128 {
        formatdoc! {r#"

            {arcane}
            fn store_rgba_u8_into(self, r: {inner}, g: {inner}, b: {inner}, a: {inner}, out: &mut [u8; 16]) {{
                let max = _mm_set1_ps(255.0);
                let rg = _mm_packs_epi32(
                    _mm_cvtps_epi32(_mm_min_ps(max, r)),
                    _mm_cvtps_epi32(_mm_min_ps(max, g)),
                );
                let ba = _mm_packs_epi32(
                    _mm_cvtps_epi32(_mm_min_ps(max, b)),
                    _mm_cvtps_epi32(_mm_min_ps(max, a)),
                );
                // [R0-3,G0-3,B0-3,A0-3] -> interleaved RGBA pixels 0-3.
                let packed = _mm_packus_epi16(rg, ba);
                let shuf = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
                crate::simd_storage::store(_mm_shuffle_epi8(packed, shuf), out);
            }}
        "#}
    } else if elem == "f32" && bits == 256 {
        formatdoc! {r#"

            {arcane}
            fn store_rgba_u8_into(self, r: {inner}, g: {inner}, b: {inner}, a: {inner}, out: &mut [u8; 32]) {{
                // AVX2 packs are lane-wise: lane0 holds pixels 0-3, lane1 4-7.
                let max = _mm256_set1_ps(255.0);
                let rg = _mm256_packs_epi32(
                    _mm256_cvtps_epi32(_mm256_min_ps(max, r)),
                    _mm256_cvtps_epi32(_mm256_min_ps(max, g)),
                );
                let ba = _mm256_packs_epi32(
                    _mm256_cvtps_epi32(_mm256_min_ps(max, b)),
                    _mm256_cvtps_epi32(_mm256_min_ps(max, a)),
                );
                let packed = _mm256_packus_epi16(rg, ba);
                let shuf = _mm256_setr_epi8(
                    0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
                    0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
                );
                crate::simd_storage::store(_mm256_shuffle_epi8(packed, shuf), out);
            }}
        "#}
    } else {
        String::new()
    };

    // Native 3-channel RGB AoS<->SoA shuffles and wide u8 loads/stores (issue #169).
    let float_layout = if elem == "f32" {
        super::float_layout::x86(ty.lanes, inner, &arcane)
    } else {
        String::new()
    };
    let rgb_3ch_x86 = if elem == "f32" && bits == 128 {
        formatdoc! {r#"

            {arcane}
            fn deinterleave_3ch_repr(self, rgb: [{inner}; 3]) -> [{inner}; 3] {{
                let [v0, v1, v2] = rgb;
                let t_r = _mm_shuffle_ps::<0b01_01_10_10>(v1, v2);
                let r = _mm_shuffle_ps::<0b10_00_11_00>(v0, t_r);
                let t_g0 = _mm_shuffle_ps::<0b00_00_01_01>(v0, v1);
                let t_g1 = _mm_shuffle_ps::<0b10_10_11_11>(v1, v2);
                let g = _mm_shuffle_ps::<0b10_00_10_00>(t_g0, t_g1);
                let t_b = _mm_shuffle_ps::<0b01_01_10_10>(v0, v1);
                let b = _mm_shuffle_ps::<0b11_00_10_00>(t_b, v2);
                [r, g, b]
            }}

            {arcane}
            fn interleave_3ch_repr(self, channels: [{inner}; 3]) -> [{inner}; 3] {{
                let [r, g, b] = channels;
                let rg_lo = _mm_unpacklo_ps(r, g);
                let rg_hi = _mm_unpackhi_ps(r, g);
                let gb_lo = _mm_unpacklo_ps(g, b);
                let gb_hi = _mm_unpackhi_ps(g, b);
                let br01 = _mm_shuffle_ps::<0b01_01_00_00>(b, r);
                let out0 = _mm_shuffle_ps::<0b10_00_01_00>(rg_lo, br01);
                let out1 = _mm_shuffle_ps::<0b01_00_11_10>(gb_lo, rg_hi);
                let br23 = _mm_shuffle_ps::<0b11_11_10_10>(b, r);
                let out2 = _mm_shuffle_ps::<0b11_10_10_00>(br23, gb_hi);
                [out0, out1, out2]
            }}

            {arcane}
            fn load_rgb_bytes(self, rgb: &[u8; 12]) -> ({inner}, {inner}, {inner}) {{
                let lo8: u64 = crate::simd_storage::copy(rgb.first_chunk::<8>().unwrap());
                let hi4: u32 = crate::simd_storage::copy(rgb.last_chunk::<4>().unwrap());
                let raw = _mm_insert_epi32::<2>(_mm_cvtsi64_si128(lo8 as i64), hi4 as i32);
                let r32 = _mm_shuffle_epi8(
                    raw,
                    _mm_setr_epi8(0, -1, -1, -1, 3, -1, -1, -1, 6, -1, -1, -1, 9, -1, -1, -1),
                );
                let g32 = _mm_shuffle_epi8(
                    raw,
                    _mm_setr_epi8(1, -1, -1, -1, 4, -1, -1, -1, 7, -1, -1, -1, 10, -1, -1, -1),
                );
                let b32 = _mm_shuffle_epi8(
                    raw,
                    _mm_setr_epi8(2, -1, -1, -1, 5, -1, -1, -1, 8, -1, -1, -1, 11, -1, -1, -1),
                );
                (_mm_cvtepi32_ps(r32), _mm_cvtepi32_ps(g32), _mm_cvtepi32_ps(b32))
            }}

            {arcane}
            fn store_rgb_u8_into(self, r: {inner}, g: {inner}, b: {inner}, out: &mut [u8; 12]) {{
                let max = _mm_set1_ps(255.0);
                let rg16 = _mm_packs_epi32(
                    _mm_cvtps_epi32(_mm_min_ps(max, r)),
                    _mm_cvtps_epi32(_mm_min_ps(max, g)),
                );
                let b16 = _mm_packs_epi32(
                    _mm_cvtps_epi32(_mm_min_ps(max, b)),
                    _mm_setzero_si128(),
                );
                let rgb8 = _mm_packus_epi16(rg16, b16);
                let shuf = _mm_setr_epi8(0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, -1, -1, -1, -1);
                let packed = _mm_shuffle_epi8(rgb8, shuf);
                crate::simd_storage::store(
                    _mm_cvtsi128_si64(packed) as u64,
                    out.first_chunk_mut::<8>().unwrap(),
                );
                crate::simd_storage::store(
                    _mm_extract_epi32::<2>(packed) as u32,
                    out.last_chunk_mut::<4>().unwrap(),
                );
            }}
        "#}
    } else if elem == "f32" && bits == 256 {
        // Since gcd(3, 8) = 1, 2 blends + 1 vpermps deinterleave (and inverse);
        // load_rgb_bytes reads overlapping 16B chunks at +0 and +8 within [u8; 24].
        formatdoc! {r#"

            {arcane}
            fn deinterleave_3ch_repr(self, rgb: [{inner}; 3]) -> [{inner}; 3] {{
                let [v0, v1, v2] = rgb;
                let r_mix = _mm256_blend_ps::<0x24>(_mm256_blend_ps::<0x92>(v0, v1), v2);
                let g_mix = _mm256_blend_ps::<0x49>(_mm256_blend_ps::<0x24>(v0, v1), v2);
                let b_mix = _mm256_blend_ps::<0x92>(_mm256_blend_ps::<0x49>(v0, v1), v2);
                [
                    _mm256_permutevar8x32_ps(r_mix, _mm256_setr_epi32(0, 3, 6, 1, 4, 7, 2, 5)),
                    _mm256_permutevar8x32_ps(g_mix, _mm256_setr_epi32(1, 4, 7, 2, 5, 0, 3, 6)),
                    _mm256_permutevar8x32_ps(b_mix, _mm256_setr_epi32(2, 5, 0, 3, 6, 1, 4, 7)),
                ]
            }}

            {arcane}
            fn interleave_3ch_repr(self, channels: [{inner}; 3]) -> [{inner}; 3] {{
                let [r, g, b] = channels;
                let rp = _mm256_permutevar8x32_ps(r, _mm256_setr_epi32(0, 3, 6, 1, 4, 7, 2, 5));
                let gp = _mm256_permutevar8x32_ps(g, _mm256_setr_epi32(5, 0, 3, 6, 1, 4, 7, 2));
                let bp = _mm256_permutevar8x32_ps(b, _mm256_setr_epi32(2, 5, 0, 3, 6, 1, 4, 7));
                [
                    _mm256_blend_ps::<0x24>(_mm256_blend_ps::<0x92>(rp, gp), bp),
                    _mm256_blend_ps::<0x24>(_mm256_blend_ps::<0x49>(rp, bp), gp),
                    _mm256_blend_ps::<0x24>(_mm256_blend_ps::<0x92>(gp, bp), rp),
                ]
            }}

            {arcane}
            fn load_rgb_bytes(self, rgb: &[u8; 24]) -> ({inner}, {inner}, {inner}) {{
                let lo: __m128i = crate::simd_storage::copy(rgb.first_chunk::<16>().unwrap());
                let hi: __m128i = crate::simd_storage::copy(rgb.last_chunk::<16>().unwrap());
                let raw = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi);
                let r32 = _mm256_shuffle_epi8(
                    raw,
                    _mm256_setr_epi8(
                        0, -1, -1, -1, 3, -1, -1, -1, 6, -1, -1, -1, 9, -1, -1, -1,
                        4, -1, -1, -1, 7, -1, -1, -1, 10, -1, -1, -1, 13, -1, -1, -1,
                    ),
                );
                let g32 = _mm256_shuffle_epi8(
                    raw,
                    _mm256_setr_epi8(
                        1, -1, -1, -1, 4, -1, -1, -1, 7, -1, -1, -1, 10, -1, -1, -1,
                        5, -1, -1, -1, 8, -1, -1, -1, 11, -1, -1, -1, 14, -1, -1, -1,
                    ),
                );
                let b32 = _mm256_shuffle_epi8(
                    raw,
                    _mm256_setr_epi8(
                        2, -1, -1, -1, 5, -1, -1, -1, 8, -1, -1, -1, 11, -1, -1, -1,
                        6, -1, -1, -1, 9, -1, -1, -1, 12, -1, -1, -1, 15, -1, -1, -1,
                    ),
                );
                (
                    _mm256_cvtepi32_ps(r32),
                    _mm256_cvtepi32_ps(g32),
                    _mm256_cvtepi32_ps(b32),
                )
            }}

            {arcane}
            fn store_rgb_u8_into(self, r: {inner}, g: {inner}, b: {inner}, out: &mut [u8; 24]) {{
                let max = _mm256_set1_ps(255.0);
                let rg16 = _mm256_packs_epi32(
                    _mm256_cvtps_epi32(_mm256_min_ps(max, r)),
                    _mm256_cvtps_epi32(_mm256_min_ps(max, g)),
                );
                let b16 = _mm256_packs_epi32(
                    _mm256_cvtps_epi32(_mm256_min_ps(max, b)),
                    _mm256_setzero_si256(),
                );
                let rgb8 = _mm256_packus_epi16(rg16, b16);
                let shuf = _mm256_setr_epi8(
                    0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, -1, -1, -1, -1,
                    0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, -1, -1, -1, -1,
                );
                let packed = _mm256_shuffle_epi8(rgb8, shuf);
                let lo12 = _mm256_castsi256_si128(packed);
                let hi12 = _mm256_extracti128_si256::<1>(packed);
                let lo16 = _mm_or_si128(lo12, _mm_slli_si128::<12>(hi12));
                let hi8 = _mm_srli_si128::<4>(hi12);
                crate::simd_storage::store(lo16, out.first_chunk_mut::<16>().unwrap());
                crate::simd_storage::store(
                    _mm_cvtsi128_si64(hi8) as u64,
                    out.last_chunk_mut::<8>().unwrap(),
                );
            }}
        "#}
    } else {
        String::new()
    };

    // Native AVX2 8x8 transpose (f32x8): unpck + shuffle_ps + permute2f128
    // network (24 ops), overriding the scalar gather default which AVX2 expands
    // to ~60 cross-lane ops.
    let transpose_8x8_x86 = if elem == "f32" && bits == 256 {
        formatdoc! {r#"

            {arcane}
            fn transpose_8x8_repr(self, rows: [{inner}; 8]) -> [{inner}; 8] {{
                let t0 = _mm256_unpacklo_ps(rows[0], rows[1]);
                let t1 = _mm256_unpackhi_ps(rows[0], rows[1]);
                let t2 = _mm256_unpacklo_ps(rows[2], rows[3]);
                let t3 = _mm256_unpackhi_ps(rows[2], rows[3]);
                let t4 = _mm256_unpacklo_ps(rows[4], rows[5]);
                let t5 = _mm256_unpackhi_ps(rows[4], rows[5]);
                let t6 = _mm256_unpacklo_ps(rows[6], rows[7]);
                let t7 = _mm256_unpackhi_ps(rows[6], rows[7]);
                let s0 = _mm256_shuffle_ps::<0x44>(t0, t2);
                let s1 = _mm256_shuffle_ps::<0xEE>(t0, t2);
                let s2 = _mm256_shuffle_ps::<0x44>(t1, t3);
                let s3 = _mm256_shuffle_ps::<0xEE>(t1, t3);
                let s4 = _mm256_shuffle_ps::<0x44>(t4, t6);
                let s5 = _mm256_shuffle_ps::<0xEE>(t4, t6);
                let s6 = _mm256_shuffle_ps::<0x44>(t5, t7);
                let s7 = _mm256_shuffle_ps::<0xEE>(t5, t7);
                [
                    _mm256_permute2f128_ps::<0x20>(s0, s4),
                    _mm256_permute2f128_ps::<0x20>(s1, s5),
                    _mm256_permute2f128_ps::<0x20>(s2, s6),
                    _mm256_permute2f128_ps::<0x20>(s3, s7),
                    _mm256_permute2f128_ps::<0x31>(s0, s4),
                    _mm256_permute2f128_ps::<0x31>(s1, s5),
                    _mm256_permute2f128_ps::<0x31>(s2, s6),
                    _mm256_permute2f128_ps::<0x31>(s3, s7),
                ]
            }}
        "#}
    } else {
        String::new()
    };

    formatdoc! {r#"
        impl {trait_name} for archmage::X64V3Token {{
            type Repr = {inner};

            // ====== Construction ======

            {arcane}
            fn splat(self, v: {elem}) -> {inner} {{
                {set1}(v)
            }}

            {arcane}
            fn zero(self) -> {inner} {{
                {setzero}()
            }}

            #[inline(always)]
            fn load(self, data: &{array}) -> {inner} {{
                crate::simd_storage::copy(data)
            }}

            #[inline(always)]
            fn from_array(self, arr: {array}) -> {inner} {{
                crate::simd_storage::cast(arr)
            }}

            #[inline(always)]
            fn store(self, repr: {inner}, out: &mut {array}) {{
                crate::simd_storage::store(repr, out);
            }}

            #[inline(always)]
            fn to_array(self, repr: {inner}) -> {array} {{
                crate::simd_storage::cast(repr)
            }}

            // ====== Arithmetic ======

            {baseline}
            fn add(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_add_{s}(a, b)
            }}
            {baseline_end}

            {baseline}
            fn sub(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_sub_{s}(a, b)
            }}
            {baseline_end}

            {arcane}
            fn mul(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_mul_{s}(a, b)
            }}

            {arcane}
            fn div(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_div_{s}(a, b)
            }}

            {arcane}
            fn neg(self, a: {inner}) -> {inner} {{
                {p}_xor_{s}(a, {p}_set1_{s}(-0.0))
            }}

            // ====== Math ======

            {arcane}
            fn min(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_blendv_{s}({p}_min_{s}(a, b), a, {cmp}::<_CMP_UNORD_Q>(b, b))
            }}

            {arcane}
            fn max(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_blendv_{s}({p}_max_{s}(a, b), a, {cmp}::<_CMP_UNORD_Q>(b, b))
            }}

            {arcane}
            fn minimum(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_or_{s}({p}_or_{s}({p}_min_{s}(a, b), {p}_min_{s}(b, a)), {cmp}::<_CMP_UNORD_Q>(a, b))
            }}

            {arcane}
            fn maximum(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_or_{s}({p}_and_{s}({p}_max_{s}(a, b), {p}_max_{s}(b, a)), {cmp}::<_CMP_UNORD_Q>(a, b))
            }}

            {arcane}
            fn min_fast(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_min_{s}(a, b)
            }}

            {arcane}
            fn max_fast(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_max_{s}(a, b)
            }}

            {baseline}
            fn sqrt(self, a: {inner}) -> {inner} {{
                {p}_sqrt_{s}(a)
            }}
            {baseline_end}

            {arcane}
            fn abs(self, a: {inner}) -> {inner} {{
                let mask = {cast_int_to_float}({p}_set1_{set1_int}({abs_mask}));
                {p}_and_{s}(a, mask)
            }}

            {arcane}
            fn trunc(self, a: {inner}) -> {inner} {{ {p}_round_{s}::<0x0b>(a) }}

            {arcane}
            fn mul_neg_add_relaxed(self, a: {inner}, b: {inner}, c: {inner}) -> {inner} {{ {p}_fnmadd_{s}(a, b, c) }}

            {arcane}
            fn mul_neg_add_fused(self, a: {inner}, b: {inner}, c: {inner}) -> {inner} {{ {p}_fnmadd_{s}(a, b, c) }}

            {arcane}
            fn floor(self, a: {inner}) -> {inner} {{
                {floor_intr}(a)
            }}

            {arcane}
            fn ceil(self, a: {inner}) -> {inner} {{
                {ceil_intr}(a)
            }}

            {arcane}
            fn round(self, a: {inner}) -> {inner} {{
                {round_intr}(a)
            }}

            {arcane}
            fn mul_add(self, a: {inner}, b: {inner}, c: {inner}) -> {inner} {{
                {p}_fmadd_{s}(a, b, c)
            }}

            {arcane}
            fn mul_sub(self, a: {inner}, b: {inner}, c: {inner}) -> {inner} {{
                {p}_fmsub_{s}(a, b, c)
            }}

            {arcane}
            fn mul_add_portable(self, a: {inner}, b: {inner}, c: {inner}) -> {inner} {{
                {p}_fmadd_{s}(a, b, c)
            }}

            {arcane}
            fn mul_sub_portable(self, a: {inner}, b: {inner}, c: {inner}) -> {inner} {{
                {p}_fmsub_{s}(a, b, c)
            }}

            // ====== Comparisons ======

            {arcane}
            fn simd_eq(self, a: {inner}, b: {inner}) -> {inner} {{
                {cmp}::<_CMP_EQ_OQ>(a, b)
            }}

            {arcane}
            fn simd_ne(self, a: {inner}, b: {inner}) -> {inner} {{
                {cmp}::<_CMP_NEQ_UQ>(a, b)
            }}

            {arcane}
            fn simd_lt(self, a: {inner}, b: {inner}) -> {inner} {{
                {cmp}::<_CMP_LT_OQ>(a, b)
            }}

            {arcane}
            fn simd_le(self, a: {inner}, b: {inner}) -> {inner} {{
                {cmp}::<_CMP_LE_OQ>(a, b)
            }}

            {arcane}
            fn simd_gt(self, a: {inner}, b: {inner}) -> {inner} {{
                {cmp}::<_CMP_GT_OQ>(a, b)
            }}

            {arcane}
            fn simd_ge(self, a: {inner}, b: {inner}) -> {inner} {{
                {cmp}::<_CMP_GE_OQ>(a, b)
            }}

            {arcane}
            fn blend(self, mask: {inner}, if_true: {inner}, if_false: {inner}) -> {inner} {{
                {p}_blendv_{s}(if_false, if_true, mask)
            }}

            // ====== Reductions ======

            {arcane}
            fn reduce_add(self, a: {inner}) -> {elem} {{
        {reduce_add_body}
            }}

            {arcane}
            fn reduce_min(self, a: {inner}) -> {elem} {{
        {reduce_min_body}
            }}

            {arcane}
            fn reduce_max(self, a: {inner}) -> {elem} {{
        {reduce_max_body}
            }}

            // ====== Approximations ======

        {approx_section}
            // ====== Bitwise ======

            {baseline}
            fn not(self, a: {inner}) -> {inner} {{
                let ones = {p}_set1_{set1_int}(-1);
                let as_int = {cast_float_to_int}(a);
                {cast_int_to_float}({p}_xor_si{bits}(as_int, ones))
            }}
            {baseline_end}

            {baseline}
            fn bitand(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_and_{s}(a, b)
            }}
            {baseline_end}

            {baseline}
            fn bitor(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_or_{s}(a, b)
            }}
            {baseline_end}

            {baseline}
            fn bitxor(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_xor_{s}(a, b)
            }}
            {baseline_end}
            {to_u8_x86}
            {transpose_8x8_x86}
        }}{internal_impl}
    "#,
        internal_impl = if elem == "f32" {
            formatdoc! {r#"

                #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
                impl crate::simd::backends::F32x{lanes}Internal for archmage::X64V3Token {{
                    type InternalRepr = {inner};
                {store_rgba_x86}
                {float_layout}
                {rgb_3ch_x86}
                }}"#,
                lanes = ty.lanes,
            }
        } else {
            String::new()
        },
    }
}

fn generate_x86_reduce_add(ty: &FloatVecType) -> String {
    let p = ty.x86_prefix();
    let s = ty.x86_suffix();
    let cvt = if ty.elem == "f32" {
        "_mm_cvtss_f32"
    } else {
        "_mm_cvtsd_f64"
    };
    let hadd = if ty.elem == "f32" {
        "_mm_hadd_ps"
    } else {
        "_mm_hadd_pd"
    };

    // f32 uses shuffle+add instead of _mm_hadd_ps — same adjacent-pair tree
    // `(v0+v1)+(v2+v3)`, fewer µops (2 vs 3 per step on Skylake/Zen).
    // f64 keeps _mm_hadd_pd — benchmarks show no difference at 2 f64 lanes.
    match (ty.elem, ty.width_bits) {
        ("f32", 128) => formatdoc! {"
                {{
                    let shuf = _mm_shuffle_ps::<0b10_11_00_01>(a, a);
                    let s1 = _mm_add_ps(a, shuf);
                    let s2 = _mm_add_ps(s1, _mm_movehl_ps(s1, s1));
                    {cvt}(s2)
                }}"},
        ("f64", 128) => formatdoc! {"
                {{
                    let h = {hadd}(a, a);
                    {cvt}(h)
                }}"},
        ("f32", 256) => formatdoc! {"
                {{
                    let hi = {p}_extractf128_{s}::<1>(a);
                    let lo = {p}_cast{s}256_{s}128(a);
                    let sum = _mm_add_ps(lo, hi);
                    let shuf = _mm_shuffle_ps::<0b10_11_00_01>(sum, sum);
                    let s1 = _mm_add_ps(sum, shuf);
                    let s2 = _mm_add_ps(s1, _mm_movehl_ps(s1, s1));
                    {cvt}(s2)
                }}"},
        ("f64", 256) => formatdoc! {"
                {{
                    let hi = {p}_extractf128_{s}::<1>(a);
                    let lo = {p}_cast{s}256_{s}128(a);
                    let sum = _mm_add_{s}(lo, hi);
                    let h = {hadd}(sum, sum);
                    {cvt}(h)
                }}"},
        _ => unreachable!(),
    }
}

fn generate_x86_reduce_min(ty: &FloatVecType) -> String {
    let p = ty.x86_prefix();
    let s = ty.x86_suffix();
    let cvt = if ty.elem == "f32" {
        "_mm_cvtss_f32"
    } else {
        "_mm_cvtsd_f64"
    };

    match (ty.elem, ty.width_bits) {
        ("f32", 128) => formatdoc! {"
                {{
                    let shuf = _mm_shuffle_ps::<0b10_11_00_01>(a, a);
                    let m1 = _mm_min_ps(a, shuf);
                    let shuf2 = _mm_shuffle_ps::<0b00_00_10_10>(m1, m1);
                    let m2 = _mm_min_ps(m1, shuf2);
                    {cvt}(m2)
                }}"},
        ("f64", 128) => formatdoc! {"
                {{
                    let shuf = _mm_shuffle_pd::<0b01>(a, a);
                    let m = _mm_min_pd(a, shuf);
                    {cvt}(m)
                }}"},
        ("f32", 256) => formatdoc! {"
                {{
                    let hi = {p}_extractf128_{s}::<1>(a);
                    let lo = {p}_cast{s}256_{s}128(a);
                    let m = _mm_min_{s}(lo, hi);
                    let shuf = _mm_shuffle_ps::<0b10_11_00_01>(m, m);
                    let m1 = _mm_min_ps(m, shuf);
                    let shuf2 = _mm_shuffle_ps::<0b00_00_10_10>(m1, m1);
                    let m2 = _mm_min_ps(m1, shuf2);
                    {cvt}(m2)
                }}"},
        ("f64", 256) => formatdoc! {"
                {{
                    let hi = {p}_extractf128_{s}::<1>(a);
                    let lo = {p}_cast{s}256_{s}128(a);
                    let m = _mm_min_{s}(lo, hi);
                    let shuf = _mm_shuffle_pd::<0b01>(m, m);
                    let m2 = _mm_min_pd(m, shuf);
                    {cvt}(m2)
                }}"},
        _ => unreachable!(),
    }
}

fn generate_x86_reduce_max(ty: &FloatVecType) -> String {
    let p = ty.x86_prefix();
    let s = ty.x86_suffix();
    let cvt = if ty.elem == "f32" {
        "_mm_cvtss_f32"
    } else {
        "_mm_cvtsd_f64"
    };

    match (ty.elem, ty.width_bits) {
        ("f32", 128) => formatdoc! {"
                {{
                    let shuf = _mm_shuffle_ps::<0b10_11_00_01>(a, a);
                    let m1 = _mm_max_ps(a, shuf);
                    let shuf2 = _mm_shuffle_ps::<0b00_00_10_10>(m1, m1);
                    let m2 = _mm_max_ps(m1, shuf2);
                    {cvt}(m2)
                }}"},
        ("f64", 128) => formatdoc! {"
                {{
                    let shuf = _mm_shuffle_pd::<0b01>(a, a);
                    let m = _mm_max_pd(a, shuf);
                    {cvt}(m)
                }}"},
        ("f32", 256) => formatdoc! {"
                {{
                    let hi = {p}_extractf128_{s}::<1>(a);
                    let lo = {p}_cast{s}256_{s}128(a);
                    let m = _mm_max_{s}(lo, hi);
                    let shuf = _mm_shuffle_ps::<0b10_11_00_01>(m, m);
                    let m1 = _mm_max_ps(m, shuf);
                    let shuf2 = _mm_shuffle_ps::<0b00_00_10_10>(m1, m1);
                    let m2 = _mm_max_ps(m1, shuf2);
                    {cvt}(m2)
                }}"},
        ("f64", 256) => formatdoc! {"
                {{
                    let hi = {p}_extractf128_{s}::<1>(a);
                    let lo = {p}_cast{s}256_{s}128(a);
                    let m = _mm_max_{s}(lo, hi);
                    let shuf = _mm_shuffle_pd::<0b01>(m, m);
                    let m2 = _mm_max_pd(m, shuf);
                    {cvt}(m2)
                }}"},
        _ => unreachable!(),
    }
}

// ============================================================================
// x86 I32 Implementation Generation
// ============================================================================

fn generate_x86_i32_impls(types: &[I32VecType]) -> String {
    let mut code = String::new();

    for ty in types {
        code.push_str("\n#[cfg(all(target_arch = \"x86_64\", target_feature = \"sse2\"))]\n");
        code.push_str(&generate_x86_i32_impl(ty));
        code.push('\n');
    }

    code
}

fn generate_x86_i32_impl(ty: &I32VecType) -> String {
    let integer_methods = super::backend_gen_widen_narrow::methods(&ty.name(), "X64V3Token");
    let arcane = super::backend_syntax::arcane("X64V3Token");
    let baseline = super::backend_syntax::sse2_or_arcane("X64V3Token", ty.width_bits);
    let baseline_end = if ty.width_bits == 128 { "}" } else { "" };
    let trait_name = ty.trait_name();
    let inner = ty.x86_inner_type();
    let p = ty.x86_prefix();
    let bits = ty.width_bits;
    let array = ty.array_type();
    let lanes = ty.lanes;

    let reduce_add_body = generate_x86_i32_reduce_add(ty);

    formatdoc! {r#"
        impl {trait_name} for archmage::X64V3Token {{
            type Repr = {inner};

            // ====== Construction ======

            {arcane}
            fn splat(self, v: i32) -> {inner} {{
                {p}_set1_epi32(v)
            }}

            {arcane}
            fn zero(self) -> {inner} {{
                {p}_setzero_si{bits}()
            }}

            #[inline(always)]
            fn load(self, data: &{array}) -> {inner} {{
                crate::simd_storage::copy(data)
            }}

            #[inline(always)]
            fn from_array(self, arr: {array}) -> {inner} {{
                crate::simd_storage::cast(arr)
            }}

            #[inline(always)]
            fn store(self, repr: {inner}, out: &mut {array}) {{
                crate::simd_storage::store(repr, out);
            }}

            #[inline(always)]
            fn to_array(self, repr: {inner}) -> {array} {{
                crate::simd_storage::cast(repr)
            }}

            // ====== Arithmetic ======

            {baseline}
            fn add(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_add_epi32(a, b)
            }}
            {baseline_end}

            {baseline}
            fn sub(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_sub_epi32(a, b)
            }}
            {baseline_end}

            {arcane}
            fn mul(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_mullo_epi32(a, b)
            }}

            {arcane}
            fn neg(self, a: {inner}) -> {inner} {{
                {p}_sub_epi32({p}_setzero_si{bits}(), a)
            }}

            // ====== Math ======

            {arcane}
            fn min(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_min_epi32(a, b)
            }}

            {arcane}
            fn max(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_max_epi32(a, b)
            }}

            {arcane}
            fn abs(self, a: {inner}) -> {inner} {{
                {p}_abs_epi32(a)
            }}

            // ====== Comparisons ======

            {arcane}
            fn simd_eq(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_cmpeq_epi32(a, b)
            }}

            {arcane}
            fn simd_ne(self, a: {inner}, b: {inner}) -> {inner} {{
                let eq = {p}_cmpeq_epi32(a, b);
                {p}_andnot_si{bits}(eq, {p}_set1_epi32(-1))
            }}

            {arcane}
            fn simd_lt(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_cmpgt_epi32(b, a)
            }}

            {arcane}
            fn simd_le(self, a: {inner}, b: {inner}) -> {inner} {{
                let gt = {p}_cmpgt_epi32(a, b);
                {p}_andnot_si{bits}(gt, {p}_set1_epi32(-1))
            }}

            {arcane}
            fn simd_gt(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_cmpgt_epi32(a, b)
            }}

            {arcane}
            fn simd_ge(self, a: {inner}, b: {inner}) -> {inner} {{
                let lt = {p}_cmpgt_epi32(b, a);
                {p}_andnot_si{bits}(lt, {p}_set1_epi32(-1))
            }}

            {arcane}
            fn blend(self, mask: {inner}, if_true: {inner}, if_false: {inner}) -> {inner} {{
                {p}_blendv_epi8(if_false, if_true, mask)
            }}

            // ====== Reductions ======

            {arcane}
            fn reduce_add(self, a: {inner}) -> i32 {{
        {reduce_add_body}
            }}

            // ====== Bitwise ======

            {baseline}
            fn not(self, a: {inner}) -> {inner} {{
                {p}_andnot_si{bits}(a, {p}_set1_epi32(-1))
            }}
            {baseline_end}

            {baseline}
            fn bitand(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_and_si{bits}(a, b)
            }}
            {baseline_end}

            {baseline}
            fn bitor(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_or_si{bits}(a, b)
            }}
            {baseline_end}

            {baseline}
            fn bitxor(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_xor_si{bits}(a, b)
            }}
            {baseline_end}

            // ====== Shifts ======

            {arcane}
            fn shl_const<const N: i32>(self, a: {inner}) -> {inner} {{
                {p}_slli_epi32::<N>(a)
            }}

            {arcane}
            fn shr_arithmetic_const<const N: i32>(self, a: {inner}) -> {inner} {{
                {p}_srai_epi32::<N>(a)
            }}

            {arcane}
            fn shr_logical_const<const N: i32>(self, a: {inner}) -> {inner} {{
                {p}_srli_epi32::<N>(a)
            }}

            // ====== Uniform variable shifts ======
            // PSLLD/PSRLD give 0 once the count reaches 32 and PSRAD clamps to
            // a sign fill — the portable contract is the hardware behaviour.
            // _mm_cvtsi32_si128 zero-extends, so u32::MAX is just a large
            // out-of-range count, not a negative one.

            {arcane}
            fn shl_uniform(self, a: {inner}, count: u32) -> {inner} {{
                {p}_sll_epi32(a, _mm_cvtsi32_si128(count as i32))
            }}

            {arcane}
            fn shr_logical_uniform(self, a: {inner}, count: u32) -> {inner} {{
                {p}_srl_epi32(a, _mm_cvtsi32_si128(count as i32))
            }}

            {arcane}
            fn shr_arithmetic_uniform(self, a: {inner}, count: u32) -> {inner} {{
                {p}_sra_epi32(a, _mm_cvtsi32_si128(count as i32))
            }}

            // ====== Boolean ======

            {arcane}
            fn all_true(self, a: {inner}) -> bool {{
                {p}_movemask_ps({p}_castsi{bits}_ps(a)) == {all_mask}
            }}

            {arcane}
            fn any_true(self, a: {inner}) -> bool {{
                {p}_movemask_ps({p}_castsi{bits}_ps(a)) != 0
            }}

            {arcane}
            fn bitmask(self, a: {inner}) -> u32 {{
                {p}_movemask_ps({p}_castsi{bits}_ps(a)) as u32
            }}
        {integer_methods}
        }}
    "#,
        all_mask = if lanes == 4 { "0xF" } else { "0xFF" },
    }
}

fn generate_x86_i32_reduce_add(ty: &I32VecType) -> String {
    match ty.width_bits {
        128 => formatdoc! {"
                {{
                    let hi = _mm_shuffle_epi32::<0b01_00_11_10>(a);
                    let sum = _mm_add_epi32(a, hi);
                    let hi2 = _mm_shuffle_epi32::<0b00_00_00_01>(sum);
                    let sum2 = _mm_add_epi32(sum, hi2);
                    _mm_cvtsi128_si32(sum2)
                }}"},
        256 => formatdoc! {"
                {{
                    let lo = _mm256_castsi256_si128(a);
                    let hi = _mm256_extracti128_si256::<1>(a);
                    let sum = _mm_add_epi32(lo, hi);
                    let hi2 = _mm_shuffle_epi32::<0b01_00_11_10>(sum);
                    let sum2 = _mm_add_epi32(sum, hi2);
                    let hi3 = _mm_shuffle_epi32::<0b00_00_00_01>(sum2);
                    let sum3 = _mm_add_epi32(sum2, hi3);
                    _mm_cvtsi128_si32(sum3)
                }}"},
        _ => unreachable!(),
    }
}

// ============================================================================
// x86 Conversion Implementation Generation
// ============================================================================

fn generate_x86_convert_impls() -> String {
    let arcane = super::backend_syntax::arcane("X64V3Token");
    formatdoc! {r#"

        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        impl F32x4Convert for archmage::X64V3Token {{
            {arcane}
            fn bitcast_f32_to_i32(self, a: __m128) -> __m128i {{
                _mm_castps_si128(a)
            }}

            {arcane}
            fn bitcast_i32_to_f32(self, a: __m128i) -> __m128 {{
                _mm_castsi128_ps(a)
            }}

            {arcane}
            fn convert_f32_to_i32(self, a: __m128) -> __m128i {{
                _mm_cvttps_epi32(a)
            }}

            // Issue #80 fixup: cvttps yields the i32::MIN sentinel for +overflow
            // and NaN; patch those two classes (-overflow already saturates to MIN).
            {arcane}
            fn convert_f32_to_i32_saturating(self, a: __m128) -> __m128i {{
                let t = _mm_cvttps_epi32(a);
                let big = _mm_cmp_ps::<_CMP_GE_OQ>(a, _mm_set1_ps(2_147_483_648.0));
                let t = _mm_blendv_epi8(t, _mm_set1_epi32(i32::MAX), _mm_castps_si128(big));
                let nan = _mm_cmp_ps::<_CMP_UNORD_Q>(a, a);
                _mm_andnot_si128(_mm_castps_si128(nan), t)
            }}

            {arcane}
            fn convert_f32_to_i32_round(self, a: __m128) -> __m128i {{
                _mm_cvtps_epi32(a)
            }}

            {arcane}
            fn convert_i32_to_f32(self, a: __m128i) -> __m128 {{
                _mm_cvtepi32_ps(a)
            }}
        }}

        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        impl F32x8Convert for archmage::X64V3Token {{
            {arcane}
            fn bitcast_f32_to_i32(self, a: __m256) -> __m256i {{
                _mm256_castps_si256(a)
            }}

            {arcane}
            fn bitcast_i32_to_f32(self, a: __m256i) -> __m256 {{
                _mm256_castsi256_ps(a)
            }}

            {arcane}
            fn convert_f32_to_i32(self, a: __m256) -> __m256i {{
                _mm256_cvttps_epi32(a)
            }}

            // Issue #80 fixup — see the 128-bit impl.
            {arcane}
            fn convert_f32_to_i32_saturating(self, a: __m256) -> __m256i {{
                let t = _mm256_cvttps_epi32(a);
                let big = _mm256_cmp_ps::<_CMP_GE_OQ>(a, _mm256_set1_ps(2_147_483_648.0));
                let t = _mm256_blendv_epi8(t, _mm256_set1_epi32(i32::MAX), _mm256_castps_si256(big));
                let nan = _mm256_cmp_ps::<_CMP_UNORD_Q>(a, a);
                _mm256_andnot_si256(_mm256_castps_si256(nan), t)
            }}

            {arcane}
            fn convert_f32_to_i32_round(self, a: __m256) -> __m256i {{
                _mm256_cvtps_epi32(a)
            }}

            {arcane}
            fn convert_i32_to_f32(self, a: __m256i) -> __m256 {{
                _mm256_cvtepi32_ps(a)
            }}
        }}

        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        impl U32x4Bitcast for archmage::X64V3Token {{
            #[inline(always)]
            fn bitcast_u32_to_i32(self, a: __m128i) -> __m128i {{
                a
            }}

            #[inline(always)]
            fn bitcast_i32_to_u32(self, a: __m128i) -> __m128i {{
                a
            }}
        }}

        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        impl U32x8Bitcast for archmage::X64V3Token {{
            #[inline(always)]
            fn bitcast_u32_to_i32(self, a: __m256i) -> __m256i {{
                a
            }}

            #[inline(always)]
            fn bitcast_i32_to_u32(self, a: __m256i) -> __m256i {{
                a
            }}
        }}

        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        impl I64x2Bitcast for archmage::X64V3Token {{
            {arcane}
            fn bitcast_i64_to_f64(self, a: __m128i) -> __m128d {{
                _mm_castsi128_pd(a)
            }}

            {arcane}
            fn bitcast_f64_to_i64(self, a: __m128d) -> __m128i {{
                _mm_castpd_si128(a)
            }}
        }}

        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        impl I64x4Bitcast for archmage::X64V3Token {{
            {arcane}
            fn bitcast_i64_to_f64(self, a: __m256i) -> __m256d {{
                _mm256_castsi256_pd(a)
            }}

            {arcane}
            fn bitcast_f64_to_i64(self, a: __m256d) -> __m256i {{
                _mm256_castpd_si256(a)
            }}
        }}
    "#}
}

// ============================================================================
// x86 U32 Implementation Generation
// ============================================================================

fn generate_x86_u32_impls(types: &[U32VecType]) -> String {
    let mut code = String::new();

    for ty in types {
        code.push_str("\n#[cfg(all(target_arch = \"x86_64\", target_feature = \"sse2\"))]\n");
        code.push_str(&generate_x86_u32_impl(ty));
        code.push('\n');
    }

    code
}

fn generate_x86_u32_impl(ty: &U32VecType) -> String {
    let arcane = super::backend_syntax::arcane("X64V3Token");
    let baseline = super::backend_syntax::sse2_or_arcane("X64V3Token", ty.width_bits);
    let baseline_end = if ty.width_bits == 128 { "}" } else { "" };
    let trait_name = ty.trait_name();
    let inner = ty.x86_inner_type();
    let p = ty.x86_prefix();
    let bits = ty.width_bits;
    let array = ty.array_type();
    let lanes = ty.lanes;

    let reduce_add_body = generate_x86_u32_reduce_add(ty);

    formatdoc! {r#"
        impl {trait_name} for archmage::X64V3Token {{
            type Repr = {inner};

            // ====== Construction ======

            {arcane}
            fn splat(self, v: u32) -> {inner} {{
                {p}_set1_epi32(v as i32)
            }}

            {arcane}
            fn zero(self) -> {inner} {{
                {p}_setzero_si{bits}()
            }}

            #[inline(always)]
            fn load(self, data: &{array}) -> {inner} {{
                crate::simd_storage::copy(data)
            }}

            #[inline(always)]
            fn from_array(self, arr: {array}) -> {inner} {{
                crate::simd_storage::cast(arr)
            }}

            #[inline(always)]
            fn store(self, repr: {inner}, out: &mut {array}) {{
                crate::simd_storage::store(repr, out);
            }}

            #[inline(always)]
            fn to_array(self, repr: {inner}) -> {array} {{
                crate::simd_storage::cast(repr)
            }}

            // ====== Arithmetic ======

            {baseline}
            fn add(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_add_epi32(a, b)
            }}
            {baseline_end}

            {baseline}
            fn sub(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_sub_epi32(a, b)
            }}
            {baseline_end}

            {arcane}
            fn mul(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_mullo_epi32(a, b)
            }}

            // ====== Math ======

            {arcane}
            fn min(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_min_epu32(a, b)
            }}

            {arcane}
            fn max(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_max_epu32(a, b)
            }}

            // ====== Comparisons ======

            {arcane}
            fn simd_eq(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_cmpeq_epi32(a, b)
            }}

            {arcane}
            fn simd_ne(self, a: {inner}, b: {inner}) -> {inner} {{
                let eq = {p}_cmpeq_epi32(a, b);
                {p}_andnot_si{bits}(eq, {p}_set1_epi32(-1))
            }}

            {arcane}
            fn simd_gt(self, a: {inner}, b: {inner}) -> {inner} {{
                // Unsigned comparison via bias trick: XOR both with 0x80000000
                // to convert to signed range, then use signed cmpgt.
                {{
                    let bias = {p}_set1_epi32(i32::MIN);
                    let sa = {p}_xor_si{bits}(a, bias);
                    let sb = {p}_xor_si{bits}(b, bias);
                    {p}_cmpgt_epi32(sa, sb)
                }}
            }}

            {arcane}
            fn simd_lt(self, a: {inner}, b: {inner}) -> {inner} {{
                <Self as {trait_name}>::simd_gt(_self, b, a)
            }}

            {arcane}
            fn simd_le(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_cmpeq_epi32({p}_min_epu32(a, b), a)
            }}

            {arcane}
            fn simd_ge(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_cmpeq_epi32({p}_min_epu32(a, b), b)
            }}

            {arcane}
            fn blend(self, mask: {inner}, if_true: {inner}, if_false: {inner}) -> {inner} {{
                {p}_blendv_epi8(if_false, if_true, mask)
            }}

            // ====== Reductions ======

            {arcane}
            fn reduce_add(self, a: {inner}) -> u32 {{
        {reduce_add_body}
            }}

            // ====== Bitwise ======

            {baseline}
            fn not(self, a: {inner}) -> {inner} {{
                {p}_andnot_si{bits}(a, {p}_set1_epi32(-1))
            }}
            {baseline_end}

            {baseline}
            fn bitand(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_and_si{bits}(a, b)
            }}
            {baseline_end}

            {baseline}
            fn bitor(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_or_si{bits}(a, b)
            }}
            {baseline_end}

            {baseline}
            fn bitxor(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_xor_si{bits}(a, b)
            }}
            {baseline_end}

            // ====== Shifts ======

            {arcane}
            fn shl_const<const N: i32>(self, a: {inner}) -> {inner} {{
                {p}_slli_epi32::<N>(a)
            }}

            {arcane}
            fn shr_logical_const<const N: i32>(self, a: {inner}) -> {inner} {{
                {p}_srli_epi32::<N>(a)
            }}

            // ====== Uniform variable shifts ======
            // PSLLD/PSRLD give 0 once the count reaches 32 — the portable
            // contract is the hardware behaviour. _mm_cvtsi32_si128
            // zero-extends, so u32::MAX is just a large out-of-range count.

            {arcane}
            fn shl_uniform(self, a: {inner}, count: u32) -> {inner} {{
                {p}_sll_epi32(a, _mm_cvtsi32_si128(count as i32))
            }}

            {arcane}
            fn shr_logical_uniform(self, a: {inner}, count: u32) -> {inner} {{
                {p}_srl_epi32(a, _mm_cvtsi32_si128(count as i32))
            }}

            // ====== Boolean ======

            {arcane}
            fn all_true(self, a: {inner}) -> bool {{
                {p}_movemask_ps({p}_castsi{bits}_ps(a)) == {all_mask}
            }}

            {arcane}
            fn any_true(self, a: {inner}) -> bool {{
                {p}_movemask_ps({p}_castsi{bits}_ps(a)) != 0
            }}

            {arcane}
            fn bitmask(self, a: {inner}) -> u32 {{
                {p}_movemask_ps({p}_castsi{bits}_ps(a)) as u32
            }}
        }}
    "#,
        all_mask = if lanes == 4 { "0xF" } else { "0xFF" },
    }
}

fn generate_x86_u32_reduce_add(ty: &U32VecType) -> String {
    match ty.width_bits {
        128 => formatdoc! {"
                {{
                    let hi = _mm_shuffle_epi32::<0b01_00_11_10>(a);
                    let sum = _mm_add_epi32(a, hi);
                    let hi2 = _mm_shuffle_epi32::<0b00_00_00_01>(sum);
                    let sum2 = _mm_add_epi32(sum, hi2);
                    _mm_cvtsi128_si32(sum2) as u32
                }}"},
        256 => formatdoc! {"
                {{
                    let lo = _mm256_castsi256_si128(a);
                    let hi = _mm256_extracti128_si256::<1>(a);
                    let sum = _mm_add_epi32(lo, hi);
                    let hi2 = _mm_shuffle_epi32::<0b01_00_11_10>(sum);
                    let sum2 = _mm_add_epi32(sum, hi2);
                    let hi3 = _mm_shuffle_epi32::<0b00_00_00_01>(sum2);
                    let sum3 = _mm_add_epi32(sum2, hi3);
                    _mm_cvtsi128_si32(sum3) as u32
                }}"},
        _ => unreachable!(),
    }
}
