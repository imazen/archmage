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
    /// Type name: "f32x8", "f64x4", etc.
    fn name(&self) -> String {
        format!("{}x{}", self.elem, self.lanes)
    }

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

    /// x86 intrinsic prefix: "_mm", "_mm256", "_mm512"
    fn x86_prefix(&self) -> &'static str {
        match self.width_bits {
            128 => "_mm",
            256 => "_mm256",
            512 => "_mm512",
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

    /// x86 inner type: "__m128", "__m256", "__m512", "__m128d", etc.
    fn x86_inner_type(&self) -> &'static str {
        match (self.elem, self.width_bits) {
            ("f32", 128) => "__m128",
            ("f32", 256) => "__m256",
            ("f32", 512) => "__m512",
            ("f64", 128) => "__m128d",
            ("f64", 256) => "__m256d",
            ("f64", 512) => "__m512d",
            _ => unreachable!(),
        }
    }

    /// Whether this type is native on NEON (128-bit only)
    fn native_on_neon(&self) -> bool {
        self.width_bits == 128
    }

    /// Whether this type is native on WASM (128-bit only)
    fn native_on_wasm(&self) -> bool {
        self.width_bits == 128
    }

    /// NEON repr type
    fn neon_repr(&self) -> String {
        if self.native_on_neon() {
            match self.elem {
                "f32" => "float32x4_t".to_string(),
                "f64" => "float64x2_t".to_string(),
                _ => unreachable!(),
            }
        } else {
            let native = match self.elem {
                "f32" => "float32x4_t",
                "f64" => "float64x2_t",
                _ => unreachable!(),
            };
            let count = self.width_bits / 128;
            format!("[{native}; {count}]")
        }
    }

    /// NEON element intrinsic suffix: "f32" or "f64"
    fn neon_suffix(&self) -> &'static str {
        self.elem
    }

    /// WASM element prefix: "f32x4" or "f64x2"
    fn wasm_prefix(&self) -> &'static str {
        match self.elem {
            "f32" => "f32x4",
            "f64" => "f64x2",
            _ => unreachable!(),
        }
    }

    /// Whether f32 (has rcp/rsqrt approximation intrinsics on x86)
    #[allow(dead_code)]
    fn _has_native_approx(&self) -> bool {
        self.elem == "f32"
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

    /// Whether this type is native on NEON (128-bit only)
    fn native_on_neon(&self) -> bool {
        self.width_bits == 128
    }

    /// Whether this type is native on WASM (128-bit only)
    fn native_on_wasm(&self) -> bool {
        self.width_bits == 128
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
    /// Type name: "u32x4", "u32x8"
    fn name(&self) -> String {
        format!("u32x{}", self.lanes)
    }

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

    /// Whether this type is native on NEON (128-bit only)
    fn native_on_neon(&self) -> bool {
        self.width_bits == 128
    }

    /// Whether this type is native on WASM (128-bit only)
    fn native_on_wasm(&self) -> bool {
        self.width_bits == 128
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

/// Prepend `#[cfg(feature = "w512")]` before every `impl …Backend/…Convert for`
/// block in the input so the entire W512 impl section (scalar / V3 polyfill /
/// NEON / WASM / V4 native) disappears from the build when `w512` is off.
/// Applied at the call site in `generate_backend_files` rather than threaded
/// through every per-impl generator.
///
/// The caller passes a string composed ONLY of W512 impls. For each impl block
/// we insert the feature cfg above whatever attribute already precedes it
/// (usually `#[cfg(target_arch = "…")]`) — rustc ANDs multiple `#[cfg]`s on
/// the same item, so both conditions must hold for the impl to be included.
fn gate_w512_impls(src: String) -> String {
    let mut out = String::with_capacity(src.len() + src.len() / 10);
    let mut prev_was_target_arch = false;
    for line in src.split_inclusive('\n') {
        let trimmed = line.trim_start();
        if trimmed.starts_with("#[cfg(target_arch") {
            // Gate above the existing target_arch attribute so both apply.
            out.push_str("#[cfg(feature = \"w512\")]\n");
            prev_was_target_arch = true;
        } else if trimmed.starts_with("impl ") && !prev_was_target_arch {
            // Bare `impl` with no preceding attribute (e.g. scalar impls).
            out.push_str("#[cfg(feature = \"w512\")]\n");
            prev_was_target_arch = false;
        } else if !trimmed.is_empty() {
            // Any non-blank, non-attribute line resets the look-behind.
            prev_was_target_arch = false;
        }
        out.push_str(line);
    }
    out
}

/// Generate all backend trait definitions and implementations.
///
/// Returns a map of relative paths (under `magetypes/src/simd/`) to file contents.
pub fn generate_backend_files() -> BTreeMap<String, String> {
    use super::backend_gen_i64::{
        all_i64_types, generate_i64_backend_trait, generate_neon_i64_impls,
        generate_scalar_i64_impls, generate_wasm_i64_impls, generate_x86_i64_impls,
    };
    use super::backend_gen_remaining_int::{
        all_remaining_int_types, generate_additional_convert_traits, generate_int_backend_trait,
        generate_neon_additional_convert_impls, generate_neon_int_impls,
        generate_scalar_additional_convert_impls, generate_scalar_int_impls,
        generate_wasm_additional_convert_impls, generate_wasm_int_impls,
        generate_x86_additional_convert_impls, generate_x86_int_impls,
    };
    use super::backend_gen_w512::{
        all_w512_types, generate_popcnt_backend_traits, generate_scalar_w512_impls,
        generate_w512_backend_trait,
    };

    let types = all_float_types();
    let i32_types = all_i32_types();
    let u32_types = all_u32_types();
    let i64_types = all_i64_types();
    let remaining_int_types = all_remaining_int_types();
    let w512_types = all_w512_types();
    let mut files = BTreeMap::new();

    // 1. sealed.rs
    files.insert("backends/sealed.rs".to_string(), generate_sealed());

    // 2. Backend trait definitions (float)
    for ty in &types {
        files.insert(
            format!("backends/{}.rs", ty.name()),
            generate_float_backend_trait(ty),
        );
    }

    // 3. Backend trait definitions (i32)
    for ty in &i32_types {
        files.insert(
            format!("backends/{}.rs", ty.name()),
            generate_i32_backend_trait(ty),
        );
    }

    // 4. Backend trait definitions (u32)
    for ty in &u32_types {
        files.insert(
            format!("backends/{}.rs", ty.name()),
            generate_u32_backend_trait(ty),
        );
    }

    // 5. Backend trait definitions (i64)
    for ty in &i64_types {
        files.insert(
            format!("backends/{}.rs", ty.name()),
            generate_i64_backend_trait(ty),
        );
    }

    // 6. Backend trait definitions (remaining int: i8, u8, i16, u16, u64)
    for ty in &remaining_int_types {
        files.insert(
            format!("backends/{}.rs", ty.name()),
            generate_int_backend_trait(ty),
        );
    }

    // 6b. Backend trait definitions (W512: f32x16, f64x8, i8x64, etc.)
    for ty in &w512_types {
        files.insert(
            format!("backends/{}.rs", ty.name()),
            generate_w512_backend_trait(ty),
        );
    }

    // 6c. Extension backend traits (popcnt for Modern token)
    files.insert(
        "backends/popcnt.rs".to_string(),
        generate_popcnt_backend_traits(&w512_types),
    );

    // 7. Conversion trait definitions (float/i32/u32/i64)
    files.insert("backends/convert.rs".to_string(), generate_convert_traits());

    // 8. Additional conversion traits (i8↔u8, i16↔u16, u64↔i64 bitcasts)
    files.insert(
        "backends/convert_int.rs".to_string(),
        generate_additional_convert_traits(),
    );

    // 9. backends/mod.rs
    files.insert(
        "backends/mod.rs".to_string(),
        generate_backends_mod(
            &types,
            &i32_types,
            &u32_types,
            &i64_types,
            &remaining_int_types,
            &w512_types,
        ),
    );

    // 10. Implementation files (W512 polyfills live hand-written in `impls/polyfill.rs`)
    files.insert(
        "impls/x86_v3.rs".to_string(),
        generate_x86_impls(&types, "X64V3Token", 256)
            + &generate_x86_i32_impls(&i32_types, "X64V3Token", 256)
            + &generate_x86_u32_impls(&u32_types, "X64V3Token", 256)
            + &generate_x86_i64_impls(&i64_types, "X64V3Token", 256)
            + &generate_x86_int_impls(&remaining_int_types, "X64V3Token", 256)
            + &generate_x86_convert_impls("X64V3Token")
            + &generate_x86_additional_convert_impls("X64V3Token")
            + &super::float_layout::x86_v3_f32x16_to_u16_helper(),
    );
    files.insert(
        "impls/scalar.rs".to_string(),
        generate_scalar_impls(&types)
            + &generate_scalar_i32_impls(&i32_types)
            + &generate_scalar_u32_impls(&u32_types)
            + &generate_scalar_i64_impls(&i64_types)
            + &generate_scalar_int_impls(&remaining_int_types)
            + &generate_scalar_convert_impls()
            + &generate_scalar_additional_convert_impls()
            + &gate_w512_impls(generate_scalar_w512_impls(&w512_types)),
    );
    files.insert(
        "impls/arm_neon.rs".to_string(),
        generate_neon_impls(&types)
            + &generate_neon_i32_impls(&i32_types)
            + &generate_neon_u32_impls(&u32_types)
            + &generate_neon_i64_impls(&i64_types)
            + &generate_neon_int_impls(&remaining_int_types)
            + &generate_neon_convert_impls()
            + &generate_neon_additional_convert_impls(),
    );
    files.insert(
        "impls/wasm128.rs".to_string(),
        generate_wasm_impls(&types)
            + &generate_wasm_i32_impls(&i32_types)
            + &generate_wasm_u32_impls(&u32_types)
            + &generate_wasm_i64_impls(&i64_types)
            + &generate_wasm_int_impls(&remaining_int_types)
            + &generate_wasm_convert_impls()
            + &generate_wasm_additional_convert_impls(),
    );

    // 10b. x86 V4 native AVX-512 implementation (W512 types only). Already
    // gated by `feature = "avx512"` at the `mod x86_v4` declaration, but we
    // also need the individual impls to be `w512`-gated because `avx512`
    // implies `w512` (and we want a single gate scheme).
    files.insert(
        "impls/x86_v4.rs".to_string(),
        gate_w512_impls(generate_x86_v4_impls_file(&w512_types)),
    );

    // 10b2. `X64V4xToken`'s 12 base W512 backends delegated to `X64V4Token`
    // via `self.v4()`. `X64V4xToken` retains only its native VPOPCNTDQ/BITALG
    // popcnt extension impls in `x86_v4.rs`.
    {
        let mut w512_specs: Vec<(String, String)> = vec![
            (
                "F32x16Backend".to_string(),
                files["backends/f32x16.rs"].clone(),
            ),
            (
                "F32x16Internal".to_string(),
                files["backends/f32x16.rs"].clone(),
            ),
        ];
        for name in [
            "f64x8", "i8x64", "u8x64", "i16x32", "u16x32", "i32x16", "u32x16", "i64x8", "u64x8",
        ] {
            let trait_name = format!("{}Backend", name[..1].to_uppercase() + &name[1..]);
            w512_specs.push((trait_name, files[&format!("backends/{name}.rs")].clone()));
        }
        w512_specs.push((
            "F32x16Convert".to_string(),
            files["backends/convert.rs"].clone(),
        ));
        files.insert(
            "impls/x86_v4_w512_delegated.rs".to_string(),
            super::v4_delegation_gen::generate_v4_w512_delegated(&w512_specs),
        );
    }

    // 10c. The AVX-512 tokens' W128/W256 f32 backends, forwarded to V3's,
    // derived from the trait definitions generated above so no method can be
    // left to its scalar default body.
    files.insert(
        "impls/x86_v4_f32_delegated.rs".to_string(),
        super::v4_delegation_gen::generate(
            &files["backends/f32x4.rs"],
            &files["backends/f32x8.rs"],
        ),
    );

    // 10d. Every other W128/W256 backend — f64, i8/i16/i32/i64, u8/u16/u32/u64
    // — plus the float↔int `*Convert` and `*Bitcast` traits, delegated to
    // V3 the same way. Without these, `T: F64x4Backend` / `T: F32x8Convert`
    // generic kernels cannot accept a V4 token (the iwssim `+v4` blocker).
    // Popcnt extension backends are excluded: they exist only natively on
    // V4x (AVX-512 VPOPCNTDQ) and have no V3 impls to forward to.
    {
        let mut specs: Vec<(String, String)> = Vec::new();
        for name in [
            "f64x2", "f64x4", "i8x16", "i8x32", "i16x8", "i16x16", "i32x4", "i32x8", "i64x2",
            "i64x4", "u8x16", "u8x32", "u16x8", "u16x16", "u32x4", "u32x8", "u64x2", "u64x4",
        ] {
            // `f64x4` → `F64x4Backend`, `u8x16` → `U8x16Backend`, …
            let trait_name = format!("{}Backend", name[..1].to_uppercase() + &name[1..]);
            specs.push((trait_name, files[&format!("backends/{name}.rs")].clone()));
        }
        for name in [
            "F32x4Convert",
            "F32x8Convert",
            "U32x4Bitcast",
            "U32x8Bitcast",
            "I64x2Bitcast",
            "I64x4Bitcast",
        ] {
            specs.push((name.to_string(), files["backends/convert.rs"].clone()));
        }
        for name in [
            "I8x16Bitcast",
            "I8x32Bitcast",
            "I16x8Bitcast",
            "I16x16Bitcast",
            "U64x2Bitcast",
            "U64x4Bitcast",
        ] {
            specs.push((name.to_string(), files["backends/convert_int.rs"].clone()));
        }
        files.insert(
            "impls/x86_v3_delegated.rs".to_string(),
            super::v4_delegation_gen::generate_v3_delegated(
                &files["backends/f32x4.rs"],
                &files["backends/f32x8.rs"],
                &specs,
            ),
        );
    }

    // 11. impls/mod.rs
    files.insert("impls/mod.rs".to_string(), generate_impls_mod());

    files
}

// ============================================================================
// Sealed Trait
// ============================================================================

fn generate_sealed() -> String {
    formatdoc! {r#"
        //! Sealed trait to prevent external implementations of backend traits.
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.

        /// Sealed trait — only archmage token types can implement backend traits.
        pub trait Sealed {{}}

        impl Sealed for archmage::X64V1Token {{}}
        impl Sealed for archmage::X64V2Token {{}}
        impl Sealed for archmage::X64V3Token {{}}
        impl Sealed for archmage::X64V3VnniToken {{}}
        impl Sealed for archmage::X64V4Token {{}}
        impl Sealed for archmage::X64V4xToken {{}}
        impl Sealed for archmage::Avx512Fp16Token {{}}
        impl Sealed for archmage::NeonToken {{}}
        impl Sealed for archmage::NeonAesToken {{}}
        impl Sealed for archmage::NeonSha3Token {{}}
        impl Sealed for archmage::NeonCrcToken {{}}
        impl Sealed for archmage::Wasm128Token {{}}
        impl Sealed for archmage::ScalarToken {{}}
    "#}
}

// ============================================================================
// Backend Trait Definition Generation
// ============================================================================

fn generate_float_backend_trait(ty: &FloatVecType) -> String {
    let trait_name = ty.trait_name();
    let elem = ty.elem;
    let lanes = ty.lanes;
    let array = ty.array_type();
    let internal_trait = super::float_layout::internal_trait(elem, lanes);

    // Pixel-pack default (f32 only): round-to-nearest + saturate each lane to u8.
    // Scalar fallback; x86/ARM backends override with native cvt+pack. Restores the
    // native codegen the retired concrete types had (see issue #60).
    let to_u8_trait = if elem == "f32" {
        formatdoc! {r#"

            // ====== Pixel pack (f32 only) ======

            /// Round-to-nearest-even then saturate each lane to `u8` (0..=255).
            ///
            /// Default body is scalar; `x86`/`aarch64` backends override with a
            /// native `cvt`+saturating-`pack` sequence.
            #[inline(always)]
            fn to_u8_bytes(self, a: Self::Repr) -> [u8; {lanes}] {{
                let arr = <Self as {trait_name}>::to_array(self, a);
                arr.map(|x| crate::nostd_math::round_to_small_uint(x, 255.0) as u8)
            }}

            /// Round/clamp 4 planar channels and interleave into RGBA bytes
            /// ({lanes} pixels = {rgba_bytes} bytes). Each channel converts via
            /// the native `to_u8_bytes` (so x86/aarch64 get cvt+pack, not scalar
            /// `roundevenf`); the byte interleave is left to LLVM, which recovers
            /// it to native shuffles. A backend may still override for a tighter
            /// fused sequence.
            #[deprecated(since = "0.9.32", note = "Use {vector}::store_rgba_u8")]
            #[inline(always)]
            fn store_rgba_bytes(self, r: Self::Repr, g: Self::Repr, b: Self::Repr, a: Self::Repr) -> [u8; {rgba_bytes}] {{
                let mut out = [0u8; {rgba_bytes}];
                <Self as F32x{lanes}Internal>::store_rgba_u8_into(self, r, g, b, a, &mut out);
                out
            }}
        "#, rgba_bytes = lanes * 4, vector = ty.name()}
    } else {
        String::new()
    };

    let min_max_fast_trait = formatdoc! {r#"

        /// IEEE 754-2019 minimum (`NaN`-propagating, `-0.0 < +0.0`).
        fn minimum(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

        /// IEEE 754-2019 maximum (`NaN`-propagating, `-0.0 < +0.0`).
        fn maximum(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

        /// Fast lane-wise minimum (`NaN` and `±0.0` unspecified).
        #[inline(always)]
        fn min_fast(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {{
            Self::min(self, a, b)
        }}

        /// Fast lane-wise maximum (`NaN` and `±0.0` unspecified).
        #[inline(always)]
        fn max_fast(self, a: Self::Repr, b: Self::Repr) -> Self::Repr {{
            Self::max(self, a, b)
        }}
    "#};

    // 8x8 transpose lives only on f32x8 (8 rows of 8). Default is a scalar
    // gather (LLVM recovers it to native shuffles on ARM/WASM); x86 overrides
    // with the AVX2 unpck+shuffle+permute2f128 network, since the gather bloats
    // to ~60 cross-lane ops (vbroadcastss/vblendps) on AVX2.
    let transpose_trait = if elem == "f32" && lanes == 8 {
        formatdoc! {r#"

            // ====== 8x8 transpose (f32x8 only) ======

            /// Transpose 8 row vectors of an 8x8 f32 matrix.
            #[inline(always)]
            fn transpose_8x8_repr(self, rows: [Self::Repr; 8]) -> [Self::Repr; 8] {{
                let r: [[f32; 8]; 8] =
                    core::array::from_fn(|i| <Self as {trait_name}>::to_array(self, rows[i]));
                core::array::from_fn(|i| {{
                    <Self as {trait_name}>::from_array(self, core::array::from_fn(|j| r[j][i]))
                }})
            }}
        "#}
    } else {
        String::new()
    };

    let scalar_bound = if elem == "f32" {
        format!(
            " + F32x{lanes}Internal<InternalRepr = <Self as {trait_name}>::Repr> + crate::scalar::F32ScalarBackend"
        )
    } else {
        " + crate::scalar::F64ScalarBackend".to_string()
    };

    formatdoc! {r#"
        //! Backend trait for `{name}<T>` — {lanes}-lane {elem} SIMD vector.
        //!
        //! Each token type implements this trait with its platform-native representation.
        //! The generic wrapper `{name}<T>` delegates all operations to these trait methods.
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.

        use super::sealed::Sealed;
        use archmage::SimdToken;

        {internal_trait}/// Backend implementation for {lanes}-lane {elem} SIMD vectors.
        ///
        /// Trait methods take `self` (the token) as receiver — the token value
        /// is the proof of CPU support, and requiring it as the receiver means
        /// the methods cannot be invoked via UFCS without holding one. The
        /// implementing type `Self` (a token type) determines which platform
        /// intrinsics are used. All methods are `#[inline(always)]` in
        /// implementations.
        ///
        /// # Sealed
        ///
        /// This trait is sealed — only archmage token types can implement it.
        /// The token proves CPU support was verified via `summon()`.
        #[allow(private_bounds)]
        pub trait {trait_name}: SimdToken + Sealed{scalar_bound} + Copy + 'static {{
            /// Platform-native SIMD representation.
            #[allow(private_bounds)]
            type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;

            // ====== Construction ======

            /// Broadcast scalar to all {lanes} lanes.
            fn splat(self, v: {elem}) -> Self::Repr;

            /// All lanes zero.
            fn zero(self) -> Self::Repr;

            /// Load from an aligned array.
            fn load(self, data: &{array}) -> Self::Repr;

            /// Create from array (zero-cost transmute where possible).
            fn from_array(self, arr: {array}) -> Self::Repr;

            /// Store to array.
            fn store(self, repr: Self::Repr, out: &mut {array});

            /// Convert to array.
            fn to_array(self, repr: Self::Repr) -> {array};

            // ====== Arithmetic ======

            /// Lane-wise addition.
            fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise subtraction.
            fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise multiplication.
            fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise division.
            fn div(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise negation.
            fn neg(self, a: Self::Repr) -> Self::Repr;

            // ====== Math ======

            /// Lane-wise minimum matching `{elem}::min` (returns the numeric operand on a single quiet `NaN`; signalling `NaN` and `±0.0` unspecified).
            fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise maximum matching `{elem}::max` (returns the numeric operand on a single quiet `NaN`; signalling `NaN` and `±0.0` unspecified).
            fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
            {min_max_fast_trait}

            /// Square root.
            fn sqrt(self, a: Self::Repr) -> Self::Repr;

            /// Absolute value.
            fn abs(self, a: Self::Repr) -> Self::Repr;

            /// Truncate toward zero without changing the sign of zero.
            #[inline(always)]
            fn trunc(self, a: Self::Repr) -> Self::Repr {{
                let lanes = <Self as {trait_name}>::to_array(self, a);
                <Self as {trait_name}>::from_array(self, lanes.map(crate::scalar::trunc_{elem}))
            }}

            /// Multiply-negate-add with backend-dependent rounding.
            #[inline(always)]
            fn mul_neg_add_relaxed(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {{
                <Self as {trait_name}>::mul_add(self, <Self as {trait_name}>::neg(self, a), b, c)
            }}

            /// Multiply-negate-add with one rounding.
            #[inline(always)]
            fn mul_neg_add_fused(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr {{
                <Self as {trait_name}>::mul_add_portable(self, <Self as {trait_name}>::neg(self, a), b, c)
            }}

            /// Round toward negative infinity.
            fn floor(self, a: Self::Repr) -> Self::Repr;

            /// Round toward positive infinity.
            fn ceil(self, a: Self::Repr) -> Self::Repr;

            /// Round to nearest integer.
            fn round(self, a: Self::Repr) -> Self::Repr;

            /// Multiply-add: `a * b + c`, fused where the hardware fuses.
            ///
            /// One rounding on x86 v3/v4 and NEON. Two roundings (multiply,
            /// then add) on the scalar backend and on WASM without
            /// `relaxed-simd`; relaxed WASM uses the engine's madd, which may
            /// round either way. NaN payload/sign are unspecified.
            fn mul_add(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;

            /// Multiply-sub: `a * b - c`. Same rounding contract as
            /// [`mul_add`](Self::mul_add).
            fn mul_sub(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;

            /// Multiply-add with one rounding on every backend: `a * b + c`.
            ///
            /// The same result everywhere with round-to-nearest and gradual
            /// underflow, NaN payload/sign aside. Software FMA where the
            /// selected backend has no guaranteed hardware FMA.
            fn mul_add_portable(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;

            /// Multiply-sub with one rounding on every backend: `a * b - c`.
            fn mul_sub_portable(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;

            // ====== Comparisons ======
            // Return masks where each lane is all-1s (true) or all-0s (false).

            /// Lane-wise equality.
            fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise inequality.
            fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise less-than.
            fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise less-than-or-equal.
            fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise greater-than.
            fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise greater-than-or-equal.
            fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Select lanes: where mask is all-1s pick `if_true`, else `if_false`.
            fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;

            // ====== Reductions ======

            /// Sum all {lanes} lanes.
            fn reduce_add(self, a: Self::Repr) -> {elem};

            /// Minimum across all {lanes} lanes.
            fn reduce_min(self, a: Self::Repr) -> {elem};

            /// Maximum across all {lanes} lanes.
            fn reduce_max(self, a: Self::Repr) -> {elem};

            // ====== Approximations ======

            /// Raw per-backend reciprocal seed (1/x). On x86/ARM this is the
            /// hardware estimate; the generic `rcp_approx` refines it to a ≥12-bit
            /// floor. WASM/scalar override `rcp_approx` on the generic type.
            ///
            /// **Default body returns the input unchanged** — every shipped
            /// backend overrides this with a native intrinsic. The original
            /// default `Self::div(Self::splat(1.0), a)` would require `splat`
            /// to be tokenless; with the soundness fix on `splat`, the default
            /// can no longer construct a `1.0` constant. New backends MUST
            /// override.
            #[inline(always)]
            fn rcp_approx(self, a: Self::Repr) -> Self::Repr {{ a }}

            /// Fast reciprocal square root approximation: backend-dependent
            /// (x86 ~12-bit, ARM ~8-bit, WASM full). [`rsqrt`](Self::rsqrt) is full f32 everywhere.
            ///
            /// See [`rcp_approx`](Self::rcp_approx) for default-body rationale.
            #[inline(always)]
            fn rsqrt_approx(self, a: Self::Repr) -> Self::Repr {{ a }}

            // ====== Bitwise ======

            /// Bitwise NOT.
            fn not(self, a: Self::Repr) -> Self::Repr;

            /// Bitwise AND.
            fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Bitwise OR.
            fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Bitwise XOR.
            fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            // ====== Default implementations ======

            /// Clamp values between `lo` and `hi`: `min_fast(max_fast(a, lo), hi)`,
            /// two instructions on native backends.
            ///
            /// A quiet `NaN` lane in `a` becomes `lo` on x86, NEON, strict WASM, and
            /// `ScalarToken`; signalling `NaN`, WASM `relaxed-simd` `NaN` lanes, `NaN`
            /// bounds, and `lo > hi` are unspecified per backend, and `clamp` never panics.
            #[inline(always)]
            fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {{
                <Self as {trait_name}>::min_fast(self, <Self as {trait_name}>::max_fast(self, a, lo), hi)
            }}

            /// Working-tier reciprocal — defaults to delegating to
            /// [`rcp_approx`](Self::rcp_approx) (which itself defaults to identity), so every
            /// backend MUST override. Contract: <= 4 ULP by the backend's
            /// fastest conforming path (estimate + Newton on x86/NEON f32,
            /// exact division elsewhere); rails are per-backend and may be
            /// NaN at `±0`/`±inf` where an estimate is refined. The 0 ULP +
            /// IEEE-rails tier is the generic `recip_portable` (issue #64).
            #[inline(always)]
            fn recip(self, a: Self::Repr) -> Self::Repr {{ Self::rcp_approx(self, a) }}

            /// Working-tier reciprocal square root — see [`recip`](Self::recip) for the
            /// contract shape.
            #[inline(always)]
            fn rsqrt(self, a: Self::Repr) -> Self::Repr {{ Self::rsqrt_approx(self, a) }}
            {to_u8_trait}
            {transpose_trait}
        }}
    "#,
        name = ty.name(),
    }
}

// ============================================================================
// backends/mod.rs Generation
// ============================================================================

fn generate_backends_mod(
    types: &[FloatVecType],
    i32_types: &[I32VecType],
    u32_types: &[U32VecType],
    i64_types: &[super::backend_gen_i64::I64VecType],
    remaining_int_types: &[super::backend_gen_remaining_int::IntVecType],
    w512_types: &[super::backend_gen_w512::W512Type],
) -> String {
    let mut code = formatdoc! {r#"
        //! Backend traits for generic SIMD types.
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.

        #![allow(non_camel_case_types)]

        pub(crate) mod sealed;

    "#};

    // Module declarations and re-exports (float)
    for ty in types {
        let name = ty.name();
        let trait_name = ty.trait_name();
        code.push_str(&format!("mod {name};\npub use {name}::{trait_name};\n"));
        if ty.elem == "f32" {
            code.push_str(&format!(
                "pub(crate) use {name}::F32x{}Internal;\n",
                ty.lanes
            ));
        }
        code.push('\n');
    }

    // Module declarations and re-exports (i32)
    for ty in i32_types {
        let name = ty.name();
        let trait_name = ty.trait_name();
        code.push_str(&format!("mod {name};\npub use {name}::{trait_name};\n\n"));
    }

    // Module declarations and re-exports (u32)
    for ty in u32_types {
        let name = ty.name();
        let trait_name = ty.trait_name();
        code.push_str(&format!("mod {name};\npub use {name}::{trait_name};\n\n"));
    }

    // Module declarations and re-exports (i64)
    for ty in i64_types {
        let name = ty.name();
        let trait_name = ty.trait_name();
        code.push_str(&format!("mod {name};\npub use {name}::{trait_name};\n\n"));
    }

    // Module declarations and re-exports (remaining int: i8, u8, i16, u16, u64)
    for ty in remaining_int_types {
        let name = ty.name();
        let trait_name = ty.trait_name();
        code.push_str(&format!("mod {name};\npub use {name}::{trait_name};\n\n"));
    }

    // Module declarations and re-exports (W512: f32x16, f64x8, i8x64, etc.)
    // Gated behind `w512` feature — avx512 implies w512 via the magetypes feature graph.
    for ty in w512_types {
        let name = ty.name();
        let trait_name = ty.trait_name();
        code.push_str(&format!(
            "#[cfg(feature = \"w512\")]\nmod {name};\n#[cfg(feature = \"w512\")]\npub use {name}::{trait_name};\n"
        ));
        if ty.elem == "f32" {
            code.push_str(&format!(
                "#[cfg(feature = \"w512\")]\npub(crate) use {name}::F32x{}Internal;\n",
                ty.lanes
            ));
        }
        code.push('\n');
    }

    // Extension traits (popcnt for Modern token)
    code.push_str("#[cfg(feature = \"avx512\")]\n");
    code.push_str("mod popcnt;\n");
    code.push_str("#[cfg(feature = \"avx512\")]\n");
    code.push_str("pub use popcnt::*;\n\n");

    // Conversion and bitcast traits (float/i32/u32/i64). F32x16Convert is gated on w512.
    code.push_str("mod convert;\n");
    code.push_str("pub use convert::{F32x4Convert, F32x8Convert, U32x4Bitcast, U32x8Bitcast, I64x2Bitcast, I64x4Bitcast};\n");
    code.push_str("#[cfg(feature = \"w512\")]\npub use convert::F32x16Convert;\n\n");

    // Additional conversion traits (i8↔u8, i16↔u16, u64↔i64 bitcasts)
    code.push_str("mod convert_int;\npub use convert_int::{I8x16Bitcast, I8x32Bitcast, I16x8Bitcast, I16x16Bitcast, U64x2Bitcast, U64x4Bitcast};\n\n");

    // Type aliases for ergonomic use
    for (alias, full, doc) in [
        ("x64v1", "archmage::X64V1Token", "x86-64 baseline (SSE2)."),
        (
            "x64v2",
            "archmage::X64V2Token",
            "x86-64 v2 (SSE4.2 + POPCNT).",
        ),
        ("x64v3", "archmage::X64V3Token", "x86-64 v3 (AVX2 + FMA)."),
        ("x64v4", "archmage::X64V4Token", "x86-64 v4 (AVX-512)."),
        (
            "x86_v4x",
            "archmage::X64V4xToken",
            "AVX-512 with v4x extensions.",
        ),
        ("neon", "archmage::NeonToken", "AArch64 NEON."),
        ("wasm128", "archmage::Wasm128Token", "WASM SIMD128."),
        ("scalar", "archmage::ScalarToken", "Scalar fallback."),
    ] {
        code.push_str(&format!("/// {doc}\npub type {alias} = {full};\n"));
    }

    code
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

        #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "avx512"))]
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
fn impls_safety_contract(token_desc: &str) -> String {
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
    code.push_str(&generate_x86_v4_f32x16_convert("X64V4Token"));

    code
}

fn generate_x86_v4_f32x16_convert(token: &str) -> String {
    let arcane = super::backend_syntax::arcane(token);
    formatdoc! {r#"
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "w512"))]
        impl F32x16Convert for archmage::{token} {{
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

fn generate_x86_impls(types: &[FloatVecType], token: &str, max_width: usize) -> String {
    let sse2_boundary = super::backend_syntax::SSE2_BOUNDARY;
    let safety = impls_safety_contract(token);
    let mut code = formatdoc! {r#"
        //! Backend implementations for {token} (x86-64).
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.
        //!
        {safety}
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        use core::arch::x86_64::*;

        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        use archmage::{{attune, {token}}};

        use crate::simd::backends::*;

        {sse2_boundary}

    "#};

    for ty in types {
        if ty.width_bits > max_width {
            continue;
        }
        code.push_str("#[cfg(all(target_arch = \"x86_64\", target_feature = \"sse2\"))]\n");
        code.push_str(&generate_x86_float_impl(ty, token));
        code.push('\n');
    }

    code
}

fn generate_x86_float_impl(ty: &FloatVecType, token: &str) -> String {
    let arcane = super::backend_syntax::arcane(token);
    let baseline = super::backend_syntax::sse2_or_arcane(token, ty.width_bits);
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
    let set1_int = if elem == "f32" {
        "epi32"
    } else if bits == 512 {
        "epi64"
    } else {
        "epi64x"
    };
    let abs_mask = if elem == "f32" {
        "0x7FFF_FFFFi32"
    } else {
        "0x7FFF_FFFF_FFFF_FFFFi64"
    };

    // Round intrinsics differ by width
    let (floor_intr, ceil_intr, round_intr) = if bits == 512 {
        (
            format!("{p}_roundscale_{s}::<0x01>"),
            format!("{p}_roundscale_{s}::<0x02>"),
            format!("{p}_roundscale_{s}::<0x00>"),
        )
    } else {
        (
            format!("{p}_floor_{s}"),
            format!("{p}_ceil_{s}"),
            format!("{p}_round_{s}::<{{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }}>"),
        )
    };

    // Reduction helpers
    let reduce_add_body = generate_x86_reduce_add(ty);
    let reduce_min_body = generate_x86_reduce_min(ty);
    let reduce_max_body = generate_x86_reduce_max(ty);

    // Comparison predicates
    let cmp = format!("{p}_cmp_{s}");

    // Approximation intrinsics (f32 only, or f64 on AVX-512)
    let (rcp_fn, rsqrt_fn) = if elem == "f32" {
        if bits == 512 {
            ("rcp14", "rsqrt14")
        } else {
            ("rcp", "rsqrt")
        }
    } else if bits == 512 {
        ("rcp14", "rsqrt14")
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

    // Extract/cast for reduction helper types
    let extract_hi = match (elem, bits) {
        ("f32", 256) => format!(
            "let hi = {p}_extractf128_ps::<1>(a);\n            let lo = {p}_castps256_ps128(a);"
        ),
        ("f64", 256) => format!(
            "let hi = {p}_extractf128_pd::<1>(a);\n            let lo = {p}_castpd256_pd128(a);"
        ),
        ("f32", 128) | ("f64", 128) => String::new(), // No extraction needed
        _ => String::new(),
    };

    let _ = extract_hi; // Used in reduce_* bodies

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
        impl {trait_name} for archmage::{token} {{
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
                impl crate::simd::backends::F32x{lanes}Internal for archmage::{token} {{
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
        ("f32", 512) => "        { _mm512_reduce_add_ps(a) }".to_string(),
        ("f64", 512) => "        { _mm512_reduce_add_pd(a) }".to_string(),
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
        ("f32", 512) => "        { _mm512_reduce_min_ps(a) }".to_string(),
        ("f64", 512) => "        { _mm512_reduce_min_pd(a) }".to_string(),
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
        ("f32", 512) => "        { _mm512_reduce_max_ps(a) }".to_string(),
        ("f64", 512) => "        { _mm512_reduce_max_pd(a) }".to_string(),
        _ => unreachable!(),
    }
}

// ============================================================================
// Scalar Implementation Generation
// ============================================================================

fn generate_scalar_impls(types: &[FloatVecType]) -> String {
    let mut code = formatdoc! {r#"
        //! Backend implementations for ScalarToken (fallback).
        //!
        //! All operations are plain array math. Always available on all platforms.
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.

        #![allow(clippy::needless_range_loop, clippy::identity_op, clippy::collapsible_else_if, clippy::unnecessary_cast)]

        use crate::simd::backends::*;

    "#};

    // Emit helper functions once per element type (not per vector type).
    // The scalar helpers select hardware where the build guarantees it.
    let mut seen_elems = std::collections::BTreeSet::new();
    for ty in types {
        if seen_elems.insert(ty.elem) {
            let (sqrt_fn, floor_fn, ceil_fn, round_fn) = if ty.elem == "f32" {
                (
                    "crate::scalar_sqrt::sqrtf",
                    "crate::scalar::floor_f32",
                    "crate::scalar::ceil_f32",
                    "crate::scalar::round_f32",
                )
            } else {
                (
                    "crate::scalar_sqrt::sqrt",
                    "crate::scalar::floor_f64",
                    "crate::scalar::ceil_f64",
                    "crate::scalar::round_f64",
                )
            };
            code.push_str(&formatdoc! {r#"
                // Helpers to avoid trait method name shadowing inside the impl block.
                // Inside `impl XxxBackend`, names like `sqrt`, `floor`, etc. resolve to
                // the trait's associated functions instead of {elem}'s inherent methods.
                #[inline(always)]
                fn {elem}_sqrt(x: {elem}) -> {elem} {{
                    {sqrt_fn}(x)
                }}
                #[inline(always)]
                fn {elem}_floor(x: {elem}) -> {elem} {{
                    {floor_fn}(x)
                }}
                #[inline(always)]
                fn {elem}_ceil(x: {elem}) -> {elem} {{
                    {ceil_fn}(x)
                }}
                #[inline(always)]
                fn {elem}_round(x: {elem}) -> {elem} {{
                    {round_fn}(x)
                }}

            "#, elem = ty.elem});
        }
    }

    for ty in types {
        code.push_str(&generate_scalar_float_impl(ty));
        code.push('\n');
    }

    code
}

fn generate_scalar_float_impl(ty: &FloatVecType) -> String {
    let trait_name = ty.trait_name();
    let elem = ty.elem;
    let lanes = ty.lanes;
    let array = ty.array_type();
    let zero_lit = if elem == "f32" { "0.0f32" } else { "0.0f64" };
    let abs_mask = if elem == "f32" {
        "0x7FFF_FFFF"
    } else {
        "0x7FFF_FFFF_FFFF_FFFF"
    };
    let fma_fn = if elem == "f32" {
        "fmaf_lanes"
    } else {
        "fma_lanes"
    };

    // Adjacent-pair tree for f32 — matches x86 v3 and NEON reduce_add
    // shape. Better ILP than linear left-fold and stays consistent across
    // backends (#50). f64 with 2 or 4 lanes has no meaningful difference.
    let reduce_add = match (elem, lanes) {
        ("f32", 4) => "(a[0] + a[1]) + (a[2] + a[3])".to_string(),
        ("f32", 8) => {
            "((a[0] + a[4]) + (a[1] + a[5])) + ((a[2] + a[6]) + (a[3] + a[7]))".to_string()
        }
        _ => (0..lanes)
            .map(|i| format!("a[{i}]"))
            .collect::<Vec<_>>()
            .join(" + "),
    };

    // Helper functions (f32_sqrt etc.) are emitted once per element type
    // at the file level by generate_scalar_impls, not per-type.

    let true_mask = if elem == "f32" {
        "f32::from_bits(0xFFFF_FFFF)"
    } else {
        "f64::from_bits(0xFFFF_FFFF_FFFF_FFFF)"
    };

    let sign_shift = if elem == "f32" { "31" } else { "63" };

    // No hardware reciprocal estimate on the scalar fallback. `rcp_approx` is
    // exact division (measured fastest: a bit-hack estimate is ~1.8x slower per
    // `examples/scalar_reciprocal_bench.rs`). `rsqrt_approx` replaces the very slow
    // scalar sqrt+div with a bit-hack seed + 2 Newton steps (~17-bit) — measured
    // ~11x faster — bit-identical to `rsqrt_approx_portable` then one more
    // `rsqrt_newton_portable`. `recip`/`rsqrt` stay exact; f64 keeps exact division.
    let recip_section = if elem == "f32" {
        formatdoc! {r#"
            #[inline(always)]
            fn rcp_approx(self, a: {array}) -> {array} {{
                core::array::from_fn(|i| 1.0 / a[i])
            }}
            #[inline(always)]
            fn rsqrt_approx(self, a: {array}) -> {array} {{
                core::array::from_fn(|i| {{
                    let x = a[i];
                    let seed = f32::from_bits(0x5f37_59df_u32.wrapping_sub(x.to_bits() >> 1));
                    let hx = 0.5 * x;
                    let y = seed * (1.5 - hx * seed * seed);
                    y * (1.5 - hx * y * y)
                }})
            }}
            #[inline(always)]
            fn recip(self, a: {array}) -> {array} {{
                core::array::from_fn(|i| 1.0 / a[i])
            }}
            #[inline(always)]
            fn rsqrt(self, a: {array}) -> {array} {{
                core::array::from_fn(|i| 1.0 / {elem}_sqrt(a[i]))
            }}
        "#}
    } else {
        formatdoc! {r#"
            #[inline(always)]
            fn rcp_approx(self, a: {array}) -> {array} {{
                core::array::from_fn(|i| 1.0 / a[i])
            }}
            #[inline(always)]
            fn rsqrt_approx(self, a: {array}) -> {array} {{
                core::array::from_fn(|i| 1.0 / {elem}_sqrt(a[i]))
            }}
            #[inline(always)]
            fn recip(self, a: {array}) -> {array} {{
                core::array::from_fn(|i| 1.0 / a[i])
            }}
            #[inline(always)]
            fn rsqrt(self, a: {array}) -> {array} {{
                core::array::from_fn(|i| 1.0 / {elem}_sqrt(a[i]))
            }}
        "#}
    };
    let (min_ieee, max_ieee) = if elem == "f32" {
        ("minimumf", "maximumf")
    } else {
        ("minimum", "maximum")
    };
    let min_max_fast_scalar = formatdoc! {r#"

        #[inline(always)]
        fn minimum(self, a: {array}, b: {array}) -> {array} {{
            core::array::from_fn(|i| crate::scalar_extrema::{min_ieee}(a[i], b[i]))
        }}

        #[inline(always)]
        fn maximum(self, a: {array}, b: {array}) -> {array} {{
            core::array::from_fn(|i| crate::scalar_extrema::{max_ieee}(a[i], b[i]))
        }}

        #[inline(always)]
        fn min_fast(self, a: {array}, b: {array}) -> {array} {{
            core::array::from_fn(|i| if a[i] < b[i] {{ a[i] }} else {{ b[i] }})
        }}

        #[inline(always)]
        fn max_fast(self, a: {array}, b: {array}) -> {array} {{
            core::array::from_fn(|i| if a[i] > b[i] {{ a[i] }} else {{ b[i] }})
        }}
    "#};

    formatdoc! {r#"
        impl {trait_name} for archmage::ScalarToken {{
            type Repr = {array};

            // ====== Construction ======

            #[inline(always)]
            fn splat(self, v: {elem}) -> {array} {{
                [v; {lanes}]
            }}

            #[inline(always)]
            fn zero(self) -> {array} {{
                [{zero_lit}; {lanes}]
            }}

            #[inline(always)]
            fn load(self, data: &{array}) -> {array} {{
                *data
            }}

            #[inline(always)]
            fn from_array(self, arr: {array}) -> {array} {{
                arr
            }}

            #[inline(always)]
            fn store(self, repr: {array}, out: &mut {array}) {{
                *out = repr;
            }}

            #[inline(always)]
            fn to_array(self, repr: {array}) -> {array} {{
                repr
            }}

            // ====== Arithmetic ======

            #[inline(always)]
            fn add(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i] + b[i])
            }}

            #[inline(always)]
            fn sub(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i] - b[i])
            }}

            #[inline(always)]
            fn mul(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i] * b[i])
            }}

            #[inline(always)]
            fn div(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i] / b[i])
            }}

            #[inline(always)]
            fn neg(self, a: {array}) -> {array} {{
                a.map(|x| -x)
            }}

            // ====== Math ======

            #[inline(always)]
            fn min(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i].min(b[i]))
            }}

            #[inline(always)]
            fn max(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i].max(b[i]))
            }}
            {min_max_fast_scalar}

            #[inline(always)]
            fn sqrt(self, a: {array}) -> {array} {{
                a.map({elem}_sqrt)
            }}

            #[inline(always)]
            fn abs(self, a: {array}) -> {array} {{
                a.map(|x| {elem}::from_bits(x.to_bits() & {abs_mask}))
            }}

            #[inline(always)]
            fn floor(self, a: {array}) -> {array} {{
                a.map({elem}_floor)
            }}

            #[inline(always)]
            fn ceil(self, a: {array}) -> {array} {{
                a.map({elem}_ceil)
            }}

            #[inline(always)]
            fn round(self, a: {array}) -> {array} {{
                a.map({elem}_round)
            }}

            #[inline(always)]
            fn mul_add(self, a: {array}, b: {array}, c: {array}) -> {array} {{
                core::array::from_fn(|i| a[i] * b[i] + c[i])
            }}

            #[inline(always)]
            fn mul_sub(self, a: {array}, b: {array}, c: {array}) -> {array} {{
                core::array::from_fn(|i| a[i] * b[i] - c[i])
            }}

            #[inline(always)]
            fn mul_add_portable(self, a: {array}, b: {array}, c: {array}) -> {array} {{
                crate::soft_fma::{fma_fn}(a, b, c)
            }}

            #[inline(always)]
            fn mul_sub_portable(self, a: {array}, b: {array}, c: {array}) -> {array} {{
                crate::soft_fma::{fma_fn}(a, b, c.map(|x| -x))
            }}

            // ====== Comparisons ======

            #[inline(always)]
            fn simd_eq(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] == b[i] {{ {true_mask} }} else {{ 0.0 }})
            }}

            #[inline(always)]
            fn simd_ne(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] != b[i] {{ {true_mask} }} else {{ 0.0 }})
            }}

            #[inline(always)]
            fn simd_lt(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] < b[i] {{ {true_mask} }} else {{ 0.0 }})
            }}

            #[inline(always)]
            fn simd_le(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] <= b[i] {{ {true_mask} }} else {{ 0.0 }})
            }}

            #[inline(always)]
            fn simd_gt(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] > b[i] {{ {true_mask} }} else {{ 0.0 }})
            }}

            #[inline(always)]
            fn simd_ge(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] >= b[i] {{ {true_mask} }} else {{ 0.0 }})
            }}

            #[inline(always)]
            fn blend(self, mask: {array}, if_true: {array}, if_false: {array}) -> {array} {{
                // Check sign bit of mask (all-1s has sign bit set)
                core::array::from_fn(|i| {{
                    if (mask[i].to_bits() >> {sign_shift}) != 0 {{
                        if_true[i]
                    }} else {{
                        if_false[i]
                    }}
                }})
            }}

            // ====== Reductions ======

            #[inline(always)]
            fn reduce_add(self, a: {array}) -> {elem} {{
                {reduce_add}
            }}

            #[inline(always)]
            fn reduce_min(self, a: {array}) -> {elem} {{
                let mut m = a[0];
                for &v in &a[1..] {{
                    m = m.min(v);
                }}
                m
            }}

            #[inline(always)]
            fn reduce_max(self, a: {array}) -> {elem} {{
                let mut m = a[0];
                for &v in &a[1..] {{
                    m = m.max(v);
                }}
                m
            }}

            // ====== Approximations ======
{recip_section}
            // ====== Bitwise ======

            #[inline(always)]
            fn not(self, a: {array}) -> {array} {{
                a.map(|x| {elem}::from_bits(!x.to_bits()))
            }}

            #[inline(always)]
            fn bitand(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| {elem}::from_bits(a[i].to_bits() & b[i].to_bits()))
            }}

            #[inline(always)]
            fn bitor(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| {elem}::from_bits(a[i].to_bits() | b[i].to_bits()))
            }}

            #[inline(always)]
            fn bitxor(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| {elem}::from_bits(a[i].to_bits() ^ b[i].to_bits()))
            }}
        }}{scalar_internal}
    "#,
        scalar_internal = if elem == "f32" {
            super::float_layout::scalar_internal_impl(lanes)
        } else {
            String::new()
        },
    }
}

// ============================================================================
// NEON Implementation Generation
// ============================================================================

fn generate_neon_impls(types: &[FloatVecType]) -> String {
    let safety = impls_safety_contract("NeonToken");
    let mut code = formatdoc! {r#"
        //! Backend implementations for NeonToken (AArch64 NEON).
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.
        //!
        {safety}
        #[cfg(target_arch = "aarch64")]
        use core::arch::aarch64::*;

        #[cfg(target_arch = "aarch64")]
        use archmage::{{attune, NeonToken}};

        use crate::simd::backends::*;

    "#};

    for ty in types {
        if ty.native_on_neon() {
            code.push_str("#[cfg(target_arch = \"aarch64\")]\n");
            code.push_str(&generate_neon_native_impl(ty));
            code.push('\n');
        } else if ty.elem == "f32" && ty.lanes == 8 {
            code.push_str(&generate_neon_f32x8_helpers());
            code.push('\n');
        }
    }

    code
}

/// Native NEON pixel-packing helpers for `f32x8` (`[float32x4_t; 2]`):
/// - `f32x8_to_u8_bytes` (called from `impls/polyfill.rs` `F32x8Backend::to_u8_bytes`)
/// - `F32x8Internal` (5 hand-tuned 24B/32B RGB/RGBA shuffle kernels + 11 delegating hooks)
fn generate_neon_f32x8_helpers() -> String {
    let arcane = super::backend_syntax::arcane("NeonToken");
    let float_layout = super::float_layout::polyfill_f32x8("[float32x4_t; 2]");
    formatdoc! {r#"
        #[cfg(target_arch = "aarch64")]
        #[attune(wrap, suppress_const_test)]
        pub(super) fn f32x8_to_u8_bytes(_token: NeonToken, a: [float32x4_t; 2]) -> [u8; 8] {{
            let i0 = vqmovn_s32(vcvtnq_s32_f32(a[0]));
            let i1 = vqmovn_s32(vcvtnq_s32_f32(a[1]));
            let u8s = vqmovun_s16(vcombine_s16(i0, i1));
            crate::simd_storage::cast(u8s)
        }}

        #[cfg(target_arch = "aarch64")]
        impl crate::simd::backends::F32x8Internal for archmage::NeonToken {{
            type InternalRepr = [float32x4_t; 2];

            {arcane}
            fn store_rgba_u8_into(
                self,
                r: [float32x4_t; 2],
                g: [float32x4_t; 2],
                b: [float32x4_t; 2],
                a: [float32x4_t; 2],
                out: &mut [u8; 32],
            ) {{
                let lo = vdupq_n_s32(0);
                let hi = vdupq_n_s32(255);
                let r0 = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(r[0]), lo), hi));
                let g0 = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(g[0]), lo), hi));
                let b0 = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(b[0]), lo), hi));
                let a0 = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(a[0]), lo), hi));
                let p0 = vorrq_u32(
                    vorrq_u32(r0, vshlq_n_u32::<8>(g0)),
                    vorrq_u32(vshlq_n_u32::<16>(b0), vshlq_n_u32::<24>(a0)),
                );
                let r1 = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(r[1]), lo), hi));
                let g1 = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(g[1]), lo), hi));
                let b1 = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(b[1]), lo), hi));
                let a1 = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(a[1]), lo), hi));
                let p1 = vorrq_u32(
                    vorrq_u32(r1, vshlq_n_u32::<8>(g1)),
                    vorrq_u32(vshlq_n_u32::<16>(b1), vshlq_n_u32::<24>(a1)),
                );
                let b0 = vreinterpretq_u8_u32(p0);
                let b1 = vreinterpretq_u8_u32(p1);
                #[cfg(target_endian = "big")]
                let (b0, b1) = (vrev32q_u8(b0), vrev32q_u8(b1));
                crate::simd_storage::store([b0, b1], out);
            }}

        {float_layout}

            {arcane}
            fn deinterleave_3ch_repr(self, rgb: [[float32x4_t; 2]; 3]) -> [[float32x4_t; 2]; 3] {{
                let idx_r: uint8x16_t = crate::simd_storage::cast([
                    0u8, 1, 2, 3, 12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39,
                ]);
                let idx_g: uint8x16_t = crate::simd_storage::cast([
                    4u8, 5, 6, 7, 16, 17, 18, 19, 28, 29, 30, 31, 40, 41, 42, 43,
                ]);
                let idx_b: uint8x16_t = crate::simd_storage::cast([
                    8u8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35, 44, 45, 46, 47,
                ]);
                let tbl_lo = uint8x16x3_t(
                    vreinterpretq_u8_f32(rgb[0][0]),
                    vreinterpretq_u8_f32(rgb[0][1]),
                    vreinterpretq_u8_f32(rgb[1][0]),
                );
                let tbl_hi = uint8x16x3_t(
                    vreinterpretq_u8_f32(rgb[1][1]),
                    vreinterpretq_u8_f32(rgb[2][0]),
                    vreinterpretq_u8_f32(rgb[2][1]),
                );
                [
                    [
                        vreinterpretq_f32_u8(vqtbl3q_u8(tbl_lo, idx_r)),
                        vreinterpretq_f32_u8(vqtbl3q_u8(tbl_hi, idx_r)),
                    ],
                    [
                        vreinterpretq_f32_u8(vqtbl3q_u8(tbl_lo, idx_g)),
                        vreinterpretq_f32_u8(vqtbl3q_u8(tbl_hi, idx_g)),
                    ],
                    [
                        vreinterpretq_f32_u8(vqtbl3q_u8(tbl_lo, idx_b)),
                        vreinterpretq_f32_u8(vqtbl3q_u8(tbl_hi, idx_b)),
                    ],
                ]
            }}

            {arcane}
            fn interleave_3ch_repr(self, channels: [[float32x4_t; 2]; 3]) -> [[float32x4_t; 2]; 3] {{
                let [r, g, b] = channels;
                let idx_0: uint8x16_t = crate::simd_storage::cast([
                    0u8, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 4, 5, 6, 7,
                ]);
                let idx_1: uint8x16_t = crate::simd_storage::cast([
                    20u8, 21, 22, 23, 36, 37, 38, 39, 8, 9, 10, 11, 24, 25, 26, 27,
                ]);
                let idx_2: uint8x16_t = crate::simd_storage::cast([
                    40u8, 41, 42, 43, 12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47,
                ]);
                let tbl_lo = uint8x16x3_t(
                    vreinterpretq_u8_f32(r[0]),
                    vreinterpretq_u8_f32(g[0]),
                    vreinterpretq_u8_f32(b[0]),
                );
                let tbl_hi = uint8x16x3_t(
                    vreinterpretq_u8_f32(r[1]),
                    vreinterpretq_u8_f32(g[1]),
                    vreinterpretq_u8_f32(b[1]),
                );
                [
                    [
                        vreinterpretq_f32_u8(vqtbl3q_u8(tbl_lo, idx_0)),
                        vreinterpretq_f32_u8(vqtbl3q_u8(tbl_lo, idx_1)),
                    ],
                    [
                        vreinterpretq_f32_u8(vqtbl3q_u8(tbl_lo, idx_2)),
                        vreinterpretq_f32_u8(vqtbl3q_u8(tbl_hi, idx_0)),
                    ],
                    [
                        vreinterpretq_f32_u8(vqtbl3q_u8(tbl_hi, idx_1)),
                        vreinterpretq_f32_u8(vqtbl3q_u8(tbl_hi, idx_2)),
                    ],
                ]
            }}

            {arcane}
            fn load_rgb_bytes(self, rgb: &[u8; 24]) -> ([float32x4_t; 2], [float32x4_t; 2], [float32x4_t; 2]) {{
                let lo16: uint8x16_t = crate::simd_storage::copy(rgb.first_chunk::<16>().unwrap());
                let hi8: uint8x8_t = crate::simd_storage::copy(rgb.last_chunk::<8>().unwrap());
                let tbl = uint8x16x2_t(lo16, vcombine_u8(hi8, vdup_n_u8(0)));
                let rg_idx: uint8x16_t = crate::simd_storage::cast([
                    0u8, 3, 6, 9, 12, 15, 18, 21, 1, 4, 7, 10, 13, 16, 19, 22,
                ]);
                let b_idx: uint8x16_t = crate::simd_storage::cast([
                    2u8, 5, 8, 11, 14, 17, 20, 23, 255, 255, 255, 255, 255, 255, 255, 255,
                ]);
                let rg8 = vqtbl2q_u8(tbl, rg_idx);
                let b8 = vqtbl2q_u8(tbl, b_idx);
                let r16 = vmovl_u8(vget_low_u8(rg8));
                let g16 = vmovl_high_u8(rg8);
                let b16 = vmovl_u8(vget_low_u8(b8));
                (
                    [
                        vcvtq_f32_u32(vmovl_u16(vget_low_u16(r16))),
                        vcvtq_f32_u32(vmovl_high_u16(r16)),
                    ],
                    [
                        vcvtq_f32_u32(vmovl_u16(vget_low_u16(g16))),
                        vcvtq_f32_u32(vmovl_high_u16(g16)),
                    ],
                    [
                        vcvtq_f32_u32(vmovl_u16(vget_low_u16(b16))),
                        vcvtq_f32_u32(vmovl_high_u16(b16)),
                    ],
                )
            }}

            {arcane}
            fn store_rgb_u8_into(
                self,
                r: [float32x4_t; 2],
                g: [float32x4_t; 2],
                b: [float32x4_t; 2],
                out: &mut [u8; 24],
            ) {{
                let r8 = vqmovun_s16(vcombine_s16(
                    vqmovn_s32(vcvtnq_s32_f32(r[0])),
                    vqmovn_s32(vcvtnq_s32_f32(r[1])),
                ));
                let g8 = vqmovun_s16(vcombine_s16(
                    vqmovn_s32(vcvtnq_s32_f32(g[0])),
                    vqmovn_s32(vcvtnq_s32_f32(g[1])),
                ));
                let b8 = vqmovun_s16(vcombine_s16(
                    vqmovn_s32(vcvtnq_s32_f32(b[0])),
                    vqmovn_s32(vcvtnq_s32_f32(b[1])),
                ));
                let tbl = uint8x16x2_t(vcombine_u8(r8, g8), vcombine_u8(b8, b8));
                let idx_lo: uint8x16_t = crate::simd_storage::cast([
                    0u8, 8, 16, 1, 9, 17, 2, 10, 18, 3, 11, 19, 4, 12, 20, 5,
                ]);
                let idx_hi: uint8x16_t = crate::simd_storage::cast([
                    13u8, 21, 6, 14, 22, 7, 15, 23, 255, 255, 255, 255, 255, 255, 255, 255,
                ]);
                let lo16 = vqtbl2q_u8(tbl, idx_lo);
                let hi8 = vqtbl2q_u8(tbl, idx_hi);
                crate::simd_storage::store(lo16, out.first_chunk_mut::<16>().unwrap());
                crate::simd_storage::store(
                    vget_low_u8(hi8),
                    out.last_chunk_mut::<8>().unwrap(),
                );
            }}
        }}
    "#}
}

/// Generate NEON impl for a type that's native 128-bit (f32x4, f64x2).
fn generate_neon_native_impl(ty: &FloatVecType) -> String {
    let arcane = super::backend_syntax::arcane("NeonToken");
    let trait_name = ty.trait_name();
    let repr = ty.neon_repr();
    let elem = ty.elem;
    let lanes = ty.lanes;
    let array = ty.array_type();
    let ns = ty.neon_suffix();
    let eb = if elem == "f32" { 32 } else { 64 };
    let native_lanes = if elem == "f32" { 4 } else { 2 };

    // NEON lacks vmvnq_u64 — pre-compute NOT expression for f64
    let not_inner = if eb == 64 {
        format!("veorq_u64(vreinterpretq_u64_{ns}(a), vdupq_n_u64(u64::MAX))")
    } else {
        format!("vmvnq_u{eb}(vreinterpretq_u{eb}_{ns}(a))")
    };
    let ne_inner = if eb == 64 {
        format!("veorq_u64(vceqq_{ns}(a, b), vdupq_n_u64(u64::MAX))")
    } else {
        format!("vmvnq_u{eb}(vceqq_{ns}(a, b))")
    };

    // Working-tier `recip`/`rsqrt` bodies, chosen per element type:
    //
    // - f32: estimate + one more fused FRECPS/FRSQRTS step than `_approx`
    //   (~2-3 ULP, inside the <= 4 ULP bare-name contract) — the published
    //   0.9.28 lowering, faster than FDIV on Neoverse-class cores (see the
    //   perf note below; on Apple exact division is the faster form, and
    //   Apple callers wanting both speed and 0 ULP use `recip_portable`).
    //   Rails: the refine step computes a*y, so ±0/±inf lanes yield NaN —
    //   documented per-backend behavior at this tier (issue #64's footgun
    //   lives in the 0 ULP tier, `recip_portable`, which divides).
    // - f64: exact FDIV / FDIV+FSQRT. No f64 estimate is worth refining —
    //   division is BOTH more precise and measured faster on Apple (rcp
    //   3.2x, benchmarks/rsqrt_f64_arm_apple-m4-pro_2026-08-02.md).
    let (recip_body, rsqrt_body) = if elem == "f32" {
        (
            format!(
                "let y = <Self as {trait_name}>::rcp_approx(_self, a); vmulq_{ns}(vrecpsq_{ns}(a, y), y)"
            ),
            format!(
                "let y = <Self as {trait_name}>::rsqrt_approx(_self, a); vmulq_{ns}(y, vrsqrtsq_{ns}(a, vmulq_{ns}(y, y)))"
            ),
        )
    } else {
        (
            format!("vdivq_{ns}(vdupq_n_{ns}(1.0), a)"),
            format!("vdivq_{ns}(vdupq_n_{ns}(1.0), vsqrtq_{ns}(a))"),
        )
    };
    let recip_exact = recip_body;
    let rsqrt_exact = rsqrt_body;

    // Measurements are PER ELEMENT TYPE — do not let the f32 numbers leak into
    // the f64 impl or vice versa. They were taken on different cores and differ
    // by more than 2x.
    let recip_perf_note = if elem == "f32" {
        "\
            //   Apple Silicon (M-series): exact is FASTER\n            \
            //     recip 1.35x, rsqrt 1.18x  (commits defbbc2 / 1b36fc7)\n            \
            //   Neoverse-N1 (Ampere Altra): exact is SLOWER\n            \
            //     rcp 1.39x, rsqrt 2.15x\n            \
            //     (benchmarks/rsqrt_arm_neoverse-n1_2026-06-21.md)"
    } else {
        "\
            //   Apple M4 Pro: exact is FASTER\n            \
            //     rcp 3.2x, rsqrt 1.23x\n            \
            //     (benchmarks/rsqrt_f64_arm_apple-m4-pro_2026-08-02.md)\n            \
            //   Neoverse / server ARM: NOT MEASURED for f64. The f32 form is\n            \
            //     slower there, so treat an f64 regression as plausible until\n            \
            //     someone runs `bench_rsqrt_f64` on that class of core."
    };

    // For native types, reduce pattern is different
    let reduce_pairwise = |pairwise: &str| -> String {
        let mut body = format!("{{\n            let pair = {pairwise}(a, a);\n");
        if native_lanes > 2 {
            body.push_str(&format!("            let pair = {pairwise}(pair, pair);\n"));
        }
        body.push_str(&format!("            vgetq_lane_{ns}::<0>(pair)\n"));
        body.push_str("        }");
        body
    };

    // Native pixel pack (f32x4 only): round-to-nearest-even via FCVTNS, then two
    // saturating narrows (i32→i16→u8). Mirrors the x86 cvtps+packs+packus path
    // and the scalar `roundevenf(x).clamp(0,255) as u8` semantics (negatives→0,
    // >255→255, NaN→0). Overrides the scalar trait default, which LLVM does NOT
    // recover to FCVTNS from the libm-style roundeven call.
    let to_u8_arm = if elem == "f32" && lanes == 4 {
        formatdoc! {r#"

            {arcane}
            fn to_u8_bytes(self, a: {repr}) -> [u8; 4] {{
                let i16s = vqmovn_s32(vcvtnq_s32_f32(a));
                let u8s = vqmovun_s16(vcombine_s16(i16s, i16s));
                let bytes: [u8; 8] = crate::simd_storage::cast(u8s);
                [bytes[0], bytes[1], bytes[2], bytes[3]]
            }}
        "#}
    } else {
        String::new()
    };

    // Native pixel interleave (f32x4): round-to-even + clamp [0,255] each plane,
    // then pack R | G<<8 | B<<16 | A<<24 per lane — each clamped byte lands in
    // its RGBA slot, no shuffle needed. Overrides the to_u8_bytes-based default.
    let store_rgba_arm = if elem == "f32" && lanes == 4 {
        formatdoc! {r#"

            {arcane}
            fn store_rgba_u8_into(self, r: {repr}, g: {repr}, b: {repr}, a: {repr}, out: &mut [u8; 16]) {{
                let lo = vdupq_n_s32(0);
                let hi = vdupq_n_s32(255);
                let ri = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(r), lo), hi));
                let gi = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(g), lo), hi));
                let bi = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(b), lo), hi));
                let ai = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(a), lo), hi));
                let pixels = vorrq_u32(
                    vorrq_u32(ri, vshlq_n_u32::<8>(gi)),
                    vorrq_u32(vshlq_n_u32::<16>(bi), vshlq_n_u32::<24>(ai)),
                );
                let bytes = vreinterpretq_u8_u32(pixels);
                #[cfg(target_endian = "big")]
                let bytes = vrev32q_u8(bytes);
                crate::simd_storage::store(bytes, out);
            }}
        "#}
    } else {
        String::new()
    };

    let float_layout = if elem == "f32" {
        super::float_layout::neon(&arcane)
    } else {
        String::new()
    };
    let rgb_3ch_arm = if elem == "f32" && lanes == 4 {
        formatdoc! {r#"

            {arcane}
            fn deinterleave_3ch_repr(self, rgb: [{repr}; 3]) -> [{repr}; 3] {{
                let idx_r: uint8x16_t = crate::simd_storage::cast([
                    0u8, 1, 2, 3, 12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39,
                ]);
                let idx_g: uint8x16_t = crate::simd_storage::cast([
                    4u8, 5, 6, 7, 16, 17, 18, 19, 28, 29, 30, 31, 40, 41, 42, 43,
                ]);
                let idx_b: uint8x16_t = crate::simd_storage::cast([
                    8u8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35, 44, 45, 46, 47,
                ]);
                let tbl = uint8x16x3_t(
                    vreinterpretq_u8_f32(rgb[0]),
                    vreinterpretq_u8_f32(rgb[1]),
                    vreinterpretq_u8_f32(rgb[2]),
                );
                [
                    vreinterpretq_f32_u8(vqtbl3q_u8(tbl, idx_r)),
                    vreinterpretq_f32_u8(vqtbl3q_u8(tbl, idx_g)),
                    vreinterpretq_f32_u8(vqtbl3q_u8(tbl, idx_b)),
                ]
            }}

            {arcane}
            fn interleave_3ch_repr(self, channels: [{repr}; 3]) -> [{repr}; 3] {{
                let idx_0: uint8x16_t = crate::simd_storage::cast([
                    0u8, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 4, 5, 6, 7,
                ]);
                let idx_1: uint8x16_t = crate::simd_storage::cast([
                    20u8, 21, 22, 23, 36, 37, 38, 39, 8, 9, 10, 11, 24, 25, 26, 27,
                ]);
                let idx_2: uint8x16_t = crate::simd_storage::cast([
                    40u8, 41, 42, 43, 12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47,
                ]);
                let tbl = uint8x16x3_t(
                    vreinterpretq_u8_f32(channels[0]),
                    vreinterpretq_u8_f32(channels[1]),
                    vreinterpretq_u8_f32(channels[2]),
                );
                [
                    vreinterpretq_f32_u8(vqtbl3q_u8(tbl, idx_0)),
                    vreinterpretq_f32_u8(vqtbl3q_u8(tbl, idx_1)),
                    vreinterpretq_f32_u8(vqtbl3q_u8(tbl, idx_2)),
                ]
            }}

            {arcane}
            fn load_rgb_bytes(self, rgb: &[u8; 12]) -> ({repr}, {repr}, {repr}) {{
                let lo8: uint8x8_t = crate::simd_storage::copy(rgb.first_chunk::<8>().unwrap());
                let hi4: u32 = crate::simd_storage::copy(rgb.last_chunk::<4>().unwrap());
                let raw = vreinterpretq_u8_u32(vsetq_lane_u32::<2>(
                    hi4,
                    vreinterpretq_u32_u8(vcombine_u8(lo8, vdup_n_u8(0))),
                ));
                // All 12 channel bytes fit in one 16B tbl: [R0..3, G0..3, B0..3, 0..0].
                let idx: uint8x16_t = crate::simd_storage::cast([
                    0u8, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 255, 255, 255, 255,
                ]);
                let rgb8 = vqtbl1q_u8(raw, idx);
                let rg16 = vmovl_u8(vget_low_u8(rgb8));
                let b16 = vmovl_high_u8(rgb8);
                (
                    vcvtq_f32_u32(vmovl_u16(vget_low_u16(rg16))),
                    vcvtq_f32_u32(vmovl_high_u16(rg16)),
                    vcvtq_f32_u32(vmovl_u16(vget_low_u16(b16))),
                )
            }}

            {arcane}
            fn store_rgb_u8_into(self, r: {repr}, g: {repr}, b: {repr}, out: &mut [u8; 12]) {{
                let ri = vqmovn_s32(vcvtnq_s32_f32(r));
                let gi = vqmovn_s32(vcvtnq_s32_f32(g));
                let bi = vqmovn_s32(vcvtnq_s32_f32(b));
                let rg8 = vqmovun_s16(vcombine_s16(ri, gi));
                let b8 = vqmovun_s16(vcombine_s16(bi, bi));
                let planes = vcombine_u8(rg8, b8);
                let idx: uint8x16_t = crate::simd_storage::cast([
                    0u8, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 255, 255, 255, 255,
                ]);
                let packed = vqtbl1q_u8(planes, idx);
                crate::simd_storage::store(
                    vget_low_u8(packed),
                    out.first_chunk_mut::<8>().unwrap(),
                );
                crate::simd_storage::store(
                    vgetq_lane_u32::<2>(vreinterpretq_u32_u8(packed)),
                    out.last_chunk_mut::<4>().unwrap(),
                );
            }}
        "#}
    } else {
        String::new()
    };

    formatdoc! {r#"
        impl {trait_name} for archmage::NeonToken {{
            type Repr = {repr};

            {arcane}
            fn splat(self, v: {elem}) -> {repr} {{
                vdupq_n_{ns}(v)
            }}

            {arcane}
            fn zero(self) -> {repr} {{
                vdupq_n_{ns}(0.0)
            }}

            #[inline(always)]
            fn load(self, data: &{array}) -> {repr} {{
                crate::simd_storage::copy(data)
            }}

            #[inline(always)]
            fn from_array(self, arr: {array}) -> {repr} {{
                crate::simd_storage::cast(arr)
            }}

            #[inline(always)]
            fn store(self, repr: {repr}, out: &mut {array}) {{
                crate::simd_storage::store(repr, out);
            }}

            #[inline(always)]
            fn to_array(self, repr: {repr}) -> {array} {{
                crate::simd_storage::cast(repr)
            }}

            {arcane}
            fn add(self, a: {repr}, b: {repr}) -> {repr} {{ vaddq_{ns}(a, b) }}
            {arcane}
            fn sub(self, a: {repr}, b: {repr}) -> {repr} {{ vsubq_{ns}(a, b) }}
            {arcane}
            fn mul(self, a: {repr}, b: {repr}) -> {repr} {{ vmulq_{ns}(a, b) }}
            {arcane}
            fn div(self, a: {repr}, b: {repr}) -> {repr} {{ vdivq_{ns}(a, b) }}
            {arcane}
            fn neg(self, a: {repr}) -> {repr} {{ vnegq_{ns}(a) }}
            {arcane}
            fn min(self, a: {repr}, b: {repr}) -> {repr} {{ vminnmq_{ns}(a, b) }}
            {arcane}
            fn max(self, a: {repr}, b: {repr}) -> {repr} {{ vmaxnmq_{ns}(a, b) }}
            {arcane}
            fn minimum(self, a: {repr}, b: {repr}) -> {repr} {{ vminq_{ns}(a, b) }}
            {arcane}
            fn maximum(self, a: {repr}, b: {repr}) -> {repr} {{ vmaxq_{ns}(a, b) }}
            {arcane}
            fn min_fast(self, a: {repr}, b: {repr}) -> {repr} {{ vminnmq_{ns}(a, b) }}
            {arcane}
            fn max_fast(self, a: {repr}, b: {repr}) -> {repr} {{ vmaxnmq_{ns}(a, b) }}
            {arcane}
            fn sqrt(self, a: {repr}) -> {repr} {{ vsqrtq_{ns}(a) }}
            {arcane}
            fn abs(self, a: {repr}) -> {repr} {{ vabsq_{ns}(a) }}
            {arcane}
            fn trunc(self, a: {repr}) -> {repr} {{ vrndq_{ns}(a) }}

            {arcane}
            fn mul_neg_add_relaxed(self, a: {repr}, b: {repr}, c: {repr}) -> {repr} {{ vfmsq_{ns}(c, a, b) }}

            {arcane}
            fn mul_neg_add_fused(self, a: {repr}, b: {repr}, c: {repr}) -> {repr} {{ vfmsq_{ns}(c, a, b) }}

            {arcane}
            fn floor(self, a: {repr}) -> {repr} {{ vrndmq_{ns}(a) }}
            {arcane}
            fn ceil(self, a: {repr}) -> {repr} {{ vrndpq_{ns}(a) }}
            {arcane}
            fn round(self, a: {repr}) -> {repr} {{ vrndnq_{ns}(a) }}

            {arcane}
            fn mul_add(self, a: {repr}, b: {repr}, c: {repr}) -> {repr} {{
                vfmaq_{ns}(c, a, b)
            }}

            {arcane}
            fn mul_sub(self, a: {repr}, b: {repr}, c: {repr}) -> {repr} {{
                vfmaq_{ns}(vnegq_{ns}(c), a, b)
            }}

            {arcane}
            fn mul_add_portable(self, a: {repr}, b: {repr}, c: {repr}) -> {repr} {{
                vfmaq_{ns}(c, a, b)
            }}

            {arcane}
            fn mul_sub_portable(self, a: {repr}, b: {repr}, c: {repr}) -> {repr} {{
                vfmaq_{ns}(vnegq_{ns}(c), a, b)
            }}

            {arcane}
            fn simd_eq(self, a: {repr}, b: {repr}) -> {repr} {{
                vreinterpretq_{ns}_u{eb}(vceqq_{ns}(a, b))
            }}
            {arcane}
            fn simd_ne(self, a: {repr}, b: {repr}) -> {repr} {{
                vreinterpretq_{ns}_u{eb}({ne_inner})
            }}
            {arcane}
            fn simd_lt(self, a: {repr}, b: {repr}) -> {repr} {{
                vreinterpretq_{ns}_u{eb}(vcltq_{ns}(a, b))
            }}
            {arcane}
            fn simd_le(self, a: {repr}, b: {repr}) -> {repr} {{
                vreinterpretq_{ns}_u{eb}(vcleq_{ns}(a, b))
            }}
            {arcane}
            fn simd_gt(self, a: {repr}, b: {repr}) -> {repr} {{
                vreinterpretq_{ns}_u{eb}(vcgtq_{ns}(a, b))
            }}
            {arcane}
            fn simd_ge(self, a: {repr}, b: {repr}) -> {repr} {{
                vreinterpretq_{ns}_u{eb}(vcgeq_{ns}(a, b))
            }}

            {arcane}
            fn blend(self, mask: {repr}, if_true: {repr}, if_false: {repr}) -> {repr} {{
                vbslq_{ns}(vreinterpretq_u{eb}_{ns}(mask), if_true, if_false)
            }}

            {arcane}
            fn reduce_add(self, a: {repr}) -> {elem} {{
                {reduce_add}
            }}

            {arcane}
            fn reduce_min(self, a: {repr}) -> {elem} {{
                {reduce_min}
            }}

            {arcane}
            fn reduce_max(self, a: {repr}) -> {elem} {{
                {reduce_max}
            }}

            // ====== Reciprocals: estimate tier vs working tier ======
            //
            // `_approx` = raw vrecpe/vrsqrte (~8-bit) + one fused FRECPS/FRSQRTS
            // step (~16-bit): the documented >=12-bit fast path. FRECPS
            // (`vrecpsq`) computes `2 - a*y` and FRSQRTS (`vrsqrtsq`) computes
            // `(3 - a*y*y)/2` as one fused step each — no intermediate rounding,
            // no 2.0/3.0/0.5 splats, measurably faster than hand-rolled mul/sub
            // on real silicon (see `benchmarks/rsqrt_arm_neoverse-n1`). Each step
            // roughly doubles the correct bits (~8 -> ~16 -> ~24). FRECPS/FRSQRTS
            // are baseline NEON.
            //
            // `recip`/`rsqrt` carry the working-tier contract (<= 4 ULP AND
            // exact IEEE rails at ±0/±inf/NaN), filled by the fastest
            // conforming lowering per element type. f32 adds one more fused
            // step to `_approx` (~2-3 ULP; faster than FDIV on Neoverse-class
            // cores, the published 0.9.28 behavior). Rails come free from the
            // hardware: FRECPS/FRSQRTS special-case inf*0 to 2.0/1.5 BY
            // DESIGN so Newton iterations pass rails through — PROVIDED the
            // dangerous product is formed INSIDE the fused op. recip does
            // this naturally (FRECPS(a, y)); rsqrt is therefore written
            // y*FRSQRTS(a, y*y), NOT y*FRSQRTS(a*y, y): a plain a*y mul is
            // NaN at the rails and was exactly how the pre-fix NaN leaked
            // (probe-verified under qemu, 2026-09-04). Cost: identical
            // instruction count. Tradeoff: y*y overflows f32 for a below
            // ~3e-39, so deep-subnormal inputs are unspecified at this tier.
            // f64 divides (no worthwhile f64 estimate exists, and FDIV is
            // both exact and measured faster on Apple).
            //
            // Cost of exactness is per-core and does NOT generalize — these
            // numbers are for {elem} specifically:
{recip_perf_note}
            {arcane}
            fn rcp_approx(self, a: {repr}) -> {repr} {{
                let y = vrecpeq_{ns}(a); vmulq_{ns}(vrecpsq_{ns}(a, y), y)
            }}
            {arcane}
            fn rsqrt_approx(self, a: {repr}) -> {repr} {{
                let y = vrsqrteq_{ns}(a); vmulq_{ns}(y, vrsqrtsq_{ns}(a, vmulq_{ns}(y, y)))
            }}
            {arcane}
            fn recip(self, a: {repr}) -> {repr} {{
                {recip_exact}
            }}
            {arcane}
            fn rsqrt(self, a: {repr}) -> {repr} {{
                {rsqrt_exact}
            }}

            {arcane}
            fn not(self, a: {repr}) -> {repr} {{
                vreinterpretq_{ns}_u{eb}({not_inner})
            }}
            {arcane}
            fn bitand(self, a: {repr}, b: {repr}) -> {repr} {{
                vreinterpretq_{ns}_u{eb}(vandq_u{eb}(vreinterpretq_u{eb}_{ns}(a), vreinterpretq_u{eb}_{ns}(b)))
            }}
            {arcane}
            fn bitor(self, a: {repr}, b: {repr}) -> {repr} {{
                vreinterpretq_{ns}_u{eb}(vorrq_u{eb}(vreinterpretq_u{eb}_{ns}(a), vreinterpretq_u{eb}_{ns}(b)))
            }}
            {arcane}
            fn bitxor(self, a: {repr}, b: {repr}) -> {repr} {{
                vreinterpretq_{ns}_u{eb}(veorq_u{eb}(vreinterpretq_u{eb}_{ns}(a), vreinterpretq_u{eb}_{ns}(b)))
            }}
            {to_u8_arm}
        }}{internal_impl}
    "#,
        internal_impl = if elem == "f32" {
            formatdoc! {r#"

                #[cfg(target_arch = "aarch64")]
                impl crate::simd::backends::F32x{lanes}Internal for archmage::NeonToken {{
                    type InternalRepr = {repr};
                {store_rgba_arm}
                {float_layout}
                {rgb_3ch_arm}
                }}"#}
        } else {
            String::new()
        },
        reduce_add = reduce_pairwise(&format!("vpaddq_{ns}")),
        reduce_min = if elem == "f32" { String::from("vminnmvq_f32(a)") } else { reduce_pairwise(&format!("vpminq_{ns}")) },
        reduce_max = if elem == "f32" { String::from("vmaxnmvq_f32(a)") } else { reduce_pairwise(&format!("vpmaxq_{ns}")) },
    }
}

// ============================================================================
// WASM Implementation Generation
// ============================================================================

fn generate_wasm_impls(types: &[FloatVecType]) -> String {
    let safety = impls_safety_contract("Wasm128Token");
    let mut code = formatdoc! {r#"
        //! Backend implementations for Wasm128Token (WebAssembly SIMD).
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.
        //!
        {safety}
        #[cfg(target_arch = "wasm32")]
        use core::arch::wasm32::*;

        use crate::simd::backends::*;

    "#};

    for ty in types {
        if ty.native_on_wasm() {
            code.push_str("#[cfg(target_arch = \"wasm32\")]\n");
            code.push_str(&generate_wasm_native_impl(ty));
            code.push('\n');
        } else if ty.elem == "f32" && ty.lanes == 8 {
            code.push_str(&generate_wasm_f32x8_helpers());
            code.push('\n');
        }
    }

    code
}

/// Native WASM128 pixel-packing helpers for `f32x8` (`[v128; 2]`):
/// - `f32x8_to_u8_bytes` (called from `impls/polyfill.rs` `F32x8Backend::to_u8_bytes`)
/// - `F32x8Internal` (5 hand-tuned 24B/32B RGB/RGBA shuffle kernels + 11 delegating hooks)
fn generate_wasm_f32x8_helpers() -> String {
    let float_layout = super::float_layout::polyfill_f32x8("[v128; 2]");
    formatdoc! {r#"
        #[cfg(target_arch = "wasm32")]
        #[inline(always)]
        pub(super) fn f32x8_to_u8_bytes(_token: archmage::Wasm128Token, a: [v128; 2]) -> [u8; 8] {{
            let i0 = i32x4_trunc_sat_f32x4(f32x4_nearest(a[0]));
            let i1 = i32x4_trunc_sat_f32x4(f32x4_nearest(a[1]));
            let i16s = i16x8_narrow_i32x4(i0, i1);
            let u8s = u8x16_narrow_i16x8(i16s, i16s);
            let lo = u32x4_extract_lane::<0>(u8s);
            let hi = u32x4_extract_lane::<1>(u8s);
            ((u64::from(hi) << 32) | u64::from(lo)).to_ne_bytes()
        }}

        #[cfg(target_arch = "wasm32")]
        impl crate::simd::backends::F32x8Internal for archmage::Wasm128Token {{
            type InternalRepr = [v128; 2];
        {float_layout}

            #[inline(always)]
            fn deinterleave_3ch_repr(self, rgb: [[v128; 2]; 3]) -> [[v128; 2]; 3] {{
                let [[v0, v1], [v2, v3], [v4, v5]] = rgb;
                let [r0, g0, b0] = <Self as F32x4Internal>::deinterleave_3ch_repr(self, [v0, v1, v2]);
                let [r1, g1, b1] = <Self as F32x4Internal>::deinterleave_3ch_repr(self, [v3, v4, v5]);
                [[r0, r1], [g0, g1], [b0, b1]]
            }}

            #[inline(always)]
            fn store_rgba_u8_into(
                self,
                r: [v128; 2],
                g: [v128; 2],
                b: [v128; 2],
                a: [v128; 2],
                out: &mut [u8; 32],
            ) {{
                let lo = i32x4_splat(0);
                let hi = i32x4_splat(255);
                let clamp = |v: v128| i32x4_min(i32x4_max(i32x4_trunc_sat_f32x4(f32x4_nearest(v)), lo), hi);
                let pack = |r: v128, g: v128, b: v128, a: v128| {{
                    v128_or(
                        v128_or(clamp(r), i32x4_shl(clamp(g), 8)),
                        v128_or(i32x4_shl(clamp(b), 16), i32x4_shl(clamp(a), 24)),
                    )
                }};
                crate::simd_storage::store([pack(r[0], g[0], b[0], a[0]), pack(r[1], g[1], b[1], a[1])], out);
            }}

            #[inline(always)]
            fn interleave_3ch_repr(self, channels: [[v128; 2]; 3]) -> [[v128; 2]; 3] {{
                let [r, g, b] = channels;
                let int4 = |r: v128, g: v128, b: v128| -> (v128, v128, v128) {{
                    let rg01 = i32x4_shuffle::<0, 4, 1, 5>(r, g);
                    let rg23 = i32x4_shuffle::<2, 6, 3, 7>(r, g);
                    let gb123 = i32x4_shuffle::<1, 5, 6, 7>(g, b);
                    let out0 = i32x4_shuffle::<0, 1, 4, 2>(rg01, b);
                    let out1 = i32x4_shuffle::<0, 1, 4, 5>(gb123, rg23);
                    let out2 = i32x4_shuffle::<2, 6, 7, 3>(gb123, rg23);
                    (out0, out1, out2)
                }};
                let (q0, q1, q2) = int4(r[0], g[0], b[0]);
                let (q3, q4, q5) = int4(r[1], g[1], b[1]);
                [[q0, q1], [q2, q3], [q4, q5]]
            }}

            #[inline(always)]
            fn load_rgb_bytes(self, rgb: &[u8; 24]) -> ([v128; 2], [v128; 2], [v128; 2]) {{
                let lo16: v128 = crate::simd_storage::copy(rgb.first_chunk::<16>().unwrap());
                let hi8 = u64x2(crate::simd_storage::copy(rgb.last_chunk::<8>().unwrap()), 0);
                let rg8 = i8x16_shuffle::<
                    0, 3, 6, 9, 12, 15, 18, 21, 1, 4, 7, 10, 13, 16, 19, 22,
                >(lo16, hi8);
                let b8 = i8x16_shuffle::<
                    2, 5, 8, 11, 14, 17, 20, 23, 24, 24, 24, 24, 24, 24, 24, 24,
                >(lo16, hi8);
                let r16 = u16x8_extend_low_u8x16(rg8);
                let g16 = u16x8_extend_high_u8x16(rg8);
                let b16 = u16x8_extend_low_u8x16(b8);
                (
                    [
                        f32x4_convert_u32x4(u32x4_extend_low_u16x8(r16)),
                        f32x4_convert_u32x4(u32x4_extend_high_u16x8(r16)),
                    ],
                    [
                        f32x4_convert_u32x4(u32x4_extend_low_u16x8(g16)),
                        f32x4_convert_u32x4(u32x4_extend_high_u16x8(g16)),
                    ],
                    [
                        f32x4_convert_u32x4(u32x4_extend_low_u16x8(b16)),
                        f32x4_convert_u32x4(u32x4_extend_high_u16x8(b16)),
                    ],
                )
            }}

            #[inline(always)]
            fn store_rgb_u8_into(self, r: [v128; 2], g: [v128; 2], b: [v128; 2], out: &mut [u8; 24]) {{
                // Narrow each channel to u8x16 individually before combining so LLVM
                // preserves hardware i16x8.narrow_i32x4_s + i8x16.narrow_i16x8_u.
                let narrow8 = |v: [v128; 2]| -> v128 {{
                    let i16s = i16x8_narrow_i32x4(
                        i32x4_trunc_sat_f32x4(f32x4_nearest(v[0])),
                        i32x4_trunc_sat_f32x4(f32x4_nearest(v[1])),
                    );
                    u8x16_narrow_i16x8(i16s, i16s)
                }};
                let r8 = narrow8(r);
                let g8 = narrow8(g);
                let b8 = narrow8(b);
                let rg8 = i64x2_shuffle::<0, 2>(r8, g8);
                let lo16 = i8x16_shuffle::<
                    0, 8, 16, 1, 9, 17, 2, 10, 18, 3, 11, 19, 4, 12, 20, 5,
                >(rg8, b8);
                let hi8 = i8x16_shuffle::<
                    13, 21, 6, 14, 22, 7, 15, 23, 0, 0, 0, 0, 0, 0, 0, 0,
                >(rg8, b8);
                crate::simd_storage::store(lo16, out.first_chunk_mut::<16>().unwrap());
                crate::simd_storage::store(
                    u64x2_extract_lane::<0>(hi8),
                    out.last_chunk_mut::<8>().unwrap(),
                );
            }}
        }}
    "#}
}

/// Generate WASM impl for native 128-bit types (f32x4, f64x2).
#[allow(clippy::too_many_lines)]
fn generate_wasm_native_impl(ty: &FloatVecType) -> String {
    let trait_name = ty.trait_name();
    let elem = ty.elem;
    let lanes = ty.lanes;
    let array = ty.array_type();
    let wp = ty.wasm_prefix();
    let zero_lit = if elem == "f32" { "0.0f32" } else { "0.0f64" };

    // Native pixel pack + 3-channel RGB helpers for wasm f32x4. `f32x4_nearest`
    // is round-half-to-even, matching the scalar default's `roundevenf`;
    // `i32x4_trunc_sat_f32x4` then truncates an already-integral value exactly,
    // and the two saturating narrows clamp into 0..=255.
    let float_layout = if elem == "f32" {
        super::float_layout::wasm()
    } else {
        String::new()
    };
    let to_u8_wasm_native = if elem == "f32" && lanes == 4 {
        formatdoc! {r#"

            #[inline(always)]
            fn to_u8_bytes(self, a: v128) -> [u8; 4] {{
                let i32s = i32x4_trunc_sat_f32x4(f32x4_nearest(a));
                let i16s = i16x8_narrow_i32x4(i32s, i32s);
                let u8s = u8x16_narrow_i16x8(i16s, i16s);
                (u32x4_extract_lane::<0>(u8s)).to_ne_bytes()
            }}
        "#}
    } else {
        String::new()
    };
    let pixel_pack_wasm_native = if elem == "f32" && lanes == 4 {
        formatdoc! {r#"

            #[inline(always)]
            fn deinterleave_3ch_repr(self, rgb: [v128; 3]) -> [v128; 3] {{
                let [v0, v1, v2] = rgb;
                let r = i32x4_shuffle::<0, 3, 6, 0>(v0, v1);
                let g = i32x4_shuffle::<1, 4, 7, 0>(v0, v1);
                let b = i32x4_shuffle::<2, 5, 0, 0>(v0, v1);
                [
                    i32x4_shuffle::<0, 1, 2, 5>(r, v2),
                    i32x4_shuffle::<0, 1, 2, 6>(g, v2),
                    i32x4_shuffle::<0, 1, 4, 7>(b, v2),
                ]
            }}

            #[inline(always)]
            fn store_rgba_u8_into(self, r: v128, g: v128, b: v128, a: v128, out: &mut [u8; 16]) {{
                let lo = i32x4_splat(0);
                let hi = i32x4_splat(255);
                let clamp = |v: v128| i32x4_min(i32x4_max(i32x4_trunc_sat_f32x4(f32x4_nearest(v)), lo), hi);
                let pixels = v128_or(
                    v128_or(clamp(r), i32x4_shl(clamp(g), 8)),
                    v128_or(i32x4_shl(clamp(b), 16), i32x4_shl(clamp(a), 24)),
                );
                crate::simd_storage::store(pixels, out);
            }}

            #[inline(always)]
            fn interleave_3ch_repr(self, channels: [v128; 3]) -> [v128; 3] {{
                let [r, g, b] = channels;
                let rg01 = i32x4_shuffle::<0, 4, 1, 5>(r, g);
                let rg23 = i32x4_shuffle::<2, 6, 3, 7>(r, g);
                let gb123 = i32x4_shuffle::<1, 5, 6, 7>(g, b);
                let out0 = i32x4_shuffle::<0, 1, 4, 2>(rg01, b);
                let out1 = i32x4_shuffle::<0, 1, 4, 5>(gb123, rg23);
                let out2 = i32x4_shuffle::<2, 6, 7, 3>(gb123, rg23);
                [out0, out1, out2]
            }}

            #[inline(always)]
            fn load_rgb_bytes(self, rgb: &[u8; 12]) -> (v128, v128, v128) {{
                let lo8 = u64x2(crate::simd_storage::copy(rgb.first_chunk::<8>().unwrap()), 0);
                let hi4 = u32x4(
                    crate::simd_storage::copy(rgb.last_chunk::<4>().unwrap()),
                    0,
                    0,
                    0,
                );
                // Extract all 12 bytes [R0..3, G0..3, B0..3, 0..0] in one shuffle;
                // lane 8 references lo8's zeroed high half so LLVM emits v128.load64_zero.
                let rgb8 = i8x16_shuffle::<
                    0, 3, 6, 17, 1, 4, 7, 18, 2, 5, 16, 19, 8, 8, 8, 8,
                >(lo8, hi4);
                let rg16 = u16x8_extend_low_u8x16(rgb8);
                let b16 = u16x8_extend_high_u8x16(rgb8);
                (
                    f32x4_convert_u32x4(u32x4_extend_low_u16x8(rg16)),
                    f32x4_convert_u32x4(u32x4_extend_high_u16x8(rg16)),
                    f32x4_convert_u32x4(u32x4_extend_low_u16x8(b16)),
                )
            }}

            #[inline(always)]
            fn store_rgb_u8_into(self, r: v128, g: v128, b: v128, out: &mut [u8; 12]) {{
                let narrow4 = |v: v128| -> v128 {{
                    let i32s = i32x4_trunc_sat_f32x4(f32x4_nearest(v));
                    let i16s = i16x8_narrow_i32x4(i32s, i32s);
                    u8x16_narrow_i16x8(i16s, i16s)
                }};
                let rg8 = i32x4_shuffle::<0, 4, 0, 0>(narrow4(r), narrow4(g));
                let packed = i8x16_shuffle::<
                    0, 4, 16, 1, 5, 17, 2, 6, 18, 3, 7, 19, 0, 0, 0, 0,
                >(rg8, narrow4(b));
                crate::simd_storage::store(
                    u64x2_extract_lane::<0>(packed),
                    out.first_chunk_mut::<8>().unwrap(),
                );
                crate::simd_storage::store(
                    u32x4_extract_lane::<2>(packed),
                    out.last_chunk_mut::<4>().unwrap(),
                );
            }}
        "#}
    } else {
        String::new()
    };

    let min_max_fast_wasm_native = formatdoc! {r#"

        #[inline(always)]
        fn min_fast(self, a: v128, b: v128) -> v128 {{ crate::wasm_fma::min_{wp}_fast(self, a, b) }}
        #[inline(always)]
        fn max_fast(self, a: v128, b: v128) -> v128 {{ crate::wasm_fma::max_{wp}_fast(self, a, b) }}
    "#};

    // Adjacent-pair tree for f32x4 to match x86/NEON shape (#50).
    let reduce_add_body = || -> String {
        if elem == "f32" && lanes == 4 {
            format!(
                "({wp}_extract_lane::<0>(a) + {wp}_extract_lane::<1>(a)) + ({wp}_extract_lane::<2>(a) + {wp}_extract_lane::<3>(a))"
            )
        } else {
            let items: Vec<String> = (0..lanes)
                .map(|j| format!("{wp}_extract_lane::<{j}>(a)"))
                .collect();
            items.join(" + ")
        }
    };

    let reduce_minmax = |fold_method: &str| -> String {
        let extracts: Vec<String> = (0..lanes)
            .map(|j| format!("let v{j} = {wp}_extract_lane::<{j}>(a);"))
            .collect();
        let fold = if lanes == 4 {
            format!("v0.{fold_method}(v1).{fold_method}(v2.{fold_method}(v3))")
        } else {
            format!("v0.{fold_method}(v1)")
        };
        format!("{}\n        {fold}", extracts.join("\n        "))
    };

    // WASM has no hardware reciprocal estimate. For f32 the fast `_approx` is an
    // integer bit-hack seed + one Newton step (~16-bit) — faster than f32x4
    // division and bit-identical to `rcp_approx_portable` (same magic constant
    // and mul/sub order). `recip`/`rsqrt` stay exact (full-precision division).
    // f64 has no cheap estimate worth the divergence, so it keeps exact division
    // for both the approx and full methods.
    let recip_section = if elem == "f32" {
        // `rcp_approx` is exact division: it is the fastest >=12-bit reciprocal on
        // WASM (a single f32x4.div), and a bit-hack estimate is no faster while
        // being less accurate (measured). `rsqrt_approx`, by contrast, replaces the
        // expensive sqrt+div with a bit-hack seed + 2 Newton steps (~17-bit) —
        // ~1.9x faster on WASM (`examples/wasm_reciprocal_bench.rs`). The seed +
        // 2 steps is bit-identical to `rsqrt_approx_portable` then one more
        // `rsqrt_newton_portable` (asserted in the tests). `recip`/`rsqrt` are
        // exact.
        r#"
            #[inline(always)]
            fn rcp_approx(self, a: v128) -> v128 { f32x4_div(f32x4_splat(1.0), a) }
            #[inline(always)]
            fn rsqrt_approx(self, a: v128) -> v128 {
                let onehalf = f32x4_splat(1.5);
                let hx = f32x4_mul(f32x4_splat(0.5), a);
                let seed = i32x4_sub(i32x4_splat(0x5f37_59df_i32), u32x4_shr(a, 1));
                let y = f32x4_mul(seed, f32x4_sub(onehalf, f32x4_mul(f32x4_mul(hx, seed), seed)));
                f32x4_mul(y, f32x4_sub(onehalf, f32x4_mul(f32x4_mul(hx, y), y)))
            }
            #[inline(always)]
            fn recip(self, a: v128) -> v128 { f32x4_div(f32x4_splat(1.0), a) }
            #[inline(always)]
            fn rsqrt(self, a: v128) -> v128 { f32x4_div(f32x4_splat(1.0), f32x4_sqrt(a)) }
"#
        .to_string()
    } else {
        formatdoc! {r#"
            #[inline(always)]
            fn rcp_approx(self, a: v128) -> v128 {{ {wp}_div({wp}_splat(1.0), a) }}
            #[inline(always)]
            fn rsqrt_approx(self, a: v128) -> v128 {{ {wp}_div({wp}_splat(1.0), {wp}_sqrt(a)) }}
            #[inline(always)]
            fn recip(self, a: v128) -> v128 {{ {wp}_div({wp}_splat(1.0), a) }}
            #[inline(always)]
            fn rsqrt(self, a: v128) -> v128 {{ {wp}_div({wp}_splat(1.0), {wp}_sqrt(a)) }}
        "#}
    };
    formatdoc! {r#"
        impl {trait_name} for archmage::Wasm128Token {{
            type Repr = v128;

            #[inline(always)]
            fn splat(self, v: {elem}) -> v128 {{ {wp}_splat(v) }}
            #[inline(always)]
            fn zero(self) -> v128 {{ {wp}_splat(0.0) }}
            #[inline(always)]
            fn load(self, data: &{array}) -> v128 {{ crate::simd_storage::copy(data) }}
            #[inline(always)]
            fn from_array(self, arr: {array}) -> v128 {{ crate::simd_storage::cast(arr) }}
            #[inline(always)]
            fn store(self, repr: v128, out: &mut {array}) {{ crate::simd_storage::store(repr, out); }}
            #[inline(always)]
            fn to_array(self, repr: v128) -> {array} {{
                let mut out = [{zero_lit}; {lanes}];
                crate::simd_storage::store(repr, &mut out);
                out
            }}

            #[inline(always)]
            fn add(self, a: v128, b: v128) -> v128 {{ {wp}_add(a, b) }}
            #[inline(always)]
            fn sub(self, a: v128, b: v128) -> v128 {{ {wp}_sub(a, b) }}
            #[inline(always)]
            fn mul(self, a: v128, b: v128) -> v128 {{ {wp}_mul(a, b) }}
            #[inline(always)]
            fn div(self, a: v128, b: v128) -> v128 {{ {wp}_div(a, b) }}
            #[inline(always)]
            fn neg(self, a: v128) -> v128 {{ {wp}_neg(a) }}
            #[inline(always)]
            fn min(self, a: v128, b: v128) -> v128 {{ v128_bitselect(a, {wp}_pmin(b, a), {wp}_ne(b, b)) }}
            #[inline(always)]
            fn max(self, a: v128, b: v128) -> v128 {{ v128_bitselect(a, {wp}_pmax(b, a), {wp}_ne(b, b)) }}
            #[inline(always)]
            fn minimum(self, a: v128, b: v128) -> v128 {{ {wp}_min(a, b) }}
            #[inline(always)]
            fn maximum(self, a: v128, b: v128) -> v128 {{ {wp}_max(a, b) }}
            {min_max_fast}
            #[inline(always)]
            fn sqrt(self, a: v128) -> v128 {{ {wp}_sqrt(a) }}
            #[inline(always)]
            fn abs(self, a: v128) -> v128 {{ {wp}_abs(a) }}
            #[inline(always)]
            fn trunc(self, a: v128) -> v128 {{ {wp}_trunc(a) }}

            #[inline(always)]
            fn mul_neg_add_relaxed(self, a: v128, b: v128, c: v128) -> v128 {{ crate::wasm_fma::nmadd_{wp}(self, a, b, c) }}

            #[inline(always)]
            fn mul_neg_add_fused(self, a: v128, b: v128, c: v128) -> v128 {{ crate::wasm_fma::fused_{wp}(self, {wp}_neg(a), b, c) }}

            #[inline(always)]
            fn floor(self, a: v128) -> v128 {{ {wp}_floor(a) }}
            #[inline(always)]
            fn ceil(self, a: v128) -> v128 {{ {wp}_ceil(a) }}
            #[inline(always)]
            fn round(self, a: v128) -> v128 {{ {wp}_nearest(a) }}
            #[inline(always)]
            fn mul_add(self, a: v128, b: v128, c: v128) -> v128 {{ crate::wasm_fma::madd_{wp}(self, a, b, c) }}
            #[inline(always)]
            fn mul_sub(self, a: v128, b: v128, c: v128) -> v128 {{ crate::wasm_fma::msub_{wp}(self, a, b, c) }}
            #[inline(always)]
            fn mul_add_portable(self, a: v128, b: v128, c: v128) -> v128 {{ crate::wasm_fma::fused_{wp}(self, a, b, c) }}
            #[inline(always)]
            fn mul_sub_portable(self, a: v128, b: v128, c: v128) -> v128 {{ crate::wasm_fma::fused_{wp}(self, a, b, {wp}_neg(c)) }}
            #[inline(always)]
            fn simd_eq(self, a: v128, b: v128) -> v128 {{ {wp}_eq(a, b) }}
            #[inline(always)]
            fn simd_ne(self, a: v128, b: v128) -> v128 {{ {wp}_ne(a, b) }}
            #[inline(always)]
            fn simd_lt(self, a: v128, b: v128) -> v128 {{ {wp}_lt(a, b) }}
            #[inline(always)]
            fn simd_le(self, a: v128, b: v128) -> v128 {{ {wp}_le(a, b) }}
            #[inline(always)]
            fn simd_gt(self, a: v128, b: v128) -> v128 {{ {wp}_gt(a, b) }}
            #[inline(always)]
            fn simd_ge(self, a: v128, b: v128) -> v128 {{ {wp}_ge(a, b) }}
            #[inline(always)]
            fn blend(self, mask: v128, if_true: v128, if_false: v128) -> v128 {{
                v128_bitselect(if_true, if_false, mask)
            }}

            #[inline(always)]
            fn reduce_add(self, a: v128) -> {elem} {{ {reduce_add} }}
            #[inline(always)]
            fn reduce_min(self, a: v128) -> {elem} {{
                {reduce_min}
            }}
            #[inline(always)]
            fn reduce_max(self, a: v128) -> {elem} {{
                {reduce_max}
            }}

{recip_section}
            #[inline(always)]
            fn not(self, a: v128) -> v128 {{ v128_not(a) }}
            #[inline(always)]
            fn bitand(self, a: v128, b: v128) -> v128 {{ v128_and(a, b) }}
            #[inline(always)]
            fn bitor(self, a: v128, b: v128) -> v128 {{ v128_or(a, b) }}
            #[inline(always)]
            fn bitxor(self, a: v128, b: v128) -> v128 {{ v128_xor(a, b) }}
            {to_u8_wasm_native}
        }}{internal_impl}
    "#,
        internal_impl = if elem == "f32" {
            formatdoc! {r#"

                #[cfg(target_arch = "wasm32")]
                impl crate::simd::backends::F32x{lanes}Internal for archmage::Wasm128Token {{
                    type InternalRepr = v128;
                {float_layout}
                {pixel_pack}
                }}"#,
                pixel_pack = pixel_pack_wasm_native,
            }
        } else {
            String::new()
        },
        reduce_add = reduce_add_body(),
        reduce_min = reduce_minmax("min"),
        reduce_max = reduce_minmax("max"),
        min_max_fast = min_max_fast_wasm_native,
    }
}

// ============================================================================
// I32 Backend Trait Definition Generation
// ============================================================================

fn generate_i32_backend_trait(ty: &I32VecType) -> String {
    let integer_methods = super::backend_gen_widen_narrow::trait_methods(&ty.name());
    let trait_name = ty.trait_name();
    let lanes = ty.lanes;
    let array = ty.array_type();

    formatdoc! {r#"
        //! Backend trait for `{name}<T>` — {lanes}-lane i32 SIMD vector.
        //!
        //! Each token type implements this trait with its platform-native representation.
        //! The generic wrapper `{name}<T>` delegates all operations to these trait methods.
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.

        use super::sealed::Sealed;
        use archmage::SimdToken;

        /// Backend implementation for {lanes}-lane i32 SIMD vectors.
        ///
        /// Trait methods take `self` (the token) as receiver — the token value
        /// is the proof of CPU support, and requiring it as the receiver means
        /// the methods cannot be invoked via UFCS without holding one. The
        /// implementing type `Self` (a token type) determines which platform
        /// intrinsics are used. All methods are `#[inline(always)]` in
        /// implementations.
        ///
        /// # Sealed
        ///
        /// This trait is sealed — only archmage token types can implement it.
        /// The token proves CPU support was verified via `summon()`.
        pub trait {trait_name}: SimdToken + Sealed + Copy + 'static {{
            /// Platform-native SIMD representation.
            #[allow(private_bounds)]
            type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;

            // ====== Construction ======

            /// Broadcast scalar to all {lanes} lanes.
            fn splat(self, v: i32) -> Self::Repr;

            /// All lanes zero.
            fn zero(self) -> Self::Repr;

            /// Load from an aligned array.
            fn load(self, data: &{array}) -> Self::Repr;

            /// Create from array (zero-cost transmute where possible).
            fn from_array(self, arr: {array}) -> Self::Repr;

            /// Store to array.
            fn store(self, repr: Self::Repr, out: &mut {array});

            /// Convert to array.
            fn to_array(self, repr: Self::Repr) -> {array};

            // ====== Arithmetic ======

            /// Lane-wise addition.
            fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise subtraction.
            fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise multiplication (low 32 bits of each 32x32 product).
            fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise negation.
            fn neg(self, a: Self::Repr) -> Self::Repr;

            // ====== Math ======

            /// Lane-wise minimum.
            fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise maximum.
            fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise absolute value.
            fn abs(self, a: Self::Repr) -> Self::Repr;

            // ====== Comparisons ======
            // Return masks where each lane is all-1s (true) or all-0s (false).

            /// Lane-wise equality.
            fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise inequality.
            fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise less-than.
            fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise less-than-or-equal.
            fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise greater-than.
            fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise greater-than-or-equal.
            fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Select lanes: where mask is all-1s pick `if_true`, else `if_false`.
            fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;

            // ====== Reductions ======

            /// Sum all {lanes} lanes.
            fn reduce_add(self, a: Self::Repr) -> i32;

            // ====== Bitwise ======

            /// Bitwise NOT.
            fn not(self, a: Self::Repr) -> Self::Repr;

            /// Bitwise AND.
            fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Bitwise OR.
            fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Bitwise XOR.
            fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            // ====== Shifts ======

            /// Shift left by constant.
            fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;

            /// Arithmetic shift right by constant (sign-extending).
            /// `N` must be in `0..=31`; the generic front-ends reject out-of-range `N` at compile time.
            fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;

            /// Logical shift right by constant (zero-filling).
            /// `N` must be in `0..=31`; the generic front-ends reject out-of-range `N` at compile time.
            fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;

            // ====== Uniform variable shifts ======

            /// Shift left by a runtime `count` applied identically to every lane.
            ///
            /// `count >= 32` produces all-zero lanes on every backend.
            fn shl_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;

            /// Logical (zero-filling) shift right by a runtime `count` applied
            /// identically to every lane.
            ///
            /// `count >= 32` produces all-zero lanes on every backend.
            fn shr_logical_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;

            /// Arithmetic (sign-filling) shift right by a runtime `count`
            /// applied identically to every lane.
            ///
            /// `count >= 32` produces a sign fill (every lane becomes `0` or
            /// `-1`) on every backend.
            fn shr_arithmetic_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;

            // ====== Boolean ======

            /// True if **every** lane has its sign bit set.
            ///
            /// This is a sign-bit test, not a "lane is nonzero" test, on
            /// every backend and at every width. A lane holding `1` is
            /// false; a lane holding `-1` (or `0x80` in its top bit) is
            /// true. Comparison results are all-ones or all-zeros per lane,
            /// so for masks — the intended input — the two readings agree
            /// and this is the cheap native reduction. They diverge only on
            /// hand-built vectors, which is where the backends used to
            /// disagree with each other.
            fn all_true(self, a: Self::Repr) -> bool;

            /// True if **any** lane has its sign bit set.
            ///
            /// Sign-bit test, not "lane is nonzero" — see `all_true`.
            fn any_true(self, a: Self::Repr) -> bool;

            /// Extract the high bit of each 32-bit lane as a bitmask.
            fn bitmask(self, a: Self::Repr) -> u32;

            // ====== Default implementations ======

            /// Clamp values between lo and hi.
            #[inline(always)]
            fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {{
                <Self as {trait_name}>::min(self, <Self as {trait_name}>::max(self, a, lo), hi)
            }}
        {integer_methods}
        }}
    "#,
        name = ty.name(),
    }
}

// ============================================================================
// Conversion Trait Generation
// ============================================================================

fn generate_convert_traits() -> String {
    formatdoc! {r#"
        //! Conversion traits between float and integer SIMD backends.
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.

        use super::sealed::Sealed;
        use super::F32x4Backend;
        use super::F32x8Backend;
        #[cfg(feature = "w512")]
        use super::F32x16Backend;
        use super::F64x2Backend;
        use super::F64x4Backend;
        use super::I32x4Backend;
        use super::I32x8Backend;
        #[cfg(feature = "w512")]
        use super::I32x16Backend;
        use super::I64x2Backend;
        use super::I64x4Backend;
        use super::U32x4Backend;
        use super::U32x8Backend;
        use archmage::SimdToken;

        /// Conversions between f32x4 and i32x4 representations.
        ///
        /// Requires both `F32x4Backend` and `I32x4Backend` to be implemented.
        pub trait F32x4Convert: F32x4Backend + I32x4Backend + SimdToken + Sealed + Copy + 'static {{
            /// Bitcast f32x4 to i32x4 (reinterpret bits, no conversion).
            fn bitcast_f32_to_i32(self, a: <Self as F32x4Backend>::Repr) -> <Self as I32x4Backend>::Repr;

            /// Bitcast i32x4 to f32x4 (reinterpret bits, no conversion).
            fn bitcast_i32_to_f32(self, a: <Self as I32x4Backend>::Repr) -> <Self as F32x4Backend>::Repr;

            /// Convert f32x4 to i32x4 with truncation toward zero.
            fn convert_f32_to_i32(self, a: <Self as F32x4Backend>::Repr) -> <Self as I32x4Backend>::Repr;

            /// Convert with truncation and UNIFORM saturation: out-of-range
            /// lanes clamp to `i32::MIN`/`i32::MAX`, NaN lanes become 0 —
            /// identical on every backend (Rust `as` / NEON FCVTZS / WASM
            /// trunc_sat semantics). Default delegates to
            /// [`convert_f32_to_i32`](Self::convert_f32_to_i32) (already conformant on NEON/WASM/
            /// scalar); x86 overrides with a cvttps + compare/blend fixup
            /// (the bare op yields the `i32::MIN` sentinel for overflow AND
            /// NaN there — issue #80).
            #[inline(always)]
            fn convert_f32_to_i32_saturating(self, a: <Self as F32x4Backend>::Repr) -> <Self as I32x4Backend>::Repr {{
                <Self as F32x4Convert>::convert_f32_to_i32(self, a)
            }}

            /// Fast in-range truncation toward zero (overflow/NaN unspecified; relaxed trunc on WASM `relaxed-simd`).
            #[inline(always)]
            fn convert_f32_to_i32_fast(self, a: <Self as F32x4Backend>::Repr) -> <Self as I32x4Backend>::Repr {{
                <Self as F32x4Convert>::convert_f32_to_i32(self, a)
            }}

            /// Convert f32x4 to i32x4 with rounding to nearest.
            fn convert_f32_to_i32_round(self, a: <Self as F32x4Backend>::Repr) -> <Self as I32x4Backend>::Repr;

            /// Convert i32x4 to f32x4.
            fn convert_i32_to_f32(self, a: <Self as I32x4Backend>::Repr) -> <Self as F32x4Backend>::Repr;
        }}

        /// Conversions between f32x8 and i32x8 representations.
        ///
        /// Requires both `F32x8Backend` and `I32x8Backend` to be implemented.
        pub trait F32x8Convert: F32x8Backend + I32x8Backend + SimdToken + Sealed + Copy + 'static {{
            /// Bitcast f32x8 to i32x8 (reinterpret bits, no conversion).
            fn bitcast_f32_to_i32(self, a: <Self as F32x8Backend>::Repr) -> <Self as I32x8Backend>::Repr;

            /// Bitcast i32x8 to f32x8 (reinterpret bits, no conversion).
            fn bitcast_i32_to_f32(self, a: <Self as I32x8Backend>::Repr) -> <Self as F32x8Backend>::Repr;

            /// Convert f32x8 to i32x8 with truncation toward zero.
            fn convert_f32_to_i32(self, a: <Self as F32x8Backend>::Repr) -> <Self as I32x8Backend>::Repr;

            /// Convert with truncation and UNIFORM saturation: out-of-range
            /// lanes clamp to `i32::MIN`/`i32::MAX`, NaN lanes become 0 —
            /// identical on every backend (Rust `as` / NEON FCVTZS / WASM
            /// trunc_sat semantics). Default delegates to
            /// [`convert_f32_to_i32`](Self::convert_f32_to_i32) (already conformant on NEON/WASM/
            /// scalar); x86 overrides with a cvttps + compare/blend fixup
            /// (the bare op yields the `i32::MIN` sentinel for overflow AND
            /// NaN there — issue #80).
            #[inline(always)]
            fn convert_f32_to_i32_saturating(self, a: <Self as F32x8Backend>::Repr) -> <Self as I32x8Backend>::Repr {{
                <Self as F32x8Convert>::convert_f32_to_i32(self, a)
            }}

            /// Fast in-range truncation toward zero (overflow/NaN unspecified; relaxed trunc on WASM `relaxed-simd`).
            #[inline(always)]
            fn convert_f32_to_i32_fast(self, a: <Self as F32x8Backend>::Repr) -> <Self as I32x8Backend>::Repr {{
                <Self as F32x8Convert>::convert_f32_to_i32(self, a)
            }}

            /// Convert f32x8 to i32x8 with rounding to nearest.
            fn convert_f32_to_i32_round(self, a: <Self as F32x8Backend>::Repr) -> <Self as I32x8Backend>::Repr;

            /// Convert i32x8 to f32x8.
            fn convert_i32_to_f32(self, a: <Self as I32x8Backend>::Repr) -> <Self as F32x8Backend>::Repr;
        }}

        /// Conversions between f32x16 and i32x16 representations.
        ///
        /// Requires both `F32x16Backend` and `I32x16Backend` to be implemented.
        #[cfg(feature = "w512")]
        pub trait F32x16Convert: F32x16Backend + I32x16Backend + SimdToken + Sealed + Copy + 'static {{
            /// Bitcast f32x16 to i32x16 (reinterpret bits, no conversion).
            fn bitcast_f32_to_i32(self, a: <Self as F32x16Backend>::Repr) -> <Self as I32x16Backend>::Repr;

            /// Bitcast i32x16 to f32x16 (reinterpret bits, no conversion).
            fn bitcast_i32_to_f32(self, a: <Self as I32x16Backend>::Repr) -> <Self as F32x16Backend>::Repr;

            /// Convert f32x16 to i32x16 with truncation toward zero.
            fn convert_f32_to_i32(self, a: <Self as F32x16Backend>::Repr) -> <Self as I32x16Backend>::Repr;

            /// Convert with truncation and UNIFORM saturation: out-of-range
            /// lanes clamp to `i32::MIN`/`i32::MAX`, NaN lanes become 0 —
            /// identical on every backend (Rust `as` / NEON FCVTZS / WASM
            /// trunc_sat semantics). Default delegates to
            /// [`convert_f32_to_i32`](Self::convert_f32_to_i32) (already conformant on NEON/WASM/
            /// scalar); x86 overrides with a cvttps + compare/blend fixup
            /// (the bare op yields the `i32::MIN` sentinel for overflow AND
            /// NaN there — issue #80).
            #[inline(always)]
            fn convert_f32_to_i32_saturating(self, a: <Self as F32x16Backend>::Repr) -> <Self as I32x16Backend>::Repr {{
                <Self as F32x16Convert>::convert_f32_to_i32(self, a)
            }}

            /// Fast in-range truncation toward zero (overflow/NaN unspecified; relaxed trunc on WASM `relaxed-simd`).
            #[inline(always)]
            fn convert_f32_to_i32_fast(self, a: <Self as F32x16Backend>::Repr) -> <Self as I32x16Backend>::Repr {{
                <Self as F32x16Convert>::convert_f32_to_i32(self, a)
            }}

            /// Convert f32x16 to i32x16 with rounding to nearest.
            fn convert_f32_to_i32_round(self, a: <Self as F32x16Backend>::Repr) -> <Self as I32x16Backend>::Repr;

            /// Convert i32x16 to f32x16.
            fn convert_i32_to_f32(self, a: <Self as I32x16Backend>::Repr) -> <Self as F32x16Backend>::Repr;
        }}

        /// Bitcast conversions between u32x4 and i32x4 representations.
        ///
        /// Requires both `U32x4Backend` and `I32x4Backend` to be implemented.
        pub trait U32x4Bitcast: U32x4Backend + I32x4Backend + SimdToken + Sealed + Copy + 'static {{
            /// Bitcast u32x4 to i32x4 (reinterpret bits, no conversion).
            fn bitcast_u32_to_i32(self, a: <Self as U32x4Backend>::Repr) -> <Self as I32x4Backend>::Repr;

            /// Bitcast i32x4 to u32x4 (reinterpret bits, no conversion).
            fn bitcast_i32_to_u32(self, a: <Self as I32x4Backend>::Repr) -> <Self as U32x4Backend>::Repr;
        }}

        /// Bitcast conversions between u32x8 and i32x8 representations.
        ///
        /// Requires both `U32x8Backend` and `I32x8Backend` to be implemented.
        pub trait U32x8Bitcast: U32x8Backend + I32x8Backend + SimdToken + Sealed + Copy + 'static {{
            /// Bitcast u32x8 to i32x8 (reinterpret bits, no conversion).
            fn bitcast_u32_to_i32(self, a: <Self as U32x8Backend>::Repr) -> <Self as I32x8Backend>::Repr;

            /// Bitcast i32x8 to u32x8 (reinterpret bits, no conversion).
            fn bitcast_i32_to_u32(self, a: <Self as I32x8Backend>::Repr) -> <Self as U32x8Backend>::Repr;
        }}

        /// Bitcast conversions between i64x2 and f64x2 representations.
        ///
        /// Requires both `I64x2Backend` and `F64x2Backend` to be implemented.
        pub trait I64x2Bitcast: I64x2Backend + F64x2Backend + SimdToken + Sealed + Copy + 'static {{
            /// Bitcast i64x2 to f64x2 (reinterpret bits, no conversion).
            fn bitcast_i64_to_f64(self, a: <Self as I64x2Backend>::Repr) -> <Self as F64x2Backend>::Repr;

            /// Bitcast f64x2 to i64x2 (reinterpret bits, no conversion).
            fn bitcast_f64_to_i64(self, a: <Self as F64x2Backend>::Repr) -> <Self as I64x2Backend>::Repr;
        }}

        /// Bitcast conversions between i64x4 and f64x4 representations.
        ///
        /// Requires both `I64x4Backend` and `F64x4Backend` to be implemented.
        pub trait I64x4Bitcast: I64x4Backend + F64x4Backend + SimdToken + Sealed + Copy + 'static {{
            /// Bitcast i64x4 to f64x4 (reinterpret bits, no conversion).
            fn bitcast_i64_to_f64(self, a: <Self as I64x4Backend>::Repr) -> <Self as F64x4Backend>::Repr;

            /// Bitcast f64x4 to i64x4 (reinterpret bits, no conversion).
            fn bitcast_f64_to_i64(self, a: <Self as F64x4Backend>::Repr) -> <Self as I64x4Backend>::Repr;
        }}
    "#}
}

// ============================================================================
// x86 I32 Implementation Generation
// ============================================================================

fn generate_x86_i32_impls(types: &[I32VecType], token: &str, max_width: usize) -> String {
    let mut code = String::new();

    for ty in types {
        if ty.width_bits > max_width {
            continue;
        }
        code.push_str("\n#[cfg(all(target_arch = \"x86_64\", target_feature = \"sse2\"))]\n");
        code.push_str(&generate_x86_i32_impl(ty, token));
        code.push('\n');
    }

    code
}

fn generate_x86_i32_impl(ty: &I32VecType, token: &str) -> String {
    let integer_methods = super::backend_gen_widen_narrow::methods(&ty.name(), token);
    let arcane = super::backend_syntax::arcane(token);
    let baseline = super::backend_syntax::sse2_or_arcane(token, ty.width_bits);
    let baseline_end = if ty.width_bits == 128 { "}" } else { "" };
    let trait_name = ty.trait_name();
    let inner = ty.x86_inner_type();
    let p = ty.x86_prefix();
    let bits = ty.width_bits;
    let array = ty.array_type();
    let lanes = ty.lanes;

    let reduce_add_body = generate_x86_i32_reduce_add(ty);

    formatdoc! {r#"
        impl {trait_name} for archmage::{token} {{
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

fn generate_x86_convert_impls(token: &str) -> String {
    let arcane = super::backend_syntax::arcane(token);
    formatdoc! {r#"

        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        impl F32x4Convert for archmage::{token} {{
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
        impl F32x8Convert for archmage::{token} {{
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
        impl U32x4Bitcast for archmage::{token} {{
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
        impl U32x8Bitcast for archmage::{token} {{
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
        impl I64x2Bitcast for archmage::{token} {{
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
        impl I64x4Bitcast for archmage::{token} {{
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
// Scalar I32 Implementation Generation
// ============================================================================

fn generate_scalar_i32_impls(types: &[I32VecType]) -> String {
    let mut code = String::new();

    for ty in types {
        code.push_str(&generate_scalar_i32_impl(ty));
        code.push('\n');
    }

    code
}

fn generate_scalar_i32_impl(ty: &I32VecType) -> String {
    let integer_methods = super::backend_gen_widen_narrow::methods(&ty.name(), "ScalarToken");
    let trait_name = ty.trait_name();
    let lanes = ty.lanes;
    let array = ty.array_type();

    let reduce_add = (0..lanes)
        .map(|i| format!("a[{i}]"))
        .collect::<Vec<_>>()
        .join(".wrapping_add(")
        + &")".repeat(lanes - 1);

    let bitmask = (0..lanes)
        .map(|i| {
            if i == 0 {
                "((a[0] as u32) >> 31)".to_string()
            } else {
                format!("(((a[{i}] as u32) >> 31) << {i})")
            }
        })
        .collect::<Vec<_>>()
        .join(" | ");

    // Sign-bit contract: a lane is "true" when its top bit is set, not
    // merely when it is nonzero. Shifting keeps this correct for both the
    // signed lane types (arithmetic shift -> 0 or -1) and the unsigned ones.
    let all_true = (0..lanes)
        .map(|i| format!("(a[{i}] >> 31) != 0"))
        .collect::<Vec<_>>()
        .join(" && ");

    let any_true = (0..lanes)
        .map(|i| format!("(a[{i}] >> 31) != 0"))
        .collect::<Vec<_>>()
        .join(" || ");

    formatdoc! {r#"
        impl {trait_name} for archmage::ScalarToken {{
            type Repr = {array};

            // ====== Construction ======

            #[inline(always)]
            fn splat(self, v: i32) -> {array} {{
                [v; {lanes}]
            }}

            #[inline(always)]
            fn zero(self) -> {array} {{
                [0i32; {lanes}]
            }}

            #[inline(always)]
            fn load(self, data: &{array}) -> {array} {{
                *data
            }}

            #[inline(always)]
            fn from_array(self, arr: {array}) -> {array} {{
                arr
            }}

            #[inline(always)]
            fn store(self, repr: {array}, out: &mut {array}) {{
                *out = repr;
            }}

            #[inline(always)]
            fn to_array(self, repr: {array}) -> {array} {{
                repr
            }}

            // ====== Arithmetic ======

            #[inline(always)]
            fn add(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i].wrapping_add(b[i]))
            }}

            #[inline(always)]
            fn sub(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i].wrapping_sub(b[i]))
            }}

            #[inline(always)]
            fn mul(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i].wrapping_mul(b[i]))
            }}

            #[inline(always)]
            fn neg(self, a: {array}) -> {array} {{
                a.map(i32::wrapping_neg)
            }}

            // ====== Math ======

            #[inline(always)]
            fn min(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] < b[i] {{ a[i] }} else {{ b[i] }})
            }}

            #[inline(always)]
            fn max(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] > b[i] {{ a[i] }} else {{ b[i] }})
            }}

            #[inline(always)]
            fn abs(self, a: {array}) -> {array} {{
                a.map(i32::wrapping_abs)
            }}

            // ====== Comparisons ======

            #[inline(always)]
            fn simd_eq(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] == b[i] {{ -1 }} else {{ 0 }})
            }}

            #[inline(always)]
            fn simd_ne(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] != b[i] {{ -1 }} else {{ 0 }})
            }}

            #[inline(always)]
            fn simd_lt(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] < b[i] {{ -1 }} else {{ 0 }})
            }}

            #[inline(always)]
            fn simd_le(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] <= b[i] {{ -1 }} else {{ 0 }})
            }}

            #[inline(always)]
            fn simd_gt(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] > b[i] {{ -1 }} else {{ 0 }})
            }}

            #[inline(always)]
            fn simd_ge(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] >= b[i] {{ -1 }} else {{ 0 }})
            }}

            #[inline(always)]
            fn blend(self, mask: {array}, if_true: {array}, if_false: {array}) -> {array} {{
                core::array::from_fn(|i| if mask[i] != 0 {{ if_true[i] }} else {{ if_false[i] }})
            }}

            // ====== Reductions ======

            #[inline(always)]
            fn reduce_add(self, a: {array}) -> i32 {{
                {reduce_add}
            }}

            // ====== Bitwise ======

            #[inline(always)]
            fn not(self, a: {array}) -> {array} {{
                a.map(|x| !x)
            }}

            #[inline(always)]
            fn bitand(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i] & b[i])
            }}

            #[inline(always)]
            fn bitor(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i] | b[i])
            }}

            #[inline(always)]
            fn bitxor(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i] ^ b[i])
            }}

            // ====== Shifts ======

            #[inline(always)]
            fn shl_const<const N: i32>(self, a: {array}) -> {array} {{
                a.map(|x| x << N)
            }}

            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(self, a: {array}) -> {array} {{
                a.map(|x| x >> N)
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(self, a: {array}) -> {array} {{
                a.map(|x| ((x as u32) >> N) as i32)
            }}

            // ====== Uniform variable shifts ======

            #[inline(always)]
            fn shl_uniform(self, a: {array}, count: u32) -> {array} {{
                a.map(|x| if count >= 32 {{ 0 }} else {{ x.wrapping_shl(count) }})
            }}

            #[inline(always)]
            fn shr_logical_uniform(self, a: {array}, count: u32) -> {array} {{
                a.map(|x| if count >= 32 {{ 0 }} else {{ ((x as u32).wrapping_shr(count)) as i32 }})
            }}

            #[inline(always)]
            fn shr_arithmetic_uniform(self, a: {array}, count: u32) -> {array} {{
                let c = if count > 31 {{ 31 }} else {{ count }};
                a.map(|x| x.wrapping_shr(c))
            }}

            // ====== Boolean ======

            #[inline(always)]
            fn all_true(self, a: {array}) -> bool {{
                {all_true}
            }}

            #[inline(always)]
            fn any_true(self, a: {array}) -> bool {{
                {any_true}
            }}

            #[inline(always)]
            fn bitmask(self, a: {array}) -> u32 {{
                {bitmask}
            }}
        {integer_methods}
        }}
    "#}
}

// ============================================================================
// Scalar Conversion Implementation Generation
// ============================================================================

fn generate_scalar_convert_impls() -> String {
    let mut code = String::new();

    for lanes in [4, 8, 16] {
        let f_array = format!("[f32; {lanes}]");
        let i_array = format!("[i32; {lanes}]");
        let trait_name = format!("F32x{lanes}Convert");

        // F32x16Convert (lanes=16) is W512 and needs a feature gate.
        let w512_gate = if lanes == 16 {
            "#[cfg(feature = \"w512\")]\n"
        } else {
            ""
        };

        code.push_str(&formatdoc! {r#"
            {w512_gate}impl {trait_name} for archmage::ScalarToken {{
                #[inline(always)]
                fn bitcast_f32_to_i32(self, a: {f_array}) -> {i_array} {{
                    a.map(|x| x.to_bits() as i32)
                }}

                #[inline(always)]
                fn bitcast_i32_to_f32(self, a: {i_array}) -> {f_array} {{
                    a.map(|x| f32::from_bits(x as u32))
                }}

                #[inline(always)]
                fn convert_f32_to_i32(self, a: {f_array}) -> {i_array} {{
                    a.map(|x| x as i32)
                }}

                #[inline(always)]
                fn convert_f32_to_i32_round(self, a: {f_array}) -> {i_array} {{
                    a.map(|x| f32_round(x) as i32)
                }}

                #[inline(always)]
                fn convert_i32_to_f32(self, a: {i_array}) -> {f_array} {{
                    a.map(|x| x as f32)
                }}
            }}

        "#});
    }

    // U32 <-> I32 bitcast impls
    for lanes in [4, 8] {
        let u_array = format!("[u32; {lanes}]");
        let i_array = format!("[i32; {lanes}]");
        let trait_name = format!("U32x{lanes}Bitcast");

        code.push_str(&formatdoc! {r#"
            impl {trait_name} for archmage::ScalarToken {{
                #[inline(always)]
                fn bitcast_u32_to_i32(self, a: {u_array}) -> {i_array} {{
                    a.map(|x| x as i32)
                }}

                #[inline(always)]
                fn bitcast_i32_to_u32(self, a: {i_array}) -> {u_array} {{
                    a.map(|x| x as u32)
                }}
            }}

        "#});
    }

    // I64 <-> F64 bitcast impls
    for lanes in [2, 4] {
        let i_array = format!("[i64; {lanes}]");
        let f_array = format!("[f64; {lanes}]");
        let trait_name = format!("I64x{lanes}Bitcast");

        code.push_str(&formatdoc! {r#"
            impl {trait_name} for archmage::ScalarToken {{
                #[inline(always)]
                fn bitcast_i64_to_f64(self, a: {i_array}) -> {f_array} {{
                    a.map(|x| f64::from_bits(x as u64))
                }}

                #[inline(always)]
                fn bitcast_f64_to_i64(self, a: {f_array}) -> {i_array} {{
                    a.map(|x| x.to_bits() as i64)
                }}
            }}

        "#});
    }

    code
}

// ============================================================================
// NEON I32 Implementation Generation
// ============================================================================

fn generate_neon_i32_impls(types: &[I32VecType]) -> String {
    let mut code = String::new();

    for ty in types {
        if ty.native_on_neon() {
            code.push_str("\n#[cfg(target_arch = \"aarch64\")]\n");
            code.push_str(&generate_neon_native_i32_impl(ty));
            code.push('\n');
        }
    }

    code
}

fn generate_neon_native_i32_impl(ty: &I32VecType) -> String {
    let integer_methods = super::backend_gen_widen_narrow::methods(&ty.name(), "NeonToken");
    let arcane = super::backend_syntax::arcane("NeonToken");
    let trait_name = ty.trait_name();
    let array = ty.array_type();

    formatdoc! {r#"
        impl {trait_name} for archmage::NeonToken {{
            type Repr = int32x4_t;

            {arcane}
            fn splat(self, v: i32) -> int32x4_t {{
                vdupq_n_s32(v)
            }}

            {arcane}
            fn zero(self) -> int32x4_t {{
                vdupq_n_s32(0)
            }}

            #[inline(always)]
            fn load(self, data: &{array}) -> int32x4_t {{
                crate::simd_storage::copy(data)
            }}

            #[inline(always)]
            fn from_array(self, arr: {array}) -> int32x4_t {{
                crate::simd_storage::cast(arr)
            }}

            #[inline(always)]
            fn store(self, repr: int32x4_t, out: &mut {array}) {{
                crate::simd_storage::store(repr, out);
            }}

            #[inline(always)]
            fn to_array(self, repr: int32x4_t) -> {array} {{
                crate::simd_storage::cast(repr)
            }}

            {arcane}
            fn add(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {{ vaddq_s32(a, b) }}
            {arcane}
            fn sub(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {{ vsubq_s32(a, b) }}
            {arcane}
            fn mul(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {{ vmulq_s32(a, b) }}
            {arcane}
            fn neg(self, a: int32x4_t) -> int32x4_t {{ vnegq_s32(a) }}
            {arcane}
            fn min(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {{ vminq_s32(a, b) }}
            {arcane}
            fn max(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {{ vmaxq_s32(a, b) }}
            {arcane}
            fn abs(self, a: int32x4_t) -> int32x4_t {{ vabsq_s32(a) }}

            {arcane}
            fn simd_eq(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {{
                vreinterpretq_s32_u32(vceqq_s32(a, b))
            }}
            {arcane}
            fn simd_ne(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {{
                vreinterpretq_s32_u32(vmvnq_u32(vceqq_s32(a, b)))
            }}
            {arcane}
            fn simd_lt(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {{
                vreinterpretq_s32_u32(vcltq_s32(a, b))
            }}
            {arcane}
            fn simd_le(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {{
                vreinterpretq_s32_u32(vcleq_s32(a, b))
            }}
            {arcane}
            fn simd_gt(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {{
                vreinterpretq_s32_u32(vcgtq_s32(a, b))
            }}
            {arcane}
            fn simd_ge(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {{
                vreinterpretq_s32_u32(vcgeq_s32(a, b))
            }}

            {arcane}
            fn blend(self, mask: int32x4_t, if_true: int32x4_t, if_false: int32x4_t) -> int32x4_t {{
                vbslq_s32(vreinterpretq_u32_s32(mask), if_true, if_false)
            }}

            {arcane}
            fn reduce_add(self, a: int32x4_t) -> i32 {{
                vaddvq_s32(a)
            }}

            {arcane}
            fn not(self, a: int32x4_t) -> int32x4_t {{
                vmvnq_s32(a)
            }}
            {arcane}
            fn bitand(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {{
                vandq_s32(a, b)
            }}
            {arcane}
            fn bitor(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {{
                vorrq_s32(a, b)
            }}
            {arcane}
            fn bitxor(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {{
                veorq_s32(a, b)
            }}

            {arcane}
            fn shl_const<const N: i32>(self, a: int32x4_t) -> int32x4_t {{
                vshlq_n_s32::<N>(a)
            }}

            {arcane}
            fn shr_arithmetic_const<const N: i32>(self, a: int32x4_t) -> int32x4_t {{
                const {{ assert!(N >= 0 && N <= 31) }};
                vshlq_s32(a, vdupq_n_s32(-N))
            }}

            {arcane}
            fn shr_logical_const<const N: i32>(self, a: int32x4_t) -> int32x4_t {{
                const {{ assert!(N >= 0 && N <= 31) }};
                vreinterpretq_s32_u32(vshlq_u32(vreinterpretq_u32_s32(a), vdupq_n_s32(-N)))
            }}

            // ====== Uniform variable shifts ======
            // Same vshlq lowering as the const forms; the count is clamped
            // because USHL/SSHL read only the low byte of each amount lane as
            // a signed value, so an unclamped 256 would wrap to a no-op
            // instead of the contracted zero.

            {arcane}
            fn shl_uniform(self, a: int32x4_t, count: u32) -> int32x4_t {{
                vshlq_s32(a, vdupq_n_s32(count.min(32) as i32))
            }}

            {arcane}
            fn shr_logical_uniform(self, a: int32x4_t, count: u32) -> int32x4_t {{
                vreinterpretq_s32_u32(vshlq_u32(
                    vreinterpretq_u32_s32(a),
                    vdupq_n_s32(-(count.min(32) as i32)),
                ))
            }}

            {arcane}
            fn shr_arithmetic_uniform(self, a: int32x4_t, count: u32) -> int32x4_t {{
                // Clamping to 31 gives the contracted sign fill.
                vshlq_s32(a, vdupq_n_s32(-(count.min(31) as i32)))
            }}

            {arcane}
            fn all_true(self, a: int32x4_t) -> bool {{
                // Every lane's sign bit set <=> every lane negative <=> the
                // signed maximum is negative. Measured 28% faster on M4 Pro
                // than the unsigned-min form this replaces.
                vmaxvq_s32(a) < 0
            }}

            {arcane}
            fn any_true(self, a: int32x4_t) -> bool {{
                vminvq_s32(a) < 0
            }}

            {arcane}
            fn bitmask(self, a: int32x4_t) -> u32 {{
                // One sign bit per word -> 4-bit mask. Shift each 0/1 lane
                // into position, then one horizontal add. The shift on the
                // u32 view must be LOGICAL — an arithmetic s32 shift
                // sign-extends to 0xFFFF_FFFF and corrupts the mask.
                let shifts: [i32; 4] = [0, 1, 2, 3];
                let shift_vec: int32x4_t = crate::simd_storage::copy(&shifts);
                vaddvq_u32(vshlq_u32(vshrq_n_u32::<31>(vreinterpretq_u32_s32(a)), shift_vec))
            }}
        {integer_methods}
        }}
    "#}
}

// ============================================================================
// NEON Conversion Implementation Generation
// ============================================================================

fn generate_neon_convert_impls() -> String {
    let arcane = super::backend_syntax::arcane("NeonToken");
    formatdoc! {r#"

        #[cfg(target_arch = "aarch64")]
        impl F32x4Convert for archmage::NeonToken {{
            {arcane}
            fn bitcast_f32_to_i32(self, a: float32x4_t) -> int32x4_t {{
                vreinterpretq_s32_f32(a)
            }}

            {arcane}
            fn bitcast_i32_to_f32(self, a: int32x4_t) -> float32x4_t {{
                vreinterpretq_f32_s32(a)
            }}

            {arcane}
            fn convert_f32_to_i32(self, a: float32x4_t) -> int32x4_t {{
                vcvtq_s32_f32(a)
            }}

            {arcane}
            fn convert_f32_to_i32_round(self, a: float32x4_t) -> int32x4_t {{
                vcvtnq_s32_f32(a)
            }}

            {arcane}
            fn convert_i32_to_f32(self, a: int32x4_t) -> float32x4_t {{
                vcvtq_f32_s32(a)
            }}
        }}

        #[cfg(target_arch = "aarch64")]
        impl U32x4Bitcast for archmage::NeonToken {{
            {arcane}
            fn bitcast_u32_to_i32(self, a: uint32x4_t) -> int32x4_t {{
                vreinterpretq_s32_u32(a)
            }}

            {arcane}
            fn bitcast_i32_to_u32(self, a: int32x4_t) -> uint32x4_t {{
                vreinterpretq_u32_s32(a)
            }}
        }}

        #[cfg(target_arch = "aarch64")]
        impl I64x2Bitcast for archmage::NeonToken {{
            {arcane}
            fn bitcast_i64_to_f64(self, a: int64x2_t) -> float64x2_t {{
                vreinterpretq_f64_s64(a)
            }}

            {arcane}
            fn bitcast_f64_to_i64(self, a: float64x2_t) -> int64x2_t {{
                vreinterpretq_s64_f64(a)
            }}
        }}
    "#}
}

// ============================================================================
// WASM I32 Implementation Generation
// ============================================================================

fn generate_wasm_i32_impls(types: &[I32VecType]) -> String {
    let mut code = String::new();

    for ty in types {
        if ty.native_on_wasm() {
            code.push_str("\n#[cfg(target_arch = \"wasm32\")]\n");
            code.push_str(&generate_wasm_native_i32_impl(ty));
            code.push('\n');
        }
    }

    code
}

fn generate_wasm_native_i32_impl(ty: &I32VecType) -> String {
    let integer_methods = super::backend_gen_widen_narrow::methods(&ty.name(), "Wasm128Token");
    let trait_name = ty.trait_name();
    let array = ty.array_type();
    let lanes = ty.lanes;

    let reduce_add_body = (0..lanes)
        .map(|j| format!("i32x4_extract_lane::<{j}>(a)"))
        .collect::<Vec<_>>()
        .join(" + ");

    formatdoc! {r#"
        impl {trait_name} for archmage::Wasm128Token {{
            type Repr = v128;

            #[inline(always)]
            fn splat(self, v: i32) -> v128 {{ i32x4_splat(v) }}
            #[inline(always)]
            fn zero(self) -> v128 {{ i32x4_splat(0) }}
            #[inline(always)]
            fn load(self, data: &{array}) -> v128 {{ crate::simd_storage::copy(data) }}
            #[inline(always)]
            fn from_array(self, arr: {array}) -> v128 {{ crate::simd_storage::cast(arr) }}
            #[inline(always)]
            fn store(self, repr: v128, out: &mut {array}) {{ crate::simd_storage::store(repr, out); }}
            #[inline(always)]
            fn to_array(self, repr: v128) -> {array} {{
                let mut out = [0i32; {lanes}];
                crate::simd_storage::store(repr, &mut out);
                out
            }}

            #[inline(always)]
            fn add(self, a: v128, b: v128) -> v128 {{ i32x4_add(a, b) }}
            #[inline(always)]
            fn sub(self, a: v128, b: v128) -> v128 {{ i32x4_sub(a, b) }}
            #[inline(always)]
            fn mul(self, a: v128, b: v128) -> v128 {{ i32x4_mul(a, b) }}
            #[inline(always)]
            fn neg(self, a: v128) -> v128 {{ i32x4_neg(a) }}
            #[inline(always)]
            fn min(self, a: v128, b: v128) -> v128 {{ i32x4_min(a, b) }}
            #[inline(always)]
            fn max(self, a: v128, b: v128) -> v128 {{ i32x4_max(a, b) }}
            #[inline(always)]
            fn abs(self, a: v128) -> v128 {{ i32x4_abs(a) }}

            #[inline(always)]
            fn simd_eq(self, a: v128, b: v128) -> v128 {{ i32x4_eq(a, b) }}
            #[inline(always)]
            fn simd_ne(self, a: v128, b: v128) -> v128 {{ i32x4_ne(a, b) }}
            #[inline(always)]
            fn simd_lt(self, a: v128, b: v128) -> v128 {{ i32x4_lt(a, b) }}
            #[inline(always)]
            fn simd_le(self, a: v128, b: v128) -> v128 {{ i32x4_le(a, b) }}
            #[inline(always)]
            fn simd_gt(self, a: v128, b: v128) -> v128 {{ i32x4_gt(a, b) }}
            #[inline(always)]
            fn simd_ge(self, a: v128, b: v128) -> v128 {{ i32x4_ge(a, b) }}
            #[inline(always)]
            fn blend(self, mask: v128, if_true: v128, if_false: v128) -> v128 {{
                v128_bitselect(if_true, if_false, mask)
            }}

            #[inline(always)]
            fn reduce_add(self, a: v128) -> i32 {{ {reduce_add_body} }}

            #[inline(always)]
            fn not(self, a: v128) -> v128 {{ v128_not(a) }}
            #[inline(always)]
            fn bitand(self, a: v128, b: v128) -> v128 {{ v128_and(a, b) }}
            #[inline(always)]
            fn bitor(self, a: v128, b: v128) -> v128 {{ v128_or(a, b) }}
            #[inline(always)]
            fn bitxor(self, a: v128, b: v128) -> v128 {{ v128_xor(a, b) }}

            #[inline(always)]
            fn shl_const<const N: i32>(self, a: v128) -> v128 {{ i32x4_shl(a, N as u32) }}
            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(self, a: v128) -> v128 {{ i32x4_shr(a, N as u32) }}
            #[inline(always)]
            fn shr_logical_const<const N: i32>(self, a: v128) -> v128 {{ u32x4_shr(a, N as u32) }}

            // ====== Uniform variable shifts ======
            // wasm takes the count modulo the lane width; the keep mask
            // restores the contracted zero at count >= 32, and clamping
            // gives the arithmetic sign fill.

            #[inline(always)]
            fn shl_uniform(self, a: v128, count: u32) -> v128 {{
                let keep: i32 = if count < 32 {{ -1 }} else {{ 0 }};
                v128_and(i32x4_shl(a, count), i32x4_splat(keep))
            }}

            #[inline(always)]
            fn shr_logical_uniform(self, a: v128, count: u32) -> v128 {{
                let keep: i32 = if count < 32 {{ -1 }} else {{ 0 }};
                v128_and(u32x4_shr(a, count), i32x4_splat(keep))
            }}

            #[inline(always)]
            fn shr_arithmetic_uniform(self, a: v128, count: u32) -> v128 {{
                i32x4_shr(a, if count > 31 {{ 31 }} else {{ count }})
            }}

            #[inline(always)]
            fn all_true(self, a: v128) -> bool {{ i32x4_bitmask(a) == 0x0F }}
            #[inline(always)]
            fn any_true(self, a: v128) -> bool {{ i32x4_bitmask(a) != 0 }}
            #[inline(always)]
            fn bitmask(self, a: v128) -> u32 {{ i32x4_bitmask(a) as u32 }}
        {integer_methods}
        }}
    "#}
}

// ============================================================================
// WASM Conversion Implementation Generation
// ============================================================================

fn generate_wasm_convert_impls() -> String {
    formatdoc! {r#"

        #[cfg(target_arch = "wasm32")]
        impl F32x4Convert for archmage::Wasm128Token {{
            #[inline(always)]
            fn bitcast_f32_to_i32(self, a: v128) -> v128 {{ a }}

            #[inline(always)]
            fn bitcast_i32_to_f32(self, a: v128) -> v128 {{ a }}

            #[inline(always)]
            fn convert_f32_to_i32(self, a: v128) -> v128 {{
                i32x4_trunc_sat_f32x4(a)
            }}

            #[inline(always)]
            fn convert_f32_to_i32_fast(self, a: v128) -> v128 {{
                crate::wasm_fma::trunc_f32x4_fast(self, a)
            }}

            #[inline(always)]
            fn convert_f32_to_i32_round(self, a: v128) -> v128 {{
                i32x4_trunc_sat_f32x4(f32x4_nearest(a))
            }}

            #[inline(always)]
            fn convert_i32_to_f32(self, a: v128) -> v128 {{
                f32x4_convert_i32x4(a)
            }}
        }}

        #[cfg(target_arch = "wasm32")]
        impl U32x4Bitcast for archmage::Wasm128Token {{
            #[inline(always)]
            fn bitcast_u32_to_i32(self, a: v128) -> v128 {{ a }}

            #[inline(always)]
            fn bitcast_i32_to_u32(self, a: v128) -> v128 {{ a }}
        }}

        #[cfg(target_arch = "wasm32")]
        impl I64x2Bitcast for archmage::Wasm128Token {{
            #[inline(always)]
            fn bitcast_i64_to_f64(self, a: v128) -> v128 {{ a }}

            #[inline(always)]
            fn bitcast_f64_to_i64(self, a: v128) -> v128 {{ a }}
        }}
    "#}
}

// ============================================================================
// U32 Backend Trait Generation
// ============================================================================

fn generate_u32_backend_trait(ty: &U32VecType) -> String {
    let trait_name = ty.trait_name();
    let lanes = ty.lanes;
    let array = ty.array_type();

    formatdoc! {r#"
        //! Backend trait for `{name}<T>` — {lanes}-lane u32 SIMD vector.
        //!
        //! Each token type implements this trait with its platform-native representation.
        //! The generic wrapper `{name}<T>` delegates all operations to these trait methods.
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.

        use super::sealed::Sealed;
        use archmage::SimdToken;

        /// Backend implementation for {lanes}-lane u32 SIMD vectors.
        ///
        /// Trait methods take `self` (the token) as receiver — the token value
        /// is the proof of CPU support, and requiring it as the receiver means
        /// the methods cannot be invoked via UFCS without holding one. The
        /// implementing type `Self` (a token type) determines which platform
        /// intrinsics are used. All methods are `#[inline(always)]` in
        /// implementations.
        ///
        /// # Sealed
        ///
        /// This trait is sealed — only archmage token types can implement it.
        /// The token proves CPU support was verified via `summon()`.
        pub trait {trait_name}: SimdToken + Sealed + Copy + 'static {{
            /// Platform-native SIMD representation.
            #[allow(private_bounds)]
            type Repr: Copy + Clone + Send + Sync + crate::simd_storage::Pod;

            // ====== Construction ======

            /// Broadcast scalar to all {lanes} lanes.
            fn splat(self, v: u32) -> Self::Repr;

            /// All lanes zero.
            fn zero(self) -> Self::Repr;

            /// Load from an aligned array.
            fn load(self, data: &{array}) -> Self::Repr;

            /// Create from array (zero-cost transmute where possible).
            fn from_array(self, arr: {array}) -> Self::Repr;

            /// Store to array.
            fn store(self, repr: Self::Repr, out: &mut {array});

            /// Convert to array.
            fn to_array(self, repr: Self::Repr) -> {array};

            // ====== Arithmetic ======

            /// Lane-wise addition (wrapping).
            fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise subtraction (wrapping).
            fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise multiplication (low 32 bits of each 32x32 product).
            fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            // ====== Math ======

            /// Lane-wise unsigned minimum.
            fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise unsigned maximum.
            fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            // ====== Comparisons ======
            // Return masks where each lane is all-1s (true) or all-0s (false).
            // All comparisons are unsigned.

            /// Lane-wise equality.
            fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise inequality.
            fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise unsigned less-than.
            fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise unsigned less-than-or-equal.
            fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise unsigned greater-than.
            fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise unsigned greater-than-or-equal.
            fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Select lanes: where mask is all-1s pick `if_true`, else `if_false`.
            fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;

            // ====== Reductions ======

            /// Sum all {lanes} lanes (wrapping).
            fn reduce_add(self, a: Self::Repr) -> u32;

            // ====== Bitwise ======

            /// Bitwise NOT.
            fn not(self, a: Self::Repr) -> Self::Repr;

            /// Bitwise AND.
            fn bitand(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Bitwise OR.
            fn bitor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Bitwise XOR.
            fn bitxor(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            // ====== Shifts ======

            /// Shift left by constant.
            fn shl_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;

            /// Logical shift right by constant (zero-filling).
            /// `N` must be in `0..=31`; the generic front-ends reject out-of-range `N` at compile time.
            fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;

            // ====== Uniform variable shifts ======

            /// Shift left by a runtime `count` applied identically to every lane.
            ///
            /// `count >= 32` produces all-zero lanes on every backend.
            fn shl_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;

            /// Logical (zero-filling) shift right by a runtime `count` applied
            /// identically to every lane.
            ///
            /// `count >= 32` produces all-zero lanes on every backend.
            fn shr_logical_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;

            // ====== Boolean ======

            /// True if **every** lane has its sign bit set.
            ///
            /// This is a sign-bit test, not a "lane is nonzero" test, on
            /// every backend and at every width. A lane holding `1` is
            /// false; a lane holding `-1` (or `0x80` in its top bit) is
            /// true. Comparison results are all-ones or all-zeros per lane,
            /// so for masks — the intended input — the two readings agree
            /// and this is the cheap native reduction. They diverge only on
            /// hand-built vectors, which is where the backends used to
            /// disagree with each other.
            fn all_true(self, a: Self::Repr) -> bool;

            /// True if **any** lane has its sign bit set.
            ///
            /// Sign-bit test, not "lane is nonzero" — see `all_true`.
            fn any_true(self, a: Self::Repr) -> bool;

            /// Extract the high bit of each 32-bit lane as a bitmask.
            fn bitmask(self, a: Self::Repr) -> u32;

            // ====== Default implementations ======

            /// Clamp values between lo and hi (unsigned comparison).
            #[inline(always)]
            fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {{
                <Self as {trait_name}>::min(self, <Self as {trait_name}>::max(self, a, lo), hi)
            }}
        }}
    "#,
        name = ty.name(),
    }
}

// ============================================================================
// x86 U32 Implementation Generation
// ============================================================================

fn generate_x86_u32_impls(types: &[U32VecType], token: &str, max_width: usize) -> String {
    let mut code = String::new();

    for ty in types {
        if ty.width_bits > max_width {
            continue;
        }
        code.push_str("\n#[cfg(all(target_arch = \"x86_64\", target_feature = \"sse2\"))]\n");
        code.push_str(&generate_x86_u32_impl(ty, token));
        code.push('\n');
    }

    code
}

fn generate_x86_u32_impl(ty: &U32VecType, token: &str) -> String {
    let arcane = super::backend_syntax::arcane(token);
    let baseline = super::backend_syntax::sse2_or_arcane(token, ty.width_bits);
    let baseline_end = if ty.width_bits == 128 { "}" } else { "" };
    let trait_name = ty.trait_name();
    let inner = ty.x86_inner_type();
    let p = ty.x86_prefix();
    let bits = ty.width_bits;
    let array = ty.array_type();
    let lanes = ty.lanes;

    let reduce_add_body = generate_x86_u32_reduce_add(ty);

    formatdoc! {r#"
        impl {trait_name} for archmage::{token} {{
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

// ============================================================================
// Scalar U32 Implementation Generation
// ============================================================================

fn generate_scalar_u32_impls(types: &[U32VecType]) -> String {
    let mut code = String::new();

    for ty in types {
        code.push_str(&generate_scalar_u32_impl(ty));
        code.push('\n');
    }

    code
}

fn generate_scalar_u32_impl(ty: &U32VecType) -> String {
    let trait_name = ty.trait_name();
    let lanes = ty.lanes;
    let array = ty.array_type();

    let reduce_add = (0..lanes)
        .map(|i| format!("a[{i}]"))
        .collect::<Vec<_>>()
        .join(".wrapping_add(")
        + &")".repeat(lanes - 1);

    let bitmask = (0..lanes)
        .map(|i| {
            if i == 0 {
                "(a[0] >> 31)".to_string()
            } else {
                format!("((a[{i}] >> 31) << {i})")
            }
        })
        .collect::<Vec<_>>()
        .join(" | ");

    // Sign-bit contract: a lane is "true" when its top bit is set, not
    // merely when it is nonzero. Shifting keeps this correct for both the
    // signed lane types (arithmetic shift -> 0 or -1) and the unsigned ones.
    let all_true = (0..lanes)
        .map(|i| format!("(a[{i}] >> 31) != 0"))
        .collect::<Vec<_>>()
        .join(" && ");

    let any_true = (0..lanes)
        .map(|i| format!("(a[{i}] >> 31) != 0"))
        .collect::<Vec<_>>()
        .join(" || ");

    formatdoc! {r#"
        impl {trait_name} for archmage::ScalarToken {{
            type Repr = {array};

            // ====== Construction ======

            #[inline(always)]
            fn splat(self, v: u32) -> {array} {{
                [v; {lanes}]
            }}

            #[inline(always)]
            fn zero(self) -> {array} {{
                [0u32; {lanes}]
            }}

            #[inline(always)]
            fn load(self, data: &{array}) -> {array} {{
                *data
            }}

            #[inline(always)]
            fn from_array(self, arr: {array}) -> {array} {{
                arr
            }}

            #[inline(always)]
            fn store(self, repr: {array}, out: &mut {array}) {{
                *out = repr;
            }}

            #[inline(always)]
            fn to_array(self, repr: {array}) -> {array} {{
                repr
            }}

            // ====== Arithmetic ======

            #[inline(always)]
            fn add(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i].wrapping_add(b[i]))
            }}

            #[inline(always)]
            fn sub(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i].wrapping_sub(b[i]))
            }}

            #[inline(always)]
            fn mul(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i].wrapping_mul(b[i]))
            }}

            // ====== Math ======

            #[inline(always)]
            fn min(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] < b[i] {{ a[i] }} else {{ b[i] }})
            }}

            #[inline(always)]
            fn max(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] > b[i] {{ a[i] }} else {{ b[i] }})
            }}

            // ====== Comparisons ======

            #[inline(always)]
            fn simd_eq(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] == b[i] {{ u32::MAX }} else {{ 0 }})
            }}

            #[inline(always)]
            fn simd_ne(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] != b[i] {{ u32::MAX }} else {{ 0 }})
            }}

            #[inline(always)]
            fn simd_lt(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] < b[i] {{ u32::MAX }} else {{ 0 }})
            }}

            #[inline(always)]
            fn simd_le(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] <= b[i] {{ u32::MAX }} else {{ 0 }})
            }}

            #[inline(always)]
            fn simd_gt(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] > b[i] {{ u32::MAX }} else {{ 0 }})
            }}

            #[inline(always)]
            fn simd_ge(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| if a[i] >= b[i] {{ u32::MAX }} else {{ 0 }})
            }}

            #[inline(always)]
            fn blend(self, mask: {array}, if_true: {array}, if_false: {array}) -> {array} {{
                core::array::from_fn(|i| if mask[i] != 0 {{ if_true[i] }} else {{ if_false[i] }})
            }}

            // ====== Reductions ======

            #[inline(always)]
            fn reduce_add(self, a: {array}) -> u32 {{
                {reduce_add}
            }}

            // ====== Bitwise ======

            #[inline(always)]
            fn not(self, a: {array}) -> {array} {{
                a.map(|x| !x)
            }}

            #[inline(always)]
            fn bitand(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i] & b[i])
            }}

            #[inline(always)]
            fn bitor(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i] | b[i])
            }}

            #[inline(always)]
            fn bitxor(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i] ^ b[i])
            }}

            // ====== Shifts ======

            #[inline(always)]
            fn shl_const<const N: i32>(self, a: {array}) -> {array} {{
                a.map(|x| x << N)
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(self, a: {array}) -> {array} {{
                a.map(|x| x >> N)
            }}

            // ====== Uniform variable shifts ======

            #[inline(always)]
            fn shl_uniform(self, a: {array}, count: u32) -> {array} {{
                a.map(|x| if count >= 32 {{ 0 }} else {{ x.wrapping_shl(count) }})
            }}

            #[inline(always)]
            fn shr_logical_uniform(self, a: {array}, count: u32) -> {array} {{
                a.map(|x| if count >= 32 {{ 0 }} else {{ x.wrapping_shr(count) }})
            }}

            // ====== Boolean ======

            #[inline(always)]
            fn all_true(self, a: {array}) -> bool {{
                {all_true}
            }}

            #[inline(always)]
            fn any_true(self, a: {array}) -> bool {{
                {any_true}
            }}

            #[inline(always)]
            fn bitmask(self, a: {array}) -> u32 {{
                {bitmask}
            }}
        }}
    "#}
}

// ============================================================================
// NEON U32 Implementation Generation
// ============================================================================

fn generate_neon_u32_impls(types: &[U32VecType]) -> String {
    let mut code = String::new();

    for ty in types {
        if ty.native_on_neon() {
            code.push_str("\n#[cfg(target_arch = \"aarch64\")]\n");
            code.push_str(&generate_neon_native_u32_impl(ty));
            code.push('\n');
        }
    }

    code
}

fn generate_neon_native_u32_impl(ty: &U32VecType) -> String {
    let arcane = super::backend_syntax::arcane("NeonToken");
    let trait_name = ty.trait_name();
    let array = ty.array_type();

    formatdoc! {r#"
        impl {trait_name} for archmage::NeonToken {{
            type Repr = uint32x4_t;

            {arcane}
            fn splat(self, v: u32) -> uint32x4_t {{
                vdupq_n_u32(v)
            }}

            {arcane}
            fn zero(self) -> uint32x4_t {{
                vdupq_n_u32(0)
            }}

            #[inline(always)]
            fn load(self, data: &{array}) -> uint32x4_t {{
                crate::simd_storage::copy(data)
            }}

            #[inline(always)]
            fn from_array(self, arr: {array}) -> uint32x4_t {{
                crate::simd_storage::cast(arr)
            }}

            #[inline(always)]
            fn store(self, repr: uint32x4_t, out: &mut {array}) {{
                crate::simd_storage::store(repr, out);
            }}

            #[inline(always)]
            fn to_array(self, repr: uint32x4_t) -> {array} {{
                crate::simd_storage::cast(repr)
            }}

            {arcane}
            fn add(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{ vaddq_u32(a, b) }}
            {arcane}
            fn sub(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{ vsubq_u32(a, b) }}
            {arcane}
            fn mul(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{ vmulq_u32(a, b) }}
            {arcane}
            fn min(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{ vminq_u32(a, b) }}
            {arcane}
            fn max(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{ vmaxq_u32(a, b) }}

            {arcane}
            fn simd_eq(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{
                vceqq_u32(a, b)
            }}
            {arcane}
            fn simd_ne(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{
                vmvnq_u32(vceqq_u32(a, b))
            }}
            {arcane}
            fn simd_lt(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{
                vcltq_u32(a, b)
            }}
            {arcane}
            fn simd_le(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{
                vcleq_u32(a, b)
            }}
            {arcane}
            fn simd_gt(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{
                vcgtq_u32(a, b)
            }}
            {arcane}
            fn simd_ge(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{
                vcgeq_u32(a, b)
            }}

            {arcane}
            fn blend(self, mask: uint32x4_t, if_true: uint32x4_t, if_false: uint32x4_t) -> uint32x4_t {{
                vbslq_u32(mask, if_true, if_false)
            }}

            {arcane}
            fn reduce_add(self, a: uint32x4_t) -> u32 {{
                vaddvq_u32(a)
            }}

            {arcane}
            fn not(self, a: uint32x4_t) -> uint32x4_t {{
                vmvnq_u32(a)
            }}
            {arcane}
            fn bitand(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{
                vandq_u32(a, b)
            }}
            {arcane}
            fn bitor(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{
                vorrq_u32(a, b)
            }}
            {arcane}
            fn bitxor(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{
                veorq_u32(a, b)
            }}

            {arcane}
            fn shl_const<const N: i32>(self, a: uint32x4_t) -> uint32x4_t {{
                vshlq_n_u32::<N>(a)
            }}

            {arcane}
            fn shr_logical_const<const N: i32>(self, a: uint32x4_t) -> uint32x4_t {{
                const {{ assert!(N >= 0 && N <= 31) }};
                vshlq_u32(a, vdupq_n_s32(-N))
            }}

            // ====== Uniform variable shifts ======
            // Same vshlq lowering as the const forms; the count is clamped
            // because USHL reads only the low byte of each amount lane as a
            // signed value, so an unclamped 256 would wrap to a no-op instead
            // of the contracted zero.

            {arcane}
            fn shl_uniform(self, a: uint32x4_t, count: u32) -> uint32x4_t {{
                vshlq_u32(a, vdupq_n_s32(count.min(32) as i32))
            }}

            {arcane}
            fn shr_logical_uniform(self, a: uint32x4_t, count: u32) -> uint32x4_t {{
                vshlq_u32(a, vdupq_n_s32(-(count.min(32) as i32)))
            }}

            {arcane}
            fn all_true(self, a: uint32x4_t) -> bool {{
                // Sign-bit contract on the signed reinterpretation. This lane
                // type previously used a third rule (all lanes == u32::MAX).
                vmaxvq_s32(vreinterpretq_s32_u32(a)) < 0
            }}

            {arcane}
            fn any_true(self, a: uint32x4_t) -> bool {{
                vminvq_s32(vreinterpretq_s32_u32(a)) < 0
            }}

            {arcane}
            fn bitmask(self, a: uint32x4_t) -> u32 {{
                // One sign bit per word -> 4-bit mask. Shift each 0/1 lane
                // into position, then one horizontal add. The shift on the
                // u32 view must be LOGICAL — an arithmetic s32 shift
                // sign-extends to 0xFFFF_FFFF and corrupts the mask.
                let shifts: [i32; 4] = [0, 1, 2, 3];
                let shift_vec: int32x4_t = crate::simd_storage::copy(&shifts);
                vaddvq_u32(vshlq_u32(vshrq_n_u32::<31>(a), shift_vec))
            }}
        }}
    "#}
}

// ============================================================================
// WASM U32 Implementation Generation
// ============================================================================

fn generate_wasm_u32_impls(types: &[U32VecType]) -> String {
    let mut code = String::new();

    for ty in types {
        if ty.native_on_wasm() {
            code.push_str("\n#[cfg(target_arch = \"wasm32\")]\n");
            code.push_str(&generate_wasm_native_u32_impl(ty));
            code.push('\n');
        }
    }

    code
}

fn generate_wasm_native_u32_impl(ty: &U32VecType) -> String {
    let trait_name = ty.trait_name();
    let array = ty.array_type();
    let lanes = ty.lanes;

    // Only method receivers need the cast parenthesized; the innermost
    // argument does not (clippy::double_parens, unused_parens).
    let reduce_add_body = (0..lanes)
        .map(|j| {
            if j + 1 == lanes {
                format!("i32x4_extract_lane::<{j}>(a) as u32")
            } else {
                format!("(i32x4_extract_lane::<{j}>(a) as u32)")
            }
        })
        .collect::<Vec<_>>()
        .join(".wrapping_add(")
        + &")".repeat(lanes - 1);

    formatdoc! {r#"
        impl {trait_name} for archmage::Wasm128Token {{
            type Repr = v128;

            #[inline(always)]
            fn splat(self, v: u32) -> v128 {{ u32x4_splat(v) }}
            #[inline(always)]
            fn zero(self) -> v128 {{ u32x4_splat(0) }}
            #[inline(always)]
            fn load(self, data: &{array}) -> v128 {{ crate::simd_storage::copy(data) }}
            #[inline(always)]
            fn from_array(self, arr: {array}) -> v128 {{ crate::simd_storage::cast(arr) }}
            #[inline(always)]
            fn store(self, repr: v128, out: &mut {array}) {{ crate::simd_storage::store(repr, out); }}
            #[inline(always)]
            fn to_array(self, repr: v128) -> {array} {{
                let mut out = [0u32; {lanes}];
                crate::simd_storage::store(repr, &mut out);
                out
            }}

            #[inline(always)]
            fn add(self, a: v128, b: v128) -> v128 {{ i32x4_add(a, b) }}
            #[inline(always)]
            fn sub(self, a: v128, b: v128) -> v128 {{ i32x4_sub(a, b) }}
            #[inline(always)]
            fn mul(self, a: v128, b: v128) -> v128 {{ i32x4_mul(a, b) }}
            #[inline(always)]
            fn min(self, a: v128, b: v128) -> v128 {{ u32x4_min(a, b) }}
            #[inline(always)]
            fn max(self, a: v128, b: v128) -> v128 {{ u32x4_max(a, b) }}

            #[inline(always)]
            fn simd_eq(self, a: v128, b: v128) -> v128 {{ i32x4_eq(a, b) }}
            #[inline(always)]
            fn simd_ne(self, a: v128, b: v128) -> v128 {{ i32x4_ne(a, b) }}
            #[inline(always)]
            fn simd_lt(self, a: v128, b: v128) -> v128 {{ u32x4_lt(a, b) }}
            #[inline(always)]
            fn simd_le(self, a: v128, b: v128) -> v128 {{ u32x4_le(a, b) }}
            #[inline(always)]
            fn simd_gt(self, a: v128, b: v128) -> v128 {{ u32x4_gt(a, b) }}
            #[inline(always)]
            fn simd_ge(self, a: v128, b: v128) -> v128 {{ u32x4_ge(a, b) }}
            #[inline(always)]
            fn blend(self, mask: v128, if_true: v128, if_false: v128) -> v128 {{
                v128_bitselect(if_true, if_false, mask)
            }}

            #[inline(always)]
            fn reduce_add(self, a: v128) -> u32 {{ {reduce_add_body} }}

            #[inline(always)]
            fn not(self, a: v128) -> v128 {{ v128_not(a) }}
            #[inline(always)]
            fn bitand(self, a: v128, b: v128) -> v128 {{ v128_and(a, b) }}
            #[inline(always)]
            fn bitor(self, a: v128, b: v128) -> v128 {{ v128_or(a, b) }}
            #[inline(always)]
            fn bitxor(self, a: v128, b: v128) -> v128 {{ v128_xor(a, b) }}

            #[inline(always)]
            fn shl_const<const N: i32>(self, a: v128) -> v128 {{ u32x4_shl(a, N as u32) }}
            #[inline(always)]
            fn shr_logical_const<const N: i32>(self, a: v128) -> v128 {{ u32x4_shr(a, N as u32) }}

            // ====== Uniform variable shifts ======
            // wasm takes the count modulo the lane width; the keep mask
            // restores the contracted zero at count >= 32.

            #[inline(always)]
            fn shl_uniform(self, a: v128, count: u32) -> v128 {{
                let keep: i32 = if count < 32 {{ -1 }} else {{ 0 }};
                v128_and(u32x4_shl(a, count), i32x4_splat(keep))
            }}

            #[inline(always)]
            fn shr_logical_uniform(self, a: v128, count: u32) -> v128 {{
                let keep: i32 = if count < 32 {{ -1 }} else {{ 0 }};
                v128_and(u32x4_shr(a, count), i32x4_splat(keep))
            }}

            #[inline(always)]
            fn all_true(self, a: v128) -> bool {{ i32x4_bitmask(a) == 0x0F }}
            #[inline(always)]
            fn any_true(self, a: v128) -> bool {{ i32x4_bitmask(a) != 0 }}
            #[inline(always)]
            fn bitmask(self, a: v128) -> u32 {{ i32x4_bitmask(a) as u32 }}
        }}
    "#}
}
