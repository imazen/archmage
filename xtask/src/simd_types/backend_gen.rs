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

use indoc::formatdoc;

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

    /// x86 token for this width
    #[allow(dead_code)]
    fn _x86_token(&self) -> &'static str {
        match self.width_bits {
            128 | 256 => "X64V3Token",
            512 => "X64V4Token",
            _ => unreachable!(),
        }
    }

    /// Whether this type is native on x86 V3 (AVX2+FMA)
    #[allow(dead_code)]
    fn _native_on_v3(&self) -> bool {
        self.width_bits <= 256
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

    /// WASM repr type
    fn wasm_repr(&self) -> String {
        if self.native_on_wasm() {
            "v128".to_string()
        } else {
            let count = self.width_bits / 128;
            format!("[v128; {count}]")
        }
    }

    /// Scalar repr type
    #[allow(dead_code)]
    fn _scalar_repr(&self) -> String {
        self.array_type()
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

    /// Number of 128-bit sub-vectors for polyfill
    fn sub_count(&self) -> usize {
        self.width_bits / 128
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

    /// NEON repr type
    fn neon_repr(&self) -> String {
        if self.native_on_neon() {
            "int32x4_t".to_string()
        } else {
            let count = self.width_bits / 128;
            format!("[int32x4_t; {count}]")
        }
    }

    /// WASM repr type
    fn wasm_repr(&self) -> String {
        if self.native_on_wasm() {
            "v128".to_string()
        } else {
            let count = self.width_bits / 128;
            format!("[v128; {count}]")
        }
    }

    /// Number of 128-bit sub-vectors for polyfill
    fn sub_count(&self) -> usize {
        self.width_bits / 128
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

    /// NEON repr type (uint32x4_t for unsigned)
    fn neon_repr(&self) -> String {
        if self.native_on_neon() {
            "uint32x4_t".to_string()
        } else {
            let count = self.width_bits / 128;
            format!("[uint32x4_t; {count}]")
        }
    }

    /// WASM repr type
    fn wasm_repr(&self) -> String {
        if self.native_on_wasm() {
            "v128".to_string()
        } else {
            let count = self.width_bits / 128;
            format!("[v128; {count}]")
        }
    }

    /// Number of 128-bit sub-vectors for polyfill
    fn sub_count(&self) -> usize {
        self.width_bits / 128
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
        all_w512_types, generate_neon_w512_impls, generate_popcnt_backend_traits,
        generate_scalar_w512_impls, generate_w512_backend_trait, generate_wasm_w512_impls,
        generate_x86_v3_w512_impls,
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

    // 10. Implementation files
    files.insert(
        "impls/x86_v3.rs".to_string(),
        generate_x86_impls(&types, "X64V3Token", 256)
            + &generate_x86_i32_impls(&i32_types, "X64V3Token", 256)
            + &generate_x86_u32_impls(&u32_types, "X64V3Token", 256)
            + &generate_x86_i64_impls(&i64_types, "X64V3Token", 256)
            + &generate_x86_int_impls(&remaining_int_types, "X64V3Token", 256)
            + &generate_x86_convert_impls("X64V3Token")
            + &generate_x86_additional_convert_impls("X64V3Token")
            + &gate_w512_impls(generate_x86_v3_w512_impls(&w512_types)),
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
            + &generate_neon_additional_convert_impls()
            + &gate_w512_impls(generate_neon_w512_impls(&w512_types)),
    );
    files.insert(
        "impls/wasm128.rs".to_string(),
        generate_wasm_impls(&types)
            + &generate_wasm_i32_impls(&i32_types)
            + &generate_wasm_u32_impls(&u32_types)
            + &generate_wasm_i64_impls(&i64_types)
            + &generate_wasm_int_impls(&remaining_int_types)
            + &generate_wasm_convert_impls()
            + &generate_wasm_additional_convert_impls()
            + &gate_w512_impls(generate_wasm_w512_impls(&w512_types)),
    );

    // 10b. x86 V4 native AVX-512 implementation (W512 types only). Already
    // gated by `feature = "avx512"` at the `mod x86_v4` declaration, but we
    // also need the individual impls to be `w512`-gated because `avx512`
    // implies `w512` (and we want a single gate scheme).
    files.insert(
        "impls/x86_v4.rs".to_string(),
        gate_w512_impls(generate_x86_v4_impls_file(&w512_types)),
    );

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
                core::array::from_fn(|i| crate::nostd_math::roundevenf(arr[i]).clamp(0.0, 255.0) as u8)
            }}

            /// Round/clamp 4 planar channels and interleave into RGBA bytes
            /// ({lanes} pixels = {rgba_bytes} bytes). Each channel converts via
            /// the native `to_u8_bytes` (so x86/aarch64 get cvt+pack, not scalar
            /// `roundevenf`); the byte interleave is left to LLVM, which recovers
            /// it to native shuffles. A backend may still override for a tighter
            /// fused sequence.
            #[inline(always)]
            fn store_rgba_bytes(self, r: Self::Repr, g: Self::Repr, b: Self::Repr, a: Self::Repr) -> [u8; {rgba_bytes}] {{
                let rb = <Self as {trait_name}>::to_u8_bytes(self, r);
                let gb = <Self as {trait_name}>::to_u8_bytes(self, g);
                let bb = <Self as {trait_name}>::to_u8_bytes(self, b);
                let ab = <Self as {trait_name}>::to_u8_bytes(self, a);
                let mut out = [0u8; {rgba_bytes}];
                let mut i = 0;
                while i < {lanes} {{
                    out[i * 4] = rb[i];
                    out[i * 4 + 1] = gb[i];
                    out[i * 4 + 2] = bb[i];
                    out[i * 4 + 3] = ab[i];
                    i += 1;
                }}
                out
            }}
        "#, rgba_bytes = lanes * 4}
    } else {
        String::new()
    };

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

    formatdoc! {r#"
        //! Backend trait for `{name}<T>` — {lanes}-lane {elem} SIMD vector.
        //!
        //! Each token type implements this trait with its platform-native representation.
        //! The generic wrapper `{name}<T>` delegates all operations to these trait methods.
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.

        use super::sealed::Sealed;
        use archmage::SimdToken;

        /// Backend implementation for {lanes}-lane {elem} SIMD vectors.
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
            type Repr: Copy + Clone + Send + Sync;

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

            /// Lane-wise minimum.
            fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise maximum.
            fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Square root.
            fn sqrt(self, a: Self::Repr) -> Self::Repr;

            /// Absolute value.
            fn abs(self, a: Self::Repr) -> Self::Repr;

            /// Round toward negative infinity.
            fn floor(self, a: Self::Repr) -> Self::Repr;

            /// Round toward positive infinity.
            fn ceil(self, a: Self::Repr) -> Self::Repr;

            /// Round to nearest integer.
            fn round(self, a: Self::Repr) -> Self::Repr;

            /// Fused multiply-add: a * b + c.
            fn mul_add(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;

            /// Fused multiply-sub: a * b - c.
            fn mul_sub(self, a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;

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
            /// (x86 ~12-bit, ARM ~8-bit, WASM full). [`rsqrt`] is full f32 everywhere.
            ///
            /// See [`rcp_approx`] for default-body rationale.
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

            /// Clamp values between lo and hi.
            #[inline(always)]
            fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {{
                <Self as {trait_name}>::min(self, <Self as {trait_name}>::max(self, a, lo), hi)
            }}

            /// Precise reciprocal — defaults to delegating to [`rcp_approx`]
            /// (which itself defaults to identity). Backends override with
            /// Newton-Raphson refinement using a native splat for the constant.
            #[inline(always)]
            fn recip(self, a: Self::Repr) -> Self::Repr {{ Self::rcp_approx(self, a) }}

            /// Precise reciprocal square root — see [`recip`] for rationale.
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
        code.push_str(&format!("mod {name};\npub use {name}::{trait_name};\n\n"));
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
            "#[cfg(feature = \"w512\")]\nmod {name};\n#[cfg(feature = \"w512\")]\npub use {name}::{trait_name};\n\n"
        ));
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

        #[cfg(target_arch = "x86_64")]
        mod x86_v3;

        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        mod x86_v4;

        // V4-family delegation of W128 / W256 f32 backends to V3 (hand-written —
        // future generators may take it over). V4 ⊃ V3, so delegating is sound.
        #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
        mod x86_v4_f32_delegated;

        #[cfg(target_arch = "aarch64")]
        mod arm_neon;

        #[cfg(target_arch = "wasm32")]
        mod wasm128;

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
    formatdoc! {r#"
        //! # Safety (audit contract for every `unsafe` block in this file)
        //!
        //! All `unsafe` blocks below are inside `impl ... for {token_desc}`
        //! blocks and fall into exactly three shapes:
        //!
        //! 1. **Value-based intrinsic calls** — sound because the receiver
        //!    token is a proof the CPU supports the intrinsic's required
        //!    features (`cargo xtask soundness` statically verifies every
        //!    intrinsic's feature set against the impl's token on every
        //!    generate/CI run; tokens are only obtainable via runtime
        //!    detection).
        //! 2. **Loads/stores through references** (`as_ptr`/`as_mut_ptr` on
        //!    sized arrays) — sound because the reference guarantees a valid,
        //!    correctly-sized allocation, and the unaligned-tolerant
        //!    instructions are used.
        //! 3. **`transmute` between fixed-size arrays and vector types** —
        //!    sound because both sides are plain-old-data of equal size
        //!    (compile-time checked by `transmute` itself).
        //!
        //! Anything outside these shapes must carry its own `// SAFETY:`
        //! comment and be added to the audit notes in `docs/SOUNDNESS.md`.
    "#}
}

/// Contains W512 backend impls for both X64V4Token and X64V4xToken,
/// plus Modern-specific extension impls (popcnt).
/// W128 and W256 types use X64V3Token (V4 downcasts to V3 for narrower widths).
fn generate_x86_v4_impls_file(w512_types: &[super::backend_gen_w512::W512Type]) -> String {
    use super::backend_gen_w512::{
        generate_popcnt_impls, generate_x86_modern_w512_impls, generate_x86_v4_w512_impls,
    };

    let safety = impls_safety_contract("X64V4Token / X64V4xToken");
    let mut code = formatdoc! {r#"
        //! Backend implementations for X64V4Token and X64V4xToken (native AVX-512).
        //!
        //! Implements the W512 backend traits using native 512-bit AVX-512 intrinsics
        //! for both X64V4Token (base AVX-512) and X64V4xToken (+ VPOPCNTDQ, BITALG, etc.).
        //!
        //! X64V4xToken also gets extension trait impls (popcnt) for Modern-only features.
        //!
        //! W128 and W256 types use X64V3Token (V4 downcasts to V3 for narrower widths).
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.
        //!
        {safety}
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;

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

    // Base backend impls for X64V4xToken (identical intrinsics, different token)
    code.push_str(
        "\n// ============================================================================\n",
    );
    code.push_str("// X64V4xToken — base AVX-512 (same intrinsics as V4)\n");
    code.push_str(
        "// ============================================================================\n\n",
    );
    code.push_str(&generate_x86_modern_w512_impls(w512_types));

    // Extension impls: popcnt (X64V4xToken only)
    code.push_str(
        "\n// ============================================================================\n",
    );
    code.push_str("// X64V4xToken — extension: popcnt (VPOPCNTDQ + BITALG)\n");
    code.push_str(
        "// ============================================================================\n\n",
    );
    code.push_str(&generate_popcnt_impls(w512_types));

    // F32x16Convert impls for V4 and V4x (native 512-bit)
    code.push_str(
        "\n// ============================================================================\n",
    );
    code.push_str("// F32x16Convert — native AVX-512 conversions\n");
    code.push_str(
        "// ============================================================================\n\n",
    );
    for token in ["X64V4Token", "X64V4xToken"] {
        code.push_str(&generate_x86_v4_f32x16_convert(token));
    }

    code
}

fn generate_x86_v4_f32x16_convert(token: &str) -> String {
    formatdoc! {r#"
        #[cfg(all(target_arch = "x86_64", feature = "w512"))]
        impl F32x16Convert for archmage::{token} {{
            #[inline(always)]
            fn bitcast_f32_to_i32(self, a: __m512) -> __m512i {{
                unsafe {{ _mm512_castps_si512(a) }}
            }}

            #[inline(always)]
            fn bitcast_i32_to_f32(self, a: __m512i) -> __m512 {{
                unsafe {{ _mm512_castsi512_ps(a) }}
            }}

            #[inline(always)]
            fn convert_f32_to_i32(self, a: __m512) -> __m512i {{
                unsafe {{ _mm512_cvttps_epi32(a) }}
            }}

            #[inline(always)]
            fn convert_f32_to_i32_round(self, a: __m512) -> __m512i {{
                unsafe {{ _mm512_cvtps_epi32(a) }}
            }}

            #[inline(always)]
            fn convert_i32_to_f32(self, a: __m512i) -> __m512 {{
                unsafe {{ _mm512_cvtepi32_ps(a) }}
            }}
        }}

    "#}
}

// ============================================================================
// x86 Implementation Generation
// ============================================================================

fn generate_x86_impls(types: &[FloatVecType], token: &str, max_width: usize) -> String {
    let safety = impls_safety_contract(token);
    let mut code = formatdoc! {r#"
        //! Backend implementations for {token} (x86-64).
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.
        //!
        {safety}
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;

        use crate::simd::backends::*;

    "#};

    for ty in types {
        if ty.width_bits > max_width {
            continue;
        }
        code.push_str("#[cfg(target_arch = \"x86_64\")]\n");
        code.push_str(&generate_x86_float_impl(ty, token));
        code.push('\n');
    }

    code
}

fn generate_x86_float_impl(ty: &FloatVecType, token: &str) -> String {
    let trait_name = ty.trait_name();
    let inner = ty.x86_inner_type();
    let p = ty.x86_prefix();
    let s = ty.x86_suffix();
    let bits = ty.width_bits;
    let array = ty.array_type();
    let elem = ty.elem;
    let lanes = ty.lanes;

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
        // The hardware estimate is ~12-14 bit and each Newton step doubles the
        // correct bits, so f32 (24-bit mantissa) needs 1 step but f64 (53-bit)
        // needs 2 to reach full precision.
        let n_steps = if elem == "f64" { 2 } else { 1 };
        // Last step is a tail expression (no `let`, so no clippy::let_and_return).
        let recip_steps = (0..n_steps).map(|i| {
            let step = format!("<Self as {trait_name}>::mul(self, r, <Self as {trait_name}>::sub(self, two, <Self as {trait_name}>::mul(self, a, r)))");
            if i + 1 < n_steps { format!("let r = {step};") } else { step }
        }).collect::<Vec<_>>().join("\n");
        let rsqrt_steps = (0..n_steps).map(|i| {
            let step = format!("<Self as {trait_name}>::mul(self, <Self as {trait_name}>::mul(self, half, y), <Self as {trait_name}>::sub(self, three, <Self as {trait_name}>::mul(self, a, <Self as {trait_name}>::mul(self, y, y))))");
            if i + 1 < n_steps { format!("let y = {step};") } else { step }
        }).collect::<Vec<_>>().join("\n");
        formatdoc! {r#"
            #[inline(always)]
            fn rcp_approx(self, a: {inner}) -> {inner} {{
                unsafe {{ {p}_{rcp_fn}_{s}(a) }}
            }}

            #[inline(always)]
            fn rsqrt_approx(self, a: {inner}) -> {inner} {{
                unsafe {{ {p}_{rsqrt_fn}_{s}(a) }}
            }}

            // Newton-Raphson refinement to full precision ({n_steps} step(s)).
            // Constants via value-based splat (impl block is target-feature gated).
            #[inline(always)]
            fn recip(self, a: {inner}) -> {inner} {{
                let two = unsafe {{ {p}_set1_{s}(2.0) }};
                let r = <Self as {trait_name}>::rcp_approx(self, a);
                {recip_steps}
            }}

            #[inline(always)]
            fn rsqrt(self, a: {inner}) -> {inner} {{
                let half = unsafe {{ {p}_set1_{s}(0.5) }};
                let three = unsafe {{ {p}_set1_{s}(3.0) }};
                let y = <Self as {trait_name}>::rsqrt_approx(self, a);
                {rsqrt_steps}
            }}
        "#}
    } else if elem == "f64" {
        // No hardware reciprocal estimate for f64 below AVX-512. Both the
        // "approx" and full methods use exact IEEE division / sqrt — full
        // precision and deterministic. Mirrors the WASM/scalar fallback, and
        // overrides the trait default (which is identity) so f64x2/f64x4 are
        // correct rather than returning the input unchanged.
        formatdoc! {r#"
            #[inline(always)]
            fn rcp_approx(self, a: {inner}) -> {inner} {{
                let one = unsafe {{ {p}_set1_{s}(1.0) }};
                <Self as {trait_name}>::div(self, one, a)
            }}

            #[inline(always)]
            fn rsqrt_approx(self, a: {inner}) -> {inner} {{
                let one = unsafe {{ {p}_set1_{s}(1.0) }};
                <Self as {trait_name}>::div(self, one, <Self as {trait_name}>::sqrt(self, a))
            }}

            #[inline(always)]
            fn recip(self, a: {inner}) -> {inner} {{
                let one = unsafe {{ {p}_set1_{s}(1.0) }};
                <Self as {trait_name}>::div(self, one, a)
            }}

            #[inline(always)]
            fn rsqrt(self, a: {inner}) -> {inner} {{
                let one = unsafe {{ {p}_set1_{s}(1.0) }};
                <Self as {trait_name}>::div(self, one, <Self as {trait_name}>::sqrt(self, a))
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

            #[inline(always)]
            fn to_u8_bytes(self, a: {inner}) -> [u8; 4] {{
                unsafe {{
                    let i32s = _mm_cvtps_epi32(a);
                    let i16s = _mm_packs_epi32(i32s, i32s);
                    let u8s = _mm_packus_epi16(i16s, i16s);
                    (_mm_cvtsi128_si32(u8s) as u32).to_ne_bytes()
                }}
            }}
        "#}
    } else if elem == "f32" && bits == 256 {
        formatdoc! {r#"

            #[inline(always)]
            fn to_u8_bytes(self, a: {inner}) -> [u8; 8] {{
                unsafe {{
                    let i32s = _mm256_cvtps_epi32(a);
                    let lo = _mm256_castsi256_si128(i32s);
                    let hi = _mm256_extracti128_si256::<1>(i32s);
                    let i16s = _mm_packs_epi32(lo, hi);
                    let u8s = _mm_packus_epi16(i16s, i16s);
                    (_mm_cvtsi128_si64(u8s) as u64).to_ne_bytes()
                }}
            }}
        "#}
    } else {
        String::new()
    };

    // Native pixel interleave (overrides the to_u8_bytes-based default, whose
    // generic byte interleave LLVM expands to ~8 pshufb). All 4 planes packed
    // together in-register, one pshufb to RGBA order. Round-to-even via cvtps
    // (MXCSR default); packs/packus saturate to [0,255] (= clamp).
    let store_rgba_x86 = if elem == "f32" && bits == 128 {
        formatdoc! {r#"

            #[inline(always)]
            fn store_rgba_bytes(self, r: {inner}, g: {inner}, b: {inner}, a: {inner}) -> [u8; 16] {{
                unsafe {{
                    let rg = _mm_packs_epi32(_mm_cvtps_epi32(r), _mm_cvtps_epi32(g));
                    let ba = _mm_packs_epi32(_mm_cvtps_epi32(b), _mm_cvtps_epi32(a));
                    // [R0-3,G0-3,B0-3,A0-3] -> interleaved RGBA pixels 0-3.
                    let packed = _mm_packus_epi16(rg, ba);
                    let shuf = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
                    core::mem::transmute(_mm_shuffle_epi8(packed, shuf))
                }}
            }}
        "#}
    } else if elem == "f32" && bits == 256 {
        formatdoc! {r#"

            #[inline(always)]
            fn store_rgba_bytes(self, r: {inner}, g: {inner}, b: {inner}, a: {inner}) -> [u8; 32] {{
                unsafe {{
                    // AVX2 packs are lane-wise: lane0 holds pixels 0-3, lane1 4-7.
                    let rg = _mm256_packs_epi32(_mm256_cvtps_epi32(r), _mm256_cvtps_epi32(g));
                    let ba = _mm256_packs_epi32(_mm256_cvtps_epi32(b), _mm256_cvtps_epi32(a));
                    let packed = _mm256_packus_epi16(rg, ba);
                    let shuf = _mm256_setr_epi8(
                        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
                        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
                    );
                    core::mem::transmute(_mm256_shuffle_epi8(packed, shuf))
                }}
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

            #[inline(always)]
            fn transpose_8x8_repr(self, rows: [{inner}; 8]) -> [{inner}; 8] {{
                unsafe {{
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
            }}
        "#}
    } else {
        String::new()
    };

    formatdoc! {r#"
        impl {trait_name} for archmage::{token} {{
            type Repr = {inner};

            // ====== Construction ======

            #[inline(always)]
            fn splat(self, v: {elem}) -> {inner} {{
                unsafe {{ {set1}(v) }}
            }}

            #[inline(always)]
            fn zero(self) -> {inner} {{
                unsafe {{ {setzero}() }}
            }}

            #[inline(always)]
            fn load(self, data: &{array}) -> {inner} {{
                unsafe {{ {p}_loadu_{s}(data.as_ptr()) }}
            }}

            #[inline(always)]
            fn from_array(self, arr: {array}) -> {inner} {{
                // SAFETY: {array} and {inner} have identical size and layout.
                unsafe {{ core::mem::transmute(arr) }}
            }}

            #[inline(always)]
            fn store(self, repr: {inner}, out: &mut {array}) {{
                unsafe {{ {p}_storeu_{s}(out.as_mut_ptr(), repr) }};
            }}

            #[inline(always)]
            fn to_array(self, repr: {inner}) -> {array} {{
                let mut out = [{zero_lit}; {lanes}];
                unsafe {{ {p}_storeu_{s}(out.as_mut_ptr(), repr) }};
                out
            }}

            // ====== Arithmetic ======

            #[inline(always)]
            fn add(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_add_{s}(a, b) }}
            }}

            #[inline(always)]
            fn sub(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_sub_{s}(a, b) }}
            }}

            #[inline(always)]
            fn mul(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_mul_{s}(a, b) }}
            }}

            #[inline(always)]
            fn div(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_div_{s}(a, b) }}
            }}

            #[inline(always)]
            fn neg(self, a: {inner}) -> {inner} {{
                unsafe {{ {p}_sub_{s}({setzero}(), a) }}
            }}

            // ====== Math ======

            #[inline(always)]
            fn min(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_min_{s}(a, b) }}
            }}

            #[inline(always)]
            fn max(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_max_{s}(a, b) }}
            }}

            #[inline(always)]
            fn sqrt(self, a: {inner}) -> {inner} {{
                unsafe {{ {p}_sqrt_{s}(a) }}
            }}

            #[inline(always)]
            fn abs(self, a: {inner}) -> {inner} {{
                unsafe {{
                    let mask = {cast_int_to_float}({p}_set1_{set1_int}({abs_mask}));
                    {p}_and_{s}(a, mask)
                }}
            }}

            #[inline(always)]
            fn floor(self, a: {inner}) -> {inner} {{
                unsafe {{ {floor_intr}(a) }}
            }}

            #[inline(always)]
            fn ceil(self, a: {inner}) -> {inner} {{
                unsafe {{ {ceil_intr}(a) }}
            }}

            #[inline(always)]
            fn round(self, a: {inner}) -> {inner} {{
                unsafe {{ {round_intr}(a) }}
            }}

            #[inline(always)]
            fn mul_add(self, a: {inner}, b: {inner}, c: {inner}) -> {inner} {{
                unsafe {{ {p}_fmadd_{s}(a, b, c) }}
            }}

            #[inline(always)]
            fn mul_sub(self, a: {inner}, b: {inner}, c: {inner}) -> {inner} {{
                unsafe {{ {p}_fmsub_{s}(a, b, c) }}
            }}

            // ====== Comparisons ======

            #[inline(always)]
            fn simd_eq(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {cmp}::<_CMP_EQ_OQ>(a, b) }}
            }}

            #[inline(always)]
            fn simd_ne(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {cmp}::<_CMP_NEQ_OQ>(a, b) }}
            }}

            #[inline(always)]
            fn simd_lt(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {cmp}::<_CMP_LT_OQ>(a, b) }}
            }}

            #[inline(always)]
            fn simd_le(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {cmp}::<_CMP_LE_OQ>(a, b) }}
            }}

            #[inline(always)]
            fn simd_gt(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {cmp}::<_CMP_GT_OQ>(a, b) }}
            }}

            #[inline(always)]
            fn simd_ge(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {cmp}::<_CMP_GE_OQ>(a, b) }}
            }}

            #[inline(always)]
            fn blend(self, mask: {inner}, if_true: {inner}, if_false: {inner}) -> {inner} {{
                unsafe {{ {p}_blendv_{s}(if_false, if_true, mask) }}
            }}

            // ====== Reductions ======

            #[inline(always)]
            fn reduce_add(self, a: {inner}) -> {elem} {{
        {reduce_add_body}
            }}

            #[inline(always)]
            fn reduce_min(self, a: {inner}) -> {elem} {{
        {reduce_min_body}
            }}

            #[inline(always)]
            fn reduce_max(self, a: {inner}) -> {elem} {{
        {reduce_max_body}
            }}

            // ====== Approximations ======

        {approx_section}
            // ====== Bitwise ======

            #[inline(always)]
            fn not(self, a: {inner}) -> {inner} {{
                unsafe {{
                    let ones = {p}_set1_{set1_int}(-1);
                    let as_int = {cast_float_to_int}(a);
                    {cast_int_to_float}({p}_xor_si{bits}(as_int, ones))
                }}
            }}

            #[inline(always)]
            fn bitand(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_and_{s}(a, b) }}
            }}

            #[inline(always)]
            fn bitor(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_or_{s}(a, b) }}
            }}

            #[inline(always)]
            fn bitxor(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_xor_{s}(a, b) }}
            }}
            {to_u8_x86}
            {store_rgba_x86}
            {transpose_8x8_x86}
        }}
    "#,
        zero_lit = if elem == "f32" { "0.0f32" } else { "0.0f64" },
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
                unsafe {{
                    let shuf = _mm_shuffle_ps::<0b10_11_00_01>(a, a);
                    let s1 = _mm_add_ps(a, shuf);
                    let s2 = _mm_add_ps(s1, _mm_movehl_ps(s1, s1));
                    {cvt}(s2)
                }}"},
        ("f64", 128) => formatdoc! {"
                unsafe {{
                    let h = {hadd}(a, a);
                    {cvt}(h)
                }}"},
        ("f32", 256) => formatdoc! {"
                unsafe {{
                    let hi = {p}_extractf128_{s}::<1>(a);
                    let lo = {p}_cast{s}256_{s}128(a);
                    let sum = _mm_add_ps(lo, hi);
                    let shuf = _mm_shuffle_ps::<0b10_11_00_01>(sum, sum);
                    let s1 = _mm_add_ps(sum, shuf);
                    let s2 = _mm_add_ps(s1, _mm_movehl_ps(s1, s1));
                    {cvt}(s2)
                }}"},
        ("f64", 256) => formatdoc! {"
                unsafe {{
                    let hi = {p}_extractf128_{s}::<1>(a);
                    let lo = {p}_cast{s}256_{s}128(a);
                    let sum = _mm_add_{s}(lo, hi);
                    let h = {hadd}(sum, sum);
                    {cvt}(h)
                }}"},
        ("f32", 512) => "        unsafe { _mm512_reduce_add_ps(a) }".to_string(),
        ("f64", 512) => "        unsafe { _mm512_reduce_add_pd(a) }".to_string(),
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
                unsafe {{
                    let shuf = _mm_shuffle_ps::<0b10_11_00_01>(a, a);
                    let m1 = _mm_min_ps(a, shuf);
                    let shuf2 = _mm_shuffle_ps::<0b00_00_10_10>(m1, m1);
                    let m2 = _mm_min_ps(m1, shuf2);
                    {cvt}(m2)
                }}"},
        ("f64", 128) => formatdoc! {"
                unsafe {{
                    let shuf = _mm_shuffle_pd::<0b01>(a, a);
                    let m = _mm_min_pd(a, shuf);
                    {cvt}(m)
                }}"},
        ("f32", 256) => formatdoc! {"
                unsafe {{
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
                unsafe {{
                    let hi = {p}_extractf128_{s}::<1>(a);
                    let lo = {p}_cast{s}256_{s}128(a);
                    let m = _mm_min_{s}(lo, hi);
                    let shuf = _mm_shuffle_pd::<0b01>(m, m);
                    let m2 = _mm_min_pd(m, shuf);
                    {cvt}(m2)
                }}"},
        ("f32", 512) => "        unsafe { _mm512_reduce_min_ps(a) }".to_string(),
        ("f64", 512) => "        unsafe { _mm512_reduce_min_pd(a) }".to_string(),
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
                unsafe {{
                    let shuf = _mm_shuffle_ps::<0b10_11_00_01>(a, a);
                    let m1 = _mm_max_ps(a, shuf);
                    let shuf2 = _mm_shuffle_ps::<0b00_00_10_10>(m1, m1);
                    let m2 = _mm_max_ps(m1, shuf2);
                    {cvt}(m2)
                }}"},
        ("f64", 128) => formatdoc! {"
                unsafe {{
                    let shuf = _mm_shuffle_pd::<0b01>(a, a);
                    let m = _mm_max_pd(a, shuf);
                    {cvt}(m)
                }}"},
        ("f32", 256) => formatdoc! {"
                unsafe {{
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
                unsafe {{
                    let hi = {p}_extractf128_{s}::<1>(a);
                    let lo = {p}_cast{s}256_{s}128(a);
                    let m = _mm_max_{s}(lo, hi);
                    let shuf = _mm_shuffle_pd::<0b01>(m, m);
                    let m2 = _mm_max_pd(m, shuf);
                    {cvt}(m2)
                }}"},
        ("f32", 512) => "        unsafe { _mm512_reduce_max_ps(a) }".to_string(),
        ("f64", 512) => "        unsafe { _mm512_reduce_max_pd(a) }".to_string(),
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
    // These use crate::nostd_math for no_std compatibility — the scalar backend
    // runs on platforms without hardware SIMD or std math intrinsics.
    let mut seen_elems = std::collections::BTreeSet::new();
    for ty in types {
        if seen_elems.insert(ty.elem) {
            let (sqrt_fn, floor_fn, ceil_fn, round_fn) = if ty.elem == "f32" {
                (
                    "crate::nostd_math::sqrtf",
                    "crate::nostd_math::floorf",
                    "crate::nostd_math::ceilf",
                    "crate::nostd_math::roundevenf",
                )
            } else {
                (
                    "crate::nostd_math::sqrt",
                    "crate::nostd_math::floor",
                    "crate::nostd_math::ceil",
                    "crate::nostd_math::roundeven",
                )
            };
            code.push_str(&formatdoc! {r#"
                // Helpers to avoid trait method name shadowing inside the impl block.
                // Inside `impl XxxBackend`, names like `sqrt`, `floor`, etc. resolve to
                // the trait's associated functions instead of {elem}'s inherent methods.
                // Uses crate::nostd_math for no_std compatibility.
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

    // Generate lane-by-lane operations
    let binary_lanes = |op: &str| -> String {
        let items: Vec<String> = (0..lanes).map(|i| format!("a[{i}] {op} b[{i}]")).collect();
        format!("[{}]", items.join(", "))
    };

    let min_lanes_fn = || -> String {
        let items: Vec<String> = (0..lanes).map(|i| format!("a[{i}].min(b[{i}])")).collect();
        format!("[{}]", items.join(", "))
    };

    let max_lanes_fn = || -> String {
        let items: Vec<String> = (0..lanes).map(|i| format!("a[{i}].max(b[{i}])")).collect();
        format!("[{}]", items.join(", "))
    };

    let neg_lanes = || -> String {
        let items: Vec<String> = (0..lanes).map(|i| format!("-a[{i}]")).collect();
        format!("[{}]", items.join(", "))
    };

    let abs_mask = if elem == "f32" {
        "0x7FFF_FFFF"
    } else {
        "0x7FFF_FFFF_FFFF_FFFF"
    };
    let abs_lanes = || -> String {
        let items: Vec<String> = (0..lanes)
            .map(|i| format!("{elem}::from_bits(a[{i}].to_bits() & {abs_mask})"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let mul_add_lanes = || -> String {
        let items: Vec<String> = (0..lanes)
            .map(|i| format!("a[{i}] * b[{i}] + c[{i}]"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let mul_sub_lanes = || -> String {
        let items: Vec<String> = (0..lanes)
            .map(|i| format!("a[{i}] * b[{i}] - c[{i}]"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let bitwise_unary_lanes = |op: &str| -> String {
        let items: Vec<String> = (0..lanes)
            .map(|i| format!("{elem}::from_bits({op}a[{i}].to_bits())"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let bitwise_binary_lanes = |op: &str| -> String {
        let items: Vec<String> = (0..lanes)
            .map(|i| format!("{elem}::from_bits(a[{i}].to_bits() {op} b[{i}].to_bits())"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    // Adjacent-pair tree for f32 — matches x86 v3 and NEON reduce_add
    // shape. Better ILP than linear left-fold and stays consistent across
    // backends (#50). f64 with 2 or 4 lanes has no meaningful difference.
    let reduce_add = || -> String {
        match (elem, lanes) {
            ("f32", 4) => "(a[0] + a[1]) + (a[2] + a[3])".to_string(),
            ("f32", 8) => {
                "((a[0] + a[4]) + (a[1] + a[5])) + ((a[2] + a[6]) + (a[3] + a[7]))".to_string()
            }
            _ => {
                let items: Vec<String> = (0..lanes).map(|i| format!("a[{i}]")).collect();
                items.join(" + ")
            }
        }
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
                {add_lanes}
            }}

            #[inline(always)]
            fn sub(self, a: {array}, b: {array}) -> {array} {{
                {sub_lanes}
            }}

            #[inline(always)]
            fn mul(self, a: {array}, b: {array}) -> {array} {{
                {mul_lanes}
            }}

            #[inline(always)]
            fn div(self, a: {array}, b: {array}) -> {array} {{
                {div_lanes}
            }}

            #[inline(always)]
            fn neg(self, a: {array}) -> {array} {{
                {neg}
            }}

            // ====== Math ======

            #[inline(always)]
            fn min(self, a: {array}, b: {array}) -> {array} {{
                {min_lanes}
            }}

            #[inline(always)]
            fn max(self, a: {array}, b: {array}) -> {array} {{
                {max_lanes}
            }}

            #[inline(always)]
            fn sqrt(self, a: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = {elem}_sqrt(a[i]);
                }}
                r
            }}

            #[inline(always)]
            fn abs(self, a: {array}) -> {array} {{
                {abs}
            }}

            #[inline(always)]
            fn floor(self, a: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = {elem}_floor(a[i]);
                }}
                r
            }}

            #[inline(always)]
            fn ceil(self, a: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = {elem}_ceil(a[i]);
                }}
                r
            }}

            #[inline(always)]
            fn round(self, a: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = {elem}_round(a[i]);
                }}
                r
            }}

            #[inline(always)]
            fn mul_add(self, a: {array}, b: {array}, c: {array}) -> {array} {{
                {mul_add}
            }}

            #[inline(always)]
            fn mul_sub(self, a: {array}, b: {array}, c: {array}) -> {array} {{
                {mul_sub}
            }}

            // ====== Comparisons ======

            #[inline(always)]
            fn simd_eq(self, a: {array}, b: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] == b[i] {{ {true_mask} }} else {{ 0.0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_ne(self, a: {array}, b: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] != b[i] {{ {true_mask} }} else {{ 0.0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_lt(self, a: {array}, b: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] < b[i] {{ {true_mask} }} else {{ 0.0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_le(self, a: {array}, b: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] <= b[i] {{ {true_mask} }} else {{ 0.0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_gt(self, a: {array}, b: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] > b[i] {{ {true_mask} }} else {{ 0.0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_ge(self, a: {array}, b: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] >= b[i] {{ {true_mask} }} else {{ 0.0 }};
                }}
                r
            }}

            #[inline(always)]
            fn blend(self, mask: {array}, if_true: {array}, if_false: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    // Check sign bit of mask (all-1s has sign bit set)
                    r[i] = if (mask[i].to_bits() >> {sign_shift}) != 0 {{
                        if_true[i]
                    }} else {{
                        if_false[i]
                    }};
                }}
                r
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
                {not_lanes}
            }}

            #[inline(always)]
            fn bitand(self, a: {array}, b: {array}) -> {array} {{
                {and_lanes}
            }}

            #[inline(always)]
            fn bitor(self, a: {array}, b: {array}) -> {array} {{
                {or_lanes}
            }}

            #[inline(always)]
            fn bitxor(self, a: {array}, b: {array}) -> {array} {{
                {xor_lanes}
            }}
        }}
    "#,
        add_lanes = binary_lanes("+"),
        sub_lanes = binary_lanes("-"),
        mul_lanes = binary_lanes("*"),
        div_lanes = binary_lanes("/"),
        neg = neg_lanes(),
        min_lanes = min_lanes_fn(),
        max_lanes = max_lanes_fn(),
        // Actually, min/max need a different pattern...
        abs = abs_lanes(),
        mul_add = mul_add_lanes(),
        mul_sub = mul_sub_lanes(),
        reduce_add = reduce_add(),
        not_lanes = bitwise_unary_lanes("!"),
        and_lanes = bitwise_binary_lanes("&"),
        or_lanes = bitwise_binary_lanes("|"),
        xor_lanes = bitwise_binary_lanes("^"),
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

        use crate::simd::backends::*;

    "#};

    for ty in types {
        code.push_str("#[cfg(target_arch = \"aarch64\")]\n");
        code.push_str(&generate_neon_float_impl(ty));
        code.push('\n');
    }

    code
}

fn generate_neon_float_impl(ty: &FloatVecType) -> String {
    let trait_name = ty.trait_name();
    let repr = ty.neon_repr();
    let elem = ty.elem;
    let lanes = ty.lanes;
    let array = ty.array_type();
    let ns = ty.neon_suffix();
    let zero_lit = if elem == "f32" { "0.0f32" } else { "0.0f64" };
    let sub_count = ty.sub_count();

    // NEON native type names
    let _native_type = match elem {
        "f32" => "float32x4_t",
        "f64" => "float64x2_t",
        _ => unreachable!(),
    };
    let native_lanes = if elem == "f32" { 4 } else { 2 };

    // If this is a native 128-bit type, no polyfill needed
    if ty.native_on_neon() {
        return generate_neon_native_impl(ty);
    }

    // Polyfill: apply operation to each sub-vector
    let binary_op = |intrinsic: &str| -> String {
        let items: Vec<String> = (0..sub_count)
            .map(|i| format!("{intrinsic}(a[{i}], b[{i}])"))
            .collect();
        format!("unsafe {{ [{}] }}", items.join(", "))
    };

    let unary_op = |intrinsic: &str| -> String {
        let items: Vec<String> = (0..sub_count)
            .map(|i| format!("{intrinsic}(a[{i}])"))
            .collect();
        format!("unsafe {{ [{}] }}", items.join(", "))
    };

    let cmp_op = |intrinsic: &str| -> String {
        let items: Vec<String> = (0..sub_count)
            .map(|i| {
                format!(
                    "vreinterpretq_{ns}_u{elem_bits}({intrinsic}(a[{i}], b[{i}]))",
                    elem_bits = if elem == "f32" { 32 } else { 64 }
                )
            })
            .collect();
        format!("unsafe {{ [{}] }}", items.join(", "))
    };

    let ne_op = || -> String {
        let eb = if elem == "f32" { 32 } else { 64 };
        let items: Vec<String> = (0..sub_count)
            .map(|i| {
                if eb == 64 {
                    // NEON lacks vmvnq_u64 — use XOR with all-ones
                    format!(
                        "vreinterpretq_{ns}_u64(veorq_u64(vceqq_{ns}(a[{i}], b[{i}]), vdupq_n_u64(u64::MAX)))",
                        ns = ns
                    )
                } else {
                    format!(
                        "vreinterpretq_{ns}_u{eb}(vmvnq_u{eb}(vceqq_{ns}(a[{i}], b[{i}])))",
                        ns = ns
                    )
                }
            })
            .collect();
        format!("unsafe {{ [{}] }}", items.join(", "))
    };

    let blend_op = || -> String {
        let eb = if elem == "f32" { 32 } else { 64 };
        let items: Vec<String> = (0..sub_count)
            .map(|i| {
                format!(
                    "vbslq_{ns}(vreinterpretq_u{eb}_{ns}(mask[{i}]), if_true[{i}], if_false[{i}])"
                )
            })
            .collect();
        format!("unsafe {{ [{}] }}", items.join(", "))
    };

    let bitwise_not_op = || -> String {
        let eb = if elem == "f32" { 32 } else { 64 };
        let items: Vec<String> = (0..sub_count)
            .map(|i| {
                if eb == 64 {
                    // NEON lacks vmvnq_u64 — use XOR with all-ones
                    format!("vreinterpretq_{ns}_u64(veorq_u64(vreinterpretq_u64_{ns}(a[{i}]), vdupq_n_u64(u64::MAX)))")
                } else {
                    format!("vreinterpretq_{ns}_u{eb}(vmvnq_u{eb}(vreinterpretq_u{eb}_{ns}(a[{i}])))")
                }
            })
            .collect();
        format!("unsafe {{ [{}] }}", items.join(", "))
    };

    let bitwise_binary_op = |neon_op: &str| -> String {
        let eb = if elem == "f32" { 32 } else { 64 };
        let items: Vec<String> = (0..sub_count)
            .map(|i| format!("vreinterpretq_{ns}_u{eb}({neon_op}(vreinterpretq_u{eb}_{ns}(a[{i}]), vreinterpretq_u{eb}_{ns}(b[{i}])))"))
            .collect();
        format!("unsafe {{ [{}] }}", items.join(", "))
    };

    // Reduction: combine halves then reduce
    let reduce_combine = |combine_intrinsic: &str, pairwise: &str| -> String {
        let mut body = String::from("unsafe {\n");
        body.push_str(&format!(
            "            let m = {combine_intrinsic}(a[0], a[1]);\n"
        ));
        // For wider types, combine more
        for i in 2..sub_count {
            body.push_str(&format!(
                "            let m = {combine_intrinsic}(m, a[{i}]);\n"
            ));
        }
        body.push_str(&format!("            let pair = {pairwise}(m, m);\n"));
        if native_lanes > 2 {
            body.push_str(&format!("            let pair = {pairwise}(pair, pair);\n"));
        }
        body.push_str(&format!("            vgetq_lane_{ns}::<0>(pair)\n"));
        body.push_str("        }");
        body
    };

    // Native pixel pack for the f32x8 polyfill ([float32x4_t; 2] → [u8; 8]):
    // round-to-even each half via FCVTNS, narrow i32→i16, combine, then one
    // saturating unsigned narrow i16x8→u8x8. Mirrors the native f32x4 path and
    // the scalar `roundevenf(x).clamp(0,255) as u8` semantics; overrides the
    // scalar trait default LLVM does not recover to FCVTNS.
    let to_u8_arm_poly = if elem == "f32" && lanes == 8 {
        formatdoc! {r#"

            #[inline(always)]
            fn to_u8_bytes(self, a: {repr}) -> [u8; {lanes}] {{
                unsafe {{
                    let i0 = vqmovn_s32(vcvtnq_s32_f32(a[0]));
                    let i1 = vqmovn_s32(vcvtnq_s32_f32(a[1]));
                    let u8s = vqmovun_s16(vcombine_s16(i0, i1));
                    core::mem::transmute(u8s)
                }}
            }}
        "#}
    } else {
        String::new()
    };

    // Native pixel interleave for the f32x8 polyfill: shift-or pack each half
    // ([float32x4_t; 2] -> 2x 16 RGBA bytes). Overrides the to_u8_bytes default.
    let store_rgba_arm_poly = if elem == "f32" && lanes == 8 {
        formatdoc! {r#"

            #[inline(always)]
            fn store_rgba_bytes(self, r: {repr}, g: {repr}, b: {repr}, a: {repr}) -> [u8; 32] {{
                unsafe {{
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
                    core::mem::transmute([vreinterpretq_u8_u32(p0), vreinterpretq_u8_u32(p1)])
                }}
            }}
        "#}
    } else {
        String::new()
    };

    formatdoc! {r#"
        impl {trait_name} for archmage::NeonToken {{
            type Repr = {repr};

            // ====== Construction ======

            #[inline(always)]
            fn splat(self, v: {elem}) -> {repr} {{
                unsafe {{
                    let v4 = vdupq_n_{ns}(v);
                    [{v4_copies}]
                }}
            }}

            #[inline(always)]
            fn zero(self) -> {repr} {{
                unsafe {{
                    let z = vdupq_n_{ns}(0.0);
                    [{z_copies}]
                }}
            }}

            #[inline(always)]
            fn load(self, data: &{array}) -> {repr} {{
                unsafe {{
                    [{load_lanes}]
                }}
            }}

            #[inline(always)]
            fn from_array(self, arr: {array}) -> {repr} {{
                <Self as {trait_name}>::load(self, &arr)
            }}

            #[inline(always)]
            fn store(self, repr: {repr}, out: &mut {array}) {{
                unsafe {{
                    {store_lanes}
                }}
            }}

            #[inline(always)]
            fn to_array(self, repr: {repr}) -> {array} {{
                let mut out = [{zero_lit}; {lanes}];
                <Self as {trait_name}>::store(self, repr, &mut out);
                out
            }}

            // ====== Arithmetic ======

            #[inline(always)]
            fn add(self, a: {repr}, b: {repr}) -> {repr} {{
                {add_body}
            }}

            #[inline(always)]
            fn sub(self, a: {repr}, b: {repr}) -> {repr} {{
                {sub_body}
            }}

            #[inline(always)]
            fn mul(self, a: {repr}, b: {repr}) -> {repr} {{
                {mul_body}
            }}

            #[inline(always)]
            fn div(self, a: {repr}, b: {repr}) -> {repr} {{
                {div_body}
            }}

            #[inline(always)]
            fn neg(self, a: {repr}) -> {repr} {{
                {neg_body}
            }}

            // ====== Math ======

            #[inline(always)]
            fn min(self, a: {repr}, b: {repr}) -> {repr} {{
                {min_body}
            }}

            #[inline(always)]
            fn max(self, a: {repr}, b: {repr}) -> {repr} {{
                {max_body}
            }}

            #[inline(always)]
            fn sqrt(self, a: {repr}) -> {repr} {{
                {sqrt_body}
            }}

            #[inline(always)]
            fn abs(self, a: {repr}) -> {repr} {{
                {abs_body}
            }}

            #[inline(always)]
            fn floor(self, a: {repr}) -> {repr} {{
                {floor_body}
            }}

            #[inline(always)]
            fn ceil(self, a: {repr}) -> {repr} {{
                {ceil_body}
            }}

            #[inline(always)]
            fn round(self, a: {repr}) -> {repr} {{
                {round_body}
            }}

            #[inline(always)]
            fn mul_add(self, a: {repr}, b: {repr}, c: {repr}) -> {repr} {{
                // vfmaq = acc + x*y, so mul_add(a, b, c) = a*b + c => vfmaq(c, a, b)
                {mul_add_body}
            }}

            #[inline(always)]
            fn mul_sub(self, a: {repr}, b: {repr}, c: {repr}) -> {repr} {{
                // a*b - c => vfmaq(-c, a, b) = -c + a*b
                {mul_sub_body}
            }}

            // ====== Comparisons ======

            #[inline(always)]
            fn simd_eq(self, a: {repr}, b: {repr}) -> {repr} {{
                {eq_body}
            }}

            #[inline(always)]
            fn simd_ne(self, a: {repr}, b: {repr}) -> {repr} {{
                {ne_body}
            }}

            #[inline(always)]
            fn simd_lt(self, a: {repr}, b: {repr}) -> {repr} {{
                {lt_body}
            }}

            #[inline(always)]
            fn simd_le(self, a: {repr}, b: {repr}) -> {repr} {{
                {le_body}
            }}

            #[inline(always)]
            fn simd_gt(self, a: {repr}, b: {repr}) -> {repr} {{
                {gt_body}
            }}

            #[inline(always)]
            fn simd_ge(self, a: {repr}, b: {repr}) -> {repr} {{
                {ge_body}
            }}

            #[inline(always)]
            fn blend(self, mask: {repr}, if_true: {repr}, if_false: {repr}) -> {repr} {{
                {blend_body}
            }}

            // ====== Reductions ======

            #[inline(always)]
            fn reduce_add(self, a: {repr}) -> {elem} {{
                {reduce_add_body}
            }}

            #[inline(always)]
            fn reduce_min(self, a: {repr}) -> {elem} {{
                {reduce_min_body}
            }}

            #[inline(always)]
            fn reduce_max(self, a: {repr}) -> {elem} {{
                {reduce_max_body}
            }}

            // ====== Approximations ======
            //
            // Delegate to the native f32x4/f64x2 backend so the polyfill inherits
            // its >=12-bit fused `_approx` (raw vrecpe + 1 FRECPS) and exact full
            // methods — one source of truth for the estimate.
            #[inline(always)]
            fn rcp_approx(self, a: {repr}) -> {repr} {{
                core::array::from_fn(|i| <Self as {sub_trait}>::rcp_approx(self, a[i]))
            }}

            #[inline(always)]
            fn rsqrt_approx(self, a: {repr}) -> {repr} {{
                core::array::from_fn(|i| <Self as {sub_trait}>::rsqrt_approx(self, a[i]))
            }}

            #[inline(always)]
            fn recip(self, a: {repr}) -> {repr} {{
                core::array::from_fn(|i| <Self as {sub_trait}>::recip(self, a[i]))
            }}

            #[inline(always)]
            fn rsqrt(self, a: {repr}) -> {repr} {{
                core::array::from_fn(|i| <Self as {sub_trait}>::rsqrt(self, a[i]))
            }}

            // ====== Bitwise ======

            #[inline(always)]
            fn not(self, a: {repr}) -> {repr} {{
                {not_body}
            }}

            #[inline(always)]
            fn bitand(self, a: {repr}, b: {repr}) -> {repr} {{
                {and_body}
            }}

            #[inline(always)]
            fn bitor(self, a: {repr}, b: {repr}) -> {repr} {{
                {or_body}
            }}

            #[inline(always)]
            fn bitxor(self, a: {repr}, b: {repr}) -> {repr} {{
                {xor_body}
            }}
            {to_u8_arm_poly}
            {store_rgba_arm_poly}
        }}
    "#,
        v4_copies = (0..sub_count).map(|_| "v4").collect::<Vec<_>>().join(", "),
        z_copies = (0..sub_count).map(|_| "z").collect::<Vec<_>>().join(", "),
        load_lanes = (0..sub_count)
            .map(|i| format!("vld1q_{ns}(data.as_ptr().add({}))", i * native_lanes))
            .collect::<Vec<_>>().join(", "),
        store_lanes = (0..sub_count)
            .map(|i| format!("vst1q_{ns}(out.as_mut_ptr().add({}), repr[{i}]);", i * native_lanes))
            .collect::<Vec<_>>().join("\n            "),
        add_body = binary_op(&format!("vaddq_{ns}")),
        sub_body = binary_op(&format!("vsubq_{ns}")),
        mul_body = binary_op(&format!("vmulq_{ns}")),
        div_body = binary_op(&format!("vdivq_{ns}")),
        neg_body = unary_op(&format!("vnegq_{ns}")),
        min_body = binary_op(&format!("vminq_{ns}")),
        max_body = binary_op(&format!("vmaxq_{ns}")),
        sqrt_body = unary_op(&format!("vsqrtq_{ns}")),
        abs_body = unary_op(&format!("vabsq_{ns}")),
        floor_body = unary_op(&format!("vrndmq_{ns}")),
        ceil_body = unary_op(&format!("vrndpq_{ns}")),
        round_body = unary_op(&format!("vrndnq_{ns}")),
        mul_add_body = {
            let items: Vec<String> = (0..sub_count)
                .map(|i| format!("vfmaq_{ns}(c[{i}], a[{i}], b[{i}])"))
                .collect();
            format!("unsafe {{ [{}] }}", items.join(", "))
        },
        mul_sub_body = {
            let items: Vec<String> = (0..sub_count)
                .map(|i| format!("vfmaq_{ns}(vnegq_{ns}(c[{i}]), a[{i}], b[{i}])"))
                .collect();
            format!("unsafe {{ [{}] }}", items.join(", "))
        },
        eq_body = cmp_op(&format!("vceqq_{ns}")),
        ne_body = ne_op(),
        lt_body = cmp_op(&format!("vcltq_{ns}")),
        le_body = cmp_op(&format!("vcleq_{ns}")),
        gt_body = cmp_op(&format!("vcgtq_{ns}")),
        ge_body = cmp_op(&format!("vcgeq_{ns}")),
        blend_body = blend_op(),
        reduce_add_body = reduce_combine(&format!("vaddq_{ns}"), &format!("vpaddq_{ns}")),
        reduce_min_body = reduce_combine(&format!("vminq_{ns}"), &format!("vpminq_{ns}")),
        reduce_max_body = reduce_combine(&format!("vmaxq_{ns}"), &format!("vpmaxq_{ns}")),
        sub_trait = format!("F{}x{}Backend", &elem[1..], if elem == "f32" { 4 } else { 2 }),
        not_body = bitwise_not_op(),
        and_body = bitwise_binary_op(&format!("vandq_u{}", if elem == "f32" { 32 } else { 64 })),
        or_body = bitwise_binary_op(&format!("vorrq_u{}", if elem == "f32" { 32 } else { 64 })),
        xor_body = bitwise_binary_op(&format!("veorq_u{}", if elem == "f32" { 32 } else { 64 })),
    }
}

/// Generate NEON impl for a type that's native 128-bit (f32x4, f64x2).
fn generate_neon_native_impl(ty: &FloatVecType) -> String {
    let trait_name = ty.trait_name();
    let repr = ty.neon_repr();
    let elem = ty.elem;
    let lanes = ty.lanes;
    let array = ty.array_type();
    let ns = ty.neon_suffix();
    let zero_lit = if elem == "f32" { "0.0f32" } else { "0.0f64" };
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

    // f64 (53-bit mantissa) needs 3 Newton steps from the 8-bit NEON estimate to
    // reach full precision; f32 (24-bit) needs 2.
    let n_nr_steps = if elem == "f64" { 3 } else { 2 };
    // Last step is a tail expression (no `let`, so no clippy::let_and_return).
    let recip_nr_steps = (0..n_nr_steps)
        .map(|i| {
            let step = format!("vmulq_{ns}(vrecpsq_{ns}(a, y), y)");
            if i + 1 < n_nr_steps {
                format!("let y = {step};")
            } else {
                step
            }
        })
        .collect::<Vec<_>>()
        .join("\n                    ");
    let rsqrt_nr_steps = (0..n_nr_steps)
        .map(|i| {
            let step = format!("vmulq_{ns}(vrsqrtsq_{ns}(vmulq_{ns}(a, y), y), y)");
            if i + 1 < n_nr_steps {
                format!("let y = {step};")
            } else {
                step
            }
        })
        .collect::<Vec<_>>()
        .join("\n                    ");

    // For native types, reduce pattern is different
    let reduce_pairwise = |pairwise: &str| -> String {
        let mut body = format!("unsafe {{\n            let pair = {pairwise}(a, a);\n");
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

            #[inline(always)]
            fn to_u8_bytes(self, a: {repr}) -> [u8; 4] {{
                unsafe {{
                    let i16s = vqmovn_s32(vcvtnq_s32_f32(a));
                    let u8s = vqmovun_s16(vcombine_s16(i16s, i16s));
                    let bytes: [u8; 8] = core::mem::transmute(u8s);
                    [bytes[0], bytes[1], bytes[2], bytes[3]]
                }}
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

            #[inline(always)]
            fn store_rgba_bytes(self, r: {repr}, g: {repr}, b: {repr}, a: {repr}) -> [u8; 16] {{
                unsafe {{
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
                    core::mem::transmute(vreinterpretq_u8_u32(pixels))
                }}
            }}
        "#}
    } else {
        String::new()
    };

    formatdoc! {r#"
        impl {trait_name} for archmage::NeonToken {{
            type Repr = {repr};

            #[inline(always)]
            fn splat(self, v: {elem}) -> {repr} {{
                unsafe {{ vdupq_n_{ns}(v) }}
            }}

            #[inline(always)]
            fn zero(self) -> {repr} {{
                unsafe {{ vdupq_n_{ns}(0.0) }}
            }}

            #[inline(always)]
            fn load(self, data: &{array}) -> {repr} {{
                unsafe {{ vld1q_{ns}(data.as_ptr()) }}
            }}

            #[inline(always)]
            fn from_array(self, arr: {array}) -> {repr} {{
                <Self as {trait_name}>::load(self, &arr)
            }}

            #[inline(always)]
            fn store(self, repr: {repr}, out: &mut {array}) {{
                unsafe {{ vst1q_{ns}(out.as_mut_ptr(), repr) }};
            }}

            #[inline(always)]
            fn to_array(self, repr: {repr}) -> {array} {{
                let mut out = [{zero_lit}; {lanes}];
                <Self as {trait_name}>::store(self, repr, &mut out);
                out
            }}

            #[inline(always)]
            fn add(self, a: {repr}, b: {repr}) -> {repr} {{ unsafe {{ vaddq_{ns}(a, b) }} }}
            #[inline(always)]
            fn sub(self, a: {repr}, b: {repr}) -> {repr} {{ unsafe {{ vsubq_{ns}(a, b) }} }}
            #[inline(always)]
            fn mul(self, a: {repr}, b: {repr}) -> {repr} {{ unsafe {{ vmulq_{ns}(a, b) }} }}
            #[inline(always)]
            fn div(self, a: {repr}, b: {repr}) -> {repr} {{ unsafe {{ vdivq_{ns}(a, b) }} }}
            #[inline(always)]
            fn neg(self, a: {repr}) -> {repr} {{ unsafe {{ vnegq_{ns}(a) }} }}
            #[inline(always)]
            fn min(self, a: {repr}, b: {repr}) -> {repr} {{ unsafe {{ vminq_{ns}(a, b) }} }}
            #[inline(always)]
            fn max(self, a: {repr}, b: {repr}) -> {repr} {{ unsafe {{ vmaxq_{ns}(a, b) }} }}
            #[inline(always)]
            fn sqrt(self, a: {repr}) -> {repr} {{ unsafe {{ vsqrtq_{ns}(a) }} }}
            #[inline(always)]
            fn abs(self, a: {repr}) -> {repr} {{ unsafe {{ vabsq_{ns}(a) }} }}
            #[inline(always)]
            fn floor(self, a: {repr}) -> {repr} {{ unsafe {{ vrndmq_{ns}(a) }} }}
            #[inline(always)]
            fn ceil(self, a: {repr}) -> {repr} {{ unsafe {{ vrndpq_{ns}(a) }} }}
            #[inline(always)]
            fn round(self, a: {repr}) -> {repr} {{ unsafe {{ vrndnq_{ns}(a) }} }}

            #[inline(always)]
            fn mul_add(self, a: {repr}, b: {repr}, c: {repr}) -> {repr} {{
                unsafe {{ vfmaq_{ns}(c, a, b) }}
            }}

            #[inline(always)]
            fn mul_sub(self, a: {repr}, b: {repr}, c: {repr}) -> {repr} {{
                unsafe {{ vfmaq_{ns}(vnegq_{ns}(c), a, b) }}
            }}

            #[inline(always)]
            fn simd_eq(self, a: {repr}, b: {repr}) -> {repr} {{
                unsafe {{ vreinterpretq_{ns}_u{eb}(vceqq_{ns}(a, b)) }}
            }}
            #[inline(always)]
            fn simd_ne(self, a: {repr}, b: {repr}) -> {repr} {{
                unsafe {{ vreinterpretq_{ns}_u{eb}({ne_inner}) }}
            }}
            #[inline(always)]
            fn simd_lt(self, a: {repr}, b: {repr}) -> {repr} {{
                unsafe {{ vreinterpretq_{ns}_u{eb}(vcltq_{ns}(a, b)) }}
            }}
            #[inline(always)]
            fn simd_le(self, a: {repr}, b: {repr}) -> {repr} {{
                unsafe {{ vreinterpretq_{ns}_u{eb}(vcleq_{ns}(a, b)) }}
            }}
            #[inline(always)]
            fn simd_gt(self, a: {repr}, b: {repr}) -> {repr} {{
                unsafe {{ vreinterpretq_{ns}_u{eb}(vcgtq_{ns}(a, b)) }}
            }}
            #[inline(always)]
            fn simd_ge(self, a: {repr}, b: {repr}) -> {repr} {{
                unsafe {{ vreinterpretq_{ns}_u{eb}(vcgeq_{ns}(a, b)) }}
            }}

            #[inline(always)]
            fn blend(self, mask: {repr}, if_true: {repr}, if_false: {repr}) -> {repr} {{
                unsafe {{ vbslq_{ns}(vreinterpretq_u{eb}_{ns}(mask), if_true, if_false) }}
            }}

            #[inline(always)]
            fn reduce_add(self, a: {repr}) -> {elem} {{
                {reduce_add}
            }}

            #[inline(always)]
            fn reduce_min(self, a: {repr}) -> {elem} {{
                {reduce_min}
            }}

            #[inline(always)]
            fn reduce_max(self, a: {repr}) -> {elem} {{
                {reduce_max}
            }}

            // FRECPS (`vrecpsq`) computes `2 - a*y` and FRSQRTS (`vrsqrtsq`)
            // computes `(3 - a*y*y)/2` as one fused step each — no intermediate
            // rounding, no 2.0/3.0/0.5 splats, measurably faster than hand-rolled
            // mul/sub on real silicon (see `benchmarks/rsqrt_arm_neoverse-n1`).
            // Each step roughly doubles the correct bits (~8 → ~16 → ~24).
            // `_approx` = raw vrecpe/vrsqrte (~8-bit) + one fused FRECPS/FRSQRTS
            // step (~16-bit): the >=12-bit fast path. `recip`/`rsqrt` refine the
            // raw estimate directly with 2 (f32) / 3 (f64) fused steps — starting
            // from the raw estimate, NOT from `_approx`, so its built-in step is
            // not double-counted. FRECPS/FRSQRTS are baseline NEON.
            #[inline(always)]
            fn rcp_approx(self, a: {repr}) -> {repr} {{
                unsafe {{ let y = vrecpeq_{ns}(a); vmulq_{ns}(vrecpsq_{ns}(a, y), y) }}
            }}
            #[inline(always)]
            fn rsqrt_approx(self, a: {repr}) -> {repr} {{
                unsafe {{ let y = vrsqrteq_{ns}(a); vmulq_{ns}(vrsqrtsq_{ns}(vmulq_{ns}(a, y), y), y) }}
            }}
            #[inline(always)]
            fn recip(self, a: {repr}) -> {repr} {{
                unsafe {{
                    let y = vrecpeq_{ns}(a);
                    {recip_nr_steps}
                }}
            }}
            #[inline(always)]
            fn rsqrt(self, a: {repr}) -> {repr} {{
                unsafe {{
                    let y = vrsqrteq_{ns}(a);
                    {rsqrt_nr_steps}
                }}
            }}

            #[inline(always)]
            fn not(self, a: {repr}) -> {repr} {{
                unsafe {{ vreinterpretq_{ns}_u{eb}({not_inner}) }}
            }}
            #[inline(always)]
            fn bitand(self, a: {repr}, b: {repr}) -> {repr} {{
                unsafe {{ vreinterpretq_{ns}_u{eb}(vandq_u{eb}(vreinterpretq_u{eb}_{ns}(a), vreinterpretq_u{eb}_{ns}(b))) }}
            }}
            #[inline(always)]
            fn bitor(self, a: {repr}, b: {repr}) -> {repr} {{
                unsafe {{ vreinterpretq_{ns}_u{eb}(vorrq_u{eb}(vreinterpretq_u{eb}_{ns}(a), vreinterpretq_u{eb}_{ns}(b))) }}
            }}
            #[inline(always)]
            fn bitxor(self, a: {repr}, b: {repr}) -> {repr} {{
                unsafe {{ vreinterpretq_{ns}_u{eb}(veorq_u{eb}(vreinterpretq_u{eb}_{ns}(a), vreinterpretq_u{eb}_{ns}(b))) }}
            }}
            {to_u8_arm}
            {store_rgba_arm}
        }}
    "#,
        reduce_add = reduce_pairwise(&format!("vpaddq_{ns}")),
        reduce_min = reduce_pairwise(&format!("vpminq_{ns}")),
        reduce_max = reduce_pairwise(&format!("vpmaxq_{ns}")),
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
        code.push_str("#[cfg(target_arch = \"wasm32\")]\n");
        code.push_str(&generate_wasm_float_impl(ty));
        code.push('\n');
    }

    code
}

fn generate_wasm_float_impl(ty: &FloatVecType) -> String {
    let trait_name = ty.trait_name();
    let repr = ty.wasm_repr();
    let elem = ty.elem;
    let lanes = ty.lanes;
    let array = ty.array_type();
    let wp = ty.wasm_prefix(); // "f32x4" or "f64x2"
    let zero_lit = if elem == "f32" { "0.0f32" } else { "0.0f64" };
    let sub_count = ty.sub_count();
    let native_lanes = if elem == "f32" { 4 } else { 2 };

    if ty.native_on_wasm() {
        return generate_wasm_native_impl(ty);
    }

    // Polyfill: apply operation to each sub-vector
    let binary_op = |func: &str| -> String {
        let items: Vec<String> = (0..sub_count)
            .map(|i| format!("{func}(a[{i}], b[{i}])"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let unary_op = |func: &str| -> String {
        let items: Vec<String> = (0..sub_count).map(|i| format!("{func}(a[{i}])")).collect();
        format!("[{}]", items.join(", "))
    };

    // Cross-block add then adjacent-pair per-block tree (#50). For f32
    // this matches the NEON f32x8 shape: `vaddq(a[0],a[1])` → `vpaddq×2`.
    // Fewer total ops and consistent tree.
    let reduce_add_body = || -> String {
        if elem == "f32" && sub_count >= 2 {
            let mut body = format!("let m = {wp}_add(a[0], a[1]);\n");
            for i in 2..sub_count {
                body.push_str(&format!("        let m = {wp}_add(m, a[{i}]);\n"));
            }
            body.push_str(&format!("        ({wp}_extract_lane::<0>(m) + {wp}_extract_lane::<1>(m)) + ({wp}_extract_lane::<2>(m) + {wp}_extract_lane::<3>(m))"));
            body
        } else {
            let mut items = Vec::new();
            for i in 0..sub_count {
                for j in 0..native_lanes {
                    items.push(format!("{wp}_extract_lane::<{j}>(a[{i}])"));
                }
            }
            items.join("\n            + ")
        }
    };

    let reduce_minmax = |combine: &str, fold_method: &str| -> String {
        let mut body = format!("let m = {combine}(a[0], a[1]);\n");
        for i in 2..sub_count {
            body.push_str(&format!("        let m = {combine}(m, a[{i}]);\n"));
        }
        let extracts: Vec<String> = (0..native_lanes)
            .map(|j| format!("let v{j} = {wp}_extract_lane::<{j}>(m);"))
            .collect();
        body.push_str(&format!("        {}\n", extracts.join("\n        ")));
        // Build fold
        let fold: String = if native_lanes == 4 {
            format!("v0.{fold_method}(v1).{fold_method}(v2.{fold_method}(v3))")
        } else {
            format!("v0.{fold_method}(v1)")
        };
        body.push_str(&format!("        {fold}"));
        body
    };

    // Reciprocals delegate to the native 128-bit sub-backend so the polyfill
    // inherits its per-platform `_approx` (the f32 bit-hack) and exact full
    // methods — no second copy of the estimate to keep in sync.
    let sub_trait = format!("F{}x{}Backend", &elem[1..], native_lanes);
    formatdoc! {r#"
        impl {trait_name} for archmage::Wasm128Token {{
            type Repr = {repr};

            #[inline(always)]
            fn splat(self, v: {elem}) -> {repr} {{
                let v4 = {wp}_splat(v);
                [{v4_copies}]
            }}

            #[inline(always)]
            fn zero(self) -> {repr} {{
                let z = {wp}_splat(0.0);
                [{z_copies}]
            }}

            #[inline(always)]
            fn load(self, data: &{array}) -> {repr} {{
                unsafe {{
                    [{load_lanes}]
                }}
            }}

            #[inline(always)]
            fn from_array(self, arr: {array}) -> {repr} {{
                <Self as {trait_name}>::load(self, &arr)
            }}

            #[inline(always)]
            fn store(self, repr: {repr}, out: &mut {array}) {{
                unsafe {{
                    {store_lanes}
                }}
            }}

            #[inline(always)]
            fn to_array(self, repr: {repr}) -> {array} {{
                let mut out = [{zero_lit}; {lanes}];
                <Self as {trait_name}>::store(self, repr, &mut out);
                out
            }}

            #[inline(always)]
            fn add(self, a: {repr}, b: {repr}) -> {repr} {{ {add} }}
            #[inline(always)]
            fn sub(self, a: {repr}, b: {repr}) -> {repr} {{ {sub} }}
            #[inline(always)]
            fn mul(self, a: {repr}, b: {repr}) -> {repr} {{ {mul} }}
            #[inline(always)]
            fn div(self, a: {repr}, b: {repr}) -> {repr} {{ {div} }}
            #[inline(always)]
            fn neg(self, a: {repr}) -> {repr} {{ {neg} }}
            #[inline(always)]
            fn min(self, a: {repr}, b: {repr}) -> {repr} {{ {min} }}
            #[inline(always)]
            fn max(self, a: {repr}, b: {repr}) -> {repr} {{ {max} }}
            #[inline(always)]
            fn sqrt(self, a: {repr}) -> {repr} {{ {sqrt} }}
            #[inline(always)]
            fn abs(self, a: {repr}) -> {repr} {{ {abs} }}
            #[inline(always)]
            fn floor(self, a: {repr}) -> {repr} {{ {floor} }}
            #[inline(always)]
            fn ceil(self, a: {repr}) -> {repr} {{ {ceil} }}
            #[inline(always)]
            fn round(self, a: {repr}) -> {repr} {{ {round} }}

            #[inline(always)]
            fn mul_add(self, a: {repr}, b: {repr}, c: {repr}) -> {repr} {{
                // WASM has no native FMA
                [{mul_add_lanes}]
            }}

            #[inline(always)]
            fn mul_sub(self, a: {repr}, b: {repr}, c: {repr}) -> {repr} {{
                [{mul_sub_lanes}]
            }}

            #[inline(always)]
            fn simd_eq(self, a: {repr}, b: {repr}) -> {repr} {{ {eq} }}
            #[inline(always)]
            fn simd_ne(self, a: {repr}, b: {repr}) -> {repr} {{ {ne} }}
            #[inline(always)]
            fn simd_lt(self, a: {repr}, b: {repr}) -> {repr} {{ {lt} }}
            #[inline(always)]
            fn simd_le(self, a: {repr}, b: {repr}) -> {repr} {{ {le} }}
            #[inline(always)]
            fn simd_gt(self, a: {repr}, b: {repr}) -> {repr} {{ {gt} }}
            #[inline(always)]
            fn simd_ge(self, a: {repr}, b: {repr}) -> {repr} {{ {ge} }}

            #[inline(always)]
            fn blend(self, mask: {repr}, if_true: {repr}, if_false: {repr}) -> {repr} {{
                [{blend_lanes}]
            }}

            #[inline(always)]
            fn reduce_add(self, a: {repr}) -> {elem} {{
                {reduce_add}
            }}

            #[inline(always)]
            fn reduce_min(self, a: {repr}) -> {elem} {{
                {reduce_min}
            }}

            #[inline(always)]
            fn reduce_max(self, a: {repr}) -> {elem} {{
                {reduce_max}
            }}

            #[inline(always)]
            fn rcp_approx(self, a: {repr}) -> {repr} {{
                core::array::from_fn(|i| <Self as {sub_trait}>::rcp_approx(self, a[i]))
            }}

            #[inline(always)]
            fn rsqrt_approx(self, a: {repr}) -> {repr} {{
                core::array::from_fn(|i| <Self as {sub_trait}>::rsqrt_approx(self, a[i]))
            }}

            #[inline(always)]
            fn recip(self, a: {repr}) -> {repr} {{
                core::array::from_fn(|i| <Self as {sub_trait}>::recip(self, a[i]))
            }}

            #[inline(always)]
            fn rsqrt(self, a: {repr}) -> {repr} {{
                core::array::from_fn(|i| <Self as {sub_trait}>::rsqrt(self, a[i]))
            }}

            #[inline(always)]
            fn not(self, a: {repr}) -> {repr} {{ {not} }}
            #[inline(always)]
            fn bitand(self, a: {repr}, b: {repr}) -> {repr} {{ {and} }}
            #[inline(always)]
            fn bitor(self, a: {repr}, b: {repr}) -> {repr} {{ {or} }}
            #[inline(always)]
            fn bitxor(self, a: {repr}, b: {repr}) -> {repr} {{ {xor} }}
        }}
    "#,
        v4_copies = (0..sub_count).map(|_| "v4").collect::<Vec<_>>().join(", "),
        z_copies = (0..sub_count).map(|_| "z").collect::<Vec<_>>().join(", "),
        load_lanes = (0..sub_count)
            .map(|i| format!("v128_load(data.as_ptr().add({}).cast())", i * native_lanes))
            .collect::<Vec<_>>().join(", "),
        store_lanes = (0..sub_count)
            .map(|i| format!("v128_store(out.as_mut_ptr().add({}).cast(), repr[{i}]);", i * native_lanes))
            .collect::<Vec<_>>().join("\n            "),
        add = binary_op(&format!("{wp}_add")),
        sub = binary_op(&format!("{wp}_sub")),
        mul = binary_op(&format!("{wp}_mul")),
        div = binary_op(&format!("{wp}_div")),
        neg = unary_op(&format!("{wp}_neg")),
        min = binary_op(&format!("{wp}_min")),
        max = binary_op(&format!("{wp}_max")),
        sqrt = unary_op(&format!("{wp}_sqrt")),
        abs = unary_op(&format!("{wp}_abs")),
        floor = unary_op(&format!("{wp}_floor")),
        ceil = unary_op(&format!("{wp}_ceil")),
        round = unary_op(&format!("{wp}_nearest")),
        mul_add_lanes = (0..sub_count)
            .map(|i| format!("{wp}_add({wp}_mul(a[{i}], b[{i}]), c[{i}])"))
            .collect::<Vec<_>>().join(", "),
        mul_sub_lanes = (0..sub_count)
            .map(|i| format!("{wp}_sub({wp}_mul(a[{i}], b[{i}]), c[{i}])"))
            .collect::<Vec<_>>().join(", "),
        eq = binary_op(&format!("{wp}_eq")),
        ne = binary_op(&format!("{wp}_ne")),
        lt = binary_op(&format!("{wp}_lt")),
        le = binary_op(&format!("{wp}_le")),
        gt = binary_op(&format!("{wp}_gt")),
        ge = binary_op(&format!("{wp}_ge")),
        blend_lanes = (0..sub_count)
            .map(|i| format!("v128_bitselect(if_true[{i}], if_false[{i}], mask[{i}])"))
            .collect::<Vec<_>>().join(", "),
        reduce_add = reduce_add_body(),
        reduce_min = reduce_minmax(&format!("{wp}_min"), "min"),
        reduce_max = reduce_minmax(&format!("{wp}_max"), "max"),
        not = unary_op("v128_not"),
        and = binary_op("v128_and"),
        or = binary_op("v128_or"),
        xor = binary_op("v128_xor"),
    }
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
            fn load(self, data: &{array}) -> v128 {{ unsafe {{ v128_load(data.as_ptr().cast()) }} }}
            #[inline(always)]
            fn from_array(self, arr: {array}) -> v128 {{ unsafe {{ v128_load(arr.as_ptr().cast()) }} }}
            #[inline(always)]
            fn store(self, repr: v128, out: &mut {array}) {{ unsafe {{ v128_store(out.as_mut_ptr().cast(), repr) }}; }}
            #[inline(always)]
            fn to_array(self, repr: v128) -> {array} {{
                let mut out = [{zero_lit}; {lanes}];
                unsafe {{ v128_store(out.as_mut_ptr().cast(), repr) }};
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
            fn min(self, a: v128, b: v128) -> v128 {{ {wp}_min(a, b) }}
            #[inline(always)]
            fn max(self, a: v128, b: v128) -> v128 {{ {wp}_max(a, b) }}
            #[inline(always)]
            fn sqrt(self, a: v128) -> v128 {{ {wp}_sqrt(a) }}
            #[inline(always)]
            fn abs(self, a: v128) -> v128 {{ {wp}_abs(a) }}
            #[inline(always)]
            fn floor(self, a: v128) -> v128 {{ {wp}_floor(a) }}
            #[inline(always)]
            fn ceil(self, a: v128) -> v128 {{ {wp}_ceil(a) }}
            #[inline(always)]
            fn round(self, a: v128) -> v128 {{ {wp}_nearest(a) }}
            #[inline(always)]
            fn mul_add(self, a: v128, b: v128, c: v128) -> v128 {{ {wp}_add({wp}_mul(a, b), c) }}
            #[inline(always)]
            fn mul_sub(self, a: v128, b: v128, c: v128) -> v128 {{ {wp}_sub({wp}_mul(a, b), c) }}
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
        }}
    "#,
        reduce_add = reduce_add_body(),
        reduce_min = reduce_minmax("min"),
        reduce_max = reduce_minmax("max"),
    }
}

// ============================================================================
// I32 Backend Trait Definition Generation
// ============================================================================

fn generate_i32_backend_trait(ty: &I32VecType) -> String {
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
            type Repr: Copy + Clone + Send + Sync;

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

            // ====== Boolean ======

            /// True if all lanes have their sign bit set (all-1s mask).
            fn all_true(self, a: Self::Repr) -> bool;

            /// True if any lane has its sign bit set (any all-1s mask lane).
            fn any_true(self, a: Self::Repr) -> bool;

            /// Extract the high bit of each 32-bit lane as a bitmask.
            fn bitmask(self, a: Self::Repr) -> u32;

            // ====== Default implementations ======

            /// Clamp values between lo and hi.
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
        code.push_str("\n#[cfg(target_arch = \"x86_64\")]\n");
        code.push_str(&generate_x86_i32_impl(ty, token));
        code.push('\n');
    }

    code
}

fn generate_x86_i32_impl(ty: &I32VecType, token: &str) -> String {
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

            #[inline(always)]
            fn splat(self, v: i32) -> {inner} {{
                unsafe {{ {p}_set1_epi32(v) }}
            }}

            #[inline(always)]
            fn zero(self) -> {inner} {{
                unsafe {{ {p}_setzero_si{bits}() }}
            }}

            #[inline(always)]
            fn load(self, data: &{array}) -> {inner} {{
                unsafe {{ {p}_loadu_si{bits}(data.as_ptr().cast()) }}
            }}

            #[inline(always)]
            fn from_array(self, arr: {array}) -> {inner} {{
                // SAFETY: {array} and {inner} have identical size and layout.
                unsafe {{ core::mem::transmute(arr) }}
            }}

            #[inline(always)]
            fn store(self, repr: {inner}, out: &mut {array}) {{
                unsafe {{ {p}_storeu_si{bits}(out.as_mut_ptr().cast(), repr) }};
            }}

            #[inline(always)]
            fn to_array(self, repr: {inner}) -> {array} {{
                let mut out = [0i32; {lanes}];
                unsafe {{ {p}_storeu_si{bits}(out.as_mut_ptr().cast(), repr) }};
                out
            }}

            // ====== Arithmetic ======

            #[inline(always)]
            fn add(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_add_epi32(a, b) }}
            }}

            #[inline(always)]
            fn sub(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_sub_epi32(a, b) }}
            }}

            #[inline(always)]
            fn mul(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_mullo_epi32(a, b) }}
            }}

            #[inline(always)]
            fn neg(self, a: {inner}) -> {inner} {{
                unsafe {{ {p}_sub_epi32({p}_setzero_si{bits}(), a) }}
            }}

            // ====== Math ======

            #[inline(always)]
            fn min(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_min_epi32(a, b) }}
            }}

            #[inline(always)]
            fn max(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_max_epi32(a, b) }}
            }}

            #[inline(always)]
            fn abs(self, a: {inner}) -> {inner} {{
                unsafe {{ {p}_abs_epi32(a) }}
            }}

            // ====== Comparisons ======

            #[inline(always)]
            fn simd_eq(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_cmpeq_epi32(a, b) }}
            }}

            #[inline(always)]
            fn simd_ne(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{
                    let eq = {p}_cmpeq_epi32(a, b);
                    {p}_andnot_si{bits}(eq, {p}_set1_epi32(-1))
                }}
            }}

            #[inline(always)]
            fn simd_lt(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_cmpgt_epi32(b, a) }}
            }}

            #[inline(always)]
            fn simd_le(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{
                    let gt = {p}_cmpgt_epi32(a, b);
                    {p}_andnot_si{bits}(gt, {p}_set1_epi32(-1))
                }}
            }}

            #[inline(always)]
            fn simd_gt(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_cmpgt_epi32(a, b) }}
            }}

            #[inline(always)]
            fn simd_ge(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{
                    let lt = {p}_cmpgt_epi32(b, a);
                    {p}_andnot_si{bits}(lt, {p}_set1_epi32(-1))
                }}
            }}

            #[inline(always)]
            fn blend(self, mask: {inner}, if_true: {inner}, if_false: {inner}) -> {inner} {{
                unsafe {{ {p}_blendv_epi8(if_false, if_true, mask) }}
            }}

            // ====== Reductions ======

            #[inline(always)]
            fn reduce_add(self, a: {inner}) -> i32 {{
        {reduce_add_body}
            }}

            // ====== Bitwise ======

            #[inline(always)]
            fn not(self, a: {inner}) -> {inner} {{
                unsafe {{ {p}_andnot_si{bits}(a, {p}_set1_epi32(-1)) }}
            }}

            #[inline(always)]
            fn bitand(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_and_si{bits}(a, b) }}
            }}

            #[inline(always)]
            fn bitor(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_or_si{bits}(a, b) }}
            }}

            #[inline(always)]
            fn bitxor(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_xor_si{bits}(a, b) }}
            }}

            // ====== Shifts ======

            #[inline(always)]
            fn shl_const<const N: i32>(self, a: {inner}) -> {inner} {{
                unsafe {{ {p}_slli_epi32::<N>(a) }}
            }}

            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(self, a: {inner}) -> {inner} {{
                unsafe {{ {p}_srai_epi32::<N>(a) }}
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(self, a: {inner}) -> {inner} {{
                unsafe {{ {p}_srli_epi32::<N>(a) }}
            }}

            // ====== Boolean ======

            #[inline(always)]
            fn all_true(self, a: {inner}) -> bool {{
                unsafe {{ {p}_movemask_ps({p}_castsi{bits}_ps(a)) == {all_mask} }}
            }}

            #[inline(always)]
            fn any_true(self, a: {inner}) -> bool {{
                unsafe {{ {p}_movemask_ps({p}_castsi{bits}_ps(a)) != 0 }}
            }}

            #[inline(always)]
            fn bitmask(self, a: {inner}) -> u32 {{
                unsafe {{ {p}_movemask_ps({p}_castsi{bits}_ps(a)) as u32 }}
            }}
        }}
    "#,
        all_mask = if lanes == 4 { "0xF" } else { "0xFF" },
    }
}

fn generate_x86_i32_reduce_add(ty: &I32VecType) -> String {
    match ty.width_bits {
        128 => formatdoc! {"
                unsafe {{
                    let hi = _mm_shuffle_epi32::<0b01_00_11_10>(a);
                    let sum = _mm_add_epi32(a, hi);
                    let hi2 = _mm_shuffle_epi32::<0b00_00_00_01>(sum);
                    let sum2 = _mm_add_epi32(sum, hi2);
                    _mm_cvtsi128_si32(sum2)
                }}"},
        256 => formatdoc! {"
                unsafe {{
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
    formatdoc! {r#"

        #[cfg(target_arch = "x86_64")]
        impl F32x4Convert for archmage::{token} {{
            #[inline(always)]
            fn bitcast_f32_to_i32(self, a: __m128) -> __m128i {{
                unsafe {{ _mm_castps_si128(a) }}
            }}

            #[inline(always)]
            fn bitcast_i32_to_f32(self, a: __m128i) -> __m128 {{
                unsafe {{ _mm_castsi128_ps(a) }}
            }}

            #[inline(always)]
            fn convert_f32_to_i32(self, a: __m128) -> __m128i {{
                unsafe {{ _mm_cvttps_epi32(a) }}
            }}

            #[inline(always)]
            fn convert_f32_to_i32_round(self, a: __m128) -> __m128i {{
                unsafe {{ _mm_cvtps_epi32(a) }}
            }}

            #[inline(always)]
            fn convert_i32_to_f32(self, a: __m128i) -> __m128 {{
                unsafe {{ _mm_cvtepi32_ps(a) }}
            }}
        }}

        #[cfg(target_arch = "x86_64")]
        impl F32x8Convert for archmage::{token} {{
            #[inline(always)]
            fn bitcast_f32_to_i32(self, a: __m256) -> __m256i {{
                unsafe {{ _mm256_castps_si256(a) }}
            }}

            #[inline(always)]
            fn bitcast_i32_to_f32(self, a: __m256i) -> __m256 {{
                unsafe {{ _mm256_castsi256_ps(a) }}
            }}

            #[inline(always)]
            fn convert_f32_to_i32(self, a: __m256) -> __m256i {{
                unsafe {{ _mm256_cvttps_epi32(a) }}
            }}

            #[inline(always)]
            fn convert_f32_to_i32_round(self, a: __m256) -> __m256i {{
                unsafe {{ _mm256_cvtps_epi32(a) }}
            }}

            #[inline(always)]
            fn convert_i32_to_f32(self, a: __m256i) -> __m256 {{
                unsafe {{ _mm256_cvtepi32_ps(a) }}
            }}
        }}

        #[cfg(all(target_arch = "x86_64", feature = "w512"))]
        impl F32x16Convert for archmage::{token} {{
            #[inline(always)]
            fn bitcast_f32_to_i32(self, a: [__m256; 2]) -> [__m256i; 2] {{
                unsafe {{ [_mm256_castps_si256(a[0]), _mm256_castps_si256(a[1])] }}
            }}

            #[inline(always)]
            fn bitcast_i32_to_f32(self, a: [__m256i; 2]) -> [__m256; 2] {{
                unsafe {{ [_mm256_castsi256_ps(a[0]), _mm256_castsi256_ps(a[1])] }}
            }}

            #[inline(always)]
            fn convert_f32_to_i32(self, a: [__m256; 2]) -> [__m256i; 2] {{
                unsafe {{ [_mm256_cvttps_epi32(a[0]), _mm256_cvttps_epi32(a[1])] }}
            }}

            #[inline(always)]
            fn convert_f32_to_i32_round(self, a: [__m256; 2]) -> [__m256i; 2] {{
                unsafe {{ [_mm256_cvtps_epi32(a[0]), _mm256_cvtps_epi32(a[1])] }}
            }}

            #[inline(always)]
            fn convert_i32_to_f32(self, a: [__m256i; 2]) -> [__m256; 2] {{
                unsafe {{ [_mm256_cvtepi32_ps(a[0]), _mm256_cvtepi32_ps(a[1])] }}
            }}
        }}

        #[cfg(target_arch = "x86_64")]
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

        #[cfg(target_arch = "x86_64")]
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

        #[cfg(target_arch = "x86_64")]
        impl I64x2Bitcast for archmage::{token} {{
            #[inline(always)]
            fn bitcast_i64_to_f64(self, a: __m128i) -> __m128d {{
                unsafe {{ _mm_castsi128_pd(a) }}
            }}

            #[inline(always)]
            fn bitcast_f64_to_i64(self, a: __m128d) -> __m128i {{
                unsafe {{ _mm_castpd_si128(a) }}
            }}
        }}

        #[cfg(target_arch = "x86_64")]
        impl I64x4Bitcast for archmage::{token} {{
            #[inline(always)]
            fn bitcast_i64_to_f64(self, a: __m256i) -> __m256d {{
                unsafe {{ _mm256_castsi256_pd(a) }}
            }}

            #[inline(always)]
            fn bitcast_f64_to_i64(self, a: __m256d) -> __m256i {{
                unsafe {{ _mm256_castpd_si256(a) }}
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
    let trait_name = ty.trait_name();
    let lanes = ty.lanes;
    let array = ty.array_type();

    // For infix operators like &, |, ^
    let binary_infix = |op: &str| -> String {
        let items: Vec<String> = (0..lanes).map(|i| format!("a[{i}] {op} b[{i}]")).collect();
        format!("[{}]", items.join(", "))
    };

    // For method-style ops like .wrapping_add(b)
    let binary_method = |method: &str| -> String {
        let items: Vec<String> = (0..lanes)
            .map(|i| format!("a[{i}].{method}(b[{i}])"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let min_lanes = || -> String {
        let items: Vec<String> = (0..lanes)
            .map(|i| format!("if a[{i}] < b[{i}] {{ a[{i}] }} else {{ b[{i}] }}"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let max_lanes = || -> String {
        let items: Vec<String> = (0..lanes)
            .map(|i| format!("if a[{i}] > b[{i}] {{ a[{i}] }} else {{ b[{i}] }}"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let neg_lanes = || -> String {
        let items: Vec<String> = (0..lanes)
            .map(|i| format!("a[{i}].wrapping_neg()"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let abs_lanes = || -> String {
        let items: Vec<String> = (0..lanes)
            .map(|i| format!("a[{i}].wrapping_abs()"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let reduce_add = || -> String {
        let items: Vec<String> = (0..lanes).map(|i| format!("a[{i}]")).collect();
        items.join(".wrapping_add(") + &")".repeat(lanes - 1)
    };

    let shl_lanes = || -> String {
        let items: Vec<String> = (0..lanes).map(|i| format!("a[{i}] << N")).collect();
        format!("[{}]", items.join(", "))
    };

    let shr_arithmetic_lanes = || -> String {
        let items: Vec<String> = (0..lanes).map(|i| format!("a[{i}] >> N")).collect();
        format!("[{}]", items.join(", "))
    };

    let shr_logical_lanes = || -> String {
        let items: Vec<String> = (0..lanes)
            .map(|i| format!("((a[{i}] as u32) >> N) as i32"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let bitmask_expr = || -> String {
        let items: Vec<String> = (0..lanes)
            .map(|i| {
                if i == 0 {
                    "((a[0] as u32) >> 31)".to_string()
                } else {
                    format!("(((a[{i}] as u32) >> 31) << {i})")
                }
            })
            .collect();
        items.join(" | ")
    };

    let all_true_expr = || -> String {
        let items: Vec<String> = (0..lanes).map(|i| format!("a[{i}] != 0")).collect();
        items.join(" && ")
    };

    let any_true_expr = || -> String {
        let items: Vec<String> = (0..lanes).map(|i| format!("a[{i}] != 0")).collect();
        items.join(" || ")
    };

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
                {add_lanes}
            }}

            #[inline(always)]
            fn sub(self, a: {array}, b: {array}) -> {array} {{
                {sub_lanes}
            }}

            #[inline(always)]
            fn mul(self, a: {array}, b: {array}) -> {array} {{
                {mul_lanes}
            }}

            #[inline(always)]
            fn neg(self, a: {array}) -> {array} {{
                {neg}
            }}

            // ====== Math ======

            #[inline(always)]
            fn min(self, a: {array}, b: {array}) -> {array} {{
                {min_lanes}
            }}

            #[inline(always)]
            fn max(self, a: {array}, b: {array}) -> {array} {{
                {max_lanes}
            }}

            #[inline(always)]
            fn abs(self, a: {array}) -> {array} {{
                {abs}
            }}

            // ====== Comparisons ======

            #[inline(always)]
            fn simd_eq(self, a: {array}, b: {array}) -> {array} {{
                let mut r = [0i32; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] == b[i] {{ -1 }} else {{ 0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_ne(self, a: {array}, b: {array}) -> {array} {{
                let mut r = [0i32; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] != b[i] {{ -1 }} else {{ 0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_lt(self, a: {array}, b: {array}) -> {array} {{
                let mut r = [0i32; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] < b[i] {{ -1 }} else {{ 0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_le(self, a: {array}, b: {array}) -> {array} {{
                let mut r = [0i32; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] <= b[i] {{ -1 }} else {{ 0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_gt(self, a: {array}, b: {array}) -> {array} {{
                let mut r = [0i32; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] > b[i] {{ -1 }} else {{ 0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_ge(self, a: {array}, b: {array}) -> {array} {{
                let mut r = [0i32; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] >= b[i] {{ -1 }} else {{ 0 }};
                }}
                r
            }}

            #[inline(always)]
            fn blend(self, mask: {array}, if_true: {array}, if_false: {array}) -> {array} {{
                let mut r = [0i32; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if mask[i] != 0 {{ if_true[i] }} else {{ if_false[i] }};
                }}
                r
            }}

            // ====== Reductions ======

            #[inline(always)]
            fn reduce_add(self, a: {array}) -> i32 {{
                {reduce_add}
            }}

            // ====== Bitwise ======

            #[inline(always)]
            fn not(self, a: {array}) -> {array} {{
                {not_lanes}
            }}

            #[inline(always)]
            fn bitand(self, a: {array}, b: {array}) -> {array} {{
                {and_lanes}
            }}

            #[inline(always)]
            fn bitor(self, a: {array}, b: {array}) -> {array} {{
                {or_lanes}
            }}

            #[inline(always)]
            fn bitxor(self, a: {array}, b: {array}) -> {array} {{
                {xor_lanes}
            }}

            // ====== Shifts ======

            #[inline(always)]
            fn shl_const<const N: i32>(self, a: {array}) -> {array} {{
                {shl}
            }}

            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(self, a: {array}) -> {array} {{
                {shr_arithmetic}
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(self, a: {array}) -> {array} {{
                {shr_logical}
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
    "#,
        add_lanes = binary_method("wrapping_add"),
        sub_lanes = binary_method("wrapping_sub"),
        mul_lanes = binary_method("wrapping_mul"),
        neg = neg_lanes(),
        min_lanes = min_lanes(),
        max_lanes = max_lanes(),
        abs = abs_lanes(),
        reduce_add = reduce_add(),
        not_lanes = {
            let items: Vec<String> = (0..lanes).map(|i| format!("!a[{i}]")).collect();
            format!("[{}]", items.join(", "))
        },
        and_lanes = binary_infix("&"),
        or_lanes = binary_infix("|"),
        xor_lanes = binary_infix("^"),
        shl = shl_lanes(),
        shr_arithmetic = shr_arithmetic_lanes(),
        shr_logical = shr_logical_lanes(),
        all_true = all_true_expr(),
        any_true = any_true_expr(),
        bitmask = bitmask_expr(),
    }
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

        let bitcast_f2i = (0..lanes)
            .map(|i| format!("a[{i}].to_bits() as i32"))
            .collect::<Vec<_>>()
            .join(", ");
        let bitcast_i2f = (0..lanes)
            .map(|i| format!("f32::from_bits(a[{i}] as u32)"))
            .collect::<Vec<_>>()
            .join(", ");
        let cvt_f2i = (0..lanes)
            .map(|i| format!("a[{i}] as i32"))
            .collect::<Vec<_>>()
            .join(", ");
        let cvt_f2i_round = (0..lanes)
            .map(|i| format!("f32_round(a[{i}]) as i32"))
            .collect::<Vec<_>>()
            .join(", ");
        let cvt_i2f = (0..lanes)
            .map(|i| format!("a[{i}] as f32"))
            .collect::<Vec<_>>()
            .join(", ");

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
                    [{bitcast_f2i}]
                }}

                #[inline(always)]
                fn bitcast_i32_to_f32(self, a: {i_array}) -> {f_array} {{
                    [{bitcast_i2f}]
                }}

                #[inline(always)]
                fn convert_f32_to_i32(self, a: {f_array}) -> {i_array} {{
                    [{cvt_f2i}]
                }}

                #[inline(always)]
                fn convert_f32_to_i32_round(self, a: {f_array}) -> {i_array} {{
                    [{cvt_f2i_round}]
                }}

                #[inline(always)]
                fn convert_i32_to_f32(self, a: {i_array}) -> {f_array} {{
                    [{cvt_i2f}]
                }}
            }}

        "#});
    }

    // U32 <-> I32 bitcast impls
    for lanes in [4, 8] {
        let u_array = format!("[u32; {lanes}]");
        let i_array = format!("[i32; {lanes}]");
        let trait_name = format!("U32x{lanes}Bitcast");

        let u2i = (0..lanes)
            .map(|i| format!("a[{i}] as i32"))
            .collect::<Vec<_>>()
            .join(", ");
        let i2u = (0..lanes)
            .map(|i| format!("a[{i}] as u32"))
            .collect::<Vec<_>>()
            .join(", ");

        code.push_str(&formatdoc! {r#"
            impl {trait_name} for archmage::ScalarToken {{
                #[inline(always)]
                fn bitcast_u32_to_i32(self, a: {u_array}) -> {i_array} {{
                    [{u2i}]
                }}

                #[inline(always)]
                fn bitcast_i32_to_u32(self, a: {i_array}) -> {u_array} {{
                    [{i2u}]
                }}
            }}

        "#});
    }

    // I64 <-> F64 bitcast impls
    for lanes in [2, 4] {
        let i_array = format!("[i64; {lanes}]");
        let f_array = format!("[f64; {lanes}]");
        let trait_name = format!("I64x{lanes}Bitcast");

        let i2f = (0..lanes)
            .map(|i| format!("f64::from_bits(a[{i}] as u64)"))
            .collect::<Vec<_>>()
            .join(", ");
        let f2i = (0..lanes)
            .map(|i| format!("a[{i}].to_bits() as i64"))
            .collect::<Vec<_>>()
            .join(", ");

        code.push_str(&formatdoc! {r#"
            impl {trait_name} for archmage::ScalarToken {{
                #[inline(always)]
                fn bitcast_i64_to_f64(self, a: {i_array}) -> {f_array} {{
                    [{i2f}]
                }}

                #[inline(always)]
                fn bitcast_f64_to_i64(self, a: {f_array}) -> {i_array} {{
                    [{f2i}]
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
        code.push_str("\n#[cfg(target_arch = \"aarch64\")]\n");
        if ty.native_on_neon() {
            code.push_str(&generate_neon_native_i32_impl(ty));
        } else {
            code.push_str(&generate_neon_polyfill_i32_impl(ty));
        }
        code.push('\n');
    }

    code
}

fn generate_neon_native_i32_impl(ty: &I32VecType) -> String {
    let trait_name = ty.trait_name();
    let array = ty.array_type();
    let lanes = ty.lanes;

    formatdoc! {r#"
        impl {trait_name} for archmage::NeonToken {{
            type Repr = int32x4_t;

            #[inline(always)]
            fn splat(self, v: i32) -> int32x4_t {{
                unsafe {{ vdupq_n_s32(v) }}
            }}

            #[inline(always)]
            fn zero(self) -> int32x4_t {{
                unsafe {{ vdupq_n_s32(0) }}
            }}

            #[inline(always)]
            fn load(self, data: &{array}) -> int32x4_t {{
                unsafe {{ vld1q_s32(data.as_ptr()) }}
            }}

            #[inline(always)]
            fn from_array(self, arr: {array}) -> int32x4_t {{
                unsafe {{ vld1q_s32(arr.as_ptr()) }}
            }}

            #[inline(always)]
            fn store(self, repr: int32x4_t, out: &mut {array}) {{
                unsafe {{ vst1q_s32(out.as_mut_ptr(), repr) }};
            }}

            #[inline(always)]
            fn to_array(self, repr: int32x4_t) -> {array} {{
                let mut out = [0i32; {lanes}];
                unsafe {{ vst1q_s32(out.as_mut_ptr(), repr) }};
                out
            }}

            #[inline(always)]
            fn add(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {{ unsafe {{ vaddq_s32(a, b) }} }}
            #[inline(always)]
            fn sub(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {{ unsafe {{ vsubq_s32(a, b) }} }}
            #[inline(always)]
            fn mul(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {{ unsafe {{ vmulq_s32(a, b) }} }}
            #[inline(always)]
            fn neg(self, a: int32x4_t) -> int32x4_t {{ unsafe {{ vnegq_s32(a) }} }}
            #[inline(always)]
            fn min(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {{ unsafe {{ vminq_s32(a, b) }} }}
            #[inline(always)]
            fn max(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {{ unsafe {{ vmaxq_s32(a, b) }} }}
            #[inline(always)]
            fn abs(self, a: int32x4_t) -> int32x4_t {{ unsafe {{ vabsq_s32(a) }} }}

            #[inline(always)]
            fn simd_eq(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {{
                unsafe {{ vreinterpretq_s32_u32(vceqq_s32(a, b)) }}
            }}
            #[inline(always)]
            fn simd_ne(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {{
                unsafe {{ vreinterpretq_s32_u32(vmvnq_u32(vceqq_s32(a, b))) }}
            }}
            #[inline(always)]
            fn simd_lt(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {{
                unsafe {{ vreinterpretq_s32_u32(vcltq_s32(a, b)) }}
            }}
            #[inline(always)]
            fn simd_le(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {{
                unsafe {{ vreinterpretq_s32_u32(vcleq_s32(a, b)) }}
            }}
            #[inline(always)]
            fn simd_gt(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {{
                unsafe {{ vreinterpretq_s32_u32(vcgtq_s32(a, b)) }}
            }}
            #[inline(always)]
            fn simd_ge(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {{
                unsafe {{ vreinterpretq_s32_u32(vcgeq_s32(a, b)) }}
            }}

            #[inline(always)]
            fn blend(self, mask: int32x4_t, if_true: int32x4_t, if_false: int32x4_t) -> int32x4_t {{
                unsafe {{ vbslq_s32(vreinterpretq_u32_s32(mask), if_true, if_false) }}
            }}

            #[inline(always)]
            fn reduce_add(self, a: int32x4_t) -> i32 {{
                unsafe {{ vaddvq_s32(a) }}
            }}

            #[inline(always)]
            fn not(self, a: int32x4_t) -> int32x4_t {{
                unsafe {{ vmvnq_s32(a) }}
            }}
            #[inline(always)]
            fn bitand(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {{
                unsafe {{ vandq_s32(a, b) }}
            }}
            #[inline(always)]
            fn bitor(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {{
                unsafe {{ vorrq_s32(a, b) }}
            }}
            #[inline(always)]
            fn bitxor(self, a: int32x4_t, b: int32x4_t) -> int32x4_t {{
                unsafe {{ veorq_s32(a, b) }}
            }}

            #[inline(always)]
            fn shl_const<const N: i32>(self, a: int32x4_t) -> int32x4_t {{
                unsafe {{ vshlq_n_s32::<N>(a) }}
            }}

            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(self, a: int32x4_t) -> int32x4_t {{
                const {{ assert!(N >= 0 && N <= 31) }};
                unsafe {{ vshlq_s32(a, vdupq_n_s32(-N)) }}
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(self, a: int32x4_t) -> int32x4_t {{
                const {{ assert!(N >= 0 && N <= 31) }};
                unsafe {{ vreinterpretq_s32_u32(vshlq_u32(vreinterpretq_u32_s32(a), vdupq_n_s32(-N))) }}
            }}

            #[inline(always)]
            fn all_true(self, a: int32x4_t) -> bool {{
                unsafe {{ vminvq_u32(vreinterpretq_u32_s32(a)) != 0 }}
            }}

            #[inline(always)]
            fn any_true(self, a: int32x4_t) -> bool {{
                unsafe {{ vmaxvq_u32(vreinterpretq_u32_s32(a)) != 0 }}
            }}

            #[inline(always)]
            fn bitmask(self, a: int32x4_t) -> u32 {{
                unsafe {{
                    // Extract sign bit of each 32-bit lane as 0/1 (LOGICAL shift on
                    // the u32 view — an arithmetic s32 shift would sign-extend to
                    // 0xFFFF_FFFF and corrupt the packed mask).
                    let shift = vshrq_n_u32::<31>(vreinterpretq_u32_s32(a));
                    // Pack: lane0 | (lane1<<1) | (lane2<<2) | (lane3<<3)
                    let lane0 = vgetq_lane_u32::<0>(shift);
                    let lane1 = vgetq_lane_u32::<1>(shift);
                    let lane2 = vgetq_lane_u32::<2>(shift);
                    let lane3 = vgetq_lane_u32::<3>(shift);
                    lane0 | (lane1 << 1) | (lane2 << 2) | (lane3 << 3)
                }}
            }}
        }}
    "#}
}

fn generate_neon_polyfill_i32_impl(ty: &I32VecType) -> String {
    let trait_name = ty.trait_name();
    let repr = ty.neon_repr();
    let array = ty.array_type();
    let lanes = ty.lanes;
    let sub_count = ty.sub_count();

    let binary_op = |intrinsic: &str| -> String {
        let items: Vec<String> = (0..sub_count)
            .map(|i| format!("{intrinsic}(a[{i}], b[{i}])"))
            .collect();
        format!("unsafe {{ [{}] }}", items.join(", "))
    };

    let unary_op = |intrinsic: &str| -> String {
        let items: Vec<String> = (0..sub_count)
            .map(|i| format!("{intrinsic}(a[{i}])"))
            .collect();
        format!("unsafe {{ [{}] }}", items.join(", "))
    };

    let cmp_op = |intrinsic: &str| -> String {
        let items: Vec<String> = (0..sub_count)
            .map(|i| format!("vreinterpretq_s32_u32({intrinsic}(a[{i}], b[{i}]))"))
            .collect();
        format!("unsafe {{ [{}] }}", items.join(", "))
    };

    let ne_op = || -> String {
        let items: Vec<String> = (0..sub_count)
            .map(|i| format!("vreinterpretq_s32_u32(vmvnq_u32(vceqq_s32(a[{i}], b[{i}])))"))
            .collect();
        format!("unsafe {{ [{}] }}", items.join(", "))
    };

    formatdoc! {r#"
        impl {trait_name} for archmage::NeonToken {{
            type Repr = {repr};

            #[inline(always)]
            fn splat(self, v: i32) -> {repr} {{
                unsafe {{
                    let v4 = vdupq_n_s32(v);
                    [{v4_copies}]
                }}
            }}

            #[inline(always)]
            fn zero(self) -> {repr} {{
                unsafe {{
                    let z = vdupq_n_s32(0);
                    [{z_copies}]
                }}
            }}

            #[inline(always)]
            fn load(self, data: &{array}) -> {repr} {{
                unsafe {{
                    [{load_lanes}]
                }}
            }}

            #[inline(always)]
            fn from_array(self, arr: {array}) -> {repr} {{
                <Self as {trait_name}>::load(self, &arr)
            }}

            #[inline(always)]
            fn store(self, repr: {repr}, out: &mut {array}) {{
                unsafe {{
                    {store_lanes}
                }}
            }}

            #[inline(always)]
            fn to_array(self, repr: {repr}) -> {array} {{
                let mut out = [0i32; {lanes}];
                <Self as {trait_name}>::store(self, repr, &mut out);
                out
            }}

            #[inline(always)]
            fn add(self, a: {repr}, b: {repr}) -> {repr} {{ {add} }}
            #[inline(always)]
            fn sub(self, a: {repr}, b: {repr}) -> {repr} {{ {sub} }}
            #[inline(always)]
            fn mul(self, a: {repr}, b: {repr}) -> {repr} {{ {mul} }}
            #[inline(always)]
            fn neg(self, a: {repr}) -> {repr} {{ {neg} }}
            #[inline(always)]
            fn min(self, a: {repr}, b: {repr}) -> {repr} {{ {min} }}
            #[inline(always)]
            fn max(self, a: {repr}, b: {repr}) -> {repr} {{ {max} }}
            #[inline(always)]
            fn abs(self, a: {repr}) -> {repr} {{ {abs} }}

            #[inline(always)]
            fn simd_eq(self, a: {repr}, b: {repr}) -> {repr} {{ {eq} }}
            #[inline(always)]
            fn simd_ne(self, a: {repr}, b: {repr}) -> {repr} {{ {ne} }}
            #[inline(always)]
            fn simd_lt(self, a: {repr}, b: {repr}) -> {repr} {{ {lt} }}
            #[inline(always)]
            fn simd_le(self, a: {repr}, b: {repr}) -> {repr} {{ {le} }}
            #[inline(always)]
            fn simd_gt(self, a: {repr}, b: {repr}) -> {repr} {{ {gt} }}
            #[inline(always)]
            fn simd_ge(self, a: {repr}, b: {repr}) -> {repr} {{ {ge} }}

            #[inline(always)]
            fn blend(self, mask: {repr}, if_true: {repr}, if_false: {repr}) -> {repr} {{
                {blend}
            }}

            #[inline(always)]
            fn reduce_add(self, a: {repr}) -> i32 {{
                {reduce_add}
            }}

            #[inline(always)]
            fn not(self, a: {repr}) -> {repr} {{ {not} }}
            #[inline(always)]
            fn bitand(self, a: {repr}, b: {repr}) -> {repr} {{ {bitand} }}
            #[inline(always)]
            fn bitor(self, a: {repr}, b: {repr}) -> {repr} {{ {bitor} }}
            #[inline(always)]
            fn bitxor(self, a: {repr}, b: {repr}) -> {repr} {{ {bitxor} }}

            #[inline(always)]
            fn shl_const<const N: i32>(self, a: {repr}) -> {repr} {{
                {shl}
            }}

            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(self, a: {repr}) -> {repr} {{
                {shr_arith}
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(self, a: {repr}) -> {repr} {{
                {shr_logic}
            }}

            #[inline(always)]
            fn all_true(self, a: {repr}) -> bool {{
                {all_true}
            }}

            #[inline(always)]
            fn any_true(self, a: {repr}) -> bool {{
                {any_true}
            }}

            #[inline(always)]
            fn bitmask(self, a: {repr}) -> u32 {{
                {bitmask}
            }}
        }}
    "#,
        v4_copies = (0..sub_count).map(|_| "v4").collect::<Vec<_>>().join(", "),
        z_copies = (0..sub_count).map(|_| "z").collect::<Vec<_>>().join(", "),
        load_lanes = (0..sub_count)
            .map(|i| format!("vld1q_s32(data.as_ptr().add({}))", i * 4))
            .collect::<Vec<_>>().join(", "),
        store_lanes = (0..sub_count)
            .map(|i| format!("vst1q_s32(out.as_mut_ptr().add({}), repr[{i}]);", i * 4))
            .collect::<Vec<_>>().join("\n            "),
        add = binary_op("vaddq_s32"),
        sub = binary_op("vsubq_s32"),
        mul = binary_op("vmulq_s32"),
        neg = unary_op("vnegq_s32"),
        min = binary_op("vminq_s32"),
        max = binary_op("vmaxq_s32"),
        abs = unary_op("vabsq_s32"),
        eq = cmp_op("vceqq_s32"),
        ne = ne_op(),
        lt = cmp_op("vcltq_s32"),
        le = cmp_op("vcleq_s32"),
        gt = cmp_op("vcgtq_s32"),
        ge = cmp_op("vcgeq_s32"),
        blend = {
            let items: Vec<String> = (0..sub_count)
                .map(|i| format!("vbslq_s32(vreinterpretq_u32_s32(mask[{i}]), if_true[{i}], if_false[{i}])"))
                .collect();
            format!("unsafe {{ [{}] }}", items.join(", "))
        },
        reduce_add = {
            let mut body = "unsafe {\n".to_string();
            body.push_str("            let m = vaddq_s32(a[0], a[1]);\n");
            for i in 2..sub_count {
                body.push_str(&format!("            let m = vaddq_s32(m, a[{i}]);\n"));
            }
            body.push_str("            vaddvq_s32(m)\n");
            body.push_str("        }");
            body
        },
        not = unary_op("vmvnq_s32"),
        bitand = binary_op("vandq_s32"),
        bitor = binary_op("vorrq_s32"),
        bitxor = binary_op("veorq_s32"),
        shl = {
            let items: Vec<String> = (0..sub_count)
                .map(|i| format!("vshlq_n_s32::<N>(a[{i}])"))
                .collect();
            format!("unsafe {{ [{}] }}", items.join(", "))
        },
        shr_arith = {
            let items: Vec<String> = (0..sub_count)
                .map(|i| format!("vshlq_s32(a[{i}], vdupq_n_s32(-N))"))
                .collect();
            format!(
                "const {{ assert!(N >= 0 && N <= 31) }};\n                unsafe {{ [{}] }}",
                items.join(", ")
            )
        },
        shr_logic = {
            let items: Vec<String> = (0..sub_count)
                .map(|i| {
                    format!(
                        "vreinterpretq_s32_u32(vshlq_u32(vreinterpretq_u32_s32(a[{i}]), vdupq_n_s32(-N)))"
                    )
                })
                .collect();
            format!(
                "const {{ assert!(N >= 0 && N <= 31) }};\n                unsafe {{ [{}] }}",
                items.join(", ")
            )
        },
        all_true = {
            let items: Vec<String> = (0..sub_count)
                .map(|i| format!("vminvq_u32(vreinterpretq_u32_s32(a[{i}])) != 0"))
                .collect();
            format!("unsafe {{ {} }}", items.join(" && "))
        },
        any_true = {
            let items: Vec<String> = (0..sub_count)
                .map(|i| format!("vmaxvq_u32(vreinterpretq_u32_s32(a[{i}])) != 0"))
                .collect();
            format!("unsafe {{ {} }}", items.join(" || "))
        },
        bitmask = {
            let mut body = "unsafe {\n".to_string();
            body.push_str("            let mut bits = 0u32;\n");
            for i in 0..sub_count {
                let base = i * 4;
                body.push_str(&format!("            let s{i} = vshrq_n_u32::<31>(vreinterpretq_u32_s32(a[{i}]));\n"));
                body.push_str(&format!("            bits |= vgetq_lane_u32::<0>(s{i}) << {base};\n"));
                body.push_str(&format!("            bits |= (vgetq_lane_u32::<1>(s{i})) << {};\n", base + 1));
                body.push_str(&format!("            bits |= (vgetq_lane_u32::<2>(s{i})) << {};\n", base + 2));
                body.push_str(&format!("            bits |= (vgetq_lane_u32::<3>(s{i})) << {};\n", base + 3));
            }
            body.push_str("            bits\n");
            body.push_str("        }");
            body
        },
    }
}

// ============================================================================
// NEON Conversion Implementation Generation
// ============================================================================

fn generate_neon_convert_impls() -> String {
    formatdoc! {r#"

        #[cfg(target_arch = "aarch64")]
        impl F32x4Convert for archmage::NeonToken {{
            #[inline(always)]
            fn bitcast_f32_to_i32(self, a: float32x4_t) -> int32x4_t {{
                unsafe {{ vreinterpretq_s32_f32(a) }}
            }}

            #[inline(always)]
            fn bitcast_i32_to_f32(self, a: int32x4_t) -> float32x4_t {{
                unsafe {{ vreinterpretq_f32_s32(a) }}
            }}

            #[inline(always)]
            fn convert_f32_to_i32(self, a: float32x4_t) -> int32x4_t {{
                unsafe {{ vcvtq_s32_f32(a) }}
            }}

            #[inline(always)]
            fn convert_f32_to_i32_round(self, a: float32x4_t) -> int32x4_t {{
                unsafe {{ vcvtnq_s32_f32(a) }}
            }}

            #[inline(always)]
            fn convert_i32_to_f32(self, a: int32x4_t) -> float32x4_t {{
                unsafe {{ vcvtq_f32_s32(a) }}
            }}
        }}

        #[cfg(target_arch = "aarch64")]
        impl F32x8Convert for archmage::NeonToken {{
            #[inline(always)]
            fn bitcast_f32_to_i32(self, a: [float32x4_t; 2]) -> [int32x4_t; 2] {{
                unsafe {{ [vreinterpretq_s32_f32(a[0]), vreinterpretq_s32_f32(a[1])] }}
            }}

            #[inline(always)]
            fn bitcast_i32_to_f32(self, a: [int32x4_t; 2]) -> [float32x4_t; 2] {{
                unsafe {{ [vreinterpretq_f32_s32(a[0]), vreinterpretq_f32_s32(a[1])] }}
            }}

            #[inline(always)]
            fn convert_f32_to_i32(self, a: [float32x4_t; 2]) -> [int32x4_t; 2] {{
                unsafe {{ [vcvtq_s32_f32(a[0]), vcvtq_s32_f32(a[1])] }}
            }}

            #[inline(always)]
            fn convert_f32_to_i32_round(self, a: [float32x4_t; 2]) -> [int32x4_t; 2] {{
                unsafe {{ [vcvtnq_s32_f32(a[0]), vcvtnq_s32_f32(a[1])] }}
            }}

            #[inline(always)]
            fn convert_i32_to_f32(self, a: [int32x4_t; 2]) -> [float32x4_t; 2] {{
                unsafe {{ [vcvtq_f32_s32(a[0]), vcvtq_f32_s32(a[1])] }}
            }}
        }}

        #[cfg(all(target_arch = "aarch64", feature = "w512"))]
        impl F32x16Convert for archmage::NeonToken {{
            #[inline(always)]
            fn bitcast_f32_to_i32(self, a: [float32x4_t; 4]) -> [int32x4_t; 4] {{
                unsafe {{ [vreinterpretq_s32_f32(a[0]), vreinterpretq_s32_f32(a[1]), vreinterpretq_s32_f32(a[2]), vreinterpretq_s32_f32(a[3])] }}
            }}

            #[inline(always)]
            fn bitcast_i32_to_f32(self, a: [int32x4_t; 4]) -> [float32x4_t; 4] {{
                unsafe {{ [vreinterpretq_f32_s32(a[0]), vreinterpretq_f32_s32(a[1]), vreinterpretq_f32_s32(a[2]), vreinterpretq_f32_s32(a[3])] }}
            }}

            #[inline(always)]
            fn convert_f32_to_i32(self, a: [float32x4_t; 4]) -> [int32x4_t; 4] {{
                unsafe {{ [vcvtq_s32_f32(a[0]), vcvtq_s32_f32(a[1]), vcvtq_s32_f32(a[2]), vcvtq_s32_f32(a[3])] }}
            }}

            #[inline(always)]
            fn convert_f32_to_i32_round(self, a: [float32x4_t; 4]) -> [int32x4_t; 4] {{
                unsafe {{ [vcvtnq_s32_f32(a[0]), vcvtnq_s32_f32(a[1]), vcvtnq_s32_f32(a[2]), vcvtnq_s32_f32(a[3])] }}
            }}

            #[inline(always)]
            fn convert_i32_to_f32(self, a: [int32x4_t; 4]) -> [float32x4_t; 4] {{
                unsafe {{ [vcvtq_f32_s32(a[0]), vcvtq_f32_s32(a[1]), vcvtq_f32_s32(a[2]), vcvtq_f32_s32(a[3])] }}
            }}
        }}

        #[cfg(target_arch = "aarch64")]
        impl U32x4Bitcast for archmage::NeonToken {{
            #[inline(always)]
            fn bitcast_u32_to_i32(self, a: uint32x4_t) -> int32x4_t {{
                unsafe {{ vreinterpretq_s32_u32(a) }}
            }}

            #[inline(always)]
            fn bitcast_i32_to_u32(self, a: int32x4_t) -> uint32x4_t {{
                unsafe {{ vreinterpretq_u32_s32(a) }}
            }}
        }}

        #[cfg(target_arch = "aarch64")]
        impl U32x8Bitcast for archmage::NeonToken {{
            #[inline(always)]
            fn bitcast_u32_to_i32(self, a: [uint32x4_t; 2]) -> [int32x4_t; 2] {{
                unsafe {{ [vreinterpretq_s32_u32(a[0]), vreinterpretq_s32_u32(a[1])] }}
            }}

            #[inline(always)]
            fn bitcast_i32_to_u32(self, a: [int32x4_t; 2]) -> [uint32x4_t; 2] {{
                unsafe {{ [vreinterpretq_u32_s32(a[0]), vreinterpretq_u32_s32(a[1])] }}
            }}
        }}

        #[cfg(target_arch = "aarch64")]
        impl I64x2Bitcast for archmage::NeonToken {{
            #[inline(always)]
            fn bitcast_i64_to_f64(self, a: int64x2_t) -> float64x2_t {{
                unsafe {{ vreinterpretq_f64_s64(a) }}
            }}

            #[inline(always)]
            fn bitcast_f64_to_i64(self, a: float64x2_t) -> int64x2_t {{
                unsafe {{ vreinterpretq_s64_f64(a) }}
            }}
        }}

        #[cfg(target_arch = "aarch64")]
        impl I64x4Bitcast for archmage::NeonToken {{
            #[inline(always)]
            fn bitcast_i64_to_f64(self, a: [int64x2_t; 2]) -> [float64x2_t; 2] {{
                unsafe {{ [vreinterpretq_f64_s64(a[0]), vreinterpretq_f64_s64(a[1])] }}
            }}

            #[inline(always)]
            fn bitcast_f64_to_i64(self, a: [float64x2_t; 2]) -> [int64x2_t; 2] {{
                unsafe {{ [vreinterpretq_s64_f64(a[0]), vreinterpretq_s64_f64(a[1])] }}
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
        code.push_str("\n#[cfg(target_arch = \"wasm32\")]\n");
        if ty.native_on_wasm() {
            code.push_str(&generate_wasm_native_i32_impl(ty));
        } else {
            code.push_str(&generate_wasm_polyfill_i32_impl(ty));
        }
        code.push('\n');
    }

    code
}

fn generate_wasm_native_i32_impl(ty: &I32VecType) -> String {
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
            fn load(self, data: &{array}) -> v128 {{ unsafe {{ v128_load(data.as_ptr().cast()) }} }}
            #[inline(always)]
            fn from_array(self, arr: {array}) -> v128 {{ unsafe {{ v128_load(arr.as_ptr().cast()) }} }}
            #[inline(always)]
            fn store(self, repr: v128, out: &mut {array}) {{ unsafe {{ v128_store(out.as_mut_ptr().cast(), repr) }}; }}
            #[inline(always)]
            fn to_array(self, repr: v128) -> {array} {{
                let mut out = [0i32; {lanes}];
                unsafe {{ v128_store(out.as_mut_ptr().cast(), repr) }};
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

            #[inline(always)]
            fn all_true(self, a: v128) -> bool {{ i32x4_all_true(a) }}
            #[inline(always)]
            fn any_true(self, a: v128) -> bool {{ v128_any_true(a) }}
            #[inline(always)]
            fn bitmask(self, a: v128) -> u32 {{ i32x4_bitmask(a) as u32 }}
        }}
    "#}
}

fn generate_wasm_polyfill_i32_impl(ty: &I32VecType) -> String {
    let trait_name = ty.trait_name();
    let repr = ty.wasm_repr();
    let array = ty.array_type();
    let lanes = ty.lanes;
    let sub_count = ty.sub_count();

    let binary_op = |func: &str| -> String {
        let items: Vec<String> = (0..sub_count)
            .map(|i| format!("{func}(a[{i}], b[{i}])"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let unary_op = |func: &str| -> String {
        let items: Vec<String> = (0..sub_count).map(|i| format!("{func}(a[{i}])")).collect();
        format!("[{}]", items.join(", "))
    };

    let reduce_add_body = || -> String {
        let mut items = Vec::new();
        for i in 0..sub_count {
            for j in 0..4usize {
                items.push(format!("i32x4_extract_lane::<{j}>(a[{i}])"));
            }
        }
        items.join(" + ")
    };

    formatdoc! {r#"
        impl {trait_name} for archmage::Wasm128Token {{
            type Repr = {repr};

            #[inline(always)]
            fn splat(self, v: i32) -> {repr} {{
                let v4 = i32x4_splat(v);
                [{v4_copies}]
            }}

            #[inline(always)]
            fn zero(self) -> {repr} {{
                let z = i32x4_splat(0);
                [{z_copies}]
            }}

            #[inline(always)]
            fn load(self, data: &{array}) -> {repr} {{
                unsafe {{
                    [{load_lanes}]
                }}
            }}

            #[inline(always)]
            fn from_array(self, arr: {array}) -> {repr} {{
                <Self as {trait_name}>::load(self, &arr)
            }}

            #[inline(always)]
            fn store(self, repr: {repr}, out: &mut {array}) {{
                unsafe {{
                    {store_lanes}
                }}
            }}

            #[inline(always)]
            fn to_array(self, repr: {repr}) -> {array} {{
                let mut out = [0i32; {lanes}];
                <Self as {trait_name}>::store(self, repr, &mut out);
                out
            }}

            #[inline(always)]
            fn add(self, a: {repr}, b: {repr}) -> {repr} {{ {add} }}
            #[inline(always)]
            fn sub(self, a: {repr}, b: {repr}) -> {repr} {{ {sub} }}
            #[inline(always)]
            fn mul(self, a: {repr}, b: {repr}) -> {repr} {{ {mul} }}
            #[inline(always)]
            fn neg(self, a: {repr}) -> {repr} {{ {neg} }}
            #[inline(always)]
            fn min(self, a: {repr}, b: {repr}) -> {repr} {{ {min} }}
            #[inline(always)]
            fn max(self, a: {repr}, b: {repr}) -> {repr} {{ {max} }}
            #[inline(always)]
            fn abs(self, a: {repr}) -> {repr} {{ {abs} }}

            #[inline(always)]
            fn simd_eq(self, a: {repr}, b: {repr}) -> {repr} {{ {eq} }}
            #[inline(always)]
            fn simd_ne(self, a: {repr}, b: {repr}) -> {repr} {{ {ne} }}
            #[inline(always)]
            fn simd_lt(self, a: {repr}, b: {repr}) -> {repr} {{ {lt} }}
            #[inline(always)]
            fn simd_le(self, a: {repr}, b: {repr}) -> {repr} {{ {le} }}
            #[inline(always)]
            fn simd_gt(self, a: {repr}, b: {repr}) -> {repr} {{ {gt} }}
            #[inline(always)]
            fn simd_ge(self, a: {repr}, b: {repr}) -> {repr} {{ {ge} }}
            #[inline(always)]
            fn blend(self, mask: {repr}, if_true: {repr}, if_false: {repr}) -> {repr} {{
                [{blend_lanes}]
            }}

            #[inline(always)]
            fn reduce_add(self, a: {repr}) -> i32 {{ {reduce_add} }}

            #[inline(always)]
            fn not(self, a: {repr}) -> {repr} {{ {not} }}
            #[inline(always)]
            fn bitand(self, a: {repr}, b: {repr}) -> {repr} {{ {and} }}
            #[inline(always)]
            fn bitor(self, a: {repr}, b: {repr}) -> {repr} {{ {or} }}
            #[inline(always)]
            fn bitxor(self, a: {repr}, b: {repr}) -> {repr} {{ {xor} }}

            #[inline(always)]
            fn shl_const<const N: i32>(self, a: {repr}) -> {repr} {{
                [{shl_lanes}]
            }}

            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(self, a: {repr}) -> {repr} {{
                [{shr_arith_lanes}]
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(self, a: {repr}) -> {repr} {{
                [{shr_logic_lanes}]
            }}

            #[inline(always)]
            fn all_true(self, a: {repr}) -> bool {{
                {all_true}
            }}

            #[inline(always)]
            fn any_true(self, a: {repr}) -> bool {{
                {any_true}
            }}

            #[inline(always)]
            fn bitmask(self, a: {repr}) -> u32 {{
                {bitmask}
            }}
        }}
    "#,
        v4_copies = (0..sub_count).map(|_| "v4").collect::<Vec<_>>().join(", "),
        z_copies = (0..sub_count).map(|_| "z").collect::<Vec<_>>().join(", "),
        load_lanes = (0..sub_count)
            .map(|i| format!("v128_load(data.as_ptr().add({}).cast())", i * 4))
            .collect::<Vec<_>>().join(", "),
        store_lanes = (0..sub_count)
            .map(|i| format!("v128_store(out.as_mut_ptr().add({}).cast(), repr[{i}]);", i * 4))
            .collect::<Vec<_>>().join("\n            "),
        add = binary_op("i32x4_add"),
        sub = binary_op("i32x4_sub"),
        mul = binary_op("i32x4_mul"),
        neg = unary_op("i32x4_neg"),
        min = binary_op("i32x4_min"),
        max = binary_op("i32x4_max"),
        abs = unary_op("i32x4_abs"),
        eq = binary_op("i32x4_eq"),
        ne = binary_op("i32x4_ne"),
        lt = binary_op("i32x4_lt"),
        le = binary_op("i32x4_le"),
        gt = binary_op("i32x4_gt"),
        ge = binary_op("i32x4_ge"),
        blend_lanes = (0..sub_count)
            .map(|i| format!("v128_bitselect(if_true[{i}], if_false[{i}], mask[{i}])"))
            .collect::<Vec<_>>().join(", "),
        reduce_add = reduce_add_body(),
        not = unary_op("v128_not"),
        and = binary_op("v128_and"),
        or = binary_op("v128_or"),
        xor = binary_op("v128_xor"),
        shl_lanes = (0..sub_count)
            .map(|i| format!("i32x4_shl(a[{i}], N as u32)"))
            .collect::<Vec<_>>().join(", "),
        shr_arith_lanes = (0..sub_count)
            .map(|i| format!("i32x4_shr(a[{i}], N as u32)"))
            .collect::<Vec<_>>().join(", "),
        shr_logic_lanes = (0..sub_count)
            .map(|i| format!("u32x4_shr(a[{i}], N as u32)"))
            .collect::<Vec<_>>().join(", "),
        all_true = (0..sub_count)
            .map(|i| format!("i32x4_all_true(a[{i}])"))
            .collect::<Vec<_>>().join(" && "),
        any_true = (0..sub_count)
            .map(|i| format!("v128_any_true(a[{i}])"))
            .collect::<Vec<_>>().join(" || "),
        bitmask = {
            let items: Vec<String> = (0..sub_count)
                .map(|i| format!("((i32x4_bitmask(a[{i}]) as u32) << {})", i * 4))
                .collect();
            items.join(" | ")
        },
    }
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
            fn convert_f32_to_i32_round(self, a: v128) -> v128 {{
                i32x4_trunc_sat_f32x4(f32x4_nearest(a))
            }}

            #[inline(always)]
            fn convert_i32_to_f32(self, a: v128) -> v128 {{
                f32x4_convert_i32x4(a)
            }}
        }}

        #[cfg(target_arch = "wasm32")]
        impl F32x8Convert for archmage::Wasm128Token {{
            #[inline(always)]
            fn bitcast_f32_to_i32(self, a: [v128; 2]) -> [v128; 2] {{ a }}

            #[inline(always)]
            fn bitcast_i32_to_f32(self, a: [v128; 2]) -> [v128; 2] {{ a }}

            #[inline(always)]
            fn convert_f32_to_i32(self, a: [v128; 2]) -> [v128; 2] {{
                [i32x4_trunc_sat_f32x4(a[0]), i32x4_trunc_sat_f32x4(a[1])]
            }}

            #[inline(always)]
            fn convert_f32_to_i32_round(self, a: [v128; 2]) -> [v128; 2] {{
                [
                    i32x4_trunc_sat_f32x4(f32x4_nearest(a[0])),
                    i32x4_trunc_sat_f32x4(f32x4_nearest(a[1])),
                ]
            }}

            #[inline(always)]
            fn convert_i32_to_f32(self, a: [v128; 2]) -> [v128; 2] {{
                [f32x4_convert_i32x4(a[0]), f32x4_convert_i32x4(a[1])]
            }}
        }}

        #[cfg(all(target_arch = "wasm32", feature = "w512"))]
        impl F32x16Convert for archmage::Wasm128Token {{
            #[inline(always)]
            fn bitcast_f32_to_i32(self, a: [v128; 4]) -> [v128; 4] {{ a }}

            #[inline(always)]
            fn bitcast_i32_to_f32(self, a: [v128; 4]) -> [v128; 4] {{ a }}

            #[inline(always)]
            fn convert_f32_to_i32(self, a: [v128; 4]) -> [v128; 4] {{
                [i32x4_trunc_sat_f32x4(a[0]), i32x4_trunc_sat_f32x4(a[1]), i32x4_trunc_sat_f32x4(a[2]), i32x4_trunc_sat_f32x4(a[3])]
            }}

            #[inline(always)]
            fn convert_f32_to_i32_round(self, a: [v128; 4]) -> [v128; 4] {{
                [
                    i32x4_trunc_sat_f32x4(f32x4_nearest(a[0])),
                    i32x4_trunc_sat_f32x4(f32x4_nearest(a[1])),
                    i32x4_trunc_sat_f32x4(f32x4_nearest(a[2])),
                    i32x4_trunc_sat_f32x4(f32x4_nearest(a[3])),
                ]
            }}

            #[inline(always)]
            fn convert_i32_to_f32(self, a: [v128; 4]) -> [v128; 4] {{
                [f32x4_convert_i32x4(a[0]), f32x4_convert_i32x4(a[1]), f32x4_convert_i32x4(a[2]), f32x4_convert_i32x4(a[3])]
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
        impl U32x8Bitcast for archmage::Wasm128Token {{
            #[inline(always)]
            fn bitcast_u32_to_i32(self, a: [v128; 2]) -> [v128; 2] {{ a }}

            #[inline(always)]
            fn bitcast_i32_to_u32(self, a: [v128; 2]) -> [v128; 2] {{ a }}
        }}

        #[cfg(target_arch = "wasm32")]
        impl I64x2Bitcast for archmage::Wasm128Token {{
            #[inline(always)]
            fn bitcast_i64_to_f64(self, a: v128) -> v128 {{ a }}

            #[inline(always)]
            fn bitcast_f64_to_i64(self, a: v128) -> v128 {{ a }}
        }}

        #[cfg(target_arch = "wasm32")]
        impl I64x4Bitcast for archmage::Wasm128Token {{
            #[inline(always)]
            fn bitcast_i64_to_f64(self, a: [v128; 2]) -> [v128; 2] {{ a }}

            #[inline(always)]
            fn bitcast_f64_to_i64(self, a: [v128; 2]) -> [v128; 2] {{ a }}
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
            type Repr: Copy + Clone + Send + Sync;

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

            // ====== Boolean ======

            /// True if all lanes have their sign bit set (all-1s mask).
            fn all_true(self, a: Self::Repr) -> bool;

            /// True if any lane has its sign bit set (any all-1s mask lane).
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
        code.push_str("\n#[cfg(target_arch = \"x86_64\")]\n");
        code.push_str(&generate_x86_u32_impl(ty, token));
        code.push('\n');
    }

    code
}

fn generate_x86_u32_impl(ty: &U32VecType, token: &str) -> String {
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

            #[inline(always)]
            fn splat(self, v: u32) -> {inner} {{
                unsafe {{ {p}_set1_epi32(v as i32) }}
            }}

            #[inline(always)]
            fn zero(self) -> {inner} {{
                unsafe {{ {p}_setzero_si{bits}() }}
            }}

            #[inline(always)]
            fn load(self, data: &{array}) -> {inner} {{
                unsafe {{ {p}_loadu_si{bits}(data.as_ptr().cast()) }}
            }}

            #[inline(always)]
            fn from_array(self, arr: {array}) -> {inner} {{
                // SAFETY: {array} and {inner} have identical size and layout.
                unsafe {{ core::mem::transmute(arr) }}
            }}

            #[inline(always)]
            fn store(self, repr: {inner}, out: &mut {array}) {{
                unsafe {{ {p}_storeu_si{bits}(out.as_mut_ptr().cast(), repr) }};
            }}

            #[inline(always)]
            fn to_array(self, repr: {inner}) -> {array} {{
                let mut out = [0u32; {lanes}];
                unsafe {{ {p}_storeu_si{bits}(out.as_mut_ptr().cast(), repr) }};
                out
            }}

            // ====== Arithmetic ======

            #[inline(always)]
            fn add(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_add_epi32(a, b) }}
            }}

            #[inline(always)]
            fn sub(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_sub_epi32(a, b) }}
            }}

            #[inline(always)]
            fn mul(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_mullo_epi32(a, b) }}
            }}

            // ====== Math ======

            #[inline(always)]
            fn min(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_min_epu32(a, b) }}
            }}

            #[inline(always)]
            fn max(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_max_epu32(a, b) }}
            }}

            // ====== Comparisons ======

            #[inline(always)]
            fn simd_eq(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_cmpeq_epi32(a, b) }}
            }}

            #[inline(always)]
            fn simd_ne(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{
                    let eq = {p}_cmpeq_epi32(a, b);
                    {p}_andnot_si{bits}(eq, {p}_set1_epi32(-1))
                }}
            }}

            #[inline(always)]
            fn simd_gt(self, a: {inner}, b: {inner}) -> {inner} {{
                // Unsigned comparison via bias trick: XOR both with 0x80000000
                // to convert to signed range, then use signed cmpgt.
                unsafe {{
                    let bias = {p}_set1_epi32(i32::MIN);
                    let sa = {p}_xor_si{bits}(a, bias);
                    let sb = {p}_xor_si{bits}(b, bias);
                    {p}_cmpgt_epi32(sa, sb)
                }}
            }}

            #[inline(always)]
            fn simd_lt(self, a: {inner}, b: {inner}) -> {inner} {{
                <Self as {trait_name}>::simd_gt(self, b, a)
            }}

            #[inline(always)]
            fn simd_le(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{
                    let gt = <Self as {trait_name}>::simd_gt(self, a, b);
                    {p}_andnot_si{bits}(gt, {p}_set1_epi32(-1))
                }}
            }}

            #[inline(always)]
            fn simd_ge(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{
                    let lt = <Self as {trait_name}>::simd_gt(self, b, a);
                    {p}_andnot_si{bits}(lt, {p}_set1_epi32(-1))
                }}
            }}

            #[inline(always)]
            fn blend(self, mask: {inner}, if_true: {inner}, if_false: {inner}) -> {inner} {{
                unsafe {{ {p}_blendv_epi8(if_false, if_true, mask) }}
            }}

            // ====== Reductions ======

            #[inline(always)]
            fn reduce_add(self, a: {inner}) -> u32 {{
        {reduce_add_body}
            }}

            // ====== Bitwise ======

            #[inline(always)]
            fn not(self, a: {inner}) -> {inner} {{
                unsafe {{ {p}_andnot_si{bits}(a, {p}_set1_epi32(-1)) }}
            }}

            #[inline(always)]
            fn bitand(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_and_si{bits}(a, b) }}
            }}

            #[inline(always)]
            fn bitor(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_or_si{bits}(a, b) }}
            }}

            #[inline(always)]
            fn bitxor(self, a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_xor_si{bits}(a, b) }}
            }}

            // ====== Shifts ======

            #[inline(always)]
            fn shl_const<const N: i32>(self, a: {inner}) -> {inner} {{
                unsafe {{ {p}_slli_epi32::<N>(a) }}
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(self, a: {inner}) -> {inner} {{
                unsafe {{ {p}_srli_epi32::<N>(a) }}
            }}

            // ====== Boolean ======

            #[inline(always)]
            fn all_true(self, a: {inner}) -> bool {{
                unsafe {{ {p}_movemask_ps({p}_castsi{bits}_ps(a)) == {all_mask} }}
            }}

            #[inline(always)]
            fn any_true(self, a: {inner}) -> bool {{
                unsafe {{ {p}_movemask_ps({p}_castsi{bits}_ps(a)) != 0 }}
            }}

            #[inline(always)]
            fn bitmask(self, a: {inner}) -> u32 {{
                unsafe {{ {p}_movemask_ps({p}_castsi{bits}_ps(a)) as u32 }}
            }}
        }}
    "#,
        all_mask = if lanes == 4 { "0xF" } else { "0xFF" },
    }
}

fn generate_x86_u32_reduce_add(ty: &U32VecType) -> String {
    match ty.width_bits {
        128 => formatdoc! {"
                unsafe {{
                    let hi = _mm_shuffle_epi32::<0b01_00_11_10>(a);
                    let sum = _mm_add_epi32(a, hi);
                    let hi2 = _mm_shuffle_epi32::<0b00_00_00_01>(sum);
                    let sum2 = _mm_add_epi32(sum, hi2);
                    _mm_cvtsi128_si32(sum2) as u32
                }}"},
        256 => formatdoc! {"
                unsafe {{
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

    // For infix operators like &, |, ^
    let binary_infix = |op: &str| -> String {
        let items: Vec<String> = (0..lanes).map(|i| format!("a[{i}] {op} b[{i}]")).collect();
        format!("[{}]", items.join(", "))
    };

    // For method-style ops like .wrapping_add(b)
    let binary_method = |method: &str| -> String {
        let items: Vec<String> = (0..lanes)
            .map(|i| format!("a[{i}].{method}(b[{i}])"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let min_lanes = || -> String {
        let items: Vec<String> = (0..lanes)
            .map(|i| format!("if a[{i}] < b[{i}] {{ a[{i}] }} else {{ b[{i}] }}"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let max_lanes = || -> String {
        let items: Vec<String> = (0..lanes)
            .map(|i| format!("if a[{i}] > b[{i}] {{ a[{i}] }} else {{ b[{i}] }}"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let reduce_add = || -> String {
        let items: Vec<String> = (0..lanes).map(|i| format!("a[{i}]")).collect();
        items.join(".wrapping_add(") + &")".repeat(lanes - 1)
    };

    let shl_lanes = || -> String {
        let items: Vec<String> = (0..lanes).map(|i| format!("a[{i}] << N")).collect();
        format!("[{}]", items.join(", "))
    };

    let shr_logical_lanes = || -> String {
        let items: Vec<String> = (0..lanes).map(|i| format!("a[{i}] >> N")).collect();
        format!("[{}]", items.join(", "))
    };

    let bitmask_expr = || -> String {
        let items: Vec<String> = (0..lanes)
            .map(|i| {
                if i == 0 {
                    "(a[0] >> 31)".to_string()
                } else {
                    format!("((a[{i}] >> 31) << {i})")
                }
            })
            .collect();
        items.join(" | ")
    };

    let all_true_expr = || -> String {
        let items: Vec<String> = (0..lanes).map(|i| format!("a[{i}] != 0")).collect();
        items.join(" && ")
    };

    let any_true_expr = || -> String {
        let items: Vec<String> = (0..lanes).map(|i| format!("a[{i}] != 0")).collect();
        items.join(" || ")
    };

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
                {add_lanes}
            }}

            #[inline(always)]
            fn sub(self, a: {array}, b: {array}) -> {array} {{
                {sub_lanes}
            }}

            #[inline(always)]
            fn mul(self, a: {array}, b: {array}) -> {array} {{
                {mul_lanes}
            }}

            // ====== Math ======

            #[inline(always)]
            fn min(self, a: {array}, b: {array}) -> {array} {{
                {min_lanes}
            }}

            #[inline(always)]
            fn max(self, a: {array}, b: {array}) -> {array} {{
                {max_lanes}
            }}

            // ====== Comparisons ======

            #[inline(always)]
            fn simd_eq(self, a: {array}, b: {array}) -> {array} {{
                let mut r = [0u32; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] == b[i] {{ u32::MAX }} else {{ 0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_ne(self, a: {array}, b: {array}) -> {array} {{
                let mut r = [0u32; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] != b[i] {{ u32::MAX }} else {{ 0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_lt(self, a: {array}, b: {array}) -> {array} {{
                let mut r = [0u32; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] < b[i] {{ u32::MAX }} else {{ 0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_le(self, a: {array}, b: {array}) -> {array} {{
                let mut r = [0u32; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] <= b[i] {{ u32::MAX }} else {{ 0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_gt(self, a: {array}, b: {array}) -> {array} {{
                let mut r = [0u32; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] > b[i] {{ u32::MAX }} else {{ 0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_ge(self, a: {array}, b: {array}) -> {array} {{
                let mut r = [0u32; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] >= b[i] {{ u32::MAX }} else {{ 0 }};
                }}
                r
            }}

            #[inline(always)]
            fn blend(self, mask: {array}, if_true: {array}, if_false: {array}) -> {array} {{
                let mut r = [0u32; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if mask[i] != 0 {{ if_true[i] }} else {{ if_false[i] }};
                }}
                r
            }}

            // ====== Reductions ======

            #[inline(always)]
            fn reduce_add(self, a: {array}) -> u32 {{
                {reduce_add}
            }}

            // ====== Bitwise ======

            #[inline(always)]
            fn not(self, a: {array}) -> {array} {{
                {not_lanes}
            }}

            #[inline(always)]
            fn bitand(self, a: {array}, b: {array}) -> {array} {{
                {and_lanes}
            }}

            #[inline(always)]
            fn bitor(self, a: {array}, b: {array}) -> {array} {{
                {or_lanes}
            }}

            #[inline(always)]
            fn bitxor(self, a: {array}, b: {array}) -> {array} {{
                {xor_lanes}
            }}

            // ====== Shifts ======

            #[inline(always)]
            fn shl_const<const N: i32>(self, a: {array}) -> {array} {{
                {shl}
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(self, a: {array}) -> {array} {{
                {shr_logical}
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
    "#,
        add_lanes = binary_method("wrapping_add"),
        sub_lanes = binary_method("wrapping_sub"),
        mul_lanes = binary_method("wrapping_mul"),
        min_lanes = min_lanes(),
        max_lanes = max_lanes(),
        reduce_add = reduce_add(),
        not_lanes = {
            let items: Vec<String> = (0..lanes).map(|i| format!("!a[{i}]")).collect();
            format!("[{}]", items.join(", "))
        },
        and_lanes = binary_infix("&"),
        or_lanes = binary_infix("|"),
        xor_lanes = binary_infix("^"),
        shl = shl_lanes(),
        shr_logical = shr_logical_lanes(),
        all_true = all_true_expr(),
        any_true = any_true_expr(),
        bitmask = bitmask_expr(),
    }
}

// ============================================================================
// NEON U32 Implementation Generation
// ============================================================================

fn generate_neon_u32_impls(types: &[U32VecType]) -> String {
    let mut code = String::new();

    for ty in types {
        code.push_str("\n#[cfg(target_arch = \"aarch64\")]\n");
        if ty.native_on_neon() {
            code.push_str(&generate_neon_native_u32_impl(ty));
        } else {
            code.push_str(&generate_neon_polyfill_u32_impl(ty));
        }
        code.push('\n');
    }

    code
}

fn generate_neon_native_u32_impl(ty: &U32VecType) -> String {
    let trait_name = ty.trait_name();
    let array = ty.array_type();
    let lanes = ty.lanes;

    formatdoc! {r#"
        impl {trait_name} for archmage::NeonToken {{
            type Repr = uint32x4_t;

            #[inline(always)]
            fn splat(self, v: u32) -> uint32x4_t {{
                unsafe {{ vdupq_n_u32(v) }}
            }}

            #[inline(always)]
            fn zero(self) -> uint32x4_t {{
                unsafe {{ vdupq_n_u32(0) }}
            }}

            #[inline(always)]
            fn load(self, data: &{array}) -> uint32x4_t {{
                unsafe {{ vld1q_u32(data.as_ptr()) }}
            }}

            #[inline(always)]
            fn from_array(self, arr: {array}) -> uint32x4_t {{
                unsafe {{ vld1q_u32(arr.as_ptr()) }}
            }}

            #[inline(always)]
            fn store(self, repr: uint32x4_t, out: &mut {array}) {{
                unsafe {{ vst1q_u32(out.as_mut_ptr(), repr) }};
            }}

            #[inline(always)]
            fn to_array(self, repr: uint32x4_t) -> {array} {{
                let mut out = [0u32; {lanes}];
                unsafe {{ vst1q_u32(out.as_mut_ptr(), repr) }};
                out
            }}

            #[inline(always)]
            fn add(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{ unsafe {{ vaddq_u32(a, b) }} }}
            #[inline(always)]
            fn sub(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{ unsafe {{ vsubq_u32(a, b) }} }}
            #[inline(always)]
            fn mul(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{ unsafe {{ vmulq_u32(a, b) }} }}
            #[inline(always)]
            fn min(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{ unsafe {{ vminq_u32(a, b) }} }}
            #[inline(always)]
            fn max(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{ unsafe {{ vmaxq_u32(a, b) }} }}

            #[inline(always)]
            fn simd_eq(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{
                unsafe {{ vceqq_u32(a, b) }}
            }}
            #[inline(always)]
            fn simd_ne(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{
                unsafe {{ vmvnq_u32(vceqq_u32(a, b)) }}
            }}
            #[inline(always)]
            fn simd_lt(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{
                unsafe {{ vcltq_u32(a, b) }}
            }}
            #[inline(always)]
            fn simd_le(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{
                unsafe {{ vcleq_u32(a, b) }}
            }}
            #[inline(always)]
            fn simd_gt(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{
                unsafe {{ vcgtq_u32(a, b) }}
            }}
            #[inline(always)]
            fn simd_ge(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{
                unsafe {{ vcgeq_u32(a, b) }}
            }}

            #[inline(always)]
            fn blend(self, mask: uint32x4_t, if_true: uint32x4_t, if_false: uint32x4_t) -> uint32x4_t {{
                unsafe {{ vbslq_u32(mask, if_true, if_false) }}
            }}

            #[inline(always)]
            fn reduce_add(self, a: uint32x4_t) -> u32 {{
                unsafe {{ vaddvq_u32(a) }}
            }}

            #[inline(always)]
            fn not(self, a: uint32x4_t) -> uint32x4_t {{
                unsafe {{ vmvnq_u32(a) }}
            }}
            #[inline(always)]
            fn bitand(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{
                unsafe {{ vandq_u32(a, b) }}
            }}
            #[inline(always)]
            fn bitor(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{
                unsafe {{ vorrq_u32(a, b) }}
            }}
            #[inline(always)]
            fn bitxor(self, a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{
                unsafe {{ veorq_u32(a, b) }}
            }}

            #[inline(always)]
            fn shl_const<const N: i32>(self, a: uint32x4_t) -> uint32x4_t {{
                unsafe {{ vshlq_n_u32::<N>(a) }}
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(self, a: uint32x4_t) -> uint32x4_t {{
                const {{ assert!(N >= 0 && N <= 31) }};
                unsafe {{ vshlq_u32(a, vdupq_n_s32(-N)) }}
            }}

            #[inline(always)]
            fn all_true(self, a: uint32x4_t) -> bool {{
                unsafe {{ vminvq_u32(a) == u32::MAX }}
            }}

            #[inline(always)]
            fn any_true(self, a: uint32x4_t) -> bool {{
                unsafe {{ vmaxvq_u32(a) != 0 }}
            }}

            #[inline(always)]
            fn bitmask(self, a: uint32x4_t) -> u32 {{
                unsafe {{
                    // Extract sign bit of each 32-bit lane
                    let shift = vshrq_n_u32::<31>(a);
                    // Pack: lane0 | (lane1<<1) | (lane2<<2) | (lane3<<3)
                    let lane0 = vgetq_lane_u32::<0>(shift);
                    let lane1 = vgetq_lane_u32::<1>(shift);
                    let lane2 = vgetq_lane_u32::<2>(shift);
                    let lane3 = vgetq_lane_u32::<3>(shift);
                    lane0 | (lane1 << 1) | (lane2 << 2) | (lane3 << 3)
                }}
            }}
        }}
    "#}
}

fn generate_neon_polyfill_u32_impl(ty: &U32VecType) -> String {
    let trait_name = ty.trait_name();
    let repr = ty.neon_repr();
    let array = ty.array_type();
    let lanes = ty.lanes;
    let sub_count = ty.sub_count();

    let binary_op = |intrinsic: &str| -> String {
        let items: Vec<String> = (0..sub_count)
            .map(|i| format!("{intrinsic}(a[{i}], b[{i}])"))
            .collect();
        format!("unsafe {{ [{}] }}", items.join(", "))
    };

    let unary_op = |intrinsic: &str| -> String {
        let items: Vec<String> = (0..sub_count)
            .map(|i| format!("{intrinsic}(a[{i}])"))
            .collect();
        format!("unsafe {{ [{}] }}", items.join(", "))
    };

    let cmp_op = |intrinsic: &str| -> String {
        let items: Vec<String> = (0..sub_count)
            .map(|i| format!("{intrinsic}(a[{i}], b[{i}])"))
            .collect();
        format!("unsafe {{ [{}] }}", items.join(", "))
    };

    let ne_op = || -> String {
        let items: Vec<String> = (0..sub_count)
            .map(|i| format!("vmvnq_u32(vceqq_u32(a[{i}], b[{i}]))"))
            .collect();
        format!("unsafe {{ [{}] }}", items.join(", "))
    };

    formatdoc! {r#"
        impl {trait_name} for archmage::NeonToken {{
            type Repr = {repr};

            #[inline(always)]
            fn splat(self, v: u32) -> {repr} {{
                unsafe {{
                    let v4 = vdupq_n_u32(v);
                    [{v4_copies}]
                }}
            }}

            #[inline(always)]
            fn zero(self) -> {repr} {{
                unsafe {{
                    let z = vdupq_n_u32(0);
                    [{z_copies}]
                }}
            }}

            #[inline(always)]
            fn load(self, data: &{array}) -> {repr} {{
                unsafe {{
                    [{load_lanes}]
                }}
            }}

            #[inline(always)]
            fn from_array(self, arr: {array}) -> {repr} {{
                <Self as {trait_name}>::load(self, &arr)
            }}

            #[inline(always)]
            fn store(self, repr: {repr}, out: &mut {array}) {{
                unsafe {{
                    {store_lanes}
                }}
            }}

            #[inline(always)]
            fn to_array(self, repr: {repr}) -> {array} {{
                let mut out = [0u32; {lanes}];
                <Self as {trait_name}>::store(self, repr, &mut out);
                out
            }}

            #[inline(always)]
            fn add(self, a: {repr}, b: {repr}) -> {repr} {{ {add} }}
            #[inline(always)]
            fn sub(self, a: {repr}, b: {repr}) -> {repr} {{ {sub} }}
            #[inline(always)]
            fn mul(self, a: {repr}, b: {repr}) -> {repr} {{ {mul} }}
            #[inline(always)]
            fn min(self, a: {repr}, b: {repr}) -> {repr} {{ {min} }}
            #[inline(always)]
            fn max(self, a: {repr}, b: {repr}) -> {repr} {{ {max} }}

            #[inline(always)]
            fn simd_eq(self, a: {repr}, b: {repr}) -> {repr} {{ {eq} }}
            #[inline(always)]
            fn simd_ne(self, a: {repr}, b: {repr}) -> {repr} {{ {ne} }}
            #[inline(always)]
            fn simd_lt(self, a: {repr}, b: {repr}) -> {repr} {{ {lt} }}
            #[inline(always)]
            fn simd_le(self, a: {repr}, b: {repr}) -> {repr} {{ {le} }}
            #[inline(always)]
            fn simd_gt(self, a: {repr}, b: {repr}) -> {repr} {{ {gt} }}
            #[inline(always)]
            fn simd_ge(self, a: {repr}, b: {repr}) -> {repr} {{ {ge} }}

            #[inline(always)]
            fn blend(self, mask: {repr}, if_true: {repr}, if_false: {repr}) -> {repr} {{
                {blend}
            }}

            #[inline(always)]
            fn reduce_add(self, a: {repr}) -> u32 {{
                {reduce_add}
            }}

            #[inline(always)]
            fn not(self, a: {repr}) -> {repr} {{ {not} }}
            #[inline(always)]
            fn bitand(self, a: {repr}, b: {repr}) -> {repr} {{ {bitand} }}
            #[inline(always)]
            fn bitor(self, a: {repr}, b: {repr}) -> {repr} {{ {bitor} }}
            #[inline(always)]
            fn bitxor(self, a: {repr}, b: {repr}) -> {repr} {{ {bitxor} }}

            #[inline(always)]
            fn shl_const<const N: i32>(self, a: {repr}) -> {repr} {{
                {shl}
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(self, a: {repr}) -> {repr} {{
                {shr_logic}
            }}

            #[inline(always)]
            fn all_true(self, a: {repr}) -> bool {{
                {all_true}
            }}

            #[inline(always)]
            fn any_true(self, a: {repr}) -> bool {{
                {any_true}
            }}

            #[inline(always)]
            fn bitmask(self, a: {repr}) -> u32 {{
                {bitmask}
            }}
        }}
    "#,
        v4_copies = (0..sub_count).map(|_| "v4").collect::<Vec<_>>().join(", "),
        z_copies = (0..sub_count).map(|_| "z").collect::<Vec<_>>().join(", "),
        load_lanes = (0..sub_count)
            .map(|i| format!("vld1q_u32(data.as_ptr().add({}))", i * 4))
            .collect::<Vec<_>>().join(", "),
        store_lanes = (0..sub_count)
            .map(|i| format!("vst1q_u32(out.as_mut_ptr().add({}), repr[{i}]);", i * 4))
            .collect::<Vec<_>>().join("\n            "),
        add = binary_op("vaddq_u32"),
        sub = binary_op("vsubq_u32"),
        mul = binary_op("vmulq_u32"),
        min = binary_op("vminq_u32"),
        max = binary_op("vmaxq_u32"),
        eq = cmp_op("vceqq_u32"),
        ne = ne_op(),
        lt = cmp_op("vcltq_u32"),
        le = cmp_op("vcleq_u32"),
        gt = cmp_op("vcgtq_u32"),
        ge = cmp_op("vcgeq_u32"),
        blend = {
            let items: Vec<String> = (0..sub_count)
                .map(|i| format!("vbslq_u32(mask[{i}], if_true[{i}], if_false[{i}])"))
                .collect();
            format!("unsafe {{ [{}] }}", items.join(", "))
        },
        reduce_add = {
            let mut body = "unsafe {\n".to_string();
            body.push_str("            let m = vaddq_u32(a[0], a[1]);\n");
            for i in 2..sub_count {
                body.push_str(&format!("            let m = vaddq_u32(m, a[{i}]);\n"));
            }
            body.push_str("            vaddvq_u32(m)\n");
            body.push_str("        }");
            body
        },
        not = unary_op("vmvnq_u32"),
        bitand = binary_op("vandq_u32"),
        bitor = binary_op("vorrq_u32"),
        bitxor = binary_op("veorq_u32"),
        shl = {
            let items: Vec<String> = (0..sub_count)
                .map(|i| format!("vshlq_n_u32::<N>(a[{i}])"))
                .collect();
            format!("unsafe {{ [{}] }}", items.join(", "))
        },
        shr_logic = {
            let items: Vec<String> = (0..sub_count)
                .map(|i| format!("vshlq_u32(a[{i}], vdupq_n_s32(-N))"))
                .collect();
            format!(
                "const {{ assert!(N >= 0 && N <= 31) }};\n                unsafe {{ [{}] }}",
                items.join(", ")
            )
        },
        all_true = {
            let items: Vec<String> = (0..sub_count)
                .map(|i| format!("vminvq_u32(a[{i}]) == u32::MAX"))
                .collect();
            format!("unsafe {{ {} }}", items.join(" && "))
        },
        any_true = {
            let items: Vec<String> = (0..sub_count)
                .map(|i| format!("vmaxvq_u32(a[{i}]) != 0"))
                .collect();
            format!("unsafe {{ {} }}", items.join(" || "))
        },
        bitmask = {
            let mut body = "unsafe {\n".to_string();
            body.push_str("            let mut bits = 0u32;\n");
            for i in 0..sub_count {
                let base = i * 4;
                body.push_str(&format!("            let s{i} = vshrq_n_u32::<31>(a[{i}]);\n"));
                body.push_str(&format!("            bits |= vgetq_lane_u32::<0>(s{i}) << {base};\n"));
                body.push_str(&format!("            bits |= (vgetq_lane_u32::<1>(s{i})) << {};\n", base + 1));
                body.push_str(&format!("            bits |= (vgetq_lane_u32::<2>(s{i})) << {};\n", base + 2));
                body.push_str(&format!("            bits |= (vgetq_lane_u32::<3>(s{i})) << {};\n", base + 3));
            }
            body.push_str("            bits\n");
            body.push_str("        }");
            body
        },
    }
}

// ============================================================================
// WASM U32 Implementation Generation
// ============================================================================

fn generate_wasm_u32_impls(types: &[U32VecType]) -> String {
    let mut code = String::new();

    for ty in types {
        code.push_str("\n#[cfg(target_arch = \"wasm32\")]\n");
        if ty.native_on_wasm() {
            code.push_str(&generate_wasm_native_u32_impl(ty));
        } else {
            code.push_str(&generate_wasm_polyfill_u32_impl(ty));
        }
        code.push('\n');
    }

    code
}

fn generate_wasm_native_u32_impl(ty: &U32VecType) -> String {
    let trait_name = ty.trait_name();
    let array = ty.array_type();
    let lanes = ty.lanes;

    let reduce_add_body = (0..lanes)
        .map(|j| format!("(i32x4_extract_lane::<{j}>(a) as u32)"))
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
            fn load(self, data: &{array}) -> v128 {{ unsafe {{ v128_load(data.as_ptr().cast()) }} }}
            #[inline(always)]
            fn from_array(self, arr: {array}) -> v128 {{ unsafe {{ v128_load(arr.as_ptr().cast()) }} }}
            #[inline(always)]
            fn store(self, repr: v128, out: &mut {array}) {{ unsafe {{ v128_store(out.as_mut_ptr().cast(), repr) }}; }}
            #[inline(always)]
            fn to_array(self, repr: v128) -> {array} {{
                let mut out = [0u32; {lanes}];
                unsafe {{ v128_store(out.as_mut_ptr().cast(), repr) }};
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

            #[inline(always)]
            fn all_true(self, a: v128) -> bool {{ i32x4_all_true(a) }}
            #[inline(always)]
            fn any_true(self, a: v128) -> bool {{ v128_any_true(a) }}
            #[inline(always)]
            fn bitmask(self, a: v128) -> u32 {{ i32x4_bitmask(a) as u32 }}
        }}
    "#}
}

fn generate_wasm_polyfill_u32_impl(ty: &U32VecType) -> String {
    let trait_name = ty.trait_name();
    let repr = ty.wasm_repr();
    let array = ty.array_type();
    let lanes = ty.lanes;
    let sub_count = ty.sub_count();

    let binary_op = |func: &str| -> String {
        let items: Vec<String> = (0..sub_count)
            .map(|i| format!("{func}(a[{i}], b[{i}])"))
            .collect();
        format!("[{}]", items.join(", "))
    };

    let unary_op = |func: &str| -> String {
        let items: Vec<String> = (0..sub_count).map(|i| format!("{func}(a[{i}])")).collect();
        format!("[{}]", items.join(", "))
    };

    let reduce_add_body = || -> String {
        let mut items = Vec::new();
        for i in 0..sub_count {
            for j in 0..4usize {
                items.push(format!("(i32x4_extract_lane::<{j}>(a[{i}]) as u32)"));
            }
        }
        items.join(".wrapping_add(") + &")".repeat(items.len() - 1)
    };

    formatdoc! {r#"
        impl {trait_name} for archmage::Wasm128Token {{
            type Repr = {repr};

            #[inline(always)]
            fn splat(self, v: u32) -> {repr} {{
                let v4 = u32x4_splat(v);
                [{v4_copies}]
            }}

            #[inline(always)]
            fn zero(self) -> {repr} {{
                let z = u32x4_splat(0);
                [{z_copies}]
            }}

            #[inline(always)]
            fn load(self, data: &{array}) -> {repr} {{
                unsafe {{
                    [{load_lanes}]
                }}
            }}

            #[inline(always)]
            fn from_array(self, arr: {array}) -> {repr} {{
                <Self as {trait_name}>::load(self, &arr)
            }}

            #[inline(always)]
            fn store(self, repr: {repr}, out: &mut {array}) {{
                unsafe {{
                    {store_lanes}
                }}
            }}

            #[inline(always)]
            fn to_array(self, repr: {repr}) -> {array} {{
                let mut out = [0u32; {lanes}];
                <Self as {trait_name}>::store(self, repr, &mut out);
                out
            }}

            #[inline(always)]
            fn add(self, a: {repr}, b: {repr}) -> {repr} {{ {add} }}
            #[inline(always)]
            fn sub(self, a: {repr}, b: {repr}) -> {repr} {{ {sub} }}
            #[inline(always)]
            fn mul(self, a: {repr}, b: {repr}) -> {repr} {{ {mul} }}
            #[inline(always)]
            fn min(self, a: {repr}, b: {repr}) -> {repr} {{ {min} }}
            #[inline(always)]
            fn max(self, a: {repr}, b: {repr}) -> {repr} {{ {max} }}

            #[inline(always)]
            fn simd_eq(self, a: {repr}, b: {repr}) -> {repr} {{ {eq} }}
            #[inline(always)]
            fn simd_ne(self, a: {repr}, b: {repr}) -> {repr} {{ {ne} }}
            #[inline(always)]
            fn simd_lt(self, a: {repr}, b: {repr}) -> {repr} {{ {lt} }}
            #[inline(always)]
            fn simd_le(self, a: {repr}, b: {repr}) -> {repr} {{ {le} }}
            #[inline(always)]
            fn simd_gt(self, a: {repr}, b: {repr}) -> {repr} {{ {gt} }}
            #[inline(always)]
            fn simd_ge(self, a: {repr}, b: {repr}) -> {repr} {{ {ge} }}
            #[inline(always)]
            fn blend(self, mask: {repr}, if_true: {repr}, if_false: {repr}) -> {repr} {{
                [{blend_lanes}]
            }}

            #[inline(always)]
            fn reduce_add(self, a: {repr}) -> u32 {{ {reduce_add} }}

            #[inline(always)]
            fn not(self, a: {repr}) -> {repr} {{ {not} }}
            #[inline(always)]
            fn bitand(self, a: {repr}, b: {repr}) -> {repr} {{ {and} }}
            #[inline(always)]
            fn bitor(self, a: {repr}, b: {repr}) -> {repr} {{ {or} }}
            #[inline(always)]
            fn bitxor(self, a: {repr}, b: {repr}) -> {repr} {{ {xor} }}

            #[inline(always)]
            fn shl_const<const N: i32>(self, a: {repr}) -> {repr} {{
                [{shl_lanes}]
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(self, a: {repr}) -> {repr} {{
                [{shr_logic_lanes}]
            }}

            #[inline(always)]
            fn all_true(self, a: {repr}) -> bool {{
                {all_true}
            }}

            #[inline(always)]
            fn any_true(self, a: {repr}) -> bool {{
                {any_true}
            }}

            #[inline(always)]
            fn bitmask(self, a: {repr}) -> u32 {{
                {bitmask}
            }}
        }}
    "#,
        v4_copies = (0..sub_count).map(|_| "v4").collect::<Vec<_>>().join(", "),
        z_copies = (0..sub_count).map(|_| "z").collect::<Vec<_>>().join(", "),
        load_lanes = (0..sub_count)
            .map(|i| format!("v128_load(data.as_ptr().add({}).cast())", i * 4))
            .collect::<Vec<_>>().join(", "),
        store_lanes = (0..sub_count)
            .map(|i| format!("v128_store(out.as_mut_ptr().add({}).cast(), repr[{i}]);", i * 4))
            .collect::<Vec<_>>().join("\n            "),
        add = binary_op("i32x4_add"),
        sub = binary_op("i32x4_sub"),
        mul = binary_op("i32x4_mul"),
        min = binary_op("u32x4_min"),
        max = binary_op("u32x4_max"),
        eq = binary_op("i32x4_eq"),
        ne = binary_op("i32x4_ne"),
        lt = binary_op("u32x4_lt"),
        le = binary_op("u32x4_le"),
        gt = binary_op("u32x4_gt"),
        ge = binary_op("u32x4_ge"),
        blend_lanes = (0..sub_count)
            .map(|i| format!("v128_bitselect(if_true[{i}], if_false[{i}], mask[{i}])"))
            .collect::<Vec<_>>().join(", "),
        reduce_add = reduce_add_body(),
        not = unary_op("v128_not"),
        and = binary_op("v128_and"),
        or = binary_op("v128_or"),
        xor = binary_op("v128_xor"),
        shl_lanes = (0..sub_count)
            .map(|i| format!("u32x4_shl(a[{i}], N as u32)"))
            .collect::<Vec<_>>().join(", "),
        shr_logic_lanes = (0..sub_count)
            .map(|i| format!("u32x4_shr(a[{i}], N as u32)"))
            .collect::<Vec<_>>().join(", "),
        all_true = (0..sub_count)
            .map(|i| format!("i32x4_all_true(a[{i}])"))
            .collect::<Vec<_>>().join(" && "),
        any_true = (0..sub_count)
            .map(|i| format!("v128_any_true(a[{i}])"))
            .collect::<Vec<_>>().join(" || "),
        bitmask = {
            let items: Vec<String> = (0..sub_count)
                .map(|i| format!("((i32x4_bitmask(a[{i}]) as u32) << {})", i * 4))
                .collect();
            items.join(" | ")
        },
    }
}
