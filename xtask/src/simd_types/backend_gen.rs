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
            + &generate_x86_v3_w512_impls(&w512_types),
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
            + &generate_scalar_w512_impls(&w512_types),
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
            + &generate_neon_w512_impls(&w512_types),
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
            + &generate_wasm_w512_impls(&w512_types),
    );

    // 10b. x86 V4 native AVX-512 implementation (W512 types only)
    files.insert(
        "impls/x86_v4.rs".to_string(),
        generate_x86_v4_impls_file(&w512_types),
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
        impl Sealed for archmage::Avx512ModernToken {{}}
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
        /// Trait methods are **associated functions** (no `self`/token parameter).
        /// The implementing type `Self` (a token type) determines which platform
        /// intrinsics are used. All methods are `#[inline(always)]` in implementations.
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
            fn splat(v: {elem}) -> Self::Repr;

            /// All lanes zero.
            fn zero() -> Self::Repr;

            /// Load from an aligned array.
            fn load(data: &{array}) -> Self::Repr;

            /// Create from array (zero-cost transmute where possible).
            fn from_array(arr: {array}) -> Self::Repr;

            /// Store to array.
            fn store(repr: Self::Repr, out: &mut {array});

            /// Convert to array.
            fn to_array(repr: Self::Repr) -> {array};

            // ====== Arithmetic ======

            /// Lane-wise addition.
            fn add(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise subtraction.
            fn sub(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise multiplication.
            fn mul(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise division.
            fn div(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise negation.
            fn neg(a: Self::Repr) -> Self::Repr;

            // ====== Math ======

            /// Lane-wise minimum.
            fn min(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise maximum.
            fn max(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Square root.
            fn sqrt(a: Self::Repr) -> Self::Repr;

            /// Absolute value.
            fn abs(a: Self::Repr) -> Self::Repr;

            /// Round toward negative infinity.
            fn floor(a: Self::Repr) -> Self::Repr;

            /// Round toward positive infinity.
            fn ceil(a: Self::Repr) -> Self::Repr;

            /// Round to nearest integer.
            fn round(a: Self::Repr) -> Self::Repr;

            /// Fused multiply-add: a * b + c.
            fn mul_add(a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;

            /// Fused multiply-sub: a * b - c.
            fn mul_sub(a: Self::Repr, b: Self::Repr, c: Self::Repr) -> Self::Repr;

            // ====== Comparisons ======
            // Return masks where each lane is all-1s (true) or all-0s (false).

            /// Lane-wise equality.
            fn simd_eq(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise inequality.
            fn simd_ne(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise less-than.
            fn simd_lt(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise less-than-or-equal.
            fn simd_le(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise greater-than.
            fn simd_gt(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise greater-than-or-equal.
            fn simd_ge(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Select lanes: where mask is all-1s pick `if_true`, else `if_false`.
            fn blend(mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;

            // ====== Reductions ======

            /// Sum all {lanes} lanes.
            fn reduce_add(a: Self::Repr) -> {elem};

            /// Minimum across all {lanes} lanes.
            fn reduce_min(a: Self::Repr) -> {elem};

            /// Maximum across all {lanes} lanes.
            fn reduce_max(a: Self::Repr) -> {elem};

            // ====== Approximations ======

            /// Fast reciprocal approximation (~12-bit precision where available).
            ///
            /// On platforms without native approximation, falls back to full division.
            fn rcp_approx(a: Self::Repr) -> Self::Repr {{
                Self::div(Self::splat(1.0), a)
            }}

            /// Fast reciprocal square root approximation (~12-bit precision where available).
            ///
            /// On platforms without native approximation, falls back to 1/sqrt.
            fn rsqrt_approx(a: Self::Repr) -> Self::Repr {{
                Self::div(Self::splat(1.0), Self::sqrt(a))
            }}

            // ====== Bitwise ======

            /// Bitwise NOT.
            fn not(a: Self::Repr) -> Self::Repr;

            /// Bitwise AND.
            fn bitand(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Bitwise OR.
            fn bitor(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Bitwise XOR.
            fn bitxor(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            // ====== Default implementations ======

            /// Clamp values between lo and hi.
            #[inline(always)]
            fn clamp(a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {{
                Self::min(Self::max(a, lo), hi)
            }}

            /// Precise reciprocal (Newton-Raphson from rcp_approx).
            #[inline(always)]
            fn recip(a: Self::Repr) -> Self::Repr {{
                let approx = Self::rcp_approx(a);
                let two = Self::splat(2.0);
                // x' = x * (2 - a*x)
                Self::mul(approx, Self::sub(two, Self::mul(a, approx)))
            }}

            /// Precise reciprocal square root (Newton-Raphson from rsqrt_approx).
            #[inline(always)]
            fn rsqrt(a: Self::Repr) -> Self::Repr {{
                let approx = Self::rsqrt_approx(a);
                let half = Self::splat(0.5);
                let three = Self::splat(3.0);
                // y' = 0.5 * y * (3 - x * y * y)
                Self::mul(
                    Self::mul(half, approx),
                    Self::sub(three, Self::mul(a, Self::mul(approx, approx))),
                )
            }}
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
    for ty in w512_types {
        let name = ty.name();
        let trait_name = ty.trait_name();
        code.push_str(&format!("mod {name};\npub use {name}::{trait_name};\n\n"));
    }

    // Extension traits (popcnt for Modern token)
    code.push_str("#[cfg(feature = \"avx512\")]\n");
    code.push_str("mod popcnt;\n");
    code.push_str("#[cfg(feature = \"avx512\")]\n");
    code.push_str("pub use popcnt::*;\n\n");

    // Conversion and bitcast traits (float/i32/u32/i64)
    code.push_str("mod convert;\npub use convert::{F32x4Convert, F32x8Convert, U32x4Bitcast, U32x8Bitcast, I64x2Bitcast, I64x4Bitcast};\n\n");

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
            "avx512_modern",
            "archmage::Avx512ModernToken",
            "AVX-512 with modern extensions.",
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

        #[cfg(target_arch = "aarch64")]
        mod arm_neon;

        #[cfg(target_arch = "wasm32")]
        mod wasm128;

        mod scalar;
    "#}
}

/// Generate the x86 V4 (native AVX-512) implementation file.
///
/// Contains W512 backend impls for both X64V4Token and Avx512ModernToken,
/// plus Modern-specific extension impls (popcnt).
/// W128 and W256 types use X64V3Token (V4 downcasts to V3 for narrower widths).
fn generate_x86_v4_impls_file(w512_types: &[super::backend_gen_w512::W512Type]) -> String {
    use super::backend_gen_w512::{
        generate_popcnt_impls, generate_x86_modern_w512_impls, generate_x86_v4_w512_impls,
    };

    let mut code = formatdoc! {r#"
        //! Backend implementations for X64V4Token and Avx512ModernToken (native AVX-512).
        //!
        //! Implements the W512 backend traits using native 512-bit AVX-512 intrinsics
        //! for both X64V4Token (base AVX-512) and Avx512ModernToken (+ VPOPCNTDQ, BITALG, etc.).
        //!
        //! Avx512ModernToken also gets extension trait impls (popcnt) for Modern-only features.
        //!
        //! W128 and W256 types use X64V3Token (V4 downcasts to V3 for narrower widths).
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.

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

    // Base backend impls for Avx512ModernToken (identical intrinsics, different token)
    code.push_str(
        "\n// ============================================================================\n",
    );
    code.push_str("// Avx512ModernToken — base AVX-512 (same intrinsics as V4)\n");
    code.push_str(
        "// ============================================================================\n\n",
    );
    code.push_str(&generate_x86_modern_w512_impls(w512_types));

    // Extension impls: popcnt (Avx512ModernToken only)
    code.push_str(
        "\n// ============================================================================\n",
    );
    code.push_str("// Avx512ModernToken — extension: popcnt (VPOPCNTDQ + BITALG)\n");
    code.push_str(
        "// ============================================================================\n\n",
    );
    code.push_str(&generate_popcnt_impls(w512_types));

    code
}

// ============================================================================
// x86 Implementation Generation
// ============================================================================

fn generate_x86_impls(types: &[FloatVecType], token: &str, max_width: usize) -> String {
    let mut code = formatdoc! {r#"
        //! Backend implementations for {token} (x86-64).
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.

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
        "0x7FFF_FFFFu32 as i32"
    } else {
        "0x7FFF_FFFF_FFFF_FFFFu64 as i64"
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
            #[inline(always)]
            fn rcp_approx(a: {inner}) -> {inner} {{
                unsafe {{ {p}_{rcp_fn}_{s}(a) }}
            }}

            #[inline(always)]
            fn rsqrt_approx(a: {inner}) -> {inner} {{
                unsafe {{ {p}_{rsqrt_fn}_{s}(a) }}
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

    formatdoc! {r#"
        impl {trait_name} for archmage::{token} {{
            type Repr = {inner};

            // ====== Construction ======

            #[inline(always)]
            fn splat(v: {elem}) -> {inner} {{
                unsafe {{ {set1}(v) }}
            }}

            #[inline(always)]
            fn zero() -> {inner} {{
                unsafe {{ {setzero}() }}
            }}

            #[inline(always)]
            fn load(data: &{array}) -> {inner} {{
                unsafe {{ {p}_loadu_{s}(data.as_ptr()) }}
            }}

            #[inline(always)]
            fn from_array(arr: {array}) -> {inner} {{
                // SAFETY: {array} and {inner} have identical size and layout.
                unsafe {{ core::mem::transmute(arr) }}
            }}

            #[inline(always)]
            fn store(repr: {inner}, out: &mut {array}) {{
                unsafe {{ {p}_storeu_{s}(out.as_mut_ptr(), repr) }};
            }}

            #[inline(always)]
            fn to_array(repr: {inner}) -> {array} {{
                let mut out = [{zero_lit}; {lanes}];
                unsafe {{ {p}_storeu_{s}(out.as_mut_ptr(), repr) }};
                out
            }}

            // ====== Arithmetic ======

            #[inline(always)]
            fn add(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_add_{s}(a, b) }}
            }}

            #[inline(always)]
            fn sub(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_sub_{s}(a, b) }}
            }}

            #[inline(always)]
            fn mul(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_mul_{s}(a, b) }}
            }}

            #[inline(always)]
            fn div(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_div_{s}(a, b) }}
            }}

            #[inline(always)]
            fn neg(a: {inner}) -> {inner} {{
                unsafe {{ {p}_sub_{s}({setzero}(), a) }}
            }}

            // ====== Math ======

            #[inline(always)]
            fn min(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_min_{s}(a, b) }}
            }}

            #[inline(always)]
            fn max(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_max_{s}(a, b) }}
            }}

            #[inline(always)]
            fn sqrt(a: {inner}) -> {inner} {{
                unsafe {{ {p}_sqrt_{s}(a) }}
            }}

            #[inline(always)]
            fn abs(a: {inner}) -> {inner} {{
                unsafe {{
                    let mask = {cast_int_to_float}({p}_set1_{set1_int}({abs_mask}));
                    {p}_and_{s}(a, mask)
                }}
            }}

            #[inline(always)]
            fn floor(a: {inner}) -> {inner} {{
                unsafe {{ {floor_intr}(a) }}
            }}

            #[inline(always)]
            fn ceil(a: {inner}) -> {inner} {{
                unsafe {{ {ceil_intr}(a) }}
            }}

            #[inline(always)]
            fn round(a: {inner}) -> {inner} {{
                unsafe {{ {round_intr}(a) }}
            }}

            #[inline(always)]
            fn mul_add(a: {inner}, b: {inner}, c: {inner}) -> {inner} {{
                unsafe {{ {p}_fmadd_{s}(a, b, c) }}
            }}

            #[inline(always)]
            fn mul_sub(a: {inner}, b: {inner}, c: {inner}) -> {inner} {{
                unsafe {{ {p}_fmsub_{s}(a, b, c) }}
            }}

            // ====== Comparisons ======

            #[inline(always)]
            fn simd_eq(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {cmp}::<_CMP_EQ_OQ>(a, b) }}
            }}

            #[inline(always)]
            fn simd_ne(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {cmp}::<_CMP_NEQ_OQ>(a, b) }}
            }}

            #[inline(always)]
            fn simd_lt(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {cmp}::<_CMP_LT_OQ>(a, b) }}
            }}

            #[inline(always)]
            fn simd_le(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {cmp}::<_CMP_LE_OQ>(a, b) }}
            }}

            #[inline(always)]
            fn simd_gt(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {cmp}::<_CMP_GT_OQ>(a, b) }}
            }}

            #[inline(always)]
            fn simd_ge(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {cmp}::<_CMP_GE_OQ>(a, b) }}
            }}

            #[inline(always)]
            fn blend(mask: {inner}, if_true: {inner}, if_false: {inner}) -> {inner} {{
                unsafe {{ {p}_blendv_{s}(if_false, if_true, mask) }}
            }}

            // ====== Reductions ======

            #[inline(always)]
            fn reduce_add(a: {inner}) -> {elem} {{
        {reduce_add_body}
            }}

            #[inline(always)]
            fn reduce_min(a: {inner}) -> {elem} {{
        {reduce_min_body}
            }}

            #[inline(always)]
            fn reduce_max(a: {inner}) -> {elem} {{
        {reduce_max_body}
            }}

            // ====== Approximations ======

        {approx_section}
            // ====== Bitwise ======

            #[inline(always)]
            fn not(a: {inner}) -> {inner} {{
                unsafe {{
                    let ones = {p}_set1_{set1_int}(-1);
                    let as_int = {cast_float_to_int}(a);
                    {cast_int_to_float}({p}_xor_si{bits}(as_int, ones))
                }}
            }}

            #[inline(always)]
            fn bitand(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_and_{s}(a, b) }}
            }}

            #[inline(always)]
            fn bitor(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_or_{s}(a, b) }}
            }}

            #[inline(always)]
            fn bitxor(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_xor_{s}(a, b) }}
            }}
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

    match (ty.elem, ty.width_bits) {
        ("f32", 128) => formatdoc! {"
                unsafe {{
                    let h1 = {hadd}(a, a);
                    let h2 = {hadd}(h1, h1);
                    {cvt}(h2)
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
                    let sum = _mm_add_{s}(lo, hi);
                    let h1 = {hadd}(sum, sum);
                    let h2 = {hadd}(h1, h1);
                    {cvt}(h2)
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

        use crate::simd::backends::*;

    "#};

    // Emit helper functions once per element type (not per vector type)
    let mut seen_elems = std::collections::BTreeSet::new();
    for ty in types {
        if seen_elems.insert(ty.elem) {
            code.push_str(&formatdoc! {r#"
                // Helpers to avoid trait method name shadowing inside the impl block.
                // Inside `impl XxxBackend`, names like `sqrt`, `floor`, etc. resolve to
                // the trait's associated functions instead of {elem}'s inherent methods.
                #[inline(always)]
                fn {elem}_sqrt(x: {elem}) -> {elem} {{
                    x.sqrt()
                }}
                #[inline(always)]
                fn {elem}_floor(x: {elem}) -> {elem} {{
                    x.floor()
                }}
                #[inline(always)]
                fn {elem}_ceil(x: {elem}) -> {elem} {{
                    x.ceil()
                }}
                #[inline(always)]
                fn {elem}_round(x: {elem}) -> {elem} {{
                    x.round()
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

    let reduce_add = || -> String {
        let items: Vec<String> = (0..lanes).map(|i| format!("a[{i}]")).collect();
        items.join(" + ")
    };

    // Helper functions (f32_sqrt etc.) are emitted once per element type
    // at the file level by generate_scalar_impls, not per-type.

    let true_mask = if elem == "f32" {
        "f32::from_bits(0xFFFF_FFFF)"
    } else {
        "f64::from_bits(0xFFFF_FFFF_FFFF_FFFF)"
    };

    let sign_shift = if elem == "f32" { "31" } else { "63" };

    formatdoc! {r#"
        impl {trait_name} for archmage::ScalarToken {{
            type Repr = {array};

            // ====== Construction ======

            #[inline(always)]
            fn splat(v: {elem}) -> {array} {{
                [v; {lanes}]
            }}

            #[inline(always)]
            fn zero() -> {array} {{
                [{zero_lit}; {lanes}]
            }}

            #[inline(always)]
            fn load(data: &{array}) -> {array} {{
                *data
            }}

            #[inline(always)]
            fn from_array(arr: {array}) -> {array} {{
                arr
            }}

            #[inline(always)]
            fn store(repr: {array}, out: &mut {array}) {{
                *out = repr;
            }}

            #[inline(always)]
            fn to_array(repr: {array}) -> {array} {{
                repr
            }}

            // ====== Arithmetic ======

            #[inline(always)]
            fn add(a: {array}, b: {array}) -> {array} {{
                {add_lanes}
            }}

            #[inline(always)]
            fn sub(a: {array}, b: {array}) -> {array} {{
                {sub_lanes}
            }}

            #[inline(always)]
            fn mul(a: {array}, b: {array}) -> {array} {{
                {mul_lanes}
            }}

            #[inline(always)]
            fn div(a: {array}, b: {array}) -> {array} {{
                {div_lanes}
            }}

            #[inline(always)]
            fn neg(a: {array}) -> {array} {{
                {neg}
            }}

            // ====== Math ======

            #[inline(always)]
            fn min(a: {array}, b: {array}) -> {array} {{
                {min_lanes}
            }}

            #[inline(always)]
            fn max(a: {array}, b: {array}) -> {array} {{
                {max_lanes}
            }}

            #[inline(always)]
            fn sqrt(a: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = {elem}_sqrt(a[i]);
                }}
                r
            }}

            #[inline(always)]
            fn abs(a: {array}) -> {array} {{
                {abs}
            }}

            #[inline(always)]
            fn floor(a: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = {elem}_floor(a[i]);
                }}
                r
            }}

            #[inline(always)]
            fn ceil(a: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = {elem}_ceil(a[i]);
                }}
                r
            }}

            #[inline(always)]
            fn round(a: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = {elem}_round(a[i]);
                }}
                r
            }}

            #[inline(always)]
            fn mul_add(a: {array}, b: {array}, c: {array}) -> {array} {{
                {mul_add}
            }}

            #[inline(always)]
            fn mul_sub(a: {array}, b: {array}, c: {array}) -> {array} {{
                {mul_sub}
            }}

            // ====== Comparisons ======

            #[inline(always)]
            fn simd_eq(a: {array}, b: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] == b[i] {{ {true_mask} }} else {{ 0.0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_ne(a: {array}, b: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] != b[i] {{ {true_mask} }} else {{ 0.0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_lt(a: {array}, b: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] < b[i] {{ {true_mask} }} else {{ 0.0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_le(a: {array}, b: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] <= b[i] {{ {true_mask} }} else {{ 0.0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_gt(a: {array}, b: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] > b[i] {{ {true_mask} }} else {{ 0.0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_ge(a: {array}, b: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] >= b[i] {{ {true_mask} }} else {{ 0.0 }};
                }}
                r
            }}

            #[inline(always)]
            fn blend(mask: {array}, if_true: {array}, if_false: {array}) -> {array} {{
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
            fn reduce_add(a: {array}) -> {elem} {{
                {reduce_add}
            }}

            #[inline(always)]
            fn reduce_min(a: {array}) -> {elem} {{
                let mut m = a[0];
                for &v in &a[1..] {{
                    m = m.min(v);
                }}
                m
            }}

            #[inline(always)]
            fn reduce_max(a: {array}) -> {elem} {{
                let mut m = a[0];
                for &v in &a[1..] {{
                    m = m.max(v);
                }}
                m
            }}

            // ====== Approximations ======

            #[inline(always)]
            fn rcp_approx(a: {array}) -> {array} {{
                {rcp_lanes}
            }}

            #[inline(always)]
            fn rsqrt_approx(a: {array}) -> {array} {{
                let mut r = [{zero_lit}; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = 1.0 / {elem}_sqrt(a[i]);
                }}
                r
            }}

            // Override defaults: scalar doesn't need Newton-Raphson (already full precision)
            // Use FQS because ScalarToken implements multiple backend traits.
            #[inline(always)]
            fn recip(a: {array}) -> {array} {{
                <archmage::ScalarToken as {trait_name}>::rcp_approx(a)
            }}

            #[inline(always)]
            fn rsqrt(a: {array}) -> {array} {{
                <archmage::ScalarToken as {trait_name}>::rsqrt_approx(a)
            }}

            // ====== Bitwise ======

            #[inline(always)]
            fn not(a: {array}) -> {array} {{
                {not_lanes}
            }}

            #[inline(always)]
            fn bitand(a: {array}, b: {array}) -> {array} {{
                {and_lanes}
            }}

            #[inline(always)]
            fn bitor(a: {array}, b: {array}) -> {array} {{
                {or_lanes}
            }}

            #[inline(always)]
            fn bitxor(a: {array}, b: {array}) -> {array} {{
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
        rcp_lanes = {
            let items: Vec<String> = (0..lanes).map(|i| format!("1.0 / a[{i}]")).collect();
            format!("[{}]", items.join(", "))
        },
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
    let mut code = formatdoc! {r#"
        //! Backend implementations for NeonToken (AArch64 NEON).
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.

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
        let items: Vec<String> = (0..sub_count)
            .map(|i| {
                format!(
                    "vreinterpretq_{ns}_u{eb}(vmvnq_u{eb}(vceqq_{ns}(a[{i}], b[{i}])))",
                    eb = if elem == "f32" { 32 } else { 64 },
                    ns = ns
                )
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
                format!("vreinterpretq_{ns}_u{eb}(vmvnq_u{eb}(vreinterpretq_u{eb}_{ns}(a[{i}])))")
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

    formatdoc! {r#"
        impl {trait_name} for archmage::NeonToken {{
            type Repr = {repr};

            // ====== Construction ======

            #[inline(always)]
            fn splat(v: {elem}) -> {repr} {{
                unsafe {{
                    let v4 = vdupq_n_{ns}(v);
                    [{v4_copies}]
                }}
            }}

            #[inline(always)]
            fn zero() -> {repr} {{
                unsafe {{
                    let z = vdupq_n_{ns}(0.0);
                    [{z_copies}]
                }}
            }}

            #[inline(always)]
            fn load(data: &{array}) -> {repr} {{
                unsafe {{
                    [{load_lanes}]
                }}
            }}

            #[inline(always)]
            fn from_array(arr: {array}) -> {repr} {{
                Self::load(&arr)
            }}

            #[inline(always)]
            fn store(repr: {repr}, out: &mut {array}) {{
                unsafe {{
                    {store_lanes}
                }}
            }}

            #[inline(always)]
            fn to_array(repr: {repr}) -> {array} {{
                let mut out = [{zero_lit}; {lanes}];
                Self::store(repr, &mut out);
                out
            }}

            // ====== Arithmetic ======

            #[inline(always)]
            fn add(a: {repr}, b: {repr}) -> {repr} {{
                {add_body}
            }}

            #[inline(always)]
            fn sub(a: {repr}, b: {repr}) -> {repr} {{
                {sub_body}
            }}

            #[inline(always)]
            fn mul(a: {repr}, b: {repr}) -> {repr} {{
                {mul_body}
            }}

            #[inline(always)]
            fn div(a: {repr}, b: {repr}) -> {repr} {{
                {div_body}
            }}

            #[inline(always)]
            fn neg(a: {repr}) -> {repr} {{
                {neg_body}
            }}

            // ====== Math ======

            #[inline(always)]
            fn min(a: {repr}, b: {repr}) -> {repr} {{
                {min_body}
            }}

            #[inline(always)]
            fn max(a: {repr}, b: {repr}) -> {repr} {{
                {max_body}
            }}

            #[inline(always)]
            fn sqrt(a: {repr}) -> {repr} {{
                {sqrt_body}
            }}

            #[inline(always)]
            fn abs(a: {repr}) -> {repr} {{
                {abs_body}
            }}

            #[inline(always)]
            fn floor(a: {repr}) -> {repr} {{
                {floor_body}
            }}

            #[inline(always)]
            fn ceil(a: {repr}) -> {repr} {{
                {ceil_body}
            }}

            #[inline(always)]
            fn round(a: {repr}) -> {repr} {{
                {round_body}
            }}

            #[inline(always)]
            fn mul_add(a: {repr}, b: {repr}, c: {repr}) -> {repr} {{
                // vfmaq = acc + x*y, so mul_add(a, b, c) = a*b + c => vfmaq(c, a, b)
                {mul_add_body}
            }}

            #[inline(always)]
            fn mul_sub(a: {repr}, b: {repr}, c: {repr}) -> {repr} {{
                // a*b - c => vfmaq(-c, a, b) = -c + a*b
                {mul_sub_body}
            }}

            // ====== Comparisons ======

            #[inline(always)]
            fn simd_eq(a: {repr}, b: {repr}) -> {repr} {{
                {eq_body}
            }}

            #[inline(always)]
            fn simd_ne(a: {repr}, b: {repr}) -> {repr} {{
                {ne_body}
            }}

            #[inline(always)]
            fn simd_lt(a: {repr}, b: {repr}) -> {repr} {{
                {lt_body}
            }}

            #[inline(always)]
            fn simd_le(a: {repr}, b: {repr}) -> {repr} {{
                {le_body}
            }}

            #[inline(always)]
            fn simd_gt(a: {repr}, b: {repr}) -> {repr} {{
                {gt_body}
            }}

            #[inline(always)]
            fn simd_ge(a: {repr}, b: {repr}) -> {repr} {{
                {ge_body}
            }}

            #[inline(always)]
            fn blend(mask: {repr}, if_true: {repr}, if_false: {repr}) -> {repr} {{
                {blend_body}
            }}

            // ====== Reductions ======

            #[inline(always)]
            fn reduce_add(a: {repr}) -> {elem} {{
                {reduce_add_body}
            }}

            #[inline(always)]
            fn reduce_min(a: {repr}) -> {elem} {{
                {reduce_min_body}
            }}

            #[inline(always)]
            fn reduce_max(a: {repr}) -> {elem} {{
                {reduce_max_body}
            }}

            // ====== Approximations ======

            #[inline(always)]
            fn rcp_approx(a: {repr}) -> {repr} {{
                {rcp_body}
            }}

            #[inline(always)]
            fn rsqrt_approx(a: {repr}) -> {repr} {{
                {rsqrt_body}
            }}

            // ====== Bitwise ======

            #[inline(always)]
            fn not(a: {repr}) -> {repr} {{
                {not_body}
            }}

            #[inline(always)]
            fn bitand(a: {repr}, b: {repr}) -> {repr} {{
                {and_body}
            }}

            #[inline(always)]
            fn bitor(a: {repr}, b: {repr}) -> {repr} {{
                {or_body}
            }}

            #[inline(always)]
            fn bitxor(a: {repr}, b: {repr}) -> {repr} {{
                {xor_body}
            }}
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
        rcp_body = unary_op(&format!("vrecpeq_{ns}")),
        rsqrt_body = unary_op(&format!("vrsqrteq_{ns}")),
        not_body = bitwise_not_op(),
        and_body = bitwise_binary_op("vandq_u32"),
        or_body = bitwise_binary_op("vorrq_u32"),
        xor_body = bitwise_binary_op("veorq_u32"),
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

    formatdoc! {r#"
        impl {trait_name} for archmage::NeonToken {{
            type Repr = {repr};

            #[inline(always)]
            fn splat(v: {elem}) -> {repr} {{
                unsafe {{ vdupq_n_{ns}(v) }}
            }}

            #[inline(always)]
            fn zero() -> {repr} {{
                unsafe {{ vdupq_n_{ns}(0.0) }}
            }}

            #[inline(always)]
            fn load(data: &{array}) -> {repr} {{
                unsafe {{ vld1q_{ns}(data.as_ptr()) }}
            }}

            #[inline(always)]
            fn from_array(arr: {array}) -> {repr} {{
                Self::load(&arr)
            }}

            #[inline(always)]
            fn store(repr: {repr}, out: &mut {array}) {{
                unsafe {{ vst1q_{ns}(out.as_mut_ptr(), repr) }};
            }}

            #[inline(always)]
            fn to_array(repr: {repr}) -> {array} {{
                let mut out = [{zero_lit}; {lanes}];
                Self::store(repr, &mut out);
                out
            }}

            #[inline(always)]
            fn add(a: {repr}, b: {repr}) -> {repr} {{ unsafe {{ vaddq_{ns}(a, b) }} }}
            #[inline(always)]
            fn sub(a: {repr}, b: {repr}) -> {repr} {{ unsafe {{ vsubq_{ns}(a, b) }} }}
            #[inline(always)]
            fn mul(a: {repr}, b: {repr}) -> {repr} {{ unsafe {{ vmulq_{ns}(a, b) }} }}
            #[inline(always)]
            fn div(a: {repr}, b: {repr}) -> {repr} {{ unsafe {{ vdivq_{ns}(a, b) }} }}
            #[inline(always)]
            fn neg(a: {repr}) -> {repr} {{ unsafe {{ vnegq_{ns}(a) }} }}
            #[inline(always)]
            fn min(a: {repr}, b: {repr}) -> {repr} {{ unsafe {{ vminq_{ns}(a, b) }} }}
            #[inline(always)]
            fn max(a: {repr}, b: {repr}) -> {repr} {{ unsafe {{ vmaxq_{ns}(a, b) }} }}
            #[inline(always)]
            fn sqrt(a: {repr}) -> {repr} {{ unsafe {{ vsqrtq_{ns}(a) }} }}
            #[inline(always)]
            fn abs(a: {repr}) -> {repr} {{ unsafe {{ vabsq_{ns}(a) }} }}
            #[inline(always)]
            fn floor(a: {repr}) -> {repr} {{ unsafe {{ vrndmq_{ns}(a) }} }}
            #[inline(always)]
            fn ceil(a: {repr}) -> {repr} {{ unsafe {{ vrndpq_{ns}(a) }} }}
            #[inline(always)]
            fn round(a: {repr}) -> {repr} {{ unsafe {{ vrndnq_{ns}(a) }} }}

            #[inline(always)]
            fn mul_add(a: {repr}, b: {repr}, c: {repr}) -> {repr} {{
                unsafe {{ vfmaq_{ns}(c, a, b) }}
            }}

            #[inline(always)]
            fn mul_sub(a: {repr}, b: {repr}, c: {repr}) -> {repr} {{
                unsafe {{ vfmaq_{ns}(vnegq_{ns}(c), a, b) }}
            }}

            #[inline(always)]
            fn simd_eq(a: {repr}, b: {repr}) -> {repr} {{
                unsafe {{ vreinterpretq_{ns}_u{eb}(vceqq_{ns}(a, b)) }}
            }}
            #[inline(always)]
            fn simd_ne(a: {repr}, b: {repr}) -> {repr} {{
                unsafe {{ vreinterpretq_{ns}_u{eb}(vmvnq_u{eb}(vceqq_{ns}(a, b))) }}
            }}
            #[inline(always)]
            fn simd_lt(a: {repr}, b: {repr}) -> {repr} {{
                unsafe {{ vreinterpretq_{ns}_u{eb}(vcltq_{ns}(a, b)) }}
            }}
            #[inline(always)]
            fn simd_le(a: {repr}, b: {repr}) -> {repr} {{
                unsafe {{ vreinterpretq_{ns}_u{eb}(vcleq_{ns}(a, b)) }}
            }}
            #[inline(always)]
            fn simd_gt(a: {repr}, b: {repr}) -> {repr} {{
                unsafe {{ vreinterpretq_{ns}_u{eb}(vcgtq_{ns}(a, b)) }}
            }}
            #[inline(always)]
            fn simd_ge(a: {repr}, b: {repr}) -> {repr} {{
                unsafe {{ vreinterpretq_{ns}_u{eb}(vcgeq_{ns}(a, b)) }}
            }}

            #[inline(always)]
            fn blend(mask: {repr}, if_true: {repr}, if_false: {repr}) -> {repr} {{
                unsafe {{ vbslq_{ns}(vreinterpretq_u{eb}_{ns}(mask), if_true, if_false) }}
            }}

            #[inline(always)]
            fn reduce_add(a: {repr}) -> {elem} {{
                {reduce_add}
            }}

            #[inline(always)]
            fn reduce_min(a: {repr}) -> {elem} {{
                {reduce_min}
            }}

            #[inline(always)]
            fn reduce_max(a: {repr}) -> {elem} {{
                {reduce_max}
            }}

            #[inline(always)]
            fn rcp_approx(a: {repr}) -> {repr} {{ unsafe {{ vrecpeq_{ns}(a) }} }}
            #[inline(always)]
            fn rsqrt_approx(a: {repr}) -> {repr} {{ unsafe {{ vrsqrteq_{ns}(a) }} }}

            #[inline(always)]
            fn not(a: {repr}) -> {repr} {{
                unsafe {{ vreinterpretq_{ns}_u{eb}(vmvnq_u{eb}(vreinterpretq_u{eb}_{ns}(a))) }}
            }}
            #[inline(always)]
            fn bitand(a: {repr}, b: {repr}) -> {repr} {{
                unsafe {{ vreinterpretq_{ns}_u{eb}(vandq_u{eb}(vreinterpretq_u{eb}_{ns}(a), vreinterpretq_u{eb}_{ns}(b))) }}
            }}
            #[inline(always)]
            fn bitor(a: {repr}, b: {repr}) -> {repr} {{
                unsafe {{ vreinterpretq_{ns}_u{eb}(vorrq_u{eb}(vreinterpretq_u{eb}_{ns}(a), vreinterpretq_u{eb}_{ns}(b))) }}
            }}
            #[inline(always)]
            fn bitxor(a: {repr}, b: {repr}) -> {repr} {{
                unsafe {{ vreinterpretq_{ns}_u{eb}(veorq_u{eb}(vreinterpretq_u{eb}_{ns}(a), vreinterpretq_u{eb}_{ns}(b))) }}
            }}
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
    let mut code = formatdoc! {r#"
        //! Backend implementations for Wasm128Token (WebAssembly SIMD).
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.

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

    // Reductions: extract all lanes and fold
    let reduce_add_body = || -> String {
        let mut items = Vec::new();
        for i in 0..sub_count {
            for j in 0..native_lanes {
                items.push(format!("{wp}_extract_lane::<{j}>(a[{i}])"));
            }
        }
        items.join("\n            + ")
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

    formatdoc! {r#"
        impl {trait_name} for archmage::Wasm128Token {{
            type Repr = {repr};

            #[inline(always)]
            fn splat(v: {elem}) -> {repr} {{
                let v4 = {wp}_splat(v);
                [{v4_copies}]
            }}

            #[inline(always)]
            fn zero() -> {repr} {{
                let z = {wp}_splat(0.0);
                [{z_copies}]
            }}

            #[inline(always)]
            fn load(data: &{array}) -> {repr} {{
                unsafe {{
                    [{load_lanes}]
                }}
            }}

            #[inline(always)]
            fn from_array(arr: {array}) -> {repr} {{
                Self::load(&arr)
            }}

            #[inline(always)]
            fn store(repr: {repr}, out: &mut {array}) {{
                unsafe {{
                    {store_lanes}
                }}
            }}

            #[inline(always)]
            fn to_array(repr: {repr}) -> {array} {{
                let mut out = [{zero_lit}; {lanes}];
                Self::store(repr, &mut out);
                out
            }}

            #[inline(always)]
            fn add(a: {repr}, b: {repr}) -> {repr} {{ {add} }}
            #[inline(always)]
            fn sub(a: {repr}, b: {repr}) -> {repr} {{ {sub} }}
            #[inline(always)]
            fn mul(a: {repr}, b: {repr}) -> {repr} {{ {mul} }}
            #[inline(always)]
            fn div(a: {repr}, b: {repr}) -> {repr} {{ {div} }}
            #[inline(always)]
            fn neg(a: {repr}) -> {repr} {{ {neg} }}
            #[inline(always)]
            fn min(a: {repr}, b: {repr}) -> {repr} {{ {min} }}
            #[inline(always)]
            fn max(a: {repr}, b: {repr}) -> {repr} {{ {max} }}
            #[inline(always)]
            fn sqrt(a: {repr}) -> {repr} {{ {sqrt} }}
            #[inline(always)]
            fn abs(a: {repr}) -> {repr} {{ {abs} }}
            #[inline(always)]
            fn floor(a: {repr}) -> {repr} {{ {floor} }}
            #[inline(always)]
            fn ceil(a: {repr}) -> {repr} {{ {ceil} }}
            #[inline(always)]
            fn round(a: {repr}) -> {repr} {{ {round} }}

            #[inline(always)]
            fn mul_add(a: {repr}, b: {repr}, c: {repr}) -> {repr} {{
                // WASM has no native FMA
                [{mul_add_lanes}]
            }}

            #[inline(always)]
            fn mul_sub(a: {repr}, b: {repr}, c: {repr}) -> {repr} {{
                [{mul_sub_lanes}]
            }}

            #[inline(always)]
            fn simd_eq(a: {repr}, b: {repr}) -> {repr} {{ {eq} }}
            #[inline(always)]
            fn simd_ne(a: {repr}, b: {repr}) -> {repr} {{ {ne} }}
            #[inline(always)]
            fn simd_lt(a: {repr}, b: {repr}) -> {repr} {{ {lt} }}
            #[inline(always)]
            fn simd_le(a: {repr}, b: {repr}) -> {repr} {{ {le} }}
            #[inline(always)]
            fn simd_gt(a: {repr}, b: {repr}) -> {repr} {{ {gt} }}
            #[inline(always)]
            fn simd_ge(a: {repr}, b: {repr}) -> {repr} {{ {ge} }}

            #[inline(always)]
            fn blend(mask: {repr}, if_true: {repr}, if_false: {repr}) -> {repr} {{
                [{blend_lanes}]
            }}

            #[inline(always)]
            fn reduce_add(a: {repr}) -> {elem} {{
                {reduce_add}
            }}

            #[inline(always)]
            fn reduce_min(a: {repr}) -> {elem} {{
                {reduce_min}
            }}

            #[inline(always)]
            fn reduce_max(a: {repr}) -> {elem} {{
                {reduce_max}
            }}

            #[inline(always)]
            fn rcp_approx(a: {repr}) -> {repr} {{
                let one = {wp}_splat(1.0);
                [{rcp_lanes}]
            }}

            #[inline(always)]
            fn rsqrt_approx(a: {repr}) -> {repr} {{
                let one = {wp}_splat(1.0);
                [{rsqrt_lanes}]
            }}

            // Override defaults: WASM has no fast approximation, already full precision
            #[inline(always)]
            fn recip(a: {repr}) -> {repr} {{
                <archmage::Wasm128Token as {trait_name}>::rcp_approx(a)
            }}

            #[inline(always)]
            fn rsqrt(a: {repr}) -> {repr} {{
                <archmage::Wasm128Token as {trait_name}>::rsqrt_approx(a)
            }}

            #[inline(always)]
            fn not(a: {repr}) -> {repr} {{ {not} }}
            #[inline(always)]
            fn bitand(a: {repr}, b: {repr}) -> {repr} {{ {and} }}
            #[inline(always)]
            fn bitor(a: {repr}, b: {repr}) -> {repr} {{ {or} }}
            #[inline(always)]
            fn bitxor(a: {repr}, b: {repr}) -> {repr} {{ {xor} }}
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
        rcp_lanes = (0..sub_count)
            .map(|i| format!("{wp}_div(one, a[{i}])"))
            .collect::<Vec<_>>().join(", "),
        rsqrt_lanes = (0..sub_count)
            .map(|i| format!("{wp}_div(one, {wp}_sqrt(a[{i}]))"))
            .collect::<Vec<_>>().join(", "),
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

    // Reduction: extract all lanes
    let reduce_add_body = || -> String {
        let items: Vec<String> = (0..lanes)
            .map(|j| format!("{wp}_extract_lane::<{j}>(a)"))
            .collect();
        items.join(" + ")
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

    formatdoc! {r#"
        impl {trait_name} for archmage::Wasm128Token {{
            type Repr = v128;

            #[inline(always)]
            fn splat(v: {elem}) -> v128 {{ {wp}_splat(v) }}
            #[inline(always)]
            fn zero() -> v128 {{ {wp}_splat(0.0) }}
            #[inline(always)]
            fn load(data: &{array}) -> v128 {{ unsafe {{ v128_load(data.as_ptr().cast()) }} }}
            #[inline(always)]
            fn from_array(arr: {array}) -> v128 {{ Self::load(&arr) }}
            #[inline(always)]
            fn store(repr: v128, out: &mut {array}) {{ unsafe {{ v128_store(out.as_mut_ptr().cast(), repr) }}; }}
            #[inline(always)]
            fn to_array(repr: v128) -> {array} {{
                let mut out = [{zero_lit}; {lanes}];
                Self::store(repr, &mut out);
                out
            }}

            #[inline(always)]
            fn add(a: v128, b: v128) -> v128 {{ {wp}_add(a, b) }}
            #[inline(always)]
            fn sub(a: v128, b: v128) -> v128 {{ {wp}_sub(a, b) }}
            #[inline(always)]
            fn mul(a: v128, b: v128) -> v128 {{ {wp}_mul(a, b) }}
            #[inline(always)]
            fn div(a: v128, b: v128) -> v128 {{ {wp}_div(a, b) }}
            #[inline(always)]
            fn neg(a: v128) -> v128 {{ {wp}_neg(a) }}
            #[inline(always)]
            fn min(a: v128, b: v128) -> v128 {{ {wp}_min(a, b) }}
            #[inline(always)]
            fn max(a: v128, b: v128) -> v128 {{ {wp}_max(a, b) }}
            #[inline(always)]
            fn sqrt(a: v128) -> v128 {{ {wp}_sqrt(a) }}
            #[inline(always)]
            fn abs(a: v128) -> v128 {{ {wp}_abs(a) }}
            #[inline(always)]
            fn floor(a: v128) -> v128 {{ {wp}_floor(a) }}
            #[inline(always)]
            fn ceil(a: v128) -> v128 {{ {wp}_ceil(a) }}
            #[inline(always)]
            fn round(a: v128) -> v128 {{ {wp}_nearest(a) }}
            #[inline(always)]
            fn mul_add(a: v128, b: v128, c: v128) -> v128 {{ {wp}_add({wp}_mul(a, b), c) }}
            #[inline(always)]
            fn mul_sub(a: v128, b: v128, c: v128) -> v128 {{ {wp}_sub({wp}_mul(a, b), c) }}
            #[inline(always)]
            fn simd_eq(a: v128, b: v128) -> v128 {{ {wp}_eq(a, b) }}
            #[inline(always)]
            fn simd_ne(a: v128, b: v128) -> v128 {{ {wp}_ne(a, b) }}
            #[inline(always)]
            fn simd_lt(a: v128, b: v128) -> v128 {{ {wp}_lt(a, b) }}
            #[inline(always)]
            fn simd_le(a: v128, b: v128) -> v128 {{ {wp}_le(a, b) }}
            #[inline(always)]
            fn simd_gt(a: v128, b: v128) -> v128 {{ {wp}_gt(a, b) }}
            #[inline(always)]
            fn simd_ge(a: v128, b: v128) -> v128 {{ {wp}_ge(a, b) }}
            #[inline(always)]
            fn blend(mask: v128, if_true: v128, if_false: v128) -> v128 {{
                v128_bitselect(if_true, if_false, mask)
            }}

            #[inline(always)]
            fn reduce_add(a: v128) -> {elem} {{ {reduce_add} }}
            #[inline(always)]
            fn reduce_min(a: v128) -> {elem} {{
                {reduce_min}
            }}
            #[inline(always)]
            fn reduce_max(a: v128) -> {elem} {{
                {reduce_max}
            }}

            #[inline(always)]
            fn rcp_approx(a: v128) -> v128 {{ {wp}_div({wp}_splat(1.0), a) }}
            #[inline(always)]
            fn rsqrt_approx(a: v128) -> v128 {{ {wp}_div({wp}_splat(1.0), {wp}_sqrt(a)) }}
            #[inline(always)]
            fn recip(a: v128) -> v128 {{ <archmage::Wasm128Token as {trait_name}>::rcp_approx(a) }}
            #[inline(always)]
            fn rsqrt(a: v128) -> v128 {{ <archmage::Wasm128Token as {trait_name}>::rsqrt_approx(a) }}

            #[inline(always)]
            fn not(a: v128) -> v128 {{ v128_not(a) }}
            #[inline(always)]
            fn bitand(a: v128, b: v128) -> v128 {{ v128_and(a, b) }}
            #[inline(always)]
            fn bitor(a: v128, b: v128) -> v128 {{ v128_or(a, b) }}
            #[inline(always)]
            fn bitxor(a: v128, b: v128) -> v128 {{ v128_xor(a, b) }}
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
        /// Trait methods are **associated functions** (no `self`/token parameter).
        /// The implementing type `Self` (a token type) determines which platform
        /// intrinsics are used. All methods are `#[inline(always)]` in implementations.
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
            fn splat(v: i32) -> Self::Repr;

            /// All lanes zero.
            fn zero() -> Self::Repr;

            /// Load from an aligned array.
            fn load(data: &{array}) -> Self::Repr;

            /// Create from array (zero-cost transmute where possible).
            fn from_array(arr: {array}) -> Self::Repr;

            /// Store to array.
            fn store(repr: Self::Repr, out: &mut {array});

            /// Convert to array.
            fn to_array(repr: Self::Repr) -> {array};

            // ====== Arithmetic ======

            /// Lane-wise addition.
            fn add(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise subtraction.
            fn sub(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise multiplication (low 32 bits of each 32x32 product).
            fn mul(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise negation.
            fn neg(a: Self::Repr) -> Self::Repr;

            // ====== Math ======

            /// Lane-wise minimum.
            fn min(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise maximum.
            fn max(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise absolute value.
            fn abs(a: Self::Repr) -> Self::Repr;

            // ====== Comparisons ======
            // Return masks where each lane is all-1s (true) or all-0s (false).

            /// Lane-wise equality.
            fn simd_eq(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise inequality.
            fn simd_ne(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise less-than.
            fn simd_lt(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise less-than-or-equal.
            fn simd_le(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise greater-than.
            fn simd_gt(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise greater-than-or-equal.
            fn simd_ge(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Select lanes: where mask is all-1s pick `if_true`, else `if_false`.
            fn blend(mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;

            // ====== Reductions ======

            /// Sum all {lanes} lanes.
            fn reduce_add(a: Self::Repr) -> i32;

            // ====== Bitwise ======

            /// Bitwise NOT.
            fn not(a: Self::Repr) -> Self::Repr;

            /// Bitwise AND.
            fn bitand(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Bitwise OR.
            fn bitor(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Bitwise XOR.
            fn bitxor(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            // ====== Shifts ======

            /// Shift left by constant.
            fn shl_const<const N: i32>(a: Self::Repr) -> Self::Repr;

            /// Arithmetic shift right by constant (sign-extending).
            fn shr_arithmetic_const<const N: i32>(a: Self::Repr) -> Self::Repr;

            /// Logical shift right by constant (zero-filling).
            fn shr_logical_const<const N: i32>(a: Self::Repr) -> Self::Repr;

            // ====== Boolean ======

            /// True if all lanes have their sign bit set (all-1s mask).
            fn all_true(a: Self::Repr) -> bool;

            /// True if any lane has its sign bit set (any all-1s mask lane).
            fn any_true(a: Self::Repr) -> bool;

            /// Extract the high bit of each 32-bit lane as a bitmask.
            fn bitmask(a: Self::Repr) -> u32;

            // ====== Default implementations ======

            /// Clamp values between lo and hi.
            #[inline(always)]
            fn clamp(a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {{
                Self::min(Self::max(a, lo), hi)
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
        use super::F64x2Backend;
        use super::F64x4Backend;
        use super::I32x4Backend;
        use super::I32x8Backend;
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
            fn bitcast_f32_to_i32(a: <Self as F32x4Backend>::Repr) -> <Self as I32x4Backend>::Repr;

            /// Bitcast i32x4 to f32x4 (reinterpret bits, no conversion).
            fn bitcast_i32_to_f32(a: <Self as I32x4Backend>::Repr) -> <Self as F32x4Backend>::Repr;

            /// Convert f32x4 to i32x4 with truncation toward zero.
            fn convert_f32_to_i32(a: <Self as F32x4Backend>::Repr) -> <Self as I32x4Backend>::Repr;

            /// Convert f32x4 to i32x4 with rounding to nearest.
            fn convert_f32_to_i32_round(a: <Self as F32x4Backend>::Repr) -> <Self as I32x4Backend>::Repr;

            /// Convert i32x4 to f32x4.
            fn convert_i32_to_f32(a: <Self as I32x4Backend>::Repr) -> <Self as F32x4Backend>::Repr;
        }}

        /// Conversions between f32x8 and i32x8 representations.
        ///
        /// Requires both `F32x8Backend` and `I32x8Backend` to be implemented.
        pub trait F32x8Convert: F32x8Backend + I32x8Backend + SimdToken + Sealed + Copy + 'static {{
            /// Bitcast f32x8 to i32x8 (reinterpret bits, no conversion).
            fn bitcast_f32_to_i32(a: <Self as F32x8Backend>::Repr) -> <Self as I32x8Backend>::Repr;

            /// Bitcast i32x8 to f32x8 (reinterpret bits, no conversion).
            fn bitcast_i32_to_f32(a: <Self as I32x8Backend>::Repr) -> <Self as F32x8Backend>::Repr;

            /// Convert f32x8 to i32x8 with truncation toward zero.
            fn convert_f32_to_i32(a: <Self as F32x8Backend>::Repr) -> <Self as I32x8Backend>::Repr;

            /// Convert f32x8 to i32x8 with rounding to nearest.
            fn convert_f32_to_i32_round(a: <Self as F32x8Backend>::Repr) -> <Self as I32x8Backend>::Repr;

            /// Convert i32x8 to f32x8.
            fn convert_i32_to_f32(a: <Self as I32x8Backend>::Repr) -> <Self as F32x8Backend>::Repr;
        }}

        /// Bitcast conversions between u32x4 and i32x4 representations.
        ///
        /// Requires both `U32x4Backend` and `I32x4Backend` to be implemented.
        pub trait U32x4Bitcast: U32x4Backend + I32x4Backend + SimdToken + Sealed + Copy + 'static {{
            /// Bitcast u32x4 to i32x4 (reinterpret bits, no conversion).
            fn bitcast_u32_to_i32(a: <Self as U32x4Backend>::Repr) -> <Self as I32x4Backend>::Repr;

            /// Bitcast i32x4 to u32x4 (reinterpret bits, no conversion).
            fn bitcast_i32_to_u32(a: <Self as I32x4Backend>::Repr) -> <Self as U32x4Backend>::Repr;
        }}

        /// Bitcast conversions between u32x8 and i32x8 representations.
        ///
        /// Requires both `U32x8Backend` and `I32x8Backend` to be implemented.
        pub trait U32x8Bitcast: U32x8Backend + I32x8Backend + SimdToken + Sealed + Copy + 'static {{
            /// Bitcast u32x8 to i32x8 (reinterpret bits, no conversion).
            fn bitcast_u32_to_i32(a: <Self as U32x8Backend>::Repr) -> <Self as I32x8Backend>::Repr;

            /// Bitcast i32x8 to u32x8 (reinterpret bits, no conversion).
            fn bitcast_i32_to_u32(a: <Self as I32x8Backend>::Repr) -> <Self as U32x8Backend>::Repr;
        }}

        /// Bitcast conversions between i64x2 and f64x2 representations.
        ///
        /// Requires both `I64x2Backend` and `F64x2Backend` to be implemented.
        pub trait I64x2Bitcast: I64x2Backend + F64x2Backend + SimdToken + Sealed + Copy + 'static {{
            /// Bitcast i64x2 to f64x2 (reinterpret bits, no conversion).
            fn bitcast_i64_to_f64(a: <Self as I64x2Backend>::Repr) -> <Self as F64x2Backend>::Repr;

            /// Bitcast f64x2 to i64x2 (reinterpret bits, no conversion).
            fn bitcast_f64_to_i64(a: <Self as F64x2Backend>::Repr) -> <Self as I64x2Backend>::Repr;
        }}

        /// Bitcast conversions between i64x4 and f64x4 representations.
        ///
        /// Requires both `I64x4Backend` and `F64x4Backend` to be implemented.
        pub trait I64x4Bitcast: I64x4Backend + F64x4Backend + SimdToken + Sealed + Copy + 'static {{
            /// Bitcast i64x4 to f64x4 (reinterpret bits, no conversion).
            fn bitcast_i64_to_f64(a: <Self as I64x4Backend>::Repr) -> <Self as F64x4Backend>::Repr;

            /// Bitcast f64x4 to i64x4 (reinterpret bits, no conversion).
            fn bitcast_f64_to_i64(a: <Self as F64x4Backend>::Repr) -> <Self as I64x4Backend>::Repr;
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
            fn splat(v: i32) -> {inner} {{
                unsafe {{ {p}_set1_epi32(v) }}
            }}

            #[inline(always)]
            fn zero() -> {inner} {{
                unsafe {{ {p}_setzero_si{bits}() }}
            }}

            #[inline(always)]
            fn load(data: &{array}) -> {inner} {{
                unsafe {{ {p}_loadu_si{bits}(data.as_ptr().cast()) }}
            }}

            #[inline(always)]
            fn from_array(arr: {array}) -> {inner} {{
                // SAFETY: {array} and {inner} have identical size and layout.
                unsafe {{ core::mem::transmute(arr) }}
            }}

            #[inline(always)]
            fn store(repr: {inner}, out: &mut {array}) {{
                unsafe {{ {p}_storeu_si{bits}(out.as_mut_ptr().cast(), repr) }};
            }}

            #[inline(always)]
            fn to_array(repr: {inner}) -> {array} {{
                let mut out = [0i32; {lanes}];
                unsafe {{ {p}_storeu_si{bits}(out.as_mut_ptr().cast(), repr) }};
                out
            }}

            // ====== Arithmetic ======

            #[inline(always)]
            fn add(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_add_epi32(a, b) }}
            }}

            #[inline(always)]
            fn sub(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_sub_epi32(a, b) }}
            }}

            #[inline(always)]
            fn mul(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_mullo_epi32(a, b) }}
            }}

            #[inline(always)]
            fn neg(a: {inner}) -> {inner} {{
                unsafe {{ {p}_sub_epi32({p}_setzero_si{bits}(), a) }}
            }}

            // ====== Math ======

            #[inline(always)]
            fn min(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_min_epi32(a, b) }}
            }}

            #[inline(always)]
            fn max(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_max_epi32(a, b) }}
            }}

            #[inline(always)]
            fn abs(a: {inner}) -> {inner} {{
                unsafe {{ {p}_abs_epi32(a) }}
            }}

            // ====== Comparisons ======

            #[inline(always)]
            fn simd_eq(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_cmpeq_epi32(a, b) }}
            }}

            #[inline(always)]
            fn simd_ne(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{
                    let eq = {p}_cmpeq_epi32(a, b);
                    {p}_andnot_si{bits}(eq, {p}_set1_epi32(-1))
                }}
            }}

            #[inline(always)]
            fn simd_lt(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_cmpgt_epi32(b, a) }}
            }}

            #[inline(always)]
            fn simd_le(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{
                    let gt = {p}_cmpgt_epi32(a, b);
                    {p}_andnot_si{bits}(gt, {p}_set1_epi32(-1))
                }}
            }}

            #[inline(always)]
            fn simd_gt(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_cmpgt_epi32(a, b) }}
            }}

            #[inline(always)]
            fn simd_ge(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{
                    let lt = {p}_cmpgt_epi32(b, a);
                    {p}_andnot_si{bits}(lt, {p}_set1_epi32(-1))
                }}
            }}

            #[inline(always)]
            fn blend(mask: {inner}, if_true: {inner}, if_false: {inner}) -> {inner} {{
                unsafe {{ {p}_blendv_epi8(if_false, if_true, mask) }}
            }}

            // ====== Reductions ======

            #[inline(always)]
            fn reduce_add(a: {inner}) -> i32 {{
        {reduce_add_body}
            }}

            // ====== Bitwise ======

            #[inline(always)]
            fn not(a: {inner}) -> {inner} {{
                unsafe {{ {p}_andnot_si{bits}(a, {p}_set1_epi32(-1)) }}
            }}

            #[inline(always)]
            fn bitand(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_and_si{bits}(a, b) }}
            }}

            #[inline(always)]
            fn bitor(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_or_si{bits}(a, b) }}
            }}

            #[inline(always)]
            fn bitxor(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_xor_si{bits}(a, b) }}
            }}

            // ====== Shifts ======

            #[inline(always)]
            fn shl_const<const N: i32>(a: {inner}) -> {inner} {{
                unsafe {{ {p}_slli_epi32::<N>(a) }}
            }}

            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(a: {inner}) -> {inner} {{
                unsafe {{ {p}_srai_epi32::<N>(a) }}
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(a: {inner}) -> {inner} {{
                unsafe {{ {p}_srli_epi32::<N>(a) }}
            }}

            // ====== Boolean ======

            #[inline(always)]
            fn all_true(a: {inner}) -> bool {{
                unsafe {{ {p}_movemask_ps({p}_castsi{bits}_ps(a)) == {all_mask} }}
            }}

            #[inline(always)]
            fn any_true(a: {inner}) -> bool {{
                unsafe {{ {p}_movemask_ps({p}_castsi{bits}_ps(a)) != 0 }}
            }}

            #[inline(always)]
            fn bitmask(a: {inner}) -> u32 {{
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
            fn bitcast_f32_to_i32(a: __m128) -> __m128i {{
                unsafe {{ _mm_castps_si128(a) }}
            }}

            #[inline(always)]
            fn bitcast_i32_to_f32(a: __m128i) -> __m128 {{
                unsafe {{ _mm_castsi128_ps(a) }}
            }}

            #[inline(always)]
            fn convert_f32_to_i32(a: __m128) -> __m128i {{
                unsafe {{ _mm_cvttps_epi32(a) }}
            }}

            #[inline(always)]
            fn convert_f32_to_i32_round(a: __m128) -> __m128i {{
                unsafe {{ _mm_cvtps_epi32(a) }}
            }}

            #[inline(always)]
            fn convert_i32_to_f32(a: __m128i) -> __m128 {{
                unsafe {{ _mm_cvtepi32_ps(a) }}
            }}
        }}

        #[cfg(target_arch = "x86_64")]
        impl F32x8Convert for archmage::{token} {{
            #[inline(always)]
            fn bitcast_f32_to_i32(a: __m256) -> __m256i {{
                unsafe {{ _mm256_castps_si256(a) }}
            }}

            #[inline(always)]
            fn bitcast_i32_to_f32(a: __m256i) -> __m256 {{
                unsafe {{ _mm256_castsi256_ps(a) }}
            }}

            #[inline(always)]
            fn convert_f32_to_i32(a: __m256) -> __m256i {{
                unsafe {{ _mm256_cvttps_epi32(a) }}
            }}

            #[inline(always)]
            fn convert_f32_to_i32_round(a: __m256) -> __m256i {{
                unsafe {{ _mm256_cvtps_epi32(a) }}
            }}

            #[inline(always)]
            fn convert_i32_to_f32(a: __m256i) -> __m256 {{
                unsafe {{ _mm256_cvtepi32_ps(a) }}
            }}
        }}

        #[cfg(target_arch = "x86_64")]
        impl U32x4Bitcast for archmage::{token} {{
            #[inline(always)]
            fn bitcast_u32_to_i32(a: __m128i) -> __m128i {{
                a
            }}

            #[inline(always)]
            fn bitcast_i32_to_u32(a: __m128i) -> __m128i {{
                a
            }}
        }}

        #[cfg(target_arch = "x86_64")]
        impl U32x8Bitcast for archmage::{token} {{
            #[inline(always)]
            fn bitcast_u32_to_i32(a: __m256i) -> __m256i {{
                a
            }}

            #[inline(always)]
            fn bitcast_i32_to_u32(a: __m256i) -> __m256i {{
                a
            }}
        }}

        #[cfg(target_arch = "x86_64")]
        impl I64x2Bitcast for archmage::{token} {{
            #[inline(always)]
            fn bitcast_i64_to_f64(a: __m128i) -> __m128d {{
                unsafe {{ _mm_castsi128_pd(a) }}
            }}

            #[inline(always)]
            fn bitcast_f64_to_i64(a: __m128d) -> __m128i {{
                unsafe {{ _mm_castpd_si128(a) }}
            }}
        }}

        #[cfg(target_arch = "x86_64")]
        impl I64x4Bitcast for archmage::{token} {{
            #[inline(always)]
            fn bitcast_i64_to_f64(a: __m256i) -> __m256d {{
                unsafe {{ _mm256_castsi256_pd(a) }}
            }}

            #[inline(always)]
            fn bitcast_f64_to_i64(a: __m256d) -> __m256i {{
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
            fn splat(v: i32) -> {array} {{
                [v; {lanes}]
            }}

            #[inline(always)]
            fn zero() -> {array} {{
                [0i32; {lanes}]
            }}

            #[inline(always)]
            fn load(data: &{array}) -> {array} {{
                *data
            }}

            #[inline(always)]
            fn from_array(arr: {array}) -> {array} {{
                arr
            }}

            #[inline(always)]
            fn store(repr: {array}, out: &mut {array}) {{
                *out = repr;
            }}

            #[inline(always)]
            fn to_array(repr: {array}) -> {array} {{
                repr
            }}

            // ====== Arithmetic ======

            #[inline(always)]
            fn add(a: {array}, b: {array}) -> {array} {{
                {add_lanes}
            }}

            #[inline(always)]
            fn sub(a: {array}, b: {array}) -> {array} {{
                {sub_lanes}
            }}

            #[inline(always)]
            fn mul(a: {array}, b: {array}) -> {array} {{
                {mul_lanes}
            }}

            #[inline(always)]
            fn neg(a: {array}) -> {array} {{
                {neg}
            }}

            // ====== Math ======

            #[inline(always)]
            fn min(a: {array}, b: {array}) -> {array} {{
                {min_lanes}
            }}

            #[inline(always)]
            fn max(a: {array}, b: {array}) -> {array} {{
                {max_lanes}
            }}

            #[inline(always)]
            fn abs(a: {array}) -> {array} {{
                {abs}
            }}

            // ====== Comparisons ======

            #[inline(always)]
            fn simd_eq(a: {array}, b: {array}) -> {array} {{
                let mut r = [0i32; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] == b[i] {{ -1 }} else {{ 0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_ne(a: {array}, b: {array}) -> {array} {{
                let mut r = [0i32; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] != b[i] {{ -1 }} else {{ 0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_lt(a: {array}, b: {array}) -> {array} {{
                let mut r = [0i32; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] < b[i] {{ -1 }} else {{ 0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_le(a: {array}, b: {array}) -> {array} {{
                let mut r = [0i32; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] <= b[i] {{ -1 }} else {{ 0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_gt(a: {array}, b: {array}) -> {array} {{
                let mut r = [0i32; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] > b[i] {{ -1 }} else {{ 0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_ge(a: {array}, b: {array}) -> {array} {{
                let mut r = [0i32; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] >= b[i] {{ -1 }} else {{ 0 }};
                }}
                r
            }}

            #[inline(always)]
            fn blend(mask: {array}, if_true: {array}, if_false: {array}) -> {array} {{
                let mut r = [0i32; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if mask[i] != 0 {{ if_true[i] }} else {{ if_false[i] }};
                }}
                r
            }}

            // ====== Reductions ======

            #[inline(always)]
            fn reduce_add(a: {array}) -> i32 {{
                {reduce_add}
            }}

            // ====== Bitwise ======

            #[inline(always)]
            fn not(a: {array}) -> {array} {{
                {not_lanes}
            }}

            #[inline(always)]
            fn bitand(a: {array}, b: {array}) -> {array} {{
                {and_lanes}
            }}

            #[inline(always)]
            fn bitor(a: {array}, b: {array}) -> {array} {{
                {or_lanes}
            }}

            #[inline(always)]
            fn bitxor(a: {array}, b: {array}) -> {array} {{
                {xor_lanes}
            }}

            // ====== Shifts ======

            #[inline(always)]
            fn shl_const<const N: i32>(a: {array}) -> {array} {{
                {shl}
            }}

            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(a: {array}) -> {array} {{
                {shr_arithmetic}
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(a: {array}) -> {array} {{
                {shr_logical}
            }}

            // ====== Boolean ======

            #[inline(always)]
            fn all_true(a: {array}) -> bool {{
                {all_true}
            }}

            #[inline(always)]
            fn any_true(a: {array}) -> bool {{
                {any_true}
            }}

            #[inline(always)]
            fn bitmask(a: {array}) -> u32 {{
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

    for lanes in [4, 8] {
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

        code.push_str(&formatdoc! {r#"
            impl {trait_name} for archmage::ScalarToken {{
                #[inline(always)]
                fn bitcast_f32_to_i32(a: {f_array}) -> {i_array} {{
                    [{bitcast_f2i}]
                }}

                #[inline(always)]
                fn bitcast_i32_to_f32(a: {i_array}) -> {f_array} {{
                    [{bitcast_i2f}]
                }}

                #[inline(always)]
                fn convert_f32_to_i32(a: {f_array}) -> {i_array} {{
                    [{cvt_f2i}]
                }}

                #[inline(always)]
                fn convert_f32_to_i32_round(a: {f_array}) -> {i_array} {{
                    [{cvt_f2i_round}]
                }}

                #[inline(always)]
                fn convert_i32_to_f32(a: {i_array}) -> {f_array} {{
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
                fn bitcast_u32_to_i32(a: {u_array}) -> {i_array} {{
                    [{u2i}]
                }}

                #[inline(always)]
                fn bitcast_i32_to_u32(a: {i_array}) -> {u_array} {{
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
                fn bitcast_i64_to_f64(a: {i_array}) -> {f_array} {{
                    [{i2f}]
                }}

                #[inline(always)]
                fn bitcast_f64_to_i64(a: {f_array}) -> {i_array} {{
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
            fn splat(v: i32) -> int32x4_t {{
                unsafe {{ vdupq_n_s32(v) }}
            }}

            #[inline(always)]
            fn zero() -> int32x4_t {{
                unsafe {{ vdupq_n_s32(0) }}
            }}

            #[inline(always)]
            fn load(data: &{array}) -> int32x4_t {{
                unsafe {{ vld1q_s32(data.as_ptr()) }}
            }}

            #[inline(always)]
            fn from_array(arr: {array}) -> int32x4_t {{
                unsafe {{ vld1q_s32(arr.as_ptr()) }}
            }}

            #[inline(always)]
            fn store(repr: int32x4_t, out: &mut {array}) {{
                unsafe {{ vst1q_s32(out.as_mut_ptr(), repr) }};
            }}

            #[inline(always)]
            fn to_array(repr: int32x4_t) -> {array} {{
                let mut out = [0i32; {lanes}];
                unsafe {{ vst1q_s32(out.as_mut_ptr(), repr) }};
                out
            }}

            #[inline(always)]
            fn add(a: int32x4_t, b: int32x4_t) -> int32x4_t {{ unsafe {{ vaddq_s32(a, b) }} }}
            #[inline(always)]
            fn sub(a: int32x4_t, b: int32x4_t) -> int32x4_t {{ unsafe {{ vsubq_s32(a, b) }} }}
            #[inline(always)]
            fn mul(a: int32x4_t, b: int32x4_t) -> int32x4_t {{ unsafe {{ vmulq_s32(a, b) }} }}
            #[inline(always)]
            fn neg(a: int32x4_t) -> int32x4_t {{ unsafe {{ vnegq_s32(a) }} }}
            #[inline(always)]
            fn min(a: int32x4_t, b: int32x4_t) -> int32x4_t {{ unsafe {{ vminq_s32(a, b) }} }}
            #[inline(always)]
            fn max(a: int32x4_t, b: int32x4_t) -> int32x4_t {{ unsafe {{ vmaxq_s32(a, b) }} }}
            #[inline(always)]
            fn abs(a: int32x4_t) -> int32x4_t {{ unsafe {{ vabsq_s32(a) }} }}

            #[inline(always)]
            fn simd_eq(a: int32x4_t, b: int32x4_t) -> int32x4_t {{
                unsafe {{ vreinterpretq_s32_u32(vceqq_s32(a, b)) }}
            }}
            #[inline(always)]
            fn simd_ne(a: int32x4_t, b: int32x4_t) -> int32x4_t {{
                unsafe {{ vreinterpretq_s32_u32(vmvnq_u32(vceqq_s32(a, b))) }}
            }}
            #[inline(always)]
            fn simd_lt(a: int32x4_t, b: int32x4_t) -> int32x4_t {{
                unsafe {{ vreinterpretq_s32_u32(vcltq_s32(a, b)) }}
            }}
            #[inline(always)]
            fn simd_le(a: int32x4_t, b: int32x4_t) -> int32x4_t {{
                unsafe {{ vreinterpretq_s32_u32(vcleq_s32(a, b)) }}
            }}
            #[inline(always)]
            fn simd_gt(a: int32x4_t, b: int32x4_t) -> int32x4_t {{
                unsafe {{ vreinterpretq_s32_u32(vcgtq_s32(a, b)) }}
            }}
            #[inline(always)]
            fn simd_ge(a: int32x4_t, b: int32x4_t) -> int32x4_t {{
                unsafe {{ vreinterpretq_s32_u32(vcgeq_s32(a, b)) }}
            }}

            #[inline(always)]
            fn blend(mask: int32x4_t, if_true: int32x4_t, if_false: int32x4_t) -> int32x4_t {{
                unsafe {{ vbslq_s32(vreinterpretq_u32_s32(mask), if_true, if_false) }}
            }}

            #[inline(always)]
            fn reduce_add(a: int32x4_t) -> i32 {{
                unsafe {{ vaddvq_s32(a) }}
            }}

            #[inline(always)]
            fn not(a: int32x4_t) -> int32x4_t {{
                unsafe {{ vmvnq_s32(a) }}
            }}
            #[inline(always)]
            fn bitand(a: int32x4_t, b: int32x4_t) -> int32x4_t {{
                unsafe {{ vandq_s32(a, b) }}
            }}
            #[inline(always)]
            fn bitor(a: int32x4_t, b: int32x4_t) -> int32x4_t {{
                unsafe {{ vorrq_s32(a, b) }}
            }}
            #[inline(always)]
            fn bitxor(a: int32x4_t, b: int32x4_t) -> int32x4_t {{
                unsafe {{ veorq_s32(a, b) }}
            }}

            #[inline(always)]
            fn shl_const<const N: i32>(a: int32x4_t) -> int32x4_t {{
                unsafe {{ vshlq_n_s32::<N>(a) }}
            }}

            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(a: int32x4_t) -> int32x4_t {{
                unsafe {{ vshrq_n_s32::<N>(a) }}
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(a: int32x4_t) -> int32x4_t {{
                unsafe {{ vreinterpretq_s32_u32(vshrq_n_u32::<N>(vreinterpretq_u32_s32(a))) }}
            }}

            #[inline(always)]
            fn all_true(a: int32x4_t) -> bool {{
                unsafe {{ vminvq_u32(vreinterpretq_u32_s32(a)) != 0 }}
            }}

            #[inline(always)]
            fn any_true(a: int32x4_t) -> bool {{
                unsafe {{ vmaxvq_u32(vreinterpretq_u32_s32(a)) != 0 }}
            }}

            #[inline(always)]
            fn bitmask(a: int32x4_t) -> u32 {{
                unsafe {{
                    // Extract sign bit of each 32-bit lane
                    let shift = vreinterpretq_u32_s32(vshrq_n_s32::<31>(a));
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
            fn splat(v: i32) -> {repr} {{
                unsafe {{
                    let v4 = vdupq_n_s32(v);
                    [{v4_copies}]
                }}
            }}

            #[inline(always)]
            fn zero() -> {repr} {{
                unsafe {{
                    let z = vdupq_n_s32(0);
                    [{z_copies}]
                }}
            }}

            #[inline(always)]
            fn load(data: &{array}) -> {repr} {{
                unsafe {{
                    [{load_lanes}]
                }}
            }}

            #[inline(always)]
            fn from_array(arr: {array}) -> {repr} {{
                Self::load(&arr)
            }}

            #[inline(always)]
            fn store(repr: {repr}, out: &mut {array}) {{
                unsafe {{
                    {store_lanes}
                }}
            }}

            #[inline(always)]
            fn to_array(repr: {repr}) -> {array} {{
                let mut out = [0i32; {lanes}];
                Self::store(repr, &mut out);
                out
            }}

            #[inline(always)]
            fn add(a: {repr}, b: {repr}) -> {repr} {{ {add} }}
            #[inline(always)]
            fn sub(a: {repr}, b: {repr}) -> {repr} {{ {sub} }}
            #[inline(always)]
            fn mul(a: {repr}, b: {repr}) -> {repr} {{ {mul} }}
            #[inline(always)]
            fn neg(a: {repr}) -> {repr} {{ {neg} }}
            #[inline(always)]
            fn min(a: {repr}, b: {repr}) -> {repr} {{ {min} }}
            #[inline(always)]
            fn max(a: {repr}, b: {repr}) -> {repr} {{ {max} }}
            #[inline(always)]
            fn abs(a: {repr}) -> {repr} {{ {abs} }}

            #[inline(always)]
            fn simd_eq(a: {repr}, b: {repr}) -> {repr} {{ {eq} }}
            #[inline(always)]
            fn simd_ne(a: {repr}, b: {repr}) -> {repr} {{ {ne} }}
            #[inline(always)]
            fn simd_lt(a: {repr}, b: {repr}) -> {repr} {{ {lt} }}
            #[inline(always)]
            fn simd_le(a: {repr}, b: {repr}) -> {repr} {{ {le} }}
            #[inline(always)]
            fn simd_gt(a: {repr}, b: {repr}) -> {repr} {{ {gt} }}
            #[inline(always)]
            fn simd_ge(a: {repr}, b: {repr}) -> {repr} {{ {ge} }}

            #[inline(always)]
            fn blend(mask: {repr}, if_true: {repr}, if_false: {repr}) -> {repr} {{
                {blend}
            }}

            #[inline(always)]
            fn reduce_add(a: {repr}) -> i32 {{
                {reduce_add}
            }}

            #[inline(always)]
            fn not(a: {repr}) -> {repr} {{ {not} }}
            #[inline(always)]
            fn bitand(a: {repr}, b: {repr}) -> {repr} {{ {bitand} }}
            #[inline(always)]
            fn bitor(a: {repr}, b: {repr}) -> {repr} {{ {bitor} }}
            #[inline(always)]
            fn bitxor(a: {repr}, b: {repr}) -> {repr} {{ {bitxor} }}

            #[inline(always)]
            fn shl_const<const N: i32>(a: {repr}) -> {repr} {{
                {shl}
            }}

            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(a: {repr}) -> {repr} {{
                {shr_arith}
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(a: {repr}) -> {repr} {{
                {shr_logic}
            }}

            #[inline(always)]
            fn all_true(a: {repr}) -> bool {{
                {all_true}
            }}

            #[inline(always)]
            fn any_true(a: {repr}) -> bool {{
                {any_true}
            }}

            #[inline(always)]
            fn bitmask(a: {repr}) -> u32 {{
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
                .map(|i| format!("vshrq_n_s32::<N>(a[{i}])"))
                .collect();
            format!("unsafe {{ [{}] }}", items.join(", "))
        },
        shr_logic = {
            let items: Vec<String> = (0..sub_count)
                .map(|i| format!("vreinterpretq_s32_u32(vshrq_n_u32::<N>(vreinterpretq_u32_s32(a[{i}])))"))
                .collect();
            format!("unsafe {{ [{}] }}", items.join(", "))
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
                body.push_str(&format!("            let s{i} = vreinterpretq_u32_s32(vshrq_n_s32::<31>(a[{i}]));\n"));
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
            fn bitcast_f32_to_i32(a: float32x4_t) -> int32x4_t {{
                unsafe {{ vreinterpretq_s32_f32(a) }}
            }}

            #[inline(always)]
            fn bitcast_i32_to_f32(a: int32x4_t) -> float32x4_t {{
                unsafe {{ vreinterpretq_f32_s32(a) }}
            }}

            #[inline(always)]
            fn convert_f32_to_i32(a: float32x4_t) -> int32x4_t {{
                unsafe {{ vcvtq_s32_f32(a) }}
            }}

            #[inline(always)]
            fn convert_f32_to_i32_round(a: float32x4_t) -> int32x4_t {{
                unsafe {{ vcvtnq_s32_f32(a) }}
            }}

            #[inline(always)]
            fn convert_i32_to_f32(a: int32x4_t) -> float32x4_t {{
                unsafe {{ vcvtq_f32_s32(a) }}
            }}
        }}

        #[cfg(target_arch = "aarch64")]
        impl F32x8Convert for archmage::NeonToken {{
            #[inline(always)]
            fn bitcast_f32_to_i32(a: [float32x4_t; 2]) -> [int32x4_t; 2] {{
                unsafe {{ [vreinterpretq_s32_f32(a[0]), vreinterpretq_s32_f32(a[1])] }}
            }}

            #[inline(always)]
            fn bitcast_i32_to_f32(a: [int32x4_t; 2]) -> [float32x4_t; 2] {{
                unsafe {{ [vreinterpretq_f32_s32(a[0]), vreinterpretq_f32_s32(a[1])] }}
            }}

            #[inline(always)]
            fn convert_f32_to_i32(a: [float32x4_t; 2]) -> [int32x4_t; 2] {{
                unsafe {{ [vcvtq_s32_f32(a[0]), vcvtq_s32_f32(a[1])] }}
            }}

            #[inline(always)]
            fn convert_f32_to_i32_round(a: [float32x4_t; 2]) -> [int32x4_t; 2] {{
                unsafe {{ [vcvtnq_s32_f32(a[0]), vcvtnq_s32_f32(a[1])] }}
            }}

            #[inline(always)]
            fn convert_i32_to_f32(a: [int32x4_t; 2]) -> [float32x4_t; 2] {{
                unsafe {{ [vcvtq_f32_s32(a[0]), vcvtq_f32_s32(a[1])] }}
            }}
        }}

        #[cfg(target_arch = "aarch64")]
        impl U32x4Bitcast for archmage::NeonToken {{
            #[inline(always)]
            fn bitcast_u32_to_i32(a: uint32x4_t) -> int32x4_t {{
                unsafe {{ vreinterpretq_s32_u32(a) }}
            }}

            #[inline(always)]
            fn bitcast_i32_to_u32(a: int32x4_t) -> uint32x4_t {{
                unsafe {{ vreinterpretq_u32_s32(a) }}
            }}
        }}

        #[cfg(target_arch = "aarch64")]
        impl U32x8Bitcast for archmage::NeonToken {{
            #[inline(always)]
            fn bitcast_u32_to_i32(a: [uint32x4_t; 2]) -> [int32x4_t; 2] {{
                unsafe {{ [vreinterpretq_s32_u32(a[0]), vreinterpretq_s32_u32(a[1])] }}
            }}

            #[inline(always)]
            fn bitcast_i32_to_u32(a: [int32x4_t; 2]) -> [uint32x4_t; 2] {{
                unsafe {{ [vreinterpretq_u32_s32(a[0]), vreinterpretq_u32_s32(a[1])] }}
            }}
        }}

        #[cfg(target_arch = "aarch64")]
        impl I64x2Bitcast for archmage::NeonToken {{
            #[inline(always)]
            fn bitcast_i64_to_f64(a: int64x2_t) -> float64x2_t {{
                unsafe {{ vreinterpretq_f64_s64(a) }}
            }}

            #[inline(always)]
            fn bitcast_f64_to_i64(a: float64x2_t) -> int64x2_t {{
                unsafe {{ vreinterpretq_s64_f64(a) }}
            }}
        }}

        #[cfg(target_arch = "aarch64")]
        impl I64x4Bitcast for archmage::NeonToken {{
            #[inline(always)]
            fn bitcast_i64_to_f64(a: [int64x2_t; 2]) -> [float64x2_t; 2] {{
                unsafe {{ [vreinterpretq_f64_s64(a[0]), vreinterpretq_f64_s64(a[1])] }}
            }}

            #[inline(always)]
            fn bitcast_f64_to_i64(a: [float64x2_t; 2]) -> [int64x2_t; 2] {{
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
            fn splat(v: i32) -> v128 {{ i32x4_splat(v) }}
            #[inline(always)]
            fn zero() -> v128 {{ i32x4_splat(0) }}
            #[inline(always)]
            fn load(data: &{array}) -> v128 {{ unsafe {{ v128_load(data.as_ptr().cast()) }} }}
            #[inline(always)]
            fn from_array(arr: {array}) -> v128 {{ Self::load(&arr) }}
            #[inline(always)]
            fn store(repr: v128, out: &mut {array}) {{ unsafe {{ v128_store(out.as_mut_ptr().cast(), repr) }}; }}
            #[inline(always)]
            fn to_array(repr: v128) -> {array} {{
                let mut out = [0i32; {lanes}];
                Self::store(repr, &mut out);
                out
            }}

            #[inline(always)]
            fn add(a: v128, b: v128) -> v128 {{ i32x4_add(a, b) }}
            #[inline(always)]
            fn sub(a: v128, b: v128) -> v128 {{ i32x4_sub(a, b) }}
            #[inline(always)]
            fn mul(a: v128, b: v128) -> v128 {{ i32x4_mul(a, b) }}
            #[inline(always)]
            fn neg(a: v128) -> v128 {{ i32x4_neg(a) }}
            #[inline(always)]
            fn min(a: v128, b: v128) -> v128 {{ i32x4_min(a, b) }}
            #[inline(always)]
            fn max(a: v128, b: v128) -> v128 {{ i32x4_max(a, b) }}
            #[inline(always)]
            fn abs(a: v128) -> v128 {{ i32x4_abs(a) }}

            #[inline(always)]
            fn simd_eq(a: v128, b: v128) -> v128 {{ i32x4_eq(a, b) }}
            #[inline(always)]
            fn simd_ne(a: v128, b: v128) -> v128 {{ i32x4_ne(a, b) }}
            #[inline(always)]
            fn simd_lt(a: v128, b: v128) -> v128 {{ i32x4_lt(a, b) }}
            #[inline(always)]
            fn simd_le(a: v128, b: v128) -> v128 {{ i32x4_le(a, b) }}
            #[inline(always)]
            fn simd_gt(a: v128, b: v128) -> v128 {{ i32x4_gt(a, b) }}
            #[inline(always)]
            fn simd_ge(a: v128, b: v128) -> v128 {{ i32x4_ge(a, b) }}
            #[inline(always)]
            fn blend(mask: v128, if_true: v128, if_false: v128) -> v128 {{
                v128_bitselect(if_true, if_false, mask)
            }}

            #[inline(always)]
            fn reduce_add(a: v128) -> i32 {{ {reduce_add_body} }}

            #[inline(always)]
            fn not(a: v128) -> v128 {{ v128_not(a) }}
            #[inline(always)]
            fn bitand(a: v128, b: v128) -> v128 {{ v128_and(a, b) }}
            #[inline(always)]
            fn bitor(a: v128, b: v128) -> v128 {{ v128_or(a, b) }}
            #[inline(always)]
            fn bitxor(a: v128, b: v128) -> v128 {{ v128_xor(a, b) }}

            #[inline(always)]
            fn shl_const<const N: i32>(a: v128) -> v128 {{ i32x4_shl(a, N as u32) }}
            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(a: v128) -> v128 {{ i32x4_shr(a, N as u32) }}
            #[inline(always)]
            fn shr_logical_const<const N: i32>(a: v128) -> v128 {{ u32x4_shr(a, N as u32) }}

            #[inline(always)]
            fn all_true(a: v128) -> bool {{ i32x4_all_true(a) }}
            #[inline(always)]
            fn any_true(a: v128) -> bool {{ v128_any_true(a) }}
            #[inline(always)]
            fn bitmask(a: v128) -> u32 {{ i32x4_bitmask(a) as u32 }}
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
            fn splat(v: i32) -> {repr} {{
                let v4 = i32x4_splat(v);
                [{v4_copies}]
            }}

            #[inline(always)]
            fn zero() -> {repr} {{
                let z = i32x4_splat(0);
                [{z_copies}]
            }}

            #[inline(always)]
            fn load(data: &{array}) -> {repr} {{
                unsafe {{
                    [{load_lanes}]
                }}
            }}

            #[inline(always)]
            fn from_array(arr: {array}) -> {repr} {{
                Self::load(&arr)
            }}

            #[inline(always)]
            fn store(repr: {repr}, out: &mut {array}) {{
                unsafe {{
                    {store_lanes}
                }}
            }}

            #[inline(always)]
            fn to_array(repr: {repr}) -> {array} {{
                let mut out = [0i32; {lanes}];
                Self::store(repr, &mut out);
                out
            }}

            #[inline(always)]
            fn add(a: {repr}, b: {repr}) -> {repr} {{ {add} }}
            #[inline(always)]
            fn sub(a: {repr}, b: {repr}) -> {repr} {{ {sub} }}
            #[inline(always)]
            fn mul(a: {repr}, b: {repr}) -> {repr} {{ {mul} }}
            #[inline(always)]
            fn neg(a: {repr}) -> {repr} {{ {neg} }}
            #[inline(always)]
            fn min(a: {repr}, b: {repr}) -> {repr} {{ {min} }}
            #[inline(always)]
            fn max(a: {repr}, b: {repr}) -> {repr} {{ {max} }}
            #[inline(always)]
            fn abs(a: {repr}) -> {repr} {{ {abs} }}

            #[inline(always)]
            fn simd_eq(a: {repr}, b: {repr}) -> {repr} {{ {eq} }}
            #[inline(always)]
            fn simd_ne(a: {repr}, b: {repr}) -> {repr} {{ {ne} }}
            #[inline(always)]
            fn simd_lt(a: {repr}, b: {repr}) -> {repr} {{ {lt} }}
            #[inline(always)]
            fn simd_le(a: {repr}, b: {repr}) -> {repr} {{ {le} }}
            #[inline(always)]
            fn simd_gt(a: {repr}, b: {repr}) -> {repr} {{ {gt} }}
            #[inline(always)]
            fn simd_ge(a: {repr}, b: {repr}) -> {repr} {{ {ge} }}
            #[inline(always)]
            fn blend(mask: {repr}, if_true: {repr}, if_false: {repr}) -> {repr} {{
                [{blend_lanes}]
            }}

            #[inline(always)]
            fn reduce_add(a: {repr}) -> i32 {{ {reduce_add} }}

            #[inline(always)]
            fn not(a: {repr}) -> {repr} {{ {not} }}
            #[inline(always)]
            fn bitand(a: {repr}, b: {repr}) -> {repr} {{ {and} }}
            #[inline(always)]
            fn bitor(a: {repr}, b: {repr}) -> {repr} {{ {or} }}
            #[inline(always)]
            fn bitxor(a: {repr}, b: {repr}) -> {repr} {{ {xor} }}

            #[inline(always)]
            fn shl_const<const N: i32>(a: {repr}) -> {repr} {{
                [{shl_lanes}]
            }}

            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(a: {repr}) -> {repr} {{
                [{shr_arith_lanes}]
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(a: {repr}) -> {repr} {{
                [{shr_logic_lanes}]
            }}

            #[inline(always)]
            fn all_true(a: {repr}) -> bool {{
                {all_true}
            }}

            #[inline(always)]
            fn any_true(a: {repr}) -> bool {{
                {any_true}
            }}

            #[inline(always)]
            fn bitmask(a: {repr}) -> u32 {{
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
            fn bitcast_f32_to_i32(a: v128) -> v128 {{ a }}

            #[inline(always)]
            fn bitcast_i32_to_f32(a: v128) -> v128 {{ a }}

            #[inline(always)]
            fn convert_f32_to_i32(a: v128) -> v128 {{
                i32x4_trunc_sat_f32x4(a)
            }}

            #[inline(always)]
            fn convert_f32_to_i32_round(a: v128) -> v128 {{
                i32x4_trunc_sat_f32x4(f32x4_nearest(a))
            }}

            #[inline(always)]
            fn convert_i32_to_f32(a: v128) -> v128 {{
                f32x4_convert_i32x4(a)
            }}
        }}

        #[cfg(target_arch = "wasm32")]
        impl F32x8Convert for archmage::Wasm128Token {{
            #[inline(always)]
            fn bitcast_f32_to_i32(a: [v128; 2]) -> [v128; 2] {{ a }}

            #[inline(always)]
            fn bitcast_i32_to_f32(a: [v128; 2]) -> [v128; 2] {{ a }}

            #[inline(always)]
            fn convert_f32_to_i32(a: [v128; 2]) -> [v128; 2] {{
                [i32x4_trunc_sat_f32x4(a[0]), i32x4_trunc_sat_f32x4(a[1])]
            }}

            #[inline(always)]
            fn convert_f32_to_i32_round(a: [v128; 2]) -> [v128; 2] {{
                [
                    i32x4_trunc_sat_f32x4(f32x4_nearest(a[0])),
                    i32x4_trunc_sat_f32x4(f32x4_nearest(a[1])),
                ]
            }}

            #[inline(always)]
            fn convert_i32_to_f32(a: [v128; 2]) -> [v128; 2] {{
                [f32x4_convert_i32x4(a[0]), f32x4_convert_i32x4(a[1])]
            }}
        }}

        #[cfg(target_arch = "wasm32")]
        impl U32x4Bitcast for archmage::Wasm128Token {{
            #[inline(always)]
            fn bitcast_u32_to_i32(a: v128) -> v128 {{ a }}

            #[inline(always)]
            fn bitcast_i32_to_u32(a: v128) -> v128 {{ a }}
        }}

        #[cfg(target_arch = "wasm32")]
        impl U32x8Bitcast for archmage::Wasm128Token {{
            #[inline(always)]
            fn bitcast_u32_to_i32(a: [v128; 2]) -> [v128; 2] {{ a }}

            #[inline(always)]
            fn bitcast_i32_to_u32(a: [v128; 2]) -> [v128; 2] {{ a }}
        }}

        #[cfg(target_arch = "wasm32")]
        impl I64x2Bitcast for archmage::Wasm128Token {{
            #[inline(always)]
            fn bitcast_i64_to_f64(a: v128) -> v128 {{ a }}

            #[inline(always)]
            fn bitcast_f64_to_i64(a: v128) -> v128 {{ a }}
        }}

        #[cfg(target_arch = "wasm32")]
        impl I64x4Bitcast for archmage::Wasm128Token {{
            #[inline(always)]
            fn bitcast_i64_to_f64(a: [v128; 2]) -> [v128; 2] {{ a }}

            #[inline(always)]
            fn bitcast_f64_to_i64(a: [v128; 2]) -> [v128; 2] {{ a }}
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
        /// Trait methods are **associated functions** (no `self`/token parameter).
        /// The implementing type `Self` (a token type) determines which platform
        /// intrinsics are used. All methods are `#[inline(always)]` in implementations.
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
            fn splat(v: u32) -> Self::Repr;

            /// All lanes zero.
            fn zero() -> Self::Repr;

            /// Load from an aligned array.
            fn load(data: &{array}) -> Self::Repr;

            /// Create from array (zero-cost transmute where possible).
            fn from_array(arr: {array}) -> Self::Repr;

            /// Store to array.
            fn store(repr: Self::Repr, out: &mut {array});

            /// Convert to array.
            fn to_array(repr: Self::Repr) -> {array};

            // ====== Arithmetic ======

            /// Lane-wise addition (wrapping).
            fn add(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise subtraction (wrapping).
            fn sub(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise multiplication (low 32 bits of each 32x32 product).
            fn mul(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            // ====== Math ======

            /// Lane-wise unsigned minimum.
            fn min(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise unsigned maximum.
            fn max(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            // ====== Comparisons ======
            // Return masks where each lane is all-1s (true) or all-0s (false).
            // All comparisons are unsigned.

            /// Lane-wise equality.
            fn simd_eq(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise inequality.
            fn simd_ne(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise unsigned less-than.
            fn simd_lt(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise unsigned less-than-or-equal.
            fn simd_le(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise unsigned greater-than.
            fn simd_gt(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise unsigned greater-than-or-equal.
            fn simd_ge(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Select lanes: where mask is all-1s pick `if_true`, else `if_false`.
            fn blend(mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;

            // ====== Reductions ======

            /// Sum all {lanes} lanes (wrapping).
            fn reduce_add(a: Self::Repr) -> u32;

            // ====== Bitwise ======

            /// Bitwise NOT.
            fn not(a: Self::Repr) -> Self::Repr;

            /// Bitwise AND.
            fn bitand(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Bitwise OR.
            fn bitor(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Bitwise XOR.
            fn bitxor(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            // ====== Shifts ======

            /// Shift left by constant.
            fn shl_const<const N: i32>(a: Self::Repr) -> Self::Repr;

            /// Logical shift right by constant (zero-filling).
            fn shr_logical_const<const N: i32>(a: Self::Repr) -> Self::Repr;

            // ====== Boolean ======

            /// True if all lanes have their sign bit set (all-1s mask).
            fn all_true(a: Self::Repr) -> bool;

            /// True if any lane has its sign bit set (any all-1s mask lane).
            fn any_true(a: Self::Repr) -> bool;

            /// Extract the high bit of each 32-bit lane as a bitmask.
            fn bitmask(a: Self::Repr) -> u32;

            // ====== Default implementations ======

            /// Clamp values between lo and hi (unsigned comparison).
            #[inline(always)]
            fn clamp(a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {{
                Self::min(Self::max(a, lo), hi)
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
            fn splat(v: u32) -> {inner} {{
                unsafe {{ {p}_set1_epi32(v as i32) }}
            }}

            #[inline(always)]
            fn zero() -> {inner} {{
                unsafe {{ {p}_setzero_si{bits}() }}
            }}

            #[inline(always)]
            fn load(data: &{array}) -> {inner} {{
                unsafe {{ {p}_loadu_si{bits}(data.as_ptr().cast()) }}
            }}

            #[inline(always)]
            fn from_array(arr: {array}) -> {inner} {{
                // SAFETY: {array} and {inner} have identical size and layout.
                unsafe {{ core::mem::transmute(arr) }}
            }}

            #[inline(always)]
            fn store(repr: {inner}, out: &mut {array}) {{
                unsafe {{ {p}_storeu_si{bits}(out.as_mut_ptr().cast(), repr) }};
            }}

            #[inline(always)]
            fn to_array(repr: {inner}) -> {array} {{
                let mut out = [0u32; {lanes}];
                unsafe {{ {p}_storeu_si{bits}(out.as_mut_ptr().cast(), repr) }};
                out
            }}

            // ====== Arithmetic ======

            #[inline(always)]
            fn add(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_add_epi32(a, b) }}
            }}

            #[inline(always)]
            fn sub(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_sub_epi32(a, b) }}
            }}

            #[inline(always)]
            fn mul(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_mullo_epi32(a, b) }}
            }}

            // ====== Math ======

            #[inline(always)]
            fn min(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_min_epu32(a, b) }}
            }}

            #[inline(always)]
            fn max(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_max_epu32(a, b) }}
            }}

            // ====== Comparisons ======

            #[inline(always)]
            fn simd_eq(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_cmpeq_epi32(a, b) }}
            }}

            #[inline(always)]
            fn simd_ne(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{
                    let eq = {p}_cmpeq_epi32(a, b);
                    {p}_andnot_si{bits}(eq, {p}_set1_epi32(-1))
                }}
            }}

            #[inline(always)]
            fn simd_gt(a: {inner}, b: {inner}) -> {inner} {{
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
            fn simd_lt(a: {inner}, b: {inner}) -> {inner} {{
                <Self as {trait_name}>::simd_gt(b, a)
            }}

            #[inline(always)]
            fn simd_le(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{
                    let gt = <Self as {trait_name}>::simd_gt(a, b);
                    {p}_andnot_si{bits}(gt, {p}_set1_epi32(-1))
                }}
            }}

            #[inline(always)]
            fn simd_ge(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{
                    let lt = <Self as {trait_name}>::simd_gt(b, a);
                    {p}_andnot_si{bits}(lt, {p}_set1_epi32(-1))
                }}
            }}

            #[inline(always)]
            fn blend(mask: {inner}, if_true: {inner}, if_false: {inner}) -> {inner} {{
                unsafe {{ {p}_blendv_epi8(if_false, if_true, mask) }}
            }}

            // ====== Reductions ======

            #[inline(always)]
            fn reduce_add(a: {inner}) -> u32 {{
        {reduce_add_body}
            }}

            // ====== Bitwise ======

            #[inline(always)]
            fn not(a: {inner}) -> {inner} {{
                unsafe {{ {p}_andnot_si{bits}(a, {p}_set1_epi32(-1)) }}
            }}

            #[inline(always)]
            fn bitand(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_and_si{bits}(a, b) }}
            }}

            #[inline(always)]
            fn bitor(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_or_si{bits}(a, b) }}
            }}

            #[inline(always)]
            fn bitxor(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ {p}_xor_si{bits}(a, b) }}
            }}

            // ====== Shifts ======

            #[inline(always)]
            fn shl_const<const N: i32>(a: {inner}) -> {inner} {{
                unsafe {{ {p}_slli_epi32::<N>(a) }}
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(a: {inner}) -> {inner} {{
                unsafe {{ {p}_srli_epi32::<N>(a) }}
            }}

            // ====== Boolean ======

            #[inline(always)]
            fn all_true(a: {inner}) -> bool {{
                unsafe {{ {p}_movemask_ps({p}_castsi{bits}_ps(a)) == {all_mask} }}
            }}

            #[inline(always)]
            fn any_true(a: {inner}) -> bool {{
                unsafe {{ {p}_movemask_ps({p}_castsi{bits}_ps(a)) != 0 }}
            }}

            #[inline(always)]
            fn bitmask(a: {inner}) -> u32 {{
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
            fn splat(v: u32) -> {array} {{
                [v; {lanes}]
            }}

            #[inline(always)]
            fn zero() -> {array} {{
                [0u32; {lanes}]
            }}

            #[inline(always)]
            fn load(data: &{array}) -> {array} {{
                *data
            }}

            #[inline(always)]
            fn from_array(arr: {array}) -> {array} {{
                arr
            }}

            #[inline(always)]
            fn store(repr: {array}, out: &mut {array}) {{
                *out = repr;
            }}

            #[inline(always)]
            fn to_array(repr: {array}) -> {array} {{
                repr
            }}

            // ====== Arithmetic ======

            #[inline(always)]
            fn add(a: {array}, b: {array}) -> {array} {{
                {add_lanes}
            }}

            #[inline(always)]
            fn sub(a: {array}, b: {array}) -> {array} {{
                {sub_lanes}
            }}

            #[inline(always)]
            fn mul(a: {array}, b: {array}) -> {array} {{
                {mul_lanes}
            }}

            // ====== Math ======

            #[inline(always)]
            fn min(a: {array}, b: {array}) -> {array} {{
                {min_lanes}
            }}

            #[inline(always)]
            fn max(a: {array}, b: {array}) -> {array} {{
                {max_lanes}
            }}

            // ====== Comparisons ======

            #[inline(always)]
            fn simd_eq(a: {array}, b: {array}) -> {array} {{
                let mut r = [0u32; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] == b[i] {{ u32::MAX }} else {{ 0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_ne(a: {array}, b: {array}) -> {array} {{
                let mut r = [0u32; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] != b[i] {{ u32::MAX }} else {{ 0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_lt(a: {array}, b: {array}) -> {array} {{
                let mut r = [0u32; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] < b[i] {{ u32::MAX }} else {{ 0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_le(a: {array}, b: {array}) -> {array} {{
                let mut r = [0u32; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] <= b[i] {{ u32::MAX }} else {{ 0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_gt(a: {array}, b: {array}) -> {array} {{
                let mut r = [0u32; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] > b[i] {{ u32::MAX }} else {{ 0 }};
                }}
                r
            }}

            #[inline(always)]
            fn simd_ge(a: {array}, b: {array}) -> {array} {{
                let mut r = [0u32; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if a[i] >= b[i] {{ u32::MAX }} else {{ 0 }};
                }}
                r
            }}

            #[inline(always)]
            fn blend(mask: {array}, if_true: {array}, if_false: {array}) -> {array} {{
                let mut r = [0u32; {lanes}];
                for i in 0..{lanes} {{
                    r[i] = if mask[i] != 0 {{ if_true[i] }} else {{ if_false[i] }};
                }}
                r
            }}

            // ====== Reductions ======

            #[inline(always)]
            fn reduce_add(a: {array}) -> u32 {{
                {reduce_add}
            }}

            // ====== Bitwise ======

            #[inline(always)]
            fn not(a: {array}) -> {array} {{
                {not_lanes}
            }}

            #[inline(always)]
            fn bitand(a: {array}, b: {array}) -> {array} {{
                {and_lanes}
            }}

            #[inline(always)]
            fn bitor(a: {array}, b: {array}) -> {array} {{
                {or_lanes}
            }}

            #[inline(always)]
            fn bitxor(a: {array}, b: {array}) -> {array} {{
                {xor_lanes}
            }}

            // ====== Shifts ======

            #[inline(always)]
            fn shl_const<const N: i32>(a: {array}) -> {array} {{
                {shl}
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(a: {array}) -> {array} {{
                {shr_logical}
            }}

            // ====== Boolean ======

            #[inline(always)]
            fn all_true(a: {array}) -> bool {{
                {all_true}
            }}

            #[inline(always)]
            fn any_true(a: {array}) -> bool {{
                {any_true}
            }}

            #[inline(always)]
            fn bitmask(a: {array}) -> u32 {{
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
            fn splat(v: u32) -> uint32x4_t {{
                unsafe {{ vdupq_n_u32(v) }}
            }}

            #[inline(always)]
            fn zero() -> uint32x4_t {{
                unsafe {{ vdupq_n_u32(0) }}
            }}

            #[inline(always)]
            fn load(data: &{array}) -> uint32x4_t {{
                unsafe {{ vld1q_u32(data.as_ptr()) }}
            }}

            #[inline(always)]
            fn from_array(arr: {array}) -> uint32x4_t {{
                unsafe {{ vld1q_u32(arr.as_ptr()) }}
            }}

            #[inline(always)]
            fn store(repr: uint32x4_t, out: &mut {array}) {{
                unsafe {{ vst1q_u32(out.as_mut_ptr(), repr) }};
            }}

            #[inline(always)]
            fn to_array(repr: uint32x4_t) -> {array} {{
                let mut out = [0u32; {lanes}];
                unsafe {{ vst1q_u32(out.as_mut_ptr(), repr) }};
                out
            }}

            #[inline(always)]
            fn add(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{ unsafe {{ vaddq_u32(a, b) }} }}
            #[inline(always)]
            fn sub(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{ unsafe {{ vsubq_u32(a, b) }} }}
            #[inline(always)]
            fn mul(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{ unsafe {{ vmulq_u32(a, b) }} }}
            #[inline(always)]
            fn min(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{ unsafe {{ vminq_u32(a, b) }} }}
            #[inline(always)]
            fn max(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{ unsafe {{ vmaxq_u32(a, b) }} }}

            #[inline(always)]
            fn simd_eq(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{
                unsafe {{ vceqq_u32(a, b) }}
            }}
            #[inline(always)]
            fn simd_ne(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{
                unsafe {{ vmvnq_u32(vceqq_u32(a, b)) }}
            }}
            #[inline(always)]
            fn simd_lt(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{
                unsafe {{ vcltq_u32(a, b) }}
            }}
            #[inline(always)]
            fn simd_le(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{
                unsafe {{ vcleq_u32(a, b) }}
            }}
            #[inline(always)]
            fn simd_gt(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{
                unsafe {{ vcgtq_u32(a, b) }}
            }}
            #[inline(always)]
            fn simd_ge(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{
                unsafe {{ vcgeq_u32(a, b) }}
            }}

            #[inline(always)]
            fn blend(mask: uint32x4_t, if_true: uint32x4_t, if_false: uint32x4_t) -> uint32x4_t {{
                unsafe {{ vbslq_u32(mask, if_true, if_false) }}
            }}

            #[inline(always)]
            fn reduce_add(a: uint32x4_t) -> u32 {{
                unsafe {{ vaddvq_u32(a) }}
            }}

            #[inline(always)]
            fn not(a: uint32x4_t) -> uint32x4_t {{
                unsafe {{ vmvnq_u32(a) }}
            }}
            #[inline(always)]
            fn bitand(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{
                unsafe {{ vandq_u32(a, b) }}
            }}
            #[inline(always)]
            fn bitor(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{
                unsafe {{ vorrq_u32(a, b) }}
            }}
            #[inline(always)]
            fn bitxor(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {{
                unsafe {{ veorq_u32(a, b) }}
            }}

            #[inline(always)]
            fn shl_const<const N: i32>(a: uint32x4_t) -> uint32x4_t {{
                unsafe {{ vshlq_n_u32::<N>(a) }}
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(a: uint32x4_t) -> uint32x4_t {{
                unsafe {{ vshrq_n_u32::<N>(a) }}
            }}

            #[inline(always)]
            fn all_true(a: uint32x4_t) -> bool {{
                unsafe {{ vminvq_u32(a) == u32::MAX }}
            }}

            #[inline(always)]
            fn any_true(a: uint32x4_t) -> bool {{
                unsafe {{ vmaxvq_u32(a) != 0 }}
            }}

            #[inline(always)]
            fn bitmask(a: uint32x4_t) -> u32 {{
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
            fn splat(v: u32) -> {repr} {{
                unsafe {{
                    let v4 = vdupq_n_u32(v);
                    [{v4_copies}]
                }}
            }}

            #[inline(always)]
            fn zero() -> {repr} {{
                unsafe {{
                    let z = vdupq_n_u32(0);
                    [{z_copies}]
                }}
            }}

            #[inline(always)]
            fn load(data: &{array}) -> {repr} {{
                unsafe {{
                    [{load_lanes}]
                }}
            }}

            #[inline(always)]
            fn from_array(arr: {array}) -> {repr} {{
                Self::load(&arr)
            }}

            #[inline(always)]
            fn store(repr: {repr}, out: &mut {array}) {{
                unsafe {{
                    {store_lanes}
                }}
            }}

            #[inline(always)]
            fn to_array(repr: {repr}) -> {array} {{
                let mut out = [0u32; {lanes}];
                Self::store(repr, &mut out);
                out
            }}

            #[inline(always)]
            fn add(a: {repr}, b: {repr}) -> {repr} {{ {add} }}
            #[inline(always)]
            fn sub(a: {repr}, b: {repr}) -> {repr} {{ {sub} }}
            #[inline(always)]
            fn mul(a: {repr}, b: {repr}) -> {repr} {{ {mul} }}
            #[inline(always)]
            fn min(a: {repr}, b: {repr}) -> {repr} {{ {min} }}
            #[inline(always)]
            fn max(a: {repr}, b: {repr}) -> {repr} {{ {max} }}

            #[inline(always)]
            fn simd_eq(a: {repr}, b: {repr}) -> {repr} {{ {eq} }}
            #[inline(always)]
            fn simd_ne(a: {repr}, b: {repr}) -> {repr} {{ {ne} }}
            #[inline(always)]
            fn simd_lt(a: {repr}, b: {repr}) -> {repr} {{ {lt} }}
            #[inline(always)]
            fn simd_le(a: {repr}, b: {repr}) -> {repr} {{ {le} }}
            #[inline(always)]
            fn simd_gt(a: {repr}, b: {repr}) -> {repr} {{ {gt} }}
            #[inline(always)]
            fn simd_ge(a: {repr}, b: {repr}) -> {repr} {{ {ge} }}

            #[inline(always)]
            fn blend(mask: {repr}, if_true: {repr}, if_false: {repr}) -> {repr} {{
                {blend}
            }}

            #[inline(always)]
            fn reduce_add(a: {repr}) -> u32 {{
                {reduce_add}
            }}

            #[inline(always)]
            fn not(a: {repr}) -> {repr} {{ {not} }}
            #[inline(always)]
            fn bitand(a: {repr}, b: {repr}) -> {repr} {{ {bitand} }}
            #[inline(always)]
            fn bitor(a: {repr}, b: {repr}) -> {repr} {{ {bitor} }}
            #[inline(always)]
            fn bitxor(a: {repr}, b: {repr}) -> {repr} {{ {bitxor} }}

            #[inline(always)]
            fn shl_const<const N: i32>(a: {repr}) -> {repr} {{
                {shl}
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(a: {repr}) -> {repr} {{
                {shr_logic}
            }}

            #[inline(always)]
            fn all_true(a: {repr}) -> bool {{
                {all_true}
            }}

            #[inline(always)]
            fn any_true(a: {repr}) -> bool {{
                {any_true}
            }}

            #[inline(always)]
            fn bitmask(a: {repr}) -> u32 {{
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
                .map(|i| format!("vshrq_n_u32::<N>(a[{i}])"))
                .collect();
            format!("unsafe {{ [{}] }}", items.join(", "))
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
            fn splat(v: u32) -> v128 {{ u32x4_splat(v) }}
            #[inline(always)]
            fn zero() -> v128 {{ u32x4_splat(0) }}
            #[inline(always)]
            fn load(data: &{array}) -> v128 {{ unsafe {{ v128_load(data.as_ptr().cast()) }} }}
            #[inline(always)]
            fn from_array(arr: {array}) -> v128 {{ Self::load(&arr) }}
            #[inline(always)]
            fn store(repr: v128, out: &mut {array}) {{ unsafe {{ v128_store(out.as_mut_ptr().cast(), repr) }}; }}
            #[inline(always)]
            fn to_array(repr: v128) -> {array} {{
                let mut out = [0u32; {lanes}];
                Self::store(repr, &mut out);
                out
            }}

            #[inline(always)]
            fn add(a: v128, b: v128) -> v128 {{ i32x4_add(a, b) }}
            #[inline(always)]
            fn sub(a: v128, b: v128) -> v128 {{ i32x4_sub(a, b) }}
            #[inline(always)]
            fn mul(a: v128, b: v128) -> v128 {{ i32x4_mul(a, b) }}
            #[inline(always)]
            fn min(a: v128, b: v128) -> v128 {{ u32x4_min(a, b) }}
            #[inline(always)]
            fn max(a: v128, b: v128) -> v128 {{ u32x4_max(a, b) }}

            #[inline(always)]
            fn simd_eq(a: v128, b: v128) -> v128 {{ i32x4_eq(a, b) }}
            #[inline(always)]
            fn simd_ne(a: v128, b: v128) -> v128 {{ i32x4_ne(a, b) }}
            #[inline(always)]
            fn simd_lt(a: v128, b: v128) -> v128 {{ u32x4_lt(a, b) }}
            #[inline(always)]
            fn simd_le(a: v128, b: v128) -> v128 {{ u32x4_le(a, b) }}
            #[inline(always)]
            fn simd_gt(a: v128, b: v128) -> v128 {{ u32x4_gt(a, b) }}
            #[inline(always)]
            fn simd_ge(a: v128, b: v128) -> v128 {{ u32x4_ge(a, b) }}
            #[inline(always)]
            fn blend(mask: v128, if_true: v128, if_false: v128) -> v128 {{
                v128_bitselect(if_true, if_false, mask)
            }}

            #[inline(always)]
            fn reduce_add(a: v128) -> u32 {{ {reduce_add_body} }}

            #[inline(always)]
            fn not(a: v128) -> v128 {{ v128_not(a) }}
            #[inline(always)]
            fn bitand(a: v128, b: v128) -> v128 {{ v128_and(a, b) }}
            #[inline(always)]
            fn bitor(a: v128, b: v128) -> v128 {{ v128_or(a, b) }}
            #[inline(always)]
            fn bitxor(a: v128, b: v128) -> v128 {{ v128_xor(a, b) }}

            #[inline(always)]
            fn shl_const<const N: i32>(a: v128) -> v128 {{ u32x4_shl(a, N as u32) }}
            #[inline(always)]
            fn shr_logical_const<const N: i32>(a: v128) -> v128 {{ u32x4_shr(a, N as u32) }}

            #[inline(always)]
            fn all_true(a: v128) -> bool {{ i32x4_all_true(a) }}
            #[inline(always)]
            fn any_true(a: v128) -> bool {{ v128_any_true(a) }}
            #[inline(always)]
            fn bitmask(a: v128) -> u32 {{ i32x4_bitmask(a) as u32 }}
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
            fn splat(v: u32) -> {repr} {{
                let v4 = u32x4_splat(v);
                [{v4_copies}]
            }}

            #[inline(always)]
            fn zero() -> {repr} {{
                let z = u32x4_splat(0);
                [{z_copies}]
            }}

            #[inline(always)]
            fn load(data: &{array}) -> {repr} {{
                unsafe {{
                    [{load_lanes}]
                }}
            }}

            #[inline(always)]
            fn from_array(arr: {array}) -> {repr} {{
                Self::load(&arr)
            }}

            #[inline(always)]
            fn store(repr: {repr}, out: &mut {array}) {{
                unsafe {{
                    {store_lanes}
                }}
            }}

            #[inline(always)]
            fn to_array(repr: {repr}) -> {array} {{
                let mut out = [0u32; {lanes}];
                Self::store(repr, &mut out);
                out
            }}

            #[inline(always)]
            fn add(a: {repr}, b: {repr}) -> {repr} {{ {add} }}
            #[inline(always)]
            fn sub(a: {repr}, b: {repr}) -> {repr} {{ {sub} }}
            #[inline(always)]
            fn mul(a: {repr}, b: {repr}) -> {repr} {{ {mul} }}
            #[inline(always)]
            fn min(a: {repr}, b: {repr}) -> {repr} {{ {min} }}
            #[inline(always)]
            fn max(a: {repr}, b: {repr}) -> {repr} {{ {max} }}

            #[inline(always)]
            fn simd_eq(a: {repr}, b: {repr}) -> {repr} {{ {eq} }}
            #[inline(always)]
            fn simd_ne(a: {repr}, b: {repr}) -> {repr} {{ {ne} }}
            #[inline(always)]
            fn simd_lt(a: {repr}, b: {repr}) -> {repr} {{ {lt} }}
            #[inline(always)]
            fn simd_le(a: {repr}, b: {repr}) -> {repr} {{ {le} }}
            #[inline(always)]
            fn simd_gt(a: {repr}, b: {repr}) -> {repr} {{ {gt} }}
            #[inline(always)]
            fn simd_ge(a: {repr}, b: {repr}) -> {repr} {{ {ge} }}
            #[inline(always)]
            fn blend(mask: {repr}, if_true: {repr}, if_false: {repr}) -> {repr} {{
                [{blend_lanes}]
            }}

            #[inline(always)]
            fn reduce_add(a: {repr}) -> u32 {{ {reduce_add} }}

            #[inline(always)]
            fn not(a: {repr}) -> {repr} {{ {not} }}
            #[inline(always)]
            fn bitand(a: {repr}, b: {repr}) -> {repr} {{ {and} }}
            #[inline(always)]
            fn bitor(a: {repr}, b: {repr}) -> {repr} {{ {or} }}
            #[inline(always)]
            fn bitxor(a: {repr}, b: {repr}) -> {repr} {{ {xor} }}

            #[inline(always)]
            fn shl_const<const N: i32>(a: {repr}) -> {repr} {{
                [{shl_lanes}]
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(a: {repr}) -> {repr} {{
                [{shr_logic_lanes}]
            }}

            #[inline(always)]
            fn all_true(a: {repr}) -> bool {{
                {all_true}
            }}

            #[inline(always)]
            fn any_true(a: {repr}) -> bool {{
                {any_true}
            }}

            #[inline(always)]
            fn bitmask(a: {repr}) -> u32 {{
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
