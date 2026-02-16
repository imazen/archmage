//! Backend codegen for 512-bit SIMD types: f32x16, f64x8, i8x64, u8x64,
//! i16x32, u16x32, i32x16, u32x16, i64x8, u64x8.
//!
//! Uses a unified W512Type to generate all 10 types with backends for:
//! - x86 V4 (native AVX-512 intrinsics)
//! - x86 V3 (2×256-bit polyfill)
//! - NEON (4×128-bit polyfill)
//! - WASM (4×128-bit polyfill)
//! - Scalar (array ops)

use indoc::formatdoc;

// ============================================================================
// Data Model
// ============================================================================

/// Categories of 512-bit types.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum W512Kind {
    Float,
    SignedInt,
    UnsignedInt,
}

/// A 512-bit SIMD vector type for backend generation.
#[derive(Clone, Debug)]
pub(super) struct W512Type {
    /// Element type: "f32", "f64", "i8", "u8", "i16", "u16", "i32", "u32", "i64", "u64"
    pub elem: &'static str,
    /// Number of lanes
    pub lanes: usize,
    /// Element size in bits
    pub elem_bits: usize,
    /// Kind (float, signed int, unsigned int)
    pub kind: W512Kind,
}

impl W512Type {
    /// Type name: "f32x16", "i8x64", etc.
    pub fn name(&self) -> String {
        format!("{}x{}", self.elem, self.lanes)
    }

    /// Trait name: "F32x16Backend", "I8x64Backend", etc.
    pub fn trait_name(&self) -> String {
        let upper = match self.elem {
            "f32" => "F32",
            "f64" => "F64",
            "i8" => "I8",
            "u8" => "U8",
            "i16" => "I16",
            "u16" => "U16",
            "i32" => "I32",
            "u32" => "U32",
            "i64" => "I64",
            "u64" => "U64",
            _ => unreachable!(),
        };
        format!("{upper}x{}Backend", self.lanes)
    }

    /// Array type: "[f32; 16]", "[i8; 64]", etc.
    fn array_type(&self) -> String {
        format!("[{}; {}]", self.elem, self.lanes)
    }

    fn is_float(&self) -> bool {
        self.kind == W512Kind::Float
    }

    fn is_signed(&self) -> bool {
        matches!(self.kind, W512Kind::Float | W512Kind::SignedInt)
    }

    /// x86 AVX-512 inner type
    fn x86_v4_inner(&self) -> &'static str {
        match self.kind {
            W512Kind::Float => match self.elem {
                "f32" => "__m512",
                "f64" => "__m512d",
                _ => unreachable!(),
            },
            _ => "__m512i",
        }
    }

    /// x86 AVX2 inner type for V3 polyfill (array of 2)
    fn x86_v3_repr(&self) -> String {
        let inner = match self.kind {
            W512Kind::Float => match self.elem {
                "f32" => "__m256",
                "f64" => "__m256d",
                _ => unreachable!(),
            },
            _ => "__m256i",
        };
        format!("[{inner}; 2]")
    }

    /// x86 AVX2 half type (single __m256/__m256d/__m256i)
    fn x86_v3_half(&self) -> &'static str {
        match self.kind {
            W512Kind::Float => match self.elem {
                "f32" => "__m256",
                "f64" => "__m256d",
                _ => unreachable!(),
            },
            _ => "__m256i",
        }
    }

    /// The 256-bit backend trait name for the half type
    fn half_backend_trait(&self) -> String {
        let upper = match self.elem {
            "f32" => "F32",
            "f64" => "F64",
            "i8" => "I8",
            "u8" => "U8",
            "i16" => "I16",
            "u16" => "U16",
            "i32" => "I32",
            "u32" => "U32",
            "i64" => "I64",
            "u64" => "U64",
            _ => unreachable!(),
        };
        let half_lanes = self.lanes / 2;
        format!("{upper}x{half_lanes}Backend")
    }

    /// Scalar repr type
    fn scalar_repr(&self) -> String {
        self.array_type()
    }

    /// NEON repr type (4×128-bit)
    fn neon_repr(&self) -> String {
        let native = self.neon_128_type();
        format!("[{native}; 4]")
    }

    /// NEON 128-bit native type
    fn neon_128_type(&self) -> &'static str {
        match self.elem {
            "f32" => "float32x4_t",
            "f64" => "float64x2_t",
            "i8" => "int8x16_t",
            "u8" => "uint8x16_t",
            "i16" => "int16x8_t",
            "u16" => "uint16x8_t",
            "i32" => "int32x4_t",
            "u32" => "uint32x4_t",
            "i64" => "int64x2_t",
            "u64" => "uint64x2_t",
            _ => unreachable!(),
        }
    }

    /// WASM repr type (4×v128)
    fn wasm_repr(&self) -> &'static str {
        "[v128; 4]"
    }

    /// The 128-bit backend trait name for the quarter type
    fn quarter_backend_trait(&self) -> String {
        let upper = match self.elem {
            "f32" => "F32",
            "f64" => "F64",
            "i8" => "I8",
            "u8" => "U8",
            "i16" => "I16",
            "u16" => "U16",
            "i32" => "I32",
            "u32" => "U32",
            "i64" => "I64",
            "u64" => "U64",
            _ => unreachable!(),
        };
        let quarter_lanes = self.lanes / 4;
        format!("{upper}x{quarter_lanes}Backend")
    }

    /// x86 AVX-512 intrinsic suffix for float ops: "ps" or "pd"
    fn x86_float_suffix(&self) -> &'static str {
        match self.elem {
            "f32" => "ps",
            "f64" => "pd",
            _ => panic!("not a float type"),
        }
    }

    /// x86 intrinsic suffix for integer set1
    fn x86_set1_suffix(&self) -> &'static str {
        match self.elem_bits {
            8 => "epi8",
            16 => "epi16",
            32 => "epi32",
            64 => "epi64",
            _ => unreachable!(),
        }
    }

    /// x86 intrinsic suffix for integer arithmetic
    fn x86_arith_suffix(&self) -> &'static str {
        match self.elem_bits {
            8 => "epi8",
            16 => "epi16",
            32 => "epi32",
            64 => "epi64",
            _ => unreachable!(),
        }
    }

    /// x86 AVX-512 suffix for signed min/max/comparison
    fn x86_minmax_suffix(&self) -> &'static str {
        match (self.is_signed(), self.elem_bits) {
            (_, 8) if self.elem == "i8" => "epi8",
            (_, 8) => "epu8",
            (_, 16) if self.elem == "i16" => "epi16",
            (_, 16) => "epu16",
            (_, 32) if self.elem == "i32" => "epi32",
            (_, 32) => "epu32",
            (_, 64) if self.elem == "i64" => "epi64",
            (_, 64) => "epu64",
            _ => unreachable!(),
        }
    }

    /// Whether this integer type has native AVX-512 multiply (16-bit, 32-bit)
    fn has_native_avx512_mul(&self) -> bool {
        matches!(self.elem_bits, 16 | 32)
    }

    /// AVX-512 mask type for comparisons
    fn avx512_mask_type(&self) -> String {
        format!("__mmask{}", self.lanes)
    }

    /// Lanes per 128-bit sub-vector
    fn lanes_per_128(&self) -> usize {
        128 / self.elem_bits
    }

    /// Zero literal for the element type
    fn zero_lit(&self) -> &'static str {
        match self.elem {
            "f32" => "0.0f32",
            "f64" => "0.0f64",
            "i8" | "i16" | "i32" | "i64" => "0",
            "u8" | "u16" | "u32" | "u64" => "0",
            _ => unreachable!(),
        }
    }
}

/// All 10 512-bit vector types.
pub(super) fn all_w512_types() -> Vec<W512Type> {
    vec![
        W512Type {
            elem: "f32",
            lanes: 16,
            elem_bits: 32,
            kind: W512Kind::Float,
        },
        W512Type {
            elem: "f64",
            lanes: 8,
            elem_bits: 64,
            kind: W512Kind::Float,
        },
        W512Type {
            elem: "i8",
            lanes: 64,
            elem_bits: 8,
            kind: W512Kind::SignedInt,
        },
        W512Type {
            elem: "u8",
            lanes: 64,
            elem_bits: 8,
            kind: W512Kind::UnsignedInt,
        },
        W512Type {
            elem: "i16",
            lanes: 32,
            elem_bits: 16,
            kind: W512Kind::SignedInt,
        },
        W512Type {
            elem: "u16",
            lanes: 32,
            elem_bits: 16,
            kind: W512Kind::UnsignedInt,
        },
        W512Type {
            elem: "i32",
            lanes: 16,
            elem_bits: 32,
            kind: W512Kind::SignedInt,
        },
        W512Type {
            elem: "u32",
            lanes: 16,
            elem_bits: 32,
            kind: W512Kind::UnsignedInt,
        },
        W512Type {
            elem: "i64",
            lanes: 8,
            elem_bits: 64,
            kind: W512Kind::SignedInt,
        },
        W512Type {
            elem: "u64",
            lanes: 8,
            elem_bits: 64,
            kind: W512Kind::UnsignedInt,
        },
    ]
}

// ============================================================================
// Backend Trait Generation
// ============================================================================

/// Generate a backend trait for a W512 float type.
fn generate_float_backend_trait(ty: &W512Type) -> String {
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
            fn rcp_approx(a: Self::Repr) -> Self::Repr {{
                Self::div(Self::splat(1.0), a)
            }}

            /// Fast reciprocal square root approximation (~12-bit precision where available).
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
                Self::mul(approx, Self::sub(two, Self::mul(a, approx)))
            }}

            /// Precise reciprocal square root (Newton-Raphson from rsqrt_approx).
            #[inline(always)]
            fn rsqrt(a: Self::Repr) -> Self::Repr {{
                let approx = Self::rsqrt_approx(a);
                let half = Self::splat(0.5);
                let three = Self::splat(3.0);
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

/// Generate a backend trait for a W512 integer type.
fn generate_int_backend_trait(ty: &W512Type) -> String {
    let trait_name = ty.trait_name();
    let elem = ty.elem;
    let lanes = ty.lanes;
    let array = ty.array_type();
    let name = ty.name();

    // Integer types: no div, no floor/ceil/round, no sqrt, no fma
    // Have: shifts, boolean reductions, abs (signed only)
    let abs_section = if ty.is_signed() {
        formatdoc! {r#"

            /// Lane-wise absolute value.
            fn abs(a: Self::Repr) -> Self::Repr;
        "#}
    } else {
        String::new()
    };

    // mul only for 16-bit and 32-bit
    let mul_section = if ty.elem_bits == 16 || ty.elem_bits == 32 {
        formatdoc! {r#"

            /// Lane-wise multiplication (low bits of product).
            fn mul(a: Self::Repr, b: Self::Repr) -> Self::Repr;
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
            {mul_section}
            /// Lane-wise negation.
            fn neg(a: Self::Repr) -> Self::Repr;

            // ====== Math ======

            /// Lane-wise minimum.
            fn min(a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise maximum.
            fn max(a: Self::Repr, b: Self::Repr) -> Self::Repr;
            {abs_section}
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

            /// Extract the high bit of each lane as a bitmask.
            fn bitmask(a: Self::Repr) -> u64;

            // ====== Default implementations ======

            /// Clamp values between lo and hi.
            #[inline(always)]
            fn clamp(a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {{
                Self::min(Self::max(a, lo), hi)
            }}
        }}
    "#}
}

/// Generate backend trait definition for any W512 type.
pub(super) fn generate_w512_backend_trait(ty: &W512Type) -> String {
    if ty.is_float() {
        generate_float_backend_trait(ty)
    } else {
        generate_int_backend_trait(ty)
    }
}

// ============================================================================
// Scalar Implementation Generation
// ============================================================================

/// Generate scalar backend implementation for a W512 float type.
fn generate_scalar_float_impl(ty: &W512Type) -> String {
    let trait_name = ty.trait_name();
    let elem = ty.elem;
    let lanes = ty.lanes;
    let array = ty.array_type();
    let zero_lit = ty.zero_lit();

    formatdoc! {r#"
        impl {trait_name} for archmage::ScalarToken {{
            type Repr = {array};

            #[inline(always)]
            fn splat(v: {elem}) -> {array} {{ [{elem_name}; {lanes}].map(|_| v) }}

            #[inline(always)]
            fn zero() -> {array} {{ [{zero_lit}; {lanes}] }}

            #[inline(always)]
            fn load(data: &{array}) -> {array} {{ *data }}

            #[inline(always)]
            fn from_array(arr: {array}) -> {array} {{ arr }}

            #[inline(always)]
            fn store(repr: {array}, out: &mut {array}) {{ *out = repr; }}

            #[inline(always)]
            fn to_array(repr: {array}) -> {array} {{ repr }}

            #[inline(always)]
            fn add(a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| a[i] + b[i]) }}

            #[inline(always)]
            fn sub(a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| a[i] - b[i]) }}

            #[inline(always)]
            fn mul(a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| a[i] * b[i]) }}

            #[inline(always)]
            fn div(a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| a[i] / b[i]) }}

            #[inline(always)]
            fn neg(a: {array}) -> {array} {{ core::array::from_fn(|i| -a[i]) }}

            #[inline(always)]
            fn min(a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| if a[i] < b[i] {{ a[i] }} else {{ b[i] }}) }}

            #[inline(always)]
            fn max(a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| if a[i] > b[i] {{ a[i] }} else {{ b[i] }}) }}

            #[inline(always)]
            fn sqrt(a: {array}) -> {array} {{
                core::array::from_fn(|i| {{
                    #[cfg(feature = "std")]
                    {{ a[i].sqrt() }}
                    #[cfg(not(feature = "std"))]
                    {{ libm::{sqrt_fn}(a[i]{as_f64}) {from_f64} }}
                }})
            }}

            #[inline(always)]
            fn abs(a: {array}) -> {array} {{
                core::array::from_fn(|i| {{
                    #[cfg(feature = "std")]
                    {{ a[i].abs() }}
                    #[cfg(not(feature = "std"))]
                    {{ libm::{fabs_fn}(a[i]{as_f64}) {from_f64} }}
                }})
            }}

            #[inline(always)]
            fn floor(a: {array}) -> {array} {{
                core::array::from_fn(|i| {{
                    #[cfg(feature = "std")]
                    {{ a[i].floor() }}
                    #[cfg(not(feature = "std"))]
                    {{ libm::{floor_fn}(a[i]{as_f64}) {from_f64} }}
                }})
            }}

            #[inline(always)]
            fn ceil(a: {array}) -> {array} {{
                core::array::from_fn(|i| {{
                    #[cfg(feature = "std")]
                    {{ a[i].ceil() }}
                    #[cfg(not(feature = "std"))]
                    {{ libm::{ceil_fn}(a[i]{as_f64}) {from_f64} }}
                }})
            }}

            #[inline(always)]
            fn round(a: {array}) -> {array} {{
                core::array::from_fn(|i| {{
                    #[cfg(feature = "std")]
                    {{ a[i].round() }}
                    #[cfg(not(feature = "std"))]
                    {{ libm::{round_fn}(a[i]{as_f64}) {from_f64} }}
                }})
            }}

            #[inline(always)]
            fn mul_add(a: {array}, b: {array}, c: {array}) -> {array} {{ core::array::from_fn(|i| a[i] * b[i] + c[i]) }}

            #[inline(always)]
            fn mul_sub(a: {array}, b: {array}, c: {array}) -> {array} {{ core::array::from_fn(|i| a[i] * b[i] - c[i]) }}

            #[inline(always)]
            fn simd_eq(a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| if a[i] == b[i] {{ {elem}::from_bits(!0{uint}) }} else {{ {zero_lit} }}) }}

            #[inline(always)]
            fn simd_ne(a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| if a[i] != b[i] {{ {elem}::from_bits(!0{uint}) }} else {{ {zero_lit} }}) }}

            #[inline(always)]
            fn simd_lt(a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| if a[i] < b[i] {{ {elem}::from_bits(!0{uint}) }} else {{ {zero_lit} }}) }}

            #[inline(always)]
            fn simd_le(a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| if a[i] <= b[i] {{ {elem}::from_bits(!0{uint}) }} else {{ {zero_lit} }}) }}

            #[inline(always)]
            fn simd_gt(a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| if a[i] > b[i] {{ {elem}::from_bits(!0{uint}) }} else {{ {zero_lit} }}) }}

            #[inline(always)]
            fn simd_ge(a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| if a[i] >= b[i] {{ {elem}::from_bits(!0{uint}) }} else {{ {zero_lit} }}) }}

            #[inline(always)]
            fn blend(mask: {array}, if_true: {array}, if_false: {array}) -> {array} {{
                core::array::from_fn(|i| if mask[i].to_bits() != 0 {{ if_true[i] }} else {{ if_false[i] }})
            }}

            #[inline(always)]
            fn reduce_add(a: {array}) -> {elem} {{ a.iter().sum() }}

            #[inline(always)]
            fn reduce_min(a: {array}) -> {elem} {{ a.iter().copied().fold({elem}::INFINITY, {elem}::min) }}

            #[inline(always)]
            fn reduce_max(a: {array}) -> {elem} {{ a.iter().copied().fold({elem}::NEG_INFINITY, {elem}::max) }}

            #[inline(always)]
            fn not(a: {array}) -> {array} {{ core::array::from_fn(|i| {elem}::from_bits(!a[i].to_bits())) }}

            #[inline(always)]
            fn bitand(a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| {elem}::from_bits(a[i].to_bits() & b[i].to_bits())) }}

            #[inline(always)]
            fn bitor(a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| {elem}::from_bits(a[i].to_bits() | b[i].to_bits())) }}

            #[inline(always)]
            fn bitxor(a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| {elem}::from_bits(a[i].to_bits() ^ b[i].to_bits())) }}
        }}
    "#,
        elem_name = format!("{zero_lit}"),
        uint = if elem == "f32" { "u32" } else { "u64" },
        sqrt_fn = if elem == "f32" { "sqrtf" } else { "sqrt" },
        fabs_fn = if elem == "f32" { "fabsf" } else { "fabs" },
        floor_fn = if elem == "f32" { "floorf" } else { "floor" },
        ceil_fn = if elem == "f32" { "ceilf" } else { "ceil" },
        round_fn = if elem == "f32" { "roundf" } else { "round" },
        as_f64 = if elem == "f32" { " as f64" } else { "" },
        from_f64 = if elem == "f32" { " as f32" } else { "" },
    }
}

/// Generate scalar backend implementation for a W512 integer type.
fn generate_scalar_int_impl(ty: &W512Type) -> String {
    let trait_name = ty.trait_name();
    let elem = ty.elem;
    let lanes = ty.lanes;
    let array = ty.array_type();
    let zero_lit = ty.zero_lit();

    let mul_impl = if ty.elem_bits == 16 || ty.elem_bits == 32 {
        formatdoc! {r#"

            #[inline(always)]
            fn mul(a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| a[i].wrapping_mul(b[i])) }}
        "#}
    } else {
        String::new()
    };

    let abs_impl = if ty.is_signed() {
        formatdoc! {r#"

            #[inline(always)]
            fn abs(a: {array}) -> {array} {{ core::array::from_fn(|i| a[i].wrapping_abs()) }}
        "#}
    } else {
        String::new()
    };

    let neg_impl = if ty.is_signed() {
        format!("core::array::from_fn(|i| a[i].wrapping_neg())")
    } else {
        // For unsigned: negate via wrapping
        format!("core::array::from_fn(|i| (0{elem}).wrapping_sub(a[i]))")
    };

    let cmp_signed = if ty.is_signed() { "" } else { "" };
    let _ = cmp_signed;

    // For shifts of byte types, N could exceed bit width, so use wrapping
    let shl_body = match ty.elem_bits {
        8 => format!(
            "core::array::from_fn(|i| if N < 8 {{ (a[i] as u8).wrapping_shl(N as u32) as {elem} }} else {{ 0 }})"
        ),
        16 => format!(
            "core::array::from_fn(|i| if N < 16 {{ (a[i] as u16).wrapping_shl(N as u32) as {elem} }} else {{ 0 }})"
        ),
        32 => format!(
            "core::array::from_fn(|i| if N < 32 {{ (a[i] as u32).wrapping_shl(N as u32) as {elem} }} else {{ 0 }})"
        ),
        64 => format!(
            "core::array::from_fn(|i| if N < 64 {{ (a[i] as u64).wrapping_shl(N as u32) as {elem} }} else {{ 0 }})"
        ),
        _ => unreachable!(),
    };

    let shr_arith_body = if ty.is_signed() {
        match ty.elem_bits {
            8 => format!(
                "core::array::from_fn(|i| if N < 8 {{ a[i].wrapping_shr(N as u32) }} else {{ if a[i] < 0 {{ -1 }} else {{ 0 }} }})"
            ),
            16 => format!(
                "core::array::from_fn(|i| if N < 16 {{ a[i].wrapping_shr(N as u32) }} else {{ if a[i] < 0 {{ -1 }} else {{ 0 }} }})"
            ),
            32 => format!(
                "core::array::from_fn(|i| if N < 32 {{ a[i].wrapping_shr(N as u32) }} else {{ if a[i] < 0 {{ -1 }} else {{ 0 }} }})"
            ),
            64 => format!(
                "core::array::from_fn(|i| if N < 64 {{ a[i].wrapping_shr(N as u32) }} else {{ if a[i] < 0 {{ -1 }} else {{ 0 }} }})"
            ),
            _ => unreachable!(),
        }
    } else {
        match ty.elem_bits {
            8 => format!(
                "core::array::from_fn(|i| if N < 8 {{ a[i].wrapping_shr(N as u32) }} else {{ 0 }})"
            ),
            16 => format!(
                "core::array::from_fn(|i| if N < 16 {{ a[i].wrapping_shr(N as u32) }} else {{ 0 }})"
            ),
            32 => format!(
                "core::array::from_fn(|i| if N < 32 {{ a[i].wrapping_shr(N as u32) }} else {{ 0 }})"
            ),
            64 => format!(
                "core::array::from_fn(|i| if N < 64 {{ a[i].wrapping_shr(N as u32) }} else {{ 0 }})"
            ),
            _ => unreachable!(),
        }
    };

    let shr_logical_body = match ty.elem_bits {
        8 => format!(
            "core::array::from_fn(|i| if N < 8 {{ (a[i] as u8).wrapping_shr(N as u32) as {elem} }} else {{ 0 }})"
        ),
        16 => format!(
            "core::array::from_fn(|i| if N < 16 {{ (a[i] as u16).wrapping_shr(N as u32) as {elem} }} else {{ 0 }})"
        ),
        32 => format!(
            "core::array::from_fn(|i| if N < 32 {{ (a[i] as u32).wrapping_shr(N as u32) as {elem} }} else {{ 0 }})"
        ),
        64 => format!(
            "core::array::from_fn(|i| if N < 64 {{ (a[i] as u64).wrapping_shr(N as u32) as {elem} }} else {{ 0 }})"
        ),
        _ => unreachable!(),
    };

    // reduce_add with wrapping for integers
    let reduce_add = format!("a.iter().copied().fold(0{elem}, {elem}::wrapping_add)");

    // Bitmask: extract high bit of each element lane
    let bitmask_body = format!(
        "let mut mask = 0u64; for i in 0..{lanes} {{ if (a[i] as {sign_cast}) < 0 {{ mask |= 1u64 << i; }} }} mask",
        sign_cast = if ty.is_signed() {
            elem.to_string()
        } else {
            // For unsigned, interpret as signed for sign bit check
            match ty.elem_bits {
                8 => "i8".to_string(),
                16 => "i16".to_string(),
                32 => "i32".to_string(),
                64 => "i64".to_string(),
                _ => unreachable!(),
            }
        },
    );

    formatdoc! {r#"
        impl {trait_name} for archmage::ScalarToken {{
            type Repr = {array};

            #[inline(always)]
            fn splat(v: {elem}) -> {array} {{ [v; {lanes}] }}

            #[inline(always)]
            fn zero() -> {array} {{ [{zero_lit}; {lanes}] }}

            #[inline(always)]
            fn load(data: &{array}) -> {array} {{ *data }}

            #[inline(always)]
            fn from_array(arr: {array}) -> {array} {{ arr }}

            #[inline(always)]
            fn store(repr: {array}, out: &mut {array}) {{ *out = repr; }}

            #[inline(always)]
            fn to_array(repr: {array}) -> {array} {{ repr }}

            #[inline(always)]
            fn add(a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| a[i].wrapping_add(b[i])) }}

            #[inline(always)]
            fn sub(a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| a[i].wrapping_sub(b[i])) }}
            {mul_impl}
            #[inline(always)]
            fn neg(a: {array}) -> {array} {{ {neg_impl} }}

            #[inline(always)]
            fn min(a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| if a[i] < b[i] {{ a[i] }} else {{ b[i] }}) }}

            #[inline(always)]
            fn max(a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| if a[i] > b[i] {{ a[i] }} else {{ b[i] }}) }}
            {abs_impl}
            #[inline(always)]
            fn simd_eq(a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| if a[i] == b[i] {{ !{zero_for_not} }} else {{ {zero_lit} }}) }}

            #[inline(always)]
            fn simd_ne(a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| if a[i] != b[i] {{ !{zero_for_not} }} else {{ {zero_lit} }}) }}

            #[inline(always)]
            fn simd_lt(a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| if a[i] < b[i] {{ !{zero_for_not} }} else {{ {zero_lit} }}) }}

            #[inline(always)]
            fn simd_le(a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| if a[i] <= b[i] {{ !{zero_for_not} }} else {{ {zero_lit} }}) }}

            #[inline(always)]
            fn simd_gt(a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| if a[i] > b[i] {{ !{zero_for_not} }} else {{ {zero_lit} }}) }}

            #[inline(always)]
            fn simd_ge(a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| if a[i] >= b[i] {{ !{zero_for_not} }} else {{ {zero_lit} }}) }}

            #[inline(always)]
            fn blend(mask: {array}, if_true: {array}, if_false: {array}) -> {array} {{
                core::array::from_fn(|i| if mask[i] != {zero_lit} {{ if_true[i] }} else {{ if_false[i] }})
            }}

            #[inline(always)]
            fn reduce_add(a: {array}) -> {elem} {{ {reduce_add} }}

            #[inline(always)]
            fn not(a: {array}) -> {array} {{ core::array::from_fn(|i| !a[i]) }}

            #[inline(always)]
            fn bitand(a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| a[i] & b[i]) }}

            #[inline(always)]
            fn bitor(a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| a[i] | b[i]) }}

            #[inline(always)]
            fn bitxor(a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| a[i] ^ b[i]) }}

            #[inline(always)]
            fn shl_const<const N: i32>(a: {array}) -> {array} {{ {shl_body} }}

            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(a: {array}) -> {array} {{ {shr_arith_body} }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(a: {array}) -> {array} {{ {shr_logical_body} }}

            #[inline(always)]
            fn all_true(a: {array}) -> bool {{ a.iter().all(|&v| v != {zero_lit}) }}

            #[inline(always)]
            fn any_true(a: {array}) -> bool {{ a.iter().any(|&v| v != {zero_lit}) }}

            #[inline(always)]
            fn bitmask(a: {array}) -> u64 {{ {bitmask_body} }}
        }}
    "#,
        zero_for_not = format!("0{elem}"),
    }
}

/// Generate all scalar W512 implementations.
pub(super) fn generate_scalar_w512_impls(types: &[W512Type]) -> String {
    let mut code = String::new();
    for ty in types {
        if ty.is_float() {
            code.push_str(&generate_scalar_float_impl(ty));
        } else {
            code.push_str(&generate_scalar_int_impl(ty));
        }
        code.push('\n');
    }
    code
}

// ============================================================================
// x86 V3 Polyfill Implementation (2×256-bit)
// ============================================================================

/// Generate V3 polyfill implementation that delegates to the 256-bit backend.
fn generate_v3_polyfill_impl(ty: &W512Type) -> String {
    let trait_name = ty.trait_name();
    let half_trait = ty.half_backend_trait();
    let v3_repr = ty.x86_v3_repr();
    let _half = ty.x86_v3_half();
    let elem = ty.elem;
    let lanes = ty.lanes;
    let half_lanes = lanes / 2;
    let array = ty.array_type();
    let zero_lit = ty.zero_lit();

    let mut code = formatdoc! {r#"
        #[cfg(target_arch = "x86_64")]
        impl {trait_name} for archmage::X64V3Token {{
            type Repr = {v3_repr};

            #[inline(always)]
            fn splat(v: {elem}) -> {v3_repr} {{
                let h = <archmage::X64V3Token as {half_trait}>::splat(v);
                [h, h]
            }}

            #[inline(always)]
            fn zero() -> {v3_repr} {{
                let h = <archmage::X64V3Token as {half_trait}>::zero();
                [h, h]
            }}

            #[inline(always)]
            fn load(data: &{array}) -> {v3_repr} {{
                let (lo, hi) = data.split_at({half_lanes});
                [
                    <archmage::X64V3Token as {half_trait}>::load(lo.try_into().unwrap()),
                    <archmage::X64V3Token as {half_trait}>::load(hi.try_into().unwrap()),
                ]
            }}

            #[inline(always)]
            fn from_array(arr: {array}) -> {v3_repr} {{
                let mut lo = [{zero_lit}; {half_lanes}];
                let mut hi = [{zero_lit}; {half_lanes}];
                lo.copy_from_slice(&arr[..{half_lanes}]);
                hi.copy_from_slice(&arr[{half_lanes}..]);
                [
                    <archmage::X64V3Token as {half_trait}>::from_array(lo),
                    <archmage::X64V3Token as {half_trait}>::from_array(hi),
                ]
            }}

            #[inline(always)]
            fn store(repr: {v3_repr}, out: &mut {array}) {{
                let (lo, hi) = out.split_at_mut({half_lanes});
                <archmage::X64V3Token as {half_trait}>::store(repr[0], lo.try_into().unwrap());
                <archmage::X64V3Token as {half_trait}>::store(repr[1], hi.try_into().unwrap());
            }}

            #[inline(always)]
            fn to_array(repr: {v3_repr}) -> {array} {{
                let lo = <archmage::X64V3Token as {half_trait}>::to_array(repr[0]);
                let hi = <archmage::X64V3Token as {half_trait}>::to_array(repr[1]);
                let mut out = [{zero_lit}; {lanes}];
                out[..{half_lanes}].copy_from_slice(&lo);
                out[{half_lanes}..].copy_from_slice(&hi);
                out
            }}

            #[inline(always)]
            fn add(a: {v3_repr}, b: {v3_repr}) -> {v3_repr} {{
                [
                    <archmage::X64V3Token as {half_trait}>::add(a[0], b[0]),
                    <archmage::X64V3Token as {half_trait}>::add(a[1], b[1]),
                ]
            }}

            #[inline(always)]
            fn sub(a: {v3_repr}, b: {v3_repr}) -> {v3_repr} {{
                [
                    <archmage::X64V3Token as {half_trait}>::sub(a[0], b[0]),
                    <archmage::X64V3Token as {half_trait}>::sub(a[1], b[1]),
                ]
            }}
    "#};

    // Float-specific ops
    if ty.is_float() {
        code.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn mul(a: {v3_repr}, b: {v3_repr}) -> {v3_repr} {{
                [
                    <archmage::X64V3Token as {half_trait}>::mul(a[0], b[0]),
                    <archmage::X64V3Token as {half_trait}>::mul(a[1], b[1]),
                ]
            }}

            #[inline(always)]
            fn div(a: {v3_repr}, b: {v3_repr}) -> {v3_repr} {{
                [
                    <archmage::X64V3Token as {half_trait}>::div(a[0], b[0]),
                    <archmage::X64V3Token as {half_trait}>::div(a[1], b[1]),
                ]
            }}
        "#});
    } else if ty.elem_bits == 16 || ty.elem_bits == 32 {
        // Integer mul for 16/32-bit types
        code.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn mul(a: {v3_repr}, b: {v3_repr}) -> {v3_repr} {{
                [
                    <archmage::X64V3Token as {half_trait}>::mul(a[0], b[0]),
                    <archmage::X64V3Token as {half_trait}>::mul(a[1], b[1]),
                ]
            }}
        "#});
    }

    // neg: signed delegates to half backend, unsigned uses sub(zero, a)
    if ty.kind != W512Kind::UnsignedInt {
        code.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn neg(a: {v3_repr}) -> {v3_repr} {{
                [
                    <archmage::X64V3Token as {half_trait}>::neg(a[0]),
                    <archmage::X64V3Token as {half_trait}>::neg(a[1]),
                ]
            }}
        "#});
    } else {
        code.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn neg(a: {v3_repr}) -> {v3_repr} {{
                let z = <archmage::X64V3Token as {half_trait}>::zero();
                [
                    <archmage::X64V3Token as {half_trait}>::sub(z, a[0]),
                    <archmage::X64V3Token as {half_trait}>::sub(z, a[1]),
                ]
            }}
        "#});
    }

    code.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn min(a: {v3_repr}, b: {v3_repr}) -> {v3_repr} {{
                [
                    <archmage::X64V3Token as {half_trait}>::min(a[0], b[0]),
                    <archmage::X64V3Token as {half_trait}>::min(a[1], b[1]),
                ]
            }}

            #[inline(always)]
            fn max(a: {v3_repr}, b: {v3_repr}) -> {v3_repr} {{
                [
                    <archmage::X64V3Token as {half_trait}>::max(a[0], b[0]),
                    <archmage::X64V3Token as {half_trait}>::max(a[1], b[1]),
                ]
            }}
    "#});

    // Float math ops
    if ty.is_float() {
        code.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn sqrt(a: {v3_repr}) -> {v3_repr} {{
                [
                    <archmage::X64V3Token as {half_trait}>::sqrt(a[0]),
                    <archmage::X64V3Token as {half_trait}>::sqrt(a[1]),
                ]
            }}

            #[inline(always)]
            fn abs(a: {v3_repr}) -> {v3_repr} {{
                [
                    <archmage::X64V3Token as {half_trait}>::abs(a[0]),
                    <archmage::X64V3Token as {half_trait}>::abs(a[1]),
                ]
            }}

            #[inline(always)]
            fn floor(a: {v3_repr}) -> {v3_repr} {{
                [
                    <archmage::X64V3Token as {half_trait}>::floor(a[0]),
                    <archmage::X64V3Token as {half_trait}>::floor(a[1]),
                ]
            }}

            #[inline(always)]
            fn ceil(a: {v3_repr}) -> {v3_repr} {{
                [
                    <archmage::X64V3Token as {half_trait}>::ceil(a[0]),
                    <archmage::X64V3Token as {half_trait}>::ceil(a[1]),
                ]
            }}

            #[inline(always)]
            fn round(a: {v3_repr}) -> {v3_repr} {{
                [
                    <archmage::X64V3Token as {half_trait}>::round(a[0]),
                    <archmage::X64V3Token as {half_trait}>::round(a[1]),
                ]
            }}

            #[inline(always)]
            fn mul_add(a: {v3_repr}, b: {v3_repr}, c: {v3_repr}) -> {v3_repr} {{
                [
                    <archmage::X64V3Token as {half_trait}>::mul_add(a[0], b[0], c[0]),
                    <archmage::X64V3Token as {half_trait}>::mul_add(a[1], b[1], c[1]),
                ]
            }}

            #[inline(always)]
            fn mul_sub(a: {v3_repr}, b: {v3_repr}, c: {v3_repr}) -> {v3_repr} {{
                [
                    <archmage::X64V3Token as {half_trait}>::mul_sub(a[0], b[0], c[0]),
                    <archmage::X64V3Token as {half_trait}>::mul_sub(a[1], b[1], c[1]),
                ]
            }}

            #[inline(always)]
            fn reduce_add(a: {v3_repr}) -> {elem} {{
                <archmage::X64V3Token as {half_trait}>::reduce_add(a[0])
                    + <archmage::X64V3Token as {half_trait}>::reduce_add(a[1])
            }}

            #[inline(always)]
            fn reduce_min(a: {v3_repr}) -> {elem} {{
                let lo = <archmage::X64V3Token as {half_trait}>::reduce_min(a[0]);
                let hi = <archmage::X64V3Token as {half_trait}>::reduce_min(a[1]);
                if lo < hi {{ lo }} else {{ hi }}
            }}

            #[inline(always)]
            fn reduce_max(a: {v3_repr}) -> {elem} {{
                let lo = <archmage::X64V3Token as {half_trait}>::reduce_max(a[0]);
                let hi = <archmage::X64V3Token as {half_trait}>::reduce_max(a[1]);
                if lo > hi {{ lo }} else {{ hi }}
            }}

            #[inline(always)]
            fn rcp_approx(a: {v3_repr}) -> {v3_repr} {{
                [
                    <archmage::X64V3Token as {half_trait}>::rcp_approx(a[0]),
                    <archmage::X64V3Token as {half_trait}>::rcp_approx(a[1]),
                ]
            }}

            #[inline(always)]
            fn rsqrt_approx(a: {v3_repr}) -> {v3_repr} {{
                [
                    <archmage::X64V3Token as {half_trait}>::rsqrt_approx(a[0]),
                    <archmage::X64V3Token as {half_trait}>::rsqrt_approx(a[1]),
                ]
            }}
        "#});
    } else {
        // Integer-specific: abs (signed only), reduce, shifts, boolean
        if ty.is_signed() {
            code.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn abs(a: {v3_repr}) -> {v3_repr} {{
                [
                    <archmage::X64V3Token as {half_trait}>::abs(a[0]),
                    <archmage::X64V3Token as {half_trait}>::abs(a[1]),
                ]
            }}
            "#});
        }

        // For unsigned types, 256-bit backends don't have shr_arithmetic_const,
        // so delegate to shr_logical_const (identical for unsigned).
        let shr_arith_delegate = if ty.kind == W512Kind::UnsignedInt {
            "shr_logical_const"
        } else {
            "shr_arithmetic_const"
        };

        code.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn reduce_add(a: {v3_repr}) -> {elem} {{
                <archmage::X64V3Token as {half_trait}>::reduce_add(a[0])
                    .wrapping_add(<archmage::X64V3Token as {half_trait}>::reduce_add(a[1]))
            }}

            #[inline(always)]
            fn shl_const<const N: i32>(a: {v3_repr}) -> {v3_repr} {{
                [
                    <archmage::X64V3Token as {half_trait}>::shl_const::<N>(a[0]),
                    <archmage::X64V3Token as {half_trait}>::shl_const::<N>(a[1]),
                ]
            }}

            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(a: {v3_repr}) -> {v3_repr} {{
                [
                    <archmage::X64V3Token as {half_trait}>::{shr_arith_delegate}::<N>(a[0]),
                    <archmage::X64V3Token as {half_trait}>::{shr_arith_delegate}::<N>(a[1]),
                ]
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(a: {v3_repr}) -> {v3_repr} {{
                [
                    <archmage::X64V3Token as {half_trait}>::shr_logical_const::<N>(a[0]),
                    <archmage::X64V3Token as {half_trait}>::shr_logical_const::<N>(a[1]),
                ]
            }}

            #[inline(always)]
            fn all_true(a: {v3_repr}) -> bool {{
                <archmage::X64V3Token as {half_trait}>::all_true(a[0])
                    && <archmage::X64V3Token as {half_trait}>::all_true(a[1])
            }}

            #[inline(always)]
            fn any_true(a: {v3_repr}) -> bool {{
                <archmage::X64V3Token as {half_trait}>::any_true(a[0])
                    || <archmage::X64V3Token as {half_trait}>::any_true(a[1])
            }}

            #[inline(always)]
            fn bitmask(a: {v3_repr}) -> u64 {{
                let lo = <archmage::X64V3Token as {half_trait}>::bitmask(a[0]) as u64;
                let hi = <archmage::X64V3Token as {half_trait}>::bitmask(a[1]) as u64;
                lo | (hi << {half_lanes})
            }}
        "#});
    }

    // Common ops: comparisons + bitwise
    code.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn simd_eq(a: {v3_repr}, b: {v3_repr}) -> {v3_repr} {{
                [
                    <archmage::X64V3Token as {half_trait}>::simd_eq(a[0], b[0]),
                    <archmage::X64V3Token as {half_trait}>::simd_eq(a[1], b[1]),
                ]
            }}

            #[inline(always)]
            fn simd_ne(a: {v3_repr}, b: {v3_repr}) -> {v3_repr} {{
                [
                    <archmage::X64V3Token as {half_trait}>::simd_ne(a[0], b[0]),
                    <archmage::X64V3Token as {half_trait}>::simd_ne(a[1], b[1]),
                ]
            }}

            #[inline(always)]
            fn simd_lt(a: {v3_repr}, b: {v3_repr}) -> {v3_repr} {{
                [
                    <archmage::X64V3Token as {half_trait}>::simd_lt(a[0], b[0]),
                    <archmage::X64V3Token as {half_trait}>::simd_lt(a[1], b[1]),
                ]
            }}

            #[inline(always)]
            fn simd_le(a: {v3_repr}, b: {v3_repr}) -> {v3_repr} {{
                [
                    <archmage::X64V3Token as {half_trait}>::simd_le(a[0], b[0]),
                    <archmage::X64V3Token as {half_trait}>::simd_le(a[1], b[1]),
                ]
            }}

            #[inline(always)]
            fn simd_gt(a: {v3_repr}, b: {v3_repr}) -> {v3_repr} {{
                [
                    <archmage::X64V3Token as {half_trait}>::simd_gt(a[0], b[0]),
                    <archmage::X64V3Token as {half_trait}>::simd_gt(a[1], b[1]),
                ]
            }}

            #[inline(always)]
            fn simd_ge(a: {v3_repr}, b: {v3_repr}) -> {v3_repr} {{
                [
                    <archmage::X64V3Token as {half_trait}>::simd_ge(a[0], b[0]),
                    <archmage::X64V3Token as {half_trait}>::simd_ge(a[1], b[1]),
                ]
            }}

            #[inline(always)]
            fn blend(mask: {v3_repr}, if_true: {v3_repr}, if_false: {v3_repr}) -> {v3_repr} {{
                [
                    <archmage::X64V3Token as {half_trait}>::blend(mask[0], if_true[0], if_false[0]),
                    <archmage::X64V3Token as {half_trait}>::blend(mask[1], if_true[1], if_false[1]),
                ]
            }}

            #[inline(always)]
            fn not(a: {v3_repr}) -> {v3_repr} {{
                [
                    <archmage::X64V3Token as {half_trait}>::not(a[0]),
                    <archmage::X64V3Token as {half_trait}>::not(a[1]),
                ]
            }}

            #[inline(always)]
            fn bitand(a: {v3_repr}, b: {v3_repr}) -> {v3_repr} {{
                [
                    <archmage::X64V3Token as {half_trait}>::bitand(a[0], b[0]),
                    <archmage::X64V3Token as {half_trait}>::bitand(a[1], b[1]),
                ]
            }}

            #[inline(always)]
            fn bitor(a: {v3_repr}, b: {v3_repr}) -> {v3_repr} {{
                [
                    <archmage::X64V3Token as {half_trait}>::bitor(a[0], b[0]),
                    <archmage::X64V3Token as {half_trait}>::bitor(a[1], b[1]),
                ]
            }}

            #[inline(always)]
            fn bitxor(a: {v3_repr}, b: {v3_repr}) -> {v3_repr} {{
                [
                    <archmage::X64V3Token as {half_trait}>::bitxor(a[0], b[0]),
                    <archmage::X64V3Token as {half_trait}>::bitxor(a[1], b[1]),
                ]
            }}
        }}
    "#});

    code
}

/// Generate all V3 polyfill W512 implementations.
pub(super) fn generate_x86_v3_w512_impls(types: &[W512Type]) -> String {
    let mut code = String::new();
    for ty in types {
        code.push_str(&generate_v3_polyfill_impl(ty));
        code.push('\n');
    }
    code
}

// ============================================================================
// NEON Polyfill Implementation (4×128-bit)
// Same structure as V3 but delegates to the 128-bit quarter backend.
// ============================================================================

/// Generate NEON polyfill implementation for a W512 type.
fn generate_neon_polyfill_impl(ty: &W512Type) -> String {
    generate_4way_polyfill_impl(ty, "NeonToken", "aarch64", &ty.quarter_backend_trait())
}

/// Generate WASM polyfill implementation for a W512 type.
fn generate_wasm_polyfill_impl(ty: &W512Type) -> String {
    generate_4way_polyfill_impl(ty, "Wasm128Token", "wasm32", &ty.quarter_backend_trait())
}

/// Generate a 4-way polyfill (NEON or WASM) that delegates to the 128-bit backend.
fn generate_4way_polyfill_impl(
    ty: &W512Type,
    token: &str,
    arch: &str,
    quarter_trait: &str,
) -> String {
    let trait_name = ty.trait_name();
    let elem = ty.elem;
    let lanes = ty.lanes;
    let q_lanes = lanes / 4;
    let array = ty.array_type();
    let zero_lit = ty.zero_lit();

    let repr = if token == "NeonToken" {
        ty.neon_repr()
    } else {
        ty.wasm_repr().to_string()
    };
    let _q_type = if token == "NeonToken" {
        ty.neon_128_type().to_string()
    } else {
        "v128".to_string()
    };

    let mut code = formatdoc! {r#"
        #[cfg(target_arch = "{arch}")]
        impl {trait_name} for archmage::{token} {{
            type Repr = {repr};

            #[inline(always)]
            fn splat(v: {elem}) -> {repr} {{
                let q = <archmage::{token} as {quarter_trait}>::splat(v);
                [q, q, q, q]
            }}

            #[inline(always)]
            fn zero() -> {repr} {{
                let q = <archmage::{token} as {quarter_trait}>::zero();
                [q, q, q, q]
            }}

            #[inline(always)]
            fn load(data: &{array}) -> {repr} {{
                [
                    <archmage::{token} as {quarter_trait}>::load(data[0..{q_lanes}].try_into().unwrap()),
                    <archmage::{token} as {quarter_trait}>::load(data[{q_lanes}..{q2}].try_into().unwrap()),
                    <archmage::{token} as {quarter_trait}>::load(data[{q2}..{q3}].try_into().unwrap()),
                    <archmage::{token} as {quarter_trait}>::load(data[{q3}..{lanes}].try_into().unwrap()),
                ]
            }}

            #[inline(always)]
            fn from_array(arr: {array}) -> {repr} {{
                let mut q0 = [{zero_lit}; {q_lanes}];
                let mut q1 = [{zero_lit}; {q_lanes}];
                let mut q2 = [{zero_lit}; {q_lanes}];
                let mut q3 = [{zero_lit}; {q_lanes}];
                q0.copy_from_slice(&arr[0..{q_lanes}]);
                q1.copy_from_slice(&arr[{q_lanes}..{q2_val}]);
                q2.copy_from_slice(&arr[{q2_val}..{q3_val}]);
                q3.copy_from_slice(&arr[{q3_val}..{lanes}]);
                [
                    <archmage::{token} as {quarter_trait}>::from_array(q0),
                    <archmage::{token} as {quarter_trait}>::from_array(q1),
                    <archmage::{token} as {quarter_trait}>::from_array(q2),
                    <archmage::{token} as {quarter_trait}>::from_array(q3),
                ]
            }}

            #[inline(always)]
            fn store(repr: {repr}, out: &mut {array}) {{
                let (o01, o23) = out.split_at_mut({q2_val});
                let (o0, o1) = o01.split_at_mut({q_lanes});
                let (o2, o3) = o23.split_at_mut({q_lanes});
                <archmage::{token} as {quarter_trait}>::store(repr[0], o0.try_into().unwrap());
                <archmage::{token} as {quarter_trait}>::store(repr[1], o1.try_into().unwrap());
                <archmage::{token} as {quarter_trait}>::store(repr[2], o2.try_into().unwrap());
                <archmage::{token} as {quarter_trait}>::store(repr[3], o3.try_into().unwrap());
            }}

            #[inline(always)]
            fn to_array(repr: {repr}) -> {array} {{
                let a0 = <archmage::{token} as {quarter_trait}>::to_array(repr[0]);
                let a1 = <archmage::{token} as {quarter_trait}>::to_array(repr[1]);
                let a2 = <archmage::{token} as {quarter_trait}>::to_array(repr[2]);
                let a3 = <archmage::{token} as {quarter_trait}>::to_array(repr[3]);
                let mut out = [{zero_lit}; {lanes}];
                out[0..{q_lanes}].copy_from_slice(&a0);
                out[{q_lanes}..{q2_val}].copy_from_slice(&a1);
                out[{q2_val}..{q3_val}].copy_from_slice(&a2);
                out[{q3_val}..{lanes}].copy_from_slice(&a3);
                out
            }}

            #[inline(always)]
            fn add(a: {repr}, b: {repr}) -> {repr} {{
                core::array::from_fn(|i| <archmage::{token} as {quarter_trait}>::add(a[i], b[i]))
            }}

            #[inline(always)]
            fn sub(a: {repr}, b: {repr}) -> {repr} {{
                core::array::from_fn(|i| <archmage::{token} as {quarter_trait}>::sub(a[i], b[i]))
            }}
    "#,
        q2 = q_lanes * 2,
        q3 = q_lanes * 3,
        q2_val = q_lanes * 2,
        q3_val = q_lanes * 3,
    };

    // Float: mul, div
    if ty.is_float() {
        code.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn mul(a: {repr}, b: {repr}) -> {repr} {{
                core::array::from_fn(|i| <archmage::{token} as {quarter_trait}>::mul(a[i], b[i]))
            }}

            #[inline(always)]
            fn div(a: {repr}, b: {repr}) -> {repr} {{
                core::array::from_fn(|i| <archmage::{token} as {quarter_trait}>::div(a[i], b[i]))
            }}
        "#});
    } else if ty.elem_bits == 16 || ty.elem_bits == 32 {
        code.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn mul(a: {repr}, b: {repr}) -> {repr} {{
                core::array::from_fn(|i| <archmage::{token} as {quarter_trait}>::mul(a[i], b[i]))
            }}
        "#});
    }

    // neg: signed delegates to quarter backend, unsigned uses sub(zero, a)
    if ty.kind != W512Kind::UnsignedInt {
        code.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn neg(a: {repr}) -> {repr} {{
                core::array::from_fn(|i| <archmage::{token} as {quarter_trait}>::neg(a[i]))
            }}
        "#});
    } else {
        code.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn neg(a: {repr}) -> {repr} {{
                let z = <archmage::{token} as {quarter_trait}>::zero();
                core::array::from_fn(|i| <archmage::{token} as {quarter_trait}>::sub(z, a[i]))
            }}
        "#});
    }

    code.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn min(a: {repr}, b: {repr}) -> {repr} {{
                core::array::from_fn(|i| <archmage::{token} as {quarter_trait}>::min(a[i], b[i]))
            }}

            #[inline(always)]
            fn max(a: {repr}, b: {repr}) -> {repr} {{
                core::array::from_fn(|i| <archmage::{token} as {quarter_trait}>::max(a[i], b[i]))
            }}
    "#});

    if ty.is_float() {
        code.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn sqrt(a: {repr}) -> {repr} {{
                core::array::from_fn(|i| <archmage::{token} as {quarter_trait}>::sqrt(a[i]))
            }}

            #[inline(always)]
            fn abs(a: {repr}) -> {repr} {{
                core::array::from_fn(|i| <archmage::{token} as {quarter_trait}>::abs(a[i]))
            }}

            #[inline(always)]
            fn floor(a: {repr}) -> {repr} {{
                core::array::from_fn(|i| <archmage::{token} as {quarter_trait}>::floor(a[i]))
            }}

            #[inline(always)]
            fn ceil(a: {repr}) -> {repr} {{
                core::array::from_fn(|i| <archmage::{token} as {quarter_trait}>::ceil(a[i]))
            }}

            #[inline(always)]
            fn round(a: {repr}) -> {repr} {{
                core::array::from_fn(|i| <archmage::{token} as {quarter_trait}>::round(a[i]))
            }}

            #[inline(always)]
            fn mul_add(a: {repr}, b: {repr}, c: {repr}) -> {repr} {{
                core::array::from_fn(|i| <archmage::{token} as {quarter_trait}>::mul_add(a[i], b[i], c[i]))
            }}

            #[inline(always)]
            fn mul_sub(a: {repr}, b: {repr}, c: {repr}) -> {repr} {{
                core::array::from_fn(|i| <archmage::{token} as {quarter_trait}>::mul_sub(a[i], b[i], c[i]))
            }}

            #[inline(always)]
            fn reduce_add(a: {repr}) -> {elem} {{
                <archmage::{token} as {quarter_trait}>::reduce_add(a[0])
                    + <archmage::{token} as {quarter_trait}>::reduce_add(a[1])
                    + <archmage::{token} as {quarter_trait}>::reduce_add(a[2])
                    + <archmage::{token} as {quarter_trait}>::reduce_add(a[3])
            }}

            #[inline(always)]
            fn reduce_min(a: {repr}) -> {elem} {{
                let m01 = {{
                    let l = <archmage::{token} as {quarter_trait}>::reduce_min(a[0]);
                    let r = <archmage::{token} as {quarter_trait}>::reduce_min(a[1]);
                    if l < r {{ l }} else {{ r }}
                }};
                let m23 = {{
                    let l = <archmage::{token} as {quarter_trait}>::reduce_min(a[2]);
                    let r = <archmage::{token} as {quarter_trait}>::reduce_min(a[3]);
                    if l < r {{ l }} else {{ r }}
                }};
                if m01 < m23 {{ m01 }} else {{ m23 }}
            }}

            #[inline(always)]
            fn reduce_max(a: {repr}) -> {elem} {{
                let m01 = {{
                    let l = <archmage::{token} as {quarter_trait}>::reduce_max(a[0]);
                    let r = <archmage::{token} as {quarter_trait}>::reduce_max(a[1]);
                    if l > r {{ l }} else {{ r }}
                }};
                let m23 = {{
                    let l = <archmage::{token} as {quarter_trait}>::reduce_max(a[2]);
                    let r = <archmage::{token} as {quarter_trait}>::reduce_max(a[3]);
                    if l > r {{ l }} else {{ r }}
                }};
                if m01 > m23 {{ m01 }} else {{ m23 }}
            }}
        "#});
    } else {
        // Integer: abs, reduce, shifts, boolean
        if ty.is_signed() {
            code.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn abs(a: {repr}) -> {repr} {{
                core::array::from_fn(|i| <archmage::{token} as {quarter_trait}>::abs(a[i]))
            }}
            "#});
        }

        // For unsigned types, shr_arithmetic_const delegates to shr_logical_const
        // (identical behavior for unsigned, and the quarter backends don't have shr_arithmetic_const)
        let shr_arith_delegate = if ty.kind == W512Kind::UnsignedInt {
            "shr_logical_const"
        } else {
            "shr_arithmetic_const"
        };

        code.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn reduce_add(a: {repr}) -> {elem} {{
                <archmage::{token} as {quarter_trait}>::reduce_add(a[0])
                    .wrapping_add(<archmage::{token} as {quarter_trait}>::reduce_add(a[1]))
                    .wrapping_add(<archmage::{token} as {quarter_trait}>::reduce_add(a[2]))
                    .wrapping_add(<archmage::{token} as {quarter_trait}>::reduce_add(a[3]))
            }}

            #[inline(always)]
            fn shl_const<const N: i32>(a: {repr}) -> {repr} {{
                core::array::from_fn(|i| <archmage::{token} as {quarter_trait}>::shl_const::<N>(a[i]))
            }}

            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(a: {repr}) -> {repr} {{
                core::array::from_fn(|i| <archmage::{token} as {quarter_trait}>::{shr_arith_delegate}::<N>(a[i]))
            }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(a: {repr}) -> {repr} {{
                core::array::from_fn(|i| <archmage::{token} as {quarter_trait}>::shr_logical_const::<N>(a[i]))
            }}

            #[inline(always)]
            fn all_true(a: {repr}) -> bool {{
                <archmage::{token} as {quarter_trait}>::all_true(a[0])
                    && <archmage::{token} as {quarter_trait}>::all_true(a[1])
                    && <archmage::{token} as {quarter_trait}>::all_true(a[2])
                    && <archmage::{token} as {quarter_trait}>::all_true(a[3])
            }}

            #[inline(always)]
            fn any_true(a: {repr}) -> bool {{
                <archmage::{token} as {quarter_trait}>::any_true(a[0])
                    || <archmage::{token} as {quarter_trait}>::any_true(a[1])
                    || <archmage::{token} as {quarter_trait}>::any_true(a[2])
                    || <archmage::{token} as {quarter_trait}>::any_true(a[3])
            }}

            #[inline(always)]
            fn bitmask(a: {repr}) -> u64 {{
                let q0 = <archmage::{token} as {quarter_trait}>::bitmask(a[0]) as u64;
                let q1 = <archmage::{token} as {quarter_trait}>::bitmask(a[1]) as u64;
                let q2 = <archmage::{token} as {quarter_trait}>::bitmask(a[2]) as u64;
                let q3 = <archmage::{token} as {quarter_trait}>::bitmask(a[3]) as u64;
                q0 | (q1 << {q_lanes}) | (q2 << {q2_lanes}) | (q3 << {q3_lanes})
            }}
        "#,
            q2_lanes = q_lanes * 2,
            q3_lanes = q_lanes * 3,
        });
    }

    // Common: comparisons and bitwise
    code.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn simd_eq(a: {repr}, b: {repr}) -> {repr} {{
                core::array::from_fn(|i| <archmage::{token} as {quarter_trait}>::simd_eq(a[i], b[i]))
            }}

            #[inline(always)]
            fn simd_ne(a: {repr}, b: {repr}) -> {repr} {{
                core::array::from_fn(|i| <archmage::{token} as {quarter_trait}>::simd_ne(a[i], b[i]))
            }}

            #[inline(always)]
            fn simd_lt(a: {repr}, b: {repr}) -> {repr} {{
                core::array::from_fn(|i| <archmage::{token} as {quarter_trait}>::simd_lt(a[i], b[i]))
            }}

            #[inline(always)]
            fn simd_le(a: {repr}, b: {repr}) -> {repr} {{
                core::array::from_fn(|i| <archmage::{token} as {quarter_trait}>::simd_le(a[i], b[i]))
            }}

            #[inline(always)]
            fn simd_gt(a: {repr}, b: {repr}) -> {repr} {{
                core::array::from_fn(|i| <archmage::{token} as {quarter_trait}>::simd_gt(a[i], b[i]))
            }}

            #[inline(always)]
            fn simd_ge(a: {repr}, b: {repr}) -> {repr} {{
                core::array::from_fn(|i| <archmage::{token} as {quarter_trait}>::simd_ge(a[i], b[i]))
            }}

            #[inline(always)]
            fn blend(mask: {repr}, if_true: {repr}, if_false: {repr}) -> {repr} {{
                core::array::from_fn(|i| <archmage::{token} as {quarter_trait}>::blend(mask[i], if_true[i], if_false[i]))
            }}

            #[inline(always)]
            fn not(a: {repr}) -> {repr} {{
                core::array::from_fn(|i| <archmage::{token} as {quarter_trait}>::not(a[i]))
            }}

            #[inline(always)]
            fn bitand(a: {repr}, b: {repr}) -> {repr} {{
                core::array::from_fn(|i| <archmage::{token} as {quarter_trait}>::bitand(a[i], b[i]))
            }}

            #[inline(always)]
            fn bitor(a: {repr}, b: {repr}) -> {repr} {{
                core::array::from_fn(|i| <archmage::{token} as {quarter_trait}>::bitor(a[i], b[i]))
            }}

            #[inline(always)]
            fn bitxor(a: {repr}, b: {repr}) -> {repr} {{
                core::array::from_fn(|i| <archmage::{token} as {quarter_trait}>::bitxor(a[i], b[i]))
            }}
        }}
    "#});

    code
}

/// Generate all NEON W512 polyfill implementations.
pub(super) fn generate_neon_w512_impls(types: &[W512Type]) -> String {
    let mut code = String::new();
    for ty in types {
        code.push_str(&generate_neon_polyfill_impl(ty));
        code.push('\n');
    }
    code
}

/// Generate all WASM W512 polyfill implementations.
pub(super) fn generate_wasm_w512_impls(types: &[W512Type]) -> String {
    let mut code = String::new();
    for ty in types {
        code.push_str(&generate_wasm_polyfill_impl(ty));
        code.push('\n');
    }
    code
}

// ============================================================================
// x86 V4 Native AVX-512 Implementation
// Uses __m512/__m512d/__m512i directly with AVX-512 intrinsics.
// ============================================================================

/// Generate V4 native implementation for a W512 float type (f32x16, f64x8).
fn generate_x86_v4_float_impl_for_token(ty: &W512Type, token: &str) -> String {
    let trait_name = ty.trait_name();
    let inner = ty.x86_v4_inner();
    let s = ty.x86_float_suffix();
    let elem = ty.elem;
    let array = ty.array_type();
    let epi = if elem == "f32" { "epi32" } else { "epi64" };
    let abs_mask = if elem == "f32" {
        "0x7FFF_FFFFu32 as i32"
    } else {
        "0x7FFF_FFFF_FFFF_FFFFu64 as i64"
    };

    formatdoc! {r#"
        #[cfg(target_arch = "x86_64")]
        impl {trait_name} for archmage::{token} {{
            type Repr = {inner};

            #[inline(always)]
            fn splat(v: {elem}) -> {inner} {{
                unsafe {{ _mm512_set1_{s}(v) }}
            }}

            #[inline(always)]
            fn zero() -> {inner} {{
                unsafe {{ _mm512_setzero_{s}() }}
            }}

            #[inline(always)]
            fn load(data: &{array}) -> {inner} {{
                unsafe {{ _mm512_loadu_{s}(data.as_ptr()) }}
            }}

            #[inline(always)]
            fn from_array(arr: {array}) -> {inner} {{
                unsafe {{ core::mem::transmute(arr) }}
            }}

            #[inline(always)]
            fn store(repr: {inner}, out: &mut {array}) {{
                unsafe {{ _mm512_storeu_{s}(out.as_mut_ptr(), repr) }}
            }}

            #[inline(always)]
            fn to_array(repr: {inner}) -> {array} {{
                unsafe {{ core::mem::transmute(repr) }}
            }}

            #[inline(always)]
            fn add(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ _mm512_add_{s}(a, b) }}
            }}

            #[inline(always)]
            fn sub(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ _mm512_sub_{s}(a, b) }}
            }}

            #[inline(always)]
            fn mul(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ _mm512_mul_{s}(a, b) }}
            }}

            #[inline(always)]
            fn div(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ _mm512_div_{s}(a, b) }}
            }}

            #[inline(always)]
            fn neg(a: {inner}) -> {inner} {{
                unsafe {{ _mm512_sub_{s}(_mm512_setzero_{s}(), a) }}
            }}

            #[inline(always)]
            fn min(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ _mm512_min_{s}(a, b) }}
            }}

            #[inline(always)]
            fn max(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ _mm512_max_{s}(a, b) }}
            }}

            #[inline(always)]
            fn sqrt(a: {inner}) -> {inner} {{
                unsafe {{ _mm512_sqrt_{s}(a) }}
            }}

            #[inline(always)]
            fn abs(a: {inner}) -> {inner} {{
                unsafe {{
                    let mask = _mm512_castsi512_{s}(_mm512_set1_{epi}({abs_mask}));
                    _mm512_and_{s}(a, mask)
                }}
            }}

            #[inline(always)]
            fn floor(a: {inner}) -> {inner} {{
                unsafe {{ _mm512_roundscale_{s}::<0x01>(a) }}
            }}

            #[inline(always)]
            fn ceil(a: {inner}) -> {inner} {{
                unsafe {{ _mm512_roundscale_{s}::<0x02>(a) }}
            }}

            #[inline(always)]
            fn round(a: {inner}) -> {inner} {{
                unsafe {{ _mm512_roundscale_{s}::<0x00>(a) }}
            }}

            #[inline(always)]
            fn mul_add(a: {inner}, b: {inner}, c: {inner}) -> {inner} {{
                unsafe {{ _mm512_fmadd_{s}(a, b, c) }}
            }}

            #[inline(always)]
            fn mul_sub(a: {inner}, b: {inner}, c: {inner}) -> {inner} {{
                unsafe {{ _mm512_fmsub_{s}(a, b, c) }}
            }}

            #[inline(always)]
            fn simd_eq(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{
                    let mask = _mm512_cmp_{s}_mask::<_CMP_EQ_OQ>(a, b);
                    _mm512_castsi512_{s}(_mm512_maskz_set1_{epi}(mask, -1))
                }}
            }}

            #[inline(always)]
            fn simd_ne(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{
                    let mask = _mm512_cmp_{s}_mask::<_CMP_NEQ_UQ>(a, b);
                    _mm512_castsi512_{s}(_mm512_maskz_set1_{epi}(mask, -1))
                }}
            }}

            #[inline(always)]
            fn simd_lt(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{
                    let mask = _mm512_cmp_{s}_mask::<_CMP_LT_OQ>(a, b);
                    _mm512_castsi512_{s}(_mm512_maskz_set1_{epi}(mask, -1))
                }}
            }}

            #[inline(always)]
            fn simd_le(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{
                    let mask = _mm512_cmp_{s}_mask::<_CMP_LE_OQ>(a, b);
                    _mm512_castsi512_{s}(_mm512_maskz_set1_{epi}(mask, -1))
                }}
            }}

            #[inline(always)]
            fn simd_gt(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{
                    let mask = _mm512_cmp_{s}_mask::<_CMP_GT_OQ>(a, b);
                    _mm512_castsi512_{s}(_mm512_maskz_set1_{epi}(mask, -1))
                }}
            }}

            #[inline(always)]
            fn simd_ge(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{
                    let mask = _mm512_cmp_{s}_mask::<_CMP_GE_OQ>(a, b);
                    _mm512_castsi512_{s}(_mm512_maskz_set1_{epi}(mask, -1))
                }}
            }}

            #[inline(always)]
            fn blend(mask: {inner}, if_true: {inner}, if_false: {inner}) -> {inner} {{
                unsafe {{
                    let mask_i = _mm512_cast{s}_si512(mask);
                    let k = _mm512_cmpneq_{epi}_mask(mask_i, _mm512_setzero_si512());
                    _mm512_mask_blend_{s}(k, if_false, if_true)
                }}
            }}

            #[inline(always)]
            fn reduce_add(a: {inner}) -> {elem} {{
                unsafe {{ _mm512_reduce_add_{s}(a) }}
            }}

            #[inline(always)]
            fn reduce_min(a: {inner}) -> {elem} {{
                unsafe {{ _mm512_reduce_min_{s}(a) }}
            }}

            #[inline(always)]
            fn reduce_max(a: {inner}) -> {elem} {{
                unsafe {{ _mm512_reduce_max_{s}(a) }}
            }}

            #[inline(always)]
            fn rcp_approx(a: {inner}) -> {inner} {{
                unsafe {{
                    let approx = _mm512_rcp14_{s}(a);
                    // One Newton-Raphson iteration: x' = x * (2 - a*x)
                    let two = _mm512_set1_{s}(2.0);
                    _mm512_mul_{s}(approx, _mm512_sub_{s}(two, _mm512_mul_{s}(a, approx)))
                }}
            }}

            #[inline(always)]
            fn rsqrt_approx(a: {inner}) -> {inner} {{
                unsafe {{
                    let approx = _mm512_rsqrt14_{s}(a);
                    // One Newton-Raphson iteration: x' = 0.5 * x * (3 - a*x*x)
                    let half = _mm512_set1_{s}(0.5);
                    let three = _mm512_set1_{s}(3.0);
                    _mm512_mul_{s}(
                        _mm512_mul_{s}(half, approx),
                        _mm512_sub_{s}(three, _mm512_mul_{s}(a, _mm512_mul_{s}(approx, approx)))
                    )
                }}
            }}

            #[inline(always)]
            fn not(a: {inner}) -> {inner} {{
                unsafe {{
                    let all_ones = _mm512_castsi512_{s}(_mm512_set1_{epi}(-1));
                    _mm512_xor_{s}(a, all_ones)
                }}
            }}

            #[inline(always)]
            fn bitand(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ _mm512_and_{s}(a, b) }}
            }}

            #[inline(always)]
            fn bitor(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ _mm512_or_{s}(a, b) }}
            }}

            #[inline(always)]
            fn bitxor(a: {inner}, b: {inner}) -> {inner} {{
                unsafe {{ _mm512_xor_{s}(a, b) }}
            }}
        }}
    "#}
}

/// Generate V4 native implementation for a W512 integer type.
fn generate_x86_v4_int_impl_for_token(ty: &W512Type, token: &str) -> String {
    let trait_name = ty.trait_name();
    let elem = ty.elem;
    let lanes = ty.lanes;
    let elem_bits = ty.elem_bits;
    let array = ty.array_type();
    let epi = ty.x86_arith_suffix();
    let mm = ty.x86_minmax_suffix();
    let set1 = ty.x86_set1_suffix();

    // set1 cast for unsigned types: we need to cast -1 to the unsigned element type
    // then to the signed type that _mm512_set1_epiN expects
    let set1_neg1 = match elem {
        "i8" | "i16" | "i32" | "i64" => "-1".to_string(),
        "u8" => "-1i8".to_string(),
        "u16" => "-1i16".to_string(),
        "u32" => "-1i32".to_string(),
        "u64" => "-1i64".to_string(),
        _ => unreachable!(),
    };

    // For maskz_set1, we always use the signed version
    let set1_signed = match elem_bits {
        8 => "epi8",
        16 => "epi16",
        32 => "epi32",
        64 => "epi64",
        _ => unreachable!(),
    };

    // Comparison intrinsic prefix: epi for signed, epu for unsigned
    let cmp_suffix = if ty.is_signed() {
        format!("epi{elem_bits}")
    } else {
        format!("epu{elem_bits}")
    };

    // blend suffix: epi8, epi16, epi32, epi64
    let blend_suffix = format!("epi{elem_bits}");

    // Mask type
    let full_mask = match lanes {
        64 => "0xFFFF_FFFF_FFFF_FFFFu64",
        32 => "0xFFFF_FFFFu64",
        16 => "0xFFFFu64",
        8 => "0xFFu64",
        _ => unreachable!(),
    };

    // mul
    let mul_impl = if ty.has_native_avx512_mul() {
        formatdoc! {r#"

            #[inline(always)]
            fn mul(a: __m512i, b: __m512i) -> __m512i {{
                unsafe {{ _mm512_mullo_{epi}(a, b) }}
            }}
        "#}
    } else {
        String::new()
    };

    // abs (signed only)
    let abs_impl = if ty.is_signed() {
        formatdoc! {r#"

            #[inline(always)]
            fn abs(a: __m512i) -> __m512i {{
                unsafe {{ _mm512_abs_{epi}(a) }}
            }}
        "#}
    } else {
        String::new()
    };

    // neg: sub(zero, a) works for both signed and unsigned
    let neg_impl = formatdoc! {r#"
            #[inline(always)]
            fn neg(a: __m512i) -> __m512i {{
                unsafe {{ _mm512_sub_{epi}(_mm512_setzero_si512(), a) }}
            }}
    "#};

    // Shifts: 8-bit shifts need polyfill via 16-bit, others use native
    let shift_impls = if elem_bits == 8 {
        generate_8bit_shift_polyfill(ty)
    } else {
        let shr_arith = if ty.kind == W512Kind::UnsignedInt {
            // Unsigned: arithmetic right shift = logical right shift
            // Inline the shift body to avoid Self:: ambiguity (multiple backend traits)
            formatdoc! {r#"
            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(a: __m512i) -> __m512i {{
                unsafe {{ _mm512_srl_{epi}(a, _mm_cvtsi32_si128(N)) }}
            }}
            "#}
        } else {
            formatdoc! {r#"
            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(a: __m512i) -> __m512i {{
                unsafe {{ _mm512_sra_{epi}(a, _mm_cvtsi32_si128(N)) }}
            }}
            "#}
        };

        formatdoc! {r#"
            #[inline(always)]
            fn shl_const<const N: i32>(a: __m512i) -> __m512i {{
                unsafe {{ _mm512_sll_{epi}(a, _mm_cvtsi32_si128(N)) }}
            }}

            {shr_arith}
            #[inline(always)]
            fn shr_logical_const<const N: i32>(a: __m512i) -> __m512i {{
                unsafe {{ _mm512_srl_{epi}(a, _mm_cvtsi32_si128(N)) }}
            }}
        "#}
    };

    formatdoc! {r#"
        #[cfg(target_arch = "x86_64")]
        impl {trait_name} for archmage::{token} {{
            type Repr = __m512i;

            #[inline(always)]
            fn splat(v: {elem}) -> __m512i {{
                unsafe {{ _mm512_set1_{set1}(v as _) }}
            }}

            #[inline(always)]
            fn zero() -> __m512i {{
                unsafe {{ _mm512_setzero_si512() }}
            }}

            #[inline(always)]
            fn load(data: &{array}) -> __m512i {{
                unsafe {{ _mm512_loadu_si512(data.as_ptr().cast()) }}
            }}

            #[inline(always)]
            fn from_array(arr: {array}) -> __m512i {{
                unsafe {{ core::mem::transmute(arr) }}
            }}

            #[inline(always)]
            fn store(repr: __m512i, out: &mut {array}) {{
                unsafe {{ _mm512_storeu_si512(out.as_mut_ptr().cast(), repr) }}
            }}

            #[inline(always)]
            fn to_array(repr: __m512i) -> {array} {{
                unsafe {{ core::mem::transmute(repr) }}
            }}

            #[inline(always)]
            fn add(a: __m512i, b: __m512i) -> __m512i {{
                unsafe {{ _mm512_add_{epi}(a, b) }}
            }}

            #[inline(always)]
            fn sub(a: __m512i, b: __m512i) -> __m512i {{
                unsafe {{ _mm512_sub_{epi}(a, b) }}
            }}
            {mul_impl}
            {neg_impl}
            #[inline(always)]
            fn min(a: __m512i, b: __m512i) -> __m512i {{
                unsafe {{ _mm512_min_{mm}(a, b) }}
            }}

            #[inline(always)]
            fn max(a: __m512i, b: __m512i) -> __m512i {{
                unsafe {{ _mm512_max_{mm}(a, b) }}
            }}
            {abs_impl}
            #[inline(always)]
            fn simd_eq(a: __m512i, b: __m512i) -> __m512i {{
                unsafe {{
                    let mask = _mm512_cmpeq_{cmp_suffix}_mask(a, b);
                    _mm512_maskz_set1_{set1_signed}(mask, {set1_neg1})
                }}
            }}

            #[inline(always)]
            fn simd_ne(a: __m512i, b: __m512i) -> __m512i {{
                unsafe {{
                    let mask = _mm512_cmpneq_{cmp_suffix}_mask(a, b);
                    _mm512_maskz_set1_{set1_signed}(mask, {set1_neg1})
                }}
            }}

            #[inline(always)]
            fn simd_lt(a: __m512i, b: __m512i) -> __m512i {{
                unsafe {{
                    let mask = _mm512_cmplt_{cmp_suffix}_mask(a, b);
                    _mm512_maskz_set1_{set1_signed}(mask, {set1_neg1})
                }}
            }}

            #[inline(always)]
            fn simd_le(a: __m512i, b: __m512i) -> __m512i {{
                unsafe {{
                    let mask = _mm512_cmple_{cmp_suffix}_mask(a, b);
                    _mm512_maskz_set1_{set1_signed}(mask, {set1_neg1})
                }}
            }}

            #[inline(always)]
            fn simd_gt(a: __m512i, b: __m512i) -> __m512i {{
                unsafe {{
                    // GT = LT with swapped args
                    let mask = _mm512_cmplt_{cmp_suffix}_mask(b, a);
                    _mm512_maskz_set1_{set1_signed}(mask, {set1_neg1})
                }}
            }}

            #[inline(always)]
            fn simd_ge(a: __m512i, b: __m512i) -> __m512i {{
                unsafe {{
                    // GE = LE with swapped args
                    let mask = _mm512_cmple_{cmp_suffix}_mask(b, a);
                    _mm512_maskz_set1_{set1_signed}(mask, {set1_neg1})
                }}
            }}

            #[inline(always)]
            fn blend(mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {{
                unsafe {{
                    let k = _mm512_cmpneq_{blend_suffix}_mask(mask, _mm512_setzero_si512());
                    _mm512_mask_blend_{blend_suffix}(k, if_false, if_true)
                }}
            }}

            #[inline(always)]
            fn reduce_add(a: __m512i) -> {elem} {{
                // No native integer reduce_add in AVX-512; use transmute to array
                let arr: {array} = unsafe {{ core::mem::transmute(a) }};
                arr.iter().copied().fold(0{elem}, {elem}::wrapping_add)
            }}

            #[inline(always)]
            fn not(a: __m512i) -> __m512i {{
                unsafe {{ _mm512_xor_si512(a, _mm512_set1_{set1_signed}({set1_neg1})) }}
            }}

            #[inline(always)]
            fn bitand(a: __m512i, b: __m512i) -> __m512i {{
                unsafe {{ _mm512_and_si512(a, b) }}
            }}

            #[inline(always)]
            fn bitor(a: __m512i, b: __m512i) -> __m512i {{
                unsafe {{ _mm512_or_si512(a, b) }}
            }}

            #[inline(always)]
            fn bitxor(a: __m512i, b: __m512i) -> __m512i {{
                unsafe {{ _mm512_xor_si512(a, b) }}
            }}

            {shift_impls}
            #[inline(always)]
            fn all_true(a: __m512i) -> bool {{
                unsafe {{
                    let mask = _mm512_cmpneq_{blend_suffix}_mask(a, _mm512_setzero_si512());
                    mask as u64 == {full_mask}
                }}
            }}

            #[inline(always)]
            fn any_true(a: __m512i) -> bool {{
                unsafe {{
                    let mask = _mm512_cmpneq_{blend_suffix}_mask(a, _mm512_setzero_si512());
                    mask as u64 != 0
                }}
            }}

            #[inline(always)]
            fn bitmask(a: __m512i) -> u64 {{
                unsafe {{
                    // Extract high bit of each lane: compare < 0 for signed interpretation
                    let zero = _mm512_setzero_si512();
                    _mm512_cmpneq_{blend_suffix}_mask(
                        _mm512_and_si512(a, _mm512_set1_{set1_signed}(1 << ({elem_bits} - 1) as {sign_type})),
                        zero
                    ) as u64
                }}
            }}
        }}
    "#,
        sign_type = match elem_bits {
            8 => "i8",
            16 => "i16",
            32 => "i32",
            64 => "i64",
            _ => unreachable!(),
        },
    }
}

/// Generate 8-bit shift polyfill via 16-bit operations for AVX-512.
fn generate_8bit_shift_polyfill(ty: &W512Type) -> String {
    let is_unsigned = ty.kind == W512Kind::UnsignedInt;

    let shr_arith = if is_unsigned {
        // Unsigned: arithmetic right shift = logical right shift
        // Inline the shift body to avoid Self:: ambiguity (multiple backend traits)
        formatdoc! {r#"
            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(a: __m512i) -> __m512i {{
                unsafe {{
                    let count = _mm_cvtsi32_si128(N);
                    let shifted = _mm512_srl_epi16(a, count);
                    let mask = _mm512_set1_epi8(((0xFFu16 >> N) & 0xFF) as i8);
                    _mm512_and_si512(shifted, mask)
                }}
            }}
        "#}
    } else {
        formatdoc! {r#"
            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(a: __m512i) -> __m512i {{
                unsafe {{
                    let count = _mm_cvtsi32_si128(N);
                    // Sign-extend bytes to 16-bit, shift, mask back to 8-bit
                    let lo = _mm512_sra_epi16(_mm512_slli_epi16::<8>(a), count);
                    let hi = _mm512_sra_epi16(a, count);
                    // Combine: take low byte from lo, high byte from hi
                    let mask = _mm512_set1_epi16(0x00FFu16 as i16);
                    _mm512_or_si512(
                        _mm512_and_si512(_mm512_srli_epi16::<8>(lo), mask),
                        _mm512_andnot_si512(mask, hi)
                    )
                }}
            }}
        "#}
    };

    formatdoc! {r#"
            #[inline(always)]
            fn shl_const<const N: i32>(a: __m512i) -> __m512i {{
                unsafe {{
                    let count = _mm_cvtsi32_si128(N);
                    let shifted = _mm512_sll_epi16(a, count);
                    let mask = _mm512_set1_epi8(((0xFFu16 << N) & 0xFF) as i8);
                    _mm512_and_si512(shifted, mask)
                }}
            }}

            {shr_arith}
            #[inline(always)]
            fn shr_logical_const<const N: i32>(a: __m512i) -> __m512i {{
                unsafe {{
                    let count = _mm_cvtsi32_si128(N);
                    let shifted = _mm512_srl_epi16(a, count);
                    let mask = _mm512_set1_epi8(((0xFFu16 >> N) & 0xFF) as i8);
                    _mm512_and_si512(shifted, mask)
                }}
            }}
    "#}
}

/// Generate V4 native AVX-512 W512 implementations for a specific token.
fn generate_x86_v4_w512_impls_for_token(types: &[W512Type], token: &str) -> String {
    let mut code = String::new();
    for ty in types {
        if ty.is_float() {
            code.push_str(&generate_x86_v4_float_impl_for_token(ty, token));
        } else {
            code.push_str(&generate_x86_v4_int_impl_for_token(ty, token));
        }
        code.push('\n');
    }
    code
}

/// Generate all V4 native AVX-512 W512 implementations (X64V4Token only).
pub(super) fn generate_x86_v4_w512_impls(types: &[W512Type]) -> String {
    generate_x86_v4_w512_impls_for_token(types, "X64V4Token")
}

/// Generate all Modern native AVX-512 W512 implementations (X64V4xToken).
pub(super) fn generate_x86_modern_w512_impls(types: &[W512Type]) -> String {
    generate_x86_v4_w512_impls_for_token(types, "X64V4xToken")
}

// ============================================================================
// Extension Traits for Modern Token
// ============================================================================

/// Generate popcnt extension backend trait for a W512 integer type.
fn generate_popcnt_backend_trait(ty: &W512Type) -> String {
    let trait_name = ty.trait_name();
    let name = ty.name();
    let elem = ty.elem;

    formatdoc! {r#"
        /// Population count (popcnt) extension for {name}.
        ///
        /// Returns a vector where each lane contains the number of set bits
        /// in the corresponding lane of the input.
        ///
        /// Requires AVX-512 VPOPCNTDQ (32/64-bit) or BITALG (8/16-bit).
        pub trait {name}PopcntBackend: {trait_name} {{
            /// Count set bits in each lane.
            fn popcnt(a: Self::Repr) -> Self::Repr;
        }}
    "#}
}

/// Generate popcnt extension backend traits for all integer W512 types.
pub(super) fn generate_popcnt_backend_traits(types: &[W512Type]) -> String {
    let mut code = formatdoc! {r#"
        //! Popcnt extension backend traits for W512 integer types.
        //!
        //! These traits extend the base integer backends with population count
        //! operations, available on AVX-512 Modern (VPOPCNTDQ + BITALG).
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.

        use super::*;

    "#};

    for ty in types {
        if !ty.is_float() {
            code.push_str(&generate_popcnt_backend_trait(ty));
            code.push('\n');
        }
    }
    code
}

/// Generate popcnt extension impl for X64V4xToken.
fn generate_popcnt_impl(ty: &W512Type) -> String {
    let name = ty.name();
    let epi = ty.x86_arith_suffix();

    formatdoc! {r#"
        #[cfg(target_arch = "x86_64")]
        impl {name}PopcntBackend for archmage::X64V4xToken {{
            #[inline(always)]
            fn popcnt(a: __m512i) -> __m512i {{
                unsafe {{ _mm512_popcnt_{epi}(a) }}
            }}
        }}
    "#}
}

/// Generate all popcnt extension impls for Modern token.
pub(super) fn generate_popcnt_impls(types: &[W512Type]) -> String {
    let mut code = String::new();
    for ty in types {
        if !ty.is_float() {
            code.push_str(&generate_popcnt_impl(ty));
            code.push('\n');
        }
    }
    code
}
