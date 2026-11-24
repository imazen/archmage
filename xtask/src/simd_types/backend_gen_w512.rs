//! Backend codegen for 512-bit SIMD types: f32x16, f64x8, i8x64, u8x64,
//! i16x32, u16x32, i32x16, u32x16, i64x8, u64x8.
//!
//! Uses a unified W512Type to generate all 10 types with backends for:
//! - x86 V4 (native AVX-512 intrinsics)
//! - Scalar (array ops)
//!
//! Width-polyfilled W512 backends (x86 V3 2×256-bit, NEON 4×128-bit,
//! WASM 4×128-bit) live hand-written in `magetypes/src/simd/impls/polyfill.rs`.

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

    /// Whether the uniform-variable-shift and saturating-arithmetic families
    /// are emitted for this type.
    ///
    /// 8- and 16-bit integers only: saturating add/sub does not exist above
    /// 16 bits on x86 or wasm at any tier, and the 128/256-bit halves these
    /// 512-bit types delegate to only carry the uniform shifts at those
    /// widths. See `docs/CROSS-ISA-INT-PRIMITIVES.md`.
    fn has_new_int_ops(&self) -> bool {
        !self.is_float() && (self.elem_bits == 8 || self.elem_bits == 16)
    }

    /// Whether this type carries the uniform-variable-shift family. Wider
    /// than `has_new_int_ops`: 32-bit uniform shifts are universal
    /// (`_mm512_sll_epi32` is AVX512F; the 128/256-bit sub-traits carry the
    /// clamped NEON/WASM/scalar forms), but 32-bit saturating add/sub does
    /// not exist on x86 or wasm at any tier, so the families gate separately.
    /// 64-bit stays out: x86 has no `sra_epi64` below AVX-512, so the
    /// arithmetic flavor is not universal at v3.
    fn has_uniform_shift_ops(&self) -> bool {
        !self.is_float() && matches!(self.elem_bits, 16 | 32)
    }

    /// The signed counterpart of this element type. Used by the sign-bit
    /// boolean contract, which is stated in terms of the sign bit regardless
    /// of the lane type's own signedness.
    fn signed_elem(&self) -> &'static str {
        match self.elem_bits {
            8 => "i8",
            16 => "i16",
            32 => "i32",
            64 => "i64",
            _ => unreachable!(),
        }
    }

    /// The unsigned counterpart of this element type.
    fn unsigned_elem(&self) -> &'static str {
        match self.elem_bits {
            8 => "u8",
            16 => "u16",
            32 => "u32",
            64 => "u64",
            _ => unreachable!(),
        }
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

/// `a[i]` reinterpreted as `via` for a wrapping operation, then back as
/// `elem`. Emits no casts when the two types match (`trivial_numeric_casts`).
fn via(elem: &str, via: &str, op: &str) -> String {
    if elem == via {
        format!("a[i].{op}")
    } else {
        format!("(a[i] as {via}).{op} as {elem}")
    }
}

/// Generate a backend trait for a W512 float type.
fn generate_float_backend_trait(ty: &W512Type) -> String {
    let trait_name = ty.trait_name();
    let elem = ty.elem;
    let lanes = ty.lanes;
    let array = ty.array_type();
    let internal_trait = super::float_layout::internal_trait(elem, lanes);
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

            /// Fast reciprocal approximation. Default returns the input
            /// unchanged; backends override with native intrinsics. The
            /// previous default `Self::div(Self::splat(1.0), a)` would
            /// require splat to be tokenless — incompatible with the
            /// soundness fix that gated splat on a token value.
            #[inline(always)]
            fn rcp_approx(self, a: Self::Repr) -> Self::Repr {{ a }}

            /// Fast reciprocal square root approximation — see [`rcp_approx`](Self::rcp_approx).
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

            /// Precise reciprocal — defaults to delegating to rcp_approx.
            /// Every shipped backend overrides with exact IEEE division.
            #[inline(always)]
            fn recip(self, a: Self::Repr) -> Self::Repr {{ Self::rcp_approx(self, a) }}

            /// Precise reciprocal square root — see [`recip`](Self::recip).
            #[inline(always)]
            fn rsqrt(self, a: Self::Repr) -> Self::Repr {{ Self::rsqrt_approx(self, a) }}
        }}
    "#,
        name = ty.name(),
    }
}

/// Generate a backend trait for a W512 integer type.
fn generate_int_backend_trait(ty: &W512Type) -> String {
    let integer_methods = super::backend_gen_widen_narrow::trait_methods(&ty.name());
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
            fn abs(self, a: Self::Repr) -> Self::Repr;
        "#}
    } else {
        String::new()
    };

    let new_ops_decls = w512_new_ops_trait_decls(ty);

    // mul only for 16-bit and 32-bit
    let mul_section = if ty.elem_bits == 16 || ty.elem_bits == 32 {
        formatdoc! {r#"

            /// Lane-wise multiplication (low bits of product).
            fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
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
            {mul_section}
            /// Lane-wise negation.
            fn neg(self, a: Self::Repr) -> Self::Repr;

            // ====== Math ======

            /// Lane-wise minimum.
            fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise maximum.
            fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
            {abs_section}
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
            /// `N` must be in `0..=lane_bits-1`.
            fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;

            /// Logical shift right by constant (zero-filling).
            /// `N` must be in `0..=lane_bits-1`.
            fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
        {new_ops_decls}
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

            /// Extract the high bit of each lane as a bitmask.
            fn bitmask(self, a: Self::Repr) -> u64;

            // ====== Default implementations ======

            /// Clamp values between lo and hi.
            #[inline(always)]
            fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {{
                <Self as {trait_name}>::min(self, <Self as {trait_name}>::max(self, a, lo), hi)
            }}
        {integer_methods}
        }}
    "#}
}

/// Trait declarations for the uniform-variable-shift and saturating-arithmetic
/// families on the 512-bit integer types (8/16-bit only).
fn w512_new_ops_trait_decls(ty: &W512Type) -> String {
    let mut code = String::new();
    if ty.has_uniform_shift_ops() {
        let elem_bits = ty.elem_bits;
        code.push_str(&formatdoc! {r#"
            // ====== Uniform variable shifts ======

            /// Shift left by a runtime `count` applied identically to every lane.
            ///
            /// `count >= {elem_bits}` produces all-zero lanes on every backend.
            fn shl_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;

            /// Logical (zero-filling) shift right by a runtime `count` applied
            /// identically to every lane.
            ///
            /// `count >= {elem_bits}` produces all-zero lanes on every backend.
            fn shr_logical_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;

            /// Arithmetic (sign-filling) shift right by a runtime `count`
            /// applied identically to every lane. Identical to
            /// `shr_logical_uniform` for unsigned element types.
            ///
            /// `count >= {elem_bits}` produces a sign fill on every backend.
            fn shr_arithmetic_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;

        "#});
    }
    if ty.has_new_int_ops() {
        code.push_str(&formatdoc! {r#"
            // ====== Saturating arithmetic ======

            /// Lane-wise addition that clamps to the element range instead of
            /// wrapping (`core`'s `saturating_add`, per lane).
            fn saturating_add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise subtraction that clamps to the element range instead
            /// of wrapping (`core`'s `saturating_sub`, per lane).
            fn saturating_sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

        "#});
    }
    code
}

/// Scalar bodies for the new families (also the differential-test reference).
fn w512_new_ops_scalar(ty: &W512Type) -> String {
    let mut code = String::new();
    if ty.has_uniform_shift_ops() {
        let array = ty.array_type();
        let elem = ty.elem;
        let uelem = ty.unsigned_elem();
        let eb = ty.elem_bits;
        let max_sh = eb - 1;

        let shr_arith = if ty.is_signed() {
            format!(
                "core::array::from_fn(|i| a[i].wrapping_shr(if count > {max_sh} {{ {max_sh} }} else {{ count }}))"
            )
        } else {
            format!(
                "core::array::from_fn(|i| if count >= {eb} {{ 0 }} else {{ a[i].wrapping_shr(count) }})"
            )
        };

        let wrapping_shl_via = via(elem, uelem, "wrapping_shl(count)");
        let wrapping_shr_via = via(elem, uelem, "wrapping_shr(count)");
        code.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn shl_uniform(self, a: {array}, count: u32) -> {array} {{
                core::array::from_fn(|i| if count >= {eb} {{ 0 }} else {{ {wrapping_shl_via} }})
            }}

            #[inline(always)]
            fn shr_logical_uniform(self, a: {array}, count: u32) -> {array} {{
                core::array::from_fn(|i| if count >= {eb} {{ 0 }} else {{ {wrapping_shr_via} }})
            }}

            #[inline(always)]
            fn shr_arithmetic_uniform(self, a: {array}, count: u32) -> {array} {{ {shr_arith} }}
        "#});
    }
    if ty.has_new_int_ops() {
        let array = ty.array_type();
        code.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn saturating_add(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i].saturating_add(b[i]))
            }}

            #[inline(always)]
            fn saturating_sub(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| a[i].saturating_sub(b[i]))
            }}
    "#});
    }
    code
}

/// Native AVX-512 bodies for the new families.
fn w512_new_ops_v4(ty: &W512Type) -> String {
    let arcane = super::backend_syntax::arcane("X64V4Token");

    let shifts = if ty.has_uniform_shift_ops() {
        // 16-bit: _mm512_{sll,srl,sra}_epi16 (avx512bw); 32-bit: the epi32
        // forms (avx512f). Hardware gives 0 / sign fill at count >= width,
        // and _mm_cvtsi32_si128 zero-extends, so no clamp is needed.
        let eb = ty.elem_bits;
        let arith = if ty.is_signed() { "sra" } else { "srl" };
        formatdoc! {r#"
            {arcane}
            fn shl_uniform(self, a: __m512i, count: u32) -> __m512i {{
                _mm512_sll_epi{eb}(a, _mm_cvtsi32_si128(count as i32))
            }}

            {arcane}
            fn shr_logical_uniform(self, a: __m512i, count: u32) -> __m512i {{
                _mm512_srl_epi{eb}(a, _mm_cvtsi32_si128(count as i32))
            }}

            {arcane}
            fn shr_arithmetic_uniform(self, a: __m512i, count: u32) -> __m512i {{
                _mm512_{arith}_epi{eb}(a, _mm_cvtsi32_si128(count as i32))
            }}
        "#}
    } else {
        String::new()
    };

    if ty.has_new_int_ops() {
        let sat_suffix = ty.x86_minmax_suffix(); // epi8/epu8/epi16/epu16
        formatdoc! {r#"
            {shifts}
            {arcane}
            fn saturating_add(self, a: __m512i, b: __m512i) -> __m512i {{
                _mm512_adds_{sat_suffix}(a, b)
            }}

            {arcane}
            fn saturating_sub(self, a: __m512i, b: __m512i) -> __m512i {{
                _mm512_subs_{sat_suffix}(a, b)
            }}
    "#}
    } else {
        shifts
    }
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
    let fma_fn = if ty.elem == "f32" {
        "fmaf_lanes"
    } else {
        "fma_lanes"
    };
    // Whole-array call, as in the W128/W256 generator; negation uses explicit
    // lane expressions rather than a closure.
    let fused_lanes = |negate: &str| {
        let addend = if negate.is_empty() {
            "c".to_string()
        } else {
            let lanes = (0..ty.lanes)
                .map(|i| format!("-c[{i}]"))
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{lanes}]")
        };
        format!("crate::soft_fma::{fma_fn}(a, b, {addend})")
    };
    let fused_add = fused_lanes("");
    let fused_sub = fused_lanes("-");
    let trait_name = ty.trait_name();
    let elem = ty.elem;
    let lanes = ty.lanes;
    let array = ty.array_type();
    let zero_lit = ty.zero_lit();
    let sqrt_fn = if elem == "f32" { "sqrtf" } else { "sqrt" };

    // The W512 float backend trait defaults rcp_approx/rsqrt_approx to the
    // identity, so every backend MUST override them. `rcp_approx` is exact
    // division (fastest); `rsqrt_approx` (f32) is the bit-hack seed + 2 Newton
    // steps (~17-bit, much cheaper than scalar sqrt+div). f64 is exact division
    // throughout. `recip`/`rsqrt` are always exact.
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
            fn recip(self, a: {array}) -> {array} {{ core::array::from_fn(|i| 1.0 / a[i]) }}
            #[inline(always)]
            fn rsqrt(self, a: {array}) -> {array} {{ core::array::from_fn(|i| 1.0 / crate::scalar_sqrt::{sqrt_fn}(a[i])) }}
        "#}
    } else {
        formatdoc! {r#"
            #[inline(always)]
            fn rcp_approx(self, a: {array}) -> {array} {{ core::array::from_fn(|i| 1.0 / a[i]) }}
            #[inline(always)]
            fn rsqrt_approx(self, a: {array}) -> {array} {{ core::array::from_fn(|i| 1.0 / crate::scalar_sqrt::{sqrt_fn}(a[i])) }}
            #[inline(always)]
            fn recip(self, a: {array}) -> {array} {{ core::array::from_fn(|i| 1.0 / a[i]) }}
            #[inline(always)]
            fn rsqrt(self, a: {array}) -> {array} {{ core::array::from_fn(|i| 1.0 / crate::scalar_sqrt::{sqrt_fn}(a[i])) }}
        "#}
    };

    formatdoc! {r#"
        impl {trait_name} for archmage::ScalarToken {{
            type Repr = {array};

            #[inline(always)]
            fn splat(self, v: {elem}) -> {array} {{ [{elem_name}; {lanes}].map(|_| v) }}

            #[inline(always)]
            fn zero(self) -> {array} {{ [{zero_lit}; {lanes}] }}

            #[inline(always)]
            fn load(self, data: &{array}) -> {array} {{ *data }}

            #[inline(always)]
            fn from_array(self, arr: {array}) -> {array} {{ arr }}

            #[inline(always)]
            fn store(self, repr: {array}, out: &mut {array}) {{ *out = repr; }}

            #[inline(always)]
            fn to_array(self, repr: {array}) -> {array} {{ repr }}

            #[inline(always)]
            fn add(self, a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| a[i] + b[i]) }}

            #[inline(always)]
            fn sub(self, a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| a[i] - b[i]) }}

            #[inline(always)]
            fn mul(self, a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| a[i] * b[i]) }}

            #[inline(always)]
            fn div(self, a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| a[i] / b[i]) }}

            #[inline(always)]
            fn neg(self, a: {array}) -> {array} {{ core::array::from_fn(|i| -a[i]) }}

            #[inline(always)]
            fn min(self, a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| a[i].min(b[i])) }}

            #[inline(always)]
            fn max(self, a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| a[i].max(b[i])) }}

            #[inline(always)]
            fn minimum(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| crate::scalar_extrema::{min_ieee}(a[i], b[i]))
            }}

            #[inline(always)]
            fn maximum(self, a: {array}, b: {array}) -> {array} {{
                core::array::from_fn(|i| crate::scalar_extrema::{max_ieee}(a[i], b[i]))
            }}

            #[inline(always)]
            fn min_fast(self, a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| if a[i] < b[i] {{ a[i] }} else {{ b[i] }}) }}

            #[inline(always)]
            fn max_fast(self, a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| if a[i] > b[i] {{ a[i] }} else {{ b[i] }}) }}

            #[inline(always)]
            fn sqrt(self, a: {array}) -> {array} {{
                core::array::from_fn(|i| crate::scalar_sqrt::{sqrt_fn}(a[i]))
            }}

{recip_section}
            #[inline(always)]
            fn abs(self, a: {array}) -> {array} {{
                core::array::from_fn(|i| {elem}::from_bits(a[i].to_bits() & {abs_mask}))
            }}

            #[inline(always)]
            fn floor(self, a: {array}) -> {array} {{
                a.map(crate::scalar::{floor_fn})
            }}

            #[inline(always)]
            fn ceil(self, a: {array}) -> {array} {{
                a.map(crate::scalar::{ceil_fn})
            }}

            #[inline(always)]
            fn round(self, a: {array}) -> {array} {{
                a.map(crate::scalar::{round_fn})
            }}

            #[inline(always)]
            fn mul_add(self, a: {array}, b: {array}, c: {array}) -> {array} {{ core::array::from_fn(|i| a[i] * b[i] + c[i]) }}

            #[inline(always)]
            fn mul_sub(self, a: {array}, b: {array}, c: {array}) -> {array} {{ core::array::from_fn(|i| a[i] * b[i] - c[i]) }}

            #[inline(always)]
            fn mul_add_portable(self, a: {array}, b: {array}, c: {array}) -> {array} {{ {fused_add} }}

            #[inline(always)]
            fn mul_sub_portable(self, a: {array}, b: {array}, c: {array}) -> {array} {{ {fused_sub} }}

            #[inline(always)]
            fn simd_eq(self, a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| if a[i] == b[i] {{ {elem}::from_bits(!0{uint}) }} else {{ {zero_lit} }}) }}

            #[inline(always)]
            fn simd_ne(self, a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| if a[i] != b[i] {{ {elem}::from_bits(!0{uint}) }} else {{ {zero_lit} }}) }}

            #[inline(always)]
            fn simd_lt(self, a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| if a[i] < b[i] {{ {elem}::from_bits(!0{uint}) }} else {{ {zero_lit} }}) }}

            #[inline(always)]
            fn simd_le(self, a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| if a[i] <= b[i] {{ {elem}::from_bits(!0{uint}) }} else {{ {zero_lit} }}) }}

            #[inline(always)]
            fn simd_gt(self, a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| if a[i] > b[i] {{ {elem}::from_bits(!0{uint}) }} else {{ {zero_lit} }}) }}

            #[inline(always)]
            fn simd_ge(self, a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| if a[i] >= b[i] {{ {elem}::from_bits(!0{uint}) }} else {{ {zero_lit} }}) }}

            #[inline(always)]
            fn blend(self, mask: {array}, if_true: {array}, if_false: {array}) -> {array} {{
                core::array::from_fn(|i| if mask[i].to_bits() != 0 {{ if_true[i] }} else {{ if_false[i] }})
            }}

            #[inline(always)]
            fn reduce_add(self, a: {array}) -> {elem} {{ a.iter().sum() }}

            #[inline(always)]
            fn reduce_min(self, a: {array}) -> {elem} {{ a.iter().copied().fold({elem}::INFINITY, {elem}::min) }}

            #[inline(always)]
            fn reduce_max(self, a: {array}) -> {elem} {{ a.iter().copied().fold({elem}::NEG_INFINITY, {elem}::max) }}

            #[inline(always)]
            fn not(self, a: {array}) -> {array} {{ core::array::from_fn(|i| {elem}::from_bits(!a[i].to_bits())) }}

            #[inline(always)]
            fn bitand(self, a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| {elem}::from_bits(a[i].to_bits() & b[i].to_bits())) }}

            #[inline(always)]
            fn bitor(self, a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| {elem}::from_bits(a[i].to_bits() | b[i].to_bits())) }}

            #[inline(always)]
            fn bitxor(self, a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| {elem}::from_bits(a[i].to_bits() ^ b[i].to_bits())) }}
        }}{scalar_internal}
    "#,
        scalar_internal = if elem == "f32" {
            super::float_layout::scalar_internal_impl(16)
        } else {
            String::new()
        },
        elem_name = zero_lit.to_string(),
        uint = if elem == "f32" { "u32" } else { "u64" },
        min_ieee = if elem == "f32" { "minimumf" } else { "minimum" },
        max_ieee = if elem == "f32" { "maximumf" } else { "maximum" },
        floor_fn = if elem == "f32" { "floor_f32" } else { "floor_f64" },
        ceil_fn = if elem == "f32" { "ceil_f32" } else { "ceil_f64" },
        round_fn = if elem == "f32" { "round_f32" } else { "round_f64" },
        abs_mask = if elem == "f32" { "0x7FFF_FFFF" } else { "0x7FFF_FFFF_FFFF_FFFF" },
    }
}

/// Generate scalar backend implementation for a W512 integer type.
fn generate_scalar_int_impl(ty: &W512Type) -> String {
    let signed_elem = ty.signed_elem();
    let integer_methods = super::backend_gen_widen_narrow::methods(&ty.name(), "ScalarToken");
    let trait_name = ty.trait_name();
    let elem = ty.elem;
    let lanes = ty.lanes;
    let array = ty.array_type();
    let zero_lit = ty.zero_lit();

    let mul_impl = if ty.elem_bits == 16 || ty.elem_bits == 32 {
        formatdoc! {r#"

            #[inline(always)]
            fn mul(self, a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| a[i].wrapping_mul(b[i])) }}
        "#}
    } else {
        String::new()
    };

    let abs_impl = if ty.is_signed() {
        formatdoc! {r#"

            #[inline(always)]
            fn abs(self, a: {array}) -> {array} {{ core::array::from_fn(|i| a[i].wrapping_abs()) }}
        "#}
    } else {
        String::new()
    };

    let neg_impl = if ty.is_signed() {
        "core::array::from_fn(|i| a[i].wrapping_neg())".to_string()
    } else {
        // For unsigned: negate via wrapping
        format!("core::array::from_fn(|i| (0{elem}).wrapping_sub(a[i]))")
    };

    // For shifts of byte types, N could exceed bit width, so use wrapping
    let shl_body = match ty.elem_bits {
        8 => format!(
            "core::array::from_fn(|i| if N < 8 {{ {} }} else {{ 0 }})",
            via(elem, "u8", "wrapping_shl(N as u32)")
        ),
        16 => format!(
            "core::array::from_fn(|i| if N < 16 {{ {} }} else {{ 0 }})",
            via(elem, "u16", "wrapping_shl(N as u32)")
        ),
        32 => format!(
            "core::array::from_fn(|i| if N < 32 {{ {} }} else {{ 0 }})",
            via(elem, "u32", "wrapping_shl(N as u32)")
        ),
        64 => format!(
            "core::array::from_fn(|i| if N < 64 {{ {} }} else {{ 0 }})",
            via(elem, "u64", "wrapping_shl(N as u32)")
        ),
        _ => unreachable!(),
    };

    let shr_arith_body = if ty.is_signed() {
        let bits = ty.elem_bits;
        format!(
            "core::array::from_fn(|i| if N < {bits} {{ a[i].wrapping_shr(N as u32) }} else if a[i] < 0 {{ -1 }} else {{ 0 }})"
        )
    } else {
        match ty.elem_bits {
            8 => "core::array::from_fn(|i| if N < 8 { a[i].wrapping_shr(N as u32) } else { 0 })"
                .to_string(),
            16 => "core::array::from_fn(|i| if N < 16 { a[i].wrapping_shr(N as u32) } else { 0 })"
                .to_string(),
            32 => "core::array::from_fn(|i| if N < 32 { a[i].wrapping_shr(N as u32) } else { 0 })"
                .to_string(),
            64 => "core::array::from_fn(|i| if N < 64 { a[i].wrapping_shr(N as u32) } else { 0 })"
                .to_string(),
            _ => unreachable!(),
        }
    };

    let shr_logical_body = match ty.elem_bits {
        8 => format!(
            "core::array::from_fn(|i| if N < 8 {{ {} }} else {{ 0 }})",
            via(elem, "u8", "wrapping_shr(N as u32)")
        ),
        16 => format!(
            "core::array::from_fn(|i| if N < 16 {{ {} }} else {{ 0 }})",
            via(elem, "u16", "wrapping_shr(N as u32)")
        ),
        32 => format!(
            "core::array::from_fn(|i| if N < 32 {{ {} }} else {{ 0 }})",
            via(elem, "u32", "wrapping_shr(N as u32)")
        ),
        64 => format!(
            "core::array::from_fn(|i| if N < 64 {{ {} }} else {{ 0 }})",
            via(elem, "u64", "wrapping_shr(N as u32)")
        ),
        _ => unreachable!(),
    };

    let new_ops = w512_new_ops_scalar(ty);

    // reduce_add with wrapping for integers
    let reduce_add = format!("a.iter().copied().fold(0{elem}, {elem}::wrapping_add)");

    // Bitmask: extract high bit of each element lane
    let bitmask_body = format!(
        "let mut mask = 0u64; for i in 0..{lanes} {{ if {negative} {{ mask |= 1u64 << i; }} }} mask",
        // Signed lanes test the sign directly; unsigned lanes reinterpret as
        // signed for the sign-bit check.
        negative = if ty.is_signed() {
            "a[i] < 0".to_string()
        } else {
            format!("(a[i] as {signed_elem}) < 0")
        },
    );
    let lane_negative = if ty.is_signed() {
        "v < 0".to_string()
    } else {
        format!("(v as {signed_elem}) < 0")
    };

    formatdoc! {r#"
        impl {trait_name} for archmage::ScalarToken {{
            type Repr = {array};

            #[inline(always)]
            fn splat(self, v: {elem}) -> {array} {{ [v; {lanes}] }}

            #[inline(always)]
            fn zero(self) -> {array} {{ [{zero_lit}; {lanes}] }}

            #[inline(always)]
            fn load(self, data: &{array}) -> {array} {{ *data }}

            #[inline(always)]
            fn from_array(self, arr: {array}) -> {array} {{ arr }}

            #[inline(always)]
            fn store(self, repr: {array}, out: &mut {array}) {{ *out = repr; }}

            #[inline(always)]
            fn to_array(self, repr: {array}) -> {array} {{ repr }}

            #[inline(always)]
            fn add(self, a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| a[i].wrapping_add(b[i])) }}

            #[inline(always)]
            fn sub(self, a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| a[i].wrapping_sub(b[i])) }}
            {mul_impl}
            #[inline(always)]
            fn neg(self, a: {array}) -> {array} {{ {neg_impl} }}

            #[inline(always)]
            fn min(self, a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| if a[i] < b[i] {{ a[i] }} else {{ b[i] }}) }}

            #[inline(always)]
            fn max(self, a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| if a[i] > b[i] {{ a[i] }} else {{ b[i] }}) }}
            {abs_impl}
            #[inline(always)]
            fn simd_eq(self, a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| if a[i] == b[i] {{ !{zero_for_not} }} else {{ {zero_lit} }}) }}

            #[inline(always)]
            fn simd_ne(self, a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| if a[i] != b[i] {{ !{zero_for_not} }} else {{ {zero_lit} }}) }}

            #[inline(always)]
            fn simd_lt(self, a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| if a[i] < b[i] {{ !{zero_for_not} }} else {{ {zero_lit} }}) }}

            #[inline(always)]
            fn simd_le(self, a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| if a[i] <= b[i] {{ !{zero_for_not} }} else {{ {zero_lit} }}) }}

            #[inline(always)]
            fn simd_gt(self, a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| if a[i] > b[i] {{ !{zero_for_not} }} else {{ {zero_lit} }}) }}

            #[inline(always)]
            fn simd_ge(self, a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| if a[i] >= b[i] {{ !{zero_for_not} }} else {{ {zero_lit} }}) }}

            #[inline(always)]
            fn blend(self, mask: {array}, if_true: {array}, if_false: {array}) -> {array} {{
                core::array::from_fn(|i| if mask[i] != {zero_lit} {{ if_true[i] }} else {{ if_false[i] }})
            }}

            #[inline(always)]
            fn reduce_add(self, a: {array}) -> {elem} {{ {reduce_add} }}

            #[inline(always)]
            fn not(self, a: {array}) -> {array} {{ core::array::from_fn(|i| !a[i]) }}

            #[inline(always)]
            fn bitand(self, a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| a[i] & b[i]) }}

            #[inline(always)]
            fn bitor(self, a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| a[i] | b[i]) }}

            #[inline(always)]
            fn bitxor(self, a: {array}, b: {array}) -> {array} {{ core::array::from_fn(|i| a[i] ^ b[i]) }}

            #[inline(always)]
            fn shl_const<const N: i32>(self, a: {array}) -> {array} {{ {shl_body} }}

            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(self, a: {array}) -> {array} {{ {shr_arith_body} }}

            #[inline(always)]
            fn shr_logical_const<const N: i32>(self, a: {array}) -> {array} {{ {shr_logical_body} }}

        {new_ops}
            #[inline(always)]
            fn all_true(self, a: {array}) -> bool {{ a.iter().all(|&v| {lane_negative}) }}

            #[inline(always)]
            fn any_true(self, a: {array}) -> bool {{ a.iter().any(|&v| {lane_negative}) }}

            #[inline(always)]
            fn bitmask(self, a: {array}) -> u64 {{ {bitmask_body} }}
        {integer_methods}
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
// x86 V4 Native AVX-512 Implementation
// Uses __m512/__m512d/__m512i directly with AVX-512 intrinsics.
// ============================================================================

/// Generate V4 native implementation for a W512 float type (f32x16, f64x8).
fn generate_x86_v4_float_impl(ty: &W512Type) -> String {
    let arcane = super::backend_syntax::arcane("X64V4Token");
    let trait_name = ty.trait_name();
    let inner = ty.x86_v4_inner();
    let s = ty.x86_float_suffix();
    let elem = ty.elem;
    let float_layout = if elem == "f32" {
        super::float_layout::x86(16, inner, &arcane)
    } else {
        String::new()
    };
    let array = ty.array_type();
    let epi = if elem == "f32" { "epi32" } else { "epi64" };
    let abs_mask = if elem == "f32" {
        "0x7FFF_FFFFu32 as i32"
    } else {
        "0x7FFF_FFFF_FFFF_FFFFu64 as i64"
    };
    // Working-tier recip/rsqrt, chosen per element type. f32: rcp14/rsqrt14
    // (14-bit) + one Newton step ≈ 24 bits — inside the <= 4 ULP bare-name
    // contract and the published 0.9.28 lowering (rails per-backend: the
    // refine step computes a*r, so ±0/±inf lanes yield NaN — the 0 ULP +
    // IEEE-rails tier is the generic `recip_portable`, issue #64). f64:
    // exact division — one step from 14 bits is ~28 bits, far short of the
    // contract, and the two-step form was never measured against
    // _mm512_div_pd.
    let numeric_min = if elem == "f32" {
        "_mm512_range_ps::<0x04>(a, b)".to_string()
    } else {
        format!(
            "_mm512_mask_blend_{s}(_mm512_cmp_{s}_mask::<_CMP_UNORD_Q>(b, b), _mm512_min_{s}(a, b), a)"
        )
    };
    let numeric_max = if elem == "f32" {
        "_mm512_range_ps::<0x05>(a, b)".to_string()
    } else {
        format!(
            "_mm512_mask_blend_{s}(_mm512_cmp_{s}_mask::<_CMP_UNORD_Q>(b, b), _mm512_max_{s}(a, b), a)"
        )
    };
    let working_recip_512 = if elem == "f32" {
        // rcp14/rsqrt14 + one FMA-Newton step + VFIXUPIMM rail patch. The
        // fixup tables classify the ORIGINAL input and overwrite exactly the
        // rail lanes with the IEEE values (recip: ±0 -> signed inf,
        // +inf -> +0, -inf -> -0; rsqrt: ±0 -> signed inf, +inf -> +0,
        // -inf and negatives -> NaN; NaN propagates quietened) — this is the
        // instruction's designed use, one op vs the 2-op cmp+blend rescue the
        // 128/256-bit v3 impls need (VFIXUPIMM needs AVX-512). Tables
        // probe-verified on Zen 5 silicon 2026-09-04; measured rails premium
        // +11.7% over plain Newton vs +181% for exact division. Deep
        // subnormals unspecified at this tier (rcp14/rsqrt14 DO handle most
        // denormals, better than legacy rcpps).
        formatdoc! {r#"
            {arcane}
            fn recip(self, a: {inner}) -> {inner} {{
                // token nibbles: QNAN->2 SNAN->2 ZERO->6 ONE->0 -INF->7 +INF->8 NEG->0 POS->0
                const RECIP_FIXUP_TBL: i32 = 0x0087_0622u32 as i32;
                {{
                    let r = _mm512_rcp14_{s}(a);
                    let e = _mm512_fnmadd_{s}(a, r, _mm512_set1_{s}(1.0));
                    let refined = _mm512_fmadd_{s}(r, e, r);
                    _mm512_fixupimm_{s}::<0>(refined, a, _mm512_set1_epi32(RECIP_FIXUP_TBL))
                }}
            }}

            {arcane}
            fn rsqrt(self, a: {inner}) -> {inner} {{
                // token nibbles: QNAN->2 SNAN->2 ZERO->6 ONE->0 -INF->3 +INF->8 NEG->3 POS->0
                const RSQRT_FIXUP_TBL: i32 = 0x0383_0622u32 as i32;
                {{
                    let y = _mm512_rsqrt14_{s}(a);
                    let t = _mm512_mul_{s}(a, _mm512_mul_{s}(y, y));
                    let refined =
                        _mm512_mul_{s}(y, _mm512_fnmadd_{s}(_mm512_set1_{s}(0.5), t, _mm512_set1_{s}(1.5)));
                    _mm512_fixupimm_{s}::<0>(refined, a, _mm512_set1_epi32(RSQRT_FIXUP_TBL))
                }}
            }}
        "#}
    } else {
        formatdoc! {r#"
            {arcane}
            fn recip(self, a: {inner}) -> {inner} {{
                _mm512_div_{s}(_mm512_set1_{s}(1.0), a)
            }}

            {arcane}
            fn rsqrt(self, a: {inner}) -> {inner} {{
                _mm512_div_{s}(_mm512_set1_{s}(1.0), _mm512_sqrt_{s}(a))
            }}
        "#}
    };
    formatdoc! {r#"
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        impl {trait_name} for archmage::X64V4Token {{
            type Repr = {inner};

            {arcane}
            fn splat(self, v: {elem}) -> {inner} {{
                _mm512_set1_{s}(v)
            }}

            {arcane}
            fn zero(self) -> {inner} {{
                _mm512_setzero_{s}()
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

            {arcane}
            fn add(self, a: {inner}, b: {inner}) -> {inner} {{
                _mm512_add_{s}(a, b)
            }}

            {arcane}
            fn sub(self, a: {inner}, b: {inner}) -> {inner} {{
                _mm512_sub_{s}(a, b)
            }}

            {arcane}
            fn mul(self, a: {inner}, b: {inner}) -> {inner} {{
                _mm512_mul_{s}(a, b)
            }}

            {arcane}
            fn div(self, a: {inner}, b: {inner}) -> {inner} {{
                _mm512_div_{s}(a, b)
            }}

            {arcane}
            fn neg(self, a: {inner}) -> {inner} {{
                _mm512_xor_{s}(a, _mm512_set1_{s}(-0.0))
            }}

            {arcane}
            fn min(self, a: {inner}, b: {inner}) -> {inner} {{
                {numeric_min}
            }}

            {arcane}
            fn max(self, a: {inner}, b: {inner}) -> {inner} {{
                {numeric_max}
            }}

            {arcane}
            fn minimum(self, a: {inner}, b: {inner}) -> {inner} {{
                _mm512_mask_blend_{s}(
                    _mm512_cmp_{s}_mask::<_CMP_UNORD_Q>(a, b),
                    _mm512_or_{s}(_mm512_min_{s}(a, b), _mm512_min_{s}(b, a)),
                    _mm512_set1_{s}({elem}::NAN),
                )
            }}

            {arcane}
            fn maximum(self, a: {inner}, b: {inner}) -> {inner} {{
                _mm512_mask_blend_{s}(
                    _mm512_cmp_{s}_mask::<_CMP_UNORD_Q>(a, b),
                    _mm512_and_{s}(_mm512_max_{s}(a, b), _mm512_max_{s}(b, a)),
                    _mm512_set1_{s}({elem}::NAN),
                )
            }}

            {arcane}
            fn min_fast(self, a: {inner}, b: {inner}) -> {inner} {{
                _mm512_min_{s}(a, b)
            }}

            {arcane}
            fn max_fast(self, a: {inner}, b: {inner}) -> {inner} {{
                _mm512_max_{s}(a, b)
            }}

            {arcane}
            fn sqrt(self, a: {inner}) -> {inner} {{
                _mm512_sqrt_{s}(a)
            }}

            {arcane}
            fn abs(self, a: {inner}) -> {inner} {{
                let mask = _mm512_castsi512_{s}(_mm512_set1_{epi}({abs_mask}));
                _mm512_and_{s}(a, mask)
            }}

            {arcane}
            fn trunc(self, a: {inner}) -> {inner} {{ _mm512_roundscale_{s}::<0x03>(a) }}

            {arcane}
            fn mul_neg_add_relaxed(self, a: {inner}, b: {inner}, c: {inner}) -> {inner} {{ _mm512_fnmadd_{s}(a, b, c) }}

            {arcane}
            fn mul_neg_add_fused(self, a: {inner}, b: {inner}, c: {inner}) -> {inner} {{ _mm512_fnmadd_{s}(a, b, c) }}

            {arcane}
            fn floor(self, a: {inner}) -> {inner} {{
                _mm512_roundscale_{s}::<0x01>(a)
            }}

            {arcane}
            fn ceil(self, a: {inner}) -> {inner} {{
                _mm512_roundscale_{s}::<0x02>(a)
            }}

            {arcane}
            fn round(self, a: {inner}) -> {inner} {{
                _mm512_roundscale_{s}::<0x00>(a)
            }}

            {arcane}
            fn mul_add(self, a: {inner}, b: {inner}, c: {inner}) -> {inner} {{
                _mm512_fmadd_{s}(a, b, c)
            }}

            {arcane}
            fn mul_sub(self, a: {inner}, b: {inner}, c: {inner}) -> {inner} {{
                _mm512_fmsub_{s}(a, b, c)
            }}

            {arcane}
            fn mul_add_portable(self, a: {inner}, b: {inner}, c: {inner}) -> {inner} {{
                _mm512_fmadd_{s}(a, b, c)
            }}

            {arcane}
            fn mul_sub_portable(self, a: {inner}, b: {inner}, c: {inner}) -> {inner} {{
                _mm512_fmsub_{s}(a, b, c)
            }}

            {arcane}
            fn simd_eq(self, a: {inner}, b: {inner}) -> {inner} {{
                let mask = _mm512_cmp_{s}_mask::<_CMP_EQ_OQ>(a, b);
                _mm512_castsi512_{s}(_mm512_maskz_set1_{epi}(mask, -1))
            }}

            {arcane}
            fn simd_ne(self, a: {inner}, b: {inner}) -> {inner} {{
                let mask = _mm512_cmp_{s}_mask::<_CMP_NEQ_UQ>(a, b);
                _mm512_castsi512_{s}(_mm512_maskz_set1_{epi}(mask, -1))
            }}

            {arcane}
            fn simd_lt(self, a: {inner}, b: {inner}) -> {inner} {{
                let mask = _mm512_cmp_{s}_mask::<_CMP_LT_OQ>(a, b);
                _mm512_castsi512_{s}(_mm512_maskz_set1_{epi}(mask, -1))
            }}

            {arcane}
            fn simd_le(self, a: {inner}, b: {inner}) -> {inner} {{
                let mask = _mm512_cmp_{s}_mask::<_CMP_LE_OQ>(a, b);
                _mm512_castsi512_{s}(_mm512_maskz_set1_{epi}(mask, -1))
            }}

            {arcane}
            fn simd_gt(self, a: {inner}, b: {inner}) -> {inner} {{
                let mask = _mm512_cmp_{s}_mask::<_CMP_GT_OQ>(a, b);
                _mm512_castsi512_{s}(_mm512_maskz_set1_{epi}(mask, -1))
            }}

            {arcane}
            fn simd_ge(self, a: {inner}, b: {inner}) -> {inner} {{
                let mask = _mm512_cmp_{s}_mask::<_CMP_GE_OQ>(a, b);
                _mm512_castsi512_{s}(_mm512_maskz_set1_{epi}(mask, -1))
            }}

            {arcane}
            fn blend(self, mask: {inner}, if_true: {inner}, if_false: {inner}) -> {inner} {{
                let mask_i = _mm512_cast{s}_si512(mask);
                let k = _mm512_cmpneq_{epi}_mask(mask_i, _mm512_setzero_si512());
                _mm512_mask_blend_{s}(k, if_false, if_true)
            }}

            {arcane}
            fn reduce_add(self, a: {inner}) -> {elem} {{
                _mm512_reduce_add_{s}(a)
            }}

            {arcane}
            fn reduce_min(self, a: {inner}) -> {elem} {{
                _mm512_reduce_min_{s}(a)
            }}

            {arcane}
            fn reduce_max(self, a: {inner}) -> {elem} {{
                _mm512_reduce_max_{s}(a)
            }}

            // Raw 14-bit hardware estimate — the cheapest >=12-bit path on AVX-512.
            {arcane}
            fn rcp_approx(self, a: {inner}) -> {inner} {{
                _mm512_rcp14_{s}(a)
            }}

            {arcane}
            fn rsqrt_approx(self, a: {inner}) -> {inner} {{
                _mm512_rsqrt14_{s}(a)
            }}

            {working_recip_512}

            {arcane}
            fn not(self, a: {inner}) -> {inner} {{
                let all_ones = _mm512_castsi512_{s}(_mm512_set1_{epi}(-1));
                _mm512_xor_{s}(a, all_ones)
            }}

            {arcane}
            fn bitand(self, a: {inner}, b: {inner}) -> {inner} {{
                _mm512_and_{s}(a, b)
            }}

            {arcane}
            fn bitor(self, a: {inner}, b: {inner}) -> {inner} {{
                _mm512_or_{s}(a, b)
            }}

            {arcane}
            fn bitxor(self, a: {inner}, b: {inner}) -> {inner} {{
                _mm512_xor_{s}(a, b)
            }}
        }}{internal_impl}
    "#,
        internal_impl = if elem == "f32" {
            formatdoc! {r#"

                #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
                impl crate::simd::backends::F32x16Internal for archmage::X64V4Token {{
                    type InternalRepr = {inner};
                {float_layout}
                }}"#}
        } else {
            String::new()
        },
    }
}

/// Generate V4 native implementation for a W512 integer type.
fn generate_x86_v4_int_impl(ty: &W512Type) -> String {
    let integer_methods = super::backend_gen_widen_narrow::methods(&ty.name(), "X64V4Token");
    let arcane = super::backend_syntax::arcane("X64V4Token");
    let trait_name = ty.trait_name();
    let elem = ty.elem;
    let lanes = ty.lanes;
    let elem_bits = ty.elem_bits;
    let array = ty.array_type();
    let epi = ty.x86_arith_suffix();
    let mm = ty.x86_minmax_suffix();
    let set1 = ty.x86_set1_suffix();
    // `_mm512_set1_epi*` take the signed lane type; only unsigned lanes cast.
    let splat_arg = if elem.starts_with('i') { "v" } else { "v as _" };

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

            {arcane}
            fn mul(self, a: __m512i, b: __m512i) -> __m512i {{
                _mm512_mullo_{epi}(a, b)
            }}
        "#}
    } else {
        String::new()
    };

    // abs (signed only)
    let abs_impl = if ty.is_signed() {
        formatdoc! {r#"

            {arcane}
            fn abs(self, a: __m512i) -> __m512i {{
                _mm512_abs_{epi}(a)
            }}
        "#}
    } else {
        String::new()
    };

    // neg: sub(zero, a) works for both signed and unsigned
    let neg_impl = formatdoc! {r#"
            {arcane}
            fn neg(self, a: __m512i) -> __m512i {{
                _mm512_sub_{epi}(_mm512_setzero_si512(), a)
            }}
    "#};

    let new_ops = w512_new_ops_v4(ty);

    // Shifts: 8-bit shifts need polyfill via 16-bit, others use native
    let shift_impls = if elem_bits == 8 {
        generate_8bit_shift_polyfill(ty)
    } else {
        let shr_arith = if ty.kind == W512Kind::UnsignedInt {
            // Unsigned: arithmetic right shift = logical right shift
            // Inline the shift body to avoid Self:: ambiguity (multiple backend traits)
            formatdoc! {r#"
            {arcane}
            fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {{
                _mm512_srl_{epi}(a, _mm_cvtsi32_si128(N))
            }}
            "#}
        } else {
            formatdoc! {r#"
            {arcane}
            fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {{
                _mm512_sra_{epi}(a, _mm_cvtsi32_si128(N))
            }}
            "#}
        };

        let shl = if elem_bits == 32 {
            let cases = (0..32)
                .map(|shift| format!("    {shift} => _mm512_slli_epi32::<{shift}>(a),"))
                .collect::<Vec<_>>()
                .join("\n");
            formatdoc! {r#"
                match N {{
                {cases}
                    _ => unreachable!("shift count checked by the public API"),
                }}
            "#}
        } else {
            format!("_mm512_sll_{epi}(a, _mm_cvtsi32_si128(N))")
        };
        formatdoc! {r#"
            {arcane}
            fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {{
                {shl}
            }}

            {shr_arith}
            {arcane}
            fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {{
                _mm512_srl_{epi}(a, _mm_cvtsi32_si128(N))
            }}
        "#}
    };

    formatdoc! {r#"
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        impl {trait_name} for archmage::X64V4Token {{
            type Repr = __m512i;

            {arcane}
            fn splat(self, v: {elem}) -> __m512i {{
                _mm512_set1_{set1}({splat_arg})
            }}

            {arcane}
            fn zero(self) -> __m512i {{
                _mm512_setzero_si512()
            }}

            #[inline(always)]
            fn load(self, data: &{array}) -> __m512i {{
                crate::simd_storage::copy(data)
            }}

            #[inline(always)]
            fn from_array(self, arr: {array}) -> __m512i {{
                crate::simd_storage::cast(arr)
            }}

            #[inline(always)]
            fn store(self, repr: __m512i, out: &mut {array}) {{
                crate::simd_storage::store(repr, out);
            }}

            #[inline(always)]
            fn to_array(self, repr: __m512i) -> {array} {{
                crate::simd_storage::cast(repr)
            }}

            {arcane}
            fn add(self, a: __m512i, b: __m512i) -> __m512i {{
                _mm512_add_{epi}(a, b)
            }}

            {arcane}
            fn sub(self, a: __m512i, b: __m512i) -> __m512i {{
                _mm512_sub_{epi}(a, b)
            }}
            {mul_impl}
            {neg_impl}
            {arcane}
            fn min(self, a: __m512i, b: __m512i) -> __m512i {{
                _mm512_min_{mm}(a, b)
            }}

            {arcane}
            fn max(self, a: __m512i, b: __m512i) -> __m512i {{
                _mm512_max_{mm}(a, b)
            }}
            {abs_impl}
            {arcane}
            fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {{
                let mask = _mm512_cmpeq_{cmp_suffix}_mask(a, b);
                _mm512_maskz_set1_{set1_signed}(mask, {set1_neg1})
            }}

            {arcane}
            fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {{
                let mask = _mm512_cmpneq_{cmp_suffix}_mask(a, b);
                _mm512_maskz_set1_{set1_signed}(mask, {set1_neg1})
            }}

            {arcane}
            fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {{
                let mask = _mm512_cmplt_{cmp_suffix}_mask(a, b);
                _mm512_maskz_set1_{set1_signed}(mask, {set1_neg1})
            }}

            {arcane}
            fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {{
                let mask = _mm512_cmple_{cmp_suffix}_mask(a, b);
                _mm512_maskz_set1_{set1_signed}(mask, {set1_neg1})
            }}

            {arcane}
            fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {{
                // GT = LT with swapped args
                let mask = _mm512_cmplt_{cmp_suffix}_mask(b, a);
                _mm512_maskz_set1_{set1_signed}(mask, {set1_neg1})
            }}

            {arcane}
            fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {{
                // GE = LE with swapped args
                let mask = _mm512_cmple_{cmp_suffix}_mask(b, a);
                _mm512_maskz_set1_{set1_signed}(mask, {set1_neg1})
            }}

            {arcane}
            fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {{
                let k = _mm512_cmpneq_{blend_suffix}_mask(mask, _mm512_setzero_si512());
                _mm512_mask_blend_{blend_suffix}(k, if_false, if_true)
            }}

            {arcane}
            fn reduce_add(self, a: __m512i) -> {elem} {{
                // AVX-512 has no single-instruction integer reduction; fold the lanes
                let arr: {array} = crate::simd_storage::cast(a);
                arr.iter().copied().fold(0{elem}, {elem}::wrapping_add)
            }}

            {arcane}
            fn not(self, a: __m512i) -> __m512i {{
                _mm512_xor_si512(a, _mm512_set1_{set1_signed}({set1_neg1}))
            }}

            {arcane}
            fn bitand(self, a: __m512i, b: __m512i) -> __m512i {{
                _mm512_and_si512(a, b)
            }}

            {arcane}
            fn bitor(self, a: __m512i, b: __m512i) -> __m512i {{
                _mm512_or_si512(a, b)
            }}

            {arcane}
            fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {{
                _mm512_xor_si512(a, b)
            }}

            {shift_impls}
        {new_ops}
            {arcane}
            fn all_true(self, a: __m512i) -> bool {{
                // Sign-bit contract: `movepi*_mask` reads the sign bits
                // directly. The `cmpneq` form this replaces answered a
                // different question (lane nonzero) and disagreed with the
                // V3 movemask backends.
                _mm512_movepi{elem_bits}_mask(a){mask_as_u64} == {full_mask}
            }}

            {arcane}
            fn any_true(self, a: __m512i) -> bool {{
                _mm512_movepi{elem_bits}_mask(a){mask_as_u64} != 0
            }}

            {arcane}
            fn bitmask(self, a: __m512i) -> u64 {{
                _mm512_movepi{elem_bits}_mask(a){mask_as_u64}
            }}
        {integer_methods}
        }}
    "#,
        // 8-bit lanes have a 64-bit mask already; narrower masks widen.
        mask_as_u64 = if elem_bits == 8 { "" } else { " as u64" },
    }
}

/// Generate 8-bit shift polyfill via 16-bit operations for AVX-512.
fn generate_8bit_shift_polyfill(ty: &W512Type) -> String {
    let arcane = super::backend_syntax::arcane("X64V4Token");
    let is_unsigned = ty.kind == W512Kind::UnsignedInt;

    let shr_arith = if is_unsigned {
        // Unsigned: arithmetic right shift = logical right shift
        // Inline the shift body to avoid Self:: ambiguity (multiple backend traits)
        formatdoc! {r#"
            {arcane}
            fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {{
                let count = _mm_cvtsi32_si128(N);
                let shifted = _mm512_srl_epi16(a, count);
                let byte_mask = _mm512_set1_epi8(((0xFFu16 >> N) & 0xFF) as i8);
                let logical = _mm512_and_si512(shifted, byte_mask);
                // Sign-extend: fill the high N bits of negative lanes. The u16
                // shift keeps the fill mask 0x00 at N == 0 (identity shift).
                let sign = _mm512_movm_epi8(_mm512_cmplt_epi8_mask(a, _mm512_setzero_si512()));
                let fill = _mm512_set1_epi8(((0xFF00u16 >> N) & 0xFF) as i8);
                _mm512_or_si512(logical, _mm512_and_si512(sign, fill))
            }}
        "#}
    } else {
        formatdoc! {r#"
            {arcane}
            fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {{
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
        "#}
    };

    formatdoc! {r#"
            {arcane}
            fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {{
                let count = _mm_cvtsi32_si128(N);
                let shifted = _mm512_sll_epi16(a, count);
                let mask = _mm512_set1_epi8(((0xFFu16 << N) & 0xFF) as i8);
                _mm512_and_si512(shifted, mask)
            }}

            {shr_arith}
            {arcane}
            fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {{
                let count = _mm_cvtsi32_si128(N);
                let shifted = _mm512_srl_epi16(a, count);
                let mask = _mm512_set1_epi8(((0xFFu16 >> N) & 0xFF) as i8);
                _mm512_and_si512(shifted, mask)
            }}
    "#}
}

/// Generate all V4 native AVX-512 W512 implementations (`X64V4Token`).
pub(super) fn generate_x86_v4_w512_impls(types: &[W512Type]) -> String {
    let mut code = String::new();
    for ty in types {
        if ty.is_float() {
            code.push_str(&generate_x86_v4_float_impl(ty));
        } else {
            code.push_str(&generate_x86_v4_int_impl(ty));
        }
        code.push('\n');
    }
    code
}

// ============================================================================
// Extension Traits for Modern Token
// ============================================================================

/// Generate popcnt extension backend trait for a W512 integer type.
fn generate_popcnt_backend_trait(ty: &W512Type) -> String {
    let trait_name = ty.trait_name();
    let name = ty.name();

    formatdoc! {r#"
        /// Population count (popcnt) extension for {name}.
        ///
        /// Returns a vector where each lane contains the number of set bits
        /// in the corresponding lane of the input.
        ///
        /// Requires AVX-512 VPOPCNTDQ (32/64-bit) or BITALG (8/16-bit).
        pub trait {name}PopcntBackend: {trait_name} {{
            /// Count set bits in each lane.
            fn popcnt(self, a: Self::Repr) -> Self::Repr;
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
    let arcane = super::backend_syntax::arcane("X64V4xToken");
    let name = ty.name();
    let epi = ty.x86_arith_suffix();

    formatdoc! {r#"
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        impl {name}PopcntBackend for archmage::X64V4xToken {{
            {arcane}
            fn popcnt(self, a: __m512i) -> __m512i {{
                _mm512_popcnt_{epi}(a)
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
