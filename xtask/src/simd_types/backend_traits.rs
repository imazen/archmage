//! Unified backend trait generation for all 30 SIMD vector types.
//!
//! Generates every file under `magetypes/src/simd/backends/`:
//! - `sealed.rs`
//! - One `<type>.rs` per vector type (float and integer, W128 / W256 / W512)
//! - `popcnt.rs` (AVX-512 VPOPCNTDQ / BITALG extension traits)
//! - `convert.rs` and `convert_int.rs` (float↔int and signed↔unsigned bitcasts)
//! - `mod.rs` (re-exports and token type aliases)

use indoc::formatdoc;

use super::types::{ElementType, SimdType, SimdWidth};

/// All 20 narrow (W128 + W256) backend types in `backends/mod.rs` declaration order.
pub(super) fn narrow_backend_types() -> [SimdType; 20] {
    use ElementType::*;
    use SimdWidth::*;
    [
        // Float
        SimdType::new(F32, W128),
        SimdType::new(F32, W256),
        SimdType::new(F64, W128),
        SimdType::new(F64, W256),
        // i32
        SimdType::new(I32, W128),
        SimdType::new(I32, W256),
        // u32
        SimdType::new(U32, W128),
        SimdType::new(U32, W256),
        // i64
        SimdType::new(I64, W128),
        SimdType::new(I64, W256),
        // Remaining int (i8, u8, i16, u16, u64)
        SimdType::new(I8, W128),
        SimdType::new(I8, W256),
        SimdType::new(U8, W128),
        SimdType::new(U8, W256),
        SimdType::new(I16, W128),
        SimdType::new(I16, W256),
        SimdType::new(U16, W128),
        SimdType::new(U16, W256),
        SimdType::new(U64, W128),
        SimdType::new(U64, W256),
    ]
}

/// All 10 W512 backend types in `backends/mod.rs` declaration order.
pub(super) fn w512_backend_types() -> [SimdType; 10] {
    use ElementType::*;
    use SimdWidth::W512;
    [
        SimdType::new(F32, W512),
        SimdType::new(F64, W512),
        SimdType::new(I8, W512),
        SimdType::new(U8, W512),
        SimdType::new(I16, W512),
        SimdType::new(U16, W512),
        SimdType::new(I32, W512),
        SimdType::new(U32, W512),
        SimdType::new(I64, W512),
        SimdType::new(U64, W512),
    ]
}

// ============================================================================
// Sealed Trait
// ============================================================================

pub(super) fn generate_sealed() -> String {
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
// Per-Type Backend Trait Generation
// ============================================================================

/// Generate the backend trait file (`backends/<name>.rs`) for any SIMD vector type.
pub(super) fn generate_backend_trait(ty: SimdType) -> String {
    if ty.is_float() {
        generate_float_backend_trait(ty)
    } else {
        generate_int_backend_trait(ty)
    }
}

fn generate_float_backend_trait(ty: SimdType) -> String {
    let trait_name = ty.trait_name();
    let elem = ty.elem.name();
    let lanes = ty.lanes();
    let array = ty.array_type();
    let name = ty.name();
    let internal_trait = super::float_layout::internal_trait(elem, lanes);

    // Pixel-pack default (narrow f32 only): round-to-nearest + saturate each lane to u8.
    let to_u8_trait = if ty.elem == ElementType::F32 && !ty.is_w512() {
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
            #[deprecated(since = "0.9.32", note = "Use {name}::store_rgba_u8")]
            #[inline(always)]
            fn store_rgba_bytes(self, r: Self::Repr, g: Self::Repr, b: Self::Repr, a: Self::Repr) -> [u8; {rgba_bytes}] {{
                let mut out = [0u8; {rgba_bytes}];
                <Self as F32x{lanes}Internal>::store_rgba_u8_into(self, r, g, b, a, &mut out);
                out
            }}
        "#, rgba_bytes = lanes * 4}
    } else {
        String::new()
    };

    // 8x8 transpose lives only on f32x8 (8 rows of 8).
    let transpose_trait = if ty.elem == ElementType::F32 && lanes == 8 {
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

    let scalar_bound = if ty.elem == ElementType::F32 {
        format!(
            " + F32x{lanes}Internal<InternalRepr = <Self as {trait_name}>::Repr> + crate::scalar::F32ScalarBackend"
        )
    } else {
        " + crate::scalar::F64ScalarBackend".to_string()
    };

    let (approx_docs, recip_docs) = if ty.is_w512() {
        (
            formatdoc! {r#"
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
            "#},
            formatdoc! {r#"
                /// Precise reciprocal — defaults to delegating to rcp_approx.
                /// Every shipped backend overrides with exact IEEE division.
                #[inline(always)]
                fn recip(self, a: Self::Repr) -> Self::Repr {{ Self::rcp_approx(self, a) }}

                /// Precise reciprocal square root — see [`recip`](Self::recip).
                #[inline(always)]
                fn rsqrt(self, a: Self::Repr) -> Self::Repr {{ Self::rsqrt_approx(self, a) }}
            "#},
        )
    } else {
        (
            formatdoc! {r#"
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
            "#},
            formatdoc! {r#"
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
            "#},
        )
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

            {approx_docs}
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

            {recip_docs}{to_u8_trait}{transpose_trait}
        }}
    "#}
}

fn generate_int_backend_trait(ty: SimdType) -> String {
    let trait_name = ty.trait_name();
    let elem = ty.elem.name();
    let elem_bits = ty.elem_bits();
    let lanes = ty.lanes();
    let array = ty.array_type();
    let name = ty.name();
    let is_w512 = ty.is_w512();
    let is_narrow_u32 = !is_w512 && ty.elem == ElementType::U32;
    let is_narrow_remaining = !is_w512
        && matches!(
            ty.elem,
            ElementType::I8
                | ElementType::U8
                | ElementType::I16
                | ElementType::U16
                | ElementType::U64
        );

    let wrapping_note = if is_narrow_u32 || is_narrow_remaining {
        " (wrapping)"
    } else {
        ""
    };
    let unsigned_word = if is_narrow_u32 { "unsigned " } else { "" };

    let sep = if is_narrow_remaining { "" } else { "\n" };

    let mul_section = if matches!(elem_bits, 16 | 32) {
        let doc = if is_w512 {
            "Lane-wise multiplication (low bits of product)."
        } else if elem_bits == 32 {
            "Lane-wise multiplication (low 32 bits of each 32x32 product)."
        } else {
            "Lane-wise multiplication (low 16 bits of product)."
        };
        format!(
            "{sep}    /// {doc}\n    fn mul(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;\n"
        )
    } else if !is_w512 && ty.elem == ElementType::I64 {
        "\n    // NOTE: No `mul` — no native i64 multiply on most platforms before AVX-512.\n"
            .to_string()
    } else {
        String::new()
    };

    // W512 integer traits include `neg` for unsigned types too; narrow traits gate it on signed.
    let neg_section = if ty.is_signed() || is_w512 {
        format!(
            "{sep}    /// Lane-wise negation.\n    fn neg(self, a: Self::Repr) -> Self::Repr;\n"
        )
    } else {
        String::new()
    };

    let abs_section = if ty.is_signed() {
        format!(
            "{sep}    /// Lane-wise absolute value.\n    fn abs(self, a: Self::Repr) -> Self::Repr;\n"
        )
    } else {
        String::new()
    };

    let cmp_comment = if is_narrow_remaining {
        ""
    } else if is_narrow_u32 {
        "    // Return masks where each lane is all-1s (true) or all-0s (false).\n    // All comparisons are unsigned.\n"
    } else {
        "    // Return masks where each lane is all-1s (true) or all-0s (false).\n"
    };

    let shift_range_doc = if is_w512 {
        "`N` must be in `0..=lane_bits-1`."
    } else if is_narrow_remaining {
        "`N` must be in `0..=lane_bits-1`; the generic front-ends reject out-of-range `N` at compile time."
    } else if elem_bits == 32 {
        "`N` must be in `0..=31`; the generic front-ends reject out-of-range `N` at compile time."
    } else {
        "`N` must be in `0..=63`; the generic front-ends reject out-of-range `N` at compile time."
    };

    let shr_const_section = if is_w512 || matches!(ty.elem, ElementType::I32 | ElementType::I64) {
        formatdoc! {r#"

            /// Arithmetic shift right by constant (sign-extending).
            /// {shift_range_doc}
            fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;

            /// Logical shift right by constant (zero-filling).
            /// {shift_range_doc}
            fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
        "#}
    } else {
        let mut s = formatdoc! {r#"

            /// Logical shift right by constant (zero-filling).
            /// {shift_range_doc}
            fn shr_logical_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
        "#};
        if ty.is_signed() {
            s.push_str(&formatdoc! {r#"
                {sep}/// Arithmetic shift right by constant (sign-extending).
                /// {shift_range_doc}
                fn shr_arithmetic_const<const N: i32>(self, a: Self::Repr) -> Self::Repr;
            "#});
        }
        s
    };

    let section_lead = if is_w512 { "" } else { "\n" };
    let uniform_shifts_section = if matches!(elem_bits, 16 | 32) {
        let mut s = formatdoc! {r#"
            {section_lead}// ====== Uniform variable shifts ======

            /// Shift left by a runtime `count` applied identically to every lane.
            ///
            /// `count >= {elem_bits}` produces all-zero lanes on every backend.
            fn shl_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;

            /// Logical (zero-filling) shift right by a runtime `count` applied
            /// identically to every lane.
            ///
            /// `count >= {elem_bits}` produces all-zero lanes on every backend.
            fn shr_logical_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
        "#};
        if is_w512 {
            s.push_str(&formatdoc! {r#"

                /// Arithmetic (sign-filling) shift right by a runtime `count`
                /// applied identically to every lane. Identical to
                /// `shr_logical_uniform` for unsigned element types.
                ///
                /// `count >= {elem_bits}` produces a sign fill on every backend.
                fn shr_arithmetic_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
            "#});
        } else if ty.is_signed() {
            let wrap = if elem_bits == 32 {
                "`count >= 32` produces a sign fill (every lane becomes `0` or\n/// `-1`) on every backend."
            } else {
                "`count >= 16` produces a sign fill (every lane becomes\n/// `0` or `-1`) on every backend."
            };
            s.push_str(&formatdoc! {r#"

                /// Arithmetic (sign-filling) shift right by a runtime `count`
                /// applied identically to every lane.
                ///
                /// {wrap}
                fn shr_arithmetic_uniform(self, a: Self::Repr, count: u32) -> Self::Repr;
            "#});
        }
        s
    } else {
        String::new()
    };

    let sat_lead = if is_w512 && uniform_shifts_section.is_empty() {
        ""
    } else {
        "\n"
    };
    let saturating_section = if matches!(elem_bits, 8 | 16) {
        formatdoc! {r#"
            {sat_lead}// ====== Saturating arithmetic ======

            /// Lane-wise addition that clamps to the element range instead of
            /// wrapping (`core`'s `saturating_add`, per lane).
            fn saturating_add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise subtraction that clamps to the element range instead
            /// of wrapping (`core`'s `saturating_sub`, per lane).
            fn saturating_sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
        "#}
    } else {
        String::new()
    };

    let bitmask_ret = if is_w512 { "u64" } else { "u32" };
    let bitmask_doc = if !is_w512 && matches!(elem_bits, 32 | 64) && ty.elem != ElementType::U64 {
        format!("Extract the high bit of each {elem_bits}-bit lane as a bitmask.")
    } else {
        "Extract the high bit of each lane as a bitmask.".to_string()
    };
    let clamp_doc = if is_narrow_u32 {
        "Clamp values between lo and hi (unsigned comparison)."
    } else {
        "Clamp values between lo and hi."
    };
    let integer_methods = super::backend_gen_widen_narrow::trait_methods(&name);
    let int_sep = if is_narrow_remaining && !integer_methods.is_empty() {
        "\n"
    } else {
        ""
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

            /// Lane-wise addition{wrapping_note}.
            fn add(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise subtraction{wrapping_note}.
            fn sub(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
        {mul_section}{neg_section}
            // ====== Math ======

            /// Lane-wise {unsigned_word}minimum.
            fn min(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise {unsigned_word}maximum.
            fn max(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;
        {abs_section}
            // ====== Comparisons ======
        {cmp_comment}
            /// Lane-wise equality.
            fn simd_eq(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise inequality.
            fn simd_ne(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise {unsigned_word}less-than.
            fn simd_lt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise {unsigned_word}less-than-or-equal.
            fn simd_le(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise {unsigned_word}greater-than.
            fn simd_gt(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Lane-wise {unsigned_word}greater-than-or-equal.
            fn simd_ge(self, a: Self::Repr, b: Self::Repr) -> Self::Repr;

            /// Select lanes: where mask is all-1s pick `if_true`, else `if_false`.
            fn blend(self, mask: Self::Repr, if_true: Self::Repr, if_false: Self::Repr) -> Self::Repr;

            // ====== Reductions ======

            /// Sum all {lanes} lanes{wrapping_note}.
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
        {shr_const_section}{uniform_shifts_section}{saturating_section}
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

            /// {bitmask_doc}
            fn bitmask(self, a: Self::Repr) -> {bitmask_ret};

            // ====== Default implementations ======

            /// {clamp_doc}
            #[inline(always)]
            fn clamp(self, a: Self::Repr, lo: Self::Repr, hi: Self::Repr) -> Self::Repr {{
                <Self as {trait_name}>::min(self, <Self as {trait_name}>::max(self, a, lo), hi)
            }}
        {int_sep}{integer_methods}
        }}
    "#}
}

// ============================================================================
// Extension & Conversion Traits
// ============================================================================

/// Generate popcnt extension backend traits for all integer W512 types.
pub(super) fn generate_popcnt_backend_traits(types: &[SimdType]) -> String {
    let mut code = formatdoc! {r#"
        //! Popcnt extension backend traits for W512 integer types.
        //!
        //! These traits extend the base integer backends with population count
        //! operations, available on AVX-512 Modern (VPOPCNTDQ + BITALG).
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.

        use super::*;

    "#};

    for ty in types.iter().filter(|t| !t.is_float()) {
        let trait_name = ty.trait_name();
        let name = ty.name();
        code.push_str(&formatdoc! {r#"
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

        "#});
    }
    code
}

/// Generate `backends/convert.rs` (F32↔I32 conversions + U32↔I32 and I64↔F64 bitcasts).
pub(super) fn generate_convert_traits() -> String {
    let mut code = formatdoc! {r#"
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
    "#};

    for lanes in [4usize, 8, 16] {
        let cfg = if lanes == 16 {
            "#[cfg(feature = \"w512\")]\n"
        } else {
            ""
        };
        code.push_str(&formatdoc! {r#"

            /// Conversions between f32x{lanes} and i32x{lanes} representations.
            ///
            /// Requires both `F32x{lanes}Backend` and `I32x{lanes}Backend` to be implemented.
            {cfg}pub trait F32x{lanes}Convert: F32x{lanes}Backend + I32x{lanes}Backend + SimdToken + Sealed + Copy + 'static {{
                /// Bitcast f32x{lanes} to i32x{lanes} (reinterpret bits, no conversion).
                fn bitcast_f32_to_i32(self, a: <Self as F32x{lanes}Backend>::Repr) -> <Self as I32x{lanes}Backend>::Repr;

                /// Bitcast i32x{lanes} to f32x{lanes} (reinterpret bits, no conversion).
                fn bitcast_i32_to_f32(self, a: <Self as I32x{lanes}Backend>::Repr) -> <Self as F32x{lanes}Backend>::Repr;

                /// Convert f32x{lanes} to i32x{lanes} with truncation toward zero.
                fn convert_f32_to_i32(self, a: <Self as F32x{lanes}Backend>::Repr) -> <Self as I32x{lanes}Backend>::Repr;

                /// Convert with truncation and UNIFORM saturation: out-of-range
                /// lanes clamp to `i32::MIN`/`i32::MAX`, NaN lanes become 0 —
                /// identical on every backend (Rust `as` / NEON FCVTZS / WASM
                /// trunc_sat semantics). Default delegates to
                /// [`convert_f32_to_i32`](Self::convert_f32_to_i32) (already conformant on NEON/WASM/
                /// scalar); x86 overrides with a cvttps + compare/blend fixup
                /// (the bare op yields the `i32::MIN` sentinel for overflow AND
                /// NaN there — issue #80).
                #[inline(always)]
                fn convert_f32_to_i32_saturating(self, a: <Self as F32x{lanes}Backend>::Repr) -> <Self as I32x{lanes}Backend>::Repr {{
                    <Self as F32x{lanes}Convert>::convert_f32_to_i32(self, a)
                }}

                /// Fast in-range truncation toward zero (overflow/NaN unspecified; relaxed trunc on WASM `relaxed-simd`).
                #[inline(always)]
                fn convert_f32_to_i32_fast(self, a: <Self as F32x{lanes}Backend>::Repr) -> <Self as I32x{lanes}Backend>::Repr {{
                    <Self as F32x{lanes}Convert>::convert_f32_to_i32(self, a)
                }}

                /// Convert f32x{lanes} to i32x{lanes} with rounding to nearest.
                fn convert_f32_to_i32_round(self, a: <Self as F32x{lanes}Backend>::Repr) -> <Self as I32x{lanes}Backend>::Repr;

                /// Convert i32x{lanes} to f32x{lanes}.
                fn convert_i32_to_f32(self, a: <Self as I32x{lanes}Backend>::Repr) -> <Self as F32x{lanes}Backend>::Repr;
            }}
        "#});
    }

    for (a_elem, b_elem, a_up, b_up, lanes) in [
        ("u32", "i32", "U32", "I32", 4usize),
        ("u32", "i32", "U32", "I32", 8),
        ("i64", "f64", "I64", "F64", 2),
        ("i64", "f64", "I64", "F64", 4),
    ] {
        code.push_str(&formatdoc! {r#"

            /// Bitcast conversions between {a_elem}x{lanes} and {b_elem}x{lanes} representations.
            ///
            /// Requires both `{a_up}x{lanes}Backend` and `{b_up}x{lanes}Backend` to be implemented.
            pub trait {a_up}x{lanes}Bitcast: {a_up}x{lanes}Backend + {b_up}x{lanes}Backend + SimdToken + Sealed + Copy + 'static {{
                /// Bitcast {a_elem}x{lanes} to {b_elem}x{lanes} (reinterpret bits, no conversion).
                fn bitcast_{a_elem}_to_{b_elem}(self, a: <Self as {a_up}x{lanes}Backend>::Repr) -> <Self as {b_up}x{lanes}Backend>::Repr;

                /// Bitcast {b_elem}x{lanes} to {a_elem}x{lanes} (reinterpret bits, no conversion).
                fn bitcast_{b_elem}_to_{a_elem}(self, a: <Self as {b_up}x{lanes}Backend>::Repr) -> <Self as {a_up}x{lanes}Backend>::Repr;
            }}
        "#});
    }

    code
}

/// Generate `backends/convert_int.rs` (i8↔u8, i16↔u16, u64↔i64 bitcasts).
pub(super) fn generate_additional_convert_traits() -> String {
    let mut code = formatdoc! {r#"
        //! Bitcast conversion traits for integer types.
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.

        use super::sealed::Sealed;
        use archmage::SimdToken;
    "#};

    for (a_elem, b_elem, a_up, b_up, lanes, doc_extra) in [
        ("i8", "u8", "I8", "U8", 16usize, ""),
        ("i8", "u8", "I8", "U8", 32, ""),
        ("i16", "u16", "I16", "U16", 8, ""),
        ("i16", "u16", "I16", "U16", 16, ""),
        ("u64", "i64", "U64", "I64", 2, ", and f64x2"),
        ("u64", "i64", "U64", "I64", 4, ", and f64x4"),
    ] {
        let doc = if doc_extra.is_empty() {
            format!(
                "Bitcast conversions between {a_elem}x{lanes} and {b_elem}x{lanes} representations."
            )
        } else {
            format!(
                "Bitcast conversions between {a_elem}x{lanes}, {b_elem}x{lanes}{doc_extra} representations."
            )
        };
        code.push_str(&formatdoc! {r#"

            /// {doc}
            pub trait {a_up}x{lanes}Bitcast: super::{a_up}x{lanes}Backend + super::{b_up}x{lanes}Backend + SimdToken + Sealed + Copy + 'static {{
                /// Bitcast {a_elem}x{lanes} to {b_elem}x{lanes} (reinterpret bits).
                fn bitcast_{a_elem}_to_{b_elem}(self, a: <Self as super::{a_up}x{lanes}Backend>::Repr) -> <Self as super::{b_up}x{lanes}Backend>::Repr;
                /// Bitcast {b_elem}x{lanes} to {a_elem}x{lanes} (reinterpret bits).
                fn bitcast_{b_elem}_to_{a_elem}(self, a: <Self as super::{b_up}x{lanes}Backend>::Repr) -> <Self as super::{a_up}x{lanes}Backend>::Repr;
            }}
        "#});
    }

    code
}

// ============================================================================
// backends/mod.rs
// ============================================================================

pub(super) fn generate_backends_mod(narrow: &[SimdType], w512: &[SimdType]) -> String {
    let mut code = formatdoc! {r#"
        //! Backend traits for generic SIMD types.
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.

        #![allow(non_camel_case_types)]

        pub(crate) mod sealed;

    "#};

    for ty in narrow {
        let name = ty.name();
        let trait_name = ty.trait_name();
        code.push_str(&format!("mod {name};\npub use {name}::{trait_name};\n"));
        if ty.elem == ElementType::F32 {
            code.push_str(&format!(
                "pub(crate) use {name}::F32x{}Internal;\n",
                ty.lanes()
            ));
        }
        code.push('\n');
    }

    // W512 types gated behind `w512` feature (`avx512` implies `w512`).
    for ty in w512 {
        let name = ty.name();
        let trait_name = ty.trait_name();
        code.push_str(&format!(
            "#[cfg(feature = \"w512\")]\nmod {name};\n#[cfg(feature = \"w512\")]\npub use {name}::{trait_name};\n"
        ));
        if ty.elem == ElementType::F32 {
            code.push_str(&format!(
                "#[cfg(feature = \"w512\")]\npub(crate) use {name}::F32x{}Internal;\n",
                ty.lanes()
            ));
        }
        code.push('\n');
    }

    code.push_str("#[cfg(feature = \"avx512\")]\n");
    code.push_str("mod popcnt;\n");
    code.push_str("#[cfg(feature = \"avx512\")]\n");
    code.push_str("pub use popcnt::*;\n\n");

    code.push_str("mod convert;\n");
    code.push_str("pub use convert::{F32x4Convert, F32x8Convert, U32x4Bitcast, U32x8Bitcast, I64x2Bitcast, I64x4Bitcast};\n");
    code.push_str("#[cfg(feature = \"w512\")]\npub use convert::F32x16Convert;\n\n");

    code.push_str("mod convert_int;\npub use convert_int::{I8x16Bitcast, I8x32Bitcast, I16x8Bitcast, I16x16Bitcast, U64x2Bitcast, U64x4Bitcast};\n\n");

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
