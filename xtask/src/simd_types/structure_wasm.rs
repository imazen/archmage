//! WebAssembly SIMD128 type structure generation.
//!
//! Generates WASM SIMD types parallel to x86 and ARM types.

use super::arch::Arch;
use super::arch::wasm::Wasm;
use super::ops_bitcast;
use super::types::{ElementType, SimdType, SimdWidth};
use indoc::formatdoc;

/// Generate a complete WASM SIMD type
pub fn generate_type(ty: &SimdType) -> String {
    assert!(
        ty.width == SimdWidth::W128,
        "WASM only supports 128-bit types"
    );

    let name = ty.name();
    let lanes = ty.lanes();
    let elem = ty.elem.name();
    let inner = Wasm::intrinsic_type(ty.elem, ty.width);

    let mut code = formatdoc! {"

        // ============================================================================
        // {name} - {lanes} x {elem} (128-bit WASM SIMD)
        // ============================================================================

        #[derive(Clone, Copy, Debug)]
        #[repr(transparent)]
        pub struct {name}({inner});

        impl {name} {{
        pub const LANES: usize = {lanes};

    "};

    // Construction methods
    code.push_str(&generate_construction_methods(ty));

    // Math operations
    code.push_str(&generate_math_ops(ty));

    // Horizontal operations
    code.push_str(&generate_horizontal_ops(ty));

    // Comparison operations
    code.push_str(&generate_comparison_ops(ty));

    // Bitwise operations (not, shift, blend)
    code.push_str(&generate_bitwise_ops(ty));

    // Type conversion operations (f32 <-> i32)
    code.push_str(&generate_conversion_ops(ty));

    // Transcendental operations (log, exp, pow) for float types
    code.push_str(&super::transcendental_wasm::generate_wasm_transcendental_ops(ty));

    // Bitcast operations (reinterpret bits between same-width types)
    code.push_str(&ops_bitcast::generate_wasm_bitcasts(ty));

    // Block operations (interleave, transpose)
    code.push_str(&super::block_ops_wasm::generate_wasm_block_ops(ty));

    // Extend/pack operations (integer widening/narrowing)
    code.push_str(&super::extend_ops_wasm::generate_wasm_extend_ops(ty));

    code.push_str("}\n\n");

    // Operator implementations
    code.push_str(&generate_operator_impls(ty));

    code
}

/// Generate construction and extraction methods
fn generate_construction_methods(ty: &SimdType) -> String {
    let name = ty.name();
    let lanes = ty.lanes();
    let elem = ty.elem.name();
    let inner = Wasm::intrinsic_type(ty.elem, ty.width);
    let splat_fn = Wasm::splat_intrinsic(ty.elem);
    let zero_val = ty.elem.zero_literal();
    let byte_size = 16; // WASM v128 is always 16 bytes

    formatdoc! {r#"
        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(_: archmage::Wasm128Token, data: &[{elem}; {lanes}]) -> Self {{
        Self(unsafe {{ v128_load(data.as_ptr() as *const v128) }})
        }}

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(_: archmage::Wasm128Token, v: {elem}) -> Self {{
        Self({splat_fn}(v))
        }}

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(_: archmage::Wasm128Token) -> Self {{
        Self({splat_fn}({zero_val}))
        }}

        /// Create from array (token-gated, zero-cost)
        ///
        /// This is a zero-cost transmute, not a memory load.
        #[inline(always)]
        pub fn from_array(_: archmage::Wasm128Token, arr: [{elem}; {lanes}]) -> Self {{
        // SAFETY: [{elem}; {lanes}] and {inner} have identical size and layout
        Self(unsafe {{ core::mem::transmute(arr) }})
        }}

        /// Create from slice (token-gated).
        ///
        /// # Panics
        ///
        /// Panics if `slice.len() < {lanes}`.
        #[inline(always)]
        pub fn from_slice(_: archmage::Wasm128Token, slice: &[{elem}]) -> Self {{
        let arr: [{elem}; {lanes}] = slice[..{lanes}].try_into().unwrap();
        // SAFETY: [{elem}; {lanes}] and {inner} have identical size and layout
        Self(unsafe {{ core::mem::transmute(arr) }})
        }}

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [{elem}; {lanes}]) {{
        unsafe {{ v128_store(out.as_mut_ptr() as *mut v128, self.0) }};
        }}

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [{elem}; {lanes}] {{
        let mut out = [{zero_val}; {lanes}];
        self.store(&mut out);
        out
        }}

        /// Get reference to underlying array
        #[inline(always)]
        pub fn as_array(&self) -> &[{elem}; {lanes}] {{
        unsafe {{ &*(self as *const Self as *const [{elem}; {lanes}]) }}
        }}

        /// Get mutable reference to underlying array
        #[inline(always)]
        pub fn as_array_mut(&mut self) -> &mut [{elem}; {lanes}] {{
        unsafe {{ &mut *(self as *mut Self as *mut [{elem}; {lanes}]) }}
        }}

        /// Get raw intrinsic type
        #[inline(always)]
        pub fn raw(self) -> {inner} {{
        self.0
        }}

        /// Create from raw intrinsic (unsafe - no token check)
        ///
        /// # Safety
        /// Caller must ensure the CPU supports WASM SIMD128.
        #[inline(always)]
        pub unsafe fn from_raw(v: {inner}) -> Self {{
        Self(v)
        }}

        /// Create from raw `{inner}` (token-gated, zero-cost).
        ///
        /// This is the safe alternative to [`from_raw`](Self::from_raw). The token
        /// proves the CPU supports the required SIMD features.
        #[inline(always)]
        pub fn from_{inner}(_: archmage::Wasm128Token, v: {inner}) -> Self {{
        Self(v)
        }}

        // ========== Token-gated bytemuck replacements ==========

        /// Reinterpret a slice of scalars as a slice of SIMD vectors (token-gated).
        ///
        /// Returns `None` if the slice length is not a multiple of {lanes}, or
        /// if the slice is not properly aligned.
        ///
        /// This is a safe, token-gated replacement for `bytemuck::cast_slice`.
        #[inline(always)]
        pub fn cast_slice(_: archmage::Wasm128Token, slice: &[{elem}]) -> Option<&[Self]> {{
        if slice.len() % {lanes} != 0 {{
            return None;
        }}
        let ptr = slice.as_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {{
            return None;
        }}
        let len = slice.len() / {lanes};
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe {{ core::slice::from_raw_parts(ptr as *const Self, len) }})
        }}

        /// Reinterpret a mutable slice of scalars as a slice of SIMD vectors (token-gated).
        ///
        /// Returns `None` if the slice length is not a multiple of {lanes}, or
        /// if the slice is not properly aligned.
        ///
        /// This is a safe, token-gated replacement for `bytemuck::cast_slice_mut`.
        #[inline(always)]
        pub fn cast_slice_mut(_: archmage::Wasm128Token, slice: &mut [{elem}]) -> Option<&mut [Self]> {{
        if slice.len() % {lanes} != 0 {{
            return None;
        }}
        let ptr = slice.as_mut_ptr();
        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {{
            return None;
        }}
        let len = slice.len() / {lanes};
        // SAFETY: alignment and length checked, layout is compatible
        Some(unsafe {{ core::slice::from_raw_parts_mut(ptr as *mut Self, len) }})
        }}

        /// View this vector as a byte array.
        ///
        /// This is a safe replacement for `bytemuck::bytes_of`.
        #[inline(always)]
        pub fn as_bytes(&self) -> &[u8; {byte_size}] {{
        // SAFETY: Self is repr(transparent) over v128 which is {byte_size} bytes
        unsafe {{ &*(self as *const Self as *const [u8; {byte_size}]) }}
        }}

        /// View this vector as a mutable byte array.
        ///
        /// This is a safe replacement for `bytemuck::bytes_of_mut`.
        #[inline(always)]
        pub fn as_bytes_mut(&mut self) -> &mut [u8; {byte_size}] {{
        // SAFETY: Self is repr(transparent) over v128 which is {byte_size} bytes
        unsafe {{ &mut *(self as *mut Self as *mut [u8; {byte_size}]) }}
        }}

        /// Create from a byte array reference (token-gated).
        ///
        /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
        #[inline(always)]
        pub fn from_bytes(_: archmage::Wasm128Token, bytes: &[u8; {byte_size}]) -> Self {{
        // SAFETY: [u8; {byte_size}] and Self have identical size
        Self(unsafe {{ core::mem::transmute(*bytes) }})
        }}

        /// Create from an owned byte array (token-gated, zero-cost).
        ///
        /// This is a zero-cost transmute from an owned byte array.
        #[inline(always)]
        pub fn from_bytes_owned(_: archmage::Wasm128Token, bytes: [u8; {byte_size}]) -> Self {{
        // SAFETY: [u8; {byte_size}] and Self have identical size
        Self(unsafe {{ core::mem::transmute(bytes) }})
        }}

        // ========== Implementation identification ==========

        /// Returns a string identifying this type's implementation.
        ///
        /// This is useful for verifying that the correct implementation is being used
        /// at compile time (via `-Ctarget-cpu`) or at runtime (via `#[magetypes]` dispatch).
        ///
        /// Returns `"wasm::wasm128::{name}"`.
        #[inline(always)]
        pub const fn implementation_name() -> &'static str {{
        "wasm::wasm128::{name}"
        }}

    "#}
}

/// Generate math operations for WASM
fn generate_math_ops(ty: &SimdType) -> String {
    let mut code = String::new();

    // Min/Max - available for floats and integers (except i64/u64)
    let has_minmax = ty.elem.is_float()
        || matches!(
            ty.elem,
            ElementType::I8
                | ElementType::U8
                | ElementType::I16
                | ElementType::U16
                | ElementType::I32
                | ElementType::U32
        );

    if has_minmax {
        let min_fn = Wasm::min_intrinsic(ty.elem);
        let max_fn = Wasm::max_intrinsic(ty.elem);

        code.push_str(&formatdoc! {r#"
            /// Element-wise minimum
            #[inline(always)]
            pub fn min(self, other: Self) -> Self {{
            Self({min_fn}(self.0, other.0))
            }}

            /// Element-wise maximum
            #[inline(always)]
            pub fn max(self, other: Self) -> Self {{
            Self({max_fn}(self.0, other.0))
            }}

            /// Clamp values between lo and hi
            #[inline(always)]
            pub fn clamp(self, lo: Self, hi: Self) -> Self {{
            self.max(lo).min(hi)
            }}

        "#});
    }

    // Float-only operations
    if ty.elem.is_float() {
        let sqrt_fn = Wasm::sqrt_intrinsic(ty.elem);
        let abs_fn = Wasm::abs_intrinsic(ty.elem);
        let floor_fn = Wasm::floor_intrinsic(ty.elem);
        let ceil_fn = Wasm::ceil_intrinsic(ty.elem);
        let nearest_fn = Wasm::nearest_intrinsic(ty.elem);
        let mul_fn = Wasm::arith_intrinsic("mul", ty.elem);
        let add_fn = Wasm::arith_intrinsic("add", ty.elem);
        let sub_fn = Wasm::arith_intrinsic("sub", ty.elem);

        code.push_str(&formatdoc! {r#"
            /// Square root
            #[inline(always)]
            pub fn sqrt(self) -> Self {{
            Self({sqrt_fn}(self.0))
            }}

            /// Absolute value
            #[inline(always)]
            pub fn abs(self) -> Self {{
            Self({abs_fn}(self.0))
            }}

            /// Floor
            #[inline(always)]
            pub fn floor(self) -> Self {{
            Self({floor_fn}(self.0))
            }}

            /// Ceil
            #[inline(always)]
            pub fn ceil(self) -> Self {{
            Self({ceil_fn}(self.0))
            }}

            /// Round to nearest
            #[inline(always)]
            pub fn round(self) -> Self {{
            Self({nearest_fn}(self.0))
            }}

            /// Fused multiply-add: self * a + b
            ///
            /// Note: WASM doesn't have native FMA in stable SIMD,
            /// this is emulated with separate mul and add.
            #[inline(always)]
            pub fn mul_add(self, a: Self, b: Self) -> Self {{
            Self({add_fn}({mul_fn}(self.0, a.0), b.0))
            }}

            /// Fused multiply-subtract: self * a - b
            ///
            /// Note: WASM doesn't have native FMA in stable SIMD,
            /// this is emulated with separate mul and sub.
            #[inline(always)]
            pub fn mul_sub(self, a: Self, b: Self) -> Self {{
            Self({sub_fn}({mul_fn}(self.0, a.0), b.0))
            }}

        "#});

        // Approximation operations (f32 only)
        if ty.elem == ElementType::F32 {
            code.push_str(&formatdoc! {r#"
                // ========== Approximation Operations ==========
                // WASM has no native reciprocal estimate intrinsics.
                // These use division for correct results.

                /// Reciprocal approximation (1/x) - uses division.
                ///
                /// Note: WASM has no native reciprocal estimate intrinsic,
                /// so this is equivalent to `recip()` (full precision division).
                #[inline(always)]
                pub fn rcp_approx(self) -> Self {{
                let one = Self(f32x4_splat(1.0));
                Self(f32x4_div(one.0, self.0))
                }}

                /// Precise reciprocal (1/x).
                #[inline(always)]
                pub fn recip(self) -> Self {{
                let one = Self(f32x4_splat(1.0));
                Self(f32x4_div(one.0, self.0))
                }}

                /// Reciprocal square root approximation (1/sqrt(x)) - uses sqrt+division.
                ///
                /// Note: WASM has no native rsqrt estimate intrinsic,
                /// so this is equivalent to `rsqrt()` (full precision).
                #[inline(always)]
                pub fn rsqrt_approx(self) -> Self {{
                let one = Self(f32x4_splat(1.0));
                Self(f32x4_div(one.0, f32x4_sqrt(self.0)))
                }}

                /// Precise reciprocal square root (1/sqrt(x)).
                #[inline(always)]
                pub fn rsqrt(self) -> Self {{
                let one = Self(f32x4_splat(1.0));
                Self(f32x4_div(one.0, f32x4_sqrt(self.0)))
                }}

            "#});
        }
    }

    // Signed integer abs
    if !ty.elem.is_float() && ty.elem.is_signed() && ty.elem != ElementType::I64 {
        let abs_fn = Wasm::abs_intrinsic(ty.elem);
        code.push_str(&formatdoc! {r#"
            /// Absolute value
            #[inline(always)]
            pub fn abs(self) -> Self {{
            Self({abs_fn}(self.0))
            }}

        "#});
    }

    // i64 polyfill: abs, min, max, clamp via compare+select
    if ty.elem == ElementType::I64 {
        code.push_str(&formatdoc! {r#"
            /// Absolute value (polyfill via conditional negate)
            #[inline(always)]
            pub fn abs(self) -> Self {{
            let negated = i64x2_neg(self.0);
            let zero = i64x2_splat(0);
            let mask = i64x2_lt(self.0, zero);
            Self(v128_bitselect(negated, self.0, mask))
            }}

            /// Element-wise minimum (polyfill via compare+select)
            #[inline(always)]
            pub fn min(self, other: Self) -> Self {{
            let mask = i64x2_gt(self.0, other.0);
            Self(v128_bitselect(other.0, self.0, mask))
            }}

            /// Element-wise maximum (polyfill via compare+select)
            #[inline(always)]
            pub fn max(self, other: Self) -> Self {{
            let mask = i64x2_gt(self.0, other.0);
            Self(v128_bitselect(self.0, other.0, mask))
            }}

            /// Clamp values between lo and hi
            #[inline(always)]
            pub fn clamp(self, lo: Self, hi: Self) -> Self {{
            self.max(lo).min(hi)
            }}

        "#});
    }

    // u64 polyfill: min, max, clamp via bias-to-signed compare+select
    if ty.elem == ElementType::U64 {
        code.push_str(&formatdoc! {r#"
            /// Element-wise minimum (polyfill via unsigned compare+select)
            #[inline(always)]
            pub fn min(self, other: Self) -> Self {{
            // Bias to signed domain for comparison
            let bias = i64x2_splat(i64::MIN);
            let a_biased = v128_xor(self.0, bias);
            let b_biased = v128_xor(other.0, bias);
            let mask = i64x2_gt(a_biased, b_biased);
            Self(v128_bitselect(other.0, self.0, mask))
            }}

            /// Element-wise maximum (polyfill via unsigned compare+select)
            #[inline(always)]
            pub fn max(self, other: Self) -> Self {{
            let bias = i64x2_splat(i64::MIN);
            let a_biased = v128_xor(self.0, bias);
            let b_biased = v128_xor(other.0, bias);
            let mask = i64x2_gt(a_biased, b_biased);
            Self(v128_bitselect(self.0, other.0, mask))
            }}

            /// Clamp values between lo and hi
            #[inline(always)]
            pub fn clamp(self, lo: Self, hi: Self) -> Self {{
            self.max(lo).min(hi)
            }}

        "#});
    }

    code
}

/// Generate horizontal operations for WASM
fn generate_horizontal_ops(ty: &SimdType) -> String {
    let elem = ty.elem.name();
    let lanes = ty.lanes();
    let extract_fn = Wasm::extract_lane_intrinsic(ty.elem);

    let reduce_body = match lanes {
        2 => format!("{extract_fn}::<0>(self.0) + {extract_fn}::<1>(self.0)"),
        4 => format!(
            "{extract_fn}::<0>(self.0) + {extract_fn}::<1>(self.0) + {extract_fn}::<2>(self.0) + {extract_fn}::<3>(self.0)"
        ),
        8 => formatdoc! {r#"
            let arr = self.to_array();
            arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7]"#},
        16 => formatdoc! {r#"
            let arr = self.to_array();
            arr.iter().copied().fold(0{elem}, |a, b| a.wrapping_add(b))"#},
        _ => formatdoc! {r#"
            let arr = self.to_array();
            arr.iter().copied().sum()"#},
    };

    let mut code = formatdoc! {r#"
        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> {elem} {{
        {reduce_body}
        }}

    "#};

    // reduce_max/reduce_min for floats
    if ty.elem.is_float() {
        code.push_str(&formatdoc! {r#"
            /// Reduce: max of all lanes
            #[inline(always)]
            pub fn reduce_max(self) -> {elem} {{
            let arr = self.to_array();
            arr.iter().copied().fold({elem}::NEG_INFINITY, {elem}::max)
            }}

            /// Reduce: min of all lanes
            #[inline(always)]
            pub fn reduce_min(self) -> {elem} {{
            let arr = self.to_array();
            arr.iter().copied().fold({elem}::INFINITY, {elem}::min)
            }}

        "#});
    }

    code
}

/// Generate comparison operations for WASM
fn generate_comparison_ops(ty: &SimdType) -> String {
    let name = ty.name();
    let eq_fn = Wasm::cmp_intrinsic("eq", ty.elem);

    let mut code = formatdoc! {r#"
        /// Element-wise equality comparison (returns mask)
        #[inline(always)]
        pub fn simd_eq(self, other: Self) -> Self {{
        Self({eq_fn}(self.0, other.0))
        }}

        /// Element-wise inequality comparison (returns mask)
        #[inline(always)]
        pub fn simd_ne(self, other: Self) -> Self {{
        Self(v128_not({eq_fn}(self.0, other.0)))
        }}

    "#};

    if ty.elem == ElementType::U64 {
        // WASM has no native u64 ordering comparisons.
        // Polyfill: XOR with i64::MIN to bias into signed domain, then use i64x2 comparisons.
        code.push_str(&formatdoc! {r#"
            /// Element-wise less-than comparison (returns mask)
            ///
            /// Polyfill: biases to signed domain via XOR with `i64::MIN`, then uses `i64x2_lt`.
            #[inline(always)]
            pub fn simd_lt(self, other: Self) -> Self {{
            let bias = i64x2_splat(i64::MIN);
            Self(i64x2_lt(v128_xor(self.0, bias), v128_xor(other.0, bias)))
            }}

            /// Element-wise less-than-or-equal comparison (returns mask)
            ///
            /// Polyfill: `a <= b` is `!(a > b)`.
            #[inline(always)]
            pub fn simd_le(self, other: Self) -> Self {{
            let bias = i64x2_splat(i64::MIN);
            Self(v128_not(i64x2_gt(v128_xor(self.0, bias), v128_xor(other.0, bias))))
            }}

            /// Element-wise greater-than comparison (returns mask)
            ///
            /// Polyfill: biases to signed domain via XOR with `i64::MIN`, then uses `i64x2_gt`.
            #[inline(always)]
            pub fn simd_gt(self, other: Self) -> Self {{
            let bias = i64x2_splat(i64::MIN);
            Self(i64x2_gt(v128_xor(self.0, bias), v128_xor(other.0, bias)))
            }}

            /// Element-wise greater-than-or-equal comparison (returns mask)
            ///
            /// Polyfill: `a >= b` is `!(a < b)`.
            #[inline(always)]
            pub fn simd_ge(self, other: Self) -> Self {{
            let bias = i64x2_splat(i64::MIN);
            Self(v128_not(i64x2_lt(v128_xor(self.0, bias), v128_xor(other.0, bias))))
            }}

        "#});
    } else {
        let lt_fn = Wasm::cmp_intrinsic("lt", ty.elem);
        let le_fn = Wasm::cmp_intrinsic("le", ty.elem);
        let gt_fn = Wasm::cmp_intrinsic("gt", ty.elem);
        let ge_fn = Wasm::cmp_intrinsic("ge", ty.elem);

        code.push_str(&formatdoc! {r#"
            /// Element-wise less-than comparison (returns mask)
            #[inline(always)]
            pub fn simd_lt(self, other: Self) -> Self {{
            Self({lt_fn}(self.0, other.0))
            }}

            /// Element-wise less-than-or-equal comparison (returns mask)
            #[inline(always)]
            pub fn simd_le(self, other: Self) -> Self {{
            Self({le_fn}(self.0, other.0))
            }}

            /// Element-wise greater-than comparison (returns mask)
            #[inline(always)]
            pub fn simd_gt(self, other: Self) -> Self {{
            Self({gt_fn}(self.0, other.0))
            }}

            /// Element-wise greater-than-or-equal comparison (returns mask)
            #[inline(always)]
            pub fn simd_ge(self, other: Self) -> Self {{
            Self({ge_fn}(self.0, other.0))
            }}

        "#});
    }

    // blend (conditional select using mask) — matches x86/ARM signature
    code.push_str(&formatdoc! {r#"
        /// Blend two vectors based on a mask
        ///
        /// For each lane, selects from `if_true` if the corresponding mask lane is all-ones,
        /// otherwise selects from `if_false`.
        ///
        /// The mask should come from a comparison operation like `simd_lt()`.
        ///
        /// # Example
        /// ```ignore
        /// let a = {name}::splat(token, 1.0);
        /// let b = {name}::splat(token, 2.0);
        /// let mask = a.simd_lt(b);  // all true
        /// let result = {name}::blend(mask, a, b);  // selects a
        /// ```
        #[inline(always)]
        pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {{
        Self(v128_bitselect(if_true.0, if_false.0, mask.0))
        }}

    "#});

    code
}

/// Generate bitwise operations (not, shift) for WASM
fn generate_bitwise_ops(ty: &SimdType) -> String {
    let mut code = formatdoc! {r#"
        /// Bitwise NOT
        #[inline(always)]
        pub fn not(self) -> Self {{
        Self(v128_not(self.0))
        }}

    "#};

    // Shift operations for integer types
    if !ty.elem.is_float() {
        let shl_fn = Wasm::shl_intrinsic(ty.elem);
        let shr_logical_fn = Wasm::shr_logical_intrinsic(ty.elem);
        let shr_arith_fn = Wasm::shr_intrinsic(ty.elem);

        code.push_str(&formatdoc! {r#"
            /// Shift left by constant
            #[inline(always)]
            pub fn shl<const N: u32>(self) -> Self {{
            Self({shl_fn}(self.0, N))
            }}

        "#});

        if ty.elem.is_signed() {
            // Signed shr_logical: use unsigned shr intrinsic (v128 bitwise identity)
            code.push_str(&formatdoc! {r#"
                /// Shift right by `N` bits (logical/unsigned shift).
                ///
                /// Bits shifted out are lost; zeros are shifted in.
                #[inline(always)]
                pub fn shr_logical<const N: u32>(self) -> Self {{
                Self({shr_logical_fn}(self.0, N))
                }}

            "#});
        } else {
            code.push_str(&formatdoc! {r#"
                /// Shift right by `N` bits (logical/unsigned shift).
                ///
                /// Bits shifted out are lost; zeros are shifted in.
                #[inline(always)]
                pub fn shr_logical<const N: u32>(self) -> Self {{
                Self({shr_logical_fn}(self.0, N))
                }}

            "#});
        }

        // shr_arithmetic for signed types
        if ty.elem.is_signed() {
            code.push_str(&formatdoc! {r#"
                /// Arithmetic shift right by `N` bits (sign-extending).
                ///
                /// The sign bit is replicated into the vacated positions.
                #[inline(always)]
                pub fn shr_arithmetic<const N: u32>(self) -> Self {{
                Self({shr_arith_fn}(self.0, N))
                }}

            "#});
        }
    }

    // all_true and any_true - only for integer types (WASM doesn't have float versions)
    if !ty.elem.is_float() {
        let all_true_fn = Wasm::all_true_intrinsic(ty.elem);
        let bitmask_fn = Wasm::bitmask_intrinsic(ty.elem);

        code.push_str(&formatdoc! {r#"
            /// Check if all lanes are non-zero (all true)
            #[inline(always)]
            pub fn all_true(self) -> bool {{
            {all_true_fn}(self.0)
            }}

            /// Check if any lane is non-zero (any true)
            #[inline(always)]
            pub fn any_true(self) -> bool {{
            v128_any_true(self.0)
            }}

            /// Extract the high bit of each lane as a bitmask
            #[inline(always)]
            pub fn bitmask(self) -> u32 {{
            {bitmask_fn}(self.0) as u32
            }}

        "#});
    }

    code
}

/// Generate type conversion operations (f32 <-> i32, etc.)
fn generate_conversion_ops(ty: &SimdType) -> String {
    let lanes = ty.lanes();
    let mut code = String::new();

    // f32 <-> i32 conversions
    if ty.elem == ElementType::F32 && lanes == 4 {
        code.push_str(&formatdoc! {r#"
            // ========== Type Conversions ==========

            /// Convert to signed 32-bit integers, rounding toward zero (truncation).
            ///
            /// Values outside the representable range are saturated to `i32::MIN`/`i32::MAX`.
            #[inline(always)]
            pub fn to_i32x4(self) -> i32x4 {{
            i32x4(i32x4_trunc_sat_f32x4(self.0))
            }}

            /// Convert to signed 32-bit integers, rounding to nearest.
            ///
            /// Values outside the representable range are saturated to `i32::MIN`/`i32::MAX`.
            ///
            /// Note: Uses `nearest()` intrinsic for proper round-to-nearest-even.
            #[inline(always)]
            pub fn to_i32x4_round(self) -> i32x4 {{
            i32x4(i32x4_trunc_sat_f32x4(f32x4_nearest(self.0)))
            }}

            /// Create from signed 32-bit integers.
            #[inline(always)]
            pub fn from_i32x4(v: i32x4) -> Self {{
            Self(f32x4_convert_i32x4(v.0))
            }}

        "#});
    }

    // i32 -> f32 conversion
    if ty.elem == ElementType::I32 && lanes == 4 {
        code.push_str(&formatdoc! {r#"
            // ========== Type Conversions ==========

            /// Convert to single-precision floats.
            #[inline(always)]
            pub fn to_f32x4(self) -> f32x4 {{
            f32x4(f32x4_convert_i32x4(self.0))
            }}

            /// Convert to single-precision floats (alias for `to_f32x4`).
            #[inline(always)]
            pub fn to_f32(self) -> f32x4 {{
            self.to_f32x4()
            }}

        "#});
    }

    // f64 -> i32 conversion (2 lanes -> lower 2 lanes of i32x4)
    if ty.elem == ElementType::F64 && lanes == 2 {
        code.push_str(&formatdoc! {r#"
            // ========== Type Conversions ==========

            /// Convert to signed 32-bit integers (2 lanes), rounding toward zero.
            ///
            /// Returns an `i32x4` where only the lower 2 lanes are valid (upper 2 are zero).
            #[inline(always)]
            pub fn to_i32x4_low(self) -> i32x4 {{
            // WASM: i32x4_trunc_sat_f64x2_zero converts f64x2 to lower 2 lanes of i32x4
            i32x4(i32x4_trunc_sat_f64x2_zero(self.0))
            }}

        "#});
    }

    code
}

/// Generate operator trait implementations for WASM
fn generate_operator_impls(ty: &SimdType) -> String {
    let name = ty.name();
    let elem = ty.elem.name();
    let lanes = ty.lanes();
    let inner = Wasm::intrinsic_type(ty.elem, ty.width);

    let add_fn = Wasm::arith_intrinsic("add", ty.elem);
    let sub_fn = Wasm::arith_intrinsic("sub", ty.elem);

    let mut code = formatdoc! {r#"
        impl core::ops::Add for {name} {{
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {{
            Self({add_fn}(self.0, rhs.0))
        }}
        }}

        impl core::ops::Sub for {name} {{
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {{
            Self({sub_fn}(self.0, rhs.0))
        }}
        }}

    "#};

    // Mul (NOT available for i8/u8 in WASM - no i8x16_mul or u8x16_mul intrinsics)
    let has_mul = !matches!(ty.elem, ElementType::I8 | ElementType::U8);
    if has_mul {
        let mul_fn = Wasm::arith_intrinsic("mul", ty.elem);
        code.push_str(&formatdoc! {r#"
            impl core::ops::Mul for {name} {{
            type Output = Self;
            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {{
                Self({mul_fn}(self.0, rhs.0))
            }}
            }}

        "#});
    }

    // Div (floats only)
    if ty.elem.is_float() {
        let div_fn = Wasm::arith_intrinsic("div", ty.elem);
        code.push_str(&formatdoc! {r#"
            impl core::ops::Div for {name} {{
            type Output = Self;
            #[inline(always)]
            fn div(self, rhs: Self) -> Self {{
                Self({div_fn}(self.0, rhs.0))
            }}
            }}

        "#});
    }

    // Neg (floats and signed integers)
    if ty.elem.is_float() || ty.elem.is_signed() {
        let neg_fn = Wasm::neg_intrinsic(ty.elem);
        code.push_str(&formatdoc! {r#"
            impl core::ops::Neg for {name} {{
            type Output = Self;
            #[inline(always)]
            fn neg(self) -> Self {{
                Self({neg_fn}(self.0))
            }}
            }}

        "#});
    }

    // Bitwise ops (use v128_and/or/xor)
    code.push_str(&formatdoc! {r#"
        impl core::ops::BitAnd for {name} {{
        type Output = Self;
        #[inline(always)]
        fn bitand(self, rhs: Self) -> Self {{
            Self(v128_and(self.0, rhs.0))
        }}
        }}

        impl core::ops::BitOr for {name} {{
        type Output = Self;
        #[inline(always)]
        fn bitor(self, rhs: Self) -> Self {{
            Self(v128_or(self.0, rhs.0))
        }}
        }}

        impl core::ops::BitXor for {name} {{
        type Output = Self;
        #[inline(always)]
        fn bitxor(self, rhs: Self) -> Self {{
            Self(v128_xor(self.0, rhs.0))
        }}
        }}

        impl core::ops::AddAssign for {name} {{
        #[inline(always)]
        fn add_assign(&mut self, rhs: Self) {{
            *self = *self + rhs;
        }}
        }}

        impl core::ops::SubAssign for {name} {{
        #[inline(always)]
        fn sub_assign(&mut self, rhs: Self) {{
            *self = *self - rhs;
        }}
        }}

    "#});

    // MulAssign (only when Mul is available)
    if has_mul {
        code.push_str(&formatdoc! {r#"
            impl core::ops::MulAssign for {name} {{
            #[inline(always)]
            fn mul_assign(&mut self, rhs: Self) {{
                *self = *self * rhs;
            }}
            }}

        "#});
    }

    if ty.elem.is_float() {
        code.push_str(&formatdoc! {r#"
            impl core::ops::DivAssign for {name} {{
            #[inline(always)]
            fn div_assign(&mut self, rhs: Self) {{
                *self = *self / rhs;
            }}
            }}

        "#});
    }

    code.push_str(&formatdoc! {r#"
        impl core::ops::BitAndAssign for {name} {{
        #[inline(always)]
        fn bitand_assign(&mut self, rhs: Self) {{
            *self = *self & rhs;
        }}
        }}

        impl core::ops::BitOrAssign for {name} {{
        #[inline(always)]
        fn bitor_assign(&mut self, rhs: Self) {{
            *self = *self | rhs;
        }}
        }}

        impl core::ops::BitXorAssign for {name} {{
        #[inline(always)]
        fn bitxor_assign(&mut self, rhs: Self) {{
            *self = *self ^ rhs;
        }}
        }}

        impl core::ops::Index<usize> for {name} {{
        type Output = {elem};
        #[inline(always)]
        fn index(&self, i: usize) -> &Self::Output {{
            assert!(i < {lanes}, "index out of bounds");
            unsafe {{ &*(self as *const Self as *const {elem}).add(i) }}
        }}
        }}

        impl core::ops::IndexMut<usize> for {name} {{
        #[inline(always)]
        fn index_mut(&mut self, i: usize) -> &mut Self::Output {{
            assert!(i < {lanes}, "index out of bounds");
            unsafe {{ &mut *(self as *mut Self as *mut {elem}).add(i) }}
        }}
        }}

        impl From<[{elem}; {lanes}]> for {name} {{
        #[inline(always)]
        fn from(arr: [{elem}; {lanes}]) -> Self {{
            // SAFETY: [{elem}; {lanes}] and {inner} have identical size and layout
            Self(unsafe {{ core::mem::transmute(arr) }})
        }}
        }}

        impl From<{name}> for [{elem}; {lanes}] {{
        #[inline(always)]
        fn from(v: {name}) -> Self {{
            // SAFETY: {inner} and [{elem}; {lanes}] have identical size and layout
            unsafe {{ core::mem::transmute(v.0) }}
        }}
        }}

    "#});

    code
}

/// Generate the WASM w128.rs file
pub fn generate_wasm_w128(types: &[SimdType]) -> String {
    let mut code = String::from(
        "//! 128-bit (WASM SIMD) types.\n//!\n//! **Auto-generated** by `cargo xtask generate` - do not edit manually.\n\nuse core::arch::wasm32::*;\n\n",
    );

    // Generate 128-bit types only
    let w128_types: Vec<_> = types
        .iter()
        .filter(|t| t.width == SimdWidth::W128)
        .collect();

    for ty in &w128_types {
        code.push_str(&generate_type(ty));
    }

    code
}

/// Get all WASM types to generate (128-bit only)
pub fn all_wasm_types() -> Vec<SimdType> {
    vec![
        SimdType::new(ElementType::F32, SimdWidth::W128),
        SimdType::new(ElementType::F64, SimdWidth::W128),
        SimdType::new(ElementType::I8, SimdWidth::W128),
        SimdType::new(ElementType::U8, SimdWidth::W128),
        SimdType::new(ElementType::I16, SimdWidth::W128),
        SimdType::new(ElementType::U16, SimdWidth::W128),
        SimdType::new(ElementType::I32, SimdWidth::W128),
        SimdType::new(ElementType::U32, SimdWidth::W128),
        SimdType::new(ElementType::I64, SimdWidth::W128),
        SimdType::new(ElementType::U64, SimdWidth::W128),
    ]
}
