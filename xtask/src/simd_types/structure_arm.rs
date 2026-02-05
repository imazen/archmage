//! ARM NEON type structure generation.
//!
//! Generates NEON SIMD types parallel to x86 types.

use super::arch::Arch;
use super::arch::arm::Arm;
use super::ops_bitcast;
use super::types::{ElementType, SimdType, SimdWidth};
use indoc::formatdoc;

/// Generate a complete NEON SIMD type
pub fn generate_type(ty: &SimdType) -> String {
    assert!(
        ty.width == SimdWidth::W128,
        "NEON only supports 128-bit types"
    );

    let name = ty.name();
    let lanes = ty.lanes();
    let elem = ty.elem.name();
    let inner = Arm::intrinsic_type(ty.elem, ty.width);

    let mut code = formatdoc! {"

        // ============================================================================
        // {name} - {lanes} x {elem} (128-bit NEON)
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

    // Bitwise operations (not, shift)
    code.push_str(&generate_bitwise_ops(ty));

    // Type conversion operations (f32 <-> i32)
    code.push_str(&generate_conversion_ops(ty));

    // Bitcast operations (reinterpret bits between same-width types)
    code.push_str(&ops_bitcast::generate_arm_bitcasts(ty));

    // Transcendental operations (log, exp, pow, cbrt)
    code.push_str(&super::transcendental_arm::generate_arm_transcendental_ops(
        ty,
    ));

    // Block operations (interleave, transpose)
    code.push_str(&super::block_ops_arm::generate_arm_block_ops(ty));

    // Extend/pack operations (integer widening/narrowing)
    code.push_str(&super::extend_ops_arm::generate_arm_extend_ops(ty));

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
    let inner = Arm::intrinsic_type(ty.elem, ty.width);

    let load_fn = Arm::load_intrinsic(ty.elem);
    let store_fn = Arm::store_intrinsic(ty.elem);
    let splat_fn = Arm::splat_intrinsic(ty.elem);
    let zero_val = ty.elem.zero_literal();
    let byte_size = ty.lanes() * ty.elem.size_bytes();

    formatdoc! {r#"
        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(_: archmage::NeonToken, data: &[{elem}; {lanes}]) -> Self {{
        Self(unsafe {{ {load_fn}(data.as_ptr()) }})
        }}

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(_: archmage::NeonToken, v: {elem}) -> Self {{
        Self(unsafe {{ {splat_fn}(v) }})
        }}

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(_: archmage::NeonToken) -> Self {{
        Self(unsafe {{ {splat_fn}({zero_val}) }})
        }}

        /// Create from array (token-gated, zero-cost)
        ///
        /// This is a zero-cost transmute, not a memory load.
        #[inline(always)]
        pub fn from_array(_: archmage::NeonToken, arr: [{elem}; {lanes}]) -> Self {{
        // SAFETY: [{elem}; {lanes}] and {inner} have identical size and layout
        Self(unsafe {{ core::mem::transmute(arr) }})
        }}

        /// Create from slice (token-gated).
        ///
        /// # Panics
        ///
        /// Panics if `slice.len() < {lanes}`.
        #[inline(always)]
        pub fn from_slice(_: archmage::NeonToken, slice: &[{elem}]) -> Self {{
        let arr: [{elem}; {lanes}] = slice[..{lanes}].try_into().unwrap();
        // SAFETY: [{elem}; {lanes}] and {inner} have identical size and layout
        Self(unsafe {{ core::mem::transmute(arr) }})
        }}

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [{elem}; {lanes}]) {{
        unsafe {{ {store_fn}(out.as_mut_ptr(), self.0) }};
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
        /// Caller must ensure the CPU supports the required SIMD features.
        #[inline(always)]
        pub unsafe fn from_raw(v: {inner}) -> Self {{
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
        pub fn cast_slice(_: archmage::NeonToken, slice: &[{elem}]) -> Option<&[Self]> {{
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
        pub fn cast_slice_mut(_: archmage::NeonToken, slice: &mut [{elem}]) -> Option<&mut [Self]> {{
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
        // SAFETY: Self is repr(transparent) over {inner} which is {byte_size} bytes
        unsafe {{ &*(self as *const Self as *const [u8; {byte_size}]) }}
        }}

        /// View this vector as a mutable byte array.
        ///
        /// This is a safe replacement for `bytemuck::bytes_of_mut`.
        #[inline(always)]
        pub fn as_bytes_mut(&mut self) -> &mut [u8; {byte_size}] {{
        // SAFETY: Self is repr(transparent) over {inner} which is {byte_size} bytes
        unsafe {{ &mut *(self as *mut Self as *mut [u8; {byte_size}]) }}
        }}

        /// Create from a byte array reference (token-gated).
        ///
        /// This is a safe, token-gated replacement for `bytemuck::from_bytes`.
        #[inline(always)]
        pub fn from_bytes(_: archmage::NeonToken, bytes: &[u8; {byte_size}]) -> Self {{
        // SAFETY: [u8; {byte_size}] and Self have identical size
        Self(unsafe {{ core::mem::transmute(*bytes) }})
        }}

        /// Create from an owned byte array (token-gated, zero-cost).
        ///
        /// This is a zero-cost transmute from an owned byte array.
        #[inline(always)]
        pub fn from_bytes_owned(_: archmage::NeonToken, bytes: [u8; {byte_size}]) -> Self {{
        // SAFETY: [u8; {byte_size}] and Self have identical size
        Self(unsafe {{ core::mem::transmute(bytes) }})
        }}

        // ========== Implementation identification ==========

        /// Returns a string identifying this type's implementation.
        ///
        /// This is useful for verifying that the correct implementation is being used
        /// at compile time (via `-Ctarget-cpu`) or at runtime (via `#[magetypes]` dispatch).
        ///
        /// Returns `"arm::neon::{name}"`.
        #[inline(always)]
        pub const fn implementation_name() -> &'static str {{
        "arm::neon::{name}"
        }}

    "#}
}

/// Generate math operations for NEON
fn generate_math_ops(ty: &SimdType) -> String {
    let mut code = String::new();

    // Min/Max - NEON doesn't have native 64-bit integer min/max, so we need polyfills
    let needs_polyfill = matches!(ty.elem, ElementType::I64 | ElementType::U64);

    if needs_polyfill {
        // Use comparison + select: min(a, b) = select(a < b, a, b)
        let (cmp_fn_min, bsl_fn) = match ty.elem {
            ElementType::I64 => ("vcltq_s64", "vbslq_s64"),
            ElementType::U64 => ("vcltq_u64", "vbslq_u64"),
            _ => unreachable!(),
        };
        let (cmp_fn_max, _) = match ty.elem {
            ElementType::I64 => ("vcgtq_s64", "vbslq_s64"),
            ElementType::U64 => ("vcgtq_u64", "vbslq_u64"),
            _ => unreachable!(),
        };

        code.push_str(&formatdoc! {r#"
            /// Element-wise minimum
            #[inline(always)]
            pub fn min(self, other: Self) -> Self {{
            // NEON lacks native 64-bit min, use compare+select
            let mask = unsafe {{ {cmp_fn_min}(self.0, other.0) }};
            Self(unsafe {{ {bsl_fn}(mask, self.0, other.0) }})
            }}

            /// Element-wise maximum
            #[inline(always)]
            pub fn max(self, other: Self) -> Self {{
            // NEON lacks native 64-bit max, use compare+select
            let mask = unsafe {{ {cmp_fn_max}(self.0, other.0) }};
            Self(unsafe {{ {bsl_fn}(mask, self.0, other.0) }})
            }}

        "#});
    } else {
        let min_fn = Arm::minmax_intrinsic("min", ty.elem);
        let max_fn = Arm::minmax_intrinsic("max", ty.elem);

        code.push_str(&formatdoc! {r#"
            /// Element-wise minimum
            #[inline(always)]
            pub fn min(self, other: Self) -> Self {{
            Self(unsafe {{ {min_fn}(self.0, other.0) }})
            }}

            /// Element-wise maximum
            #[inline(always)]
            pub fn max(self, other: Self) -> Self {{
            Self(unsafe {{ {max_fn}(self.0, other.0) }})
            }}

        "#});
    }

    code.push_str(&formatdoc! {r#"
        /// Clamp values between lo and hi
        #[inline(always)]
        pub fn clamp(self, lo: Self, hi: Self) -> Self {{
        self.max(lo).min(hi)
        }}

    "#});

    // Float-only operations
    if ty.elem.is_float() {
        let sqrt_fn = Arm::sqrt_intrinsic(ty.elem);
        let abs_fn = Arm::abs_intrinsic(ty.elem);
        let floor_fn = Arm::floor_intrinsic(ty.elem);
        let ceil_fn = Arm::ceil_intrinsic(ty.elem);
        let round_fn = Arm::round_intrinsic(ty.elem);
        let fma_fn = Arm::fma_intrinsic(ty.elem);
        let neg_fn = Arm::neg_intrinsic(ty.elem);

        code.push_str(&formatdoc! {r#"
            /// Square root
            #[inline(always)]
            pub fn sqrt(self) -> Self {{
            Self(unsafe {{ {sqrt_fn}(self.0) }})
            }}

            /// Absolute value
            #[inline(always)]
            pub fn abs(self) -> Self {{
            Self(unsafe {{ {abs_fn}(self.0) }})
            }}

            /// Floor
            #[inline(always)]
            pub fn floor(self) -> Self {{
            Self(unsafe {{ {floor_fn}(self.0) }})
            }}

            /// Ceil
            #[inline(always)]
            pub fn ceil(self) -> Self {{
            Self(unsafe {{ {ceil_fn}(self.0) }})
            }}

            /// Round to nearest
            #[inline(always)]
            pub fn round(self) -> Self {{
            Self(unsafe {{ {round_fn}(self.0) }})
            }}

            /// Fused multiply-add: self * a + b
            #[inline(always)]
            pub fn mul_add(self, a: Self, b: Self) -> Self {{
            Self(unsafe {{ {fma_fn}(b.0, self.0, a.0) }})
            }}

            /// Fused multiply-subtract: self * a - b
            #[inline(always)]
            pub fn mul_sub(self, a: Self, b: Self) -> Self {{
            let neg_b = unsafe {{ {neg_fn}(b.0) }};
            Self(unsafe {{ {fma_fn}(neg_b, self.0, a.0) }})
            }}

        "#});

        // Approximation operations (f32 only has native intrinsics)
        if ty.elem == ElementType::F32 {
            let recpe_fn = Arm::recpe_intrinsic(ty.elem);
            let rsqrte_fn = Arm::rsqrte_intrinsic(ty.elem);
            let splat_fn = Arm::splat_intrinsic(ty.elem);

            code.push_str(&formatdoc! {r#"
                // ========== Approximation Operations ==========

                /// Fast reciprocal approximation (1/x) with ~8-12 bit precision.
                ///
                /// For full precision, use `recip()` which applies Newton-Raphson refinement.
                #[inline(always)]
                pub fn rcp_approx(self) -> Self {{
                Self(unsafe {{ {recpe_fn}(self.0) }})
                }}

                /// Precise reciprocal (1/x) using Newton-Raphson refinement.
                ///
                /// More accurate than `rcp_approx()` but slower. For maximum speed
                /// with acceptable precision loss, use `rcp_approx()`.
                #[inline(always)]
                pub fn recip(self) -> Self {{
                // Newton-Raphson: x' = x * (2 - a*x)
                let approx = self.rcp_approx();
                let two = Self(unsafe {{ {splat_fn}(2.0) }});
                // One iteration gives ~24-bit precision from ~12-bit
                approx * (two - self * approx)
                }}

                /// Fast reciprocal square root approximation (1/sqrt(x)) with ~8-12 bit precision.
                ///
                /// For full precision, use `rsqrt()` which applies Newton-Raphson refinement.
                #[inline(always)]
                pub fn rsqrt_approx(self) -> Self {{
                Self(unsafe {{ {rsqrte_fn}(self.0) }})
                }}

                /// Precise reciprocal square root (1/sqrt(x)) using Newton-Raphson refinement.
                #[inline(always)]
                pub fn rsqrt(self) -> Self {{
                // Newton-Raphson for rsqrt: y' = 0.5 * y * (3 - x * y * y)
                let approx = self.rsqrt_approx();
                let half = Self(unsafe {{ {splat_fn}(0.5) }});
                let three = Self(unsafe {{ {splat_fn}(3.0) }});
                half * approx * (three - self * approx * approx)
                }}

            "#});
        }
    }

    // Signed integer abs
    if !ty.elem.is_float() && ty.elem.is_signed() {
        let abs_fn = Arm::abs_intrinsic(ty.elem);
        code.push_str(&formatdoc! {r#"
            /// Absolute value
            #[inline(always)]
            pub fn abs(self) -> Self {{
            Self(unsafe {{ {abs_fn}(self.0) }})
            }}

        "#});
    }

    code
}

/// Generate horizontal operations
fn generate_horizontal_ops(ty: &SimdType) -> String {
    let elem = ty.elem.name();
    let lanes = ty.lanes();
    let padd_fn = Arm::padd_intrinsic(ty.elem);
    let get_lane_fn = Arm::get_lane_intrinsic(ty.elem);

    let reduce_add_body = match lanes {
        2 => formatdoc! {r#"
            let sum = {padd_fn}(self.0, self.0);
            {get_lane_fn}::<0>(sum)"#},
        4 => formatdoc! {r#"
            let sum = {padd_fn}(self.0, self.0);
            let sum = {padd_fn}(sum, sum);
            {get_lane_fn}::<0>(sum)"#},
        8 => formatdoc! {r#"
            let sum = {padd_fn}(self.0, self.0);
            let sum = {padd_fn}(sum, sum);
            let sum = {padd_fn}(sum, sum);
            {get_lane_fn}::<0>(sum)"#},
        16 => formatdoc! {r#"
            let sum = {padd_fn}(self.0, self.0);
            let sum = {padd_fn}(sum, sum);
            let sum = {padd_fn}(sum, sum);
            let sum = {padd_fn}(sum, sum);
            {get_lane_fn}::<0>(sum)"#},
        _ => formatdoc! {r#"
            let arr = self.to_array();
            arr.iter().sum()"#},
    };

    let mut code = formatdoc! {r#"
        /// Reduce: sum all lanes
        #[inline(always)]
        pub fn reduce_add(self) -> {elem} {{
        unsafe {{
        {reduce_add_body}
        }}
        }}

    "#};

    // reduce_max/reduce_min for floats
    if ty.elem.is_float() {
        let pmax_fn = Arm::minmax_intrinsic("pmax", ty.elem);
        let pmin_fn = Arm::minmax_intrinsic("pmin", ty.elem);

        let reduce_max_body = match lanes {
            2 => formatdoc! {r#"
                let m = {pmax_fn}(self.0, self.0);
                {get_lane_fn}::<0>(m)"#},
            4 => formatdoc! {r#"
                let m = {pmax_fn}(self.0, self.0);
                let m = {pmax_fn}(m, m);
                {get_lane_fn}::<0>(m)"#},
            _ => formatdoc! {r#"
                let arr = self.to_array();
                arr.iter().copied().fold({elem}::NEG_INFINITY, {elem}::max)"#},
        };

        let reduce_min_body = match lanes {
            2 => formatdoc! {r#"
                let m = {pmin_fn}(self.0, self.0);
                {get_lane_fn}::<0>(m)"#},
            4 => formatdoc! {r#"
                let m = {pmin_fn}(self.0, self.0);
                let m = {pmin_fn}(m, m);
                {get_lane_fn}::<0>(m)"#},
            _ => formatdoc! {r#"
                let arr = self.to_array();
                arr.iter().copied().fold({elem}::INFINITY, {elem}::min)"#},
        };

        code.push_str(&formatdoc! {r#"
            /// Reduce: max of all lanes
            #[inline(always)]
            pub fn reduce_max(self) -> {elem} {{
            unsafe {{
            {reduce_max_body}
            }}
            }}

            /// Reduce: min of all lanes
            #[inline(always)]
            pub fn reduce_min(self) -> {elem} {{
            unsafe {{
            {reduce_min_body}
            }}
            }}

        "#});
    }

    code
}

/// Generate comparison operations for NEON
fn generate_comparison_ops(ty: &SimdType) -> String {
    let name = ty.name();

    let eq_fn = Arm::cmp_intrinsic("eq", ty.elem);
    let lt_fn = Arm::cmp_intrinsic("lt", ty.elem);
    let le_fn = Arm::cmp_intrinsic("le", ty.elem);
    let gt_fn = Arm::cmp_intrinsic("gt", ty.elem);
    let ge_fn = Arm::cmp_intrinsic("ge", ty.elem);

    // Get reinterpret function to cast from unsigned result back to our type
    let reinterpret_from = match ty.elem {
        ElementType::F32 => "vreinterpretq_f32_u32",
        ElementType::F64 => "vreinterpretq_f64_u64",
        ElementType::I8 => "vreinterpretq_s8_u8",
        ElementType::U8 => "",
        ElementType::I16 => "vreinterpretq_s16_u16",
        ElementType::U16 => "",
        ElementType::I32 => "vreinterpretq_s32_u32",
        ElementType::U32 => "",
        ElementType::I64 => "vreinterpretq_s64_u64",
        ElementType::U64 => "",
    };
    let needs_reinterpret = !reinterpret_from.is_empty();

    let cmp_body = |fn_name: &str| {
        if needs_reinterpret {
            format!("Self(unsafe {{ {reinterpret_from}({fn_name}(self.0, other.0)) }})")
        } else {
            format!("Self(unsafe {{ {fn_name}(self.0, other.0) }})")
        }
    };

    let eq_body = cmp_body(&eq_fn);
    let lt_body = cmp_body(&lt_fn);
    let le_body = cmp_body(&le_fn);
    let gt_body = cmp_body(&gt_fn);
    let ge_body = cmp_body(&ge_fn);

    // Get blend cast (for vbslq)
    let (mask_cast, bsl_fn) = match ty.elem {
        ElementType::F32 => ("vreinterpretq_u32_f32", "vbslq_f32"),
        ElementType::F64 => ("vreinterpretq_u64_f64", "vbslq_f64"),
        ElementType::I8 => ("vreinterpretq_u8_s8", "vbslq_s8"),
        ElementType::U8 => ("", "vbslq_u8"),
        ElementType::I16 => ("vreinterpretq_u16_s16", "vbslq_s16"),
        ElementType::U16 => ("", "vbslq_u16"),
        ElementType::I32 => ("vreinterpretq_u32_s32", "vbslq_s32"),
        ElementType::U32 => ("", "vbslq_u32"),
        ElementType::I64 => ("vreinterpretq_u64_s64", "vbslq_s64"),
        ElementType::U64 => ("", "vbslq_u64"),
    };

    let blend_body = if mask_cast.is_empty() {
        format!("Self(unsafe {{ {bsl_fn}(mask.0, if_true.0, if_false.0) }})")
    } else {
        format!("Self(unsafe {{ {bsl_fn}({mask_cast}(mask.0), if_true.0, if_false.0) }})")
    };

    formatdoc! {r#"
        // ========== Comparisons ==========
        // These return a mask where each lane is all-1s (true) or all-0s (false).
        // Use with `blend()` to select values based on the comparison result.

        /// Lane-wise equality comparison.
        ///
        /// Returns a mask where each lane is all-1s if equal, all-0s otherwise.
        /// Use with `blend(mask, if_true, if_false)` to select values.
        #[inline(always)]
        pub fn simd_eq(self, other: Self) -> Self {{
        {eq_body}
        }}

        /// Lane-wise inequality comparison.
        ///
        /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise.
        #[inline(always)]
        pub fn simd_ne(self, other: Self) -> Self {{
        self.simd_eq(other).not()
        }}

        /// Lane-wise less-than comparison.
        ///
        /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.
        #[inline(always)]
        pub fn simd_lt(self, other: Self) -> Self {{
        {lt_body}
        }}

        /// Lane-wise less-than-or-equal comparison.
        ///
        /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.
        #[inline(always)]
        pub fn simd_le(self, other: Self) -> Self {{
        {le_body}
        }}

        /// Lane-wise greater-than comparison.
        ///
        /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.
        #[inline(always)]
        pub fn simd_gt(self, other: Self) -> Self {{
        {gt_body}
        }}

        /// Lane-wise greater-than-or-equal comparison.
        ///
        /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.
        #[inline(always)]
        pub fn simd_ge(self, other: Self) -> Self {{
        {ge_body}
        }}

        // ========== Blending/Selection ==========

        /// Select lanes from `if_true` where mask is all-1s, `if_false` where mask is all-0s.
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
        {blend_body}
        }}

    "#}
}

/// Generate bitwise operations (not, shift) for NEON
fn generate_bitwise_ops(ty: &SimdType) -> String {
    let mut code = String::new();

    // not() - bitwise complement
    if ty.elem.is_float() {
        if ty.elem == ElementType::F64 {
            // NEON doesn't have vmvnq_u64, need to use EOR with all-ones
            code.push_str(&formatdoc! {r#"
                /// Bitwise NOT (complement)
                #[inline(always)]
                pub fn not(self) -> Self {{
                // NEON lacks vmvnq_u64, use XOR with all-ones
                unsafe {{
                let bits = vreinterpretq_u64_f64(self.0);
                let ones = vdupq_n_u64(u64::MAX);
                Self(vreinterpretq_f64_u64(veorq_u64(bits, ones)))
                }}
                }}

            "#});
        } else {
            // f32
            code.push_str(&formatdoc! {r#"
                /// Bitwise NOT (complement)
                #[inline(always)]
                pub fn not(self) -> Self {{
                Self(unsafe {{ vreinterpretq_f32_u32(vmvnq_u32(vreinterpretq_u32_f32(self.0))) }})
                }}

            "#});
        }
    } else {
        // For integers, use vmvnq directly or EOR for 64-bit
        let mvn_exists = matches!(
            ty.elem,
            ElementType::I8
                | ElementType::U8
                | ElementType::I16
                | ElementType::U16
                | ElementType::I32
                | ElementType::U32
        );

        if mvn_exists {
            let mvn_fn = match ty.elem {
                ElementType::I8 => "vmvnq_s8",
                ElementType::U8 => "vmvnq_u8",
                ElementType::I16 => "vmvnq_s16",
                ElementType::U16 => "vmvnq_u16",
                ElementType::I32 => "vmvnq_s32",
                ElementType::U32 => "vmvnq_u32",
                _ => unreachable!(),
            };
            code.push_str(&formatdoc! {r#"
                /// Bitwise NOT (complement)
                #[inline(always)]
                pub fn not(self) -> Self {{
                Self(unsafe {{ {mvn_fn}(self.0) }})
                }}

            "#});
        } else {
            // i64/u64: use EOR with all-ones
            let (eor_fn, dup_fn, max_val) = match ty.elem {
                ElementType::I64 => ("veorq_s64", "vdupq_n_s64", "-1i64"),
                ElementType::U64 => ("veorq_u64", "vdupq_n_u64", "u64::MAX"),
                _ => unreachable!(),
            };
            code.push_str(&formatdoc! {r#"
                /// Bitwise NOT (complement)
                #[inline(always)]
                pub fn not(self) -> Self {{
                unsafe {{
                let ones = {dup_fn}({max_val});
                Self({eor_fn}(self.0, ones))
                }}
                }}

            "#});
        }
    }

    // Shifts - only for integer types
    if !ty.elem.is_float() {
        let shl_fn = match ty.elem {
            ElementType::I8 => "vshlq_n_s8",
            ElementType::U8 => "vshlq_n_u8",
            ElementType::I16 => "vshlq_n_s16",
            ElementType::U16 => "vshlq_n_u16",
            ElementType::I32 => "vshlq_n_s32",
            ElementType::U32 => "vshlq_n_u32",
            ElementType::I64 => "vshlq_n_s64",
            ElementType::U64 => "vshlq_n_u64",
            _ => unreachable!(),
        };

        let shr_fn = match ty.elem {
            ElementType::U8 => "vshrq_n_u8",
            ElementType::U16 => "vshrq_n_u16",
            ElementType::U32 => "vshrq_n_u32",
            ElementType::U64 => "vshrq_n_u64",
            ElementType::I8 => "vshrq_n_s8",
            ElementType::I16 => "vshrq_n_s16",
            ElementType::I32 => "vshrq_n_s32",
            ElementType::I64 => "vshrq_n_s64",
            _ => unreachable!(),
        };

        code.push_str(&formatdoc! {r#"
            /// Shift left by immediate (const generic)
            #[inline(always)]
            pub fn shl<const N: i32>(self) -> Self {{
            Self(unsafe {{ {shl_fn}::<N>(self.0) }})
            }}

        "#});

        if ty.elem.is_signed() {
            code.push_str(&formatdoc! {r#"
                /// Shift right by immediate (const generic)
                ///
                /// For signed types, this is an arithmetic shift (sign-extending).
                #[inline(always)]
                pub fn shr<const N: i32>(self) -> Self {{
                Self(unsafe {{ {shr_fn}::<N>(self.0) }})
                }}

            "#});
        } else {
            code.push_str(&formatdoc! {r#"
                /// Shift right by immediate (const generic)
                ///
                /// For unsigned types, this is a logical shift (zero-extending).
                #[inline(always)]
                pub fn shr<const N: i32>(self) -> Self {{
                Self(unsafe {{ {shr_fn}::<N>(self.0) }})
                }}

            "#});
        }

        // shr_arithmetic for signed types (alias for shr on ARM since NEON shr is arithmetic)
        if ty.elem.is_signed() && !matches!(ty.elem, ElementType::I64) {
            code.push_str(&formatdoc! {r#"
                /// Arithmetic shift right by `N` bits (sign-extending).
                ///
                /// The sign bit is replicated into the vacated positions.
                /// On ARM NEON, this is the same as `shr()` for signed types.
                #[inline(always)]
                pub fn shr_arithmetic<const N: i32>(self) -> Self {{
                Self(unsafe {{ {shr_fn}::<N>(self.0) }})
                }}

            "#});
        }

        // Boolean reductions
        code.push_str(&generate_boolean_reductions(ty));
    }

    code
}

/// Generate boolean reduction operations for NEON integer types
fn generate_boolean_reductions(ty: &SimdType) -> String {
    let (u8_cast, u16_cast, u32_cast, u64_cast) = match ty.elem {
        ElementType::U8 | ElementType::U16 | ElementType::U32 | ElementType::U64 => {
            ("", "", "", "")
        }
        ElementType::I8 => ("vreinterpretq_u8_s8", "", "", ""),
        ElementType::I16 => ("", "vreinterpretq_u16_s16", "", ""),
        ElementType::I32 => ("", "", "vreinterpretq_u32_s32", ""),
        ElementType::I64 => ("", "", "", "vreinterpretq_u64_s64"),
        _ => unreachable!(),
    };

    let all_true_body = match ty.elem {
        ElementType::I8 => format!("vminvq_u8({u8_cast}(self.0)) != 0"),
        ElementType::U8 => "vminvq_u8(self.0) != 0".to_string(),
        ElementType::I16 => format!("vminvq_u16({u16_cast}(self.0)) != 0"),
        ElementType::U16 => "vminvq_u16(self.0) != 0".to_string(),
        ElementType::I32 => format!("vminvq_u32({u32_cast}(self.0)) != 0"),
        ElementType::U32 => "vminvq_u32(self.0) != 0".to_string(),
        ElementType::I64 => formatdoc! {r#"
            let as_u64 = {u64_cast}(self.0);
            vgetq_lane_u64::<0>(as_u64) != 0 && vgetq_lane_u64::<1>(as_u64) != 0"#},
        ElementType::U64 => {
            "vgetq_lane_u64::<0>(self.0) != 0 && vgetq_lane_u64::<1>(self.0) != 0".to_string()
        }
        _ => unreachable!(),
    };

    let any_true_body = match ty.elem {
        ElementType::I8 => format!("vmaxvq_u8({u8_cast}(self.0)) != 0"),
        ElementType::U8 => "vmaxvq_u8(self.0) != 0".to_string(),
        ElementType::I16 => format!("vmaxvq_u16({u16_cast}(self.0)) != 0"),
        ElementType::U16 => "vmaxvq_u16(self.0) != 0".to_string(),
        ElementType::I32 => format!("vmaxvq_u32({u32_cast}(self.0)) != 0"),
        ElementType::U32 => "vmaxvq_u32(self.0) != 0".to_string(),
        ElementType::I64 => formatdoc! {r#"
            let as_u64 = {u64_cast}(self.0);
            (vgetq_lane_u64::<0>(as_u64) | vgetq_lane_u64::<1>(as_u64)) != 0"#},
        ElementType::U64 => {
            "(vgetq_lane_u64::<0>(self.0) | vgetq_lane_u64::<1>(self.0)) != 0".to_string()
        }
        _ => unreachable!(),
    };

    let bitmask_body = match ty.elem {
        ElementType::I8 => formatdoc! {r#"
            let signs = vshrq_n_u8::<7>({u8_cast}(self.0));
            let arr: [u8; 16] = core::mem::transmute(signs);
            let mut r = 0u32;
            let mut i = 0;
            while i < 16 {{ r |= ((arr[i] & 1) as u32) << i; i += 1; }}
            r"#},
        ElementType::U8 => formatdoc! {r#"
            let signs = vshrq_n_u8::<7>(self.0);
            let arr: [u8; 16] = core::mem::transmute(signs);
            let mut r = 0u32;
            let mut i = 0;
            while i < 16 {{ r |= ((arr[i] & 1) as u32) << i; i += 1; }}
            r"#},
        ElementType::I16 => formatdoc! {r#"
            let signs = vshrq_n_u16::<15>({u16_cast}(self.0));
            let arr: [u16; 8] = core::mem::transmute(signs);
            let mut r = 0u32;
            let mut i = 0;
            while i < 8 {{ r |= ((arr[i] & 1) as u32) << i; i += 1; }}
            r"#},
        ElementType::U16 => formatdoc! {r#"
            let signs = vshrq_n_u16::<15>(self.0);
            let arr: [u16; 8] = core::mem::transmute(signs);
            let mut r = 0u32;
            let mut i = 0;
            while i < 8 {{ r |= ((arr[i] & 1) as u32) << i; i += 1; }}
            r"#},
        ElementType::I32 => formatdoc! {r#"
            let signs = vshrq_n_u32::<31>({u32_cast}(self.0));
            let arr: [u32; 4] = core::mem::transmute(signs);
            (arr[0] & 1) | ((arr[1] & 1) << 1) | ((arr[2] & 1) << 2) | ((arr[3] & 1) << 3)"#},
        ElementType::U32 => formatdoc! {r#"
            let signs = vshrq_n_u32::<31>(self.0);
            let arr: [u32; 4] = core::mem::transmute(signs);
            (arr[0] & 1) | ((arr[1] & 1) << 1) | ((arr[2] & 1) << 2) | ((arr[3] & 1) << 3)"#},
        ElementType::I64 => formatdoc! {r#"
            let signs = vshrq_n_u64::<63>({u64_cast}(self.0));
            ((vgetq_lane_u64::<0>(signs) & 1) | ((vgetq_lane_u64::<1>(signs) & 1) << 1)) as u32"#},
        ElementType::U64 => formatdoc! {r#"
            let signs = vshrq_n_u64::<63>(self.0);
            ((vgetq_lane_u64::<0>(signs) & 1) | ((vgetq_lane_u64::<1>(signs) & 1) << 1)) as u32"#},
        _ => unreachable!(),
    };

    formatdoc! {r#"
        // ========== Boolean Reductions ==========

        /// Returns true if all lanes are non-zero (truthy).
        ///
        /// Typically used with comparison results where true lanes are all-1s.
        #[inline(always)]
        pub fn all_true(self) -> bool {{
        unsafe {{ {all_true_body} }}
        }}

        /// Returns true if any lane is non-zero (truthy).
        #[inline(always)]
        pub fn any_true(self) -> bool {{
        unsafe {{ {any_true_body} }}
        }}

        /// Extract the high bit of each lane as a bitmask.
        ///
        /// Returns a u32 where bit N corresponds to the sign bit of lane N.
        #[inline(always)]
        pub fn bitmask(self) -> u32 {{
        unsafe {{
        {bitmask_body}
        }}
        }}

    "#}
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
            /// Values outside the representable range become `i32::MIN` (0x80000000).
            #[inline(always)]
            pub fn to_i32x4(self) -> i32x4 {{
            i32x4(unsafe {{ vcvtq_s32_f32(self.0) }})
            }}

            /// Convert to signed 32-bit integers, rounding to nearest even.
            ///
            /// Values outside the representable range become `i32::MIN` (0x80000000).
            #[inline(always)]
            pub fn to_i32x4_round(self) -> i32x4 {{
            i32x4(unsafe {{ vcvtnq_s32_f32(self.0) }})
            }}

            /// Create from signed 32-bit integers.
            #[inline(always)]
            pub fn from_i32x4(v: i32x4) -> Self {{
            Self(unsafe {{ vcvtq_f32_s32(v.0) }})
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
            f32x4(unsafe {{ vcvtq_f32_s32(self.0) }})
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
            /// Returns an `i32x4` where only the lower 2 lanes are valid.
            #[inline(always)]
            pub fn to_i32x4_low(self) -> i32x4 {{
            // NEON: f64->s64->s32 via vcvtq_s64_f64 + vmovn_s64
            let s64 = unsafe {{ vcvtq_s64_f64(self.0) }};
            let s32_low = unsafe {{ vmovn_s64(s64) }};
            i32x4(unsafe {{ vcombine_s32(s32_low, vdup_n_s32(0)) }})
            }}

        "#});
    }

    code
}

/// Generate operator trait implementations for NEON
fn generate_operator_impls(ty: &SimdType) -> String {
    let name = ty.name();
    let elem = ty.elem.name();
    let lanes = ty.lanes();
    let inner = Arm::intrinsic_type(ty.elem, ty.width);

    let add_fn = Arm::arith_intrinsic("add", ty.elem);
    let sub_fn = Arm::arith_intrinsic("sub", ty.elem);

    let mut code = formatdoc! {r#"
        impl core::ops::Add for {name} {{
        type Output = Self;
        #[inline(always)]
        fn add(self, rhs: Self) -> Self {{
            Self(unsafe {{ {add_fn}(self.0, rhs.0) }})
        }}
        }}

        impl core::ops::Sub for {name} {{
        type Output = Self;
        #[inline(always)]
        fn sub(self, rhs: Self) -> Self {{
            Self(unsafe {{ {sub_fn}(self.0, rhs.0) }})
        }}
        }}

    "#};

    // Mul (only for floats and some integers)
    let has_mul = ty.elem.is_float()
        || matches!(
            ty.elem,
            ElementType::I16 | ElementType::U16 | ElementType::I32 | ElementType::U32
        );
    if has_mul {
        let mul_fn = Arm::arith_intrinsic("mul", ty.elem);
        code.push_str(&formatdoc! {r#"
            impl core::ops::Mul for {name} {{
            type Output = Self;
            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {{
                Self(unsafe {{ {mul_fn}(self.0, rhs.0) }})
            }}
            }}

        "#});
    }

    // Div (floats only - NEON has vdivq)
    if ty.elem.is_float() {
        let div_fn = Arm::arith_intrinsic("div", ty.elem);
        code.push_str(&formatdoc! {r#"
            impl core::ops::Div for {name} {{
            type Output = Self;
            #[inline(always)]
            fn div(self, rhs: Self) -> Self {{
                Self(unsafe {{ {div_fn}(self.0, rhs.0) }})
            }}
            }}

        "#});
    }

    // Neg (floats and signed integers)
    if ty.elem.is_float() || ty.elem.is_signed() {
        let neg_fn = Arm::neg_intrinsic(ty.elem);
        code.push_str(&formatdoc! {r#"
            impl core::ops::Neg for {name} {{
            type Output = Self;
            #[inline(always)]
            fn neg(self) -> Self {{
                Self(unsafe {{ {neg_fn}(self.0) }})
            }}
            }}

        "#});
    }

    // Assign ops
    code.push_str(&formatdoc! {r#"
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

/// Generate the ARM w128.rs file
pub fn generate_arm_w128(types: &[SimdType]) -> String {
    let mut code = String::from(
        "//! 128-bit (NEON) SIMD types.\n//!\n//! **Auto-generated** by `cargo xtask generate` - do not edit manually.\n\nuse core::arch::aarch64::*;\n\n",
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

/// Get all NEON types to generate (128-bit only)
pub fn all_neon_types() -> Vec<SimdType> {
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
