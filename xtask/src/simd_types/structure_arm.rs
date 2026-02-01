//! ARM NEON type structure generation.
//!
//! Generates NEON SIMD types parallel to x86 types.

use super::arch::Arch;
use super::arch::arm::Arm;
use super::ops_bitcast;
use super::types::{ElementType, SimdType, SimdWidth};
use std::fmt::Write;

/// Generate a complete NEON SIMD type
pub fn generate_type(ty: &SimdType) -> String {
    assert!(
        ty.width == SimdWidth::W128,
        "NEON only supports 128-bit types"
    );

    let mut code = String::new();
    let name = ty.name();
    let lanes = ty.lanes();
    let elem = ty.elem.name();
    let inner = Arm::intrinsic_type(ty.elem, ty.width);

    // Type definition
    writeln!(
        code,
        "\n// ============================================================================"
    )
    .unwrap();
    writeln!(code, "// {} - {} x {} (128-bit NEON)", name, lanes, elem).unwrap();
    writeln!(
        code,
        "// ============================================================================\n"
    )
    .unwrap();

    writeln!(code, "#[derive(Clone, Copy, Debug)]").unwrap();
    writeln!(code, "#[repr(transparent)]").unwrap();
    writeln!(code, "pub struct {}({});\n", name, inner).unwrap();

    // Impl block
    writeln!(code, "impl {} {{", name).unwrap();
    writeln!(code, "    pub const LANES: usize = {};\n", lanes).unwrap();

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

    writeln!(code, "}}\n").unwrap();

    // Operator implementations
    code.push_str(&generate_operator_impls(ty));

    code
}

/// Generate construction and extraction methods
fn generate_construction_methods(ty: &SimdType) -> String {
    let mut code = String::new();
    let _name = ty.name();
    let lanes = ty.lanes();
    let elem = ty.elem.name();
    let inner = Arm::intrinsic_type(ty.elem, ty.width);

    let load_fn = Arm::load_intrinsic(ty.elem);
    let store_fn = Arm::store_intrinsic(ty.elem);
    let splat_fn = Arm::splat_intrinsic(ty.elem);

    // Load
    writeln!(code, "    /// Load from array (token-gated)").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(
        code,
        "    pub fn load(_: archmage::NeonToken, data: &[{}; {}]) -> Self {{",
        elem, lanes
    )
    .unwrap();
    writeln!(
        code,
        "        Self(unsafe {{ {}(data.as_ptr()) }})",
        load_fn
    )
    .unwrap();
    writeln!(code, "    }}\n").unwrap();

    // Splat
    writeln!(code, "    /// Broadcast scalar to all lanes (token-gated)").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(
        code,
        "    pub fn splat(_: archmage::NeonToken, v: {}) -> Self {{",
        elem
    )
    .unwrap();
    writeln!(code, "        Self(unsafe {{ {}(v) }})", splat_fn).unwrap();
    writeln!(code, "    }}\n").unwrap();

    // Zero
    writeln!(code, "    /// Zero vector (token-gated)").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn zero(_: archmage::NeonToken) -> Self {{").unwrap();
    let zero_val = ty.elem.zero_literal();
    writeln!(
        code,
        "        Self(unsafe {{ {}({}) }})",
        splat_fn, zero_val
    )
    .unwrap();
    writeln!(code, "    }}\n").unwrap();

    // From array (zero-cost transmute, no load instruction)
    writeln!(code, "    /// Create from array (token-gated, zero-cost)").unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// This is a zero-cost transmute, not a memory load."
    )
    .unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(
        code,
        "    pub fn from_array(_: archmage::NeonToken, arr: [{}; {}]) -> Self {{",
        elem, lanes
    )
    .unwrap();
    writeln!(
        code,
        "        // SAFETY: [{}; {}] and {} have identical size and layout",
        elem, lanes, inner
    )
    .unwrap();
    writeln!(code, "        Self(unsafe {{ core::mem::transmute(arr) }})").unwrap();
    writeln!(code, "    }}\n").unwrap();

    // Store
    writeln!(code, "    /// Store to array").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(
        code,
        "    pub fn store(self, out: &mut [{}; {}]) {{",
        elem, lanes
    )
    .unwrap();
    writeln!(
        code,
        "        unsafe {{ {}(out.as_mut_ptr(), self.0) }};",
        store_fn
    )
    .unwrap();
    writeln!(code, "    }}\n").unwrap();

    // To array
    writeln!(code, "    /// Convert to array").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(
        code,
        "    pub fn to_array(self) -> [{}; {}] {{",
        elem, lanes
    )
    .unwrap();
    writeln!(
        code,
        "        let mut out = [{}; {}];",
        ty.elem.zero_literal(),
        lanes
    )
    .unwrap();
    writeln!(code, "        self.store(&mut out);").unwrap();
    writeln!(code, "        out").unwrap();
    writeln!(code, "    }}\n").unwrap();

    // As array
    writeln!(code, "    /// Get reference to underlying array").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(
        code,
        "    pub fn as_array(&self) -> &[{}; {}] {{",
        elem, lanes
    )
    .unwrap();
    writeln!(
        code,
        "        unsafe {{ &*(self as *const Self as *const [{}; {}]) }}",
        elem, lanes
    )
    .unwrap();
    writeln!(code, "    }}\n").unwrap();

    // As array mut
    writeln!(code, "    /// Get mutable reference to underlying array").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(
        code,
        "    pub fn as_array_mut(&mut self) -> &mut [{}; {}] {{",
        elem, lanes
    )
    .unwrap();
    writeln!(
        code,
        "        unsafe {{ &mut *(self as *mut Self as *mut [{}; {}]) }}",
        elem, lanes
    )
    .unwrap();
    writeln!(code, "    }}\n").unwrap();

    // Raw
    writeln!(code, "    /// Get raw intrinsic type").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn raw(self) -> {} {{", inner).unwrap();
    writeln!(code, "        self.0").unwrap();
    writeln!(code, "    }}\n").unwrap();

    // From raw
    writeln!(
        code,
        "    /// Create from raw intrinsic (unsafe - no token check)"
    )
    .unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(code, "    /// # Safety").unwrap();
    writeln!(
        code,
        "    /// Caller must ensure the CPU supports the required SIMD features."
    )
    .unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub unsafe fn from_raw(v: {}) -> Self {{", inner).unwrap();
    writeln!(code, "        Self(v)").unwrap();
    writeln!(code, "    }}\n").unwrap();

    // Token-gated bytemuck replacements
    let byte_size = ty.lanes() * ty.elem.size_bytes();

    writeln!(
        code,
        "    // ========== Token-gated bytemuck replacements ==========\n"
    )
    .unwrap();

    // cast_slice: &[T] -> &[Self]
    writeln!(
        code,
        "    /// Reinterpret a slice of scalars as a slice of SIMD vectors (token-gated)."
    )
    .unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// Returns `None` if the slice length is not a multiple of {}, or",
        lanes
    )
    .unwrap();
    writeln!(code, "    /// if the slice is not properly aligned.").unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// This is a safe, token-gated replacement for `bytemuck::cast_slice`."
    )
    .unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(
        code,
        "    pub fn cast_slice(_: archmage::NeonToken, slice: &[{}]) -> Option<&[Self]> {{",
        elem
    )
    .unwrap();
    writeln!(code, "        if slice.len() % {} != 0 {{", lanes).unwrap();
    writeln!(code, "            return None;").unwrap();
    writeln!(code, "        }}").unwrap();
    writeln!(code, "        let ptr = slice.as_ptr();").unwrap();
    writeln!(
        code,
        "        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {{"
    )
    .unwrap();
    writeln!(code, "            return None;").unwrap();
    writeln!(code, "        }}").unwrap();
    writeln!(code, "        let len = slice.len() / {};", lanes).unwrap();
    writeln!(
        code,
        "        // SAFETY: alignment and length checked, layout is compatible"
    )
    .unwrap();
    writeln!(
        code,
        "        Some(unsafe {{ core::slice::from_raw_parts(ptr as *const Self, len) }})"
    )
    .unwrap();
    writeln!(code, "    }}\n").unwrap();

    // cast_slice_mut: &mut [T] -> &mut [Self]
    writeln!(
        code,
        "    /// Reinterpret a mutable slice of scalars as a slice of SIMD vectors (token-gated)."
    )
    .unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// Returns `None` if the slice length is not a multiple of {}, or",
        lanes
    )
    .unwrap();
    writeln!(code, "    /// if the slice is not properly aligned.").unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// This is a safe, token-gated replacement for `bytemuck::cast_slice_mut`."
    )
    .unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn cast_slice_mut(_: archmage::NeonToken, slice: &mut [{}]) -> Option<&mut [Self]> {{", elem).unwrap();
    writeln!(code, "        if slice.len() % {} != 0 {{", lanes).unwrap();
    writeln!(code, "            return None;").unwrap();
    writeln!(code, "        }}").unwrap();
    writeln!(code, "        let ptr = slice.as_mut_ptr();").unwrap();
    writeln!(
        code,
        "        if ptr.align_offset(core::mem::align_of::<Self>()) != 0 {{"
    )
    .unwrap();
    writeln!(code, "            return None;").unwrap();
    writeln!(code, "        }}").unwrap();
    writeln!(code, "        let len = slice.len() / {};", lanes).unwrap();
    writeln!(
        code,
        "        // SAFETY: alignment and length checked, layout is compatible"
    )
    .unwrap();
    writeln!(
        code,
        "        Some(unsafe {{ core::slice::from_raw_parts_mut(ptr as *mut Self, len) }})"
    )
    .unwrap();
    writeln!(code, "    }}\n").unwrap();

    // as_bytes: &self -> &[u8; N]
    writeln!(code, "    /// View this vector as a byte array.").unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// This is a safe replacement for `bytemuck::bytes_of`."
    )
    .unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(
        code,
        "    pub fn as_bytes(&self) -> &[u8; {}] {{",
        byte_size
    )
    .unwrap();
    writeln!(
        code,
        "        // SAFETY: Self is repr(transparent) over {} which is {} bytes",
        inner, byte_size
    )
    .unwrap();
    writeln!(
        code,
        "        unsafe {{ &*(self as *const Self as *const [u8; {}]) }}",
        byte_size
    )
    .unwrap();
    writeln!(code, "    }}\n").unwrap();

    // as_bytes_mut: &mut self -> &mut [u8; N]
    writeln!(code, "    /// View this vector as a mutable byte array.").unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// This is a safe replacement for `bytemuck::bytes_of_mut`."
    )
    .unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(
        code,
        "    pub fn as_bytes_mut(&mut self) -> &mut [u8; {}] {{",
        byte_size
    )
    .unwrap();
    writeln!(
        code,
        "        // SAFETY: Self is repr(transparent) over {} which is {} bytes",
        inner, byte_size
    )
    .unwrap();
    writeln!(
        code,
        "        unsafe {{ &mut *(self as *mut Self as *mut [u8; {}]) }}",
        byte_size
    )
    .unwrap();
    writeln!(code, "    }}\n").unwrap();

    // from_bytes: &[u8; N] -> Self (token-gated)
    writeln!(
        code,
        "    /// Create from a byte array reference (token-gated)."
    )
    .unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// This is a safe, token-gated replacement for `bytemuck::from_bytes`."
    )
    .unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(
        code,
        "    pub fn from_bytes(_: archmage::NeonToken, bytes: &[u8; {}]) -> Self {{",
        byte_size
    )
    .unwrap();
    writeln!(
        code,
        "        // SAFETY: [u8; {}] and Self have identical size",
        byte_size
    )
    .unwrap();
    writeln!(
        code,
        "        Self(unsafe {{ core::mem::transmute(*bytes) }})"
    )
    .unwrap();
    writeln!(code, "    }}\n").unwrap();

    // from_bytes_owned: [u8; N] -> Self (token-gated, zero-cost)
    writeln!(
        code,
        "    /// Create from an owned byte array (token-gated, zero-cost)."
    )
    .unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// This is a zero-cost transmute from an owned byte array."
    )
    .unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(
        code,
        "    pub fn from_bytes_owned(_: archmage::NeonToken, bytes: [u8; {}]) -> Self {{",
        byte_size
    )
    .unwrap();
    writeln!(
        code,
        "        // SAFETY: [u8; {}] and Self have identical size",
        byte_size
    )
    .unwrap();
    writeln!(
        code,
        "        Self(unsafe {{ core::mem::transmute(bytes) }})"
    )
    .unwrap();
    writeln!(code, "    }}\n").unwrap();

    code
}

/// Generate math operations for NEON
fn generate_math_ops(ty: &SimdType) -> String {
    use super::types::ElementType;

    let mut code = String::new();

    // Min/Max - NEON doesn't have native 64-bit integer min/max, so we need polyfills
    let needs_polyfill = matches!(ty.elem, ElementType::I64 | ElementType::U64);

    writeln!(code, "    /// Element-wise minimum").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn min(self, other: Self) -> Self {{").unwrap();

    if needs_polyfill {
        // Use comparison + select: min(a, b) = select(a < b, a, b)
        let cmp_fn = match ty.elem {
            ElementType::I64 => "vcltq_s64",
            ElementType::U64 => "vcltq_u64",
            _ => unreachable!(),
        };
        let bsl_fn = match ty.elem {
            ElementType::I64 => "vbslq_s64",
            ElementType::U64 => "vbslq_u64",
            _ => unreachable!(),
        };
        writeln!(
            code,
            "        // NEON lacks native 64-bit min, use compare+select"
        )
        .unwrap();
        writeln!(
            code,
            "        let mask = unsafe {{ {}(self.0, other.0) }};",
            cmp_fn
        )
        .unwrap();
        writeln!(
            code,
            "        Self(unsafe {{ {}(mask, self.0, other.0) }})",
            bsl_fn
        )
        .unwrap();
    } else {
        let min_fn = Arm::minmax_intrinsic("min", ty.elem);
        writeln!(
            code,
            "        Self(unsafe {{ {}(self.0, other.0) }})",
            min_fn
        )
        .unwrap();
    }
    writeln!(code, "    }}\n").unwrap();

    writeln!(code, "    /// Element-wise maximum").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn max(self, other: Self) -> Self {{").unwrap();

    if needs_polyfill {
        // Use comparison + select: max(a, b) = select(a > b, a, b)
        let cmp_fn = match ty.elem {
            ElementType::I64 => "vcgtq_s64",
            ElementType::U64 => "vcgtq_u64",
            _ => unreachable!(),
        };
        let bsl_fn = match ty.elem {
            ElementType::I64 => "vbslq_s64",
            ElementType::U64 => "vbslq_u64",
            _ => unreachable!(),
        };
        writeln!(
            code,
            "        // NEON lacks native 64-bit max, use compare+select"
        )
        .unwrap();
        writeln!(
            code,
            "        let mask = unsafe {{ {}(self.0, other.0) }};",
            cmp_fn
        )
        .unwrap();
        writeln!(
            code,
            "        Self(unsafe {{ {}(mask, self.0, other.0) }})",
            bsl_fn
        )
        .unwrap();
    } else {
        let max_fn = Arm::minmax_intrinsic("max", ty.elem);
        writeln!(
            code,
            "        Self(unsafe {{ {}(self.0, other.0) }})",
            max_fn
        )
        .unwrap();
    }
    writeln!(code, "    }}\n").unwrap();

    writeln!(code, "    /// Clamp values between lo and hi").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(
        code,
        "    pub fn clamp(self, lo: Self, hi: Self) -> Self {{"
    )
    .unwrap();
    writeln!(code, "        self.max(lo).min(hi)").unwrap();
    writeln!(code, "    }}\n").unwrap();

    // Float-only operations
    if ty.elem.is_float() {
        let sqrt_fn = Arm::sqrt_intrinsic(ty.elem);
        let abs_fn = Arm::abs_intrinsic(ty.elem);
        let _neg_fn = Arm::neg_intrinsic(ty.elem);
        let floor_fn = Arm::floor_intrinsic(ty.elem);
        let ceil_fn = Arm::ceil_intrinsic(ty.elem);
        let round_fn = Arm::round_intrinsic(ty.elem);
        let fma_fn = Arm::fma_intrinsic(ty.elem);

        writeln!(code, "    /// Square root").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn sqrt(self) -> Self {{").unwrap();
        writeln!(code, "        Self(unsafe {{ {}(self.0) }})", sqrt_fn).unwrap();
        writeln!(code, "    }}\n").unwrap();

        writeln!(code, "    /// Absolute value").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn abs(self) -> Self {{").unwrap();
        writeln!(code, "        Self(unsafe {{ {}(self.0) }})", abs_fn).unwrap();
        writeln!(code, "    }}\n").unwrap();

        writeln!(code, "    /// Floor").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn floor(self) -> Self {{").unwrap();
        writeln!(code, "        Self(unsafe {{ {}(self.0) }})", floor_fn).unwrap();
        writeln!(code, "    }}\n").unwrap();

        writeln!(code, "    /// Ceil").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn ceil(self) -> Self {{").unwrap();
        writeln!(code, "        Self(unsafe {{ {}(self.0) }})", ceil_fn).unwrap();
        writeln!(code, "    }}\n").unwrap();

        writeln!(code, "    /// Round to nearest").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn round(self) -> Self {{").unwrap();
        writeln!(code, "        Self(unsafe {{ {}(self.0) }})", round_fn).unwrap();
        writeln!(code, "    }}\n").unwrap();

        // FMA: NEON uses vfmaq_f32(a, b, c) = a + b*c
        // We want mul_add(self, a, b) = self * a + b
        // So: vfmaq(b, self, a)
        writeln!(code, "    /// Fused multiply-add: self * a + b").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(
            code,
            "    pub fn mul_add(self, a: Self, b: Self) -> Self {{"
        )
        .unwrap();
        writeln!(
            code,
            "        Self(unsafe {{ {}(b.0, self.0, a.0) }})",
            fma_fn
        )
        .unwrap();
        writeln!(code, "    }}\n").unwrap();

        // mul_sub: self * a - b
        // NEON vfmaq(a, b, c) = a + b*c, so we use vfmaq(-b, self, a) = -b + self*a = self*a - b
        let neg_fn = Arm::neg_intrinsic(ty.elem);
        writeln!(code, "    /// Fused multiply-subtract: self * a - b").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(
            code,
            "    pub fn mul_sub(self, a: Self, b: Self) -> Self {{"
        )
        .unwrap();
        writeln!(code, "        let neg_b = unsafe {{ {}(b.0) }};", neg_fn).unwrap();
        writeln!(
            code,
            "        Self(unsafe {{ {}(neg_b, self.0, a.0) }})",
            fma_fn
        )
        .unwrap();
        writeln!(code, "    }}\n").unwrap();

        // Approximation operations (f32 only has native intrinsics)
        if ty.elem == ElementType::F32 {
            let recpe_fn = Arm::recpe_intrinsic(ty.elem);
            let rsqrte_fn = Arm::rsqrte_intrinsic(ty.elem);
            let splat_fn = Arm::splat_intrinsic(ty.elem);

            writeln!(
                code,
                "    // ========== Approximation Operations ==========\n"
            )
            .unwrap();

            // rcp_approx
            writeln!(
                code,
                "    /// Fast reciprocal approximation (1/x) with ~8-12 bit precision."
            )
            .unwrap();
            writeln!(code, "    ///").unwrap();
            writeln!(
                code,
                "    /// For full precision, use `recip()` which applies Newton-Raphson refinement."
            )
            .unwrap();
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(code, "    pub fn rcp_approx(self) -> Self {{").unwrap();
            writeln!(code, "        Self(unsafe {{ {}(self.0) }})", recpe_fn).unwrap();
            writeln!(code, "    }}\n").unwrap();

            // recip - precise reciprocal via Newton-Raphson
            writeln!(
                code,
                "    /// Precise reciprocal (1/x) using Newton-Raphson refinement."
            )
            .unwrap();
            writeln!(code, "    ///").unwrap();
            writeln!(
                code,
                "    /// More accurate than `rcp_approx()` but slower. For maximum speed"
            )
            .unwrap();
            writeln!(
                code,
                "    /// with acceptable precision loss, use `rcp_approx()`."
            )
            .unwrap();
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(code, "    pub fn recip(self) -> Self {{").unwrap();
            writeln!(code, "        // Newton-Raphson: x' = x * (2 - a*x)").unwrap();
            writeln!(code, "        let approx = self.rcp_approx();").unwrap();
            writeln!(
                code,
                "        let two = Self(unsafe {{ {}(2.0) }});",
                splat_fn
            )
            .unwrap();
            writeln!(
                code,
                "        // One iteration gives ~24-bit precision from ~12-bit"
            )
            .unwrap();
            writeln!(code, "        approx * (two - self * approx)").unwrap();
            writeln!(code, "    }}\n").unwrap();

            // rsqrt_approx
            writeln!(
                code,
                "    /// Fast reciprocal square root approximation (1/sqrt(x)) with ~8-12 bit precision."
            )
            .unwrap();
            writeln!(code, "    ///").unwrap();
            writeln!(
                code,
                "    /// For full precision, use `rsqrt()` which applies Newton-Raphson refinement."
            )
            .unwrap();
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(code, "    pub fn rsqrt_approx(self) -> Self {{").unwrap();
            writeln!(code, "        Self(unsafe {{ {}(self.0) }})", rsqrte_fn).unwrap();
            writeln!(code, "    }}\n").unwrap();

            // rsqrt - precise via Newton-Raphson
            writeln!(
                code,
                "    /// Precise reciprocal square root (1/sqrt(x)) using Newton-Raphson refinement."
            )
            .unwrap();
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(code, "    pub fn rsqrt(self) -> Self {{").unwrap();
            writeln!(
                code,
                "        // Newton-Raphson for rsqrt: y' = 0.5 * y * (3 - x * y * y)"
            )
            .unwrap();
            writeln!(code, "        let approx = self.rsqrt_approx();").unwrap();
            writeln!(
                code,
                "        let half = Self(unsafe {{ {}(0.5) }});",
                splat_fn
            )
            .unwrap();
            writeln!(
                code,
                "        let three = Self(unsafe {{ {}(3.0) }});",
                splat_fn
            )
            .unwrap();
            writeln!(
                code,
                "        half * approx * (three - self * approx * approx)"
            )
            .unwrap();
            writeln!(code, "    }}\n").unwrap();
        }
    }

    // Signed integer abs
    if !ty.elem.is_float() && ty.elem.is_signed() {
        let abs_fn = Arm::abs_intrinsic(ty.elem);
        writeln!(code, "    /// Absolute value").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn abs(self) -> Self {{").unwrap();
        writeln!(code, "        Self(unsafe {{ {}(self.0) }})", abs_fn).unwrap();
        writeln!(code, "    }}\n").unwrap();
    }

    code
}

/// Generate horizontal operations
fn generate_horizontal_ops(ty: &SimdType) -> String {
    let mut code = String::new();
    let elem = ty.elem.name();
    let lanes = ty.lanes();

    // reduce_add using pairwise addition
    // For f32x4: vpaddq_f32 twice, then extract
    // Generate for all types (floats, signed, and unsigned)
    {
        let padd_fn = Arm::padd_intrinsic(ty.elem);
        let get_lane_fn = Arm::get_lane_intrinsic(ty.elem);

        writeln!(code, "    /// Reduce: sum all lanes").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn reduce_add(self) -> {} {{", elem).unwrap();
        writeln!(code, "        unsafe {{").unwrap();

        match lanes {
            2 => {
                // f64x2: one pairwise add
                writeln!(code, "            let sum = {}(self.0, self.0);", padd_fn).unwrap();
                writeln!(code, "            {}::<0>(sum)", get_lane_fn).unwrap();
            }
            4 => {
                // f32x4, i32x4: two pairwise adds
                writeln!(code, "            let sum = {}(self.0, self.0);", padd_fn).unwrap();
                writeln!(code, "            let sum = {}(sum, sum);", padd_fn).unwrap();
                writeln!(code, "            {}::<0>(sum)", get_lane_fn).unwrap();
            }
            8 => {
                // i16x8: three pairwise adds
                writeln!(code, "            let sum = {}(self.0, self.0);", padd_fn).unwrap();
                writeln!(code, "            let sum = {}(sum, sum);", padd_fn).unwrap();
                writeln!(code, "            let sum = {}(sum, sum);", padd_fn).unwrap();
                writeln!(code, "            {}::<0>(sum)", get_lane_fn).unwrap();
            }
            16 => {
                // i8x16: four pairwise adds
                writeln!(code, "            let sum = {}(self.0, self.0);", padd_fn).unwrap();
                writeln!(code, "            let sum = {}(sum, sum);", padd_fn).unwrap();
                writeln!(code, "            let sum = {}(sum, sum);", padd_fn).unwrap();
                writeln!(code, "            let sum = {}(sum, sum);", padd_fn).unwrap();
                writeln!(code, "            {}::<0>(sum)", get_lane_fn).unwrap();
            }
            _ => {
                // Fallback: scalar loop
                writeln!(code, "            let arr = self.to_array();").unwrap();
                writeln!(code, "            arr.iter().sum()").unwrap();
            }
        }

        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();
    }

    // reduce_max/reduce_min for floats
    if ty.elem.is_float() {
        let pmax_fn = Arm::minmax_intrinsic("pmax", ty.elem);
        let pmin_fn = Arm::minmax_intrinsic("pmin", ty.elem);
        let get_lane_fn = Arm::get_lane_intrinsic(ty.elem);

        writeln!(code, "    /// Reduce: max of all lanes").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn reduce_max(self) -> {} {{", elem).unwrap();
        writeln!(code, "        unsafe {{").unwrap();

        match lanes {
            2 => {
                writeln!(code, "            let m = {}(self.0, self.0);", pmax_fn).unwrap();
                writeln!(code, "            {}::<0>(m)", get_lane_fn).unwrap();
            }
            4 => {
                writeln!(code, "            let m = {}(self.0, self.0);", pmax_fn).unwrap();
                writeln!(code, "            let m = {}(m, m);", pmax_fn).unwrap();
                writeln!(code, "            {}::<0>(m)", get_lane_fn).unwrap();
            }
            _ => {
                writeln!(code, "            let arr = self.to_array();").unwrap();
                writeln!(
                    code,
                    "            arr.iter().copied().fold({}::NEG_INFINITY, {}::max)",
                    elem, elem
                )
                .unwrap();
            }
        }

        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        writeln!(code, "    /// Reduce: min of all lanes").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn reduce_min(self) -> {} {{", elem).unwrap();
        writeln!(code, "        unsafe {{").unwrap();

        match lanes {
            2 => {
                writeln!(code, "            let m = {}(self.0, self.0);", pmin_fn).unwrap();
                writeln!(code, "            {}::<0>(m)", get_lane_fn).unwrap();
            }
            4 => {
                writeln!(code, "            let m = {}(self.0, self.0);", pmin_fn).unwrap();
                writeln!(code, "            let m = {}(m, m);", pmin_fn).unwrap();
                writeln!(code, "            {}::<0>(m)", get_lane_fn).unwrap();
            }
            _ => {
                writeln!(code, "            let arr = self.to_array();").unwrap();
                writeln!(
                    code,
                    "            arr.iter().copied().fold({}::INFINITY, {}::min)",
                    elem, elem
                )
                .unwrap();
            }
        }

        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();
    }

    code
}

/// Generate comparison operations for NEON
fn generate_comparison_ops(ty: &SimdType) -> String {
    let mut code = String::new();
    let name = ty.name();

    writeln!(code, "    // ========== Comparisons ==========").unwrap();
    writeln!(
        code,
        "    // These return a mask where each lane is all-1s (true) or all-0s (false)."
    )
    .unwrap();
    writeln!(
        code,
        "    // Use with `blend()` to select values based on the comparison result.\n"
    )
    .unwrap();

    // NEON comparison intrinsics ALWAYS return unsigned integer vectors
    // vceqq_s8 -> uint8x16_t, vceqq_f32 -> uint32x4_t, etc.
    // We need to reinterpret back to the original type for all types

    let eq_fn = Arm::cmp_intrinsic("eq", ty.elem);

    // Get reinterpret function to cast from unsigned result back to our type
    let reinterpret_from = match ty.elem {
        ElementType::F32 => "vreinterpretq_f32_u32",
        ElementType::F64 => "vreinterpretq_f64_u64",
        ElementType::I8 => "vreinterpretq_s8_u8",
        ElementType::U8 => "", // already u8
        ElementType::I16 => "vreinterpretq_s16_u16",
        ElementType::U16 => "", // already u16
        ElementType::I32 => "vreinterpretq_s32_u32",
        ElementType::U32 => "", // already u32
        ElementType::I64 => "vreinterpretq_s64_u64",
        ElementType::U64 => "", // already u64
    };
    let needs_reinterpret = !reinterpret_from.is_empty();

    // simd_eq
    writeln!(code, "    /// Lane-wise equality comparison.").unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// Returns a mask where each lane is all-1s if equal, all-0s otherwise."
    )
    .unwrap();
    writeln!(
        code,
        "    /// Use with `blend(mask, if_true, if_false)` to select values."
    )
    .unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn simd_eq(self, other: Self) -> Self {{").unwrap();
    if needs_reinterpret {
        writeln!(
            code,
            "        Self(unsafe {{ {}({}(self.0, other.0)) }})",
            reinterpret_from, eq_fn
        )
        .unwrap();
    } else {
        writeln!(
            code,
            "        Self(unsafe {{ {}(self.0, other.0) }})",
            eq_fn
        )
        .unwrap();
    }
    writeln!(code, "    }}\n").unwrap();

    // simd_ne (use eq + not)
    writeln!(code, "    /// Lane-wise inequality comparison.").unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise."
    )
    .unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn simd_ne(self, other: Self) -> Self {{").unwrap();
    writeln!(code, "        self.simd_eq(other).not()").unwrap();
    writeln!(code, "    }}\n").unwrap();

    // simd_lt, simd_le, simd_gt, simd_ge
    let lt_fn = Arm::cmp_intrinsic("lt", ty.elem);
    let le_fn = Arm::cmp_intrinsic("le", ty.elem);
    let gt_fn = Arm::cmp_intrinsic("gt", ty.elem);
    let ge_fn = Arm::cmp_intrinsic("ge", ty.elem);

    // simd_lt
    writeln!(code, "    /// Lane-wise less-than comparison.").unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise."
    )
    .unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn simd_lt(self, other: Self) -> Self {{").unwrap();
    if needs_reinterpret {
        writeln!(
            code,
            "        Self(unsafe {{ {}({}(self.0, other.0)) }})",
            reinterpret_from, lt_fn
        )
        .unwrap();
    } else {
        writeln!(
            code,
            "        Self(unsafe {{ {}(self.0, other.0) }})",
            lt_fn
        )
        .unwrap();
    }
    writeln!(code, "    }}\n").unwrap();

    // simd_le
    writeln!(code, "    /// Lane-wise less-than-or-equal comparison.").unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise."
    )
    .unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn simd_le(self, other: Self) -> Self {{").unwrap();
    if needs_reinterpret {
        writeln!(
            code,
            "        Self(unsafe {{ {}({}(self.0, other.0)) }})",
            reinterpret_from, le_fn
        )
        .unwrap();
    } else {
        writeln!(
            code,
            "        Self(unsafe {{ {}(self.0, other.0) }})",
            le_fn
        )
        .unwrap();
    }
    writeln!(code, "    }}\n").unwrap();

    // simd_gt
    writeln!(code, "    /// Lane-wise greater-than comparison.").unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise."
    )
    .unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn simd_gt(self, other: Self) -> Self {{").unwrap();
    if needs_reinterpret {
        writeln!(
            code,
            "        Self(unsafe {{ {}({}(self.0, other.0)) }})",
            reinterpret_from, gt_fn
        )
        .unwrap();
    } else {
        writeln!(
            code,
            "        Self(unsafe {{ {}(self.0, other.0) }})",
            gt_fn
        )
        .unwrap();
    }
    writeln!(code, "    }}\n").unwrap();

    // simd_ge
    writeln!(code, "    /// Lane-wise greater-than-or-equal comparison.").unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise."
    )
    .unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn simd_ge(self, other: Self) -> Self {{").unwrap();
    if needs_reinterpret {
        writeln!(
            code,
            "        Self(unsafe {{ {}({}(self.0, other.0)) }})",
            reinterpret_from, ge_fn
        )
        .unwrap();
    } else {
        writeln!(
            code,
            "        Self(unsafe {{ {}(self.0, other.0) }})",
            ge_fn
        )
        .unwrap();
    }
    writeln!(code, "    }}\n").unwrap();

    // blend - conditional select using mask
    // NEON vbslq: vbslq_*(mask_as_uint, if_true, if_false)
    writeln!(code, "    // ========== Blending/Selection ==========\n").unwrap();

    writeln!(
        code,
        "    /// Select lanes from `if_true` where mask is all-1s, `if_false` where mask is all-0s."
    )
    .unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// The mask should come from a comparison operation like `simd_lt()`."
    )
    .unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(code, "    /// # Example").unwrap();
    writeln!(code, "    /// ```ignore").unwrap();
    writeln!(code, "    /// let a = {}::splat(token, 1.0);", name).unwrap();
    writeln!(code, "    /// let b = {}::splat(token, 2.0);", name).unwrap();
    writeln!(code, "    /// let mask = a.simd_lt(b);  // all true").unwrap();
    writeln!(
        code,
        "    /// let result = {}::blend(mask, a, b);  // selects a",
        name
    )
    .unwrap();
    writeln!(code, "    /// ```").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(
        code,
        "    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {{"
    )
    .unwrap();

    // For NEON vbslq, the mask must be the unsigned type
    // vbslq_f32(uint32x4_t mask, float32x4_t a, float32x4_t b)
    // vbslq_s8(uint8x16_t mask, int8x16_t a, int8x16_t b)
    let (mask_cast, bsl_fn) = match ty.elem {
        ElementType::F32 => ("vreinterpretq_u32_f32", "vbslq_f32"),
        ElementType::F64 => ("vreinterpretq_u64_f64", "vbslq_f64"),
        ElementType::I8 => ("vreinterpretq_u8_s8", "vbslq_s8"),
        ElementType::U8 => ("", "vbslq_u8"), // already unsigned
        ElementType::I16 => ("vreinterpretq_u16_s16", "vbslq_s16"),
        ElementType::U16 => ("", "vbslq_u16"),
        ElementType::I32 => ("vreinterpretq_u32_s32", "vbslq_s32"),
        ElementType::U32 => ("", "vbslq_u32"),
        ElementType::I64 => ("vreinterpretq_u64_s64", "vbslq_s64"),
        ElementType::U64 => ("", "vbslq_u64"),
    };

    if mask_cast.is_empty() {
        writeln!(
            code,
            "        Self(unsafe {{ {}(mask.0, if_true.0, if_false.0) }})",
            bsl_fn
        )
        .unwrap();
    } else {
        writeln!(
            code,
            "        Self(unsafe {{ {}({}(mask.0), if_true.0, if_false.0) }})",
            bsl_fn, mask_cast
        )
        .unwrap();
    }
    writeln!(code, "    }}\n").unwrap();

    code
}

/// Generate bitwise operations (not, shift) for NEON
fn generate_bitwise_ops(ty: &SimdType) -> String {
    let mut code = String::new();

    // not() - bitwise complement
    // For integers: vmvnq_*
    // For floats: reinterpret to uint, vmvnq, reinterpret back
    writeln!(code, "    /// Bitwise NOT (complement)").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn not(self) -> Self {{").unwrap();

    if ty.elem.is_float() {
        let (to_uint, mvn_fn, from_uint) = match ty.elem {
            ElementType::F32 => (
                "vreinterpretq_u32_f32",
                "vmvnq_u32",
                "vreinterpretq_f32_u32",
            ),
            ElementType::F64 => {
                // NEON doesn't have vmvnq_u64, need to use EOR with all-ones
                // Use veorq with all-ones vector
                writeln!(
                    code,
                    "        // NEON lacks vmvnq_u64, use XOR with all-ones"
                )
                .unwrap();
                writeln!(code, "        unsafe {{").unwrap();
                writeln!(
                    code,
                    "            let bits = vreinterpretq_u64_f64(self.0);"
                )
                .unwrap();
                writeln!(code, "            let ones = vdupq_n_u64(u64::MAX);").unwrap();
                writeln!(
                    code,
                    "            Self(vreinterpretq_f64_u64(veorq_u64(bits, ones)))"
                )
                .unwrap();
                writeln!(code, "        }}").unwrap();
                writeln!(code, "    }}\n").unwrap();
                return code;
            }
            _ => unreachable!(),
        };
        writeln!(
            code,
            "        Self(unsafe {{ {}({}({}(self.0))) }})",
            from_uint, mvn_fn, to_uint
        )
        .unwrap();
    } else {
        // For integers, use vmvnq directly
        // Note: vmvnq only exists for 8/16/32-bit, not 64-bit
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
            writeln!(code, "        Self(unsafe {{ {}(self.0) }})", mvn_fn).unwrap();
        } else {
            // i64/u64: use EOR with all-ones
            let (eor_fn, dup_fn, max_val) = match ty.elem {
                ElementType::I64 => ("veorq_s64", "vdupq_n_s64", "-1i64"),
                ElementType::U64 => ("veorq_u64", "vdupq_n_u64", "u64::MAX"),
                _ => unreachable!(),
            };
            writeln!(code, "        unsafe {{").unwrap();
            writeln!(code, "            let ones = {}({});", dup_fn, max_val).unwrap();
            writeln!(code, "            Self({}(self.0, ones))", eor_fn).unwrap();
            writeln!(code, "        }}").unwrap();
        }
    }
    writeln!(code, "    }}\n").unwrap();

    // Shifts - only for integer types
    if !ty.elem.is_float() {
        // shl - shift left by immediate
        // NEON: vshlq_n_* for immediate shifts
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

        writeln!(code, "    /// Shift left by immediate (const generic)").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn shl<const N: i32>(self) -> Self {{").unwrap();
        writeln!(code, "        Self(unsafe {{ {}::<N>(self.0) }})", shl_fn).unwrap();
        writeln!(code, "    }}\n").unwrap();

        // shr - shift right (logical for unsigned, arithmetic for signed)
        let shr_fn = match ty.elem {
            // Unsigned: logical right shift
            ElementType::U8 => "vshrq_n_u8",
            ElementType::U16 => "vshrq_n_u16",
            ElementType::U32 => "vshrq_n_u32",
            ElementType::U64 => "vshrq_n_u64",
            // Signed: arithmetic right shift
            ElementType::I8 => "vshrq_n_s8",
            ElementType::I16 => "vshrq_n_s16",
            ElementType::I32 => "vshrq_n_s32",
            ElementType::I64 => "vshrq_n_s64",
            _ => unreachable!(),
        };

        writeln!(code, "    /// Shift right by immediate (const generic)").unwrap();
        writeln!(code, "    ///").unwrap();
        if ty.elem.is_signed() {
            writeln!(
                code,
                "    /// For signed types, this is an arithmetic shift (sign-extending)."
            )
            .unwrap();
        } else {
            writeln!(
                code,
                "    /// For unsigned types, this is a logical shift (zero-extending)."
            )
            .unwrap();
        }
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn shr<const N: i32>(self) -> Self {{").unwrap();
        writeln!(code, "        Self(unsafe {{ {}::<N>(self.0) }})", shr_fn).unwrap();
        writeln!(code, "    }}\n").unwrap();

        // shr_arithmetic for signed types (alias for shr on ARM since NEON shr is arithmetic)
        if ty.elem.is_signed() && !matches!(ty.elem, ElementType::I64) {
            writeln!(
                code,
                "    /// Arithmetic shift right by `N` bits (sign-extending)."
            )
            .unwrap();
            writeln!(code, "    ///").unwrap();
            writeln!(
                code,
                "    /// The sign bit is replicated into the vacated positions."
            )
            .unwrap();
            writeln!(
                code,
                "    /// On ARM NEON, this is the same as `shr()` for signed types."
            )
            .unwrap();
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(
                code,
                "    pub fn shr_arithmetic<const N: i32>(self) -> Self {{"
            )
            .unwrap();
            writeln!(code, "        Self(unsafe {{ {}::<N>(self.0) }})", shr_fn).unwrap();
            writeln!(code, "    }}\n").unwrap();
        }

        // Boolean reductions - only for integer types
        code.push_str(&generate_boolean_reductions(ty));
    }

    code
}

/// Generate boolean reduction operations for NEON integer types
fn generate_boolean_reductions(ty: &SimdType) -> String {
    let mut code = String::new();

    writeln!(code, "    // ========== Boolean Reductions ==========\n").unwrap();

    // Helper to get reinterpret function (empty string for types that don't need it)
    let (u8_cast, u16_cast, u32_cast, u64_cast) = match ty.elem {
        // Unsigned types don't need casting
        ElementType::U8 => ("", "", "", ""),
        ElementType::U16 => ("", "", "", ""),
        ElementType::U32 => ("", "", "", ""),
        ElementType::U64 => ("", "", "", ""),
        // Signed types need reinterpret to unsigned
        ElementType::I8 => ("vreinterpretq_u8_s8", "", "", ""),
        ElementType::I16 => ("", "vreinterpretq_u16_s16", "", ""),
        ElementType::I32 => ("", "", "vreinterpretq_u32_s32", ""),
        ElementType::I64 => ("", "", "", "vreinterpretq_u64_s64"),
        _ => unreachable!(),
    };

    // all_true - check if all lanes are non-zero
    writeln!(
        code,
        "    /// Returns true if all lanes are non-zero (truthy)."
    )
    .unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// Typically used with comparison results where true lanes are all-1s."
    )
    .unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn all_true(self) -> bool {{").unwrap();

    // Strategy: use horizontal min - if min is non-zero, all lanes are non-zero
    match ty.elem {
        ElementType::I8 => {
            writeln!(
                code,
                "        unsafe {{ vminvq_u8({}(self.0)) != 0 }}",
                u8_cast
            )
            .unwrap();
        }
        ElementType::U8 => {
            writeln!(code, "        unsafe {{ vminvq_u8(self.0) != 0 }}").unwrap();
        }
        ElementType::I16 => {
            writeln!(
                code,
                "        unsafe {{ vminvq_u16({}(self.0)) != 0 }}",
                u16_cast
            )
            .unwrap();
        }
        ElementType::U16 => {
            writeln!(code, "        unsafe {{ vminvq_u16(self.0) != 0 }}").unwrap();
        }
        ElementType::I32 => {
            writeln!(
                code,
                "        unsafe {{ vminvq_u32({}(self.0)) != 0 }}",
                u32_cast
            )
            .unwrap();
        }
        ElementType::U32 => {
            writeln!(code, "        unsafe {{ vminvq_u32(self.0) != 0 }}").unwrap();
        }
        ElementType::I64 => {
            // No vminvq_u64, check both lanes manually
            writeln!(code, "        unsafe {{").unwrap();
            writeln!(code, "            let as_u64 = {}(self.0);", u64_cast).unwrap();
            writeln!(
                code,
                "            vgetq_lane_u64::<0>(as_u64) != 0 && vgetq_lane_u64::<1>(as_u64) != 0"
            )
            .unwrap();
            writeln!(code, "        }}").unwrap();
        }
        ElementType::U64 => {
            writeln!(code, "        unsafe {{").unwrap();
            writeln!(
                code,
                "            vgetq_lane_u64::<0>(self.0) != 0 && vgetq_lane_u64::<1>(self.0) != 0"
            )
            .unwrap();
            writeln!(code, "        }}").unwrap();
        }
        _ => unreachable!(),
    }
    writeln!(code, "    }}\n").unwrap();

    // any_true - check if any lane is non-zero
    writeln!(
        code,
        "    /// Returns true if any lane is non-zero (truthy)."
    )
    .unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn any_true(self) -> bool {{").unwrap();

    // Strategy: use horizontal max - if max is non-zero, at least one lane is non-zero
    match ty.elem {
        ElementType::I8 => {
            writeln!(
                code,
                "        unsafe {{ vmaxvq_u8({}(self.0)) != 0 }}",
                u8_cast
            )
            .unwrap();
        }
        ElementType::U8 => {
            writeln!(code, "        unsafe {{ vmaxvq_u8(self.0) != 0 }}").unwrap();
        }
        ElementType::I16 => {
            writeln!(
                code,
                "        unsafe {{ vmaxvq_u16({}(self.0)) != 0 }}",
                u16_cast
            )
            .unwrap();
        }
        ElementType::U16 => {
            writeln!(code, "        unsafe {{ vmaxvq_u16(self.0) != 0 }}").unwrap();
        }
        ElementType::I32 => {
            writeln!(
                code,
                "        unsafe {{ vmaxvq_u32({}(self.0)) != 0 }}",
                u32_cast
            )
            .unwrap();
        }
        ElementType::U32 => {
            writeln!(code, "        unsafe {{ vmaxvq_u32(self.0) != 0 }}").unwrap();
        }
        ElementType::I64 => {
            writeln!(code, "        unsafe {{").unwrap();
            writeln!(code, "            let as_u64 = {}(self.0);", u64_cast).unwrap();
            writeln!(
                code,
                "            (vgetq_lane_u64::<0>(as_u64) | vgetq_lane_u64::<1>(as_u64)) != 0"
            )
            .unwrap();
            writeln!(code, "        }}").unwrap();
        }
        ElementType::U64 => {
            writeln!(code, "        unsafe {{").unwrap();
            writeln!(
                code,
                "            (vgetq_lane_u64::<0>(self.0) | vgetq_lane_u64::<1>(self.0)) != 0"
            )
            .unwrap();
            writeln!(code, "        }}").unwrap();
        }
        _ => unreachable!(),
    }
    writeln!(code, "    }}\n").unwrap();

    // bitmask - extract high bit of each lane
    writeln!(
        code,
        "    /// Extract the high bit of each lane as a bitmask."
    )
    .unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// Returns a u32 where bit N corresponds to the sign bit of lane N."
    )
    .unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn bitmask(self) -> u32 {{").unwrap();

    // NEON doesn't have a direct movemask, need to emulate
    // Strategy: shift right to get sign bit in LSB, then collect via transmute
    match ty.elem {
        ElementType::I8 => {
            writeln!(code, "        unsafe {{").unwrap();
            writeln!(
                code,
                "            let signs = vshrq_n_u8::<7>({}(self.0));",
                u8_cast
            )
            .unwrap();
            writeln!(
                code,
                "            let arr: [u8; 16] = core::mem::transmute(signs);"
            )
            .unwrap();
            writeln!(code, "            let mut r = 0u32;").unwrap();
            writeln!(code, "            let mut i = 0;").unwrap();
            writeln!(
                code,
                "            while i < 16 {{ r |= ((arr[i] & 1) as u32) << i; i += 1; }}"
            )
            .unwrap();
            writeln!(code, "            r").unwrap();
            writeln!(code, "        }}").unwrap();
        }
        ElementType::U8 => {
            writeln!(code, "        unsafe {{").unwrap();
            writeln!(code, "            let signs = vshrq_n_u8::<7>(self.0);").unwrap();
            writeln!(
                code,
                "            let arr: [u8; 16] = core::mem::transmute(signs);"
            )
            .unwrap();
            writeln!(code, "            let mut r = 0u32;").unwrap();
            writeln!(code, "            let mut i = 0;").unwrap();
            writeln!(
                code,
                "            while i < 16 {{ r |= ((arr[i] & 1) as u32) << i; i += 1; }}"
            )
            .unwrap();
            writeln!(code, "            r").unwrap();
            writeln!(code, "        }}").unwrap();
        }
        ElementType::I16 => {
            writeln!(code, "        unsafe {{").unwrap();
            writeln!(
                code,
                "            let signs = vshrq_n_u16::<15>({}(self.0));",
                u16_cast
            )
            .unwrap();
            writeln!(
                code,
                "            let arr: [u16; 8] = core::mem::transmute(signs);"
            )
            .unwrap();
            writeln!(code, "            let mut r = 0u32;").unwrap();
            writeln!(code, "            let mut i = 0;").unwrap();
            writeln!(
                code,
                "            while i < 8 {{ r |= ((arr[i] & 1) as u32) << i; i += 1; }}"
            )
            .unwrap();
            writeln!(code, "            r").unwrap();
            writeln!(code, "        }}").unwrap();
        }
        ElementType::U16 => {
            writeln!(code, "        unsafe {{").unwrap();
            writeln!(code, "            let signs = vshrq_n_u16::<15>(self.0);").unwrap();
            writeln!(
                code,
                "            let arr: [u16; 8] = core::mem::transmute(signs);"
            )
            .unwrap();
            writeln!(code, "            let mut r = 0u32;").unwrap();
            writeln!(code, "            let mut i = 0;").unwrap();
            writeln!(
                code,
                "            while i < 8 {{ r |= ((arr[i] & 1) as u32) << i; i += 1; }}"
            )
            .unwrap();
            writeln!(code, "            r").unwrap();
            writeln!(code, "        }}").unwrap();
        }
        ElementType::I32 => {
            writeln!(code, "        unsafe {{").unwrap();
            writeln!(
                code,
                "            let signs = vshrq_n_u32::<31>({}(self.0));",
                u32_cast
            )
            .unwrap();
            writeln!(
                code,
                "            let arr: [u32; 4] = core::mem::transmute(signs);"
            )
            .unwrap();
            writeln!(code, "            (arr[0] & 1) | ((arr[1] & 1) << 1) | ((arr[2] & 1) << 2) | ((arr[3] & 1) << 3)").unwrap();
            writeln!(code, "        }}").unwrap();
        }
        ElementType::U32 => {
            writeln!(code, "        unsafe {{").unwrap();
            writeln!(code, "            let signs = vshrq_n_u32::<31>(self.0);").unwrap();
            writeln!(
                code,
                "            let arr: [u32; 4] = core::mem::transmute(signs);"
            )
            .unwrap();
            writeln!(code, "            (arr[0] & 1) | ((arr[1] & 1) << 1) | ((arr[2] & 1) << 2) | ((arr[3] & 1) << 3)").unwrap();
            writeln!(code, "        }}").unwrap();
        }
        ElementType::I64 => {
            writeln!(code, "        unsafe {{").unwrap();
            writeln!(
                code,
                "            let signs = vshrq_n_u64::<63>({}(self.0));",
                u64_cast
            )
            .unwrap();
            writeln!(code, "            ((vgetq_lane_u64::<0>(signs) & 1) | ((vgetq_lane_u64::<1>(signs) & 1) << 1)) as u32").unwrap();
            writeln!(code, "        }}").unwrap();
        }
        ElementType::U64 => {
            writeln!(code, "        unsafe {{").unwrap();
            writeln!(code, "            let signs = vshrq_n_u64::<63>(self.0);").unwrap();
            writeln!(code, "            ((vgetq_lane_u64::<0>(signs) & 1) | ((vgetq_lane_u64::<1>(signs) & 1) << 1)) as u32").unwrap();
            writeln!(code, "        }}").unwrap();
        }
        _ => unreachable!(),
    }
    writeln!(code, "    }}\n").unwrap();

    code
}

/// Generate type conversion operations (f32 <-> i32, etc.)
fn generate_conversion_ops(ty: &SimdType) -> String {
    use super::types::ElementType;

    let mut code = String::new();
    let lanes = ty.lanes();

    // f32 <-> i32 conversions
    if ty.elem == ElementType::F32 && lanes == 4 {
        let int_name = "i32x4";

        writeln!(code, "    // ========== Type Conversions ==========\n").unwrap();

        // to_i32x4 (truncate toward zero) - vcvtq_s32_f32
        writeln!(
            code,
            "    /// Convert to signed 32-bit integers, rounding toward zero (truncation)."
        )
        .unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Values outside the representable range become `i32::MIN` (0x80000000)."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn to_i32x4(self) -> {} {{", int_name).unwrap();
        writeln!(
            code,
            "        {}(unsafe {{ vcvtq_s32_f32(self.0) }})",
            int_name
        )
        .unwrap();
        writeln!(code, "    }}\n").unwrap();

        // to_i32x4_round (round to nearest) - vcvtnq_s32_f32
        writeln!(
            code,
            "    /// Convert to signed 32-bit integers, rounding to nearest even."
        )
        .unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Values outside the representable range become `i32::MIN` (0x80000000)."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn to_i32x4_round(self) -> {} {{", int_name).unwrap();
        writeln!(
            code,
            "        {}(unsafe {{ vcvtnq_s32_f32(self.0) }})",
            int_name
        )
        .unwrap();
        writeln!(code, "    }}\n").unwrap();

        // from_i32x4 - vcvtq_f32_s32
        writeln!(code, "    /// Create from signed 32-bit integers.").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn from_i32x4(v: {}) -> Self {{", int_name).unwrap();
        writeln!(code, "        Self(unsafe {{ vcvtq_f32_s32(v.0) }})").unwrap();
        writeln!(code, "    }}\n").unwrap();
    }

    // i32 -> f32 conversion
    if ty.elem == ElementType::I32 && lanes == 4 {
        let float_name = "f32x4";

        writeln!(code, "    // ========== Type Conversions ==========\n").unwrap();

        // to_f32x4 - vcvtq_f32_s32
        writeln!(code, "    /// Convert to single-precision floats.").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn to_f32x4(self) -> {} {{", float_name).unwrap();
        writeln!(
            code,
            "        {}(unsafe {{ vcvtq_f32_s32(self.0) }})",
            float_name
        )
        .unwrap();
        writeln!(code, "    }}\n").unwrap();

        // to_f32 (alias for to_f32x4)
        writeln!(
            code,
            "    /// Convert to single-precision floats (alias for `to_f32x4`)."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn to_f32(self) -> {} {{", float_name).unwrap();
        writeln!(code, "        self.to_f32x4()").unwrap();
        writeln!(code, "    }}\n").unwrap();
    }

    // f64 -> i32 conversion (2 lanes -> lower 2 lanes of i32x4)
    if ty.elem == ElementType::F64 && lanes == 2 {
        writeln!(code, "    // ========== Type Conversions ==========\n").unwrap();

        writeln!(
            code,
            "    /// Convert to signed 32-bit integers (2 lanes), rounding toward zero."
        )
        .unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Returns an `i32x4` where only the lower 2 lanes are valid."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn to_i32x4_low(self) -> i32x4 {{").unwrap();
        writeln!(
            code,
            "        // NEON: f64->s64->s32 via vcvtq_s64_f64 + vmovn_s64"
        )
        .unwrap();
        writeln!(
            code,
            "        let s64 = unsafe {{ vcvtq_s64_f64(self.0) }};"
        )
        .unwrap();
        writeln!(code, "        let s32_low = unsafe {{ vmovn_s64(s64) }};").unwrap();
        writeln!(
            code,
            "        i32x4(unsafe {{ vcombine_s32(s32_low, vdup_n_s32(0)) }})"
        )
        .unwrap();
        writeln!(code, "    }}\n").unwrap();
    }

    code
}

/// Generate operator trait implementations for NEON
fn generate_operator_impls(ty: &SimdType) -> String {
    let mut code = String::new();
    let name = ty.name();

    let add_fn = Arm::arith_intrinsic("add", ty.elem);
    let sub_fn = Arm::arith_intrinsic("sub", ty.elem);

    // Add
    writeln!(code, "impl core::ops::Add for {} {{", name).unwrap();
    writeln!(code, "    type Output = Self;").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    fn add(self, rhs: Self) -> Self {{").unwrap();
    writeln!(code, "        Self(unsafe {{ {}(self.0, rhs.0) }})", add_fn).unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}\n").unwrap();

    // Sub
    writeln!(code, "impl core::ops::Sub for {} {{", name).unwrap();
    writeln!(code, "    type Output = Self;").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    fn sub(self, rhs: Self) -> Self {{").unwrap();
    writeln!(code, "        Self(unsafe {{ {}(self.0, rhs.0) }})", sub_fn).unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}\n").unwrap();

    // Mul (only for floats and some integers)
    if ty.elem.is_float()
        || matches!(
            ty.elem,
            ElementType::I16 | ElementType::U16 | ElementType::I32 | ElementType::U32
        )
    {
        let mul_fn = Arm::arith_intrinsic("mul", ty.elem);
        writeln!(code, "impl core::ops::Mul for {} {{", name).unwrap();
        writeln!(code, "    type Output = Self;").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    fn mul(self, rhs: Self) -> Self {{").unwrap();
        writeln!(code, "        Self(unsafe {{ {}(self.0, rhs.0) }})", mul_fn).unwrap();
        writeln!(code, "    }}").unwrap();
        writeln!(code, "}}\n").unwrap();
    }

    // Div (floats only - NEON has vdivq)
    if ty.elem.is_float() {
        let div_fn = Arm::arith_intrinsic("div", ty.elem);
        writeln!(code, "impl core::ops::Div for {} {{", name).unwrap();
        writeln!(code, "    type Output = Self;").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    fn div(self, rhs: Self) -> Self {{").unwrap();
        writeln!(code, "        Self(unsafe {{ {}(self.0, rhs.0) }})", div_fn).unwrap();
        writeln!(code, "    }}").unwrap();
        writeln!(code, "}}\n").unwrap();
    }

    // Neg (floats and signed integers)
    if ty.elem.is_float() || ty.elem.is_signed() {
        let neg_fn = Arm::neg_intrinsic(ty.elem);
        writeln!(code, "impl core::ops::Neg for {} {{", name).unwrap();
        writeln!(code, "    type Output = Self;").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    fn neg(self) -> Self {{").unwrap();
        writeln!(code, "        Self(unsafe {{ {}(self.0) }})", neg_fn).unwrap();
        writeln!(code, "    }}").unwrap();
        writeln!(code, "}}\n").unwrap();
    }

    // Assign ops
    writeln!(code, "impl core::ops::AddAssign for {} {{", name).unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    fn add_assign(&mut self, rhs: Self) {{").unwrap();
    writeln!(code, "        *self = *self + rhs;").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}\n").unwrap();

    writeln!(code, "impl core::ops::SubAssign for {} {{", name).unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    fn sub_assign(&mut self, rhs: Self) {{").unwrap();
    writeln!(code, "        *self = *self - rhs;").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}\n").unwrap();

    if ty.elem.is_float()
        || matches!(
            ty.elem,
            ElementType::I16 | ElementType::U16 | ElementType::I32 | ElementType::U32
        )
    {
        writeln!(code, "impl core::ops::MulAssign for {} {{", name).unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    fn mul_assign(&mut self, rhs: Self) {{").unwrap();
        writeln!(code, "        *self = *self * rhs;").unwrap();
        writeln!(code, "    }}").unwrap();
        writeln!(code, "}}\n").unwrap();
    }

    if ty.elem.is_float() {
        writeln!(code, "impl core::ops::DivAssign for {} {{", name).unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    fn div_assign(&mut self, rhs: Self) {{").unwrap();
        writeln!(code, "        *self = *self / rhs;").unwrap();
        writeln!(code, "    }}").unwrap();
        writeln!(code, "}}\n").unwrap();
    }

    // Index
    let elem = ty.elem.name();
    let lanes = ty.lanes();
    writeln!(code, "impl core::ops::Index<usize> for {} {{", name).unwrap();
    writeln!(code, "    type Output = {};", elem).unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    fn index(&self, i: usize) -> &Self::Output {{").unwrap();
    writeln!(
        code,
        "        assert!(i < {}, \"index out of bounds\");",
        lanes
    )
    .unwrap();
    writeln!(
        code,
        "        unsafe {{ &*(self as *const Self as *const {}).add(i) }}",
        elem
    )
    .unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}\n").unwrap();

    writeln!(code, "impl core::ops::IndexMut<usize> for {} {{", name).unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(
        code,
        "    fn index_mut(&mut self, i: usize) -> &mut Self::Output {{"
    )
    .unwrap();
    writeln!(
        code,
        "        assert!(i < {}, \"index out of bounds\");",
        lanes
    )
    .unwrap();
    writeln!(
        code,
        "        unsafe {{ &mut *(self as *mut Self as *mut {}).add(i) }}",
        elem
    )
    .unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}\n").unwrap();

    // From<[T; N]> for zero-cost conversion
    let inner = super::arch::arm::Arm::intrinsic_type(ty.elem, ty.width);
    writeln!(code, "impl From<[{}; {}]> for {} {{", elem, lanes, name).unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    fn from(arr: [{}; {}]) -> Self {{", elem, lanes).unwrap();
    writeln!(
        code,
        "        // SAFETY: [{}; {}] and {} have identical size and layout",
        elem, lanes, inner
    )
    .unwrap();
    writeln!(code, "        Self(unsafe {{ core::mem::transmute(arr) }})").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}\n").unwrap();

    // Into<[T; N]> for zero-cost conversion back
    writeln!(code, "impl From<{}> for [{}; {}] {{", name, elem, lanes).unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    fn from(v: {}) -> Self {{", name).unwrap();
    writeln!(
        code,
        "        // SAFETY: {} and [{}; {}] have identical size and layout",
        inner, elem, lanes
    )
    .unwrap();
    writeln!(code, "        unsafe {{ core::mem::transmute(v.0) }}").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}\n").unwrap();

    code
}

/// Generate the ARM w128.rs file
pub fn generate_arm_w128(types: &[SimdType]) -> String {
    let mut code = String::new();

    code.push_str("//! 128-bit (NEON) SIMD types.\n");
    code.push_str("//!\n");
    code.push_str("//! **Auto-generated** by `cargo xtask generate` - do not edit manually.\n\n");

    code.push_str("use core::arch::aarch64::*;\n\n");

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
