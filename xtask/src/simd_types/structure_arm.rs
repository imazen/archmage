//! ARM NEON type structure generation.
//!
//! Generates NEON SIMD types parallel to x86 types.

use super::arch::Arch;
use super::arch::arm::Arm;
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
    if ty.elem.is_float() || ty.elem.is_signed() {
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
