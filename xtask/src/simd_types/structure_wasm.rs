//! WebAssembly SIMD128 type structure generation.
//!
//! Generates WASM SIMD types parallel to x86 and ARM types.

use super::arch::Arch;
use super::arch::wasm::Wasm;
use super::types::{ElementType, SimdType, SimdWidth};
use std::fmt::Write;

/// Generate a complete WASM SIMD type
pub fn generate_type(ty: &SimdType) -> String {
    assert!(
        ty.width == SimdWidth::W128,
        "WASM only supports 128-bit types"
    );

    let mut code = String::new();
    let name = ty.name();
    let lanes = ty.lanes();
    let elem = ty.elem.name();
    let inner = Wasm::intrinsic_type(ty.elem, ty.width);

    // Type definition
    writeln!(
        code,
        "\n// ============================================================================"
    )
    .unwrap();
    writeln!(
        code,
        "// {} - {} x {} (128-bit WASM SIMD)",
        name, lanes, elem
    )
    .unwrap();
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

    // Bitwise operations (not, shift, blend)
    code.push_str(&generate_bitwise_ops(ty));

    // Transcendental operations (log, exp, pow) for float types
    code.push_str(&super::transcendental_wasm::generate_wasm_transcendental_ops(ty));

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
    let inner = Wasm::intrinsic_type(ty.elem, ty.width);

    let splat_fn = Wasm::splat_intrinsic(ty.elem);

    // Load (WASM uses v128_load which is untyped)
    writeln!(code, "    /// Load from array (token-gated)").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(
        code,
        "    pub fn load(_: archmage::Simd128Token, data: &[{}; {}]) -> Self {{",
        elem, lanes
    )
    .unwrap();
    writeln!(
        code,
        "        Self(unsafe {{ v128_load(data.as_ptr() as *const v128) }})"
    )
    .unwrap();
    writeln!(code, "    }}\n").unwrap();

    // Splat
    writeln!(code, "    /// Broadcast scalar to all lanes (token-gated)").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(
        code,
        "    pub fn splat(_: archmage::Simd128Token, v: {}) -> Self {{",
        elem
    )
    .unwrap();
    writeln!(code, "        Self({}(v))", splat_fn).unwrap();
    writeln!(code, "    }}\n").unwrap();

    // Zero
    writeln!(code, "    /// Zero vector (token-gated)").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(
        code,
        "    pub fn zero(_: archmage::Simd128Token) -> Self {{"
    )
    .unwrap();
    let zero_val = ty.elem.zero_literal();
    writeln!(code, "        Self({}({}))", splat_fn, zero_val).unwrap();
    writeln!(code, "    }}\n").unwrap();

    // From array (zero-cost transmute)
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
        "    pub fn from_array(_: archmage::Simd128Token, arr: [{}; {}]) -> Self {{",
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
        "        unsafe {{ v128_store(out.as_mut_ptr() as *mut v128, self.0) }};"
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
        "    /// Caller must ensure the CPU supports WASM SIMD128."
    )
    .unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub unsafe fn from_raw(v: {}) -> Self {{", inner).unwrap();
    writeln!(code, "        Self(v)").unwrap();
    writeln!(code, "    }}\n").unwrap();

    code
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

        writeln!(code, "    /// Element-wise minimum").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn min(self, other: Self) -> Self {{").unwrap();
        writeln!(code, "        Self({}(self.0, other.0))", min_fn).unwrap();
        writeln!(code, "    }}\n").unwrap();

        writeln!(code, "    /// Element-wise maximum").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn max(self, other: Self) -> Self {{").unwrap();
        writeln!(code, "        Self({}(self.0, other.0))", max_fn).unwrap();
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
    }

    // Float-only operations
    if ty.elem.is_float() {
        let sqrt_fn = Wasm::sqrt_intrinsic(ty.elem);
        let abs_fn = Wasm::abs_intrinsic(ty.elem);
        let _neg_fn = Wasm::neg_intrinsic(ty.elem);
        let floor_fn = Wasm::floor_intrinsic(ty.elem);
        let ceil_fn = Wasm::ceil_intrinsic(ty.elem);
        let nearest_fn = Wasm::nearest_intrinsic(ty.elem);

        writeln!(code, "    /// Square root").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn sqrt(self) -> Self {{").unwrap();
        writeln!(code, "        Self({}(self.0))", sqrt_fn).unwrap();
        writeln!(code, "    }}\n").unwrap();

        writeln!(code, "    /// Absolute value").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn abs(self) -> Self {{").unwrap();
        writeln!(code, "        Self({}(self.0))", abs_fn).unwrap();
        writeln!(code, "    }}\n").unwrap();

        writeln!(code, "    /// Floor").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn floor(self) -> Self {{").unwrap();
        writeln!(code, "        Self({}(self.0))", floor_fn).unwrap();
        writeln!(code, "    }}\n").unwrap();

        writeln!(code, "    /// Ceil").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn ceil(self) -> Self {{").unwrap();
        writeln!(code, "        Self({}(self.0))", ceil_fn).unwrap();
        writeln!(code, "    }}\n").unwrap();

        writeln!(code, "    /// Round to nearest").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn round(self) -> Self {{").unwrap();
        writeln!(code, "        Self({}(self.0))", nearest_fn).unwrap();
        writeln!(code, "    }}\n").unwrap();

        // FMA: WASM relaxed-simd has f32x4_relaxed_madd, but it's not stable
        // For now, emulate with mul + add
        let mul_fn = Wasm::arith_intrinsic("mul", ty.elem);
        let add_fn = Wasm::arith_intrinsic("add", ty.elem);
        writeln!(code, "    /// Fused multiply-add: self * a + b").unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Note: WASM doesn't have native FMA in stable SIMD,"
        )
        .unwrap();
        writeln!(code, "    /// this is emulated with separate mul and add.").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(
            code,
            "    pub fn mul_add(self, a: Self, b: Self) -> Self {{"
        )
        .unwrap();
        writeln!(
            code,
            "        Self({}({}(self.0, a.0), b.0))",
            add_fn, mul_fn
        )
        .unwrap();
        writeln!(code, "    }}\n").unwrap();
    }

    // Signed integer abs (not available for i64 in WASM)
    if !ty.elem.is_float() && ty.elem.is_signed() && ty.elem != ElementType::I64 {
        let abs_fn = Wasm::abs_intrinsic(ty.elem);
        writeln!(code, "    /// Absolute value").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn abs(self) -> Self {{").unwrap();
        writeln!(code, "        Self({}(self.0))", abs_fn).unwrap();
        writeln!(code, "    }}\n").unwrap();
    }

    code
}

/// Generate horizontal operations for WASM
fn generate_horizontal_ops(ty: &SimdType) -> String {
    let mut code = String::new();
    let elem = ty.elem.name();
    let lanes = ty.lanes();

    // reduce_add - WASM doesn't have native horizontal add, use extract lanes
    let extract_fn = Wasm::extract_lane_intrinsic(ty.elem);

    if ty.elem.is_float() || ty.elem.is_signed() || !ty.elem.is_float() {
        writeln!(code, "    /// Reduce: sum all lanes").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn reduce_add(self) -> {} {{", elem).unwrap();

        // Generate lane extractions and sum
        match lanes {
            2 => {
                writeln!(
                    code,
                    "        {}::<0>(self.0) + {}::<1>(self.0)",
                    extract_fn, extract_fn
                )
                .unwrap();
            }
            4 => {
                writeln!(
                    code,
                    "        {}::<0>(self.0) + {}::<1>(self.0) + {}::<2>(self.0) + {}::<3>(self.0)",
                    extract_fn, extract_fn, extract_fn, extract_fn
                )
                .unwrap();
            }
            8 => {
                writeln!(code, "        let arr = self.to_array();").unwrap();
                writeln!(
                    code,
                    "        arr[0] + arr[1] + arr[2] + arr[3] + arr[4] + arr[5] + arr[6] + arr[7]"
                )
                .unwrap();
            }
            16 => {
                writeln!(code, "        let arr = self.to_array();").unwrap();
                writeln!(
                    code,
                    "        arr.iter().copied().fold(0{}, |a, b| a.wrapping_add(b))",
                    elem
                )
                .unwrap();
            }
            _ => {
                writeln!(code, "        let arr = self.to_array();").unwrap();
                writeln!(code, "        arr.iter().copied().sum()").unwrap();
            }
        }

        writeln!(code, "    }}\n").unwrap();
    }

    // reduce_max/reduce_min for floats
    if ty.elem.is_float() {
        writeln!(code, "    /// Reduce: max of all lanes").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn reduce_max(self) -> {} {{", elem).unwrap();
        writeln!(code, "        let arr = self.to_array();").unwrap();
        writeln!(
            code,
            "        arr.iter().copied().fold({}::NEG_INFINITY, {}::max)",
            elem, elem
        )
        .unwrap();
        writeln!(code, "    }}\n").unwrap();

        writeln!(code, "    /// Reduce: min of all lanes").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn reduce_min(self) -> {} {{", elem).unwrap();
        writeln!(code, "        let arr = self.to_array();").unwrap();
        writeln!(
            code,
            "        arr.iter().copied().fold({}::INFINITY, {}::min)",
            elem, elem
        )
        .unwrap();
        writeln!(code, "    }}\n").unwrap();
    }

    code
}

/// Generate comparison operations for WASM
fn generate_comparison_ops(ty: &SimdType) -> String {
    let mut code = String::new();
    let name = ty.name();

    let eq_fn = Wasm::cmp_intrinsic("eq", ty.elem);

    // simd_eq - available for all types
    writeln!(
        code,
        "    /// Element-wise equality comparison (returns mask)"
    )
    .unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn simd_eq(self, other: Self) -> Self {{").unwrap();
    writeln!(code, "        Self({}(self.0, other.0))", eq_fn).unwrap();
    writeln!(code, "    }}\n").unwrap();

    // simd_ne - available for all types (use not(eq))
    writeln!(
        code,
        "    /// Element-wise inequality comparison (returns mask)"
    )
    .unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn simd_ne(self, other: Self) -> Self {{").unwrap();
    writeln!(code, "        Self(v128_not({}(self.0, other.0)))", eq_fn).unwrap();
    writeln!(code, "    }}\n").unwrap();

    // WASM doesn't have lt/le/gt/ge for u64x2 - skip ordering comparisons for u64
    let has_ordering = ty.elem != ElementType::U64;

    if has_ordering {
        let lt_fn = Wasm::cmp_intrinsic("lt", ty.elem);
        // simd_lt
        writeln!(
            code,
            "    /// Element-wise less-than comparison (returns mask)"
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn simd_lt(self, other: Self) -> Self {{").unwrap();
        writeln!(code, "        Self({}(self.0, other.0))", lt_fn).unwrap();
        writeln!(code, "    }}\n").unwrap();

        // simd_le
        let le_fn = Wasm::cmp_intrinsic("le", ty.elem);
        writeln!(
            code,
            "    /// Element-wise less-than-or-equal comparison (returns mask)"
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn simd_le(self, other: Self) -> Self {{").unwrap();
        writeln!(code, "        Self({}(self.0, other.0))", le_fn).unwrap();
        writeln!(code, "    }}\n").unwrap();

        // simd_gt
        let gt_fn = Wasm::cmp_intrinsic("gt", ty.elem);
        writeln!(
            code,
            "    /// Element-wise greater-than comparison (returns mask)"
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn simd_gt(self, other: Self) -> Self {{").unwrap();
        writeln!(code, "        Self({}(self.0, other.0))", gt_fn).unwrap();
        writeln!(code, "    }}\n").unwrap();

        // simd_ge
        let ge_fn = Wasm::cmp_intrinsic("ge", ty.elem);
        writeln!(
            code,
            "    /// Element-wise greater-than-or-equal comparison (returns mask)"
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn simd_ge(self, other: Self) -> Self {{").unwrap();
        writeln!(code, "        Self({}(self.0, other.0))", ge_fn).unwrap();
        writeln!(code, "    }}\n").unwrap();
    }

    // blend (conditional select using mask)
    writeln!(code, "    /// Blend two vectors based on a mask").unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// For each lane, selects from `self` if the corresponding mask lane is all-ones,"
    )
    .unwrap();
    writeln!(code, "    /// otherwise selects from `other`.").unwrap();
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
        "    /// let result = a.blend(b, mask);  // selects from a (all ones in mask)"
    )
    .unwrap();
    writeln!(code, "    /// ```").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(
        code,
        "    pub fn blend(self, other: Self, mask: Self) -> Self {{"
    )
    .unwrap();
    writeln!(
        code,
        "        Self(v128_bitselect(self.0, other.0, mask.0))"
    )
    .unwrap();
    writeln!(code, "    }}\n").unwrap();

    code
}

/// Generate bitwise operations (not, shift) for WASM
fn generate_bitwise_ops(ty: &SimdType) -> String {
    let mut code = String::new();

    // not() - available for all types
    writeln!(code, "    /// Bitwise NOT").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn not(self) -> Self {{").unwrap();
    writeln!(code, "        Self(v128_not(self.0))").unwrap();
    writeln!(code, "    }}\n").unwrap();

    // Shift operations for integer types
    if !ty.elem.is_float() {
        let shl_fn = Wasm::shl_intrinsic(ty.elem);
        let shr_fn = Wasm::shr_intrinsic(ty.elem);

        // shl<const N>
        writeln!(code, "    /// Shift left by constant").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn shl<const N: u32>(self) -> Self {{").unwrap();
        writeln!(code, "        Self({}(self.0, N))", shl_fn).unwrap();
        writeln!(code, "    }}\n").unwrap();

        // shr<const N>
        writeln!(code, "    /// Shift right by constant").unwrap();
        if ty.elem.is_signed() {
            writeln!(code, "    ///").unwrap();
            writeln!(
                code,
                "    /// For signed types, this is an arithmetic shift (sign-extending)."
            )
            .unwrap();
        }
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn shr<const N: u32>(self) -> Self {{").unwrap();
        writeln!(code, "        Self({}(self.0, N))", shr_fn).unwrap();
        writeln!(code, "    }}\n").unwrap();
    }

    // all_true and any_true - only for integer types (WASM doesn't have float versions)
    if !ty.elem.is_float() {
        let all_true_fn = Wasm::all_true_intrinsic(ty.elem);
        writeln!(code, "    /// Check if all lanes are non-zero (all true)").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn all_true(self) -> bool {{").unwrap();
        writeln!(code, "        {}(self.0)", all_true_fn).unwrap();
        writeln!(code, "    }}\n").unwrap();

        writeln!(code, "    /// Check if any lane is non-zero (any true)").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn any_true(self) -> bool {{").unwrap();
        writeln!(code, "        v128_any_true(self.0)").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // bitmask for extracting lane results - only integers
        let bitmask_fn = Wasm::bitmask_intrinsic(ty.elem);
        writeln!(
            code,
            "    /// Extract the high bit of each lane as a bitmask"
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn bitmask(self) -> u32 {{").unwrap();
        writeln!(code, "        {}(self.0) as u32", bitmask_fn).unwrap();
        writeln!(code, "    }}\n").unwrap();
    }

    code
}

/// Generate operator trait implementations for WASM
fn generate_operator_impls(ty: &SimdType) -> String {
    let mut code = String::new();
    let name = ty.name();

    let add_fn = Wasm::arith_intrinsic("add", ty.elem);
    let sub_fn = Wasm::arith_intrinsic("sub", ty.elem);

    // Add
    writeln!(code, "impl core::ops::Add for {} {{", name).unwrap();
    writeln!(code, "    type Output = Self;").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    fn add(self, rhs: Self) -> Self {{").unwrap();
    writeln!(code, "        Self({}(self.0, rhs.0))", add_fn).unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}\n").unwrap();

    // Sub
    writeln!(code, "impl core::ops::Sub for {} {{", name).unwrap();
    writeln!(code, "    type Output = Self;").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    fn sub(self, rhs: Self) -> Self {{").unwrap();
    writeln!(code, "        Self({}(self.0, rhs.0))", sub_fn).unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}\n").unwrap();

    // Mul (NOT available for i8/u8 in WASM - no i8x16_mul or u8x16_mul intrinsics)
    let has_mul = !matches!(ty.elem, ElementType::I8 | ElementType::U8);
    if has_mul {
        let mul_fn = Wasm::arith_intrinsic("mul", ty.elem);
        writeln!(code, "impl core::ops::Mul for {} {{", name).unwrap();
        writeln!(code, "    type Output = Self;").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    fn mul(self, rhs: Self) -> Self {{").unwrap();
        writeln!(code, "        Self({}(self.0, rhs.0))", mul_fn).unwrap();
        writeln!(code, "    }}").unwrap();
        writeln!(code, "}}\n").unwrap();
    }

    // Div (floats only)
    if ty.elem.is_float() {
        let div_fn = Wasm::arith_intrinsic("div", ty.elem);
        writeln!(code, "impl core::ops::Div for {} {{", name).unwrap();
        writeln!(code, "    type Output = Self;").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    fn div(self, rhs: Self) -> Self {{").unwrap();
        writeln!(code, "        Self({}(self.0, rhs.0))", div_fn).unwrap();
        writeln!(code, "    }}").unwrap();
        writeln!(code, "}}\n").unwrap();
    }

    // Neg (floats and signed integers)
    if ty.elem.is_float() || ty.elem.is_signed() {
        let neg_fn = Wasm::neg_intrinsic(ty.elem);
        writeln!(code, "impl core::ops::Neg for {} {{", name).unwrap();
        writeln!(code, "    type Output = Self;").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    fn neg(self) -> Self {{").unwrap();
        writeln!(code, "        Self({}(self.0))", neg_fn).unwrap();
        writeln!(code, "    }}").unwrap();
        writeln!(code, "}}\n").unwrap();
    }

    // Bitwise ops (use v128_and/or/xor)
    writeln!(code, "impl core::ops::BitAnd for {} {{", name).unwrap();
    writeln!(code, "    type Output = Self;").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    fn bitand(self, rhs: Self) -> Self {{").unwrap();
    writeln!(code, "        Self(v128_and(self.0, rhs.0))").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}\n").unwrap();

    writeln!(code, "impl core::ops::BitOr for {} {{", name).unwrap();
    writeln!(code, "    type Output = Self;").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    fn bitor(self, rhs: Self) -> Self {{").unwrap();
    writeln!(code, "        Self(v128_or(self.0, rhs.0))").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}\n").unwrap();

    writeln!(code, "impl core::ops::BitXor for {} {{", name).unwrap();
    writeln!(code, "    type Output = Self;").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    fn bitxor(self, rhs: Self) -> Self {{").unwrap();
    writeln!(code, "        Self(v128_xor(self.0, rhs.0))").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}\n").unwrap();

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

    // MulAssign (only when Mul is available)
    if has_mul {
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

    writeln!(code, "impl core::ops::BitAndAssign for {} {{", name).unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    fn bitand_assign(&mut self, rhs: Self) {{").unwrap();
    writeln!(code, "        *self = *self & rhs;").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}\n").unwrap();

    writeln!(code, "impl core::ops::BitOrAssign for {} {{", name).unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    fn bitor_assign(&mut self, rhs: Self) {{").unwrap();
    writeln!(code, "        *self = *self | rhs;").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}\n").unwrap();

    writeln!(code, "impl core::ops::BitXorAssign for {} {{", name).unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    fn bitxor_assign(&mut self, rhs: Self) {{").unwrap();
    writeln!(code, "        *self = *self ^ rhs;").unwrap();
    writeln!(code, "    }}").unwrap();
    writeln!(code, "}}\n").unwrap();

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
    let inner = Wasm::intrinsic_type(ty.elem, ty.width);
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

/// Generate the WASM w128.rs file
pub fn generate_wasm_w128(types: &[SimdType]) -> String {
    let mut code = String::new();

    code.push_str("//! 128-bit (WASM SIMD) types.\n");
    code.push_str("//!\n");
    code.push_str("//! **Auto-generated** by `cargo xtask generate` - do not edit manually.\n\n");

    code.push_str("use core::arch::wasm32::*;\n\n");

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
