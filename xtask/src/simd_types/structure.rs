//! Type structure and macro generation.

use super::block_ops;
use super::ops;
use super::transcendental;
use super::types::SimdType;
use std::fmt::Write;

/// Generate comparison trait definitions
pub fn generate_comparison_traits() -> String {
    r#"
// ============================================================================
// Comparison Traits (return masks, not bool)
// ============================================================================

/// SIMD equality comparison (returns mask)
pub trait SimdEq<Rhs = Self> {
    type Output;
    fn simd_eq(self, rhs: Rhs) -> Self::Output;
}

/// SIMD inequality comparison (returns mask)
pub trait SimdNe<Rhs = Self> {
    type Output;
    fn simd_ne(self, rhs: Rhs) -> Self::Output;
}

/// SIMD less-than comparison (returns mask)
pub trait SimdLt<Rhs = Self> {
    type Output;
    fn simd_lt(self, rhs: Rhs) -> Self::Output;
}

/// SIMD less-than-or-equal comparison (returns mask)
pub trait SimdLe<Rhs = Self> {
    type Output;
    fn simd_le(self, rhs: Rhs) -> Self::Output;
}

/// SIMD greater-than comparison (returns mask)
pub trait SimdGt<Rhs = Self> {
    type Output;
    fn simd_gt(self, rhs: Rhs) -> Self::Output;
}

/// SIMD greater-than-or-equal comparison (returns mask)
pub trait SimdGe<Rhs = Self> {
    type Output;
    fn simd_ge(self, rhs: Rhs) -> Self::Output;
}

"#
    .to_string()
}

/// Generate implementation macros
pub fn generate_macros() -> String {
    r#"
// ============================================================================
// Implementation Macros
// ============================================================================

#[doc(hidden)]
#[macro_export]
macro_rules! impl_arithmetic_ops {
    ($t:ty, $add:path, $sub:path, $mul:path, $div:path) => {
        impl Add for $t {
            type Output = Self;
            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                Self(unsafe { $add(self.0, rhs.0) })
            }
        }
        impl Sub for $t {
            type Output = Self;
            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                Self(unsafe { $sub(self.0, rhs.0) })
            }
        }
        impl Mul for $t {
            type Output = Self;
            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                Self(unsafe { $mul(self.0, rhs.0) })
            }
        }
        impl Div for $t {
            type Output = Self;
            #[inline(always)]
            fn div(self, rhs: Self) -> Self {
                Self(unsafe { $div(self.0, rhs.0) })
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! impl_int_arithmetic_ops {
    ($t:ty, $add:path, $sub:path) => {
        impl Add for $t {
            type Output = Self;
            #[inline(always)]
            fn add(self, rhs: Self) -> Self {
                Self(unsafe { $add(self.0, rhs.0) })
            }
        }
        impl Sub for $t {
            type Output = Self;
            #[inline(always)]
            fn sub(self, rhs: Self) -> Self {
                Self(unsafe { $sub(self.0, rhs.0) })
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! impl_int_mul_op {
    ($t:ty, $mul:path) => {
        impl Mul for $t {
            type Output = Self;
            #[inline(always)]
            fn mul(self, rhs: Self) -> Self {
                Self(unsafe { $mul(self.0, rhs.0) })
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! impl_bitwise_ops {
    ($t:ty, $inner:ty, $and:path, $or:path, $xor:path) => {
        impl BitAnd for $t {
            type Output = Self;
            #[inline(always)]
            fn bitand(self, rhs: Self) -> Self {
                Self(unsafe { $and(self.0, rhs.0) })
            }
        }
        impl BitOr for $t {
            type Output = Self;
            #[inline(always)]
            fn bitor(self, rhs: Self) -> Self {
                Self(unsafe { $or(self.0, rhs.0) })
            }
        }
        impl BitXor for $t {
            type Output = Self;
            #[inline(always)]
            fn bitxor(self, rhs: Self) -> Self {
                Self(unsafe { $xor(self.0, rhs.0) })
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! impl_assign_ops {
    ($t:ty) => {
        impl AddAssign for $t {
            #[inline(always)]
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }
        impl SubAssign for $t {
            #[inline(always)]
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }
        impl BitAndAssign for $t {
            #[inline(always)]
            fn bitand_assign(&mut self, rhs: Self) {
                *self = *self & rhs;
            }
        }
        impl BitOrAssign for $t {
            #[inline(always)]
            fn bitor_assign(&mut self, rhs: Self) {
                *self = *self | rhs;
            }
        }
        impl BitXorAssign for $t {
            #[inline(always)]
            fn bitxor_assign(&mut self, rhs: Self) {
                *self = *self ^ rhs;
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! impl_float_assign_ops {
    ($t:ty) => {
        impl_assign_ops!($t);
        impl MulAssign for $t {
            #[inline(always)]
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }
        impl DivAssign for $t {
            #[inline(always)]
            fn div_assign(&mut self, rhs: Self) {
                *self = *self / rhs;
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! impl_neg {
    ($t:ty, $sub:path, $zero:path) => {
        impl Neg for $t {
            type Output = Self;
            #[inline(always)]
            fn neg(self) -> Self {
                Self(unsafe { $sub($zero(), self.0) })
            }
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! impl_index {
    ($t:ty, $elem:ty, $lanes:expr) => {
        impl Index<usize> for $t {
            type Output = $elem;
            #[inline(always)]
            fn index(&self, i: usize) -> &Self::Output {
                assert!(i < $lanes, "index out of bounds");
                unsafe { &*(self as *const Self as *const $elem).add(i) }
            }
        }
        impl IndexMut<usize> for $t {
            #[inline(always)]
            fn index_mut(&mut self, i: usize) -> &mut Self::Output {
                assert!(i < $lanes, "index out of bounds");
                unsafe { &mut *(self as *mut Self as *mut $elem).add(i) }
            }
        }
    };
}

"#
    .to_string()
}

/// Generate a complete SIMD type
pub fn generate_type(ty: &SimdType) -> String {
    let mut code = String::new();
    let name = ty.name();
    let lanes = ty.lanes();
    let elem = ty.elem.name();
    let inner = ty.x86_inner_type();
    let token = ty.token();
    let prefix = ty.width.x86_prefix();
    let suffix = ty.elem.x86_suffix();

    // Feature gate for AVX-512
    let cfg_attr = match ty.width.required_feature() {
        Some(feat) => format!(
            "#[cfg(all(target_arch = \"x86_64\", feature = \"{}\"))]\n",
            feat
        ),
        None => "#[cfg(target_arch = \"x86_64\")]\n".to_string(),
    };

    // Type definition
    writeln!(
        code,
        "\n// ============================================================================"
    )
    .unwrap();
    writeln!(
        code,
        "// {} - {} x {} ({}-bit)",
        name,
        lanes,
        elem,
        ty.width.bits()
    )
    .unwrap();
    writeln!(
        code,
        "// ============================================================================\n"
    )
    .unwrap();

    writeln!(code, "{}#[derive(Clone, Copy, Debug)]", cfg_attr).unwrap();
    writeln!(code, "#[repr(transparent)]").unwrap();
    writeln!(code, "pub struct {}({});\n", name, inner).unwrap();

    // Impl block
    writeln!(code, "{}impl {} {{", cfg_attr, name).unwrap();
    writeln!(code, "    pub const LANES: usize = {};\n", lanes).unwrap();

    // Construction methods
    code.push_str(&generate_construction_methods(ty));

    // Math operations
    code.push_str(&ops::generate_math_ops(ty));

    // Comparison operations
    code.push_str(&ops::generate_comparison_ops(ty));

    // Blend/select operations
    code.push_str(&ops::generate_blend_ops(ty));

    // Horizontal operations
    code.push_str(&ops::generate_horizontal_ops(ty));

    // Type conversions
    code.push_str(&ops::generate_conversion_ops(ty));

    // Approximation operations
    code.push_str(&ops::generate_approx_ops(ty));

    // Bitwise unary operations
    code.push_str(&ops::generate_bitwise_unary_ops(ty));

    // Shift operations
    code.push_str(&ops::generate_shift_ops(ty));

    // Transcendental operations
    code.push_str(&transcendental::generate_transcendental_ops(ty));

    // Block operations (transpose, etc.)
    code.push_str(&block_ops::generate_block_ops(ty));

    writeln!(code, "}}\n").unwrap();

    // Operator implementations
    code.push_str(&generate_operator_impls(ty, &cfg_attr));

    // Scalar broadcast operators
    code.push_str(&ops::generate_scalar_ops(ty, &cfg_attr));

    code
}

/// Generate construction and extraction methods
fn generate_construction_methods(ty: &SimdType) -> String {
    use super::types::ElementType;

    let mut code = String::new();
    let name = ty.name();
    let lanes = ty.lanes();
    let elem = ty.elem.name();
    let inner = ty.x86_inner_type();
    let token = ty.token();
    let prefix = ty.width.x86_prefix();
    let suffix = ty.elem.x86_suffix();

    // Load
    writeln!(code, "    /// Load from array (token-gated)").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(
        code,
        "    pub fn load(_: crate::{}, data: &[{}; {}]) -> Self {{",
        token, elem, lanes
    )
    .unwrap();
    if ty.elem.is_float() {
        writeln!(
            code,
            "        Self(unsafe {{ {}_loadu_{}(data.as_ptr()) }})",
            prefix, suffix
        )
        .unwrap();
    } else {
        writeln!(
            code,
            "        Self(unsafe {{ {}_loadu_si{}(data.as_ptr() as *const {}) }})",
            prefix,
            ty.width.bits(),
            inner
        )
        .unwrap();
    }
    writeln!(code, "    }}\n").unwrap();

    // Splat
    writeln!(code, "    /// Broadcast scalar to all lanes (token-gated)").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(
        code,
        "    pub fn splat(_: crate::{}, v: {}) -> Self {{",
        token, elem
    )
    .unwrap();
    let (set1_suffix, cast) = match (ty.elem, ty.width) {
        (ElementType::I64 | ElementType::U64, super::types::SimdWidth::W512) => {
            ("epi64", ty.elem != ElementType::I64)
        }
        (ElementType::I64 | ElementType::U64, _) => ("epi64x", ty.elem != ElementType::I64),
        (ElementType::U8, _) => ("epi8", true),
        (ElementType::U16, _) => ("epi16", true),
        (ElementType::U32, _) => ("epi32", true),
        _ => (suffix, false),
    };
    if cast && !ty.elem.is_float() {
        let signed_ty = match ty.elem {
            ElementType::U8 => "i8",
            ElementType::U16 => "i16",
            ElementType::U32 => "i32",
            ElementType::U64 => "i64",
            _ => elem,
        };
        writeln!(
            code,
            "        Self(unsafe {{ {}_set1_{}(v as {}) }})",
            prefix, set1_suffix, signed_ty
        )
        .unwrap();
    } else {
        writeln!(
            code,
            "        Self(unsafe {{ {}_set1_{}(v) }})",
            prefix, set1_suffix
        )
        .unwrap();
    }
    writeln!(code, "    }}\n").unwrap();

    // Zero
    writeln!(code, "    /// Zero vector (token-gated)").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn zero(_: crate::{}) -> Self {{", token).unwrap();
    if ty.elem.is_float() {
        writeln!(
            code,
            "        Self(unsafe {{ {}_setzero_{}() }})",
            prefix, suffix
        )
        .unwrap();
    } else {
        writeln!(
            code,
            "        Self(unsafe {{ {}_setzero_si{}() }})",
            prefix,
            ty.width.bits()
        )
        .unwrap();
    }
    writeln!(code, "    }}\n").unwrap();

    // From array
    writeln!(code, "    /// Create from array (token-gated)").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(
        code,
        "    pub fn from_array(token: crate::{}, arr: [{}; {}]) -> Self {{",
        token, elem, lanes
    )
    .unwrap();
    writeln!(code, "        Self::load(token, &arr)").unwrap();
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
    if ty.elem.is_float() {
        writeln!(
            code,
            "        unsafe {{ {}_storeu_{}(out.as_mut_ptr(), self.0) }};",
            prefix, suffix
        )
        .unwrap();
    } else {
        writeln!(
            code,
            "        unsafe {{ {}_storeu_si{}(out.as_mut_ptr() as *mut {}, self.0) }};",
            prefix,
            ty.width.bits(),
            inner
        )
        .unwrap();
    }
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
    writeln!(
        code,
        "    /// Use token-gated constructors (`load`, `splat`, `zero`) for safe construction."
    )
    .unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub unsafe fn from_raw(v: {}) -> Self {{", inner).unwrap();
    writeln!(code, "        Self(v)").unwrap();
    writeln!(code, "    }}\n").unwrap();

    code
}

/// Generate operator trait implementations
fn generate_operator_impls(ty: &SimdType, cfg_attr: &str) -> String {
    use super::types::ElementType;

    let mut code = String::new();
    let name = ty.name();
    let prefix = ty.width.x86_prefix();
    let suffix = ty.elem.x86_suffix();

    if ty.elem.is_float() {
        writeln!(
            code,
            "{}crate::impl_arithmetic_ops!({}, {}_add_{}, {}_sub_{}, {}_mul_{}, {}_div_{});",
            cfg_attr, name, prefix, suffix, prefix, suffix, prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "{}crate::impl_float_assign_ops!({});", cfg_attr, name).unwrap();
        writeln!(
            code,
            "{}crate::impl_neg!({}, {}_sub_{}, {}_setzero_{});",
            cfg_attr, name, prefix, suffix, prefix, suffix
        )
        .unwrap();
    } else {
        writeln!(
            code,
            "{}crate::impl_int_arithmetic_ops!({}, {}_add_{}, {}_sub_{});",
            cfg_attr, name, prefix, suffix, prefix, suffix
        )
        .unwrap();

        if matches!(
            ty.elem,
            ElementType::I16 | ElementType::U16 | ElementType::I32 | ElementType::U32
        ) {
            let mul_suffix = if matches!(ty.elem, ElementType::I16 | ElementType::U16) {
                "epi16"
            } else {
                "epi32"
            };
            writeln!(
                code,
                "{}crate::impl_int_mul_op!({}, {}_mullo_{});",
                cfg_attr, name, prefix, mul_suffix
            )
            .unwrap();
        }

        writeln!(code, "{}crate::impl_assign_ops!({});", cfg_attr, name).unwrap();
    }

    // Bitwise
    if ty.elem.is_float() {
        let and_fn = format!("{}_and_{}", prefix, suffix);
        let or_fn = format!("{}_or_{}", prefix, suffix);
        let xor_fn = format!("{}_xor_{}", prefix, suffix);
        writeln!(
            code,
            "{}crate::impl_bitwise_ops!({}, {}, {}, {}, {});",
            cfg_attr,
            name,
            ty.x86_inner_type(),
            and_fn,
            or_fn,
            xor_fn
        )
        .unwrap();
    } else {
        let width = ty.width.bits();
        writeln!(
            code,
            "{}crate::impl_bitwise_ops!({}, {}, {}_and_si{}, {}_or_si{}, {}_xor_si{});",
            cfg_attr,
            name,
            ty.x86_inner_type(),
            prefix,
            width,
            prefix,
            width,
            prefix,
            width
        )
        .unwrap();
    }

    // Index
    writeln!(
        code,
        "{}crate::impl_index!({}, {}, {});",
        cfg_attr,
        name,
        ty.elem.name(),
        ty.lanes()
    )
    .unwrap();

    code.push('\n');
    code
}
