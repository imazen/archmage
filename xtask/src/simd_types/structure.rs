//! Type structure and macro generation.

use super::block_ops;
use super::ops;
use super::ops_bitcast;
use super::transcendental;
use super::types::{ElementType, SimdType, SimdWidth};
use indoc::formatdoc;

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
    let name = ty.name();
    let lanes = ty.lanes();
    let elem = ty.elem.name();
    let inner = ty.x86_inner_type();
    let bits = ty.width.bits();

    let cfg_attr = match ty.width.required_feature() {
        Some(feat) => {
            format!("#[cfg(all(target_arch = \"x86_64\", feature = \"{feat}\"))]\n")
        }
        None => "#[cfg(target_arch = \"x86_64\")]\n".to_string(),
    };

    let mut code = formatdoc! {"

        // ============================================================================
        // {name} - {lanes} x {elem} ({bits}-bit)
        // ============================================================================

        {cfg_attr}#[derive(Clone, Copy, Debug)]
        #[repr(transparent)]
        pub struct {name}({inner});

        {cfg_attr}impl {name} {{
        pub const LANES: usize = {lanes};

    "};

    // Construction methods
    code.push_str(&generate_construction_methods(ty));

    // Math operations
    code.push_str(&ops::generate_math_ops(ty));

    // Comparison operations
    code.push_str(&ops::generate_comparison_ops(ty));

    // Blend/select operations
    code.push_str(&ops::generate_blend_ops(ty));

    // Boolean reductions (all_true, any_true, bitmask)
    code.push_str(&ops::generate_boolean_reductions(ty));

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

    // Bitcast operations (reinterpret bits between same-width types)
    code.push_str(&ops_bitcast::generate_x86_bitcasts(ty));

    code.push_str("}\n\n");

    // Operator implementations
    code.push_str(&generate_operator_impls(ty, &cfg_attr));

    // Scalar broadcast operators
    code.push_str(&ops::generate_scalar_ops(ty, &cfg_attr));

    code
}

/// Generate construction and extraction methods
fn generate_construction_methods(ty: &SimdType) -> String {
    let name = ty.name();
    let lanes = ty.lanes();
    let elem = ty.elem.name();
    let inner = ty.x86_inner_type();
    let token = ty.token();
    let prefix = ty.width.x86_prefix();
    let suffix = ty.elem.x86_suffix();
    let bits = ty.width.bits();
    let byte_size = ty.lanes() * ty.elem.size_bytes();
    let zero_lit = ty.elem.zero_literal();

    // Implementation name for identification (uses token tier, not width)
    let tier_name = match ty.width {
        SimdWidth::W128 | SimdWidth::W256 => "v3",
        SimdWidth::W512 => "v4",
    };
    let impl_name = format!("x86::{tier_name}::{name}");

    // Load intrinsic body
    let load_body = if ty.elem.is_float() {
        format!("Self(unsafe {{ {prefix}_loadu_{suffix}(data.as_ptr()) }})")
    } else {
        format!("Self(unsafe {{ {prefix}_loadu_si{bits}(data.as_ptr() as *const {inner}) }})")
    };

    // Splat intrinsic
    let (set1_suffix, cast) = match (ty.elem, ty.width) {
        (ElementType::I64 | ElementType::U64, SimdWidth::W512) => {
            ("epi64", ty.elem != ElementType::I64)
        }
        (ElementType::I64 | ElementType::U64, _) => ("epi64x", ty.elem != ElementType::I64),
        (ElementType::U8, _) => ("epi8", true),
        (ElementType::U16, _) => ("epi16", true),
        (ElementType::U32, _) => ("epi32", true),
        _ => (suffix, false),
    };

    let splat_body = if cast && !ty.elem.is_float() {
        let signed_ty = match ty.elem {
            ElementType::U8 => "i8",
            ElementType::U16 => "i16",
            ElementType::U32 => "i32",
            ElementType::U64 => "i64",
            _ => elem,
        };
        format!("Self(unsafe {{ {prefix}_set1_{set1_suffix}(v as {signed_ty}) }})")
    } else {
        format!("Self(unsafe {{ {prefix}_set1_{set1_suffix}(v) }})")
    };

    // Zero intrinsic
    let zero_body = if ty.elem.is_float() {
        format!("Self(unsafe {{ {prefix}_setzero_{suffix}() }})")
    } else {
        format!("Self(unsafe {{ {prefix}_setzero_si{bits}() }})")
    };

    // Store intrinsic
    let store_body = if ty.elem.is_float() {
        format!("unsafe {{ {prefix}_storeu_{suffix}(out.as_mut_ptr(), self.0) }};")
    } else {
        format!("unsafe {{ {prefix}_storeu_si{bits}(out.as_mut_ptr() as *mut {inner}, self.0) }};")
    };

    formatdoc! {"
        /// Load from array (token-gated)
        #[inline(always)]
        pub fn load(_: archmage::{token}, data: &[{elem}; {lanes}]) -> Self {{
        {load_body}
        }}

        /// Broadcast scalar to all lanes (token-gated)
        #[inline(always)]
        pub fn splat(_: archmage::{token}, v: {elem}) -> Self {{
        {splat_body}
        }}

        /// Zero vector (token-gated)
        #[inline(always)]
        pub fn zero(_: archmage::{token}) -> Self {{
        {zero_body}
        }}

        /// Create from array (token-gated, zero-cost)
        ///
        /// This is a zero-cost transmute, not a memory load.
        #[inline(always)]
        pub fn from_array(_: archmage::{token}, arr: [{elem}; {lanes}]) -> Self {{
        // SAFETY: [{elem}; {lanes}] and {inner} have identical size and layout
        Self(unsafe {{ core::mem::transmute(arr) }})
        }}

        /// Create from slice (token-gated).
        ///
        /// # Panics
        ///
        /// Panics if `slice.len() < {lanes}`.
        #[inline(always)]
        pub fn from_slice(_: archmage::{token}, slice: &[{elem}]) -> Self {{
        let arr: [{elem}; {lanes}] = slice[..{lanes}].try_into().unwrap();
        // SAFETY: [{elem}; {lanes}] and {inner} have identical size and layout
        Self(unsafe {{ core::mem::transmute(arr) }})
        }}

        /// Store to array
        #[inline(always)]
        pub fn store(self, out: &mut [{elem}; {lanes}]) {{
        {store_body}
        }}

        /// Convert to array
        #[inline(always)]
        pub fn to_array(self) -> [{elem}; {lanes}] {{
        let mut out = [{zero_lit}; {lanes}];
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
        /// Use token-gated constructors (`load`, `splat`, `zero`) for safe construction.
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
        pub fn cast_slice(_: archmage::{token}, slice: &[{elem}]) -> Option<&[Self]> {{
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
        pub fn cast_slice_mut(_: archmage::{token}, slice: &mut [{elem}]) -> Option<&mut [Self]> {{
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
        pub fn from_bytes(_: archmage::{token}, bytes: &[u8; {byte_size}]) -> Self {{
        // SAFETY: [u8; {byte_size}] and Self have identical size
        Self(unsafe {{ core::mem::transmute(*bytes) }})
        }}

        /// Create from an owned byte array (token-gated, zero-cost).
        ///
        /// This is a zero-cost transmute from an owned byte array.
        #[inline(always)]
        pub fn from_bytes_owned(_: archmage::{token}, bytes: [u8; {byte_size}]) -> Self {{
        // SAFETY: [u8; {byte_size}] and Self have identical size
        Self(unsafe {{ core::mem::transmute(bytes) }})
        }}

        // ========== Implementation identification ==========

        /// Returns a string identifying this type's implementation.
        ///
        /// This is useful for verifying that the correct implementation is being used
        /// at compile time or at runtime (via `#[magetypes]` dispatch).
        #[inline(always)]
        pub const fn implementation_name() -> &'static str {{
        \"{impl_name}\"
        }}

    "}
}

/// Generate operator trait implementations
fn generate_operator_impls(ty: &SimdType, cfg_attr: &str) -> String {
    let name = ty.name();
    let prefix = ty.width.x86_prefix();
    let suffix = ty.elem.x86_suffix();
    let inner = ty.x86_inner_type();
    let elem = ty.elem.name();
    let lanes = ty.lanes();
    let width = ty.width.bits();

    let mut code = String::new();

    if ty.elem.is_float() {
        code.push_str(&formatdoc! {"
            {cfg_attr}crate::impl_arithmetic_ops!({name}, {prefix}_add_{suffix}, {prefix}_sub_{suffix}, {prefix}_mul_{suffix}, {prefix}_div_{suffix});
            {cfg_attr}crate::impl_float_assign_ops!({name});
            {cfg_attr}crate::impl_neg!({name}, {prefix}_sub_{suffix}, {prefix}_setzero_{suffix});
        "});
    } else {
        code.push_str(&formatdoc! {"
            {cfg_attr}crate::impl_int_arithmetic_ops!({name}, {prefix}_add_{suffix}, {prefix}_sub_{suffix});
        "});

        if matches!(
            ty.elem,
            ElementType::I16 | ElementType::U16 | ElementType::I32 | ElementType::U32
        ) {
            let mul_suffix = if matches!(ty.elem, ElementType::I16 | ElementType::U16) {
                "epi16"
            } else {
                "epi32"
            };
            code.push_str(&format!(
                "{cfg_attr}crate::impl_int_mul_op!({name}, {prefix}_mullo_{mul_suffix});\n"
            ));
        }

        code.push_str(&format!("{cfg_attr}crate::impl_assign_ops!({name});\n"));
    }

    // Bitwise
    if ty.elem.is_float() {
        code.push_str(&formatdoc! {"
            {cfg_attr}crate::impl_bitwise_ops!({name}, {inner}, {prefix}_and_{suffix}, {prefix}_or_{suffix}, {prefix}_xor_{suffix});
        "});
    } else {
        code.push_str(&formatdoc! {"
            {cfg_attr}crate::impl_bitwise_ops!({name}, {inner}, {prefix}_and_si{width}, {prefix}_or_si{width}, {prefix}_xor_si{width});
        "});
    }

    // Index
    code.push_str(&format!(
        "{cfg_attr}crate::impl_index!({name}, {elem}, {lanes});\n"
    ));

    // Into<[T; N]> implementation (extraction is always safe — no intrinsics needed)
    // From<[T; N]> is deliberately NOT implemented for x86 types:
    // construction requires a token to prove CPU feature availability.
    // Use from_array(token, arr) or from_slice(token, slice) instead.
    code.push_str(&formatdoc! {"

        {cfg_attr}impl From<{name}> for [{elem}; {lanes}] {{
        #[inline(always)]
        fn from(v: {name}) -> Self {{
        // SAFETY: {inner} and [{elem}; {lanes}] have identical size and layout
        unsafe {{ core::mem::transmute(v.0) }}
        }}
        }}

    "});

    code
}
