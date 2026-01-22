//! SIMD type generation for wide-like ergonomic types.
//!
//! Generates token-gated SIMD types with:
//! - Safe construction (requires capability token)
//! - Operator overloads (+, -, *, /, &, |, ^, etc.)
//! - Math methods (min, max, abs, sqrt, fma, etc.)
//! - Integration with archmage::mem for load/store

use std::fmt::Write;

/// Element type for SIMD vectors
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ElementType {
    F32,
    F64,
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
}

impl ElementType {
    pub fn name(&self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F64 => "f64",
            Self::I8 => "i8",
            Self::U8 => "u8",
            Self::I16 => "i16",
            Self::U16 => "u16",
            Self::I32 => "i32",
            Self::U32 => "u32",
            Self::I64 => "i64",
            Self::U64 => "u64",
        }
    }

    pub fn size_bytes(&self) -> usize {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F64 | Self::I64 | Self::U64 => 8,
            Self::I16 | Self::U16 => 2,
            Self::I8 | Self::U8 => 1,
        }
    }

    pub fn is_float(&self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }

    pub fn is_signed(&self) -> bool {
        matches!(
            self,
            Self::F32 | Self::F64 | Self::I8 | Self::I16 | Self::I32 | Self::I64
        )
    }

    /// Intrinsic suffix for this type (e.g., "ps" for f32, "epi32" for i32)
    /// Note: Unsigned integers use signed intrinsics (bit patterns are identical)
    pub fn x86_suffix(&self) -> &'static str {
        match self {
            Self::F32 => "ps",
            Self::F64 => "pd",
            Self::I8 | Self::U8 => "epi8",
            Self::I16 | Self::U16 => "epi16",
            Self::I32 | Self::U32 => "epi32",
            Self::I64 | Self::U64 => "epi64",
        }
    }

    /// Intrinsic suffix for min/max operations (differs for signed vs unsigned)
    pub fn x86_minmax_suffix(&self) -> &'static str {
        match self {
            Self::F32 => "ps",
            Self::F64 => "pd",
            Self::I8 => "epi8",
            Self::U8 => "epu8",
            Self::I16 => "epi16",
            Self::U16 => "epu16",
            Self::I32 => "epi32",
            Self::U32 => "epu32",
            Self::I64 => "epi64",
            Self::U64 => "epu64",
        }
    }

    /// Default value literal
    pub fn zero_literal(&self) -> &'static str {
        match self {
            Self::F32 => "0.0f32",
            Self::F64 => "0.0f64",
            Self::I8 => "0i8",
            Self::U8 => "0u8",
            Self::I16 => "0i16",
            Self::U16 => "0u16",
            Self::I32 => "0i32",
            Self::U32 => "0u32",
            Self::I64 => "0i64",
            Self::U64 => "0u64",
        }
    }
}

/// SIMD register width
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimdWidth {
    W128, // SSE, NEON
    W256, // AVX/AVX2
    W512, // AVX-512
}

impl SimdWidth {
    pub fn bits(&self) -> usize {
        match self {
            Self::W128 => 128,
            Self::W256 => 256,
            Self::W512 => 512,
        }
    }

    pub fn bytes(&self) -> usize {
        self.bits() / 8
    }

    /// x86 intrinsic type for floats
    pub fn x86_float_type(&self, elem: ElementType) -> &'static str {
        match (self, elem) {
            (Self::W128, ElementType::F32) => "__m128",
            (Self::W128, ElementType::F64) => "__m128d",
            (Self::W256, ElementType::F32) => "__m256",
            (Self::W256, ElementType::F64) => "__m256d",
            (Self::W512, ElementType::F32) => "__m512",
            (Self::W512, ElementType::F64) => "__m512d",
            _ => panic!("Invalid float type for width"),
        }
    }

    /// x86 intrinsic type for integers
    pub fn x86_int_type(&self) -> &'static str {
        match self {
            Self::W128 => "__m128i",
            Self::W256 => "__m256i",
            Self::W512 => "__m512i",
        }
    }

    /// Intrinsic prefix (mm, mm256, mm512)
    pub fn x86_prefix(&self) -> &'static str {
        match self {
            Self::W128 => "_mm",
            Self::W256 => "_mm256",
            Self::W512 => "_mm512",
        }
    }

    /// Required token for this width
    pub fn required_token(&self, needs_int_ops: bool) -> &'static str {
        match self {
            Self::W128 => "Sse41Token", // SSE4.1 for most useful ops
            Self::W256 => {
                if needs_int_ops {
                    "Avx2FmaToken" // AVX2 for integer ops
                } else {
                    "Avx2FmaToken" // Use AVX2+FMA for consistency
                }
            }
            Self::W512 => "X64V4Token",
        }
    }

    /// Required feature flag
    pub fn required_feature(&self) -> Option<&'static str> {
        match self {
            Self::W128 | Self::W256 => None,
            Self::W512 => Some("avx512"),
        }
    }
}

/// A SIMD vector type to generate
#[derive(Clone, Debug)]
pub struct SimdType {
    pub elem: ElementType,
    pub width: SimdWidth,
}

impl SimdType {
    pub fn new(elem: ElementType, width: SimdWidth) -> Self {
        Self { elem, width }
    }

    /// Number of lanes
    pub fn lanes(&self) -> usize {
        self.width.bytes() / self.elem.size_bytes()
    }

    /// Type name (e.g., "f32x8")
    pub fn name(&self) -> String {
        format!("{}x{}", self.elem.name(), self.lanes())
    }

    /// Constants struct name (e.g., "F32x8Consts")
    pub fn consts_name(&self) -> String {
        let elem_upper = match self.elem {
            ElementType::F32 => "F32",
            ElementType::F64 => "F64",
            ElementType::I8 => "I8",
            ElementType::U8 => "U8",
            ElementType::I16 => "I16",
            ElementType::U16 => "U16",
            ElementType::I32 => "I32",
            ElementType::U32 => "U32",
            ElementType::I64 => "I64",
            ElementType::U64 => "U64",
        };
        format!("{}x{}Consts", elem_upper, self.lanes())
    }

    /// Underlying x86 intrinsic type
    pub fn x86_inner_type(&self) -> &'static str {
        if self.elem.is_float() {
            self.width.x86_float_type(self.elem)
        } else {
            self.width.x86_int_type()
        }
    }

    /// Token required for construction
    pub fn token(&self) -> &'static str {
        self.width.required_token(!self.elem.is_float())
    }
}

/// Intrinsic mapping for a specific operation
pub struct IntrinsicOp {
    pub name: &'static str,
    pub x86_128: Option<&'static str>,
    pub x86_256: Option<&'static str>,
    pub x86_512: Option<&'static str>,
}

// ============================================================================
// Code Generation
// ============================================================================

/// Generate all SIMD types
pub fn generate_simd_types() -> String {
    let mut code = String::new();

    // Header
    code.push_str(
        r#"//! Token-gated SIMD types with natural operators.
//!
//! Provides `wide`-like ergonomics with token-gated construction.
//! There is NO way to construct these types without proving CPU support.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

#![allow(dead_code)]
#![allow(non_camel_case_types)]
#![allow(missing_docs)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use core::ops::{
    Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign,
    Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

"#,
    );

    // Generate comparison traits
    code.push_str(&generate_comparison_traits());

    // Generate macros
    code.push_str(&generate_macros());

    // Define all types to generate
    let types = [
        // 128-bit (SSE)
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
        // 256-bit (AVX/AVX2)
        SimdType::new(ElementType::F32, SimdWidth::W256),
        SimdType::new(ElementType::F64, SimdWidth::W256),
        SimdType::new(ElementType::I8, SimdWidth::W256),
        SimdType::new(ElementType::U8, SimdWidth::W256),
        SimdType::new(ElementType::I16, SimdWidth::W256),
        SimdType::new(ElementType::U16, SimdWidth::W256),
        SimdType::new(ElementType::I32, SimdWidth::W256),
        SimdType::new(ElementType::U32, SimdWidth::W256),
        SimdType::new(ElementType::I64, SimdWidth::W256),
        SimdType::new(ElementType::U64, SimdWidth::W256),
        // 512-bit (AVX-512)
        SimdType::new(ElementType::F32, SimdWidth::W512),
        SimdType::new(ElementType::F64, SimdWidth::W512),
        SimdType::new(ElementType::I8, SimdWidth::W512),
        SimdType::new(ElementType::U8, SimdWidth::W512),
        SimdType::new(ElementType::I16, SimdWidth::W512),
        SimdType::new(ElementType::U16, SimdWidth::W512),
        SimdType::new(ElementType::I32, SimdWidth::W512),
        SimdType::new(ElementType::U32, SimdWidth::W512),
        SimdType::new(ElementType::I64, SimdWidth::W512),
        SimdType::new(ElementType::U64, SimdWidth::W512),
    ];

    // Generate each type
    for ty in &types {
        code.push_str(&generate_type(ty));
    }

    code
}

fn generate_comparison_traits() -> String {
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

fn generate_macros() -> String {
    r#"
// ============================================================================
// Implementation Macros
// ============================================================================

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

fn generate_type(ty: &SimdType) -> String {
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

    // Token-gated construction
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
        // Integer types need cast
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
    // 64-bit integers use epi64x for 128/256-bit, but epi64 for 512-bit
    // Unsigned types need cast to signed for set1 intrinsic
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

    // Extraction methods (don't need token)
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

    writeln!(code, "    /// Get raw intrinsic type").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn raw(self) -> {} {{", inner).unwrap();
    writeln!(code, "        self.0").unwrap();
    writeln!(code, "    }}\n").unwrap();

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

    // Math operations
    code.push_str(&generate_math_ops(ty));

    // Comparison operations
    code.push_str(&generate_comparison_ops(ty));

    // Blend/select operations
    code.push_str(&generate_blend_ops(ty));

    // Horizontal operations (reduce_add, reduce_min, reduce_max)
    code.push_str(&generate_horizontal_ops(ty));

    // Type conversions (f32 <-> i32, etc.)
    code.push_str(&generate_conversion_ops(ty));

    // Approximation operations (rcp, rsqrt with Newton-Raphson)
    code.push_str(&generate_approx_ops(ty));

    // Bitwise unary operations (not)
    code.push_str(&generate_bitwise_unary_ops(ty));

    // Shift operations (shl, shr, shr_arithmetic)
    code.push_str(&generate_shift_ops(ty));

    // Transcendental operations (log2, exp2, ln, exp)
    code.push_str(&generate_transcendental_ops(ty));

    writeln!(code, "}}\n").unwrap();

    // Operator implementations
    code.push_str(&generate_operator_impls(ty, &cfg_attr));

    // Scalar broadcast operators (v + 2.0)
    code.push_str(&generate_scalar_ops(ty, &cfg_attr));

    code
}

fn generate_comparison_ops(ty: &SimdType) -> String {
    let mut code = String::new();
    let name = ty.name();
    let prefix = ty.width.x86_prefix();
    let suffix = ty.elem.x86_suffix();
    let lanes = ty.lanes();

    // Section header
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

    if ty.elem.is_float() {
        // Float comparisons use _mm*_cmp_ps/pd with comparison predicate
        // AVX-512 uses mask registers but we'll use the vector-returning variants for consistency
        let cmp_intrinsic = format!("{}_cmp_{}", prefix, suffix);

        // For SSE, comparisons are different intrinsics, not cmp with predicate
        if ty.width == SimdWidth::W128 {
            // SSE uses separate intrinsics
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
            writeln!(
                code,
                "        Self(unsafe {{ {}_cmpeq_{}(self.0, other.0) }})",
                prefix, suffix
            )
            .unwrap();
            writeln!(code, "    }}\n").unwrap();

            writeln!(code, "    /// Lane-wise inequality comparison.").unwrap();
            writeln!(code, "    ///").unwrap();
            writeln!(
                code,
                "    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise."
            )
            .unwrap();
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(code, "    pub fn simd_ne(self, other: Self) -> Self {{").unwrap();
            writeln!(
                code,
                "        Self(unsafe {{ {}_cmpneq_{}(self.0, other.0) }})",
                prefix, suffix
            )
            .unwrap();
            writeln!(code, "    }}\n").unwrap();

            writeln!(code, "    /// Lane-wise less-than comparison.").unwrap();
            writeln!(code, "    ///").unwrap();
            writeln!(code, "    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.").unwrap();
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(code, "    pub fn simd_lt(self, other: Self) -> Self {{").unwrap();
            writeln!(
                code,
                "        Self(unsafe {{ {}_cmplt_{}(self.0, other.0) }})",
                prefix, suffix
            )
            .unwrap();
            writeln!(code, "    }}\n").unwrap();

            writeln!(code, "    /// Lane-wise less-than-or-equal comparison.").unwrap();
            writeln!(code, "    ///").unwrap();
            writeln!(code, "    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.").unwrap();
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(code, "    pub fn simd_le(self, other: Self) -> Self {{").unwrap();
            writeln!(
                code,
                "        Self(unsafe {{ {}_cmple_{}(self.0, other.0) }})",
                prefix, suffix
            )
            .unwrap();
            writeln!(code, "    }}\n").unwrap();

            writeln!(code, "    /// Lane-wise greater-than comparison.").unwrap();
            writeln!(code, "    ///").unwrap();
            writeln!(code, "    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.").unwrap();
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(code, "    pub fn simd_gt(self, other: Self) -> Self {{").unwrap();
            writeln!(
                code,
                "        Self(unsafe {{ {}_cmpgt_{}(self.0, other.0) }})",
                prefix, suffix
            )
            .unwrap();
            writeln!(code, "    }}\n").unwrap();

            writeln!(code, "    /// Lane-wise greater-than-or-equal comparison.").unwrap();
            writeln!(code, "    ///").unwrap();
            writeln!(code, "    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.").unwrap();
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(code, "    pub fn simd_ge(self, other: Self) -> Self {{").unwrap();
            writeln!(
                code,
                "        Self(unsafe {{ {}_cmpge_{}(self.0, other.0) }})",
                prefix, suffix
            )
            .unwrap();
            writeln!(code, "    }}\n").unwrap();
        } else if ty.width == SimdWidth::W512 {
            // AVX-512 uses mask-returning intrinsics, need to expand mask to vector
            // Pattern: get mask, then use mask_blend to create all-1s where true
            let mask_suffix = if ty.elem == ElementType::F32 {
                "ps"
            } else {
                "pd"
            };
            let int_suffix = if ty.elem == ElementType::F32 {
                "epi32"
            } else {
                "epi64"
            };
            let cast_fn = if ty.elem == ElementType::F32 {
                "_mm512_castsi512_ps"
            } else {
                "_mm512_castsi512_pd"
            };

            for (method, doc, cmp_const) in [
                ("simd_eq", "equality", "_CMP_EQ_OQ"),
                ("simd_ne", "inequality", "_CMP_NEQ_OQ"),
                ("simd_lt", "less-than", "_CMP_LT_OQ"),
                ("simd_le", "less-than-or-equal", "_CMP_LE_OQ"),
                ("simd_gt", "greater-than", "_CMP_GT_OQ"),
                ("simd_ge", "greater-than-or-equal", "_CMP_GE_OQ"),
            ] {
                writeln!(code, "    /// Lane-wise {} comparison.", doc).unwrap();
                writeln!(code, "    ///").unwrap();
                writeln!(
                    code,
                    "    /// Returns a mask where each lane is all-1s if {}, all-0s otherwise.",
                    doc
                )
                .unwrap();
                if method == "simd_eq" {
                    writeln!(
                        code,
                        "    /// Use with `blend(mask, if_true, if_false)` to select values."
                    )
                    .unwrap();
                }
                writeln!(code, "    #[inline(always)]").unwrap();
                writeln!(code, "    pub fn {}(self, other: Self) -> Self {{", method).unwrap();
                writeln!(code, "        Self(unsafe {{").unwrap();
                writeln!(
                    code,
                    "            let mask = {}_cmp_{}_mask::<{}>{}(self.0, other.0);",
                    prefix, mask_suffix, cmp_const, ""
                )
                .unwrap();
                writeln!(
                    code,
                    "            // Expand mask to vector: -1 where true, 0 where false"
                )
                .unwrap();
                writeln!(
                    code,
                    "            {}({}_maskz_set1_{}(mask, -1))",
                    cast_fn, prefix, int_suffix
                )
                .unwrap();
                writeln!(code, "        }})").unwrap();
                writeln!(code, "    }}\n").unwrap();
            }
        } else {
            // AVX (256-bit) uses _cmp_ps/pd with predicate constants, returns vector directly
            // _CMP_EQ_OQ = 0, _CMP_LT_OQ = 17, _CMP_LE_OQ = 18, _CMP_NEQ_OQ = 12, _CMP_GE_OQ = 29, _CMP_GT_OQ = 30
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
            writeln!(
                code,
                "        Self(unsafe {{ {}::<_CMP_EQ_OQ>(self.0, other.0) }})",
                cmp_intrinsic
            )
            .unwrap();
            writeln!(code, "    }}\n").unwrap();

            writeln!(code, "    /// Lane-wise inequality comparison.").unwrap();
            writeln!(code, "    ///").unwrap();
            writeln!(
                code,
                "    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise."
            )
            .unwrap();
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(code, "    pub fn simd_ne(self, other: Self) -> Self {{").unwrap();
            writeln!(
                code,
                "        Self(unsafe {{ {}::<_CMP_NEQ_OQ>(self.0, other.0) }})",
                cmp_intrinsic
            )
            .unwrap();
            writeln!(code, "    }}\n").unwrap();

            writeln!(code, "    /// Lane-wise less-than comparison.").unwrap();
            writeln!(code, "    ///").unwrap();
            writeln!(code, "    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.").unwrap();
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(code, "    pub fn simd_lt(self, other: Self) -> Self {{").unwrap();
            writeln!(
                code,
                "        Self(unsafe {{ {}::<_CMP_LT_OQ>(self.0, other.0) }})",
                cmp_intrinsic
            )
            .unwrap();
            writeln!(code, "    }}\n").unwrap();

            writeln!(code, "    /// Lane-wise less-than-or-equal comparison.").unwrap();
            writeln!(code, "    ///").unwrap();
            writeln!(code, "    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.").unwrap();
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(code, "    pub fn simd_le(self, other: Self) -> Self {{").unwrap();
            writeln!(
                code,
                "        Self(unsafe {{ {}::<_CMP_LE_OQ>(self.0, other.0) }})",
                cmp_intrinsic
            )
            .unwrap();
            writeln!(code, "    }}\n").unwrap();

            writeln!(code, "    /// Lane-wise greater-than comparison.").unwrap();
            writeln!(code, "    ///").unwrap();
            writeln!(code, "    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.").unwrap();
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(code, "    pub fn simd_gt(self, other: Self) -> Self {{").unwrap();
            writeln!(
                code,
                "        Self(unsafe {{ {}::<_CMP_GT_OQ>(self.0, other.0) }})",
                cmp_intrinsic
            )
            .unwrap();
            writeln!(code, "    }}\n").unwrap();

            writeln!(code, "    /// Lane-wise greater-than-or-equal comparison.").unwrap();
            writeln!(code, "    ///").unwrap();
            writeln!(code, "    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.").unwrap();
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(code, "    pub fn simd_ge(self, other: Self) -> Self {{").unwrap();
            writeln!(
                code,
                "        Self(unsafe {{ {}::<_CMP_GE_OQ>(self.0, other.0) }})",
                cmp_intrinsic
            )
            .unwrap();
            writeln!(code, "    }}\n").unwrap();
        }
    } else if ty.width == SimdWidth::W512 {
        // AVX-512 integer comparisons return mask registers, need to expand to vector
        // For signed: use cmp_*_mask intrinsics with _MM_CMPINT_*
        // For unsigned: use cmpu*_mask intrinsics
        let set1_suffix = match ty.elem {
            ElementType::I8 | ElementType::U8 => "epi8",
            ElementType::I16 | ElementType::U16 => "epi16",
            ElementType::I32 | ElementType::U32 => "epi32",
            ElementType::I64 | ElementType::U64 => "epi64",
            _ => unreachable!(),
        };

        // AVX-512 has direct unsigned comparison intrinsics
        let cmp_fn = if ty.elem.is_signed() {
            format!("{}_cmp_{}_mask", prefix, suffix)
        } else {
            // Unsigned uses epu* variants
            let unsigned_suffix = match ty.elem {
                ElementType::U8 => "epu8",
                ElementType::U16 => "epu16",
                ElementType::U32 => "epu32",
                ElementType::U64 => "epu64",
                _ => unreachable!(),
            };
            format!("{}_cmp_{}_mask", prefix, unsigned_suffix)
        };

        for (method, doc, cmp_const) in [
            ("simd_eq", "equality", "_MM_CMPINT_EQ"),
            ("simd_ne", "inequality", "_MM_CMPINT_NE"),
            ("simd_lt", "less-than", "_MM_CMPINT_LT"),
            ("simd_le", "less-than-or-equal", "_MM_CMPINT_LE"),
            ("simd_gt", "greater-than", "_MM_CMPINT_NLT"), // NLT on swapped args = GT
            ("simd_ge", "greater-than-or-equal", "_MM_CMPINT_NLE"), // NLE on swapped args = GE
        ] {
            writeln!(code, "    /// Lane-wise {} comparison.", doc).unwrap();
            writeln!(code, "    ///").unwrap();
            writeln!(
                code,
                "    /// Returns a mask where each lane is all-1s if {}, all-0s otherwise.",
                doc
            )
            .unwrap();
            if method == "simd_eq" {
                writeln!(
                    code,
                    "    /// Use with `blend(mask, if_true, if_false)` to select values."
                )
                .unwrap();
            }
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(code, "    pub fn {}(self, other: Self) -> Self {{", method).unwrap();
            writeln!(code, "        Self(unsafe {{").unwrap();
            // GT and GE need swapped arguments with NLT/NLE
            if method == "simd_gt" || method == "simd_ge" {
                let actual_cmp = if method == "simd_gt" {
                    "_MM_CMPINT_LT"
                } else {
                    "_MM_CMPINT_LE"
                };
                writeln!(
                    code,
                    "            let mask = {}::<{}>(other.0, self.0);",
                    cmp_fn, actual_cmp
                )
                .unwrap();
            } else {
                writeln!(
                    code,
                    "            let mask = {}::<{}>(self.0, other.0);",
                    cmp_fn, cmp_const
                )
                .unwrap();
            }
            writeln!(
                code,
                "            // Expand mask to vector: -1 where true, 0 where false"
            )
            .unwrap();
            writeln!(
                code,
                "            {}_maskz_set1_{}(mask, -1)",
                prefix, set1_suffix
            )
            .unwrap();
            writeln!(code, "        }})").unwrap();
            writeln!(code, "    }}\n").unwrap();
        }
    } else {
        // SSE/AVX integer comparisons return vectors directly
        // cmpeq exists for all, cmpgt exists for signed
        // For unsigned comparisons and lt/le/ge, we derive from cmpgt

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
        writeln!(
            code,
            "        Self(unsafe {{ {}_cmpeq_{}(self.0, other.0) }})",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "    }}\n").unwrap();

        writeln!(code, "    /// Lane-wise inequality comparison.").unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Returns a mask where each lane is all-1s if not equal, all-0s otherwise."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn simd_ne(self, other: Self) -> Self {{").unwrap();
        // ne = NOT eq, so XOR with all-1s
        writeln!(code, "        Self(unsafe {{").unwrap();
        writeln!(
            code,
            "            let eq = {}_cmpeq_{}(self.0, other.0);",
            prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let ones = {}_set1_{}(-1);",
            prefix,
            match ty.elem {
                ElementType::I8 | ElementType::U8 => "epi8",
                ElementType::I16 | ElementType::U16 => "epi16",
                ElementType::I32 | ElementType::U32 => "epi32",
                ElementType::I64 | ElementType::U64 => "epi64x",
                _ => unreachable!(),
            }
        )
        .unwrap();
        writeln!(
            code,
            "            {}_xor_si{}(eq, ones)",
            prefix,
            ty.width.bits()
        )
        .unwrap();
        writeln!(code, "        }})").unwrap();
        writeln!(code, "    }}\n").unwrap();

        if ty.elem.is_signed() {
            // Signed integers have cmpgt directly
            writeln!(code, "    /// Lane-wise greater-than comparison.").unwrap();
            writeln!(code, "    ///").unwrap();
            writeln!(code, "    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.").unwrap();
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(code, "    pub fn simd_gt(self, other: Self) -> Self {{").unwrap();
            writeln!(
                code,
                "        Self(unsafe {{ {}_cmpgt_{}(self.0, other.0) }})",
                prefix, suffix
            )
            .unwrap();
            writeln!(code, "    }}\n").unwrap();

            writeln!(code, "    /// Lane-wise less-than comparison.").unwrap();
            writeln!(code, "    ///").unwrap();
            writeln!(code, "    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.").unwrap();
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(code, "    pub fn simd_lt(self, other: Self) -> Self {{").unwrap();
            // lt(a, b) = gt(b, a)
            writeln!(
                code,
                "        Self(unsafe {{ {}_cmpgt_{}(other.0, self.0) }})",
                prefix, suffix
            )
            .unwrap();
            writeln!(code, "    }}\n").unwrap();

            writeln!(code, "    /// Lane-wise greater-than-or-equal comparison.").unwrap();
            writeln!(code, "    ///").unwrap();
            writeln!(code, "    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.").unwrap();
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(code, "    pub fn simd_ge(self, other: Self) -> Self {{").unwrap();
            // ge(a, b) = NOT lt(a, b) = NOT gt(b, a)
            writeln!(code, "        Self(unsafe {{").unwrap();
            writeln!(
                code,
                "            let lt = {}_cmpgt_{}(other.0, self.0);",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let ones = {}_set1_{}(-1);",
                prefix,
                match ty.elem {
                    ElementType::I8 => "epi8",
                    ElementType::I16 => "epi16",
                    ElementType::I32 => "epi32",
                    ElementType::I64 => "epi64x",
                    _ => unreachable!(),
                }
            )
            .unwrap();
            writeln!(
                code,
                "            {}_xor_si{}(lt, ones)",
                prefix,
                ty.width.bits()
            )
            .unwrap();
            writeln!(code, "        }})").unwrap();
            writeln!(code, "    }}\n").unwrap();

            writeln!(code, "    /// Lane-wise less-than-or-equal comparison.").unwrap();
            writeln!(code, "    ///").unwrap();
            writeln!(code, "    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.").unwrap();
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(code, "    pub fn simd_le(self, other: Self) -> Self {{").unwrap();
            // le(a, b) = NOT gt(a, b)
            writeln!(code, "        Self(unsafe {{").unwrap();
            writeln!(
                code,
                "            let gt = {}_cmpgt_{}(self.0, other.0);",
                prefix, suffix
            )
            .unwrap();
            writeln!(
                code,
                "            let ones = {}_set1_{}(-1);",
                prefix,
                match ty.elem {
                    ElementType::I8 => "epi8",
                    ElementType::I16 => "epi16",
                    ElementType::I32 => "epi32",
                    ElementType::I64 => "epi64x",
                    _ => unreachable!(),
                }
            )
            .unwrap();
            writeln!(
                code,
                "            {}_xor_si{}(gt, ones)",
                prefix,
                ty.width.bits()
            )
            .unwrap();
            writeln!(code, "        }})").unwrap();
            writeln!(code, "    }}\n").unwrap();
        } else {
            // Unsigned comparisons - use signed comparison with bias trick
            // For unsigned compare: XOR with 0x80..80 to flip sign bit, then do signed compare
            let bias_val = match ty.elem {
                ElementType::U8 => "0x80u8 as i8",
                ElementType::U16 => "0x8000u16 as i16",
                ElementType::U32 => "0x8000_0000u32 as i32",
                ElementType::U64 => "0x8000_0000_0000_0000u64 as i64",
                _ => unreachable!(),
            };
            let signed_suffix = match ty.elem {
                ElementType::U8 => "epi8",
                ElementType::U16 => "epi16",
                ElementType::U32 => "epi32",
                ElementType::U64 => "epi64",
                _ => unreachable!(),
            };
            let set1_suffix = match ty.elem {
                ElementType::U64 => "epi64x",
                _ => signed_suffix,
            };

            writeln!(
                code,
                "    /// Lane-wise greater-than comparison (unsigned)."
            )
            .unwrap();
            writeln!(code, "    ///").unwrap();
            writeln!(code, "    /// Returns a mask where each lane is all-1s if self > other, all-0s otherwise.").unwrap();
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(code, "    pub fn simd_gt(self, other: Self) -> Self {{").unwrap();
            writeln!(code, "        Self(unsafe {{").unwrap();
            writeln!(
                code,
                "            // Flip sign bit to convert unsigned to signed comparison"
            )
            .unwrap();
            writeln!(
                code,
                "            let bias = {}_set1_{}({});",
                prefix, set1_suffix, bias_val
            )
            .unwrap();
            writeln!(
                code,
                "            let a = {}_xor_si{}(self.0, bias);",
                prefix,
                ty.width.bits()
            )
            .unwrap();
            writeln!(
                code,
                "            let b = {}_xor_si{}(other.0, bias);",
                prefix,
                ty.width.bits()
            )
            .unwrap();
            writeln!(code, "            {}_cmpgt_{}(a, b)", prefix, signed_suffix).unwrap();
            writeln!(code, "        }})").unwrap();
            writeln!(code, "    }}\n").unwrap();

            writeln!(code, "    /// Lane-wise less-than comparison (unsigned).").unwrap();
            writeln!(code, "    ///").unwrap();
            writeln!(code, "    /// Returns a mask where each lane is all-1s if self < other, all-0s otherwise.").unwrap();
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(code, "    pub fn simd_lt(self, other: Self) -> Self {{").unwrap();
            writeln!(code, "        other.simd_gt(self)").unwrap();
            writeln!(code, "    }}\n").unwrap();

            writeln!(
                code,
                "    /// Lane-wise greater-than-or-equal comparison (unsigned)."
            )
            .unwrap();
            writeln!(code, "    ///").unwrap();
            writeln!(code, "    /// Returns a mask where each lane is all-1s if self >= other, all-0s otherwise.").unwrap();
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(code, "    pub fn simd_ge(self, other: Self) -> Self {{").unwrap();
            writeln!(code, "        Self(unsafe {{").unwrap();
            writeln!(code, "            let lt = other.simd_gt(self);").unwrap();
            writeln!(
                code,
                "            let ones = {}_set1_{}(-1);",
                prefix, set1_suffix
            )
            .unwrap();
            writeln!(
                code,
                "            {}_xor_si{}(lt.0, ones)",
                prefix,
                ty.width.bits()
            )
            .unwrap();
            writeln!(code, "        }})").unwrap();
            writeln!(code, "    }}\n").unwrap();

            writeln!(
                code,
                "    /// Lane-wise less-than-or-equal comparison (unsigned)."
            )
            .unwrap();
            writeln!(code, "    ///").unwrap();
            writeln!(code, "    /// Returns a mask where each lane is all-1s if self <= other, all-0s otherwise.").unwrap();
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(code, "    pub fn simd_le(self, other: Self) -> Self {{").unwrap();
            writeln!(code, "        Self(unsafe {{").unwrap();
            writeln!(code, "            let gt = self.simd_gt(other);").unwrap();
            writeln!(
                code,
                "            let ones = {}_set1_{}(-1);",
                prefix, set1_suffix
            )
            .unwrap();
            writeln!(
                code,
                "            {}_xor_si{}(gt.0, ones)",
                prefix,
                ty.width.bits()
            )
            .unwrap();
            writeln!(code, "        }})").unwrap();
            writeln!(code, "    }}\n").unwrap();
        }
    }

    code
}

fn generate_blend_ops(ty: &SimdType) -> String {
    let mut code = String::new();
    let prefix = ty.width.x86_prefix();
    let suffix = ty.elem.x86_suffix();

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
    writeln!(code, "    /// let a = {}::splat(token, 1.0);", ty.name()).unwrap();
    writeln!(code, "    /// let b = {}::splat(token, 2.0);", ty.name()).unwrap();
    writeln!(code, "    /// let mask = a.simd_lt(b);  // all true").unwrap();
    writeln!(
        code,
        "    /// let result = {}::blend(mask, a, b);  // selects a",
        ty.name()
    )
    .unwrap();
    writeln!(code, "    /// ```").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(
        code,
        "    pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {{"
    )
    .unwrap();

    if ty.width == SimdWidth::W512 {
        // AVX-512 uses mask registers for blend
        // Convert vector mask to mask register by comparing against zero
        if ty.elem.is_float() {
            let (int_suffix, cast_fn) = if ty.elem == ElementType::F32 {
                ("epi32", "_mm512_castps_si512")
            } else {
                ("epi64", "_mm512_castpd_si512")
            };
            writeln!(code, "        Self(unsafe {{").unwrap();
            writeln!(code, "            // Convert vector mask to mask register").unwrap();
            writeln!(
                code,
                "            let m = _mm512_cmpneq_{}_mask({}(mask.0), _mm512_setzero_si512());",
                int_suffix, cast_fn
            )
            .unwrap();
            writeln!(
                code,
                "            {}_mask_blend_{}(m, if_false.0, if_true.0)",
                prefix, suffix
            )
            .unwrap();
            writeln!(code, "        }})").unwrap();
        } else {
            // Integer types
            writeln!(code, "        Self(unsafe {{").unwrap();
            writeln!(code, "            // Convert vector mask to mask register").unwrap();
            writeln!(
                code,
                "            let m = _mm512_cmpneq_{}_mask(mask.0, _mm512_setzero_si512());",
                suffix
            )
            .unwrap();
            writeln!(
                code,
                "            {}_mask_blend_{}(m, if_false.0, if_true.0)",
                prefix, suffix
            )
            .unwrap();
            writeln!(code, "        }})").unwrap();
        }
    } else if ty.elem.is_float() {
        writeln!(
            code,
            "        Self(unsafe {{ {}_blendv_{}(if_false.0, if_true.0, mask.0) }})",
            prefix, suffix
        )
        .unwrap();
    } else {
        // Integer blend uses blendv_epi8 which operates on bytes
        writeln!(
            code,
            "        Self(unsafe {{ {}_blendv_epi8(if_false.0, if_true.0, mask.0) }})",
            prefix
        )
        .unwrap();
    }
    writeln!(code, "    }}\n").unwrap();

    code
}

fn generate_horizontal_ops(ty: &SimdType) -> String {
    let mut code = String::new();
    let prefix = ty.width.x86_prefix();
    let suffix = ty.elem.x86_suffix();
    let name = ty.name();
    let elem = ty.elem.name();
    let lanes = ty.lanes();

    writeln!(code, "    // ========== Horizontal Operations ==========\n").unwrap();

    // Horizontal sum - different strategies per width
    if ty.elem.is_float() {
        writeln!(code, "    /// Sum all lanes horizontally.").unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Returns a scalar containing the sum of all {} lanes.",
            lanes
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn reduce_add(self) -> {} {{", elem).unwrap();

        match (ty.elem, ty.width) {
            (ElementType::F32, SimdWidth::W128) => {
                // f32x4: hadd twice, extract
                writeln!(code, "        unsafe {{").unwrap();
                writeln!(code, "            let h1 = _mm_hadd_ps(self.0, self.0);").unwrap();
                writeln!(code, "            let h2 = _mm_hadd_ps(h1, h1);").unwrap();
                writeln!(code, "            _mm_cvtss_f32(h2)").unwrap();
                writeln!(code, "        }}").unwrap();
            }
            (ElementType::F64, SimdWidth::W128) => {
                // f64x2: hadd, extract
                writeln!(code, "        unsafe {{").unwrap();
                writeln!(code, "            let h = _mm_hadd_pd(self.0, self.0);").unwrap();
                writeln!(code, "            _mm_cvtsd_f64(h)").unwrap();
                writeln!(code, "        }}").unwrap();
            }
            (ElementType::F32, SimdWidth::W256) => {
                // f32x8: extract halves, add, then 128-bit reduce
                writeln!(code, "        unsafe {{").unwrap();
                writeln!(
                    code,
                    "            let hi = _mm256_extractf128_ps::<1>(self.0);"
                )
                .unwrap();
                writeln!(code, "            let lo = _mm256_castps256_ps128(self.0);").unwrap();
                writeln!(code, "            let sum = _mm_add_ps(lo, hi);").unwrap();
                writeln!(code, "            let h1 = _mm_hadd_ps(sum, sum);").unwrap();
                writeln!(code, "            let h2 = _mm_hadd_ps(h1, h1);").unwrap();
                writeln!(code, "            _mm_cvtss_f32(h2)").unwrap();
                writeln!(code, "        }}").unwrap();
            }
            (ElementType::F64, SimdWidth::W256) => {
                // f64x4: extract halves, add, then 128-bit reduce
                writeln!(code, "        unsafe {{").unwrap();
                writeln!(
                    code,
                    "            let hi = _mm256_extractf128_pd::<1>(self.0);"
                )
                .unwrap();
                writeln!(code, "            let lo = _mm256_castpd256_pd128(self.0);").unwrap();
                writeln!(code, "            let sum = _mm_add_pd(lo, hi);").unwrap();
                writeln!(code, "            let h = _mm_hadd_pd(sum, sum);").unwrap();
                writeln!(code, "            _mm_cvtsd_f64(h)").unwrap();
                writeln!(code, "        }}").unwrap();
            }
            (ElementType::F32, SimdWidth::W512) => {
                // f32x16: AVX-512 has reduce intrinsics
                writeln!(code, "        unsafe {{ _mm512_reduce_add_ps(self.0) }}").unwrap();
            }
            (ElementType::F64, SimdWidth::W512) => {
                // f64x8: AVX-512 has reduce intrinsics
                writeln!(code, "        unsafe {{ _mm512_reduce_add_pd(self.0) }}").unwrap();
            }
            _ => unreachable!(),
        }
        writeln!(code, "    }}\n").unwrap();

        // Horizontal min/max for floats
        writeln!(code, "    /// Find the minimum value across all lanes.").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn reduce_min(self) -> {} {{", elem).unwrap();
        match (ty.elem, ty.width) {
            (ElementType::F32, SimdWidth::W128) => {
                writeln!(code, "        unsafe {{").unwrap();
                writeln!(
                    code,
                    "            let shuf = _mm_shuffle_ps::<0b10_11_00_01>(self.0, self.0);"
                )
                .unwrap();
                writeln!(code, "            let m1 = _mm_min_ps(self.0, shuf);").unwrap();
                writeln!(
                    code,
                    "            let shuf2 = _mm_shuffle_ps::<0b00_00_10_10>(m1, m1);"
                )
                .unwrap();
                writeln!(code, "            let m2 = _mm_min_ps(m1, shuf2);").unwrap();
                writeln!(code, "            _mm_cvtss_f32(m2)").unwrap();
                writeln!(code, "        }}").unwrap();
            }
            (ElementType::F64, SimdWidth::W128) => {
                writeln!(code, "        unsafe {{").unwrap();
                writeln!(
                    code,
                    "            let shuf = _mm_shuffle_pd::<0b01>(self.0, self.0);"
                )
                .unwrap();
                writeln!(code, "            let m = _mm_min_pd(self.0, shuf);").unwrap();
                writeln!(code, "            _mm_cvtsd_f64(m)").unwrap();
                writeln!(code, "        }}").unwrap();
            }
            (ElementType::F32, SimdWidth::W256) => {
                writeln!(code, "        unsafe {{").unwrap();
                writeln!(
                    code,
                    "            let hi = _mm256_extractf128_ps::<1>(self.0);"
                )
                .unwrap();
                writeln!(code, "            let lo = _mm256_castps256_ps128(self.0);").unwrap();
                writeln!(code, "            let m = _mm_min_ps(lo, hi);").unwrap();
                writeln!(
                    code,
                    "            let shuf = _mm_shuffle_ps::<0b10_11_00_01>(m, m);"
                )
                .unwrap();
                writeln!(code, "            let m1 = _mm_min_ps(m, shuf);").unwrap();
                writeln!(
                    code,
                    "            let shuf2 = _mm_shuffle_ps::<0b00_00_10_10>(m1, m1);"
                )
                .unwrap();
                writeln!(code, "            let m2 = _mm_min_ps(m1, shuf2);").unwrap();
                writeln!(code, "            _mm_cvtss_f32(m2)").unwrap();
                writeln!(code, "        }}").unwrap();
            }
            (ElementType::F64, SimdWidth::W256) => {
                writeln!(code, "        unsafe {{").unwrap();
                writeln!(
                    code,
                    "            let hi = _mm256_extractf128_pd::<1>(self.0);"
                )
                .unwrap();
                writeln!(code, "            let lo = _mm256_castpd256_pd128(self.0);").unwrap();
                writeln!(code, "            let m = _mm_min_pd(lo, hi);").unwrap();
                writeln!(code, "            let shuf = _mm_shuffle_pd::<0b01>(m, m);").unwrap();
                writeln!(code, "            let m2 = _mm_min_pd(m, shuf);").unwrap();
                writeln!(code, "            _mm_cvtsd_f64(m2)").unwrap();
                writeln!(code, "        }}").unwrap();
            }
            (ElementType::F32, SimdWidth::W512) => {
                writeln!(code, "        unsafe {{ _mm512_reduce_min_ps(self.0) }}").unwrap();
            }
            (ElementType::F64, SimdWidth::W512) => {
                writeln!(code, "        unsafe {{ _mm512_reduce_min_pd(self.0) }}").unwrap();
            }
            _ => unreachable!(),
        }
        writeln!(code, "    }}\n").unwrap();

        writeln!(code, "    /// Find the maximum value across all lanes.").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn reduce_max(self) -> {} {{", elem).unwrap();
        match (ty.elem, ty.width) {
            (ElementType::F32, SimdWidth::W128) => {
                writeln!(code, "        unsafe {{").unwrap();
                writeln!(
                    code,
                    "            let shuf = _mm_shuffle_ps::<0b10_11_00_01>(self.0, self.0);"
                )
                .unwrap();
                writeln!(code, "            let m1 = _mm_max_ps(self.0, shuf);").unwrap();
                writeln!(
                    code,
                    "            let shuf2 = _mm_shuffle_ps::<0b00_00_10_10>(m1, m1);"
                )
                .unwrap();
                writeln!(code, "            let m2 = _mm_max_ps(m1, shuf2);").unwrap();
                writeln!(code, "            _mm_cvtss_f32(m2)").unwrap();
                writeln!(code, "        }}").unwrap();
            }
            (ElementType::F64, SimdWidth::W128) => {
                writeln!(code, "        unsafe {{").unwrap();
                writeln!(
                    code,
                    "            let shuf = _mm_shuffle_pd::<0b01>(self.0, self.0);"
                )
                .unwrap();
                writeln!(code, "            let m = _mm_max_pd(self.0, shuf);").unwrap();
                writeln!(code, "            _mm_cvtsd_f64(m)").unwrap();
                writeln!(code, "        }}").unwrap();
            }
            (ElementType::F32, SimdWidth::W256) => {
                writeln!(code, "        unsafe {{").unwrap();
                writeln!(
                    code,
                    "            let hi = _mm256_extractf128_ps::<1>(self.0);"
                )
                .unwrap();
                writeln!(code, "            let lo = _mm256_castps256_ps128(self.0);").unwrap();
                writeln!(code, "            let m = _mm_max_ps(lo, hi);").unwrap();
                writeln!(
                    code,
                    "            let shuf = _mm_shuffle_ps::<0b10_11_00_01>(m, m);"
                )
                .unwrap();
                writeln!(code, "            let m1 = _mm_max_ps(m, shuf);").unwrap();
                writeln!(
                    code,
                    "            let shuf2 = _mm_shuffle_ps::<0b00_00_10_10>(m1, m1);"
                )
                .unwrap();
                writeln!(code, "            let m2 = _mm_max_ps(m1, shuf2);").unwrap();
                writeln!(code, "            _mm_cvtss_f32(m2)").unwrap();
                writeln!(code, "        }}").unwrap();
            }
            (ElementType::F64, SimdWidth::W256) => {
                writeln!(code, "        unsafe {{").unwrap();
                writeln!(
                    code,
                    "            let hi = _mm256_extractf128_pd::<1>(self.0);"
                )
                .unwrap();
                writeln!(code, "            let lo = _mm256_castpd256_pd128(self.0);").unwrap();
                writeln!(code, "            let m = _mm_max_pd(lo, hi);").unwrap();
                writeln!(code, "            let shuf = _mm_shuffle_pd::<0b01>(m, m);").unwrap();
                writeln!(code, "            let m2 = _mm_max_pd(m, shuf);").unwrap();
                writeln!(code, "            _mm_cvtsd_f64(m2)").unwrap();
                writeln!(code, "        }}").unwrap();
            }
            (ElementType::F32, SimdWidth::W512) => {
                writeln!(code, "        unsafe {{ _mm512_reduce_max_ps(self.0) }}").unwrap();
            }
            (ElementType::F64, SimdWidth::W512) => {
                writeln!(code, "        unsafe {{ _mm512_reduce_max_pd(self.0) }}").unwrap();
            }
            _ => unreachable!(),
        }
        writeln!(code, "    }}\n").unwrap();
    } else {
        // Integer horizontal sum - use to_array fallback for now (complex shuffle patterns)
        writeln!(code, "    /// Sum all lanes horizontally.").unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Returns a scalar containing the sum of all {} lanes.",
            lanes
        )
        .unwrap();
        writeln!(
            code,
            "    /// Note: This uses a scalar loop. For performance-critical code,"
        )
        .unwrap();
        writeln!(
            code,
            "    /// consider keeping values in SIMD until the final reduction."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn reduce_add(self) -> {} {{", elem).unwrap();
        writeln!(
            code,
            "        self.to_array().iter().copied().fold(0_{}, {}::wrapping_add)",
            elem, elem
        )
        .unwrap();
        writeln!(code, "    }}\n").unwrap();
    }

    code
}

fn generate_conversion_ops(ty: &SimdType) -> String {
    let mut code = String::new();
    let prefix = ty.width.x86_prefix();
    let name = ty.name();
    let lanes = ty.lanes();
    let token = ty.token();

    // Only generate conversions for matching lane counts
    // f32 <-> i32 conversions
    if ty.elem == ElementType::F32 {
        let int_name = format!("i32x{}", lanes);
        let uint_name = format!("u32x{}", lanes);

        writeln!(code, "    // ========== Type Conversions ==========\n").unwrap();

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
        writeln!(code, "    pub fn to_i32x{}(self) -> {} {{", lanes, int_name).unwrap();
        writeln!(
            code,
            "        {}(unsafe {{ {}_cvttps_epi32(self.0) }})",
            int_name, prefix
        )
        .unwrap();
        writeln!(code, "    }}\n").unwrap();

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
        writeln!(
            code,
            "    pub fn to_i32x{}_round(self) -> {} {{",
            lanes, int_name
        )
        .unwrap();
        writeln!(
            code,
            "        {}(unsafe {{ {}_cvtps_epi32(self.0) }})",
            int_name, prefix
        )
        .unwrap();
        writeln!(code, "    }}\n").unwrap();

        writeln!(code, "    /// Create from signed 32-bit integers.").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(
            code,
            "    pub fn from_i32x{}(v: {}) -> Self {{",
            lanes, int_name
        )
        .unwrap();
        writeln!(
            code,
            "        Self(unsafe {{ {}_cvtepi32_ps(v.0) }})",
            prefix
        )
        .unwrap();
        writeln!(code, "    }}\n").unwrap();
    } else if ty.elem == ElementType::I32 {
        let float_name = format!("f32x{}", lanes);

        writeln!(code, "    // ========== Type Conversions ==========\n").unwrap();

        writeln!(code, "    /// Convert to single-precision floats.").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(
            code,
            "    pub fn to_f32x{}(self) -> {} {{",
            lanes, float_name
        )
        .unwrap();
        writeln!(
            code,
            "        {}(unsafe {{ {}_cvtepi32_ps(self.0) }})",
            float_name, prefix
        )
        .unwrap();
        writeln!(code, "    }}\n").unwrap();
    }

    // f64 <-> i64/i32 conversions (more limited)
    if ty.elem == ElementType::F64 && ty.width == SimdWidth::W128 {
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
        writeln!(code, "        i32x4(unsafe {{ _mm_cvttpd_epi32(self.0) }})").unwrap();
        writeln!(code, "    }}\n").unwrap();
    }

    code
}

fn generate_scalar_ops(ty: &SimdType, cfg_attr: &str) -> String {
    let mut code = String::new();
    let name = ty.name();
    let elem = ty.elem.name();
    let token = ty.token();

    writeln!(code, "\n// Scalar broadcast operations for {}", name).unwrap();
    writeln!(
        code,
        "// These allow `v + 2.0` instead of `v + {}::splat(token, 2.0)`\n",
        name
    )
    .unwrap();

    if ty.elem.is_float() {
        // Add scalar
        writeln!(code, "{}impl Add<{}> for {} {{", cfg_attr, elem, name).unwrap();
        writeln!(code, "    type Output = Self;").unwrap();
        writeln!(code, "    /// Add a scalar to all lanes: `v + 2.0`").unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Broadcasts the scalar to all lanes, then adds."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    fn add(self, rhs: {}) -> Self {{", elem).unwrap();
        writeln!(
            code,
            "        self + Self(unsafe {{ {}({}) }})",
            scalar_splat_intrinsic(ty),
            "rhs"
        )
        .unwrap();
        writeln!(code, "    }}").unwrap();
        writeln!(code, "}}\n").unwrap();

        // Sub scalar
        writeln!(code, "{}impl Sub<{}> for {} {{", cfg_attr, elem, name).unwrap();
        writeln!(code, "    type Output = Self;").unwrap();
        writeln!(code, "    /// Subtract a scalar from all lanes: `v - 2.0`").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    fn sub(self, rhs: {}) -> Self {{", elem).unwrap();
        writeln!(
            code,
            "        self - Self(unsafe {{ {}({}) }})",
            scalar_splat_intrinsic(ty),
            "rhs"
        )
        .unwrap();
        writeln!(code, "    }}").unwrap();
        writeln!(code, "}}\n").unwrap();

        // Mul scalar
        writeln!(code, "{}impl Mul<{}> for {} {{", cfg_attr, elem, name).unwrap();
        writeln!(code, "    type Output = Self;").unwrap();
        writeln!(code, "    /// Multiply all lanes by a scalar: `v * 2.0`").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    fn mul(self, rhs: {}) -> Self {{", elem).unwrap();
        writeln!(
            code,
            "        self * Self(unsafe {{ {}({}) }})",
            scalar_splat_intrinsic(ty),
            "rhs"
        )
        .unwrap();
        writeln!(code, "    }}").unwrap();
        writeln!(code, "}}\n").unwrap();

        // Div scalar
        writeln!(code, "{}impl Div<{}> for {} {{", cfg_attr, elem, name).unwrap();
        writeln!(code, "    type Output = Self;").unwrap();
        writeln!(code, "    /// Divide all lanes by a scalar: `v / 2.0`").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    fn div(self, rhs: {}) -> Self {{", elem).unwrap();
        writeln!(
            code,
            "        self / Self(unsafe {{ {}({}) }})",
            scalar_splat_intrinsic(ty),
            "rhs"
        )
        .unwrap();
        writeln!(code, "    }}").unwrap();
        writeln!(code, "}}\n").unwrap();
    }

    // Integer scalar ops - just add/sub for now
    if !ty.elem.is_float() {
        let splat_call = scalar_splat_intrinsic_int(ty);

        writeln!(code, "{}impl Add<{}> for {} {{", cfg_attr, elem, name).unwrap();
        writeln!(code, "    type Output = Self;").unwrap();
        writeln!(code, "    /// Add a scalar to all lanes.").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    fn add(self, rhs: {}) -> Self {{", elem).unwrap();
        writeln!(code, "        self + Self(unsafe {{ {} }})", splat_call).unwrap();
        writeln!(code, "    }}").unwrap();
        writeln!(code, "}}\n").unwrap();

        writeln!(code, "{}impl Sub<{}> for {} {{", cfg_attr, elem, name).unwrap();
        writeln!(code, "    type Output = Self;").unwrap();
        writeln!(code, "    /// Subtract a scalar from all lanes.").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    fn sub(self, rhs: {}) -> Self {{", elem).unwrap();
        writeln!(code, "        self - Self(unsafe {{ {} }})", splat_call).unwrap();
        writeln!(code, "    }}").unwrap();
        writeln!(code, "}}\n").unwrap();
    }

    code
}

fn scalar_splat_intrinsic(ty: &SimdType) -> String {
    let prefix = ty.width.x86_prefix();
    match ty.elem {
        ElementType::F32 => format!("{}_set1_ps", prefix),
        ElementType::F64 => format!("{}_set1_pd", prefix),
        _ => unreachable!("scalar_splat_intrinsic only for floats"),
    }
}

fn scalar_splat_intrinsic_int(ty: &SimdType) -> String {
    let prefix = ty.width.x86_prefix();
    let (suffix, cast) = match (ty.elem, ty.width) {
        (ElementType::I8, _) => ("epi8", "rhs"),
        (ElementType::U8, _) => ("epi8", "rhs as i8"),
        (ElementType::I16, _) => ("epi16", "rhs"),
        (ElementType::U16, _) => ("epi16", "rhs as i16"),
        (ElementType::I32, _) => ("epi32", "rhs"),
        (ElementType::U32, _) => ("epi32", "rhs as i32"),
        (ElementType::I64, SimdWidth::W512) => ("epi64", "rhs"),
        (ElementType::I64, _) => ("epi64x", "rhs"),
        (ElementType::U64, SimdWidth::W512) => ("epi64", "rhs as i64"),
        (ElementType::U64, _) => ("epi64x", "rhs as i64"),
        _ => unreachable!(),
    };
    format!("{}_set1_{}({})", prefix, suffix, cast)
}

fn generate_math_ops(ty: &SimdType) -> String {
    let mut code = String::new();
    let prefix = ty.width.x86_prefix();
    let suffix = ty.elem.x86_suffix();
    let minmax_suffix = ty.elem.x86_minmax_suffix();

    // Min/Max - i64/u64 only available in AVX-512
    let has_minmax = ty.elem.is_float()
        || ty.width == SimdWidth::W512
        || !matches!(ty.elem, ElementType::I64 | ElementType::U64);

    if has_minmax {
        writeln!(code, "    /// Element-wise minimum").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn min(self, other: Self) -> Self {{").unwrap();
        writeln!(
            code,
            "        Self(unsafe {{ {}_min_{}(self.0, other.0) }})",
            prefix, minmax_suffix
        )
        .unwrap();
        writeln!(code, "    }}\n").unwrap();

        writeln!(code, "    /// Element-wise maximum").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn max(self, other: Self) -> Self {{").unwrap();
        writeln!(
            code,
            "        Self(unsafe {{ {}_max_{}(self.0, other.0) }})",
            prefix, minmax_suffix
        )
        .unwrap();
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

    // Float-specific operations
    if ty.elem.is_float() {
        writeln!(code, "    /// Square root").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn sqrt(self) -> Self {{").unwrap();
        writeln!(
            code,
            "        Self(unsafe {{ {}_sqrt_{}(self.0) }})",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "    }}\n").unwrap();

        // Abs for floats uses AND with sign mask
        let abs_mask = if ty.elem == ElementType::F32 {
            "0x7FFF_FFFFu32 as i32"
        } else {
            "0x7FFF_FFFF_FFFF_FFFFu64 as i64"
        };
        // 64-bit uses epi64x for 128/256-bit, epi64 for 512-bit
        let set1_int = match (ty.elem, ty.width) {
            (ElementType::F32, _) => "epi32",
            (_, SimdWidth::W512) => "epi64",
            _ => "epi64x",
        };
        let cast_to = if ty.elem == ElementType::F32 {
            "ps"
        } else {
            "pd"
        };

        writeln!(code, "    /// Absolute value").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn abs(self) -> Self {{").unwrap();
        writeln!(code, "        Self(unsafe {{").unwrap();
        writeln!(
            code,
            "            let mask = {}_castsi{}_{}({}_set1_{}({}));",
            prefix,
            ty.width.bits(),
            cast_to,
            prefix,
            set1_int,
            abs_mask
        )
        .unwrap();
        writeln!(code, "            {}_and_{}(self.0, mask)", prefix, suffix).unwrap();
        writeln!(code, "        }})").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // Floor/ceil/round for floats
        if ty.width == SimdWidth::W512 {
            // AVX-512 uses roundscale intrinsics with different constant encoding
            // _MM_FROUND_TO_NEG_INF = 0x01, _MM_FROUND_TO_POS_INF = 0x02, _MM_FROUND_TO_NEAREST_INT = 0x00
            writeln!(code, "    /// Round toward negative infinity").unwrap();
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(code, "    pub fn floor(self) -> Self {{").unwrap();
            writeln!(
                code,
                "        Self(unsafe {{ {}_roundscale_{}::<0x01>(self.0) }})",
                prefix, suffix
            )
            .unwrap();
            writeln!(code, "    }}\n").unwrap();

            writeln!(code, "    /// Round toward positive infinity").unwrap();
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(code, "    pub fn ceil(self) -> Self {{").unwrap();
            writeln!(
                code,
                "        Self(unsafe {{ {}_roundscale_{}::<0x02>(self.0) }})",
                prefix, suffix
            )
            .unwrap();
            writeln!(code, "    }}\n").unwrap();

            writeln!(code, "    /// Round to nearest integer").unwrap();
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(code, "    pub fn round(self) -> Self {{").unwrap();
            writeln!(
                code,
                "        Self(unsafe {{ {}_roundscale_{}::<0x00>(self.0) }})",
                prefix, suffix
            )
            .unwrap();
            writeln!(code, "    }}\n").unwrap();
        } else {
            // SSE4.1/AVX use floor/ceil/round intrinsics directly
            writeln!(code, "    /// Round toward negative infinity").unwrap();
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(code, "    pub fn floor(self) -> Self {{").unwrap();
            writeln!(
                code,
                "        Self(unsafe {{ {}_floor_{}(self.0) }})",
                prefix, suffix
            )
            .unwrap();
            writeln!(code, "    }}\n").unwrap();

            writeln!(code, "    /// Round toward positive infinity").unwrap();
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(code, "    pub fn ceil(self) -> Self {{").unwrap();
            writeln!(
                code,
                "        Self(unsafe {{ {}_ceil_{}(self.0) }})",
                prefix, suffix
            )
            .unwrap();
            writeln!(code, "    }}\n").unwrap();

            writeln!(code, "    /// Round to nearest integer").unwrap();
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(code, "    pub fn round(self) -> Self {{").unwrap();
            writeln!(code, "        Self(unsafe {{ {}_round_{}::<{{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }}>(self.0) }})", prefix, suffix).unwrap();
            writeln!(code, "    }}\n").unwrap();
        }

        // FMA (fused multiply-add)
        writeln!(code, "    /// Fused multiply-add: self * a + b").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(
            code,
            "    pub fn mul_add(self, a: Self, b: Self) -> Self {{"
        )
        .unwrap();
        writeln!(
            code,
            "        Self(unsafe {{ {}_fmadd_{}(self.0, a.0, b.0) }})",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "    }}\n").unwrap();

        writeln!(code, "    /// Fused multiply-sub: self * a - b").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(
            code,
            "    pub fn mul_sub(self, a: Self, b: Self) -> Self {{"
        )
        .unwrap();
        writeln!(
            code,
            "        Self(unsafe {{ {}_fmsub_{}(self.0, a.0, b.0) }})",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "    }}\n").unwrap();
    } else if ty.elem.is_signed() {
        // Abs for signed integers - i64 only available in AVX-512
        let has_abs = ty.width == SimdWidth::W512 || !matches!(ty.elem, ElementType::I64);
        if has_abs {
            writeln!(code, "    /// Absolute value").unwrap();
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(code, "    pub fn abs(self) -> Self {{").unwrap();
            writeln!(
                code,
                "        Self(unsafe {{ {}_abs_{}(self.0) }})",
                prefix, suffix
            )
            .unwrap();
            writeln!(code, "    }}\n").unwrap();
        }
    }

    code
}

fn generate_approx_ops(ty: &SimdType) -> String {
    let mut code = String::new();
    let prefix = ty.width.x86_prefix();
    let suffix = ty.elem.x86_suffix();

    // Only for f32 types (f64 doesn't have rcp/rsqrt in SSE/AVX, only AVX-512)
    if ty.elem == ElementType::F32 {
        writeln!(
            code,
            "    // ========== Approximation Operations ==========\n"
        )
        .unwrap();

        // rcp - reciprocal approximation
        if ty.width == SimdWidth::W512 {
            // AVX-512 has rcp14 with ~14-bit precision
            writeln!(
                code,
                "    /// Fast reciprocal approximation (1/x) with ~14-bit precision."
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
            writeln!(
                code,
                "        Self(unsafe {{ {}_rcp14_{}(self.0) }})",
                prefix, suffix
            )
            .unwrap();
            writeln!(code, "    }}\n").unwrap();
        } else {
            // SSE/AVX have rcp with ~12-bit precision
            writeln!(
                code,
                "    /// Fast reciprocal approximation (1/x) with ~12-bit precision."
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
            writeln!(
                code,
                "        Self(unsafe {{ {}_rcp_{}(self.0) }})",
                prefix, suffix
            )
            .unwrap();
            writeln!(code, "    }}\n").unwrap();
        }

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
            "        let two = Self(unsafe {{ {}_set1_ps(2.0) }});",
            prefix
        )
        .unwrap();
        writeln!(
            code,
            "        // One iteration gives ~24-bit precision from ~12-bit"
        )
        .unwrap();
        writeln!(code, "        approx * (two - self * approx)").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // rsqrt - reciprocal sqrt approximation
        if ty.width == SimdWidth::W512 {
            writeln!(code, "    /// Fast reciprocal square root approximation (1/sqrt(x)) with ~14-bit precision.").unwrap();
            writeln!(code, "    ///").unwrap();
            writeln!(
                code,
                "    /// For full precision, use `sqrt().recip()` or apply Newton-Raphson manually."
            )
            .unwrap();
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(code, "    pub fn rsqrt_approx(self) -> Self {{").unwrap();
            writeln!(
                code,
                "        Self(unsafe {{ {}_rsqrt14_{}(self.0) }})",
                prefix, suffix
            )
            .unwrap();
            writeln!(code, "    }}\n").unwrap();
        } else {
            writeln!(code, "    /// Fast reciprocal square root approximation (1/sqrt(x)) with ~12-bit precision.").unwrap();
            writeln!(code, "    ///").unwrap();
            writeln!(
                code,
                "    /// For full precision, use `sqrt().recip()` or apply Newton-Raphson manually."
            )
            .unwrap();
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(code, "    pub fn rsqrt_approx(self) -> Self {{").unwrap();
            writeln!(
                code,
                "        Self(unsafe {{ {}_rsqrt_{}(self.0) }})",
                prefix, suffix
            )
            .unwrap();
            writeln!(code, "    }}\n").unwrap();
        }

        // rsqrt with refinement
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
            "        let half = Self(unsafe {{ {}_set1_ps(0.5) }});",
            prefix
        )
        .unwrap();
        writeln!(
            code,
            "        let three = Self(unsafe {{ {}_set1_ps(3.0) }});",
            prefix
        )
        .unwrap();
        writeln!(
            code,
            "        half * approx * (three - self * approx * approx)"
        )
        .unwrap();
        writeln!(code, "    }}\n").unwrap();
    }

    // AVX-512 also has rcp14/rsqrt14 for f64
    if ty.elem == ElementType::F64 && ty.width == SimdWidth::W512 {
        writeln!(
            code,
            "    // ========== Approximation Operations ==========\n"
        )
        .unwrap();

        writeln!(
            code,
            "    /// Fast reciprocal approximation (1/x) with ~14-bit precision."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn rcp_approx(self) -> Self {{").unwrap();
        writeln!(
            code,
            "        Self(unsafe {{ {}_rcp14_{}(self.0) }})",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "    }}\n").unwrap();

        writeln!(
            code,
            "    /// Precise reciprocal (1/x) using Newton-Raphson refinement."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn recip(self) -> Self {{").unwrap();
        writeln!(code, "        let approx = self.rcp_approx();").unwrap();
        writeln!(
            code,
            "        let two = Self(unsafe {{ {}_set1_pd(2.0) }});",
            prefix
        )
        .unwrap();
        writeln!(code, "        approx * (two - self * approx)").unwrap();
        writeln!(code, "    }}\n").unwrap();

        writeln!(
            code,
            "    /// Fast reciprocal square root approximation (1/sqrt(x)) with ~14-bit precision."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn rsqrt_approx(self) -> Self {{").unwrap();
        writeln!(
            code,
            "        Self(unsafe {{ {}_rsqrt14_{}(self.0) }})",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "    }}\n").unwrap();

        writeln!(
            code,
            "    /// Precise reciprocal square root (1/sqrt(x)) using Newton-Raphson refinement."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn rsqrt(self) -> Self {{").unwrap();
        writeln!(code, "        let approx = self.rsqrt_approx();").unwrap();
        writeln!(
            code,
            "        let half = Self(unsafe {{ {}_set1_pd(0.5) }});",
            prefix
        )
        .unwrap();
        writeln!(
            code,
            "        let three = Self(unsafe {{ {}_set1_pd(3.0) }});",
            prefix
        )
        .unwrap();
        writeln!(
            code,
            "        half * approx * (three - self * approx * approx)"
        )
        .unwrap();
        writeln!(code, "    }}\n").unwrap();
    }

    code
}

fn generate_bitwise_unary_ops(ty: &SimdType) -> String {
    let mut code = String::new();
    let prefix = ty.width.x86_prefix();
    let bits = ty.width.bits();

    writeln!(
        code,
        "    // ========== Bitwise Unary Operations ==========\n"
    )
    .unwrap();

    // not - bitwise complement (XOR with all 1s)
    writeln!(code, "    /// Bitwise NOT (complement): flips all bits.").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn not(self) -> Self {{").unwrap();

    if ty.elem.is_float() {
        let suffix = ty.elem.x86_suffix();
        let int_suffix = if ty.elem == ElementType::F32 {
            "epi32"
        } else {
            "epi64"
        };
        let set1_suffix = match (ty.elem, ty.width) {
            (ElementType::F32, _) => "epi32",
            (_, SimdWidth::W512) => "epi64",
            _ => "epi64x",
        };
        let cast_to = if ty.elem == ElementType::F32 {
            "ps"
        } else {
            "pd"
        };
        let cast_from = if ty.elem == ElementType::F32 {
            "ps"
        } else {
            "pd"
        };

        writeln!(code, "        Self(unsafe {{").unwrap();
        writeln!(
            code,
            "            let ones = {}_set1_{}(-1);",
            prefix, set1_suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let as_int = {}_cast{}_si{}(self.0);",
            prefix, cast_from, bits
        )
        .unwrap();
        writeln!(
            code,
            "            {}_castsi{}_{} ({}_xor_si{}(as_int, ones))",
            prefix, bits, cast_to, prefix, bits
        )
        .unwrap();
        writeln!(code, "        }})").unwrap();
    } else {
        let set1_suffix = match (ty.elem, ty.width) {
            (ElementType::I8 | ElementType::U8, _) => "epi8",
            (ElementType::I16 | ElementType::U16, _) => "epi16",
            (ElementType::I32 | ElementType::U32, _) => "epi32",
            (ElementType::I64 | ElementType::U64, SimdWidth::W512) => "epi64",
            (ElementType::I64 | ElementType::U64, _) => "epi64x",
            _ => unreachable!(),
        };
        writeln!(code, "        Self(unsafe {{").unwrap();
        writeln!(
            code,
            "            let ones = {}_set1_{}(-1);",
            prefix, set1_suffix
        )
        .unwrap();
        writeln!(code, "            {}_xor_si{}(self.0, ones)", prefix, bits).unwrap();
        writeln!(code, "        }})").unwrap();
    }
    writeln!(code, "    }}\n").unwrap();

    code
}

fn generate_shift_ops(ty: &SimdType) -> String {
    let mut code = String::new();

    // Only for integer types, and NOT for 8-bit (no 8-bit shift intrinsics in x86 SIMD)
    if ty.elem.is_float() || matches!(ty.elem, ElementType::I8 | ElementType::U8) {
        return code;
    }

    let prefix = ty.width.x86_prefix();
    let suffix = ty.elem.x86_suffix();
    // AVX-512 intrinsics use u32 for const generic, SSE/AVX use i32
    let const_type = if ty.width == SimdWidth::W512 {
        "u32"
    } else {
        "i32"
    };

    writeln!(code, "    // ========== Shift Operations ==========\n").unwrap();

    // Shift left logical
    writeln!(code, "    /// Shift each lane left by `N` bits.").unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// Bits shifted out are lost; zeros are shifted in."
    )
    .unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(
        code,
        "    pub fn shl<const N: {}>(self) -> Self {{",
        const_type
    )
    .unwrap();
    writeln!(
        code,
        "        Self(unsafe {{ {}_slli_{}::<N>(self.0) }})",
        prefix, suffix
    )
    .unwrap();
    writeln!(code, "    }}\n").unwrap();

    // Shift right logical
    writeln!(
        code,
        "    /// Shift each lane right by `N` bits (logical/unsigned shift)."
    )
    .unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// Bits shifted out are lost; zeros are shifted in."
    )
    .unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(
        code,
        "    pub fn shr<const N: {}>(self) -> Self {{",
        const_type
    )
    .unwrap();
    writeln!(
        code,
        "        Self(unsafe {{ {}_srli_{}::<N>(self.0) }})",
        prefix, suffix
    )
    .unwrap();
    writeln!(code, "    }}\n").unwrap();

    // Arithmetic shift right (signed types; 64-bit only in AVX-512)
    if ty.elem.is_signed() {
        let has_sra = ty.elem != ElementType::I64 || ty.width == SimdWidth::W512;
        if has_sra {
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
            writeln!(code, "    #[inline(always)]").unwrap();
            writeln!(
                code,
                "    pub fn shr_arithmetic<const N: {}>(self) -> Self {{",
                const_type
            )
            .unwrap();
            writeln!(
                code,
                "        Self(unsafe {{ {}_srai_{}::<N>(self.0) }})",
                prefix, suffix
            )
            .unwrap();
            writeln!(code, "    }}\n").unwrap();
        }
    }

    code
}

fn generate_transcendental_ops(ty: &SimdType) -> String {
    let mut code = String::new();

    // Only for float types
    if !ty.elem.is_float() {
        return code;
    }

    let prefix = ty.width.x86_prefix();
    let suffix = ty.elem.x86_suffix();
    let bits = ty.width.bits();

    // Determine integer suffix for cast operations
    let int_suffix = if ty.elem == ElementType::F32 {
        "si256"
    } else {
        "si256"
    };
    let int_suffix_512 = if ty.elem == ElementType::F32 {
        "si512"
    } else {
        "si512"
    };
    let int_suffix_128 = if ty.elem == ElementType::F32 {
        "si128"
    } else {
        "si128"
    };

    let actual_int_suffix = match ty.width {
        SimdWidth::W128 => int_suffix_128,
        SimdWidth::W256 => int_suffix,
        SimdWidth::W512 => int_suffix_512,
    };

    writeln!(
        code,
        "    // ========== Transcendental Operations ==========\n"
    )
    .unwrap();

    if ty.elem == ElementType::F32 {
        // ===== F32 log2_lowp =====
        writeln!(
            code,
            "    /// Low-precision base-2 logarithm (~7.7e-5 max relative error)."
        )
        .unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(code, "    /// Uses rational polynomial approximation. Fast but not suitable for color-accurate work.").unwrap();
        writeln!(code, "    /// For higher precision, use `log2_midp()`.").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn log2_lowp(self) -> Self {{").unwrap();
        writeln!(
            code,
            "        // Rational polynomial coefficients from butteraugli/jpegli"
        )
        .unwrap();
        writeln!(code, "        const P0: f32 = -1.850_383_34e-6;").unwrap();
        writeln!(code, "        const P1: f32 = 1.428_716_05;").unwrap();
        writeln!(code, "        const P2: f32 = 0.742_458_73;").unwrap();
        writeln!(code, "        const Q0: f32 = 0.990_328_14;").unwrap();
        writeln!(code, "        const Q1: f32 = 1.009_671_86;").unwrap();
        writeln!(code, "        const Q2: f32 = 0.174_093_43;").unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(
            code,
            "            let x_bits = {}_cast{}_{}(self.0);",
            prefix, suffix, actual_int_suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let offset = {}_set1_epi32(0x3f2aaaab_u32 as i32);",
            prefix
        )
        .unwrap();
        writeln!(
            code,
            "            let exp_bits = {}_sub_epi32(x_bits, offset);",
            prefix
        )
        .unwrap();
        writeln!(
            code,
            "            let exp_shifted = {}_srai_epi32::<23>(exp_bits);",
            prefix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "            let mantissa_bits = {}_sub_epi32(x_bits, {}_slli_epi32::<23>(exp_shifted));", prefix, prefix).unwrap();
        writeln!(
            code,
            "            let mantissa = {}_cast{}_{}(mantissa_bits);",
            prefix, actual_int_suffix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let exp_val = {}_cvtepi32_{}(exp_shifted);",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "            let one = {}_set1_{}(1.0);",
            prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let m = {}_sub_{}(mantissa, one);",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "            // Horner's for numerator: P2*m^2 + P1*m + P0"
        )
        .unwrap();
        writeln!(
            code,
            "            let yp = {}_fmadd_{}({}_set1_{}(P2), m, {}_set1_{}(P1));",
            prefix, suffix, prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let yp = {}_fmadd_{}(yp, m, {}_set1_{}(P0));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "            // Horner's for denominator: Q2*m^2 + Q1*m + Q0"
        )
        .unwrap();
        writeln!(
            code,
            "            let yq = {}_fmadd_{}({}_set1_{}(Q2), m, {}_set1_{}(Q1));",
            prefix, suffix, prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let yq = {}_fmadd_{}(yq, m, {}_set1_{}(Q0));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "            Self({}_add_{}({}_div_{}(yp, yq), exp_val))",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F32 exp2_lowp =====
        writeln!(
            code,
            "    /// Low-precision base-2 exponential (~5.5e-3 max relative error)."
        )
        .unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(code, "    /// Uses degree-3 polynomial approximation. Fast but not suitable for color-accurate work.").unwrap();
        writeln!(code, "    /// For higher precision, use `exp2_midp()`.").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn exp2_lowp(self) -> Self {{").unwrap();
        writeln!(code, "        // Polynomial coefficients").unwrap();
        writeln!(code, "        const C0: f32 = 1.0;").unwrap();
        writeln!(code, "        const C1: f32 = core::f32::consts::LN_2;").unwrap();
        writeln!(code, "        const C2: f32 = 0.240_226_5;").unwrap();
        writeln!(code, "        const C3: f32 = 0.055_504_11;").unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(code, "            // Clamp to safe range").unwrap();
        writeln!(
            code,
            "            let x = {}_max_{}(self.0, {}_set1_{}(-126.0));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let x = {}_min_{}(x, {}_set1_{}(126.0));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "            // Split into integer and fractional parts"
        )
        .unwrap();

        // Use appropriate floor/round intrinsic based on width
        if ty.width == SimdWidth::W512 {
            writeln!(
                code,
                "            let xi = {}_roundscale_{}::<0x01>(x); // floor",
                prefix, suffix
            )
            .unwrap();
        } else {
            writeln!(code, "            let xi = {}_floor_{}(x);", prefix, suffix).unwrap();
        }

        writeln!(
            code,
            "            let xf = {}_sub_{}(x, xi);",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "            // Polynomial for 2^frac").unwrap();
        writeln!(
            code,
            "            let poly = {}_fmadd_{}({}_set1_{}(C3), xf, {}_set1_{}(C2));",
            prefix, suffix, prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let poly = {}_fmadd_{}(poly, xf, {}_set1_{}(C1));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let poly = {}_fmadd_{}(poly, xf, {}_set1_{}(C0));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "            // Scale by 2^integer using bit manipulation"
        )
        .unwrap();
        writeln!(
            code,
            "            let xi_i32 = {}_cvt{}_epi32(xi);",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "            let bias = {}_set1_epi32(127);", prefix).unwrap();
        writeln!(
            code,
            "            let scale_bits = {}_slli_epi32::<23>({}_add_epi32(xi_i32, bias));",
            prefix, prefix
        )
        .unwrap();
        writeln!(
            code,
            "            let scale = {}_cast{}_{}(scale_bits);",
            prefix, actual_int_suffix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "            Self({}_mul_{}(poly, scale))",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F32 ln_lowp (natural log) =====
        writeln!(code, "    /// Low-precision natural logarithm.").unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Computed as `log2_lowp(x) * ln(2)`. For higher precision, use `ln_midp()`."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn ln_lowp(self) -> Self {{").unwrap();
        writeln!(code, "        const LN2: f32 = core::f32::consts::LN_2;").unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(
            code,
            "            Self({}_mul_{}(self.log2_lowp().0, {}_set1_{}(LN2)))",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F32 exp_lowp (natural exp) =====
        writeln!(code, "    /// Low-precision natural exponential (e^x).").unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Computed as `exp2_lowp(x * log2(e))`. For higher precision, use `exp_midp()`."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn exp_lowp(self) -> Self {{").unwrap();
        writeln!(
            code,
            "        const LOG2_E: f32 = core::f32::consts::LOG2_E;"
        )
        .unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(
            code,
            "            Self({}_mul_{}(self.0, {}_set1_{}(LOG2_E))).exp2_lowp()",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F32 log10_lowp =====
        writeln!(code, "    /// Low-precision base-10 logarithm.").unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(code, "    /// Computed as `log2_lowp(x) / log2(10)`.").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn log10_lowp(self) -> Self {{").unwrap();
        writeln!(
            code,
            "        const LOG10_2: f32 = core::f32::consts::LOG10_2; // 1/log2(10)"
        )
        .unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(
            code,
            "            Self({}_mul_{}(self.log2_lowp().0, {}_set1_{}(LOG10_2)))",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F32 pow_lowp =====
        writeln!(code, "    /// Low-precision power function (self^n).").unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(code, "    /// Computed as `exp2_lowp(n * log2_lowp(self))`. For higher precision, use `pow_midp()`.").unwrap();
        writeln!(code, "    /// Note: Only valid for positive self values.").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn pow_lowp(self, n: f32) -> Self {{").unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(
            code,
            "            Self({}_mul_{}(self.log2_lowp().0, {}_set1_{}(n))).exp2_lowp()",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ========== Mid-Precision Transcendental Operations ==========
        writeln!(
            code,
            "    // ========== Mid-Precision Transcendental Operations ==========\n"
        )
        .unwrap();

        // ===== F32 log2_midp =====
        writeln!(
            code,
            "    /// Mid-precision base-2 logarithm (~3 ULP max error)."
        )
        .unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Uses (a-1)/(a+1) transform with degree-6 odd polynomial."
        )
        .unwrap();
        writeln!(
            code,
            "    /// Suitable for 8-bit, 10-bit, and 12-bit color processing."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn log2_midp(self) -> Self {{").unwrap();
        writeln!(code, "        // Constants for range reduction").unwrap();
        writeln!(
            code,
            "        const SQRT2_OVER_2: u32 = 0x3f3504f3; // sqrt(2)/2 in f32 bits"
        )
        .unwrap();
        writeln!(
            code,
            "        const ONE: u32 = 0x3f800000;          // 1.0 in f32 bits"
        )
        .unwrap();
        writeln!(
            code,
            "        const MANTISSA_MASK: i32 = 0x007fffff_u32 as i32;"
        )
        .unwrap();
        writeln!(code, "        const EXPONENT_BIAS: i32 = 127;").unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "        // Coefficients for odd polynomial on y = (a-1)/(a+1)"
        )
        .unwrap();
        writeln!(code, "        const C0: f32 = 2.885_390_08;  // 2/ln(2)").unwrap();
        writeln!(
            code,
            "        const C1: f32 = 0.961_800_76;  // y^2 coefficient"
        )
        .unwrap();
        writeln!(
            code,
            "        const C2: f32 = 0.576_974_45;  // y^4 coefficient"
        )
        .unwrap();
        writeln!(
            code,
            "        const C3: f32 = 0.434_411_97;  // y^6 coefficient"
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(
            code,
            "            let x_bits = {}_cast{}_{}(self.0);",
            prefix, suffix, actual_int_suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "            // Normalize mantissa to [sqrt(2)/2, sqrt(2)]"
        )
        .unwrap();
        writeln!(
            code,
            "            let offset = {}_set1_epi32((ONE - SQRT2_OVER_2) as i32);",
            prefix
        )
        .unwrap();
        writeln!(
            code,
            "            let adjusted = {}_add_epi32(x_bits, offset);",
            prefix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "            // Extract exponent").unwrap();
        writeln!(
            code,
            "            let exp_raw = {}_srai_epi32::<23>(adjusted);",
            prefix
        )
        .unwrap();
        writeln!(
            code,
            "            let exp_biased = {}_sub_epi32(exp_raw, {}_set1_epi32(EXPONENT_BIAS));",
            prefix, prefix
        )
        .unwrap();
        writeln!(
            code,
            "            let n = {}_cvtepi32_{}(exp_biased);",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "            // Reconstruct normalized mantissa").unwrap();
        writeln!(
            code,
            "            let mantissa_bits = {}_and_{}(adjusted, {}_set1_epi32(MANTISSA_MASK));",
            prefix, actual_int_suffix, prefix
        )
        .unwrap();
        writeln!(code, "            let a_bits = {}_add_epi32(mantissa_bits, {}_set1_epi32(SQRT2_OVER_2 as i32));", prefix, prefix).unwrap();
        writeln!(
            code,
            "            let a = {}_cast{}_{}(a_bits);",
            prefix, actual_int_suffix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "            // y = (a - 1) / (a + 1)").unwrap();
        writeln!(
            code,
            "            let one = {}_set1_{}(1.0);",
            prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let a_minus_1 = {}_sub_{}(a, one);",
            prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let a_plus_1 = {}_add_{}(a, one);",
            prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let y = {}_div_{}(a_minus_1, a_plus_1);",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "            // y^2").unwrap();
        writeln!(
            code,
            "            let y2 = {}_mul_{}(y, y);",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "            // Polynomial: c0 + y^2*(c1 + y^2*(c2 + y^2*c3))"
        )
        .unwrap();
        writeln!(
            code,
            "            let poly = {}_fmadd_{}({}_set1_{}(C3), y2, {}_set1_{}(C2));",
            prefix, suffix, prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let poly = {}_fmadd_{}(poly, y2, {}_set1_{}(C1));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let poly = {}_fmadd_{}(poly, y2, {}_set1_{}(C0));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "            // Result: y * poly + n").unwrap();
        writeln!(
            code,
            "            Self({}_fmadd_{}(y, poly, n))",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F32 exp2_midp =====
        writeln!(
            code,
            "    /// Mid-precision base-2 exponential (~140 ULP, ~8e-6 max relative error)."
        )
        .unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(code, "    /// Uses degree-6 minimax polynomial.").unwrap();
        writeln!(
            code,
            "    /// Suitable for 8-bit, 10-bit, and 12-bit color processing."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn exp2_midp(self) -> Self {{").unwrap();
        writeln!(
            code,
            "        // Degree-6 minimax polynomial for 2^x on [0, 1]"
        )
        .unwrap();
        writeln!(code, "        const C0: f32 = 1.0;").unwrap();
        writeln!(code, "        const C1: f32 = 0.693_147_180_559_945;").unwrap();
        writeln!(code, "        const C2: f32 = 0.240_226_506_959_101;").unwrap();
        writeln!(code, "        const C3: f32 = 0.055_504_108_664_822;").unwrap();
        writeln!(code, "        const C4: f32 = 0.009_618_129_107_629;").unwrap();
        writeln!(code, "        const C5: f32 = 0.001_333_355_814_497;").unwrap();
        writeln!(code, "        const C6: f32 = 0.000_154_035_303_933;").unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(code, "            // Clamp to safe range").unwrap();
        writeln!(
            code,
            "            let x = {}_max_{}(self.0, {}_set1_{}(-126.0));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let x = {}_min_{}(x, {}_set1_{}(126.0));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();

        // Use appropriate floor intrinsic based on width
        if ty.width == SimdWidth::W512 {
            writeln!(
                code,
                "            let xi = {}_roundscale_{}::<0x01>(x); // floor",
                prefix, suffix
            )
            .unwrap();
        } else {
            writeln!(code, "            let xi = {}_floor_{}(x);", prefix, suffix).unwrap();
        }

        writeln!(
            code,
            "            let xf = {}_sub_{}(x, xi);",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "            // Horner's method with 6 coefficients").unwrap();
        writeln!(
            code,
            "            let poly = {}_fmadd_{}({}_set1_{}(C6), xf, {}_set1_{}(C5));",
            prefix, suffix, prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let poly = {}_fmadd_{}(poly, xf, {}_set1_{}(C4));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let poly = {}_fmadd_{}(poly, xf, {}_set1_{}(C3));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let poly = {}_fmadd_{}(poly, xf, {}_set1_{}(C2));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let poly = {}_fmadd_{}(poly, xf, {}_set1_{}(C1));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let poly = {}_fmadd_{}(poly, xf, {}_set1_{}(C0));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "            // Scale by 2^integer").unwrap();
        writeln!(
            code,
            "            let xi_i32 = {}_cvt{}_epi32(xi);",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "            let bias = {}_set1_epi32(127);", prefix).unwrap();
        writeln!(
            code,
            "            let scale_bits = {}_slli_epi32::<23>({}_add_epi32(xi_i32, bias));",
            prefix, prefix
        )
        .unwrap();
        writeln!(
            code,
            "            let scale = {}_cast{}_{}(scale_bits);",
            prefix, actual_int_suffix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "            Self({}_mul_{}(poly, scale))",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F32 pow_midp =====
        writeln!(code, "    /// Mid-precision power function (self^n).").unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Computed as `exp2_midp(n * log2_midp(self))`."
        )
        .unwrap();
        writeln!(
            code,
            "    /// Achieves 100% exact round-trips for 8-bit, 10-bit, and 12-bit values."
        )
        .unwrap();
        writeln!(code, "    /// Note: Only valid for positive self values.").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn pow_midp(self, n: f32) -> Self {{").unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(
            code,
            "            Self({}_mul_{}(self.log2_midp().0, {}_set1_{}(n))).exp2_midp()",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F32 ln_midp =====
        writeln!(code, "    /// Mid-precision natural logarithm.").unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(code, "    /// Computed as `log2_midp(x) * ln(2)`.").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn ln_midp(self) -> Self {{").unwrap();
        writeln!(code, "        const LN2: f32 = core::f32::consts::LN_2;").unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(
            code,
            "            Self({}_mul_{}(self.log2_midp().0, {}_set1_{}(LN2)))",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F32 exp_midp =====
        writeln!(code, "    /// Mid-precision natural exponential (e^x).").unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(code, "    /// Computed as `exp2_midp(x * log2(e))`.").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn exp_midp(self) -> Self {{").unwrap();
        writeln!(
            code,
            "        const LOG2_E: f32 = core::f32::consts::LOG2_E;"
        )
        .unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(
            code,
            "            Self({}_mul_{}(self.0, {}_set1_{}(LOG2_E))).exp2_midp()",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ========== Cube Root ==========
        writeln!(code, "    // ========== Cube Root ==========\n").unwrap();

        // ===== F32 cbrt_lowp =====
        writeln!(code, "    /// Low-precision cube root (x^(1/3)).").unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Computed via `pow_lowp(x, 1/3)`. For negative inputs, returns NaN."
        )
        .unwrap();
        writeln!(code, "    /// For higher precision, use `cbrt_midp()`.").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn cbrt_lowp(self) -> Self {{").unwrap();
        writeln!(code, "        self.pow_lowp(1.0 / 3.0)").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F32 cbrt_midp =====
        writeln!(code, "    /// Mid-precision cube root (x^(1/3)).").unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Uses pow_midp with scalar extraction for initial guess + Newton-Raphson."
        )
        .unwrap();
        writeln!(
            code,
            "    /// Handles negative values correctly (returns -cbrt(|x|))."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn cbrt_midp(self) -> Self {{").unwrap();
        writeln!(
            code,
            "        // B1 magic constant for cube root initial approximation"
        )
        .unwrap();
        writeln!(
            code,
            "        // B1 = (127 - 127.0/3 - 0.03306235651) * 2^23 = 709958130"
        )
        .unwrap();
        writeln!(code, "        const B1: u32 = 709_958_130;").unwrap();
        writeln!(code, "        const ONE_THIRD: f32 = 1.0 / 3.0;").unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(
            code,
            "            // Extract to array for initial approximation (scalar division by 3)"
        )
        .unwrap();
        writeln!(
            code,
            "            let x_arr: [f32; {}] = core::mem::transmute(self.0);",
            ty.lanes()
        )
        .unwrap();
        writeln!(
            code,
            "            let mut y_arr = [0.0f32; {}];",
            ty.lanes()
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "            for i in 0..{} {{", ty.lanes()).unwrap();
        writeln!(code, "                let xi = x_arr[i];").unwrap();
        writeln!(code, "                let ui = xi.to_bits();").unwrap();
        writeln!(
            code,
            "                let hx = ui & 0x7FFF_FFFF; // abs bits"
        )
        .unwrap();
        writeln!(
            code,
            "                // Initial approximation: bits/3 + B1 (always positive)"
        )
        .unwrap();
        writeln!(code, "                let approx = hx / 3 + B1;").unwrap();
        writeln!(code, "                y_arr[i] = f32::from_bits(approx);").unwrap();
        writeln!(code, "            }}").unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "            let abs_x = {}_andnot_{}({}_set1_{}(-0.0), self.0);",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let sign_bits = {}_and_{}(self.0, {}_set1_{}(-0.0));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let mut y = core::mem::transmute::<_, _>(y_arr);"
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "            // Newton-Raphson: y = y * (2*x + y^3) / (x + 2*y^3)"
        )
        .unwrap();
        writeln!(code, "            // Two iterations for full f32 precision").unwrap();
        writeln!(
            code,
            "            let two = {}_set1_{}(2.0);",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "            // Iteration 1").unwrap();
        writeln!(
            code,
            "            let y3 = {}_mul_{}({}_mul_{}(y, y), y);",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let num = {}_fmadd_{}(two, abs_x, y3);",
            prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let den = {}_fmadd_{}(two, y3, abs_x);",
            prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            y = {}_mul_{}(y, {}_div_{}(num, den));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "            // Iteration 2").unwrap();
        writeln!(
            code,
            "            let y3 = {}_mul_{}({}_mul_{}(y, y), y);",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let num = {}_fmadd_{}(two, abs_x, y3);",
            prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let den = {}_fmadd_{}(two, y3, abs_x);",
            prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            y = {}_mul_{}(y, {}_div_{}(num, den));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "            // Restore sign").unwrap();
        writeln!(
            code,
            "            Self({}_or_{}(y, sign_bits))",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();
    } else if ty.elem == ElementType::F64 {
        // ===== F64 log2_lowp =====
        // For f64, we use a similar algorithm but with f64 constants
        writeln!(code, "    /// Low-precision base-2 logarithm.").unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Uses polynomial approximation. For natural log, use `ln_lowp()`."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn log2_lowp(self) -> Self {{").unwrap();
        writeln!(code, "        // Polynomial coefficients for f64").unwrap();
        writeln!(code, "        const P0: f64 = -1.850_383_340_051_831e-6;").unwrap();
        writeln!(code, "        const P1: f64 = 1.428_716_047_008_376;").unwrap();
        writeln!(code, "        const P2: f64 = 0.742_458_733_278_206;").unwrap();
        writeln!(code, "        const Q0: f64 = 0.990_328_142_775_907;").unwrap();
        writeln!(code, "        const Q1: f64 = 1.009_671_857_224_115;").unwrap();
        writeln!(code, "        const Q2: f64 = 0.174_093_430_036_669;").unwrap();
        writeln!(
            code,
            "        const OFFSET: i64 = 0x3fe6a09e667f3bcd_u64 as i64; // 2/3 in f64 bits"
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(
            code,
            "            let x_bits = {}_cast{}_{}(self.0);",
            prefix, suffix, actual_int_suffix
        )
        .unwrap();

        // For 64-bit integers, we need different intrinsics
        // For set1 with i64, SSE/AVX use epi64x, AVX-512 uses epi64
        let epi64_suffix = if ty.width == SimdWidth::W512 {
            "epi64"
        } else {
            "epi64x"
        };

        writeln!(
            code,
            "            let offset = {}_set1_{}(OFFSET);",
            prefix, epi64_suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let exp_bits = {}_sub_epi64(x_bits, offset);",
            prefix
        )
        .unwrap();
        writeln!(
            code,
            "            let exp_shifted = {}_srai_epi64::<52>(exp_bits);",
            prefix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "            let mantissa_bits = {}_sub_epi64(x_bits, {}_slli_epi64::<52>(exp_shifted));", prefix, prefix).unwrap();
        writeln!(
            code,
            "            let mantissa = {}_cast{}_{}(mantissa_bits);",
            prefix, actual_int_suffix, suffix
        )
        .unwrap();

        // Convert i64 to f64 - extract and convert via scalar
        writeln!(code, "            // Convert exponent to f64").unwrap();
        writeln!(
            code,
            "            let exp_arr: [i64; {}] = core::mem::transmute(exp_shifted);",
            ty.lanes()
        )
        .unwrap();
        writeln!(code, "            let exp_f64: [f64; {}] = [", ty.lanes()).unwrap();
        for i in 0..ty.lanes() {
            if i > 0 {
                write!(code, ", ").unwrap();
            }
            write!(code, "exp_arr[{}] as f64", i).unwrap();
        }
        writeln!(code, "];").unwrap();
        writeln!(
            code,
            "            let exp_val = {}_loadu_{}(exp_f64.as_ptr());",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "            let one = {}_set1_{}(1.0);",
            prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let m = {}_sub_{}(mantissa, one);",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "            // Horner's for numerator").unwrap();
        writeln!(
            code,
            "            let yp = {}_fmadd_{}({}_set1_{}(P2), m, {}_set1_{}(P1));",
            prefix, suffix, prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let yp = {}_fmadd_{}(yp, m, {}_set1_{}(P0));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "            // Horner's for denominator").unwrap();
        writeln!(
            code,
            "            let yq = {}_fmadd_{}({}_set1_{}(Q2), m, {}_set1_{}(Q1));",
            prefix, suffix, prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let yq = {}_fmadd_{}(yq, m, {}_set1_{}(Q0));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "            Self({}_add_{}({}_div_{}(yp, yq), exp_val))",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F64 exp2_lowp =====
        writeln!(code, "    /// Low-precision base-2 exponential (2^x).").unwrap();
        writeln!(code, "    ///").unwrap();
        writeln!(
            code,
            "    /// Uses polynomial approximation. For natural exp, use `exp_lowp()`."
        )
        .unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn exp2_lowp(self) -> Self {{").unwrap();
        writeln!(code, "        const C0: f64 = 1.0;").unwrap();
        writeln!(code, "        const C1: f64 = core::f64::consts::LN_2;").unwrap();
        writeln!(code, "        const C2: f64 = 0.240_226_506_959_101;").unwrap();
        writeln!(code, "        const C3: f64 = 0.055_504_108_664_822;").unwrap();
        writeln!(code, "        const C4: f64 = 0.009_618_129_107_629;").unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(code, "            // Clamp to safe range").unwrap();
        writeln!(
            code,
            "            let x = {}_max_{}(self.0, {}_set1_{}(-1022.0));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let x = {}_min_{}(x, {}_set1_{}(1022.0));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();

        if ty.width == SimdWidth::W512 {
            writeln!(
                code,
                "            let xi = {}_roundscale_{}::<0x01>(x); // floor",
                prefix, suffix
            )
            .unwrap();
        } else {
            writeln!(code, "            let xi = {}_floor_{}(x);", prefix, suffix).unwrap();
        }

        writeln!(
            code,
            "            let xf = {}_sub_{}(x, xi);",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(code, "            // Polynomial for 2^frac").unwrap();
        writeln!(
            code,
            "            let poly = {}_fmadd_{}({}_set1_{}(C4), xf, {}_set1_{}(C3));",
            prefix, suffix, prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let poly = {}_fmadd_{}(poly, xf, {}_set1_{}(C2));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let poly = {}_fmadd_{}(poly, xf, {}_set1_{}(C1));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(
            code,
            "            let poly = {}_fmadd_{}(poly, xf, {}_set1_{}(C0));",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "            // Scale by 2^integer - extract, convert, scale"
        )
        .unwrap();
        writeln!(
            code,
            "            let xi_arr: [f64; {}] = core::mem::transmute(xi);",
            ty.lanes()
        )
        .unwrap();
        writeln!(code, "            let scale_arr: [f64; {}] = [", ty.lanes()).unwrap();
        for i in 0..ty.lanes() {
            if i > 0 {
                write!(code, ", ").unwrap();
            }
            write!(
                code,
                "f64::from_bits(((xi_arr[{}] as i64 + 1023) << 52) as u64)",
                i
            )
            .unwrap();
        }
        writeln!(code, "];").unwrap();
        writeln!(
            code,
            "            let scale = {}_loadu_{}(scale_arr.as_ptr());",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "").unwrap();
        writeln!(
            code,
            "            Self({}_mul_{}(poly, scale))",
            prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F64 ln_lowp =====
        writeln!(code, "    /// Low-precision natural logarithm.").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn ln_lowp(self) -> Self {{").unwrap();
        writeln!(code, "        const LN2: f64 = core::f64::consts::LN_2;").unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(
            code,
            "            Self({}_mul_{}(self.log2_lowp().0, {}_set1_{}(LN2)))",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F64 exp_lowp =====
        writeln!(code, "    /// Low-precision natural exponential (e^x).").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn exp_lowp(self) -> Self {{").unwrap();
        writeln!(
            code,
            "        const LOG2_E: f64 = core::f64::consts::LOG2_E;"
        )
        .unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(
            code,
            "            Self({}_mul_{}(self.0, {}_set1_{}(LOG2_E))).exp2_lowp()",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F64 log10_lowp =====
        writeln!(code, "    /// Low-precision base-10 logarithm.").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn log10_lowp(self) -> Self {{").unwrap();
        writeln!(
            code,
            "        const LOG10_2: f64 = core::f64::consts::LOG10_2;"
        )
        .unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(
            code,
            "            Self({}_mul_{}(self.log2_lowp().0, {}_set1_{}(LOG10_2)))",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();

        // ===== F64 pow_lowp =====
        writeln!(code, "    /// Low-precision power function (self^n).").unwrap();
        writeln!(code, "    #[inline(always)]").unwrap();
        writeln!(code, "    pub fn pow_lowp(self, n: f64) -> Self {{").unwrap();
        writeln!(code, "        unsafe {{").unwrap();
        writeln!(
            code,
            "            Self({}_mul_{}(self.log2_lowp().0, {}_set1_{}(n))).exp2_lowp()",
            prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "        }}").unwrap();
        writeln!(code, "    }}\n").unwrap();
    }

    code
}

fn generate_operator_impls(ty: &SimdType, cfg_attr: &str) -> String {
    let mut code = String::new();
    let name = ty.name();
    let prefix = ty.width.x86_prefix();
    let suffix = ty.elem.x86_suffix();

    if ty.elem.is_float() {
        // Float arithmetic
        writeln!(
            code,
            "{}impl_arithmetic_ops!({}, {}_add_{}, {}_sub_{}, {}_mul_{}, {}_div_{});",
            cfg_attr, name, prefix, suffix, prefix, suffix, prefix, suffix, prefix, suffix
        )
        .unwrap();
        writeln!(code, "{}impl_float_assign_ops!({});", cfg_attr, name).unwrap();

        // Neg for floats
        writeln!(
            code,
            "{}impl_neg!({}, {}_sub_{}, {}_setzero_{});",
            cfg_attr, name, prefix, suffix, prefix, suffix
        )
        .unwrap();
    } else {
        // Integer arithmetic (add/sub only, mul depends on type)
        writeln!(
            code,
            "{}impl_int_arithmetic_ops!({}, {}_add_{}, {}_sub_{});",
            cfg_attr, name, prefix, suffix, prefix, suffix
        )
        .unwrap();

        // Integer mul available for some sizes
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
                "{}impl_int_mul_op!({}, {}_mullo_{});",
                cfg_attr, name, prefix, mul_suffix
            )
            .unwrap();
        }

        writeln!(code, "{}impl_assign_ops!({});", cfg_attr, name).unwrap();
    }

    // Bitwise (use si version for floats, direct for ints)
    if ty.elem.is_float() {
        let and_fn = format!("{}_and_{}", prefix, suffix);
        let or_fn = format!("{}_or_{}", prefix, suffix);
        let xor_fn = format!("{}_xor_{}", prefix, suffix);
        writeln!(
            code,
            "{}impl_bitwise_ops!({}, {}, {}, {}, {});",
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
            "{}impl_bitwise_ops!({}, {}, {}_and_si{}, {}_or_si{}, {}_xor_si{});",
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
        "{}impl_index!({}, {}, {});",
        cfg_attr,
        name,
        ty.elem.name(),
        ty.lanes()
    )
    .unwrap();

    code.push_str("\n");
    code
}

/// Generate tests for SIMD types
pub fn generate_simd_tests() -> String {
    let mut code = String::from(
        r#"//! Auto-generated tests for SIMD types.
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.

#![cfg(target_arch = "x86_64")]
#![allow(unused)]

use archmage::SimdToken;
use archmage::simd::*;

#[test]
fn test_f32x8_basic() {
    let Some(token) = archmage::Avx2FmaToken::try_new() else {
        eprintln!("AVX2+FMA not available");
        return;
    };

    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let v = f32x8::load(token, &data);

    // Test round-trip
    let out = v.to_array();
    assert_eq!(data, out);

    // Test arithmetic
    let two = f32x8::splat(token, 2.0);
    let sum = v + two;
    let expected = [3.0f32, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    assert_eq!(sum.to_array(), expected);

    // Test min/max
    let a = f32x8::splat(token, 5.0);
    assert_eq!(v.min(a).to_array(), [1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0]);
    assert_eq!(v.max(a).to_array(), [5.0, 5.0, 5.0, 5.0, 5.0, 6.0, 7.0, 8.0]);

    // Test FMA
    let b = f32x8::splat(token, 1.0);
    let fma = v.mul_add(two, b); // v * 2 + 1
    assert_eq!(fma.to_array(), [3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0]);
}

#[test]
fn test_f32x8_comparisons() {
    let Some(token) = archmage::Avx2FmaToken::try_new() else {
        eprintln!("AVX2+FMA not available");
        return;
    };

    let a = f32x8::load(token, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = f32x8::load(token, &[2.0, 2.0, 2.0, 2.0, 6.0, 6.0, 6.0, 6.0]);

    // Test simd_lt: lanes where a < b should be all-1s (as f32: NaN)
    let lt = a.simd_lt(b);
    let lt_arr = lt.to_array();
    assert!(lt_arr[0].is_nan()); // 1 < 2 = true (all-1s = NaN)
    assert_eq!(lt_arr[1].to_bits(), 0); // 2 < 2 = false
    assert_eq!(lt_arr[2].to_bits(), 0); // 3 < 2 = false

    // Test simd_eq
    let eq = a.simd_eq(b);
    let eq_arr = eq.to_array();
    assert_eq!(eq_arr[0].to_bits(), 0); // 1 == 2 = false
    assert!(eq_arr[1].is_nan()); // 2 == 2 = true

    // Test simd_gt
    let gt = a.simd_gt(b);
    let gt_arr = gt.to_array();
    assert_eq!(gt_arr[0].to_bits(), 0); // 1 > 2 = false
    assert_eq!(gt_arr[1].to_bits(), 0); // 2 > 2 = false
    assert!(gt_arr[2].is_nan()); // 3 > 2 = true
}

#[test]
fn test_f32x8_blend() {
    let Some(token) = archmage::Avx2FmaToken::try_new() else {
        eprintln!("AVX2+FMA not available");
        return;
    };

    let a = f32x8::load(token, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = f32x8::splat(token, 0.0);
    let threshold = f32x8::splat(token, 4.5);

    // Select a where a < threshold, else b
    let mask = a.simd_lt(threshold);
    let result = f32x8::blend(mask, a, b);

    assert_eq!(result.to_array(), [1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn test_f32x8_horizontal() {
    let Some(token) = archmage::Avx2FmaToken::try_new() else {
        eprintln!("AVX2+FMA not available");
        return;
    };

    let v = f32x8::load(token, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    // Sum: 1+2+3+4+5+6+7+8 = 36
    let sum = v.reduce_add();
    assert!((sum - 36.0).abs() < 0.001);

    // Min: 1.0
    let min = v.reduce_min();
    assert!((min - 1.0).abs() < 0.001);

    // Max: 8.0
    let max = v.reduce_max();
    assert!((max - 8.0).abs() < 0.001);
}

#[test]
fn test_f32x8_scalar_ops() {
    let Some(token) = archmage::Avx2FmaToken::try_new() else {
        eprintln!("AVX2+FMA not available");
        return;
    };

    let v = f32x8::load(token, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

    // Test v + scalar
    let sum = v + 10.0;
    assert_eq!(sum.to_array(), [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0]);

    // Test v * scalar
    let prod = v * 2.0;
    assert_eq!(prod.to_array(), [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);

    // Test v - scalar
    let diff = v - 0.5;
    assert_eq!(diff.to_array(), [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5]);

    // Test v / scalar
    let quot = v / 2.0;
    assert_eq!(quot.to_array(), [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]);
}

#[test]
fn test_f32x8_conversions() {
    let Some(token) = archmage::Avx2FmaToken::try_new() else {
        eprintln!("AVX2+FMA not available");
        return;
    };

    // f32 -> i32 (truncate)
    let f = f32x8::load(token, &[1.9, -2.9, 3.1, -4.1, 5.5, -6.5, 7.0, -8.0]);
    let i = f.to_i32x8();
    assert_eq!(i.to_array(), [1, -2, 3, -4, 5, -6, 7, -8]);

    // f32 -> i32 (round)
    let rounded = f.to_i32x8_round();
    assert_eq!(rounded.to_array(), [2, -3, 3, -4, 6, -6, 7, -8]);

    // i32 -> f32
    let i2 = i32x8::load(token, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let f2 = f32x8::from_i32x8(i2);
    assert_eq!(f2.to_array(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn test_i32x8_basic() {
    let Some(token) = archmage::Avx2FmaToken::try_new() else {
        eprintln!("AVX2+FMA not available");
        return;
    };

    let data = [1i32, 2, 3, 4, 5, 6, 7, 8];
    let v = i32x8::load(token, &data);

    // Test round-trip
    assert_eq!(v.to_array(), data);

    // Test arithmetic
    let two = i32x8::splat(token, 2);
    let sum = v + two;
    assert_eq!(sum.to_array(), [3, 4, 5, 6, 7, 8, 9, 10]);

    // Test mul
    let prod = v * two;
    assert_eq!(prod.to_array(), [2, 4, 6, 8, 10, 12, 14, 16]);
}

#[test]
fn test_i32x8_comparisons() {
    let Some(token) = archmage::Avx2FmaToken::try_new() else {
        eprintln!("AVX2+FMA not available");
        return;
    };

    let a = i32x8::load(token, &[1, 2, 3, 4, 5, 6, 7, 8]);
    let b = i32x8::load(token, &[2, 2, 2, 2, 6, 6, 6, 6]);

    // simd_eq: compare each lane
    let eq = a.simd_eq(b);
    let eq_arr = eq.to_array();
    assert_eq!(eq_arr[0], 0);  // 1 == 2 = false
    assert_eq!(eq_arr[1], -1); // 2 == 2 = true (all-1s = -1 as i32)
    assert_eq!(eq_arr[2], 0);  // 3 == 2 = false

    // simd_gt
    let gt = a.simd_gt(b);
    let gt_arr = gt.to_array();
    assert_eq!(gt_arr[0], 0);  // 1 > 2 = false
    assert_eq!(gt_arr[1], 0);  // 2 > 2 = false
    assert_eq!(gt_arr[2], -1); // 3 > 2 = true

    // simd_lt
    let lt = a.simd_lt(b);
    let lt_arr = lt.to_array();
    assert_eq!(lt_arr[0], -1); // 1 < 2 = true
    assert_eq!(lt_arr[1], 0);  // 2 < 2 = false
}

#[test]
fn test_i32x8_scalar_ops() {
    let Some(token) = archmage::Avx2FmaToken::try_new() else {
        eprintln!("AVX2+FMA not available");
        return;
    };

    let v = i32x8::load(token, &[1, 2, 3, 4, 5, 6, 7, 8]);

    // Test v + scalar
    let sum = v + 10;
    assert_eq!(sum.to_array(), [11, 12, 13, 14, 15, 16, 17, 18]);

    // Test v - scalar
    let diff = v - 1;
    assert_eq!(diff.to_array(), [0, 1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn test_u32x8_comparisons() {
    let Some(token) = archmage::Avx2FmaToken::try_new() else {
        eprintln!("AVX2+FMA not available");
        return;
    };

    // Test unsigned comparison (important: different from signed!)
    let a = u32x8::load(token, &[1, 2, 0xFFFF_FFFF, 4, 5, 6, 7, 8]);
    let b = u32x8::load(token, &[2, 2, 1, 4, 4, 4, 4, 4]);

    // simd_gt for unsigned: 0xFFFF_FFFF > 1 should be true
    let gt = a.simd_gt(b);
    let gt_arr = gt.to_array();
    assert_eq!(gt_arr[0], 0);           // 1 > 2 = false
    assert_eq!(gt_arr[1], 0);           // 2 > 2 = false
    assert_eq!(gt_arr[2], 0xFFFF_FFFF); // 0xFFFF_FFFF > 1 = true (unsigned!)
    assert_eq!(gt_arr[3], 0);           // 4 > 4 = false
    assert_eq!(gt_arr[4], 0xFFFF_FFFF); // 5 > 4 = true
}

#[test]
fn test_f32x4_basic() {
    let Some(token) = archmage::Sse41Token::try_new() else {
        eprintln!("SSE4.1 not available");
        return;
    };

    let data = [1.0f32, 2.0, 3.0, 4.0];
    let v = f32x4::load(token, &data);
    assert_eq!(v.to_array(), data);

    // Test horizontal sum
    let sum = v.reduce_add();
    assert!((sum - 10.0).abs() < 0.001);
}

#[test]
fn test_f32x4_comparisons() {
    let Some(token) = archmage::Sse41Token::try_new() else {
        eprintln!("SSE4.1 not available");
        return;
    };

    let a = f32x4::load(token, &[1.0, 2.0, 3.0, 4.0]);
    let b = f32x4::load(token, &[2.0, 2.0, 2.0, 2.0]);

    // Test simd_lt
    let lt = a.simd_lt(b);
    let lt_arr = lt.to_array();
    assert!(lt_arr[0].is_nan()); // 1 < 2 = true
    assert_eq!(lt_arr[1].to_bits(), 0); // 2 < 2 = false
    assert_eq!(lt_arr[2].to_bits(), 0); // 3 < 2 = false
}
"#,
    );

    code
}
