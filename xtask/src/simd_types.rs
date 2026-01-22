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

    writeln!(code, "}}\n").unwrap();

    // Operator implementations
    code.push_str(&generate_operator_impls(ty, &cfg_attr));

    code
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
fn test_f32x4_basic() {
    let Some(token) = archmage::Sse41Token::try_new() else {
        eprintln!("SSE4.1 not available");
        return;
    };

    let data = [1.0f32, 2.0, 3.0, 4.0];
    let v = f32x4::load(token, &data);
    assert_eq!(v.to_array(), data);
}
"#,
    );

    code
}
