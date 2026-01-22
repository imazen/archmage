//! Core types and helper functions for SIMD code generation.

use indoc::formatdoc;

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

// ============================================================================
// Code Generation Helpers
// ============================================================================

/// Indent each line of a string by a given number of spaces.
pub fn indent(s: &str, spaces: usize) -> String {
    let prefix = " ".repeat(spaces);
    s.lines()
        .map(|line| {
            if line.is_empty() {
                String::new()
            } else {
                format!("{prefix}{line}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Indent continuation lines (all except the first) by a given number of spaces.
/// Useful for multi-line strings embedded via `{variable}` in formatdoc!.
pub fn indent_continuation(s: &str, spaces: usize) -> String {
    let prefix = " ".repeat(spaces);
    let mut lines = s.lines();
    let mut result = String::new();

    // First line: no extra indent
    if let Some(first) = lines.next() {
        result.push_str(first);
    }

    // Continuation lines: add indent
    for line in lines {
        result.push('\n');
        if !line.is_empty() {
            result.push_str(&prefix);
        }
        result.push_str(line);
    }

    result
}

/// Generate a simple unary method: `pub fn name(self) -> Self { body }`
pub fn gen_unary_method(doc: &str, name: &str, body: &str) -> String {
    indent(
        &formatdoc! {"
            /// {doc}
            #[inline(always)]
            pub fn {name}(self) -> Self {{
                {body}
            }}
        "},
        4,
    ) + "\n"
}

/// Generate a binary method: `pub fn name(self, other: Self) -> Self { body }`
pub fn gen_binary_method(doc: &str, name: &str, body: &str) -> String {
    indent(
        &formatdoc! {"
            /// {doc}
            #[inline(always)]
            pub fn {name}(self, other: Self) -> Self {{
                {body}
            }}
        "},
        4,
    ) + "\n"
}

/// Generate a method returning a scalar: `pub fn name(self) -> T { body }`
#[allow(dead_code)]
pub fn gen_scalar_method(doc: &str, name: &str, return_type: &str, body: &str) -> String {
    let body = indent_continuation(body, 4);
    indent(
        &formatdoc! {"
            /// {doc}
            #[inline(always)]
            pub fn {name}(self) -> {return_type} {{
                {body}
            }}
        "},
        4,
    ) + "\n"
}

/// List of all SIMD types to generate
pub fn all_simd_types() -> Vec<SimdType> {
    vec![
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
    ]
}
