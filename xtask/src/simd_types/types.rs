//! Core types and helper functions for SIMD code generation.

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

    /// x86 intrinsic type for floats (used by the generic backend generator)
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

    /// x86 intrinsic type for integers (used by the generic backend generator)
    pub fn x86_int_type(&self) -> &'static str {
        match self {
            Self::W128 => "__m128i",
            Self::W256 => "__m256i",
            Self::W512 => "__m512i",
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
