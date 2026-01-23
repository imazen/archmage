//! x86-64 architecture code generation.

use super::Arch;
use crate::simd_types::types::{ElementType, SimdWidth};

/// x86-64 architecture (SSE, AVX, AVX-512)
pub struct X86;

impl Arch for X86 {
    fn target_arch() -> &'static str {
        "x86_64"
    }

    fn intrinsic_type(elem: ElementType, width: SimdWidth) -> &'static str {
        match (elem, width) {
            // Floats have specific types
            (ElementType::F32, SimdWidth::W128) => "__m128",
            (ElementType::F32, SimdWidth::W256) => "__m256",
            (ElementType::F32, SimdWidth::W512) => "__m512",
            (ElementType::F64, SimdWidth::W128) => "__m128d",
            (ElementType::F64, SimdWidth::W256) => "__m256d",
            (ElementType::F64, SimdWidth::W512) => "__m512d",
            // All integers use the same type per width
            (_, SimdWidth::W128) => "__m128i",
            (_, SimdWidth::W256) => "__m256i",
            (_, SimdWidth::W512) => "__m512i",
        }
    }

    fn prefix(width: SimdWidth) -> &'static str {
        match width {
            SimdWidth::W128 => "_mm",
            SimdWidth::W256 => "_mm256",
            SimdWidth::W512 => "_mm512",
        }
    }

    fn suffix(elem: ElementType) -> &'static str {
        match elem {
            ElementType::F32 => "ps",
            ElementType::F64 => "pd",
            ElementType::I8 | ElementType::U8 => "epi8",
            ElementType::I16 | ElementType::U16 => "epi16",
            ElementType::I32 | ElementType::U32 => "epi32",
            ElementType::I64 | ElementType::U64 => "epi64",
        }
    }

    fn minmax_suffix(elem: ElementType) -> &'static str {
        match elem {
            ElementType::F32 => "ps",
            ElementType::F64 => "pd",
            ElementType::I8 => "epi8",
            ElementType::U8 => "epu8",
            ElementType::I16 => "epi16",
            ElementType::U16 => "epu16",
            ElementType::I32 => "epi32",
            ElementType::U32 => "epu32",
            ElementType::I64 => "epi64",
            ElementType::U64 => "epu64",
        }
    }

    fn required_token(width: SimdWidth, _needs_int_ops: bool) -> &'static str {
        match width {
            SimdWidth::W128 => "Sse41Token",
            SimdWidth::W256 => "Avx2FmaToken",
            SimdWidth::W512 => "X64V4Token",
        }
    }

    fn required_feature(width: SimdWidth) -> Option<&'static str> {
        match width {
            SimdWidth::W128 | SimdWidth::W256 => None,
            SimdWidth::W512 => Some("avx512"),
        }
    }

    fn supports_width(width: SimdWidth) -> bool {
        // x86-64 supports all widths
        match width {
            SimdWidth::W128 | SimdWidth::W256 | SimdWidth::W512 => true,
        }
    }
}

// ============================================================================
// x86-specific intrinsic helpers
// ============================================================================

impl X86 {
    /// Integer suffix for cast operations (si128, si256, si512)
    pub fn int_cast_suffix(width: SimdWidth) -> &'static str {
        match width {
            SimdWidth::W128 => "si128",
            SimdWidth::W256 => "si256",
            SimdWidth::W512 => "si512",
        }
    }

    /// Get the comparison intrinsic name pattern
    /// For AVX-512, uses mask variant; for SSE/AVX, uses vector variant
    pub fn cmp_intrinsic(width: SimdWidth, suffix: &str) -> String {
        let prefix = Self::prefix(width);
        if width == SimdWidth::W512 {
            format!("{}_cmp_{}_mask", prefix, suffix)
        } else {
            format!("{}_cmp_{}", prefix, suffix)
        }
    }

    /// Get the blend intrinsic name pattern
    /// For AVX-512, uses mask_blend; for SSE/AVX, uses blendv
    pub fn blend_intrinsic(width: SimdWidth, suffix: &str) -> String {
        let prefix = Self::prefix(width);
        if width == SimdWidth::W512 {
            format!("{}_mask_blend_{}", prefix, suffix)
        } else {
            format!("{}_blendv_{}", prefix, suffix)
        }
    }

    /// Get the floor intrinsic (AVX-512 uses roundscale)
    pub fn floor_intrinsic(width: SimdWidth, suffix: &str) -> String {
        let prefix = Self::prefix(width);
        if width == SimdWidth::W512 {
            format!("{}_roundscale_{}::<0x01>", prefix, suffix)
        } else {
            format!("{}_floor_{}", prefix, suffix)
        }
    }

    /// Whether this width uses mask registers for comparisons
    pub fn uses_mask_registers(width: SimdWidth) -> bool {
        width == SimdWidth::W512
    }
}
