//! Comparison and blend operations for SIMD types.

use super::types::{ElementType, SimdType, SimdWidth, indent, indent_continuation};
use indoc::formatdoc;
use std::fmt::Write;

pub fn generate_comparison_ops(ty: &SimdType) -> String {
    let mut code = String::new();
    let _name = ty.name();
    let prefix = ty.width.x86_prefix();
    let suffix = ty.elem.x86_suffix();
    let _lanes = ty.lanes();

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

pub fn generate_blend_ops(ty: &SimdType) -> String {
    let prefix = ty.width.x86_prefix();
    let suffix = ty.elem.x86_suffix();
    let name = ty.name();

    // Build the blend body based on width and element type
    let blend_body = if ty.width == SimdWidth::W512 {
        // AVX-512 uses mask registers for blend - multi-line body needs extra indent
        let inner = if ty.elem.is_float() {
            let (int_suffix, cast_fn) = if ty.elem == ElementType::F32 {
                ("epi32", "_mm512_castps_si512")
            } else {
                ("epi64", "_mm512_castpd_si512")
            };
            formatdoc! {"
                Self(unsafe {{
                    // Convert vector mask to mask register
                    let m = _mm512_cmpneq_{int_suffix}_mask({cast_fn}(mask.0), _mm512_setzero_si512());
                    {prefix}_mask_blend_{suffix}(m, if_false.0, if_true.0)
                }})"
            }
        } else {
            // Integer types
            formatdoc! {"
                Self(unsafe {{
                    // Convert vector mask to mask register
                    let m = _mm512_cmpneq_{suffix}_mask(mask.0, _mm512_setzero_si512());
                    {prefix}_mask_blend_{suffix}(m, if_false.0, if_true.0)
                }})"
            }
        };
        // Indent continuation lines (skip the first line as it's already at the right position)
        indent_continuation(&inner, 4)
    } else if ty.elem.is_float() {
        format!("Self(unsafe {{ {prefix}_blendv_{suffix}(if_false.0, if_true.0, mask.0) }})")
    } else {
        // Integer blend uses blendv_epi8 which operates on bytes
        format!("Self(unsafe {{ {prefix}_blendv_epi8(if_false.0, if_true.0, mask.0) }})")
    };

    let code = formatdoc! {"
        // ========== Blending/Selection ==========

        /// Select lanes from `if_true` where mask is all-1s, `if_false` where mask is all-0s.
        ///
        /// The mask should come from a comparison operation like `simd_lt()`.
        ///
        /// # Example
        /// ```ignore
        /// let a = {name}::splat(token, 1.0);
        /// let b = {name}::splat(token, 2.0);
        /// let mask = a.simd_lt(b);  // all true
        /// let result = {name}::blend(mask, a, b);  // selects a
        /// ```
        #[inline(always)]
        pub fn blend(mask: Self, if_true: Self, if_false: Self) -> Self {{
            {blend_body}
        }}
    "};

    indent(&code, 4) + "\n"
}

/// Generate boolean reduction operations (all_true, any_true, bitmask) for x86 integer types
pub fn generate_boolean_reductions(ty: &SimdType) -> String {
    // Only for integer types (not floats)
    if ty.elem.is_float() {
        return String::new();
    }

    let mut code = String::new();
    let prefix = ty.width.x86_prefix();
    let lanes = ty.lanes();

    writeln!(code, "    // ========== Boolean Reductions ==========\n").unwrap();

    // all_true - check if all lanes are non-zero (have their sign bit set for masks)
    // For masks from comparisons, lanes are all-1s or all-0s
    writeln!(code, "    /// Returns true if all lanes are non-zero (truthy).").unwrap();
    writeln!(code, "    ///").unwrap();
    writeln!(
        code,
        "    /// Typically used with comparison results where true lanes are all-1s."
    )
    .unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn all_true(self) -> bool {{").unwrap();

    match ty.width {
        SimdWidth::W128 => {
            // Use movemask_epi8 and check all bits
            // All lanes being truthy means all 16 bytes have high bit set (for masks)
            writeln!(
                code,
                "        unsafe {{ {}_movemask_epi8(self.0) == 0xFFFF_u32 as i32 }}",
                prefix
            )
            .unwrap();
        }
        SimdWidth::W256 => {
            // For 256-bit, movemask returns 32 bits
            writeln!(
                code,
                "        unsafe {{ {}_movemask_epi8(self.0) == -1_i32 }}",
                prefix
            )
            .unwrap();
        }
        SimdWidth::W512 => {
            // AVX-512 uses kmov to extract mask - mask type depends on lane count
            let cmp_suffix = ty.elem.x86_suffix();
            writeln!(code, "        unsafe {{").unwrap();
            writeln!(
                code,
                "            let mask = {}_cmpneq_{}_mask(self.0, {}_setzero_si512());",
                prefix, cmp_suffix, prefix
            )
            .unwrap();
            // The mask type matches the lane count: 64 lanes -> u64, 8 lanes -> u8, etc.
            let full_mask = match lanes {
                64 => "0xFFFF_FFFF_FFFF_FFFFu64",
                32 => "0xFFFF_FFFFu32",
                16 => "0xFFFFu16",
                8 => "0xFFu8",
                _ => "0u64",
            };
            writeln!(code, "            mask == {}", full_mask).unwrap();
            writeln!(code, "        }}").unwrap();
        }
    }
    writeln!(code, "    }}\n").unwrap();

    // any_true - check if any lane is non-zero
    writeln!(code, "    /// Returns true if any lane is non-zero (truthy).").unwrap();
    writeln!(code, "    #[inline(always)]").unwrap();
    writeln!(code, "    pub fn any_true(self) -> bool {{").unwrap();

    match ty.width {
        SimdWidth::W128 | SimdWidth::W256 => {
            writeln!(
                code,
                "        unsafe {{ {}_movemask_epi8(self.0) != 0 }}",
                prefix
            )
            .unwrap();
        }
        SimdWidth::W512 => {
            let cmp_suffix = ty.elem.x86_suffix();
            writeln!(code, "        unsafe {{").unwrap();
            writeln!(
                code,
                "            let mask = {}_cmpneq_{}_mask(self.0, {}_setzero_si512());",
                prefix, cmp_suffix, prefix
            )
            .unwrap();
            writeln!(code, "            mask != 0").unwrap();
            writeln!(code, "        }}").unwrap();
        }
    }
    writeln!(code, "    }}\n").unwrap();

    // bitmask - extract high bit of each lane as a bitmask
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

    match ty.width {
        SimdWidth::W128 => {
            // movemask_epi8 extracts bit 7 of each byte
            // For larger types, we need to handle differently
            match ty.elem {
                ElementType::I8 | ElementType::U8 => {
                    // 16 lanes -> 16 bits
                    writeln!(
                        code,
                        "        unsafe {{ {}_movemask_epi8(self.0) as u32 }}",
                        prefix
                    )
                    .unwrap();
                }
                ElementType::I16 | ElementType::U16 => {
                    // 8 lanes, need to pack to bytes first or use arithmetic shift
                    // Actually simpler: shift right by 15, pack to bytes, movemask
                    // But packing would lose info. Use: shift each lane right by 15, pack i16->i8
                    writeln!(code, "        unsafe {{").unwrap();
                    writeln!(
                        code,
                        "            // Shift right to get sign bit in LSB, pack to bytes"
                    )
                    .unwrap();
                    writeln!(
                        code,
                        "            let shifted = _mm_srai_epi16::<15>(self.0);"
                    )
                    .unwrap();
                    writeln!(
                        code,
                        "            let packed = _mm_packs_epi16(shifted, shifted);"
                    )
                    .unwrap();
                    writeln!(
                        code,
                        "            (_mm_movemask_epi8(packed) & 0xFF) as u32"
                    )
                    .unwrap();
                    writeln!(code, "        }}").unwrap();
                }
                ElementType::I32 | ElementType::U32 => {
                    // 4 lanes - use movemask_ps after reinterpret
                    writeln!(code, "        unsafe {{").unwrap();
                    writeln!(
                        code,
                        "            _mm_movemask_ps(_mm_castsi128_ps(self.0)) as u32"
                    )
                    .unwrap();
                    writeln!(code, "        }}").unwrap();
                }
                ElementType::I64 | ElementType::U64 => {
                    // 2 lanes - use movemask_pd after reinterpret
                    writeln!(code, "        unsafe {{").unwrap();
                    writeln!(
                        code,
                        "            _mm_movemask_pd(_mm_castsi128_pd(self.0)) as u32"
                    )
                    .unwrap();
                    writeln!(code, "        }}").unwrap();
                }
                _ => unreachable!(),
            }
        }
        SimdWidth::W256 => {
            match ty.elem {
                ElementType::I8 | ElementType::U8 => {
                    writeln!(
                        code,
                        "        unsafe {{ {}_movemask_epi8(self.0) as u32 }}",
                        prefix
                    )
                    .unwrap();
                }
                ElementType::I16 | ElementType::U16 => {
                    writeln!(code, "        unsafe {{").unwrap();
                    writeln!(
                        code,
                        "            let shifted = _mm256_srai_epi16::<15>(self.0);"
                    )
                    .unwrap();
                    writeln!(
                        code,
                        "            let packed = _mm256_packs_epi16(shifted, shifted);"
                    )
                    .unwrap();
                    writeln!(code, "            // packs interleaves, need to extract").unwrap();
                    writeln!(
                        code,
                        "            let lo = _mm256_castsi256_si128(packed);"
                    )
                    .unwrap();
                    writeln!(
                        code,
                        "            let hi = _mm256_extracti128_si256::<1>(packed);"
                    )
                    .unwrap();
                    writeln!(
                        code,
                        "            ((_mm_movemask_epi8(lo) & 0xFF) | ((_mm_movemask_epi8(hi) & 0xFF) << 8)) as u32"
                    )
                    .unwrap();
                    writeln!(code, "        }}").unwrap();
                }
                ElementType::I32 | ElementType::U32 => {
                    writeln!(code, "        unsafe {{").unwrap();
                    writeln!(
                        code,
                        "            _mm256_movemask_ps(_mm256_castsi256_ps(self.0)) as u32"
                    )
                    .unwrap();
                    writeln!(code, "        }}").unwrap();
                }
                ElementType::I64 | ElementType::U64 => {
                    writeln!(code, "        unsafe {{").unwrap();
                    writeln!(
                        code,
                        "            _mm256_movemask_pd(_mm256_castsi256_pd(self.0)) as u32"
                    )
                    .unwrap();
                    writeln!(code, "        }}").unwrap();
                }
                _ => unreachable!(),
            }
        }
        SimdWidth::W512 => {
            // AVX-512 has direct mask extraction
            let cmp_suffix = ty.elem.x86_suffix();
            writeln!(code, "        unsafe {{").unwrap();
            writeln!(
                code,
                "            {}_cmpneq_{}_mask(self.0, {}_setzero_si512()) as u32",
                prefix, cmp_suffix, prefix
            )
            .unwrap();
            writeln!(code, "        }}").unwrap();
        }
    }
    writeln!(code, "    }}\n").unwrap();

    code
}
