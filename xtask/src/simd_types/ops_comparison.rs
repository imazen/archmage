//! Comparison and blend operations for SIMD types.

use crate::simd_types::types::{ElementType, SimdType, SimdWidth};
use indoc::formatdoc;

/// Generate a comparison method
fn gen_cmp_method(method: &str, doc: &str, extra_doc: Option<&str>, body: &str) -> String {
    let extra = extra_doc.map(|s| format!("/// {s}\n")).unwrap_or_default();
    formatdoc! {"
        /// Lane-wise {doc} comparison.
        ///
        /// Returns a mask where each lane is all-1s if {doc}, all-0s otherwise.
        {extra}#[inline(always)]
        pub fn {method}(self, other: Self) -> Self {{
        {body}
        }}

    "}
}

pub fn generate_comparison_ops(ty: &SimdType) -> String {
    let prefix = ty.width.x86_prefix();
    let suffix = ty.elem.x86_suffix();

    let mut code = formatdoc! {"
        // ========== Comparisons ==========
        // These return a mask where each lane is all-1s (true) or all-0s (false).
        // Use with `blend()` to select values based on the comparison result.

    "};

    if ty.elem.is_float() {
        code.push_str(&generate_float_comparisons(ty, prefix, suffix));
    } else if ty.width == SimdWidth::W512 {
        code.push_str(&generate_avx512_int_comparisons(ty, prefix));
    } else {
        code.push_str(&generate_sse_avx_int_comparisons(ty, prefix, suffix));
    }

    code
}

fn generate_float_comparisons(ty: &SimdType, prefix: &str, suffix: &str) -> String {
    let mut code = String::new();

    if ty.width == SimdWidth::W128 {
        // SSE uses separate intrinsics
        let blend_doc = Some("Use with `blend(mask, if_true, if_false)` to select values.");
        code.push_str(&gen_cmp_method(
            "simd_eq",
            "equality",
            blend_doc,
            &format!("Self(unsafe {{ {prefix}_cmpeq_{suffix}(self.0, other.0) }})"),
        ));
        code.push_str(&gen_cmp_method(
            "simd_ne",
            "inequality",
            None,
            &format!("Self(unsafe {{ {prefix}_cmpneq_{suffix}(self.0, other.0) }})"),
        ));
        code.push_str(&gen_cmp_method(
            "simd_lt",
            "less-than",
            None,
            &format!("Self(unsafe {{ {prefix}_cmplt_{suffix}(self.0, other.0) }})"),
        ));
        code.push_str(&gen_cmp_method(
            "simd_le",
            "less-than-or-equal",
            None,
            &format!("Self(unsafe {{ {prefix}_cmple_{suffix}(self.0, other.0) }})"),
        ));
        code.push_str(&gen_cmp_method(
            "simd_gt",
            "greater-than",
            None,
            &format!("Self(unsafe {{ {prefix}_cmpgt_{suffix}(self.0, other.0) }})"),
        ));
        code.push_str(&gen_cmp_method(
            "simd_ge",
            "greater-than-or-equal",
            None,
            &format!("Self(unsafe {{ {prefix}_cmpge_{suffix}(self.0, other.0) }})"),
        ));
    } else if ty.width == SimdWidth::W512 {
        // AVX-512 uses mask-returning intrinsics
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
            let extra = if method == "simd_eq" {
                Some("Use with `blend(mask, if_true, if_false)` to select values.")
            } else {
                None
            };
            let body = formatdoc! {"
                Self(unsafe {{
                let mask = {prefix}_cmp_{mask_suffix}_mask::<{cmp_const}>(self.0, other.0);
                // Expand mask to vector: -1 where true, 0 where false
                {cast_fn}({prefix}_maskz_set1_{int_suffix}(mask, -1))
                }})"
            };
            code.push_str(&gen_cmp_method(method, doc, extra, &body));
        }
    } else {
        // AVX (256-bit) uses _cmp_ps/pd with predicate constants
        let cmp_intrinsic = format!("{prefix}_cmp_{suffix}");
        let blend_doc = Some("Use with `blend(mask, if_true, if_false)` to select values.");

        code.push_str(&gen_cmp_method(
            "simd_eq",
            "equality",
            blend_doc,
            &format!("Self(unsafe {{ {cmp_intrinsic}::<_CMP_EQ_OQ>(self.0, other.0) }})"),
        ));
        code.push_str(&gen_cmp_method(
            "simd_ne",
            "inequality",
            None,
            &format!("Self(unsafe {{ {cmp_intrinsic}::<_CMP_NEQ_OQ>(self.0, other.0) }})"),
        ));
        code.push_str(&gen_cmp_method(
            "simd_lt",
            "less-than",
            None,
            &format!("Self(unsafe {{ {cmp_intrinsic}::<_CMP_LT_OQ>(self.0, other.0) }})"),
        ));
        code.push_str(&gen_cmp_method(
            "simd_le",
            "less-than-or-equal",
            None,
            &format!("Self(unsafe {{ {cmp_intrinsic}::<_CMP_LE_OQ>(self.0, other.0) }})"),
        ));
        code.push_str(&gen_cmp_method(
            "simd_gt",
            "greater-than",
            None,
            &format!("Self(unsafe {{ {cmp_intrinsic}::<_CMP_GT_OQ>(self.0, other.0) }})"),
        ));
        code.push_str(&gen_cmp_method(
            "simd_ge",
            "greater-than-or-equal",
            None,
            &format!("Self(unsafe {{ {cmp_intrinsic}::<_CMP_GE_OQ>(self.0, other.0) }})"),
        ));
    }

    code
}

fn generate_avx512_int_comparisons(ty: &SimdType, prefix: &str) -> String {
    let mut code = String::new();
    let suffix = ty.elem.x86_suffix();

    let set1_suffix = match ty.elem {
        ElementType::I8 | ElementType::U8 => "epi8",
        ElementType::I16 | ElementType::U16 => "epi16",
        ElementType::I32 | ElementType::U32 => "epi32",
        ElementType::I64 | ElementType::U64 => "epi64",
        _ => unreachable!(),
    };

    let cmp_fn = if ty.elem.is_signed() {
        format!("{prefix}_cmp_{suffix}_mask")
    } else {
        let unsigned_suffix = match ty.elem {
            ElementType::U8 => "epu8",
            ElementType::U16 => "epu16",
            ElementType::U32 => "epu32",
            ElementType::U64 => "epu64",
            _ => unreachable!(),
        };
        format!("{prefix}_cmp_{unsigned_suffix}_mask")
    };

    for (method, doc, cmp_const) in [
        ("simd_eq", "equality", "_MM_CMPINT_EQ"),
        ("simd_ne", "inequality", "_MM_CMPINT_NE"),
        ("simd_lt", "less-than", "_MM_CMPINT_LT"),
        ("simd_le", "less-than-or-equal", "_MM_CMPINT_LE"),
        ("simd_gt", "greater-than", "_MM_CMPINT_LT"), // Use LT with swapped args
        ("simd_ge", "greater-than-or-equal", "_MM_CMPINT_LE"), // Use LE with swapped args
    ] {
        let extra = if method == "simd_eq" {
            Some("Use with `blend(mask, if_true, if_false)` to select values.")
        } else {
            None
        };

        let (arg_a, arg_b) = if method == "simd_gt" || method == "simd_ge" {
            ("other.0", "self.0")
        } else {
            ("self.0", "other.0")
        };

        let body = formatdoc! {"
            Self(unsafe {{
            let mask = {cmp_fn}::<{cmp_const}>({arg_a}, {arg_b});
            // Expand mask to vector: -1 where true, 0 where false
            {prefix}_maskz_set1_{set1_suffix}(mask, -1)
            }})"
        };
        code.push_str(&gen_cmp_method(method, doc, extra, &body));
    }

    code
}

fn generate_sse_avx_int_comparisons(ty: &SimdType, prefix: &str, suffix: &str) -> String {
    let mut code = String::new();
    let bits = ty.width.bits();

    let set1_suffix = match ty.elem {
        ElementType::I8 | ElementType::U8 => "epi8",
        ElementType::I16 | ElementType::U16 => "epi16",
        ElementType::I32 | ElementType::U32 => "epi32",
        ElementType::I64 | ElementType::U64 => "epi64x",
        _ => unreachable!(),
    };

    // simd_eq - works for all integer types
    code.push_str(&gen_cmp_method(
        "simd_eq",
        "equality",
        Some("Use with `blend(mask, if_true, if_false)` to select values."),
        &format!("Self(unsafe {{ {prefix}_cmpeq_{suffix}(self.0, other.0) }})"),
    ));

    // simd_ne = NOT eq
    let ne_body = formatdoc! {"
        Self(unsafe {{
        let eq = {prefix}_cmpeq_{suffix}(self.0, other.0);
        let ones = {prefix}_set1_{set1_suffix}(-1);
        {prefix}_xor_si{bits}(eq, ones)
        }})"
    };
    code.push_str(&gen_cmp_method("simd_ne", "inequality", None, &ne_body));

    if ty.elem.is_signed() {
        code.push_str(&generate_signed_int_comparisons(ty, prefix, suffix));
    } else {
        code.push_str(&generate_unsigned_int_comparisons(ty, prefix));
    }

    code
}

fn generate_signed_int_comparisons(ty: &SimdType, prefix: &str, suffix: &str) -> String {
    let mut code = String::new();
    let bits = ty.width.bits();

    let set1_suffix = match ty.elem {
        ElementType::I8 => "epi8",
        ElementType::I16 => "epi16",
        ElementType::I32 => "epi32",
        ElementType::I64 => "epi64x",
        _ => unreachable!(),
    };

    // simd_gt - direct intrinsic
    code.push_str(&gen_cmp_method(
        "simd_gt",
        "greater-than",
        None,
        &format!("Self(unsafe {{ {prefix}_cmpgt_{suffix}(self.0, other.0) }})"),
    ));

    // simd_lt = gt with swapped args
    code.push_str(&gen_cmp_method(
        "simd_lt",
        "less-than",
        None,
        &format!("Self(unsafe {{ {prefix}_cmpgt_{suffix}(other.0, self.0) }})"),
    ));

    // simd_ge = NOT lt = NOT gt(b, a)
    let ge_body = formatdoc! {"
        Self(unsafe {{
        let lt = {prefix}_cmpgt_{suffix}(other.0, self.0);
        let ones = {prefix}_set1_{set1_suffix}(-1);
        {prefix}_xor_si{bits}(lt, ones)
        }})"
    };
    code.push_str(&gen_cmp_method(
        "simd_ge",
        "greater-than-or-equal",
        None,
        &ge_body,
    ));

    // simd_le = NOT gt
    let le_body = formatdoc! {"
        Self(unsafe {{
        let gt = {prefix}_cmpgt_{suffix}(self.0, other.0);
        let ones = {prefix}_set1_{set1_suffix}(-1);
        {prefix}_xor_si{bits}(gt, ones)
        }})"
    };
    code.push_str(&gen_cmp_method(
        "simd_le",
        "less-than-or-equal",
        None,
        &le_body,
    ));

    code
}

fn generate_unsigned_int_comparisons(ty: &SimdType, prefix: &str) -> String {
    let mut code = String::new();
    let bits = ty.width.bits();

    let (bias_val, signed_suffix, set1_suffix) = match ty.elem {
        ElementType::U8 => ("0x80u8 as i8", "epi8", "epi8"),
        ElementType::U16 => ("0x8000u16 as i16", "epi16", "epi16"),
        ElementType::U32 => ("0x8000_0000u32 as i32", "epi32", "epi32"),
        ElementType::U64 => ("0x8000_0000_0000_0000u64 as i64", "epi64", "epi64x"),
        _ => unreachable!(),
    };

    // simd_gt - use signed comparison with bias trick
    let gt_body = formatdoc! {"
        Self(unsafe {{
        // Flip sign bit to convert unsigned to signed comparison
        let bias = {prefix}_set1_{set1_suffix}({bias_val});
        let a = {prefix}_xor_si{bits}(self.0, bias);
        let b = {prefix}_xor_si{bits}(other.0, bias);
        {prefix}_cmpgt_{signed_suffix}(a, b)
        }})"
    };
    code.push_str(&gen_cmp_method(
        "simd_gt",
        "greater-than (unsigned)",
        None,
        &gt_body,
    ));

    // simd_lt = gt with swapped args
    code.push_str(&gen_cmp_method(
        "simd_lt",
        "less-than (unsigned)",
        None,
        "other.simd_gt(self)",
    ));

    // simd_ge = NOT lt
    let ge_body = formatdoc! {"
        Self(unsafe {{
        let lt = other.simd_gt(self);
        let ones = {prefix}_set1_{set1_suffix}(-1);
        {prefix}_xor_si{bits}(lt.0, ones)
        }})"
    };
    code.push_str(&gen_cmp_method(
        "simd_ge",
        "greater-than-or-equal (unsigned)",
        None,
        &ge_body,
    ));

    // simd_le = NOT gt
    let le_body = formatdoc! {"
        Self(unsafe {{
        let gt = self.simd_gt(other);
        let ones = {prefix}_set1_{set1_suffix}(-1);
        {prefix}_xor_si{bits}(gt.0, ones)
        }})"
    };
    code.push_str(&gen_cmp_method(
        "simd_le",
        "less-than-or-equal (unsigned)",
        None,
        &le_body,
    ));

    code
}

pub fn generate_blend_ops(ty: &SimdType) -> String {
    let prefix = ty.width.x86_prefix();
    let suffix = ty.elem.x86_suffix();
    let name = ty.name();

    let blend_body = if ty.width == SimdWidth::W512 {
        if ty.elem.is_float() {
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
            formatdoc! {"
                Self(unsafe {{
                // Convert vector mask to mask register
                let m = _mm512_cmpneq_{suffix}_mask(mask.0, _mm512_setzero_si512());
                {prefix}_mask_blend_{suffix}(m, if_false.0, if_true.0)
                }})"
            }
        }
    } else if ty.elem.is_float() {
        format!("Self(unsafe {{ {prefix}_blendv_{suffix}(if_false.0, if_true.0, mask.0) }})")
    } else {
        format!("Self(unsafe {{ {prefix}_blendv_epi8(if_false.0, if_true.0, mask.0) }})")
    };

    formatdoc! {"
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

    "}
}

pub fn generate_boolean_reductions(ty: &SimdType) -> String {
    if ty.elem.is_float() {
        return String::new();
    }

    let prefix = ty.width.x86_prefix();
    let lanes = ty.lanes();
    let cmp_suffix = ty.elem.x86_suffix();

    let mut code = String::from("// ========== Boolean Reductions ==========\n\n");

    // all_true
    let all_true_body = match ty.width {
        SimdWidth::W128 => {
            format!("unsafe {{ {prefix}_movemask_epi8(self.0) == 0xFFFF_u32 as i32 }}")
        }
        SimdWidth::W256 => format!("unsafe {{ {prefix}_movemask_epi8(self.0) == -1_i32 }}"),
        SimdWidth::W512 => {
            let full_mask = match lanes {
                64 => "0xFFFF_FFFF_FFFF_FFFFu64",
                32 => "0xFFFF_FFFFu32",
                16 => "0xFFFFu16",
                8 => "0xFFu8",
                _ => "0u64",
            };
            formatdoc! {"
                unsafe {{
                let mask = {prefix}_cmpneq_{cmp_suffix}_mask(self.0, {prefix}_setzero_si512());
                mask == {full_mask}
                }}"
            }
        }
    };
    code.push_str(&formatdoc! {"
        /// Returns true if all lanes are non-zero (truthy).
        ///
        /// Typically used with comparison results where true lanes are all-1s.
        #[inline(always)]
        pub fn all_true(self) -> bool {{
        {all_true_body}
        }}

    "});

    // any_true
    let any_true_body = match ty.width {
        SimdWidth::W128 | SimdWidth::W256 => {
            format!("unsafe {{ {prefix}_movemask_epi8(self.0) != 0 }}")
        }
        SimdWidth::W512 => {
            formatdoc! {"
                unsafe {{
                let mask = {prefix}_cmpneq_{cmp_suffix}_mask(self.0, {prefix}_setzero_si512());
                mask != 0
                }}"
            }
        }
    };
    code.push_str(&formatdoc! {"
        /// Returns true if any lane is non-zero (truthy).
        #[inline(always)]
        pub fn any_true(self) -> bool {{
        {any_true_body}
        }}

    "});

    // bitmask
    let bitmask_body = generate_bitmask_body(ty, prefix);
    code.push_str(&formatdoc! {"
        /// Extract the high bit of each lane as a bitmask.
        ///
        /// Returns a u32 where bit N corresponds to the sign bit of lane N.
        #[inline(always)]
        pub fn bitmask(self) -> u32 {{
        {bitmask_body}
        }}

    "});

    code
}

fn generate_bitmask_body(ty: &SimdType, prefix: &str) -> String {
    match ty.width {
        SimdWidth::W128 => match ty.elem {
            ElementType::I8 | ElementType::U8 => {
                format!("unsafe {{ {prefix}_movemask_epi8(self.0) as u32 }}")
            }
            ElementType::I16 | ElementType::U16 => {
                formatdoc! {"
                    unsafe {{
                    // Shift right to get sign bit in LSB, pack to bytes
                    let shifted = _mm_srai_epi16::<15>(self.0);
                    let packed = _mm_packs_epi16(shifted, shifted);
                    (_mm_movemask_epi8(packed) & 0xFF) as u32
                    }}"
                }
            }
            ElementType::I32 | ElementType::U32 => {
                "unsafe { _mm_movemask_ps(_mm_castsi128_ps(self.0)) as u32 }".into()
            }
            ElementType::I64 | ElementType::U64 => {
                "unsafe { _mm_movemask_pd(_mm_castsi128_pd(self.0)) as u32 }".into()
            }
            _ => unreachable!(),
        },
        SimdWidth::W256 => match ty.elem {
            ElementType::I8 | ElementType::U8 => {
                format!("unsafe {{ {prefix}_movemask_epi8(self.0) as u32 }}")
            }
            ElementType::I16 | ElementType::U16 => {
                formatdoc! {"
                    unsafe {{
                    let shifted = _mm256_srai_epi16::<15>(self.0);
                    let packed = _mm256_packs_epi16(shifted, shifted);
                    // packs interleaves, need to extract
                    let lo = _mm256_castsi256_si128(packed);
                    let hi = _mm256_extracti128_si256::<1>(packed);
                    ((_mm_movemask_epi8(lo) & 0xFF) | ((_mm_movemask_epi8(hi) & 0xFF) << 8)) as u32
                    }}"
                }
            }
            ElementType::I32 | ElementType::U32 => {
                "unsafe { _mm256_movemask_ps(_mm256_castsi256_ps(self.0)) as u32 }".into()
            }
            ElementType::I64 | ElementType::U64 => {
                "unsafe { _mm256_movemask_pd(_mm256_castsi256_pd(self.0)) as u32 }".into()
            }
            _ => unreachable!(),
        },
        SimdWidth::W512 => {
            let cmp_suffix = ty.elem.x86_suffix();
            formatdoc! {"
                unsafe {{
                {prefix}_cmpneq_{cmp_suffix}_mask(self.0, {prefix}_setzero_si512()) as u32
                }}"
            }
        }
    }
}
