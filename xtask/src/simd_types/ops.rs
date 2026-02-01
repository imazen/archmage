//! Standard SIMD operations (horizontal, conversion, math, etc.)

pub use super::ops_comparison::{generate_blend_ops, generate_boolean_reductions, generate_comparison_ops};

use super::types::{
    ElementType, SimdType, SimdWidth, gen_binary_method, gen_unary_method, indent,
    indent_continuation,
};
use indoc::formatdoc;
use std::fmt::Write;

pub fn generate_horizontal_ops(ty: &SimdType) -> String {
    let mut code = String::new();
    let _prefix = ty.width.x86_prefix();
    let _suffix = ty.elem.x86_suffix();
    let _name = ty.name();
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
            "        self.as_array().iter().copied().fold(0_{}, {}::wrapping_add)",
            elem, elem
        )
        .unwrap();
        writeln!(code, "    }}\n").unwrap();
    }

    code
}

pub fn generate_conversion_ops(ty: &SimdType) -> String {
    let mut code = String::new();
    let prefix = ty.width.x86_prefix();
    let _name = ty.name();
    let lanes = ty.lanes();
    let _token = ty.token();

    // Only generate conversions for matching lane counts
    // f32 <-> i32 conversions
    if ty.elem == ElementType::F32 {
        let int_name = format!("i32x{}", lanes);
        let _uint_name = format!("u32x{}", lanes);

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

pub fn generate_scalar_ops(ty: &SimdType, cfg_attr: &str) -> String {
    let mut code = String::new();
    let name = ty.name();
    let elem = ty.elem.name();
    let _token = ty.token();

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

pub fn generate_math_ops(ty: &SimdType) -> String {
    let mut code = String::new();
    let prefix = ty.width.x86_prefix();
    let suffix = ty.elem.x86_suffix();
    let minmax_suffix = ty.elem.x86_minmax_suffix();
    let bits = ty.width.bits();

    // Min/Max - i64/u64 only available in AVX-512
    let has_minmax = ty.elem.is_float()
        || ty.width == SimdWidth::W512
        || !matches!(ty.elem, ElementType::I64 | ElementType::U64);

    if has_minmax {
        code.push_str(&gen_binary_method(
            "Element-wise minimum",
            "min",
            &format!("Self(unsafe {{ {prefix}_min_{minmax_suffix}(self.0, other.0) }})"),
        ));
        code.push_str(&gen_binary_method(
            "Element-wise maximum",
            "max",
            &format!("Self(unsafe {{ {prefix}_max_{minmax_suffix}(self.0, other.0) }})"),
        ));
        code.push_str(&indent(
            &formatdoc! {"
                /// Clamp values between lo and hi
                #[inline(always)]
                pub fn clamp(self, lo: Self, hi: Self) -> Self {{
                    self.max(lo).min(hi)
                }}
            "},
            4,
        ));
        code.push('\n');
    }

    // Float-specific operations
    if ty.elem.is_float() {
        code.push_str(&gen_unary_method(
            "Square root",
            "sqrt",
            &format!("Self(unsafe {{ {prefix}_sqrt_{suffix}(self.0) }})"),
        ));

        // Abs for floats uses AND with sign mask
        let abs_mask = if ty.elem == ElementType::F32 {
            "0x7FFF_FFFFu32 as i32"
        } else {
            "0x7FFF_FFFF_FFFF_FFFFu64 as i64"
        };
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

        let abs_body = formatdoc! {"
            Self(unsafe {{
                let mask = {prefix}_castsi{bits}_{cast_to}({prefix}_set1_{set1_int}({abs_mask}));
                {prefix}_and_{suffix}(self.0, mask)
            }})"
        };
        code.push_str(&gen_unary_method(
            "Absolute value",
            "abs",
            &indent_continuation(&abs_body, 4),
        ));

        // Floor/ceil/round for floats
        if ty.width == SimdWidth::W512 {
            // AVX-512 uses roundscale intrinsics
            code.push_str(&gen_unary_method(
                "Round toward negative infinity",
                "floor",
                &format!("Self(unsafe {{ {prefix}_roundscale_{suffix}::<0x01>(self.0) }})"),
            ));
            code.push_str(&gen_unary_method(
                "Round toward positive infinity",
                "ceil",
                &format!("Self(unsafe {{ {prefix}_roundscale_{suffix}::<0x02>(self.0) }})"),
            ));
            code.push_str(&gen_unary_method(
                "Round to nearest integer",
                "round",
                &format!("Self(unsafe {{ {prefix}_roundscale_{suffix}::<0x00>(self.0) }})"),
            ));
        } else {
            // SSE4.1/AVX use floor/ceil/round intrinsics directly
            code.push_str(&gen_unary_method(
                "Round toward negative infinity",
                "floor",
                &format!("Self(unsafe {{ {prefix}_floor_{suffix}(self.0) }})"),
            ));
            code.push_str(&gen_unary_method(
                "Round toward positive infinity",
                "ceil",
                &format!("Self(unsafe {{ {prefix}_ceil_{suffix}(self.0) }})"),
            ));
            code.push_str(&gen_unary_method(
                "Round to nearest integer",
                "round",
                &format!("Self(unsafe {{ {prefix}_round_{suffix}::<{{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }}>(self.0) }})"),
            ));
        }

        // FMA (fused multiply-add)
        code.push_str(&indent(
            &formatdoc! {"
                /// Fused multiply-add: self * a + b
                #[inline(always)]
                pub fn mul_add(self, a: Self, b: Self) -> Self {{
                    Self(unsafe {{ {prefix}_fmadd_{suffix}(self.0, a.0, b.0) }})
                }}
            "},
            4,
        ));
        code.push('\n');

        code.push_str(&indent(
            &formatdoc! {"
                /// Fused multiply-sub: self * a - b
                #[inline(always)]
                pub fn mul_sub(self, a: Self, b: Self) -> Self {{
                    Self(unsafe {{ {prefix}_fmsub_{suffix}(self.0, a.0, b.0) }})
                }}
            "},
            4,
        ));
        code.push('\n');
    } else if ty.elem.is_signed() {
        // Abs for signed integers - i64 only available in AVX-512
        let has_abs = ty.width == SimdWidth::W512 || !matches!(ty.elem, ElementType::I64);
        if has_abs {
            code.push_str(&gen_unary_method(
                "Absolute value",
                "abs",
                &format!("Self(unsafe {{ {prefix}_abs_{suffix}(self.0) }})"),
            ));
        }
    }

    code
}

pub fn generate_approx_ops(ty: &SimdType) -> String {
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

pub fn generate_bitwise_unary_ops(ty: &SimdType) -> String {
    let mut code = String::new();
    let prefix = ty.width.x86_prefix();
    let bits = ty.width.bits();

    code.push_str(&indent(
        "// ========== Bitwise Unary Operations ==========\n",
        4,
    ));
    code.push('\n');

    // Build the not body based on element type
    let not_body = if ty.elem.is_float() {
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

        formatdoc! {"
            Self(unsafe {{
                let ones = {prefix}_set1_{set1_suffix}(-1);
                let as_int = {prefix}_cast{cast_from}_si{bits}(self.0);
                {prefix}_castsi{bits}_{cast_to}({prefix}_xor_si{bits}(as_int, ones))
            }})"
        }
    } else {
        let set1_suffix = match (ty.elem, ty.width) {
            (ElementType::I8 | ElementType::U8, _) => "epi8",
            (ElementType::I16 | ElementType::U16, _) => "epi16",
            (ElementType::I32 | ElementType::U32, _) => "epi32",
            (ElementType::I64 | ElementType::U64, SimdWidth::W512) => "epi64",
            (ElementType::I64 | ElementType::U64, _) => "epi64x",
            _ => unreachable!(),
        };
        formatdoc! {"
            Self(unsafe {{
                let ones = {prefix}_set1_{set1_suffix}(-1);
                {prefix}_xor_si{bits}(self.0, ones)
            }})"
        }
    };

    code.push_str(&gen_unary_method(
        "Bitwise NOT (complement): flips all bits.",
        "not",
        &indent_continuation(&not_body, 4),
    ));

    code
}

pub fn generate_shift_ops(ty: &SimdType) -> String {
    // Only for integer types, and NOT for 8-bit (no 8-bit shift intrinsics in x86 SIMD)
    if ty.elem.is_float() || matches!(ty.elem, ElementType::I8 | ElementType::U8) {
        return String::new();
    }

    let mut code = String::new();
    let prefix = ty.width.x86_prefix();
    let suffix = ty.elem.x86_suffix();
    let const_type = if ty.width == SimdWidth::W512 {
        "u32"
    } else {
        "i32"
    };

    code.push_str(&indent("// ========== Shift Operations ==========\n", 4));
    code.push('\n');

    // Shift left logical
    code.push_str(&indent(
        &formatdoc! {"
            /// Shift each lane left by `N` bits.
            ///
            /// Bits shifted out are lost; zeros are shifted in.
            #[inline(always)]
            pub fn shl<const N: {const_type}>(self) -> Self {{
                Self(unsafe {{ {prefix}_slli_{suffix}::<N>(self.0) }})
            }}
        "},
        4,
    ));
    code.push('\n');

    // Shift right logical
    code.push_str(&indent(
        &formatdoc! {"
            /// Shift each lane right by `N` bits (logical/unsigned shift).
            ///
            /// Bits shifted out are lost; zeros are shifted in.
            #[inline(always)]
            pub fn shr<const N: {const_type}>(self) -> Self {{
                Self(unsafe {{ {prefix}_srli_{suffix}::<N>(self.0) }})
            }}
        "},
        4,
    ));
    code.push('\n');

    // Arithmetic shift right (signed types; 64-bit only in AVX-512)
    if ty.elem.is_signed() {
        let has_sra = ty.elem != ElementType::I64 || ty.width == SimdWidth::W512;
        if has_sra {
            code.push_str(&indent(
                &formatdoc! {"
                    /// Arithmetic shift right by `N` bits (sign-extending).
                    ///
                    /// The sign bit is replicated into the vacated positions.
                    #[inline(always)]
                    pub fn shr_arithmetic<const N: {const_type}>(self) -> Self {{
                        Self(unsafe {{ {prefix}_srai_{suffix}::<N>(self.0) }})
                    }}
                "},
                4,
            ));
            code.push('\n');
        }
    }

    code
}
