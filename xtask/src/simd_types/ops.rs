//! Standard SIMD operations (horizontal, conversion, math, etc.)

pub use super::ops_comparison::{
    generate_blend_ops, generate_boolean_reductions, generate_comparison_ops,
};

use super::types::{ElementType, SimdType, SimdWidth, gen_binary_method, gen_unary_method};
use indoc::formatdoc;

pub fn generate_horizontal_ops(ty: &SimdType) -> String {
    let elem = ty.elem.name();
    let lanes = ty.lanes();

    let mut code = String::from("// ========== Horizontal Operations ==========\n\n");

    if ty.elem.is_float() {
        let reduce_add_body = match (ty.elem, ty.width) {
            (ElementType::F32, SimdWidth::W128) => formatdoc! {"
                unsafe {{
                let h1 = _mm_hadd_ps(self.0, self.0);
                let h2 = _mm_hadd_ps(h1, h1);
                _mm_cvtss_f32(h2)
                }}"
            },
            (ElementType::F64, SimdWidth::W128) => formatdoc! {"
                unsafe {{
                let h = _mm_hadd_pd(self.0, self.0);
                _mm_cvtsd_f64(h)
                }}"
            },
            (ElementType::F32, SimdWidth::W256) => formatdoc! {"
                unsafe {{
                let hi = _mm256_extractf128_ps::<1>(self.0);
                let lo = _mm256_castps256_ps128(self.0);
                let sum = _mm_add_ps(lo, hi);
                let h1 = _mm_hadd_ps(sum, sum);
                let h2 = _mm_hadd_ps(h1, h1);
                _mm_cvtss_f32(h2)
                }}"
            },
            (ElementType::F64, SimdWidth::W256) => formatdoc! {"
                unsafe {{
                let hi = _mm256_extractf128_pd::<1>(self.0);
                let lo = _mm256_castpd256_pd128(self.0);
                let sum = _mm_add_pd(lo, hi);
                let h = _mm_hadd_pd(sum, sum);
                _mm_cvtsd_f64(h)
                }}"
            },
            (ElementType::F32, SimdWidth::W512) => "unsafe { _mm512_reduce_add_ps(self.0) }".into(),
            (ElementType::F64, SimdWidth::W512) => "unsafe { _mm512_reduce_add_pd(self.0) }".into(),
            _ => unreachable!(),
        };

        code.push_str(&formatdoc! {"
            /// Sum all lanes horizontally.
            ///
            /// Returns a scalar containing the sum of all {lanes} lanes.
            #[inline(always)]
            pub fn reduce_add(self) -> {elem} {{
            {reduce_add_body}
            }}

        "});

        // reduce_min
        let reduce_min_body = match (ty.elem, ty.width) {
            (ElementType::F32, SimdWidth::W128) => formatdoc! {"
                unsafe {{
                let shuf = _mm_shuffle_ps::<0b10_11_00_01>(self.0, self.0);
                let m1 = _mm_min_ps(self.0, shuf);
                let shuf2 = _mm_shuffle_ps::<0b00_00_10_10>(m1, m1);
                let m2 = _mm_min_ps(m1, shuf2);
                _mm_cvtss_f32(m2)
                }}"
            },
            (ElementType::F64, SimdWidth::W128) => formatdoc! {"
                unsafe {{
                let shuf = _mm_shuffle_pd::<0b01>(self.0, self.0);
                let m = _mm_min_pd(self.0, shuf);
                _mm_cvtsd_f64(m)
                }}"
            },
            (ElementType::F32, SimdWidth::W256) => formatdoc! {"
                unsafe {{
                let hi = _mm256_extractf128_ps::<1>(self.0);
                let lo = _mm256_castps256_ps128(self.0);
                let m = _mm_min_ps(lo, hi);
                let shuf = _mm_shuffle_ps::<0b10_11_00_01>(m, m);
                let m1 = _mm_min_ps(m, shuf);
                let shuf2 = _mm_shuffle_ps::<0b00_00_10_10>(m1, m1);
                let m2 = _mm_min_ps(m1, shuf2);
                _mm_cvtss_f32(m2)
                }}"
            },
            (ElementType::F64, SimdWidth::W256) => formatdoc! {"
                unsafe {{
                let hi = _mm256_extractf128_pd::<1>(self.0);
                let lo = _mm256_castpd256_pd128(self.0);
                let m = _mm_min_pd(lo, hi);
                let shuf = _mm_shuffle_pd::<0b01>(m, m);
                let m2 = _mm_min_pd(m, shuf);
                _mm_cvtsd_f64(m2)
                }}"
            },
            (ElementType::F32, SimdWidth::W512) => "unsafe { _mm512_reduce_min_ps(self.0) }".into(),
            (ElementType::F64, SimdWidth::W512) => "unsafe { _mm512_reduce_min_pd(self.0) }".into(),
            _ => unreachable!(),
        };

        code.push_str(&formatdoc! {"
            /// Find the minimum value across all lanes.
            #[inline(always)]
            pub fn reduce_min(self) -> {elem} {{
            {reduce_min_body}
            }}

        "});

        // reduce_max
        let reduce_max_body = match (ty.elem, ty.width) {
            (ElementType::F32, SimdWidth::W128) => formatdoc! {"
                unsafe {{
                let shuf = _mm_shuffle_ps::<0b10_11_00_01>(self.0, self.0);
                let m1 = _mm_max_ps(self.0, shuf);
                let shuf2 = _mm_shuffle_ps::<0b00_00_10_10>(m1, m1);
                let m2 = _mm_max_ps(m1, shuf2);
                _mm_cvtss_f32(m2)
                }}"
            },
            (ElementType::F64, SimdWidth::W128) => formatdoc! {"
                unsafe {{
                let shuf = _mm_shuffle_pd::<0b01>(self.0, self.0);
                let m = _mm_max_pd(self.0, shuf);
                _mm_cvtsd_f64(m)
                }}"
            },
            (ElementType::F32, SimdWidth::W256) => formatdoc! {"
                unsafe {{
                let hi = _mm256_extractf128_ps::<1>(self.0);
                let lo = _mm256_castps256_ps128(self.0);
                let m = _mm_max_ps(lo, hi);
                let shuf = _mm_shuffle_ps::<0b10_11_00_01>(m, m);
                let m1 = _mm_max_ps(m, shuf);
                let shuf2 = _mm_shuffle_ps::<0b00_00_10_10>(m1, m1);
                let m2 = _mm_max_ps(m1, shuf2);
                _mm_cvtss_f32(m2)
                }}"
            },
            (ElementType::F64, SimdWidth::W256) => formatdoc! {"
                unsafe {{
                let hi = _mm256_extractf128_pd::<1>(self.0);
                let lo = _mm256_castpd256_pd128(self.0);
                let m = _mm_max_pd(lo, hi);
                let shuf = _mm_shuffle_pd::<0b01>(m, m);
                let m2 = _mm_max_pd(m, shuf);
                _mm_cvtsd_f64(m2)
                }}"
            },
            (ElementType::F32, SimdWidth::W512) => "unsafe { _mm512_reduce_max_ps(self.0) }".into(),
            (ElementType::F64, SimdWidth::W512) => "unsafe { _mm512_reduce_max_pd(self.0) }".into(),
            _ => unreachable!(),
        };

        code.push_str(&formatdoc! {"
            /// Find the maximum value across all lanes.
            #[inline(always)]
            pub fn reduce_max(self) -> {elem} {{
            {reduce_max_body}
            }}

        "});
    } else {
        // Integer horizontal sum
        code.push_str(&formatdoc! {"
            /// Sum all lanes horizontally.
            ///
            /// Returns a scalar containing the sum of all {lanes} lanes.
            /// Note: This uses a scalar loop. For performance-critical code,
            /// consider keeping values in SIMD until the final reduction.
            #[inline(always)]
            pub fn reduce_add(self) -> {elem} {{
            self.as_array().iter().copied().fold(0_{elem}, {elem}::wrapping_add)
            }}

        "});
    }

    code
}

pub fn generate_conversion_ops(ty: &SimdType) -> String {
    let mut code = String::new();
    let prefix = ty.width.x86_prefix();
    let lanes = ty.lanes();

    if ty.elem == ElementType::F32 {
        let int_name = format!("i32x{lanes}");

        code.push_str(&formatdoc! {"
            // ========== Type Conversions ==========

            /// Convert to signed 32-bit integers, rounding toward zero (truncation).
            ///
            /// Values outside the representable range become `i32::MIN` (0x80000000).
            #[inline(always)]
            pub fn to_i32x{lanes}(self) -> {int_name} {{
            {int_name}(unsafe {{ {prefix}_cvttps_epi32(self.0) }})
            }}

            /// Convert to signed 32-bit integers, rounding to nearest even.
            ///
            /// Values outside the representable range become `i32::MIN` (0x80000000).
            #[inline(always)]
            pub fn to_i32x{lanes}_round(self) -> {int_name} {{
            {int_name}(unsafe {{ {prefix}_cvtps_epi32(self.0) }})
            }}

            /// Create from signed 32-bit integers.
            #[inline(always)]
            pub fn from_i32x{lanes}(v: {int_name}) -> Self {{
            Self(unsafe {{ {prefix}_cvtepi32_ps(v.0) }})
            }}

        "});
    } else if ty.elem == ElementType::I32 {
        let float_name = format!("f32x{lanes}");

        code.push_str(&formatdoc! {"
            // ========== Type Conversions ==========

            /// Convert to single-precision floats.
            #[inline(always)]
            pub fn to_f32x{lanes}(self) -> {float_name} {{
            {float_name}(unsafe {{ {prefix}_cvtepi32_ps(self.0) }})
            }}

        "});
    }

    if ty.elem == ElementType::F64 && ty.width == SimdWidth::W128 {
        code.push_str(&formatdoc! {"
            // ========== Type Conversions ==========

            /// Convert to signed 32-bit integers (2 lanes), rounding toward zero.
            ///
            /// Returns an `i32x4` where only the lower 2 lanes are valid.
            #[inline(always)]
            pub fn to_i32x4_low(self) -> i32x4 {{
            i32x4(unsafe {{ _mm_cvttpd_epi32(self.0) }})
            }}

        "});
    }

    code
}

pub fn generate_scalar_ops(ty: &SimdType, cfg_attr: &str) -> String {
    let name = ty.name();
    let elem = ty.elem.name();

    let mut code = formatdoc! {"

        // Scalar broadcast operations for {name}
        // These allow `v + 2.0` instead of `v + {name}::splat(token, 2.0)`

    "};

    if ty.elem.is_float() {
        let splat = scalar_splat_intrinsic(ty);
        code.push_str(&formatdoc! {"
            {cfg_attr}impl Add<{elem}> for {name} {{
            type Output = Self;
            /// Add a scalar to all lanes: `v + 2.0`
            ///
            /// Broadcasts the scalar to all lanes, then adds.
            #[inline(always)]
            fn add(self, rhs: {elem}) -> Self {{
            self + Self(unsafe {{ {splat}(rhs) }})
            }}
            }}

            {cfg_attr}impl Sub<{elem}> for {name} {{
            type Output = Self;
            /// Subtract a scalar from all lanes: `v - 2.0`
            #[inline(always)]
            fn sub(self, rhs: {elem}) -> Self {{
            self - Self(unsafe {{ {splat}(rhs) }})
            }}
            }}

            {cfg_attr}impl Mul<{elem}> for {name} {{
            type Output = Self;
            /// Multiply all lanes by a scalar: `v * 2.0`
            #[inline(always)]
            fn mul(self, rhs: {elem}) -> Self {{
            self * Self(unsafe {{ {splat}(rhs) }})
            }}
            }}

            {cfg_attr}impl Div<{elem}> for {name} {{
            type Output = Self;
            /// Divide all lanes by a scalar: `v / 2.0`
            #[inline(always)]
            fn div(self, rhs: {elem}) -> Self {{
            self / Self(unsafe {{ {splat}(rhs) }})
            }}
            }}

        "});
    } else {
        let splat_call = scalar_splat_intrinsic_int(ty);
        code.push_str(&formatdoc! {"
            {cfg_attr}impl Add<{elem}> for {name} {{
            type Output = Self;
            /// Add a scalar to all lanes.
            #[inline(always)]
            fn add(self, rhs: {elem}) -> Self {{
            self + Self(unsafe {{ {splat_call} }})
            }}
            }}

            {cfg_attr}impl Sub<{elem}> for {name} {{
            type Output = Self;
            /// Subtract a scalar from all lanes.
            #[inline(always)]
            fn sub(self, rhs: {elem}) -> Self {{
            self - Self(unsafe {{ {splat_call} }})
            }}
            }}

        "});
    }

    code
}

fn scalar_splat_intrinsic(ty: &SimdType) -> String {
    let prefix = ty.width.x86_prefix();
    match ty.elem {
        ElementType::F32 => format!("{prefix}_set1_ps"),
        ElementType::F64 => format!("{prefix}_set1_pd"),
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
    format!("{prefix}_set1_{suffix}({cast})")
}

pub fn generate_math_ops(ty: &SimdType) -> String {
    let mut code = String::new();
    let prefix = ty.width.x86_prefix();
    let suffix = ty.elem.x86_suffix();
    let minmax_suffix = ty.elem.x86_minmax_suffix();
    let bits = ty.width.bits();

    // Min/Max - i64/u64 only native in AVX-512, polyfill on SSE/AVX2
    let has_native_minmax = ty.elem.is_float()
        || ty.width == SimdWidth::W512
        || !matches!(ty.elem, ElementType::I64 | ElementType::U64);

    if has_native_minmax {
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
        code.push_str(&formatdoc! {"
            /// Clamp values between lo and hi
            #[inline(always)]
            pub fn clamp(self, lo: Self, hi: Self) -> Self {{
            self.max(lo).min(hi)
            }}

        "});
    } else if ty.elem == ElementType::I64 {
        // i64 polyfill: compare + blend (SSE4.2 has _cmpgt_epi64)
        code.push_str(&formatdoc! {"
            /// Element-wise minimum (polyfill via compare+select)
            #[inline(always)]
            pub fn min(self, other: Self) -> Self {{
            unsafe {{
            let mask = {prefix}_cmpgt_epi64(self.0, other.0);
            Self({prefix}_blendv_epi8(self.0, other.0, mask))
            }}
            }}

            /// Element-wise maximum (polyfill via compare+select)
            #[inline(always)]
            pub fn max(self, other: Self) -> Self {{
            unsafe {{
            let mask = {prefix}_cmpgt_epi64(self.0, other.0);
            Self({prefix}_blendv_epi8(other.0, self.0, mask))
            }}
            }}

            /// Clamp values between lo and hi
            #[inline(always)]
            pub fn clamp(self, lo: Self, hi: Self) -> Self {{
            self.max(lo).min(hi)
            }}

        "});
    } else if ty.elem == ElementType::U64 {
        // u64 polyfill: bias to signed domain, then compare + blend
        let si_suffix = format!("si{bits}");
        code.push_str(&formatdoc! {"
            /// Element-wise minimum (polyfill via unsigned compare+select)
            #[inline(always)]
            pub fn min(self, other: Self) -> Self {{
            unsafe {{
            let bias = {prefix}_set1_epi64x(i64::MIN);
            let a_biased = {prefix}_xor_{si_suffix}(self.0, bias);
            let b_biased = {prefix}_xor_{si_suffix}(other.0, bias);
            let mask = {prefix}_cmpgt_epi64(a_biased, b_biased);
            Self({prefix}_blendv_epi8(self.0, other.0, mask))
            }}
            }}

            /// Element-wise maximum (polyfill via unsigned compare+select)
            #[inline(always)]
            pub fn max(self, other: Self) -> Self {{
            unsafe {{
            let bias = {prefix}_set1_epi64x(i64::MIN);
            let a_biased = {prefix}_xor_{si_suffix}(self.0, bias);
            let b_biased = {prefix}_xor_{si_suffix}(other.0, bias);
            let mask = {prefix}_cmpgt_epi64(a_biased, b_biased);
            Self({prefix}_blendv_epi8(other.0, self.0, mask))
            }}
            }}

            /// Clamp values between lo and hi
            #[inline(always)]
            pub fn clamp(self, lo: Self, hi: Self) -> Self {{
            self.max(lo).min(hi)
            }}

        "});
    }

    if ty.elem.is_float() {
        code.push_str(&gen_unary_method(
            "Square root",
            "sqrt",
            &format!("Self(unsafe {{ {prefix}_sqrt_{suffix}(self.0) }})"),
        ));

        // Abs for floats
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
        code.push_str(&gen_unary_method("Absolute value", "abs", &abs_body));

        // Floor/ceil/round
        if ty.width == SimdWidth::W512 {
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

        // FMA
        code.push_str(&formatdoc! {"
            /// Fused multiply-add: self * a + b
            #[inline(always)]
            pub fn mul_add(self, a: Self, b: Self) -> Self {{
            Self(unsafe {{ {prefix}_fmadd_{suffix}(self.0, a.0, b.0) }})
            }}

            /// Fused multiply-sub: self * a - b
            #[inline(always)]
            pub fn mul_sub(self, a: Self, b: Self) -> Self {{
            Self(unsafe {{ {prefix}_fmsub_{suffix}(self.0, a.0, b.0) }})
            }}

        "});
    } else if ty.elem.is_signed() {
        // Abs for signed integers - i64 only native in AVX-512
        let has_native_abs = ty.width == SimdWidth::W512 || !matches!(ty.elem, ElementType::I64);
        if has_native_abs {
            code.push_str(&gen_unary_method(
                "Absolute value",
                "abs",
                &format!("Self(unsafe {{ {prefix}_abs_{suffix}(self.0) }})"),
            ));
        } else if ty.elem == ElementType::I64 {
            // i64 abs polyfill: conditional negate using sign mask
            let si_suffix = format!("si{bits}");
            code.push_str(&formatdoc! {"
                /// Absolute value (polyfill via conditional negate)
                #[inline(always)]
                pub fn abs(self) -> Self {{
                unsafe {{
                let zero = {prefix}_setzero_{si_suffix}();
                let sign = {prefix}_cmpgt_epi64(zero, self.0);
                Self({prefix}_sub_epi64({prefix}_xor_{si_suffix}(self.0, sign), sign))
                }}
                }}

            "});
        }
    }

    code
}

pub fn generate_approx_ops(ty: &SimdType) -> String {
    let prefix = ty.width.x86_prefix();
    let suffix = ty.elem.x86_suffix();

    // f32 types
    if ty.elem == ElementType::F32 {
        let (rcp_fn, rsqrt_fn, precision) = if ty.width == SimdWidth::W512 {
            ("rcp14", "rsqrt14", "~14-bit")
        } else {
            ("rcp", "rsqrt", "~12-bit")
        };

        let set1 = format!("{prefix}_set1_ps");

        return formatdoc! {"
            // ========== Approximation Operations ==========

            /// Fast reciprocal approximation (1/x) with {precision} precision.
            ///
            /// For full precision, use `recip()` which applies Newton-Raphson refinement.
            #[inline(always)]
            pub fn rcp_approx(self) -> Self {{
            Self(unsafe {{ {prefix}_{rcp_fn}_{suffix}(self.0) }})
            }}

            /// Precise reciprocal (1/x) using Newton-Raphson refinement.
            ///
            /// More accurate than `rcp_approx()` but slower. For maximum speed
            /// with acceptable precision loss, use `rcp_approx()`.
            #[inline(always)]
            pub fn recip(self) -> Self {{
            // Newton-Raphson: x' = x * (2 - a*x)
            let approx = self.rcp_approx();
            let two = Self(unsafe {{ {set1}(2.0) }});
            // One iteration gives ~24-bit precision from ~12-bit
            approx * (two - self * approx)
            }}

            /// Fast reciprocal square root approximation (1/sqrt(x)) with {precision} precision.
            ///
            /// For full precision, use `sqrt().recip()` or apply Newton-Raphson manually.
            #[inline(always)]
            pub fn rsqrt_approx(self) -> Self {{
            Self(unsafe {{ {prefix}_{rsqrt_fn}_{suffix}(self.0) }})
            }}

            /// Precise reciprocal square root (1/sqrt(x)) using Newton-Raphson refinement.
            #[inline(always)]
            pub fn rsqrt(self) -> Self {{
            // Newton-Raphson for rsqrt: y' = 0.5 * y * (3 - x * y * y)
            let approx = self.rsqrt_approx();
            let half = Self(unsafe {{ {set1}(0.5) }});
            let three = Self(unsafe {{ {set1}(3.0) }});
            half * approx * (three - self * approx * approx)
            }}

        "};
    }

    // f64 only has approximation intrinsics in AVX-512
    if ty.elem == ElementType::F64 && ty.width == SimdWidth::W512 {
        return formatdoc! {"
            // ========== Approximation Operations ==========

            /// Fast reciprocal approximation (1/x) with ~14-bit precision.
            #[inline(always)]
            pub fn rcp_approx(self) -> Self {{
            Self(unsafe {{ {prefix}_rcp14_{suffix}(self.0) }})
            }}

            /// Precise reciprocal (1/x) using Newton-Raphson refinement.
            #[inline(always)]
            pub fn recip(self) -> Self {{
            let approx = self.rcp_approx();
            let two = Self(unsafe {{ {prefix}_set1_pd(2.0) }});
            approx * (two - self * approx)
            }}

            /// Fast reciprocal square root approximation (1/sqrt(x)) with ~14-bit precision.
            #[inline(always)]
            pub fn rsqrt_approx(self) -> Self {{
            Self(unsafe {{ {prefix}_rsqrt14_{suffix}(self.0) }})
            }}

            /// Precise reciprocal square root (1/sqrt(x)) using Newton-Raphson refinement.
            #[inline(always)]
            pub fn rsqrt(self) -> Self {{
            let approx = self.rsqrt_approx();
            let half = Self(unsafe {{ {prefix}_set1_pd(0.5) }});
            let three = Self(unsafe {{ {prefix}_set1_pd(3.0) }});
            half * approx * (three - self * approx * approx)
            }}

        "};
    }

    String::new()
}

pub fn generate_bitwise_unary_ops(ty: &SimdType) -> String {
    let prefix = ty.width.x86_prefix();
    let bits = ty.width.bits();

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

    let mut code = String::from("// ========== Bitwise Unary Operations ==========\n\n");
    code.push_str(&gen_unary_method(
        "Bitwise NOT (complement): flips all bits.",
        "not",
        &not_body,
    ));
    code
}

pub fn generate_shift_ops(ty: &SimdType) -> String {
    if ty.elem.is_float() {
        return String::new();
    }

    // 8-bit types need polyfill (no native shift-by-immediate in any x86 ISA)
    if matches!(ty.elem, ElementType::I8 | ElementType::U8) {
        return generate_byte_shift_polyfill(ty);
    }

    let prefix = ty.width.x86_prefix();
    let suffix = ty.elem.x86_suffix();
    let const_type = if ty.width == SimdWidth::W512 {
        "u32"
    } else {
        "i32"
    };

    let mut code = formatdoc! {"
        // ========== Shift Operations ==========

        /// Shift each lane left by `N` bits.
        ///
        /// Bits shifted out are lost; zeros are shifted in.
        #[inline(always)]
        pub fn shl<const N: {const_type}>(self) -> Self {{
        Self(unsafe {{ {prefix}_slli_{suffix}::<N>(self.0) }})
        }}

        /// Shift each lane right by `N` bits (logical/unsigned shift).
        ///
        /// Bits shifted out are lost; zeros are shifted in.
        #[inline(always)]
        pub fn shr<const N: {const_type}>(self) -> Self {{
        Self(unsafe {{ {prefix}_srli_{suffix}::<N>(self.0) }})
        }}

    "};

    // Arithmetic shift right (signed types; 64-bit only in AVX-512)
    if ty.elem.is_signed() {
        let has_sra = ty.elem != ElementType::I64 || ty.width == SimdWidth::W512;
        if has_sra {
            code.push_str(&formatdoc! {"
                /// Arithmetic shift right by `N` bits (sign-extending).
                ///
                /// The sign bit is replicated into the vacated positions.
                #[inline(always)]
                pub fn shr_arithmetic<const N: {const_type}>(self) -> Self {{
                Self(unsafe {{ {prefix}_srai_{suffix}::<N>(self.0) }})
                }}

            "});
        }
    }

    code
}

/// Generate 8-bit shift polyfills using 16-bit shift + mask.
///
/// x86 has no native 8-bit shift-by-immediate at any ISA level.
/// Polyfill: shift 16-bit lanes, then AND to clear cross-byte bits.
fn generate_byte_shift_polyfill(ty: &SimdType) -> String {
    let prefix = ty.width.x86_prefix();
    let bits = ty.width.bits();
    let si_suffix = format!("si{bits}");
    let const_type = if ty.width == SimdWidth::W512 {
        "u32"
    } else {
        "i32"
    };
    // W512 const generic is already u32, avoid clippy unnecessary_cast
    let n_u32 = if ty.width == SimdWidth::W512 {
        "N"
    } else {
        "N as u32"
    };

    let mut code = formatdoc! {"
        // ========== Shift Operations (polyfill) ==========
        // x86 has no native 8-bit shift-by-immediate; uses 16-bit shift + mask.

        /// Shift each byte left by `N` bits.
        ///
        /// Bits shifted out are lost; zeros are shifted in.
        /// Implemented via 16-bit shift + byte mask (no native 8-bit shift in x86).
        #[inline(always)]
        pub fn shl<const N: {const_type}>(self) -> Self {{
        unsafe {{
        let shifted = {prefix}_slli_epi16::<N>(self.0);
        let mask = {prefix}_set1_epi8((0xFFu8.wrapping_shl({n_u32})) as i8);
        Self({prefix}_and_{si_suffix}(shifted, mask))
        }}
        }}

        /// Shift each byte right by `N` bits (logical/unsigned shift).
        ///
        /// Bits shifted out are lost; zeros are shifted in.
        /// Implemented via 16-bit shift + byte mask (no native 8-bit shift in x86).
        #[inline(always)]
        pub fn shr<const N: {const_type}>(self) -> Self {{
        unsafe {{
        let shifted = {prefix}_srli_epi16::<N>(self.0);
        let mask = {prefix}_set1_epi8((0xFFu8.wrapping_shr({n_u32})) as i8);
        Self({prefix}_and_{si_suffix}(shifted, mask))
        }}
        }}

    "};

    // Arithmetic shift right for i8 only
    if ty.elem == ElementType::I8 {
        if ty.width == SimdWidth::W512 {
            // AVX-512: cmpgt returns __mmask, use maskz_mov for sign extension
            code.push_str(&formatdoc! {"
                /// Arithmetic shift right by `N` bits (sign-extending).
                ///
                /// The sign bit is replicated into the vacated positions.
                /// Implemented via logical shift + sign bit fill (no native 8-bit shift in x86).
                #[inline(always)]
                pub fn shr_arithmetic<const N: {const_type}>(self) -> Self {{
                unsafe {{
                let shifted = {prefix}_srli_epi16::<N>(self.0);
                let byte_mask = {prefix}_set1_epi8((0xFFu8.wrapping_shr({n_u32})) as i8);
                let logical = {prefix}_and_{si_suffix}(shifted, byte_mask);
                let zero = {prefix}_setzero_{si_suffix}();
                let neg = {prefix}_cmpgt_epi8_mask(zero, self.0);
                let fill = {prefix}_set1_epi8((0xFFu8.wrapping_shl(8u32.wrapping_sub({n_u32}))) as i8);
                let sign_bits = {prefix}_maskz_mov_epi8(neg, fill);
                Self({prefix}_or_{si_suffix}(logical, sign_bits))
                }}
                }}

            "});
        } else {
            // SSE/AVX2: cmpgt returns vector mask
            code.push_str(&formatdoc! {"
                /// Arithmetic shift right by `N` bits (sign-extending).
                ///
                /// The sign bit is replicated into the vacated positions.
                /// Implemented via logical shift + sign bit fill (no native 8-bit shift in x86).
                #[inline(always)]
                pub fn shr_arithmetic<const N: {const_type}>(self) -> Self {{
                unsafe {{
                let shifted = {prefix}_srli_epi16::<N>(self.0);
                let byte_mask = {prefix}_set1_epi8((0xFFu8.wrapping_shr(N as u32)) as i8);
                let logical = {prefix}_and_{si_suffix}(shifted, byte_mask);
                let zero = {prefix}_setzero_{si_suffix}();
                let sign = {prefix}_cmpgt_epi8(zero, self.0);
                let fill = {prefix}_set1_epi8((0xFFu8.wrapping_shl(8u32.wrapping_sub(N as u32))) as i8);
                Self({prefix}_or_{si_suffix}(logical, {prefix}_and_{si_suffix}(sign, fill)))
                }}
                }}

            "});
        }
    }

    code
}
