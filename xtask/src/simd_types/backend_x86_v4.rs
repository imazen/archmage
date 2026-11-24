//! Unified x86-64 V4 (native AVX-512) backend implementation generation (`impls/x86_v4.rs`).
//!
//! Generates native 512-bit backend impls for all 10 W512 SIMD types (`X64V4Token`),
//! VPOPCNTDQ/BITALG popcnt extension impls (`X64V4xToken`), and `F32x16Convert`.
//! (`X64V4xToken` base W512 backends delegate via `x86_v4_w512_delegated.rs`.)

use indoc::formatdoc;

use super::types::{ElementType, SimdType};

/// Generate the entire `magetypes/src/simd/impls/x86_v4.rs` file.
pub(super) fn generate_x86_v4_file(w512_types: &[SimdType]) -> String {
    let safety = super::backend_gen::impls_safety_contract("X64V4Token / X64V4xToken");
    let mut code = formatdoc! {r#"
        //! Backend implementations for X64V4Token and X64V4xToken (native AVX-512).
        //!
        //! Implements the W512 backend traits using native 512-bit AVX-512 intrinsics
        //! for X64V4Token (base AVX-512), plus extension trait impls (popcnt) for
        //! X64V4xToken (+ VPOPCNTDQ, BITALG).
        //!
        //! X64V4xToken's base W512 backends forward to X64V4Token via `self.v4()` in
        //! `x86_v4_w512_delegated.rs`. W128 and W256 types delegate to X64V3Token.
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.
        //!
        {safety}
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        use core::arch::x86_64::*;

        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        use archmage::{{attune, X64V4Token, X64V4xToken}};

        use crate::simd::backends::*;

    "#};

    // Base backend impls for X64V4Token
    code.push_str(
        "// ============================================================================\n",
    );
    code.push_str("// X64V4Token — base AVX-512 (F/BW/CD/DQ/VL)\n");
    code.push_str(
        "// ============================================================================\n\n",
    );
    for &ty in w512_types {
        if ty.is_float() {
            code.push_str(&generate_x86_v4_float_impl(ty));
        } else {
            code.push_str(&generate_x86_v4_int_impl(ty));
        }
        code.push('\n');
    }

    // Extension impls: popcnt (X64V4xToken only)
    code.push_str(
        "\n// ============================================================================\n",
    );
    code.push_str("// X64V4xToken — extension: popcnt (VPOPCNTDQ + BITALG)\n");
    code.push_str(
        "// ============================================================================\n\n",
    );
    for &ty in w512_types {
        if !ty.is_float() {
            code.push_str(&generate_popcnt_impl(ty));
            code.push('\n');
        }
    }

    // F32x16Convert impl for V4 (V4x delegates in `x86_v4_w512_delegated.rs`)
    code.push_str(
        "\n// ============================================================================\n",
    );
    code.push_str("// F32x16Convert — native AVX-512 conversions\n");
    code.push_str(
        "// ============================================================================\n\n",
    );
    code.push_str(&generate_x86_v4_f32x16_convert());

    code
}

fn epi_suffix(elem_bits: usize) -> &'static str {
    match elem_bits {
        8 => "epi8",
        16 => "epi16",
        32 => "epi32",
        64 => "epi64",
        _ => unreachable!(),
    }
}

fn minmax_suffix(ty: SimdType) -> &'static str {
    match (ty.is_signed(), ty.elem_bits()) {
        (true, 8) => "epi8",
        (false, 8) => "epu8",
        (true, 16) => "epi16",
        (false, 16) => "epu16",
        (true, 32) => "epi32",
        (false, 32) => "epu32",
        (true, 64) => "epi64",
        (false, 64) => "epu64",
        _ => unreachable!(),
    }
}

// ============================================================================
// Float Implementations (f32x16, f64x8)
// ============================================================================

fn generate_x86_v4_float_impl(ty: SimdType) -> String {
    let arcane = super::backend_syntax::arcane("X64V4Token");
    let trait_name = ty.trait_name();
    let inner = ty.width.x86_float_type(ty.elem);
    let is_f32 = ty.elem == ElementType::F32;
    let s = if is_f32 { "ps" } else { "pd" };
    let elem = ty.elem.name();
    let float_layout = if is_f32 {
        super::float_layout::x86(16, inner, &arcane)
    } else {
        String::new()
    };
    let array = ty.array_type();
    let epi = if is_f32 { "epi32" } else { "epi64" };
    let abs_mask = if is_f32 {
        "0x7FFF_FFFFu32 as i32"
    } else {
        "0x7FFF_FFFF_FFFF_FFFFu64 as i64"
    };

    let numeric_min = if is_f32 {
        "_mm512_range_ps::<0x04>(a, b)".to_string()
    } else {
        format!(
            "_mm512_mask_blend_{s}(_mm512_cmp_{s}_mask::<_CMP_UNORD_Q>(b, b), _mm512_min_{s}(a, b), a)"
        )
    };
    let numeric_max = if is_f32 {
        "_mm512_range_ps::<0x05>(a, b)".to_string()
    } else {
        format!(
            "_mm512_mask_blend_{s}(_mm512_cmp_{s}_mask::<_CMP_UNORD_Q>(b, b), _mm512_max_{s}(a, b), a)"
        )
    };

    let working_recip_512 = if is_f32 {
        formatdoc! {r#"
            {arcane}
            fn recip(self, a: {inner}) -> {inner} {{
                // token nibbles: QNAN->2 SNAN->2 ZERO->6 ONE->0 -INF->7 +INF->8 NEG->0 POS->0
                const RECIP_FIXUP_TBL: i32 = 0x0087_0622u32 as i32;
                {{
                    let r = _mm512_rcp14_{s}(a);
                    let e = _mm512_fnmadd_{s}(a, r, _mm512_set1_{s}(1.0));
                    let refined = _mm512_fmadd_{s}(r, e, r);
                    _mm512_fixupimm_{s}::<0>(refined, a, _mm512_set1_epi32(RECIP_FIXUP_TBL))
                }}
            }}

            {arcane}
            fn rsqrt(self, a: {inner}) -> {inner} {{
                // token nibbles: QNAN->2 SNAN->2 ZERO->6 ONE->0 -INF->3 +INF->8 NEG->3 POS->0
                const RSQRT_FIXUP_TBL: i32 = 0x0383_0622u32 as i32;
                {{
                    let y = _mm512_rsqrt14_{s}(a);
                    let t = _mm512_mul_{s}(a, _mm512_mul_{s}(y, y));
                    let refined =
                        _mm512_mul_{s}(y, _mm512_fnmadd_{s}(_mm512_set1_{s}(0.5), t, _mm512_set1_{s}(1.5)));
                    _mm512_fixupimm_{s}::<0>(refined, a, _mm512_set1_epi32(RSQRT_FIXUP_TBL))
                }}
            }}
        "#}
    } else {
        formatdoc! {r#"
            {arcane}
            fn recip(self, a: {inner}) -> {inner} {{
                _mm512_div_{s}(_mm512_set1_{s}(1.0), a)
            }}

            {arcane}
            fn rsqrt(self, a: {inner}) -> {inner} {{
                _mm512_div_{s}(_mm512_set1_{s}(1.0), _mm512_sqrt_{s}(a))
            }}
        "#}
    };

    let internal_impl = if is_f32 {
        formatdoc! {r#"

            #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
            impl crate::simd::backends::F32x16Internal for archmage::X64V4Token {{
                type InternalRepr = {inner};
            {float_layout}
            }}"#}
    } else {
        String::new()
    };

    formatdoc! {r#"
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        impl {trait_name} for archmage::X64V4Token {{
            type Repr = {inner};

            {arcane}
            fn splat(self, v: {elem}) -> {inner} {{
                _mm512_set1_{s}(v)
            }}

            {arcane}
            fn zero(self) -> {inner} {{
                _mm512_setzero_{s}()
            }}

            #[inline(always)]
            fn load(self, data: &{array}) -> {inner} {{
                crate::simd_storage::copy(data)
            }}

            #[inline(always)]
            fn from_array(self, arr: {array}) -> {inner} {{
                crate::simd_storage::cast(arr)
            }}

            #[inline(always)]
            fn store(self, repr: {inner}, out: &mut {array}) {{
                crate::simd_storage::store(repr, out);
            }}

            #[inline(always)]
            fn to_array(self, repr: {inner}) -> {array} {{
                crate::simd_storage::cast(repr)
            }}

            {arcane}
            fn add(self, a: {inner}, b: {inner}) -> {inner} {{
                _mm512_add_{s}(a, b)
            }}

            {arcane}
            fn sub(self, a: {inner}, b: {inner}) -> {inner} {{
                _mm512_sub_{s}(a, b)
            }}

            {arcane}
            fn mul(self, a: {inner}, b: {inner}) -> {inner} {{
                _mm512_mul_{s}(a, b)
            }}

            {arcane}
            fn div(self, a: {inner}, b: {inner}) -> {inner} {{
                _mm512_div_{s}(a, b)
            }}

            {arcane}
            fn neg(self, a: {inner}) -> {inner} {{
                _mm512_xor_{s}(a, _mm512_set1_{s}(-0.0))
            }}

            {arcane}
            fn min(self, a: {inner}, b: {inner}) -> {inner} {{
                {numeric_min}
            }}

            {arcane}
            fn max(self, a: {inner}, b: {inner}) -> {inner} {{
                {numeric_max}
            }}

            {arcane}
            fn minimum(self, a: {inner}, b: {inner}) -> {inner} {{
                _mm512_mask_blend_{s}(
                    _mm512_cmp_{s}_mask::<_CMP_UNORD_Q>(a, b),
                    _mm512_or_{s}(_mm512_min_{s}(a, b), _mm512_min_{s}(b, a)),
                    _mm512_set1_{s}({elem}::NAN),
                )
            }}

            {arcane}
            fn maximum(self, a: {inner}, b: {inner}) -> {inner} {{
                _mm512_mask_blend_{s}(
                    _mm512_cmp_{s}_mask::<_CMP_UNORD_Q>(a, b),
                    _mm512_and_{s}(_mm512_max_{s}(a, b), _mm512_max_{s}(b, a)),
                    _mm512_set1_{s}({elem}::NAN),
                )
            }}

            {arcane}
            fn min_fast(self, a: {inner}, b: {inner}) -> {inner} {{
                _mm512_min_{s}(a, b)
            }}

            {arcane}
            fn max_fast(self, a: {inner}, b: {inner}) -> {inner} {{
                _mm512_max_{s}(a, b)
            }}

            {arcane}
            fn sqrt(self, a: {inner}) -> {inner} {{
                _mm512_sqrt_{s}(a)
            }}

            {arcane}
            fn abs(self, a: {inner}) -> {inner} {{
                let mask = _mm512_castsi512_{s}(_mm512_set1_{epi}({abs_mask}));
                _mm512_and_{s}(a, mask)
            }}

            {arcane}
            fn trunc(self, a: {inner}) -> {inner} {{ _mm512_roundscale_{s}::<0x03>(a) }}

            {arcane}
            fn mul_neg_add_relaxed(self, a: {inner}, b: {inner}, c: {inner}) -> {inner} {{ _mm512_fnmadd_{s}(a, b, c) }}

            {arcane}
            fn mul_neg_add_fused(self, a: {inner}, b: {inner}, c: {inner}) -> {inner} {{ _mm512_fnmadd_{s}(a, b, c) }}

            {arcane}
            fn floor(self, a: {inner}) -> {inner} {{
                _mm512_roundscale_{s}::<0x01>(a)
            }}

            {arcane}
            fn ceil(self, a: {inner}) -> {inner} {{
                _mm512_roundscale_{s}::<0x02>(a)
            }}

            {arcane}
            fn round(self, a: {inner}) -> {inner} {{
                _mm512_roundscale_{s}::<0x00>(a)
            }}

            {arcane}
            fn mul_add(self, a: {inner}, b: {inner}, c: {inner}) -> {inner} {{
                _mm512_fmadd_{s}(a, b, c)
            }}

            {arcane}
            fn mul_sub(self, a: {inner}, b: {inner}, c: {inner}) -> {inner} {{
                _mm512_fmsub_{s}(a, b, c)
            }}

            {arcane}
            fn mul_add_portable(self, a: {inner}, b: {inner}, c: {inner}) -> {inner} {{
                _mm512_fmadd_{s}(a, b, c)
            }}

            {arcane}
            fn mul_sub_portable(self, a: {inner}, b: {inner}, c: {inner}) -> {inner} {{
                _mm512_fmsub_{s}(a, b, c)
            }}

            {arcane}
            fn simd_eq(self, a: {inner}, b: {inner}) -> {inner} {{
                let mask = _mm512_cmp_{s}_mask::<_CMP_EQ_OQ>(a, b);
                _mm512_castsi512_{s}(_mm512_maskz_set1_{epi}(mask, -1))
            }}

            {arcane}
            fn simd_ne(self, a: {inner}, b: {inner}) -> {inner} {{
                let mask = _mm512_cmp_{s}_mask::<_CMP_NEQ_UQ>(a, b);
                _mm512_castsi512_{s}(_mm512_maskz_set1_{epi}(mask, -1))
            }}

            {arcane}
            fn simd_lt(self, a: {inner}, b: {inner}) -> {inner} {{
                let mask = _mm512_cmp_{s}_mask::<_CMP_LT_OQ>(a, b);
                _mm512_castsi512_{s}(_mm512_maskz_set1_{epi}(mask, -1))
            }}

            {arcane}
            fn simd_le(self, a: {inner}, b: {inner}) -> {inner} {{
                let mask = _mm512_cmp_{s}_mask::<_CMP_LE_OQ>(a, b);
                _mm512_castsi512_{s}(_mm512_maskz_set1_{epi}(mask, -1))
            }}

            {arcane}
            fn simd_gt(self, a: {inner}, b: {inner}) -> {inner} {{
                let mask = _mm512_cmp_{s}_mask::<_CMP_GT_OQ>(a, b);
                _mm512_castsi512_{s}(_mm512_maskz_set1_{epi}(mask, -1))
            }}

            {arcane}
            fn simd_ge(self, a: {inner}, b: {inner}) -> {inner} {{
                let mask = _mm512_cmp_{s}_mask::<_CMP_GE_OQ>(a, b);
                _mm512_castsi512_{s}(_mm512_maskz_set1_{epi}(mask, -1))
            }}

            {arcane}
            fn blend(self, mask: {inner}, if_true: {inner}, if_false: {inner}) -> {inner} {{
                let mask_i = _mm512_cast{s}_si512(mask);
                let k = _mm512_cmpneq_{epi}_mask(mask_i, _mm512_setzero_si512());
                _mm512_mask_blend_{s}(k, if_false, if_true)
            }}

            {arcane}
            fn reduce_add(self, a: {inner}) -> {elem} {{
                _mm512_reduce_add_{s}(a)
            }}

            {arcane}
            fn reduce_min(self, a: {inner}) -> {elem} {{
                _mm512_reduce_min_{s}(a)
            }}

            {arcane}
            fn reduce_max(self, a: {inner}) -> {elem} {{
                _mm512_reduce_max_{s}(a)
            }}

            // Raw 14-bit hardware estimate — the cheapest >=12-bit path on AVX-512.
            {arcane}
            fn rcp_approx(self, a: {inner}) -> {inner} {{
                _mm512_rcp14_{s}(a)
            }}

            {arcane}
            fn rsqrt_approx(self, a: {inner}) -> {inner} {{
                _mm512_rsqrt14_{s}(a)
            }}

            {working_recip_512}

            {arcane}
            fn not(self, a: {inner}) -> {inner} {{
                let all_ones = _mm512_castsi512_{s}(_mm512_set1_{epi}(-1));
                _mm512_xor_{s}(a, all_ones)
            }}

            {arcane}
            fn bitand(self, a: {inner}, b: {inner}) -> {inner} {{
                _mm512_and_{s}(a, b)
            }}

            {arcane}
            fn bitor(self, a: {inner}, b: {inner}) -> {inner} {{
                _mm512_or_{s}(a, b)
            }}

            {arcane}
            fn bitxor(self, a: {inner}, b: {inner}) -> {inner} {{
                _mm512_xor_{s}(a, b)
            }}
        }}{internal_impl}
    "#}
}

// ============================================================================
// Integer Implementations (all 8 W512 integer types)
// ============================================================================

fn w512_new_ops_v4(ty: SimdType) -> String {
    let arcane = super::backend_syntax::arcane("X64V4Token");
    let eb = ty.elem_bits();

    let shifts = if matches!(eb, 16 | 32) {
        let arith = if ty.is_signed() { "sra" } else { "srl" };
        formatdoc! {r#"
            {arcane}
            fn shl_uniform(self, a: __m512i, count: u32) -> __m512i {{
                _mm512_sll_epi{eb}(a, _mm_cvtsi32_si128(count as i32))
            }}

            {arcane}
            fn shr_logical_uniform(self, a: __m512i, count: u32) -> __m512i {{
                _mm512_srl_epi{eb}(a, _mm_cvtsi32_si128(count as i32))
            }}

            {arcane}
            fn shr_arithmetic_uniform(self, a: __m512i, count: u32) -> __m512i {{
                _mm512_{arith}_epi{eb}(a, _mm_cvtsi32_si128(count as i32))
            }}
        "#}
    } else {
        String::new()
    };

    if eb <= 16 {
        let sat_suffix = minmax_suffix(ty);
        formatdoc! {r#"
            {shifts}
            {arcane}
            fn saturating_add(self, a: __m512i, b: __m512i) -> __m512i {{
                _mm512_adds_{sat_suffix}(a, b)
            }}

            {arcane}
            fn saturating_sub(self, a: __m512i, b: __m512i) -> __m512i {{
                _mm512_subs_{sat_suffix}(a, b)
            }}
        "#}
    } else {
        shifts
    }
}

fn generate_x86_v4_int_impl(ty: SimdType) -> String {
    let integer_methods = super::backend_gen_widen_narrow::methods(&ty.name(), "X64V4Token");
    let arcane = super::backend_syntax::arcane("X64V4Token");
    let trait_name = ty.trait_name();
    let elem = ty.elem.name();
    let lanes = ty.lanes();
    let elem_bits = ty.elem_bits();
    let signed = ty.is_signed();
    let array = ty.array_type();
    let epi = epi_suffix(elem_bits);
    let mm = minmax_suffix(ty);
    let splat_arg = if signed { "v" } else { "v as _" };
    let set1_neg1 = match ty.elem {
        ElementType::U8 => "-1i8",
        ElementType::U16 => "-1i16",
        ElementType::U32 => "-1i32",
        ElementType::U64 => "-1i64",
        _ => "-1",
    };

    let full_mask = match lanes {
        64 => "0xFFFF_FFFF_FFFF_FFFFu64",
        32 => "0xFFFF_FFFFu64",
        16 => "0xFFFFu64",
        8 => "0xFFu64",
        _ => unreachable!(),
    };

    let mul_impl = if matches!(elem_bits, 16 | 32) {
        formatdoc! {r#"

            {arcane}
            fn mul(self, a: __m512i, b: __m512i) -> __m512i {{
                _mm512_mullo_{epi}(a, b)
            }}
        "#}
    } else {
        String::new()
    };

    let abs_impl = if signed {
        formatdoc! {r#"

            {arcane}
            fn abs(self, a: __m512i) -> __m512i {{
                _mm512_abs_{epi}(a)
            }}
        "#}
    } else {
        String::new()
    };

    let new_ops = w512_new_ops_v4(ty);
    let shift_impls = if elem_bits == 8 {
        generate_8bit_shift_polyfill(signed)
    } else {
        let shr_arith_op = if signed { "sra" } else { "srl" };
        let shl = if elem_bits == 32 {
            let cases = (0..32)
                .map(|shift| format!("    {shift} => _mm512_slli_epi32::<{shift}>(a),"))
                .collect::<Vec<_>>()
                .join("\n");
            formatdoc! {r#"
                match N {{
                {cases}
                    _ => unreachable!("shift count checked by the public API"),
                }}
            "#}
        } else {
            format!("_mm512_sll_{epi}(a, _mm_cvtsi32_si128(N))")
        };
        formatdoc! {r#"
            {arcane}
            fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {{
                {shl}
            }}

            {arcane}
            fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {{
                _mm512_{shr_arith_op}_{epi}(a, _mm_cvtsi32_si128(N))
            }}

            {arcane}
            fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {{
                _mm512_srl_{epi}(a, _mm_cvtsi32_si128(N))
            }}
        "#}
    };

    let mask_as_u64 = if elem_bits == 8 { "" } else { " as u64" };

    formatdoc! {r#"
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        impl {trait_name} for archmage::X64V4Token {{
            type Repr = __m512i;

            {arcane}
            fn splat(self, v: {elem}) -> __m512i {{
                _mm512_set1_{epi}({splat_arg})
            }}

            {arcane}
            fn zero(self) -> __m512i {{
                _mm512_setzero_si512()
            }}

            #[inline(always)]
            fn load(self, data: &{array}) -> __m512i {{
                crate::simd_storage::copy(data)
            }}

            #[inline(always)]
            fn from_array(self, arr: {array}) -> __m512i {{
                crate::simd_storage::cast(arr)
            }}

            #[inline(always)]
            fn store(self, repr: __m512i, out: &mut {array}) {{
                crate::simd_storage::store(repr, out);
            }}

            #[inline(always)]
            fn to_array(self, repr: __m512i) -> {array} {{
                crate::simd_storage::cast(repr)
            }}

            {arcane}
            fn add(self, a: __m512i, b: __m512i) -> __m512i {{
                _mm512_add_{epi}(a, b)
            }}

            {arcane}
            fn sub(self, a: __m512i, b: __m512i) -> __m512i {{
                _mm512_sub_{epi}(a, b)
            }}
            {mul_impl}
            {arcane}
            fn neg(self, a: __m512i) -> __m512i {{
                _mm512_sub_{epi}(_mm512_setzero_si512(), a)
            }}

            {arcane}
            fn min(self, a: __m512i, b: __m512i) -> __m512i {{
                _mm512_min_{mm}(a, b)
            }}

            {arcane}
            fn max(self, a: __m512i, b: __m512i) -> __m512i {{
                _mm512_max_{mm}(a, b)
            }}
            {abs_impl}
            {arcane}
            fn simd_eq(self, a: __m512i, b: __m512i) -> __m512i {{
                let mask = _mm512_cmpeq_{mm}_mask(a, b);
                _mm512_maskz_set1_{epi}(mask, {set1_neg1})
            }}

            {arcane}
            fn simd_ne(self, a: __m512i, b: __m512i) -> __m512i {{
                let mask = _mm512_cmpneq_{mm}_mask(a, b);
                _mm512_maskz_set1_{epi}(mask, {set1_neg1})
            }}

            {arcane}
            fn simd_lt(self, a: __m512i, b: __m512i) -> __m512i {{
                let mask = _mm512_cmplt_{mm}_mask(a, b);
                _mm512_maskz_set1_{epi}(mask, {set1_neg1})
            }}

            {arcane}
            fn simd_le(self, a: __m512i, b: __m512i) -> __m512i {{
                let mask = _mm512_cmple_{mm}_mask(a, b);
                _mm512_maskz_set1_{epi}(mask, {set1_neg1})
            }}

            {arcane}
            fn simd_gt(self, a: __m512i, b: __m512i) -> __m512i {{
                // GT = LT with swapped args
                let mask = _mm512_cmplt_{mm}_mask(b, a);
                _mm512_maskz_set1_{epi}(mask, {set1_neg1})
            }}

            {arcane}
            fn simd_ge(self, a: __m512i, b: __m512i) -> __m512i {{
                // GE = LE with swapped args
                let mask = _mm512_cmple_{mm}_mask(b, a);
                _mm512_maskz_set1_{epi}(mask, {set1_neg1})
            }}

            {arcane}
            fn blend(self, mask: __m512i, if_true: __m512i, if_false: __m512i) -> __m512i {{
                let k = _mm512_cmpneq_{epi}_mask(mask, _mm512_setzero_si512());
                _mm512_mask_blend_{epi}(k, if_false, if_true)
            }}

            {arcane}
            fn reduce_add(self, a: __m512i) -> {elem} {{
                // AVX-512 has no single-instruction integer reduction; fold the lanes
                let arr: {array} = crate::simd_storage::cast(a);
                arr.iter().copied().fold(0{elem}, {elem}::wrapping_add)
            }}

            {arcane}
            fn not(self, a: __m512i) -> __m512i {{
                _mm512_xor_si512(a, _mm512_set1_{epi}({set1_neg1}))
            }}

            {arcane}
            fn bitand(self, a: __m512i, b: __m512i) -> __m512i {{
                _mm512_and_si512(a, b)
            }}

            {arcane}
            fn bitor(self, a: __m512i, b: __m512i) -> __m512i {{
                _mm512_or_si512(a, b)
            }}

            {arcane}
            fn bitxor(self, a: __m512i, b: __m512i) -> __m512i {{
                _mm512_xor_si512(a, b)
            }}

            {shift_impls}
        {new_ops}
            {arcane}
            fn all_true(self, a: __m512i) -> bool {{
                // Sign-bit contract: `movepi*_mask` reads the sign bits
                // directly. The `cmpneq` form this replaces answered a
                // different question (lane nonzero) and disagreed with the
                // V3 movemask backends.
                _mm512_movepi{elem_bits}_mask(a){mask_as_u64} == {full_mask}
            }}

            {arcane}
            fn any_true(self, a: __m512i) -> bool {{
                _mm512_movepi{elem_bits}_mask(a){mask_as_u64} != 0
            }}

            {arcane}
            fn bitmask(self, a: __m512i) -> u64 {{
                _mm512_movepi{elem_bits}_mask(a){mask_as_u64}
            }}
        {integer_methods}
        }}
    "#}
}

/// Generate 8-bit shift polyfill via 16-bit operations for AVX-512.
fn generate_8bit_shift_polyfill(signed: bool) -> String {
    let arcane = super::backend_syntax::arcane("X64V4Token");
    let shr_arith = if !signed {
        formatdoc! {r#"
            {arcane}
            fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {{
                let count = _mm_cvtsi32_si128(N);
                let shifted = _mm512_srl_epi16(a, count);
                let byte_mask = _mm512_set1_epi8(((0xFFu16 >> N) & 0xFF) as i8);
                let logical = _mm512_and_si512(shifted, byte_mask);
                // Sign-extend: fill the high N bits of negative lanes. The u16
                // shift keeps the fill mask 0x00 at N == 0 (identity shift).
                let sign = _mm512_movm_epi8(_mm512_cmplt_epi8_mask(a, _mm512_setzero_si512()));
                let fill = _mm512_set1_epi8(((0xFF00u16 >> N) & 0xFF) as i8);
                _mm512_or_si512(logical, _mm512_and_si512(sign, fill))
            }}
        "#}
    } else {
        formatdoc! {r#"
            {arcane}
            fn shr_arithmetic_const<const N: i32>(self, a: __m512i) -> __m512i {{
                let count = _mm_cvtsi32_si128(N);
                // Sign-extend bytes to 16-bit, shift, mask back to 8-bit
                let lo = _mm512_sra_epi16(_mm512_slli_epi16::<8>(a), count);
                let hi = _mm512_sra_epi16(a, count);
                // Combine: take low byte from lo, high byte from hi
                let mask = _mm512_set1_epi16(0x00FFu16 as i16);
                _mm512_or_si512(
                    _mm512_and_si512(_mm512_srli_epi16::<8>(lo), mask),
                    _mm512_andnot_si512(mask, hi)
                )
            }}
        "#}
    };

    formatdoc! {r#"
            {arcane}
            fn shl_const<const N: i32>(self, a: __m512i) -> __m512i {{
                let count = _mm_cvtsi32_si128(N);
                let shifted = _mm512_sll_epi16(a, count);
                let mask = _mm512_set1_epi8(((0xFFu16 << N) & 0xFF) as i8);
                _mm512_and_si512(shifted, mask)
            }}

            {shr_arith}
            {arcane}
            fn shr_logical_const<const N: i32>(self, a: __m512i) -> __m512i {{
                let count = _mm_cvtsi32_si128(N);
                let shifted = _mm512_srl_epi16(a, count);
                let mask = _mm512_set1_epi8(((0xFFu16 >> N) & 0xFF) as i8);
                _mm512_and_si512(shifted, mask)
            }}
    "#}
}

// ============================================================================
// Extension & Conversion Impls
// ============================================================================

fn generate_popcnt_impl(ty: SimdType) -> String {
    let arcane = super::backend_syntax::arcane("X64V4xToken");
    let name = ty.name();
    let epi = epi_suffix(ty.elem_bits());

    formatdoc! {r#"
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        impl {name}PopcntBackend for archmage::X64V4xToken {{
            {arcane}
            fn popcnt(self, a: __m512i) -> __m512i {{
                _mm512_popcnt_{epi}(a)
            }}
        }}
    "#}
}

fn generate_x86_v4_f32x16_convert() -> String {
    let arcane = super::backend_syntax::arcane("X64V4Token");
    formatdoc! {r#"
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "w512"))]
        impl F32x16Convert for archmage::X64V4Token {{
            {arcane}
            fn bitcast_f32_to_i32(self, a: __m512) -> __m512i {{
                _mm512_castps_si512(a)
            }}

            {arcane}
            fn bitcast_i32_to_f32(self, a: __m512i) -> __m512 {{
                _mm512_castsi512_ps(a)
            }}

            {arcane}
            fn convert_f32_to_i32(self, a: __m512) -> __m512i {{
                _mm512_cvttps_epi32(a)
            }}

            // Issue #80 fixup, mask-register form.
            {arcane}
            fn convert_f32_to_i32_saturating(self, a: __m512) -> __m512i {{
                let t = _mm512_cvttps_epi32(a);
                let big = _mm512_cmp_ps_mask::<_CMP_GE_OQ>(a, _mm512_set1_ps(2_147_483_648.0));
                let t = _mm512_mask_mov_epi32(t, big, _mm512_set1_epi32(i32::MAX));
                let not_nan = _mm512_cmp_ps_mask::<_CMP_ORD_Q>(a, a);
                _mm512_maskz_mov_epi32(not_nan, t)
            }}

            {arcane}
            fn convert_f32_to_i32_round(self, a: __m512) -> __m512i {{
                _mm512_cvtps_epi32(a)
            }}

            {arcane}
            fn convert_i32_to_f32(self, a: __m512i) -> __m512 {{
                _mm512_cvtepi32_ps(a)
            }}
        }}

    "#}
}
