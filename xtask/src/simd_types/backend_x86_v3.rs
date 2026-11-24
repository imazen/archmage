//! Unified x86-64 V3 (SSE2..AVX2+FMA) backend implementation generation (`impls/x86_v3.rs`).
//!
//! Generates native 128-bit and 256-bit backend impls for all 20 narrow SIMD types
//! (`f32x4/8`, `f64x2/4`, `i32x4/8`, `u32x4/8`, `i64x2/4`, `i8x16/32`, `u8x16/32`,
//! `i16x8/16`, `u16x8/16`, `u64x2/4`), all V3 float↔int and signed↔unsigned
//! convert/bitcast impls, and the V3 `f32x16 -> u16` pixel-packing helper.

use indoc::formatdoc;

use super::types::{ElementType, SimdType, SimdWidth};

/// Generate the entire `magetypes/src/simd/impls/x86_v3.rs` file.
pub(super) fn generate_x86_v3_file(narrow_types: &[SimdType]) -> String {
    let sse2_boundary = super::backend_syntax::SSE2_BOUNDARY;
    let safety = super::backend_gen::impls_safety_contract("X64V3Token");
    let mut code = formatdoc! {r#"
        //! Backend implementations for X64V3Token (x86-64).
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.
        //!
        {safety}
        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        use core::arch::x86_64::*;

        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        use archmage::{{attune, X64V3Token}};

        use crate::simd::backends::*;

        {sse2_boundary}

    "#};

    for &ty in narrow_types {
        if ty.is_float() {
            code.push_str("#[cfg(all(target_arch = \"x86_64\", target_feature = \"sse2\"))]\n");
            code.push_str(&generate_x86_float_impl(ty));
            code.push('\n');
        } else {
            code.push_str("\n#[cfg(all(target_arch = \"x86_64\", target_feature = \"sse2\"))]\n");
            code.push_str(&generate_x86_int_impl(ty));
            code.push('\n');
        }
    }

    code.push_str(&generate_x86_convert_impls());
    code.push_str(&super::float_layout::x86_v3_f32x16_to_u16_helper());
    code
}

fn x86_prefix(width: SimdWidth) -> &'static str {
    match width {
        SimdWidth::W128 => "_mm",
        SimdWidth::W256 => "_mm256",
        SimdWidth::W512 => unreachable!("W512 belongs to backend_x86_v4"),
    }
}

fn x86_float_suffix(elem: ElementType) -> &'static str {
    match elem {
        ElementType::F32 => "ps",
        ElementType::F64 => "pd",
        _ => unreachable!(),
    }
}

fn x86_arith_suffix(elem_bits: usize) -> &'static str {
    match elem_bits {
        8 => "epi8",
        16 => "epi16",
        32 => "epi32",
        64 => "epi64",
        _ => unreachable!(),
    }
}

fn x86_set1_suffix(elem_bits: usize) -> &'static str {
    match elem_bits {
        8 => "epi8",
        16 => "epi16",
        32 => "epi32",
        64 => "epi64x",
        _ => unreachable!(),
    }
}

fn x86_minmax_suffix(ty: SimdType) -> &'static str {
    match (ty.is_signed(), ty.elem_bits()) {
        (true, 8) => "epi8",
        (false, 8) => "epu8",
        (true, 16) => "epi16",
        (false, 16) => "epu16",
        (true, 32) => "epi32",
        (false, 32) => "epu32",
        _ => unreachable!("64-bit min/max is polyfilled on V3"),
    }
}

// ============================================================================
// Float Implementations (f32x4, f32x8, f64x2, f64x4)
// ============================================================================

fn generate_x86_float_impl(ty: SimdType) -> String {
    let arcane = super::backend_syntax::arcane("X64V3Token");
    let bits = ty.width.bits();
    let baseline = super::backend_syntax::sse2_or_arcane("X64V3Token", bits);
    let baseline_end = if bits == 128 { "}" } else { "" };
    let trait_name = ty.trait_name();
    let inner = ty.width.x86_float_type(ty.elem);
    let p = x86_prefix(ty.width);
    let s = x86_float_suffix(ty.elem);
    let array = ty.array_type();
    let elem = ty.elem.name();
    let lanes = ty.lanes();
    let is_f32 = ty.elem == ElementType::F32;

    let set1 = format!("{p}_set1_{s}");
    let setzero = format!("{p}_setzero_{s}");
    let cast_int_to_float = format!("{p}_castsi{bits}_{s}");
    let cast_float_to_int = format!("{p}_cast{s}_si{bits}");
    let set1_int = if is_f32 { "epi32" } else { "epi64x" };
    let abs_mask = if is_f32 {
        "0x7FFF_FFFFi32"
    } else {
        "0x7FFF_FFFF_FFFF_FFFFi64"
    };

    let floor_intr = format!("{p}_floor_{s}");
    let ceil_intr = format!("{p}_ceil_{s}");
    let round_intr =
        format!("{p}_round_{s}::<{{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }}>");
    let cmp = format!("{p}_cmp_{s}");

    let reduce_add_body = generate_x86_float_reduce_add(ty);
    let reduce_min_body = generate_x86_float_reduce_minmax(ty, "min");
    let reduce_max_body = generate_x86_float_reduce_minmax(ty, "max");

    let approx_section = if is_f32 {
        formatdoc! {r#"
            {arcane}
            fn rcp_approx(self, a: {inner}) -> {inner} {{
                {p}_rcp_{s}(a)
            }}

            {arcane}
            fn rsqrt_approx(self, a: {inner}) -> {inner} {{
                {p}_rsqrt_{s}(a)
            }}

            // Estimate + one FMA-Newton step + a branchless rail rescue: the
            // fastest form meeting the bare-name contract (<= 4 ULP AND exact
            // IEEE rails at ±0/±inf/NaN). The Newton error term `e = 1 - a*r`
            // is NaN exactly on rail lanes (inf*0), so an unordered _self-
            // compare selects them and blendv rescues those lanes to the raw
            // estimate — which is rail-EXACT on x86 (rcpps(±0) = ±inf,
            // rcpps(±inf) = ±0, signed; likewise rsqrtps). Measured cost of
            // the rescue vs the plain Newton form on Zen 5: recip +2.6%,
            // rsqrt +20% — vs +21%/+232% for exact division
            // (benchmarks/recip_x86_zen5-9950x3d_2026-09-03.md). Subnormal
            // inputs remain unspecified at this tier (rcpps flushes them);
            // `recip_portable`/`rsqrt_portable` divide: 0 ULP + subnormals.
            {arcane}
            fn recip(self, a: {inner}) -> {inner} {{
                let one = {{ {p}_set1_{s}(1.0) }};
                let r = <Self as {trait_name}>::rcp_approx(_self, a);
                {{
                    let e = {p}_fnmadd_{s}(a, r, one);
                    let refined = {p}_fmadd_{s}(r, e, r);
                    let bad = {p}_cmp_{s}::<_CMP_UNORD_Q>(e, e);
                    {p}_blendv_{s}(refined, r, bad)
                }}
            }}

            {arcane}
            fn rsqrt(self, a: {inner}) -> {inner} {{
                let half = {{ {p}_set1_{s}(0.5) }};
                let three_halves = {{ {p}_set1_{s}(1.5) }};
                let y = <Self as {trait_name}>::rsqrt_approx(_self, a);
                {{
                    let t = {p}_mul_{s}(a, {p}_mul_{s}(y, y));
                    let refined = {p}_mul_{s}(y, {p}_fnmadd_{s}(half, t, three_halves));
                    let bad = {p}_cmp_{s}::<_CMP_UNORD_Q>(t, t);
                    {p}_blendv_{s}(refined, y, bad)
                }}
            }}
        "#}
    } else {
        formatdoc! {r#"
            {arcane}
            fn rcp_approx(self, a: {inner}) -> {inner} {{
                let one = {{ {p}_set1_{s}(1.0) }};
                <Self as {trait_name}>::div(_self, one, a)
            }}

            {arcane}
            fn rsqrt_approx(self, a: {inner}) -> {inner} {{
                let one = {{ {p}_set1_{s}(1.0) }};
                <Self as {trait_name}>::div(_self, one, <Self as {trait_name}>::sqrt(_self, a))
            }}

            {arcane}
            fn recip(self, a: {inner}) -> {inner} {{
                let one = {{ {p}_set1_{s}(1.0) }};
                <Self as {trait_name}>::div(_self, one, a)
            }}

            {arcane}
            fn rsqrt(self, a: {inner}) -> {inner} {{
                let one = {{ {p}_set1_{s}(1.0) }};
                <Self as {trait_name}>::div(_self, one, <Self as {trait_name}>::sqrt(_self, a))
            }}
        "#}
    };

    let to_u8_x86 = match (ty.elem, ty.width) {
        (ElementType::F32, SimdWidth::W128) => formatdoc! {r#"

            {arcane}
            fn to_u8_bytes(self, a: {inner}) -> [u8; 4] {{
                // Clamp above at 255 first: cvtps turns +inf and anything past
                // i32::MAX into i32::MIN, which the packs would saturate to 0.
                // `min(255, a)` keeps NaN (minps returns its second operand),
                // and NaN still converts to 0, as in the scalar reference.
                let i32s = _mm_cvtps_epi32(_mm_min_ps(_mm_set1_ps(255.0), a));
                let i16s = _mm_packs_epi32(i32s, i32s);
                let u8s = _mm_packus_epi16(i16s, i16s);
                (_mm_cvtsi128_si32(u8s) as u32).to_ne_bytes()
            }}
        "#},
        (ElementType::F32, SimdWidth::W256) => formatdoc! {r#"

            {arcane}
            fn to_u8_bytes(self, a: {inner}) -> [u8; 8] {{
                // Clamp above at 255 before cvtps; see the 128-bit form.
                let i32s = _mm256_cvtps_epi32(_mm256_min_ps(_mm256_set1_ps(255.0), a));
                let lo = _mm256_castsi256_si128(i32s);
                let hi = _mm256_extracti128_si256::<1>(i32s);
                let i16s = _mm_packs_epi32(lo, hi);
                let u8s = _mm_packus_epi16(i16s, i16s);
                (_mm_cvtsi128_si64(u8s) as u64).to_ne_bytes()
            }}
        "#},
        _ => String::new(),
    };

    let transpose_8x8_x86 = if is_f32 && ty.width == SimdWidth::W256 {
        formatdoc! {r#"

            {arcane}
            fn transpose_8x8_repr(self, rows: [{inner}; 8]) -> [{inner}; 8] {{
                let t0 = _mm256_unpacklo_ps(rows[0], rows[1]);
                let t1 = _mm256_unpackhi_ps(rows[0], rows[1]);
                let t2 = _mm256_unpacklo_ps(rows[2], rows[3]);
                let t3 = _mm256_unpackhi_ps(rows[2], rows[3]);
                let t4 = _mm256_unpacklo_ps(rows[4], rows[5]);
                let t5 = _mm256_unpackhi_ps(rows[4], rows[5]);
                let t6 = _mm256_unpacklo_ps(rows[6], rows[7]);
                let t7 = _mm256_unpackhi_ps(rows[6], rows[7]);
                let s0 = _mm256_shuffle_ps::<0x44>(t0, t2);
                let s1 = _mm256_shuffle_ps::<0xEE>(t0, t2);
                let s2 = _mm256_shuffle_ps::<0x44>(t1, t3);
                let s3 = _mm256_shuffle_ps::<0xEE>(t1, t3);
                let s4 = _mm256_shuffle_ps::<0x44>(t4, t6);
                let s5 = _mm256_shuffle_ps::<0xEE>(t4, t6);
                let s6 = _mm256_shuffle_ps::<0x44>(t5, t7);
                let s7 = _mm256_shuffle_ps::<0xEE>(t5, t7);
                [
                    _mm256_permute2f128_ps::<0x20>(s0, s4),
                    _mm256_permute2f128_ps::<0x20>(s1, s5),
                    _mm256_permute2f128_ps::<0x20>(s2, s6),
                    _mm256_permute2f128_ps::<0x20>(s3, s7),
                    _mm256_permute2f128_ps::<0x31>(s0, s4),
                    _mm256_permute2f128_ps::<0x31>(s1, s5),
                    _mm256_permute2f128_ps::<0x31>(s2, s6),
                    _mm256_permute2f128_ps::<0x31>(s3, s7),
                ]
            }}
        "#}
    } else {
        String::new()
    };

    let internal_impl = if is_f32 {
        generate_x86_f32_internal(ty.width, lanes, inner, &arcane)
    } else {
        String::new()
    };

    formatdoc! {r#"
        impl {trait_name} for archmage::X64V3Token {{
            type Repr = {inner};

            // ====== Construction ======

            {arcane}
            fn splat(self, v: {elem}) -> {inner} {{
                {set1}(v)
            }}

            {arcane}
            fn zero(self) -> {inner} {{
                {setzero}()
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

            // ====== Arithmetic ======

            {baseline}
            fn add(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_add_{s}(a, b)
            }}
            {baseline_end}

            {baseline}
            fn sub(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_sub_{s}(a, b)
            }}
            {baseline_end}

            {arcane}
            fn mul(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_mul_{s}(a, b)
            }}

            {arcane}
            fn div(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_div_{s}(a, b)
            }}

            {arcane}
            fn neg(self, a: {inner}) -> {inner} {{
                {p}_xor_{s}(a, {p}_set1_{s}(-0.0))
            }}

            // ====== Math ======

            {arcane}
            fn min(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_blendv_{s}({p}_min_{s}(a, b), a, {cmp}::<_CMP_UNORD_Q>(b, b))
            }}

            {arcane}
            fn max(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_blendv_{s}({p}_max_{s}(a, b), a, {cmp}::<_CMP_UNORD_Q>(b, b))
            }}

            {arcane}
            fn minimum(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_or_{s}({p}_or_{s}({p}_min_{s}(a, b), {p}_min_{s}(b, a)), {cmp}::<_CMP_UNORD_Q>(a, b))
            }}

            {arcane}
            fn maximum(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_or_{s}({p}_and_{s}({p}_max_{s}(a, b), {p}_max_{s}(b, a)), {cmp}::<_CMP_UNORD_Q>(a, b))
            }}

            {arcane}
            fn min_fast(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_min_{s}(a, b)
            }}

            {arcane}
            fn max_fast(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_max_{s}(a, b)
            }}

            {baseline}
            fn sqrt(self, a: {inner}) -> {inner} {{
                {p}_sqrt_{s}(a)
            }}
            {baseline_end}

            {arcane}
            fn abs(self, a: {inner}) -> {inner} {{
                let mask = {cast_int_to_float}({p}_set1_{set1_int}({abs_mask}));
                {p}_and_{s}(a, mask)
            }}

            {arcane}
            fn trunc(self, a: {inner}) -> {inner} {{ {p}_round_{s}::<0x0b>(a) }}

            {arcane}
            fn mul_neg_add_relaxed(self, a: {inner}, b: {inner}, c: {inner}) -> {inner} {{ {p}_fnmadd_{s}(a, b, c) }}

            {arcane}
            fn mul_neg_add_fused(self, a: {inner}, b: {inner}, c: {inner}) -> {inner} {{ {p}_fnmadd_{s}(a, b, c) }}

            {arcane}
            fn floor(self, a: {inner}) -> {inner} {{
                {floor_intr}(a)
            }}

            {arcane}
            fn ceil(self, a: {inner}) -> {inner} {{
                {ceil_intr}(a)
            }}

            {arcane}
            fn round(self, a: {inner}) -> {inner} {{
                {round_intr}(a)
            }}

            {arcane}
            fn mul_add(self, a: {inner}, b: {inner}, c: {inner}) -> {inner} {{
                {p}_fmadd_{s}(a, b, c)
            }}

            {arcane}
            fn mul_sub(self, a: {inner}, b: {inner}, c: {inner}) -> {inner} {{
                {p}_fmsub_{s}(a, b, c)
            }}

            {arcane}
            fn mul_add_portable(self, a: {inner}, b: {inner}, c: {inner}) -> {inner} {{
                {p}_fmadd_{s}(a, b, c)
            }}

            {arcane}
            fn mul_sub_portable(self, a: {inner}, b: {inner}, c: {inner}) -> {inner} {{
                {p}_fmsub_{s}(a, b, c)
            }}

            // ====== Comparisons ======

            {arcane}
            fn simd_eq(self, a: {inner}, b: {inner}) -> {inner} {{
                {cmp}::<_CMP_EQ_OQ>(a, b)
            }}

            {arcane}
            fn simd_ne(self, a: {inner}, b: {inner}) -> {inner} {{
                {cmp}::<_CMP_NEQ_UQ>(a, b)
            }}

            {arcane}
            fn simd_lt(self, a: {inner}, b: {inner}) -> {inner} {{
                {cmp}::<_CMP_LT_OQ>(a, b)
            }}

            {arcane}
            fn simd_le(self, a: {inner}, b: {inner}) -> {inner} {{
                {cmp}::<_CMP_LE_OQ>(a, b)
            }}

            {arcane}
            fn simd_gt(self, a: {inner}, b: {inner}) -> {inner} {{
                {cmp}::<_CMP_GT_OQ>(a, b)
            }}

            {arcane}
            fn simd_ge(self, a: {inner}, b: {inner}) -> {inner} {{
                {cmp}::<_CMP_GE_OQ>(a, b)
            }}

            {arcane}
            fn blend(self, mask: {inner}, if_true: {inner}, if_false: {inner}) -> {inner} {{
                {p}_blendv_{s}(if_false, if_true, mask)
            }}

            // ====== Reductions ======

            {arcane}
            fn reduce_add(self, a: {inner}) -> {elem} {{
        {reduce_add_body}
            }}

            {arcane}
            fn reduce_min(self, a: {inner}) -> {elem} {{
        {reduce_min_body}
            }}

            {arcane}
            fn reduce_max(self, a: {inner}) -> {elem} {{
        {reduce_max_body}
            }}

            // ====== Approximations ======

        {approx_section}
            // ====== Bitwise ======

            {baseline}
            fn not(self, a: {inner}) -> {inner} {{
                let ones = {p}_set1_{set1_int}(-1);
                let as_int = {cast_float_to_int}(a);
                {cast_int_to_float}({p}_xor_si{bits}(as_int, ones))
            }}
            {baseline_end}

            {baseline}
            fn bitand(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_and_{s}(a, b)
            }}
            {baseline_end}

            {baseline}
            fn bitor(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_or_{s}(a, b)
            }}
            {baseline_end}

            {baseline}
            fn bitxor(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_xor_{s}(a, b)
            }}
            {baseline_end}
            {to_u8_x86}
            {transpose_8x8_x86}
        }}{internal_impl}
    "#}
}

fn generate_x86_f32_internal(width: SimdWidth, lanes: usize, inner: &str, arcane: &str) -> String {
    let float_layout = super::float_layout::x86(lanes, inner, arcane);
    let (store_rgba_x86, rgb_3ch_x86) = match width {
        SimdWidth::W128 => (
            formatdoc! {r#"

                {arcane}
                fn store_rgba_u8_into(self, r: {inner}, g: {inner}, b: {inner}, a: {inner}, out: &mut [u8; 16]) {{
                    let max = _mm_set1_ps(255.0);
                    let rg = _mm_packs_epi32(
                        _mm_cvtps_epi32(_mm_min_ps(max, r)),
                        _mm_cvtps_epi32(_mm_min_ps(max, g)),
                    );
                    let ba = _mm_packs_epi32(
                        _mm_cvtps_epi32(_mm_min_ps(max, b)),
                        _mm_cvtps_epi32(_mm_min_ps(max, a)),
                    );
                    // [R0-3,G0-3,B0-3,A0-3] -> interleaved RGBA pixels 0-3.
                    let packed = _mm_packus_epi16(rg, ba);
                    let shuf = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
                    crate::simd_storage::store(_mm_shuffle_epi8(packed, shuf), out);
                }}
            "#},
            formatdoc! {r#"

                {arcane}
                fn deinterleave_3ch_repr(self, rgb: [{inner}; 3]) -> [{inner}; 3] {{
                    let [v0, v1, v2] = rgb;
                    let t_r = _mm_shuffle_ps::<0b01_01_10_10>(v1, v2);
                    let r = _mm_shuffle_ps::<0b10_00_11_00>(v0, t_r);
                    let t_g0 = _mm_shuffle_ps::<0b00_00_01_01>(v0, v1);
                    let t_g1 = _mm_shuffle_ps::<0b10_10_11_11>(v1, v2);
                    let g = _mm_shuffle_ps::<0b10_00_10_00>(t_g0, t_g1);
                    let t_b = _mm_shuffle_ps::<0b01_01_10_10>(v0, v1);
                    let b = _mm_shuffle_ps::<0b11_00_10_00>(t_b, v2);
                    [r, g, b]
                }}

                {arcane}
                fn interleave_3ch_repr(self, channels: [{inner}; 3]) -> [{inner}; 3] {{
                    let [r, g, b] = channels;
                    let rg_lo = _mm_unpacklo_ps(r, g);
                    let rg_hi = _mm_unpackhi_ps(r, g);
                    let gb_lo = _mm_unpacklo_ps(g, b);
                    let gb_hi = _mm_unpackhi_ps(g, b);
                    let br01 = _mm_shuffle_ps::<0b01_01_00_00>(b, r);
                    let out0 = _mm_shuffle_ps::<0b10_00_01_00>(rg_lo, br01);
                    let out1 = _mm_shuffle_ps::<0b01_00_11_10>(gb_lo, rg_hi);
                    let br23 = _mm_shuffle_ps::<0b11_11_10_10>(b, r);
                    let out2 = _mm_shuffle_ps::<0b11_10_10_00>(br23, gb_hi);
                    [out0, out1, out2]
                }}

                {arcane}
                fn load_rgb_bytes(self, rgb: &[u8; 12]) -> ({inner}, {inner}, {inner}) {{
                    let lo8: u64 = crate::simd_storage::copy(rgb.first_chunk::<8>().unwrap());
                    let hi4: u32 = crate::simd_storage::copy(rgb.last_chunk::<4>().unwrap());
                    let raw = _mm_insert_epi32::<2>(_mm_cvtsi64_si128(lo8 as i64), hi4 as i32);
                    let r32 = _mm_shuffle_epi8(
                        raw,
                        _mm_setr_epi8(0, -1, -1, -1, 3, -1, -1, -1, 6, -1, -1, -1, 9, -1, -1, -1),
                    );
                    let g32 = _mm_shuffle_epi8(
                        raw,
                        _mm_setr_epi8(1, -1, -1, -1, 4, -1, -1, -1, 7, -1, -1, -1, 10, -1, -1, -1),
                    );
                    let b32 = _mm_shuffle_epi8(
                        raw,
                        _mm_setr_epi8(2, -1, -1, -1, 5, -1, -1, -1, 8, -1, -1, -1, 11, -1, -1, -1),
                    );
                    (_mm_cvtepi32_ps(r32), _mm_cvtepi32_ps(g32), _mm_cvtepi32_ps(b32))
                }}

                {arcane}
                fn store_rgb_u8_into(self, r: {inner}, g: {inner}, b: {inner}, out: &mut [u8; 12]) {{
                    let max = _mm_set1_ps(255.0);
                    let rg16 = _mm_packs_epi32(
                        _mm_cvtps_epi32(_mm_min_ps(max, r)),
                        _mm_cvtps_epi32(_mm_min_ps(max, g)),
                    );
                    let b16 = _mm_packs_epi32(
                        _mm_cvtps_epi32(_mm_min_ps(max, b)),
                        _mm_setzero_si128(),
                    );
                    let rgb8 = _mm_packus_epi16(rg16, b16);
                    let shuf = _mm_setr_epi8(0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, -1, -1, -1, -1);
                    let packed = _mm_shuffle_epi8(rgb8, shuf);
                    crate::simd_storage::store(
                        _mm_cvtsi128_si64(packed) as u64,
                        out.first_chunk_mut::<8>().unwrap(),
                    );
                    crate::simd_storage::store(
                        _mm_extract_epi32::<2>(packed) as u32,
                        out.last_chunk_mut::<4>().unwrap(),
                    );
                }}
            "#},
        ),
        SimdWidth::W256 => (
            formatdoc! {r#"

                {arcane}
                fn store_rgba_u8_into(self, r: {inner}, g: {inner}, b: {inner}, a: {inner}, out: &mut [u8; 32]) {{
                    // AVX2 packs are lane-wise: lane0 holds pixels 0-3, lane1 4-7.
                    let max = _mm256_set1_ps(255.0);
                    let rg = _mm256_packs_epi32(
                        _mm256_cvtps_epi32(_mm256_min_ps(max, r)),
                        _mm256_cvtps_epi32(_mm256_min_ps(max, g)),
                    );
                    let ba = _mm256_packs_epi32(
                        _mm256_cvtps_epi32(_mm256_min_ps(max, b)),
                        _mm256_cvtps_epi32(_mm256_min_ps(max, a)),
                    );
                    let packed = _mm256_packus_epi16(rg, ba);
                    let shuf = _mm256_setr_epi8(
                        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
                        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
                    );
                    crate::simd_storage::store(_mm256_shuffle_epi8(packed, shuf), out);
                }}
            "#},
            formatdoc! {r#"

                {arcane}
                fn deinterleave_3ch_repr(self, rgb: [{inner}; 3]) -> [{inner}; 3] {{
                    let [v0, v1, v2] = rgb;
                    let r_mix = _mm256_blend_ps::<0x24>(_mm256_blend_ps::<0x92>(v0, v1), v2);
                    let g_mix = _mm256_blend_ps::<0x49>(_mm256_blend_ps::<0x24>(v0, v1), v2);
                    let b_mix = _mm256_blend_ps::<0x92>(_mm256_blend_ps::<0x49>(v0, v1), v2);
                    [
                        _mm256_permutevar8x32_ps(r_mix, _mm256_setr_epi32(0, 3, 6, 1, 4, 7, 2, 5)),
                        _mm256_permutevar8x32_ps(g_mix, _mm256_setr_epi32(1, 4, 7, 2, 5, 0, 3, 6)),
                        _mm256_permutevar8x32_ps(b_mix, _mm256_setr_epi32(2, 5, 0, 3, 6, 1, 4, 7)),
                    ]
                }}

                {arcane}
                fn interleave_3ch_repr(self, channels: [{inner}; 3]) -> [{inner}; 3] {{
                    let [r, g, b] = channels;
                    let rp = _mm256_permutevar8x32_ps(r, _mm256_setr_epi32(0, 3, 6, 1, 4, 7, 2, 5));
                    let gp = _mm256_permutevar8x32_ps(g, _mm256_setr_epi32(5, 0, 3, 6, 1, 4, 7, 2));
                    let bp = _mm256_permutevar8x32_ps(b, _mm256_setr_epi32(2, 5, 0, 3, 6, 1, 4, 7));
                    [
                        _mm256_blend_ps::<0x24>(_mm256_blend_ps::<0x92>(rp, gp), bp),
                        _mm256_blend_ps::<0x24>(_mm256_blend_ps::<0x49>(rp, bp), gp),
                        _mm256_blend_ps::<0x24>(_mm256_blend_ps::<0x92>(gp, bp), rp),
                    ]
                }}

                {arcane}
                fn load_rgb_bytes(self, rgb: &[u8; 24]) -> ({inner}, {inner}, {inner}) {{
                    let lo: __m128i = crate::simd_storage::copy(rgb.first_chunk::<16>().unwrap());
                    let hi: __m128i = crate::simd_storage::copy(rgb.last_chunk::<16>().unwrap());
                    let raw = _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi);
                    let r32 = _mm256_shuffle_epi8(
                        raw,
                        _mm256_setr_epi8(
                            0, -1, -1, -1, 3, -1, -1, -1, 6, -1, -1, -1, 9, -1, -1, -1,
                            4, -1, -1, -1, 7, -1, -1, -1, 10, -1, -1, -1, 13, -1, -1, -1,
                        ),
                    );
                    let g32 = _mm256_shuffle_epi8(
                        raw,
                        _mm256_setr_epi8(
                            1, -1, -1, -1, 4, -1, -1, -1, 7, -1, -1, -1, 10, -1, -1, -1,
                            5, -1, -1, -1, 8, -1, -1, -1, 11, -1, -1, -1, 14, -1, -1, -1,
                        ),
                    );
                    let b32 = _mm256_shuffle_epi8(
                        raw,
                        _mm256_setr_epi8(
                            2, -1, -1, -1, 5, -1, -1, -1, 8, -1, -1, -1, 11, -1, -1, -1,
                            6, -1, -1, -1, 9, -1, -1, -1, 12, -1, -1, -1, 15, -1, -1, -1,
                        ),
                    );
                    (
                        _mm256_cvtepi32_ps(r32),
                        _mm256_cvtepi32_ps(g32),
                        _mm256_cvtepi32_ps(b32),
                    )
                }}

                {arcane}
                fn store_rgb_u8_into(self, r: {inner}, g: {inner}, b: {inner}, out: &mut [u8; 24]) {{
                    let max = _mm256_set1_ps(255.0);
                    let rg16 = _mm256_packs_epi32(
                        _mm256_cvtps_epi32(_mm256_min_ps(max, r)),
                        _mm256_cvtps_epi32(_mm256_min_ps(max, g)),
                    );
                    let b16 = _mm256_packs_epi32(
                        _mm256_cvtps_epi32(_mm256_min_ps(max, b)),
                        _mm256_setzero_si256(),
                    );
                    let rgb8 = _mm256_packus_epi16(rg16, b16);
                    let shuf = _mm256_setr_epi8(
                        0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, -1, -1, -1, -1,
                        0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, -1, -1, -1, -1,
                    );
                    let packed = _mm256_shuffle_epi8(rgb8, shuf);
                    let lo12 = _mm256_castsi256_si128(packed);
                    let hi12 = _mm256_extracti128_si256::<1>(packed);
                    let lo16 = _mm_or_si128(lo12, _mm_slli_si128::<12>(hi12));
                    let hi8 = _mm_srli_si128::<4>(hi12);
                    crate::simd_storage::store(lo16, out.first_chunk_mut::<16>().unwrap());
                    crate::simd_storage::store(
                        _mm_cvtsi128_si64(hi8) as u64,
                        out.last_chunk_mut::<8>().unwrap(),
                    );
                }}
            "#},
        ),
        SimdWidth::W512 => unreachable!(),
    };

    formatdoc! {r#"

        #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
        impl crate::simd::backends::F32x{lanes}Internal for archmage::X64V3Token {{
            type InternalRepr = {inner};
        {store_rgba_x86}
        {float_layout}
        {rgb_3ch_x86}
        }}"#}
}

fn generate_x86_float_reduce_add(ty: SimdType) -> String {
    let p = x86_prefix(ty.width);
    let s = x86_float_suffix(ty.elem);
    match (ty.elem, ty.width) {
        (ElementType::F32, SimdWidth::W128) => formatdoc! {"
                {{
                    let shuf = _mm_shuffle_ps::<0b10_11_00_01>(a, a);
                    let s1 = _mm_add_ps(a, shuf);
                    let s2 = _mm_add_ps(s1, _mm_movehl_ps(s1, s1));
                    _mm_cvtss_f32(s2)
                }}"},
        (ElementType::F64, SimdWidth::W128) => formatdoc! {"
                {{
                    let h = _mm_hadd_pd(a, a);
                    _mm_cvtsd_f64(h)
                }}"},
        (ElementType::F32, SimdWidth::W256) => formatdoc! {"
                {{
                    let hi = {p}_extractf128_{s}::<1>(a);
                    let lo = {p}_cast{s}256_{s}128(a);
                    let sum = _mm_add_ps(lo, hi);
                    let shuf = _mm_shuffle_ps::<0b10_11_00_01>(sum, sum);
                    let s1 = _mm_add_ps(sum, shuf);
                    let s2 = _mm_add_ps(s1, _mm_movehl_ps(s1, s1));
                    _mm_cvtss_f32(s2)
                }}"},
        (ElementType::F64, SimdWidth::W256) => formatdoc! {"
                {{
                    let hi = {p}_extractf128_{s}::<1>(a);
                    let lo = {p}_cast{s}256_{s}128(a);
                    let sum = _mm_add_{s}(lo, hi);
                    let h = _mm_hadd_pd(sum, sum);
                    _mm_cvtsd_f64(h)
                }}"},
        _ => unreachable!(),
    }
}

fn generate_x86_float_reduce_minmax(ty: SimdType, op: &str) -> String {
    let p = x86_prefix(ty.width);
    let s = x86_float_suffix(ty.elem);
    match (ty.elem, ty.width) {
        (ElementType::F32, SimdWidth::W128) => formatdoc! {"
                {{
                    let shuf = _mm_shuffle_ps::<0b10_11_00_01>(a, a);
                    let m1 = _mm_{op}_ps(a, shuf);
                    let shuf2 = _mm_shuffle_ps::<0b00_00_10_10>(m1, m1);
                    let m2 = _mm_{op}_ps(m1, shuf2);
                    _mm_cvtss_f32(m2)
                }}"},
        (ElementType::F64, SimdWidth::W128) => formatdoc! {"
                {{
                    let shuf = _mm_shuffle_pd::<0b01>(a, a);
                    let m = _mm_{op}_pd(a, shuf);
                    _mm_cvtsd_f64(m)
                }}"},
        (ElementType::F32, SimdWidth::W256) => formatdoc! {"
                {{
                    let hi = {p}_extractf128_{s}::<1>(a);
                    let lo = {p}_cast{s}256_{s}128(a);
                    let m = _mm_{op}_{s}(lo, hi);
                    let shuf = _mm_shuffle_ps::<0b10_11_00_01>(m, m);
                    let m1 = _mm_{op}_ps(m, shuf);
                    let shuf2 = _mm_shuffle_ps::<0b00_00_10_10>(m1, m1);
                    let m2 = _mm_{op}_ps(m1, shuf2);
                    _mm_cvtss_f32(m2)
                }}"},
        (ElementType::F64, SimdWidth::W256) => formatdoc! {"
                {{
                    let hi = {p}_extractf128_{s}::<1>(a);
                    let lo = {p}_cast{s}256_{s}128(a);
                    let m = _mm_{op}_{s}(lo, hi);
                    let shuf = _mm_shuffle_pd::<0b01>(m, m);
                    let m2 = _mm_{op}_pd(m, shuf);
                    _mm_cvtsd_f64(m2)
                }}"},
        _ => unreachable!(),
    }
}

// ============================================================================
// Unified Integer Implementation (all 16 narrow W128 + W256 integer types)
// ============================================================================

fn generate_x86_int_impl(ty: SimdType) -> String {
    let integer_methods = super::backend_gen_widen_narrow::methods(&ty.name(), "X64V3Token");
    let arcane = super::backend_syntax::arcane("X64V3Token");
    let bits = ty.width.bits();
    let baseline = super::backend_syntax::sse2_or_arcane("X64V3Token", bits);
    let baseline_end = if bits == 128 { "}" } else { "" };
    let trait_name = ty.trait_name();
    let inner = ty.width.x86_int_type();
    let p = x86_prefix(ty.width);
    let array = ty.array_type();
    let elem = ty.elem.name();
    let eb = ty.elem_bits();
    let signed = ty.is_signed();
    let arith_suf = x86_arith_suffix(eb);
    let set1_suf = x86_set1_suffix(eb);
    let set1_cast = if signed {
        ""
    } else {
        match eb {
            8 => " as i8",
            16 => " as i16",
            32 => " as i32",
            64 => " as i64",
            _ => unreachable!(),
        }
    };
    let neg1_lit = match ty.elem {
        ElementType::U8 => "-1_i8",
        ElementType::U16 => "-1_i16",
        ElementType::U64 => "-1_i64",
        _ => "-1",
    };

    // Historical spacing: 32/64-bit signed/u32 emitters put blank lines before
    // `mul`/`neg`/`abs`, whereas the 8/16/u64 emitter packed them tightly.
    let spaced_arith = matches!(
        ty.elem,
        ElementType::I32 | ElementType::U32 | ElementType::I64
    );
    let arith_sep = if spaced_arith { "\n" } else { "" };

    let mut body = formatdoc! {r#"
            type Repr = {inner};

            // ====== Construction ======

            {arcane}
            fn splat(self, v: {elem}) -> {inner} {{
                {p}_set1_{set1_suf}(v{set1_cast})
            }}

            {arcane}
            fn zero(self) -> {inner} {{
                {p}_setzero_si{bits}()
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

            // ====== Arithmetic ======

            {baseline}
            fn add(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_add_{arith_suf}(a, b)
            }}
            {baseline_end}

            {baseline}
            fn sub(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_sub_{arith_suf}(a, b)
            }}
            {baseline_end}
    "#};

    if matches!(eb, 16 | 32) {
        body.push_str(arith_sep);
        body.push_str(&formatdoc! {r#"
            {arcane}
            fn mul(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_mullo_{arith_suf}(a, b)
            }}
        "#});
    }

    if signed {
        body.push_str(arith_sep);
        body.push_str(&formatdoc! {r#"
            {arcane}
            fn neg(self, a: {inner}) -> {inner} {{
                {p}_sub_{arith_suf}({p}_setzero_si{bits}(), a)
            }}
        "#});
    }

    // Math: min/max/abs
    match ty.elem {
        ElementType::I64 => {
            body.push_str(&formatdoc! {r#"

                // ====== Math ======

                {arcane}
                fn min(self, a: {inner}, b: {inner}) -> {inner} {{
                    // Polyfill: compare+select (no native i64 min on AVX2)
                    {{
                        let mask = {p}_cmpgt_epi64(a, b);
                        {p}_blendv_epi8(a, b, mask)
                    }}
                }}

                {arcane}
                fn max(self, a: {inner}, b: {inner}) -> {inner} {{
                    // Polyfill: compare+select (no native i64 max on AVX2)
                    {{
                        let mask = {p}_cmpgt_epi64(a, b);
                        {p}_blendv_epi8(b, a, mask)
                    }}
                }}

                {arcane}
                fn abs(self, a: {inner}) -> {inner} {{
                    // Polyfill: (a ^ sign) - sign (two's complement trick)
                    {{
                        let zero = {p}_setzero_si{bits}();
                        let sign = {p}_cmpgt_epi64(zero, a);
                        {p}_sub_epi64({p}_xor_si{bits}(a, sign), sign)
                    }}
                }}
            "#});
        }
        ElementType::U64 => {
            body.push_str(&formatdoc! {r#"

                // ====== Math ======

                {arcane}
                fn min(self, a: {inner}, b: {inner}) -> {inner} {{
                    let bias = {p}_set1_epi64x(i64::MIN);
                    let a_biased = {p}_xor_si{bits}(a, bias);
                    let b_biased = {p}_xor_si{bits}(b, bias);
                    let mask = {p}_cmpgt_epi64(a_biased, b_biased);
                    {p}_blendv_epi8(a, b, mask)
                }}

                {arcane}
                fn max(self, a: {inner}, b: {inner}) -> {inner} {{
                    let bias = {p}_set1_epi64x(i64::MIN);
                    let a_biased = {p}_xor_si{bits}(a, bias);
                    let b_biased = {p}_xor_si{bits}(b, bias);
                    let mask = {p}_cmpgt_epi64(a_biased, b_biased);
                    {p}_blendv_epi8(b, a, mask)
                }}
            "#});
        }
        _ => {
            let mm_suf = x86_minmax_suffix(ty);
            body.push_str(&formatdoc! {r#"

                // ====== Math ======

                {arcane}
                fn min(self, a: {inner}, b: {inner}) -> {inner} {{
                    {p}_min_{mm_suf}(a, b)
                }}

                {arcane}
                fn max(self, a: {inner}, b: {inner}) -> {inner} {{
                    {p}_max_{mm_suf}(a, b)
                }}
            "#});
            if signed {
                body.push_str(arith_sep);
                body.push_str(&formatdoc! {r#"
                    {arcane}
                    fn abs(self, a: {inner}) -> {inner} {{
                        {p}_abs_{arith_suf}(a)
                    }}
                "#});
            }
        }
    }

    // Comparisons
    if signed {
        let invert = |var: &str| -> String {
            if ty.elem == ElementType::I64 {
                format!("{p}_xor_si{bits}({var}, {p}_set1_epi64x(-1))")
            } else {
                format!("{p}_andnot_si{bits}({var}, {p}_set1_{set1_suf}(-1))")
            }
        };
        body.push_str(&formatdoc! {r#"

            // ====== Comparisons ======

            {arcane}
            fn simd_eq(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_cmpeq_{arith_suf}(a, b)
            }}

            {arcane}
            fn simd_ne(self, a: {inner}, b: {inner}) -> {inner} {{
                let eq = {p}_cmpeq_{arith_suf}(a, b);
                {ne_inv}
            }}

            {arcane}
            fn simd_lt(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_cmpgt_{arith_suf}(b, a)
            }}

            {arcane}
            fn simd_le(self, a: {inner}, b: {inner}) -> {inner} {{
                let gt = {p}_cmpgt_{arith_suf}(a, b);
                {le_inv}
            }}

            {arcane}
            fn simd_gt(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_cmpgt_{arith_suf}(a, b)
            }}

            {arcane}
            fn simd_ge(self, a: {inner}, b: {inner}) -> {inner} {{
                let lt = {p}_cmpgt_{arith_suf}(b, a);
                {ge_inv}
            }}
        "#,
            ne_inv = invert("eq"),
            le_inv = invert("gt"),
            ge_inv = invert("lt"),
        });
    } else if ty.elem == ElementType::U32 {
        body.push_str(&formatdoc! {r#"

            // ====== Comparisons ======

            {arcane}
            fn simd_eq(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_cmpeq_epi32(a, b)
            }}

            {arcane}
            fn simd_ne(self, a: {inner}, b: {inner}) -> {inner} {{
                let eq = {p}_cmpeq_epi32(a, b);
                {p}_andnot_si{bits}(eq, {p}_set1_epi32(-1))
            }}

            {arcane}
            fn simd_gt(self, a: {inner}, b: {inner}) -> {inner} {{
                // Unsigned comparison via bias trick: XOR both with 0x80000000
                // to convert to signed range, then use signed cmpgt.
                {{
                    let bias = {p}_set1_epi32(i32::MIN);
                    let sa = {p}_xor_si{bits}(a, bias);
                    let sb = {p}_xor_si{bits}(b, bias);
                    {p}_cmpgt_epi32(sa, sb)
                }}
            }}

            {arcane}
            fn simd_lt(self, a: {inner}, b: {inner}) -> {inner} {{
                <Self as {trait_name}>::simd_gt(_self, b, a)
            }}

            {arcane}
            fn simd_le(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_cmpeq_epi32({p}_min_epu32(a, b), a)
            }}

            {arcane}
            fn simd_ge(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_cmpeq_epi32({p}_min_epu32(a, b), b)
            }}
        "#});
    } else {
        let bias_lit = format!("{}::MIN", ty.elem.signed_int_name());
        body.push_str(&formatdoc! {r#"

            // ====== Comparisons ======

            {arcane}
            fn simd_eq(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_cmpeq_{arith_suf}(a, b)
            }}

            {arcane}
            fn simd_ne(self, a: {inner}, b: {inner}) -> {inner} {{
                let eq = {p}_cmpeq_{arith_suf}(a, b);
                {p}_andnot_si{bits}(eq, {p}_set1_{set1_suf}({neg1_lit}))
            }}

            {arcane}
            fn simd_gt(self, a: {inner}, b: {inner}) -> {inner} {{
                let bias = {p}_set1_{set1_suf}({bias_lit});
                let sa = {p}_xor_si{bits}(a, bias);
                let sb = {p}_xor_si{bits}(b, bias);
                {p}_cmpgt_{arith_suf}(sa, sb)
            }}

            {arcane}
            fn simd_lt(self, a: {inner}, b: {inner}) -> {inner} {{
                <Self as {trait_name}>::simd_gt(_self, b, a)
            }}

            {arcane}
            fn simd_le(self, a: {inner}, b: {inner}) -> {inner} {{
                let gt = <Self as {trait_name}>::simd_gt(_self, a, b);
                {p}_andnot_si{bits}(gt, {p}_set1_{set1_suf}({neg1_lit}))
            }}

            {arcane}
            fn simd_ge(self, a: {inner}, b: {inner}) -> {inner} {{
                let lt = <Self as {trait_name}>::simd_gt(_self, b, a);
                {p}_andnot_si{bits}(lt, {p}_set1_{set1_suf}({neg1_lit}))
            }}
        "#});
    }

    // Blend & Reduce add & Bitwise
    let reduce_body = generate_x86_int_reduce_add(ty);
    let not_body = if ty.elem == ElementType::I64 {
        format!("{p}_xor_si{bits}(a, {p}_set1_epi64x(-1))")
    } else {
        format!("{p}_andnot_si{bits}(a, {p}_set1_{set1_suf}({neg1_lit}))")
    };
    body.push_str(&formatdoc! {r#"

            {arcane}
            fn blend(self, mask: {inner}, if_true: {inner}, if_false: {inner}) -> {inner} {{
                {p}_blendv_epi8(if_false, if_true, mask)
            }}

            // ====== Reductions ======

            {arcane}
            fn reduce_add(self, a: {inner}) -> {elem} {{
        {reduce_body}
            }}

            // ====== Bitwise ======

            {baseline}
            fn not(self, a: {inner}) -> {inner} {{
                {not_body}
            }}
            {baseline_end}

            {baseline}
            fn bitand(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_and_si{bits}(a, b)
            }}
            {baseline_end}

            {baseline}
            fn bitor(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_or_si{bits}(a, b)
            }}
            {baseline_end}

            {baseline}
            fn bitxor(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_xor_si{bits}(a, b)
            }}
            {baseline_end}
    "#});

    // Shifts, uniform shifts, saturating arithmetic, boolean reductions
    body.push_str(&generate_x86_int_shifts(ty));
    body.push_str(&generate_x86_int_uniform_shifts(ty));
    body.push_str(&generate_x86_int_saturating(ty));
    body.push_str(&generate_x86_int_boolean(ty));

    let im_sep = if ty.elem == ElementType::I32 {
        ""
    } else {
        "\n"
    };
    formatdoc! {r#"
        impl {trait_name} for archmage::X64V3Token {{
        {body}{im_sep}{integer_methods}
        }}
    "#}
}

fn generate_x86_int_reduce_add(ty: SimdType) -> String {
    match (ty.elem, ty.width) {
        (ElementType::I32 | ElementType::U32, SimdWidth::W128) => {
            let cast = if ty.elem == ElementType::U32 {
                " as u32"
            } else {
                ""
            };
            formatdoc! {"
                {{
                    let hi = _mm_shuffle_epi32::<0b01_00_11_10>(a);
                    let sum = _mm_add_epi32(a, hi);
                    let hi2 = _mm_shuffle_epi32::<0b00_00_00_01>(sum);
                    let sum2 = _mm_add_epi32(sum, hi2);
                    _mm_cvtsi128_si32(sum2){cast}
                }}"}
        }
        (ElementType::I32 | ElementType::U32, SimdWidth::W256) => {
            let cast = if ty.elem == ElementType::U32 {
                " as u32"
            } else {
                ""
            };
            formatdoc! {"
                {{
                    let lo = _mm256_castsi256_si128(a);
                    let hi = _mm256_extracti128_si256::<1>(a);
                    let sum = _mm_add_epi32(lo, hi);
                    let hi2 = _mm_shuffle_epi32::<0b01_00_11_10>(sum);
                    let sum2 = _mm_add_epi32(sum, hi2);
                    let hi3 = _mm_shuffle_epi32::<0b00_00_00_01>(sum2);
                    let sum3 = _mm_add_epi32(sum2, hi3);
                    _mm_cvtsi128_si32(sum3){cast}
                }}"}
        }
        (ElementType::I64, SimdWidth::W128) => formatdoc! {"
                {{
                    let hi = _mm_unpackhi_epi64(a, a);
                    let sum = _mm_add_epi64(a, hi);
                    // Extract low 64-bit lane
                    crate::simd_storage::cast::<__m128i, [i64; 2]>(sum)[0]
                }}"},
        (ElementType::I64, SimdWidth::W256) => formatdoc! {"
                {{
                    let lo = _mm256_castsi256_si128(a);
                    let hi = _mm256_extracti128_si256::<1>(a);
                    let sum = _mm_add_epi64(lo, hi);
                    let hi2 = _mm_unpackhi_epi64(sum, sum);
                    let sum2 = _mm_add_epi64(sum, hi2);
                    crate::simd_storage::cast::<__m128i, [i64; 2]>(sum2)[0]
                }}"},
        _ => {
            let elem = ty.elem.name();
            let trait_name = ty.trait_name();
            formatdoc! {"
                let arr = <Self as {trait_name}>::to_array(_self, a);
                arr.iter().copied().fold(0{elem}, {elem}::wrapping_add)"}
        }
    }
}

fn generate_x86_int_shifts(ty: SimdType) -> String {
    let arcane = super::backend_syntax::arcane("X64V3Token");
    let inner = ty.width.x86_int_type();
    let p = x86_prefix(ty.width);
    let bits = ty.width.bits();
    let eb = ty.elem_bits();
    let signed = ty.is_signed();

    match eb {
        8 => {
            let mut code = formatdoc! {r#"

                // ====== Shifts (polyfill via 16-bit) ======

                {arcane}
                fn shl_const<const N: i32>(self, a: {inner}) -> {inner} {{
                    let shifted = {p}_slli_epi16::<N>(a);
                    let mask = {p}_set1_epi8((0xFFu8.wrapping_shl(N as u32)) as i8);
                    {p}_and_si{bits}(shifted, mask)
                }}

                {arcane}
                fn shr_logical_const<const N: i32>(self, a: {inner}) -> {inner} {{
                    let shifted = {p}_srli_epi16::<N>(a);
                    let mask = {p}_set1_epi8((0xFFu8.wrapping_shr(N as u32)) as i8);
                    {p}_and_si{bits}(shifted, mask)
                }}
            "#};
            if signed {
                code.push_str(&formatdoc! {r#"

                {arcane}
                fn shr_arithmetic_const<const N: i32>(self, a: {inner}) -> {inner} {{
                    let shifted = {p}_srli_epi16::<N>(a);
                    let byte_mask = {p}_set1_epi8((0xFFu8.wrapping_shr(N as u32)) as i8);
                    let logical = {p}_and_si{bits}(shifted, byte_mask);
                    let zero = {p}_setzero_si{bits}();
                    let sign = {p}_cmpgt_epi8(zero, a);
                    // High-N-bits fill mask via u16 shift: 0x00 at N == 0 (identity
                    // shift needs no sign fill). A u8 `0xFF << (8 - N)` would wrap
                    // the shift amount at N == 0 and corrupt negative lanes.
                    let fill = {p}_set1_epi8(((0xFF00u16 >> N) & 0xFF) as i8);
                    {p}_or_si{bits}(logical, {p}_and_si{bits}(sign, fill))
                }}
                "#});
            }
            code
        }
        16 => {
            let mut code = formatdoc! {r#"

                // ====== Shifts ======

                {arcane}
                fn shl_const<const N: i32>(self, a: {inner}) -> {inner} {{
                    {p}_slli_epi16::<N>(a)
                }}

                {arcane}
                fn shr_logical_const<const N: i32>(self, a: {inner}) -> {inner} {{
                    {p}_srli_epi16::<N>(a)
                }}
            "#};
            if signed {
                code.push_str(&formatdoc! {r#"

                {arcane}
                fn shr_arithmetic_const<const N: i32>(self, a: {inner}) -> {inner} {{
                    {p}_srai_epi16::<N>(a)
                }}
                "#});
            }
            code
        }
        32 => {
            let mut code = formatdoc! {r#"

                // ====== Shifts ======

                {arcane}
                fn shl_const<const N: i32>(self, a: {inner}) -> {inner} {{
                    {p}_slli_epi32::<N>(a)
                }}
            "#};
            if signed {
                code.push_str(&formatdoc! {r#"

                {arcane}
                fn shr_arithmetic_const<const N: i32>(self, a: {inner}) -> {inner} {{
                    {p}_srai_epi32::<N>(a)
                }}
                "#});
            }
            code.push_str(&formatdoc! {r#"

                {arcane}
                fn shr_logical_const<const N: i32>(self, a: {inner}) -> {inner} {{
                    {p}_srli_epi32::<N>(a)
                }}
            "#});
            code
        }
        64 => {
            let mut code = formatdoc! {r#"

                // ====== Shifts ======

                {arcane}
                fn shl_const<const N: i32>(self, a: {inner}) -> {inner} {{
                    {p}_slli_epi64::<N>(a)
                }}
            "#};
            if signed {
                code.push_str(&formatdoc! {r#"

                {arcane}
                fn shr_arithmetic_const<const N: i32>(self, a: {inner}) -> {inner} {{
                    // Polyfill: no native _srai_epi64 on AVX2.
                    // Use logical shift + sign extension.
                    {{
                        // Broadcast sign of each 64-bit lane to all bits of that lane
                        let sign_ext = {p}_srai_epi32::<31>(a);
                        let sign64 = {p}_shuffle_epi32::<0xF5>(sign_ext);
                        let logical = {p}_srli_epi64::<N>(a);
                        // mask = NOT(srli(-1, N)) = upper N bits set, avoids {{64 - N}} const expr
                        let all_ones = {p}_set1_epi64x(-1);
                        let mask = {p}_andnot_si{bits}({p}_srli_epi64::<N>(all_ones), all_ones);
                        {p}_or_si{bits}(logical, {p}_and_si{bits}(sign64, mask))
                    }}
                }}
                "#});
            }
            code.push_str(&formatdoc! {r#"

                {arcane}
                fn shr_logical_const<const N: i32>(self, a: {inner}) -> {inner} {{
                    {p}_srli_epi64::<N>(a)
                }}
            "#});
            code
        }
        _ => unreachable!(),
    }
}

fn generate_x86_int_uniform_shifts(ty: SimdType) -> String {
    let eb = ty.elem_bits();
    if !matches!(eb, 16 | 32) {
        return String::new();
    }
    let arcane = super::backend_syntax::arcane("X64V3Token");
    let inner = ty.width.x86_int_type();
    let p = x86_prefix(ty.width);
    let suf = x86_arith_suffix(eb);

    let header = match ty.elem {
        ElementType::I32 => {
            "\n            // ====== Uniform variable shifts ======\n            // PSLLD/PSRLD give 0 once the count reaches 32 and PSRAD clamps to\n            // a sign fill — the portable contract is the hardware behaviour.\n            // _mm_cvtsi32_si128 zero-extends, so u32::MAX is just a large\n            // out-of-range count, not a negative one.\n"
        }
        ElementType::U32 => {
            "\n            // ====== Uniform variable shifts ======\n            // PSLLD/PSRLD give 0 once the count reaches 32 — the portable\n            // contract is the hardware behaviour. _mm_cvtsi32_si128\n            // zero-extends, so u32::MAX is just a large out-of-range count.\n"
        }
        _ => "\n            // ====== Uniform variable shifts ======\n",
    };

    let mut code = formatdoc! {r#"{header}
            {arcane}
            fn shl_uniform(self, a: {inner}, count: u32) -> {inner} {{
                {p}_sll_{suf}(a, _mm_cvtsi32_si128(count as i32))
            }}

            {arcane}
            fn shr_logical_uniform(self, a: {inner}, count: u32) -> {inner} {{
                {p}_srl_{suf}(a, _mm_cvtsi32_si128(count as i32))
            }}
    "#};

    if ty.is_signed() {
        code.push_str(&formatdoc! {r#"

            {arcane}
            fn shr_arithmetic_uniform(self, a: {inner}, count: u32) -> {inner} {{
                {p}_sra_{suf}(a, _mm_cvtsi32_si128(count as i32))
            }}
        "#});
    }
    code
}

fn generate_x86_int_saturating(ty: SimdType) -> String {
    if ty.elem_bits() > 16 {
        return String::new();
    }
    let arcane = super::backend_syntax::arcane("X64V3Token");
    let inner = ty.width.x86_int_type();
    let p = x86_prefix(ty.width);
    let suf = x86_minmax_suffix(ty);

    formatdoc! {r#"

            // ====== Saturating arithmetic ======

            {arcane}
            fn saturating_add(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_adds_{suf}(a, b)
            }}

            {arcane}
            fn saturating_sub(self, a: {inner}, b: {inner}) -> {inner} {{
                {p}_subs_{suf}(a, b)
            }}
    "#}
}

fn generate_x86_int_boolean(ty: SimdType) -> String {
    let arcane = super::backend_syntax::arcane("X64V3Token");
    let inner = ty.width.x86_int_type();
    let p = x86_prefix(ty.width);
    let bits = ty.width.bits();

    match ty.elem_bits() {
        8 => {
            let all_mask = if bits == 128 {
                "0xFFFF_u32 as i32"
            } else {
                "-1_i32"
            };
            formatdoc! {r#"

            // ====== Boolean ======

            {arcane}
            fn all_true(self, a: {inner}) -> bool {{
                {p}_movemask_epi8(a) == {all_mask}
            }}

            {arcane}
            fn any_true(self, a: {inner}) -> bool {{
                {p}_movemask_epi8(a) != 0
            }}

            {arcane}
            fn bitmask(self, a: {inner}) -> u32 {{
                {p}_movemask_epi8(a) as u32
            }}
            "#}
        }
        16 => {
            let hi_bytes = if bits == 128 {
                "0xAAAA_u32"
            } else {
                "0xAAAA_AAAA_u32"
            };
            let bitmask_body = if bits == 128 {
                formatdoc! {r#"
                    let shifted = _mm_srai_epi16::<15>(a);
                    let packed = _mm_packs_epi16(shifted, shifted);
                    (_mm_movemask_epi8(packed) & 0xFF) as u32
                "#}
            } else {
                formatdoc! {r#"
                    let shifted = _mm256_srai_epi16::<15>(a);
                    let lo = _mm256_castsi256_si128(shifted);
                    let hi = _mm256_extracti128_si256::<1>(shifted);
                    let packed = _mm_packs_epi16(lo, hi);
                    (_mm_movemask_epi8(packed) as u32) & 0xFFFF
                "#}
            };
            formatdoc! {r#"

            // ====== Boolean ======

            {arcane}
            fn all_true(self, a: {inner}) -> bool {{
                (({p}_movemask_epi8(a) as u32) & {hi_bytes}) == {hi_bytes}
            }}

            {arcane}
            fn any_true(self, a: {inner}) -> bool {{
                (({p}_movemask_epi8(a) as u32) & {hi_bytes}) != 0
            }}

            {arcane}
            fn bitmask(self, a: {inner}) -> u32 {{
                {bitmask_body}
            }}
            "#}
        }
        32 => {
            let all_mask = if bits == 128 { "0xF" } else { "0xFF" };
            formatdoc! {r#"

            // ====== Boolean ======

            {arcane}
            fn all_true(self, a: {inner}) -> bool {{
                {p}_movemask_ps({p}_castsi{bits}_ps(a)) == {all_mask}
            }}

            {arcane}
            fn any_true(self, a: {inner}) -> bool {{
                {p}_movemask_ps({p}_castsi{bits}_ps(a)) != 0
            }}

            {arcane}
            fn bitmask(self, a: {inner}) -> u32 {{
                {p}_movemask_ps({p}_castsi{bits}_ps(a)) as u32
            }}
            "#}
        }
        64 => {
            let all_mask = if bits == 128 { "0x3" } else { "0xF" };
            formatdoc! {r#"

            // ====== Boolean ======

            {arcane}
            fn all_true(self, a: {inner}) -> bool {{
                {p}_movemask_pd({p}_castsi{bits}_pd(a)) == {all_mask}
            }}

            {arcane}
            fn any_true(self, a: {inner}) -> bool {{
                {p}_movemask_pd({p}_castsi{bits}_pd(a)) != 0
            }}

            {arcane}
            fn bitmask(self, a: {inner}) -> u32 {{
                {p}_movemask_pd({p}_castsi{bits}_pd(a)) as u32
            }}
            "#}
        }
        _ => unreachable!(),
    }
}

// ============================================================================
// Conversion & Bitcast Implementations
// ============================================================================

fn generate_x86_convert_impls() -> String {
    let arcane = super::backend_syntax::arcane("X64V3Token");
    let mut code = String::new();

    for &(lanes, p, bits, f_ty, i_ty, sat_comment) in &[
        (
            4,
            "_mm",
            128,
            "__m128",
            "__m128i",
            "// Issue #80 fixup: cvttps yields the i32::MIN sentinel for +overflow\n            // and NaN; patch those two classes (-overflow already saturates to MIN).",
        ),
        (
            8,
            "_mm256",
            256,
            "__m256",
            "__m256i",
            "// Issue #80 fixup — see the 128-bit impl.",
        ),
    ] {
        code.push_str(&formatdoc! {r#"

            #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
            impl F32x{lanes}Convert for archmage::X64V3Token {{
                {arcane}
                fn bitcast_f32_to_i32(self, a: {f_ty}) -> {i_ty} {{
                    {p}_castps_si{bits}(a)
                }}

                {arcane}
                fn bitcast_i32_to_f32(self, a: {i_ty}) -> {f_ty} {{
                    {p}_castsi{bits}_ps(a)
                }}

                {arcane}
                fn convert_f32_to_i32(self, a: {f_ty}) -> {i_ty} {{
                    {p}_cvttps_epi32(a)
                }}

                {sat_comment}
                {arcane}
                fn convert_f32_to_i32_saturating(self, a: {f_ty}) -> {i_ty} {{
                    let t = {p}_cvttps_epi32(a);
                    let big = {p}_cmp_ps::<_CMP_GE_OQ>(a, {p}_set1_ps(2_147_483_648.0));
                    let t = {p}_blendv_epi8(t, {p}_set1_epi32(i32::MAX), {p}_castps_si{bits}(big));
                    let nan = {p}_cmp_ps::<_CMP_UNORD_Q>(a, a);
                    {p}_andnot_si{bits}({p}_castps_si{bits}(nan), t)
                }}

                {arcane}
                fn convert_f32_to_i32_round(self, a: {f_ty}) -> {i_ty} {{
                    {p}_cvtps_epi32(a)
                }}

                {arcane}
                fn convert_i32_to_f32(self, a: {i_ty}) -> {f_ty} {{
                    {p}_cvtepi32_ps(a)
                }}
            }}
        "#});
    }

    for &(lanes, ty) in &[(4, "__m128i"), (8, "__m256i")] {
        code.push_str(&formatdoc! {r#"

            #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
            impl U32x{lanes}Bitcast for archmage::X64V3Token {{
                #[inline(always)]
                fn bitcast_u32_to_i32(self, a: {ty}) -> {ty} {{
                    a
                }}

                #[inline(always)]
                fn bitcast_i32_to_u32(self, a: {ty}) -> {ty} {{
                    a
                }}
            }}
        "#});
    }

    for &(lanes, p, bits, i_ty, f_ty) in &[
        (2, "_mm", 128, "__m128i", "__m128d"),
        (4, "_mm256", 256, "__m256i", "__m256d"),
    ] {
        code.push_str(&formatdoc! {r#"

            #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
            impl I64x{lanes}Bitcast for archmage::X64V3Token {{
                {arcane}
                fn bitcast_i64_to_f64(self, a: {i_ty}) -> {f_ty} {{
                    {p}_castsi{bits}_pd(a)
                }}

                {arcane}
                fn bitcast_f64_to_i64(self, a: {f_ty}) -> {i_ty} {{
                    {p}_castpd_si{bits}(a)
                }}
            }}
        "#});
    }

    // Identity integer signed↔unsigned bitcasts
    for &(tr, from_e, to_e, ty) in &[
        ("I8x16Bitcast", "i8", "u8", "__m128i"),
        ("I8x32Bitcast", "i8", "u8", "__m256i"),
        ("I16x8Bitcast", "i16", "u16", "__m128i"),
        ("I16x16Bitcast", "i16", "u16", "__m256i"),
        ("U64x2Bitcast", "u64", "i64", "__m128i"),
        ("U64x4Bitcast", "u64", "i64", "__m256i"),
    ] {
        code.push_str(&formatdoc! {r#"

            #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
            impl {tr} for archmage::X64V3Token {{
                #[inline(always)]
                fn bitcast_{from_e}_to_{to_e}(self, a: {ty}) -> {ty} {{ a }}
                #[inline(always)]
                fn bitcast_{to_e}_to_{from_e}(self, a: {ty}) -> {ty} {{ a }}
            }}
        "#});
    }

    code
}
