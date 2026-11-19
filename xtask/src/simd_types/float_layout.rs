//! Raw u16 conversion and RGBA lane motion for the generic f32 backends.

use indoc::formatdoc;

pub(super) fn trait_methods(elem: &str, lanes: usize, backend: &str) -> String {
    if elem != "f32" {
        return String::new();
    }
    let transpose = if lanes == 4 {
        formatdoc! {r#"
            /// Transpose four vectors without changing any lane bits.
            #[doc(hidden)]
            #[inline(always)]
            fn transpose_4x4_repr(self, rows: [Self::Repr; 4]) -> [Self::Repr; 4] {{
                let rows = rows.map(|r| <Self as {backend}>::to_array(self, r));
                core::array::from_fn(|col| <Self as {backend}>::from_array(self, rows.map(|r| r[col])))
            }}
        "#}
    } else {
        String::new()
    };
    let rgba_pack = if lanes <= 8 {
        formatdoc! {r#"
            /// Quantize and pack planar RGBA with the encoded integer policy.
            #[doc(hidden)]
            #[inline(always)]
            fn store_rgba_u8_with_into<const CFG: u32>(self, r: Self::Repr, g: Self::Repr, b: Self::Repr, a: Self::Repr, out: &mut [u8; {rgba_bytes}]) {{
                let channels = [r, g, b, a].map(|v| <Self as {backend}>::to_u8_lanes::<CFG>(self, v));
                for (i, byte) in out.iter_mut().enumerate() {{
                    *byte = channels[i % 4][i / 4];
                }}
            }}
        "#, rgba_bytes = lanes * 4}
    } else {
        String::new()
    };
    let wide = if lanes == 16 {
        formatdoc! {r#"
            /// Widen sixteen consecutive bytes to float.
            #[doc(hidden)]
            #[inline(always)]
            fn from_u8_lanes_wide(self, lanes: &[u8; 16]) -> Self::Repr {{
                <Self as {backend}>::from_array(self, lanes.map(f32::from))
            }}

            /// Deinterleave sixteen RGBA pixels into planar float storage.
            #[doc(hidden)]
            #[inline(always)]
            fn deinterleave_rgba_wide(self, rgba: [Self::Repr; 4]) -> [Self::Repr; 4] {{
                let pixels = rgba.map(|r| <Self as {backend}>::to_array(self, r));
                core::array::from_fn(|c| {{
                    <Self as {backend}>::from_array(self, core::array::from_fn(|i| pixels[i / 4][4 * (i % 4) + c]))
                }})
            }}

            /// Interleave sixteen planar float pixels into RGBA storage.
            #[doc(hidden)]
            #[inline(always)]
            fn interleave_rgba_wide(self, channels: [Self::Repr; 4]) -> [Self::Repr; 4] {{
                let channels = channels.map(|r| <Self as {backend}>::to_array(self, r));
                core::array::from_fn(|group| {{
                    <Self as {backend}>::from_array(self, core::array::from_fn(|i| channels[i % 4][4 * group + i / 4]))
                }})
            }}
        "#}
    } else {
        String::new()
    };
    formatdoc! {r#"
        /// Load unsigned lanes and convert them exactly, without normalization.
        #[doc(hidden)]
        #[inline(always)]
        fn from_u16_lanes(self, lanes: &[u16; {lanes}]) -> Self::Repr {{
            <Self as {backend}>::from_array(self, lanes.map(f32::from))
        }}

        /// Convert lanes to u16 using the encoded policy.
        #[doc(hidden)]
        #[inline(always)]
        fn to_u16_lanes<const CFG: u32>(self, value: Self::Repr) -> [u16; {lanes}] {{
            <Self as {backend}>::to_array(self, value).map(crate::scalar::to_u16_with::<CFG>)
        }}

        /// Convert lanes to u8 using the encoded policy.
        #[doc(hidden)]
        #[inline(always)]
        fn to_u8_lanes<const CFG: u32>(self, value: Self::Repr) -> [u8; {lanes}] {{
            <Self as {backend}>::to_array(self, value).map(crate::scalar::to_u8_with::<CFG>)
        }}

        /// Convert f32 lanes to unsigned u32 stored in the float representation.
        #[doc(hidden)]
        #[inline(always)]
        fn to_u32_lanes<const CFG: u32>(self, value: Self::Repr) -> Self::Repr {{
            let lanes = <Self as {backend}>::to_array(self, value);
            <Self as {backend}>::from_array(self, lanes.map(|x| f32::from_bits(crate::scalar::to_u32_with::<CFG>(x))))
        }}

        /// Widen unsigned u32 bit patterns stored in the float representation.
        #[doc(hidden)]
        #[inline(always)]
        fn from_u32_lanes(self, value: Self::Repr) -> Self::Repr {{
            let lanes = <Self as {backend}>::to_array(self, value);
            <Self as {backend}>::from_array(self, lanes.map(|x| x.to_bits() as f32))
        }}

        /// Decode binary16 using the encoded policy.
        #[doc(hidden)]
        #[inline(always)]
        fn from_f16_lanes<const CFG: u32>(self, lanes: &[u16; {lanes}]) -> Self::Repr {{
            <Self as {backend}>::from_array(self, lanes.map(crate::scalar::from_f16_with::<CFG>))
        }}

        /// Encode binary16 using the encoded policy.
        #[doc(hidden)]
        #[inline(always)]
        fn to_f16_lanes<const CFG: u32>(self, value: Self::Repr) -> [u16; {lanes}] {{
            <Self as {backend}>::to_array(self, value).map(crate::scalar::to_f16_bits_with::<CFG>)
        }}

        /// Widen bfloat16 bit patterns.
        #[doc(hidden)]
        #[inline(always)]
        fn from_bf16_lanes(self, lanes: &[u16; {lanes}]) -> Self::Repr {{
            <Self as {backend}>::from_array(self, lanes.map(crate::scalar::from_bf16))
        }}

        /// Encode bfloat16 using the encoded policy.
        #[doc(hidden)]
        #[inline(always)]
        fn to_bf16_lanes<const CFG: u32>(self, value: Self::Repr) -> [u16; {lanes}] {{
            <Self as {backend}>::to_array(self, value).map(crate::scalar::to_bf16_with::<CFG>)
        }}

        /// Broadcast lane 3 in each consecutive group of four lanes.
        #[doc(hidden)]
        #[inline(always)]
        fn broadcast_rgba_alpha_repr(self, value: Self::Repr) -> Self::Repr {{
            let value = <Self as {backend}>::to_array(self, value);
            <Self as {backend}>::from_array(self, core::array::from_fn(|i| value[i | 3]))
        }}

        /// Replace lane 3 in each consecutive group of four lanes.
        #[doc(hidden)]
        #[inline(always)]
        fn with_rgba_alpha_repr(self, value: Self::Repr, src: Self::Repr) -> Self::Repr {{
            let mut value = <Self as {backend}>::to_array(self, value);
            let src = <Self as {backend}>::to_array(self, src);
            for i in (3..{lanes}).step_by(4) {{
                value[i] = src[i];
            }}
            <Self as {backend}>::from_array(self, value)
        }}

        {transpose}
        {rgba_pack}
        {wide}
    "#}
}

pub(super) fn x86(lanes: usize, repr: &str, attr: &str) -> String {
    let (load, convert, broadcast, replace) = match lanes {
        4 => (
            "_mm_cvtepu16_epi32(_mm_cvtsi64_si128(crate::simd_storage::cast(*lanes)))",
            "crate::simd_storage::cast::<_, [u16; 8]>(_mm_packus_epi32(i, i)).first_chunk::<4>().copied().unwrap()",
            "_mm_shuffle_ps::<255>(value, value)",
            "_mm_blend_ps::<8>(value, src)",
        ),
        8 => (
            "_mm256_cvtepu16_epi32(crate::simd_storage::cast(*lanes))",
            "crate::simd_storage::cast(_mm_packus_epi32(_mm256_castsi256_si128(i), _mm256_extracti128_si256::<1>(i)))",
            "_mm256_permute_ps::<255>(value)",
            "_mm256_blend_ps::<136>(value, src)",
        ),
        16 => (
            "_mm512_cvtepu16_epi32(crate::simd_storage::cast(*lanes))",
            "crate::simd_storage::cast(_mm512_cvtusepi32_epi16(_mm512_max_epi32(i, _mm512_setzero_si512())))",
            "_mm512_permute_ps::<255>(value)",
            "_mm512_mask_blend_ps(0x8888, value, src)",
        ),
        _ => unreachable!(),
    };
    let prefix = match lanes {
        4 => "_mm",
        8 => "_mm256",
        _ => "_mm512",
    };
    let byte_pack = match lanes {
        4 => {
            "_mm_cvtsi128_si32(_mm_packus_epi16(_mm_packs_epi32(i, i), _mm_setzero_si128())) as u32"
        }
        8 => {
            "_mm_cvtsi128_si64(_mm_packus_epi16(_mm_packs_epi32(_mm256_castsi256_si128(i), _mm256_extracti128_si256::<1>(i)), _mm_setzero_si128())) as u64"
        }
        _ => "_mm512_cvtusepi32_epi8(_mm512_max_epi32(i, _mm512_setzero_si512()))",
    };
    let half_prefix = if lanes == 16 { "_mm256" } else { "_mm" };
    let load_half = match lanes {
        4 => "_mm_cvtsi64_si128(crate::simd_storage::cast(*lanes))",
        _ => "crate::simd_storage::cast(*lanes)",
    };
    let extract_half = if lanes == 4 {
        "_mm_cvtsi128_si64(h) as u64"
    } else {
        "h"
    };
    let wide = if lanes == 16 {
        x86_rgba_wide(attr)
    } else {
        String::new()
    };
    let unsigned = x86_unsigned(lanes, prefix, repr, attr);
    let transpose = if lanes == 4 {
        formatdoc! {r#"
            {attr}
            fn transpose_4x4_repr(self, rows: [{repr}; 4]) -> [{repr}; 4] {{
                let [a, b, c, d] = rows;
                let ab_lo = _mm_unpacklo_ps(a, b);
                let ab_hi = _mm_unpackhi_ps(a, b);
                let cd_lo = _mm_unpacklo_ps(c, d);
                let cd_hi = _mm_unpackhi_ps(c, d);
                [
                    _mm_movelh_ps(ab_lo, cd_lo),
                    _mm_movehl_ps(cd_lo, ab_lo),
                    _mm_movelh_ps(ab_hi, cd_hi),
                    _mm_movehl_ps(cd_hi, ab_hi),
                ]
            }}
        "#}
    } else {
        String::new()
    };
    let bits = lanes * 32;
    let cast_to_float = format!("{prefix}_castsi{bits}_ps");
    let cast_to_int = format!("{prefix}_castps_si{bits}");
    let half_bits = if lanes == 16 { 256 } else { 128 };
    let wide_unsigned_words = if lanes == 16 {
        formatdoc! {r#"
            if opts.nonnegative() {{
                let i = if opts.rounds() {{ _mm512_cvtps_epu32(value) }} else {{ _mm512_cvttps_epu32(value) }};
                return crate::simd_storage::cast(_mm512_cvtusepi32_epi16(i));
            }}
        "#}
    } else {
        String::new()
    };
    let wide_unsigned_bytes = if lanes == 16 {
        formatdoc! {r#"
            if opts.nonnegative() {{
                let i = if opts.rounds() {{ _mm512_cvtps_epu32(value) }} else {{ _mm512_cvttps_epu32(value) }};
                return crate::simd_storage::cast(_mm512_cvtusepi32_epi8(i));
            }}
        "#}
    } else {
        String::new()
    };
    formatdoc! {r#"
        {attr}
        fn from_u16_lanes(self, lanes: &[u16; {lanes}]) -> {repr} {{
            {prefix}_cvtepi32_ps({load})
        }}

        {attr}
        fn to_u16_lanes<const CFG: u32>(self, value: {repr}) -> [u16; {lanes}] {{
            let opts = const {{ crate::numerics::IntConvert::decode(CFG) }};
            let value = if opts.saturates_all() {{
                {prefix}_min_ps({prefix}_set1_ps(65535.0), value)
            }} else {{
                value
            }};
            {wide_unsigned_words}
            let i = if opts.rounds() {{ {prefix}_cvtps_epi32(value) }} else {{ {prefix}_cvttps_epi32(value) }};
            {convert}
        }}

        {attr}
        fn to_u8_lanes<const CFG: u32>(self, value: {repr}) -> [u8; {lanes}] {{
            let opts = const {{ crate::numerics::IntConvert::decode(CFG) }};
            let value = if opts.saturates_all() {{
                {prefix}_min_ps({prefix}_set1_ps(255.0), value)
            }} else {{
                value
            }};
            {wide_unsigned_bytes}let i = if opts.rounds() {{ {prefix}_cvtps_epi32(value) }} else {{ {prefix}_cvttps_epi32(value) }};
            crate::simd_storage::cast({byte_pack})
        }}

        {unsigned}

        {attr}
        fn from_f16_lanes<const CFG: u32>(self, lanes: &[u16; {lanes}]) -> {repr} {{
            let opts = const {{ crate::numerics::HalfConvert::decode(CFG) }};
            let h = {load_half};
            let h = if opts.flushes() {{
                let mag = {half_prefix}_and_si{half_bits}(h, {half_prefix}_set1_epi16(0x7fff));
                let small = {half_prefix}_cmpgt_epi16({half_prefix}_set1_epi16(0x400), mag);
                {half_prefix}_andnot_si{half_bits}({half_prefix}_and_si{half_bits}(small, {half_prefix}_set1_epi16(0x7fff)), h)
            }} else {{ h }};
            {prefix}_cvtph_ps(h)
        }}

        {attr}
        fn to_f16_lanes<const CFG: u32>(self, value: {repr}) -> [u16; {lanes}] {{
            let opts = const {{ crate::numerics::HalfConvert::decode(CFG) }};
            let h = if opts.rounds() {{ {prefix}_cvtps_ph::<0>(value) }} else {{ {prefix}_cvtps_ph::<3>(value) }};
            let h = if opts.flushes() {{
                let mag = {half_prefix}_and_si{half_bits}(h, {half_prefix}_set1_epi16(0x7fff));
                let small = {half_prefix}_cmpgt_epi16({half_prefix}_set1_epi16(0x400), mag);
                {half_prefix}_andnot_si{half_bits}({half_prefix}_and_si{half_bits}(small, {half_prefix}_set1_epi16(0x7fff)), h)
            }} else {{ h }};
            crate::simd_storage::cast({extract_half})
        }}

        {attr}
        fn from_bf16_lanes(self, lanes: &[u16; {lanes}]) -> {repr} {{
            {cast_to_float}({prefix}_slli_epi32::<16>({load}))
        }}

        {attr}
        fn to_bf16_lanes<const CFG: u32>(self, value: {repr}) -> [u16; {lanes}] {{
            let opts = const {{ crate::numerics::HalfConvert::decode(CFG) }};
            let raw = {cast_to_int}(value);
            let rounded = if opts.rounds() {{
                let lsb = {prefix}_and_si{bits}({prefix}_srli_epi32::<16>(raw), {prefix}_set1_epi32(1));
                {prefix}_add_epi32(raw, {prefix}_add_epi32(lsb, {prefix}_set1_epi32(0x7fff)))
            }} else {{ raw }};
            let i = {prefix}_srli_epi32::<16>(rounded);
            let out: [u16; {lanes}] = {convert};
            if opts.is_finite() && !opts.flushes() {{
                out
            }} else {{
                // Special and signed-subnormal handling is shared with scalar.
                let original: [f32; {lanes}] = crate::simd_storage::cast(value);
                core::array::from_fn(|i| {{
                    if original[i].is_nan() || (opts.flushes() && out[i] & 0x7f80 == 0) {{
                        crate::scalar::to_bf16_with::<CFG>(original[i])
                    }} else {{ out[i] }}
                }})
            }}
        }}

        {wide}

        {attr}
        fn broadcast_rgba_alpha_repr(self, value: {repr}) -> {repr} {{
            {broadcast}
        }}

        {attr}
        fn with_rgba_alpha_repr(self, value: {repr}, src: {repr}) -> {repr} {{
            {replace}
        }}

        {transpose}
    "#}
}

pub(super) fn neon(attr: &str) -> String {
    let neon_encode_soft = formatdoc! {r#"
        let v = crate::simd::generic::f32x4::from_repr_unchecked(self, value);
        let i = crate::simd::generic::half4::narrow_f16::<CFG, _>(v).into_repr();
        crate::simd_storage::cast(vmovn_u32(vreinterpretq_u32_s32(i)))
    "#};
    formatdoc! {r#"
        {attr}
        fn from_u16_lanes(self, lanes: &[u16; 4]) -> float32x4_t {{
            vcvtq_f32_u32(vmovl_u16(crate::simd_storage::cast(*lanes)))
        }}

        {attr}
        fn to_u16_lanes<const CFG: u32>(self, value: float32x4_t) -> [u16; 4] {{
            let i = if const {{ crate::numerics::IntConvert::decode(CFG).rounds() }} {{ vcvtnq_u32_f32(value) }} else {{ vcvtq_u32_f32(value) }};
            crate::simd_storage::cast(vqmovn_u32(i))
        }}

        {attr}
        fn to_u8_lanes<const CFG: u32>(self, value: float32x4_t) -> [u8; 4] {{
            let i = if const {{ crate::numerics::IntConvert::decode(CFG).rounds() }} {{ vcvtnq_u32_f32(value) }} else {{ vcvtq_u32_f32(value) }};
            let words = vqmovn_u32(i);
            crate::simd_storage::cast(vget_lane_u32::<0>(vreinterpret_u32_u8(vqmovn_u16(vcombine_u16(words, words)))))
        }}

        {attr}
        fn store_rgba_u8_with_into<const CFG: u32>(self, r: float32x4_t, g: float32x4_t, b: float32x4_t, a: float32x4_t, out: &mut [u8; 16]) {{
            let opts = const {{ crate::numerics::IntConvert::decode(CFG) }};
            if cfg!(target_endian = "little") && opts.is_in_range() {{
                let convert = |x| if opts.rounds() {{ vcvtnq_u32_f32(x) }} else {{ vcvtq_u32_f32(x) }};
                let rg = vsliq_n_u32::<8>(convert(r), convert(g));
                let rgb = vsliq_n_u32::<16>(rg, convert(b));
                crate::simd_storage::store(vsliq_n_u32::<24>(rgb, convert(a)), out);
                return;
            }}
            let channels = [r, g, b, a].map(|v| <Self as F32x4Backend>::to_u8_lanes::<CFG>(self, v));
            for (i, byte) in out.iter_mut().enumerate() {{
                *byte = channels[i % 4][i / 4];
            }}
        }}

        {attr}
        fn to_u32_lanes<const CFG: u32>(self, value: float32x4_t) -> float32x4_t {{
            vreinterpretq_f32_u32(if const {{ crate::numerics::IntConvert::decode(CFG).rounds() }} {{ vcvtnq_u32_f32(value) }} else {{ vcvtq_u32_f32(value) }})
        }}

        {attr}
        fn from_u32_lanes(self, value: float32x4_t) -> float32x4_t {{
            vcvtq_f32_u32(vreinterpretq_u32_f32(value))
        }}

        #[rustversion::since(1.94)]
        #[allow(clippy::incompatible_msrv)]
        {attr}
        fn from_f16_lanes<const CFG: u32>(self, lanes: &[u16; 4]) -> float32x4_t {{
            let words: uint16x4_t = crate::simd_storage::cast(*lanes);
            let words = if const {{ crate::numerics::HalfConvert::decode(CFG).flushes() }} {{
                let mag = vand_u16(words, vdup_n_u16(0x7fff));
                vbsl_u16(vclt_u16(mag, vdup_n_u16(0x400)), vand_u16(words, vdup_n_u16(0x8000)), words)
            }} else {{
                words
            }};
            vcvt_f32_f16(vreinterpret_f16_u16(words))
        }}

        #[rustversion::before(1.94)]
        {attr}
        fn from_f16_lanes<const CFG: u32>(self, lanes: &[u16; 4]) -> float32x4_t {{
            let words = vreinterpretq_s32_u32(vmovl_u16(crate::simd_storage::cast(*lanes)));
            crate::simd::generic::half4::widen_f16::<CFG, _>(
                crate::simd::generic::i32x4::from_repr_unchecked(self, words)
            ).into_repr()
        }}

        #[rustversion::since(1.94)]
        #[allow(clippy::incompatible_msrv)]
        {attr}
        fn to_f16_lanes<const CFG: u32>(self, value: float32x4_t) -> [u16; 4] {{
            let opts = const {{ crate::numerics::HalfConvert::decode(CFG) }};
            if opts.rounds() {{
                let h = vreinterpret_u16_f16(vcvt_f16_f32(value));
                let words = if opts.flushes() {{
                    let mag = vand_u16(h, vdup_n_u16(0x7fff));
                    vbsl_u16(vclt_u16(mag, vdup_n_u16(0x400)), vand_u16(h, vdup_n_u16(0x8000)), h)
                }} else {{
                    h
                }};
                crate::simd_storage::cast(words)
            }} else {{
                {neon_encode_soft}
            }}
        }}

        #[rustversion::before(1.94)]
        {attr}
        fn to_f16_lanes<const CFG: u32>(self, value: float32x4_t) -> [u16; 4] {{
            {neon_encode_soft}
        }}

        {attr}
        fn to_bf16_lanes<const CFG: u32>(self, value: float32x4_t) -> [u16; 4] {{
            let v = crate::simd::generic::f32x4::from_repr_unchecked(self, value);
            let i = crate::simd::generic::half4::narrow_bf16::<CFG, _>(v).into_repr();
            crate::simd_storage::cast(vmovn_u32(vreinterpretq_u32_s32(i)))
        }}

        {attr}
        fn from_bf16_lanes(self, lanes: &[u16; 4]) -> float32x4_t {{
            vreinterpretq_f32_u32(vshll_n_u16::<16>(crate::simd_storage::cast(*lanes)))
        }}

        {attr}
        fn broadcast_rgba_alpha_repr(self, value: float32x4_t) -> float32x4_t {{
            vdupq_laneq_f32::<3>(value)
        }}

        {attr}
        fn with_rgba_alpha_repr(self, value: float32x4_t, src: float32x4_t) -> float32x4_t {{
            vcopyq_laneq_f32::<3, 3>(value, src)
        }}

        {attr}
        fn transpose_4x4_repr(self, rows: [float32x4_t; 4]) -> [float32x4_t; 4] {{
            let [a, b, c, d] = rows;
            let ab_lo = vzip1q_f32(a, b);
            let ab_hi = vzip2q_f32(a, b);
            let cd_lo = vzip1q_f32(c, d);
            let cd_hi = vzip2q_f32(c, d);
            [
                vcombine_f32(vget_low_f32(ab_lo), vget_low_f32(cd_lo)),
                vcombine_f32(vget_high_f32(ab_lo), vget_high_f32(cd_lo)),
                vcombine_f32(vget_low_f32(ab_hi), vget_low_f32(cd_hi)),
                vcombine_f32(vget_high_f32(ab_hi), vget_high_f32(cd_hi)),
            ]
        }}
    "#}
}

pub(super) fn wasm() -> String {
    formatdoc! {r#"
        #[inline(always)]
        fn from_u16_lanes(self, lanes: &[u16; 4]) -> v128 {{
            let words = u64x2(crate::simd_storage::cast(*lanes), 0);
            f32x4_convert_u32x4(u32x4_extend_low_u16x8(words))
        }}

        #[inline(always)]
        fn to_u16_lanes<const CFG: u32>(self, value: v128) -> [u16; 4] {{
            let opts = const {{ crate::numerics::IntConvert::decode(CFG) }};
            let a = if opts.rounds() {{ f32x4_nearest(value) }} else {{ value }};
            let i = if const {{ crate::numerics::IntConvert::decode(CFG).is_in_range() }} {{ crate::wasm_fma::trunc_f32x4_fast(self, a) }} else {{ i32x4_trunc_sat_f32x4(a) }};
            crate::simd_storage::cast(u64x2_extract_lane::<0>(u16x8_narrow_i32x4(i, i)))
        }}

        #[inline(always)]
        fn to_u8_lanes<const CFG: u32>(self, value: v128) -> [u8; 4] {{
            let opts = const {{ crate::numerics::IntConvert::decode(CFG) }};
            let a = if opts.rounds() {{ f32x4_nearest(value) }} else {{ value }};
            let i = if opts.is_in_range() {{ crate::wasm_fma::trunc_f32x4_fast(self, a) }} else {{ i32x4_trunc_sat_f32x4(a) }};
            let n = i16x8_narrow_i32x4(i, i);
            crate::simd_storage::cast(i32x4_extract_lane::<0>(u8x16_narrow_i16x8(n, n)))
        }}

        #[inline(always)]
        fn to_u32_lanes<const CFG: u32>(self, value: v128) -> v128 {{
            let opts = const {{ crate::numerics::IntConvert::decode(CFG) }};
            let a = if opts.rounds() {{ f32x4_nearest(value) }} else {{ value }};
            #[cfg(target_feature = "relaxed-simd")]
            if opts.is_in_range() {{
                return crate::wasm_fma::trunc_u32_f32x4_fast(self, a);
            }}
            u32x4_trunc_sat_f32x4(a)
        }}

        #[inline(always)]
        fn from_u32_lanes(self, value: v128) -> v128 {{
            f32x4_convert_u32x4(value)
        }}

        #[inline(always)]
        fn from_f16_lanes<const CFG: u32>(self, lanes: &[u16; 4]) -> v128 {{
            let words = u64x2(crate::simd_storage::cast(*lanes), 0);
            crate::simd::generic::half4::widen_f16::<CFG, _>(
                crate::simd::generic::i32x4::from_repr_unchecked(self, u32x4_extend_low_u16x8(words))
            ).into_repr()
        }}

        #[inline(always)]
        fn to_f16_lanes<const CFG: u32>(self, value: v128) -> [u16; 4] {{
            let v = crate::simd::generic::f32x4::from_repr_unchecked(self, value);
            let i = crate::simd::generic::half4::narrow_f16::<CFG, _>(v).into_repr();
            crate::simd_storage::cast(u64x2_extract_lane::<0>(u16x8_narrow_i32x4(i, i)))
        }}

        #[inline(always)]
        fn to_bf16_lanes<const CFG: u32>(self, value: v128) -> [u16; 4] {{
            let v = crate::simd::generic::f32x4::from_repr_unchecked(self, value);
            let i = crate::simd::generic::half4::narrow_bf16::<CFG, _>(v).into_repr();
            crate::simd_storage::cast(u64x2_extract_lane::<0>(u16x8_narrow_i32x4(i, i)))
        }}

        #[inline(always)]
        fn from_bf16_lanes(self, lanes: &[u16; 4]) -> v128 {{
            let words = u64x2(crate::simd_storage::cast(*lanes), 0);
            i32x4_shl(u32x4_extend_low_u16x8(words), 16)
        }}

        #[inline(always)]
        fn broadcast_rgba_alpha_repr(self, value: v128) -> v128 {{
            i32x4_shuffle::<3, 3, 3, 3>(value, value)
        }}

        #[inline(always)]
        fn with_rgba_alpha_repr(self, value: v128, src: v128) -> v128 {{
            i32x4_shuffle::<0, 1, 2, 7>(value, src)
        }}

        #[inline(always)]
        fn transpose_4x4_repr(self, rows: [v128; 4]) -> [v128; 4] {{
            let [a, b, c, d] = rows;
            let ab_lo = i32x4_shuffle::<0, 4, 1, 5>(a, b);
            let ab_hi = i32x4_shuffle::<2, 6, 3, 7>(a, b);
            let cd_lo = i32x4_shuffle::<0, 4, 1, 5>(c, d);
            let cd_hi = i32x4_shuffle::<2, 6, 3, 7>(c, d);
            [
                i64x2_shuffle::<0, 2>(ab_lo, cd_lo),
                i64x2_shuffle::<1, 3>(ab_lo, cd_lo),
                i64x2_shuffle::<0, 2>(ab_hi, cd_hi),
                i64x2_shuffle::<1, 3>(ab_hi, cd_hi),
            ]
        }}
    "#}
}

pub(super) fn polyfill(lanes: usize, part: usize, repr: &str) -> String {
    let backend = format!("F32x{part}Backend");
    let wide = if lanes == 16 {
        polyfill_rgba_wide(part, repr)
    } else {
        String::new()
    };
    let pack_words = if part == 8 {
        let attr = super::backend_syntax::arcane("X64V3Token");
        formatdoc! {r#"
            {attr}
            fn to_u16_lanes<const CFG: u32>(self, value: {repr}) -> [u16; {lanes}] {{
                let opts = const {{ crate::numerics::IntConvert::decode(CFG) }};
                let convert = |v| {{
                    let v = if opts.saturates_all() {{ _mm256_min_ps(_mm256_set1_ps(65535.0), v) }} else {{ v }};
                    if opts.rounds() {{ _mm256_cvtps_epi32(v) }} else {{ _mm256_cvttps_epi32(v) }}
                }};
                let packed = _mm256_packus_epi32(convert(value[0]), convert(value[1]));
                crate::simd_storage::cast(_mm256_permute4x64_epi64::<0xd8>(packed))
            }}
        "#}
    } else {
        formatdoc! {r#"
            #[inline(always)]
            fn to_u16_lanes<const CFG: u32>(self, value: {repr}) -> [u16; {lanes}] {{
                crate::simd_storage::cast(value.map(|v| <Self as {backend}>::to_u16_lanes::<CFG>(self, v)))
            }}
        "#}
    };
    formatdoc! {r#"
        #[inline(always)]
        fn from_u16_lanes(self, lanes: &[u16; {lanes}]) -> {repr} {{
            let (chunks, _) = lanes.as_chunks::<{part}>();
            core::array::from_fn(|i| <Self as {backend}>::from_u16_lanes(self, &chunks[i]))
        }}

        {pack_words}

        #[inline(always)]
        fn to_u8_lanes<const CFG: u32>(self, value: {repr}) -> [u8; {lanes}] {{
            let parts = value.map(|v| <Self as {backend}>::to_u8_lanes::<CFG>(self, v));
            crate::simd_storage::cast(parts)
        }}

        #[inline(always)]
        fn to_u32_lanes<const CFG: u32>(self, value: {repr}) -> {repr} {{
            value.map(|part| <Self as {backend}>::to_u32_lanes::<CFG>(self, part))
        }}

        #[inline(always)]
        fn from_u32_lanes(self, value: {repr}) -> {repr} {{
            value.map(|part| <Self as {backend}>::from_u32_lanes(self, part))
        }}

        #[inline(always)]
        fn from_f16_lanes<const CFG: u32>(self, lanes: &[u16; {lanes}]) -> {repr} {{
            let (chunks, _) = lanes.as_chunks::<{part}>();
            core::array::from_fn(|i| <Self as {backend}>::from_f16_lanes::<CFG>(self, &chunks[i]))
        }}

        #[inline(always)]
        fn to_f16_lanes<const CFG: u32>(self, value: {repr}) -> [u16; {lanes}] {{
            crate::simd_storage::cast(value.map(|part| <Self as {backend}>::to_f16_lanes::<CFG>(self, part)))
        }}

        #[inline(always)]
        fn from_bf16_lanes(self, lanes: &[u16; {lanes}]) -> {repr} {{
            let (chunks, _) = lanes.as_chunks::<{part}>();
            core::array::from_fn(|i| <Self as {backend}>::from_bf16_lanes(self, &chunks[i]))
        }}

        #[inline(always)]
        fn to_bf16_lanes<const CFG: u32>(self, value: {repr}) -> [u16; {lanes}] {{
            crate::simd_storage::cast(value.map(|part| <Self as {backend}>::to_bf16_lanes::<CFG>(self, part)))
        }}

        {wide}

        #[inline(always)]
        fn broadcast_rgba_alpha_repr(self, value: {repr}) -> {repr} {{
            value.map(|v| <Self as {backend}>::broadcast_rgba_alpha_repr(self, v))
        }}

        #[inline(always)]
        fn with_rgba_alpha_repr(self, value: {repr}, src: {repr}) -> {repr} {{
            core::array::from_fn(|i| <Self as {backend}>::with_rgba_alpha_repr(self, value[i], src[i]))
        }}
    "#}
}

fn x86_unsigned(lanes: usize, prefix: &str, repr: &str, attr: &str) -> String {
    let bits = lanes * 32;
    let to_int = if lanes == 16 {
        formatdoc! {r#"
            let clamped = if opts.nonnegative() {{ value }} else {{ _mm512_max_ps(value, _mm512_setzero_ps()) }};
            let i = if opts.rounds() {{ _mm512_cvtps_epu32(clamped) }} else {{ _mm512_cvttps_epu32(clamped) }};
            _mm512_castsi512_ps(i)
        "#}
    } else {
        formatdoc! {r#"
            let value = if opts.nonnegative() {{ value }} else {{ {prefix}_max_ps(value, {prefix}_setzero_ps()) }};
            if const {{ !crate::numerics::IntConvert::decode(CFG).is_in_range() && !crate::numerics::IntConvert::decode(CFG).saturates_all() }} {{
                let i = if opts.rounds() {{ {prefix}_cvtps_epi32(value) }} else {{ {prefix}_cvttps_epi32(value) }};
                return {prefix}_castsi{bits}_ps(i);
            }}
            let offset = {prefix}_set1_ps(2147483648.0);
            let hi = {prefix}_cmp_ps::<_CMP_GE_OQ>(value, offset);
            let adjusted = {prefix}_sub_ps(value, {prefix}_and_ps(hi, offset));
            let i = if opts.rounds() {{ {prefix}_cvtps_epi32(adjusted) }} else {{ {prefix}_cvttps_epi32(adjusted) }};
            let i = {prefix}_or_ps({prefix}_castsi{bits}_ps(i), {prefix}_and_ps(hi, {prefix}_set1_ps(-0.0)));
            if opts.saturates_all() {{
                let overflow = {prefix}_cmp_ps::<_CMP_GE_OQ>(value, {prefix}_set1_ps(4294967296.0));
                {prefix}_or_ps(i, overflow)
            }} else {{
                i
            }}
        "#}
    };
    let from_int = if lanes == 16 {
        "_mm512_cvtepu32_ps(_mm512_castps_si512(value))".to_string()
    } else {
        formatdoc! {r#"
            let i = {prefix}_castps_si{bits}(value);
            // Both halves widen exactly. FMA rounds the reconstructed u32 once.
            let hi = {prefix}_cvtepi32_ps({prefix}_srli_epi32::<16>(i));
            let lo = {prefix}_cvtepi32_ps({prefix}_and_si{bits}(i, {prefix}_set1_epi32(65535)));
            {prefix}_fmadd_ps(hi, {prefix}_set1_ps(65536.0), lo)
        "#}
    };
    formatdoc! {r#"
        {attr}
        fn to_u32_lanes<const CFG: u32>(self, value: {repr}) -> {repr} {{
            let opts = const {{ crate::numerics::IntConvert::decode(CFG) }};
            {to_int}
        }}

        {attr}
        fn from_u32_lanes(self, value: {repr}) -> {repr} {{
            {from_int}
        }}
    "#}
}

fn x86_rgba_wide(attr: &str) -> String {
    formatdoc! {r#"
        {attr}
        fn from_u8_lanes_wide(self, lanes: &[u8; 16]) -> __m512 {{
            _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(crate::simd_storage::cast(*lanes)))
        }}

        {attr}
        fn deinterleave_rgba_wide(self, rgba: [__m512; 4]) -> [__m512; 4] {{
            let [a, b, c, d] = rgba;
            let rg = _mm512_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28, 1, 5, 9, 13, 17, 21, 25, 29);
            let ba = _mm512_add_epi32(rg, _mm512_set1_epi32(2));
            let low = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
            let high = _mm512_add_epi32(low, _mm512_set1_epi32(8));
            let ab_rg = _mm512_permutex2var_ps(a, rg, b);
            let cd_rg = _mm512_permutex2var_ps(c, rg, d);
            let ab_ba = _mm512_permutex2var_ps(a, ba, b);
            let cd_ba = _mm512_permutex2var_ps(c, ba, d);
            [
                _mm512_permutex2var_ps(ab_rg, low, cd_rg),
                _mm512_permutex2var_ps(ab_rg, high, cd_rg),
                _mm512_permutex2var_ps(ab_ba, low, cd_ba),
                _mm512_permutex2var_ps(ab_ba, high, cd_ba),
            ]
        }}

        {attr}
        fn interleave_rgba_wide(self, channels: [__m512; 4]) -> [__m512; 4] {{
            let [r, g, b, a] = channels;
            let low = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
            let high = _mm512_add_epi32(low, _mm512_set1_epi32(8));
            let first = _mm512_setr_epi32(0, 1, 16, 17, 2, 3, 18, 19, 4, 5, 20, 21, 6, 7, 22, 23);
            let second = _mm512_add_epi32(first, _mm512_set1_epi32(8));
            let rg0 = _mm512_permutex2var_ps(r, low, g);
            let ba0 = _mm512_permutex2var_ps(b, low, a);
            let rg1 = _mm512_permutex2var_ps(r, high, g);
            let ba1 = _mm512_permutex2var_ps(b, high, a);
            [
                _mm512_permutex2var_ps(rg0, first, ba0),
                _mm512_permutex2var_ps(rg0, second, ba0),
                _mm512_permutex2var_ps(rg1, first, ba1),
                _mm512_permutex2var_ps(rg1, second, ba1),
            ]
        }}
    "#}
}

fn polyfill_rgba_wide(part: usize, repr: &str) -> String {
    let count = 16 / part;
    formatdoc! {r#"
        #[inline(always)]
        fn from_u8_lanes_wide(self, lanes: &[u8; 16]) -> {repr} {{
            use crate::simd::generic::f32x{part} as V;
            let (parts, _) = lanes.as_chunks::<{part}>();
            core::array::from_fn(|i| V::from_u8_t(self, &parts[i]).into_repr())
        }}

        #[inline(always)]
        fn deinterleave_rgba_wide(self, rgba: [{repr}; 4]) -> [{repr}; 4] {{
            use crate::simd::generic::f32x{part} as V;
            let groups: [[V<Self>; 4]; {count}] = core::array::from_fn(|i| {{
                V::deinterleave_4ch(core::array::from_fn(|j| {{
                    let block = 4 * i + j;
                    V::from_repr_unchecked(self, rgba[block / {count}][block % {count}])
                }}))
            }});
            core::array::from_fn(|c| core::array::from_fn(|i| groups[i][c].into_repr()))
        }}

        #[inline(always)]
        fn interleave_rgba_wide(self, channels: [{repr}; 4]) -> [{repr}; 4] {{
            use crate::simd::generic::f32x{part} as V;
            let groups: [[V<Self>; 4]; {count}] = core::array::from_fn(|i| {{
                V::interleave_4ch(channels.map(|c| V::from_repr_unchecked(self, c[i])))
            }});
            core::array::from_fn(|i| core::array::from_fn(|j| {{
                let block = i * {count} + j;
                groups[block / 4][block % 4].into_repr()
            }}))
        }}
    "#}
}
