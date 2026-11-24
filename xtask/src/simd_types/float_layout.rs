//! Raw u16 conversion and RGBA lane motion for the generic f32 backends.

use indoc::formatdoc;

pub(super) fn internal_trait(elem: &str, lanes: usize) -> String {
    if elem != "f32" {
        return String::new();
    }
    let internal_name = format!("F32x{lanes}Internal");
    let transpose = if lanes == 4 {
        formatdoc! {r#"
            /// Transpose four vectors without changing any lane bits.
            fn transpose_4x4_repr(self, rows: [Self::InternalRepr; 4]) -> [Self::InternalRepr; 4];
        "#}
    } else {
        String::new()
    };
    let rgb_and_rgba = if lanes <= 8 {
        formatdoc! {r#"
            /// Quantize and store planar channels into packed RGBA bytes.
            fn store_rgba_u8_into(
                self,
                r: Self::InternalRepr,
                g: Self::InternalRepr,
                b: Self::InternalRepr,
                a: Self::InternalRepr,
                out: &mut [u8; {rgba_bytes}],
            );

            /// Quantize and pack planar RGBA with the encoded integer policy.
            #[inline(always)]
            fn store_rgba_u8_with_into<const CFG: u32>(
                self,
                r: Self::InternalRepr,
                g: Self::InternalRepr,
                b: Self::InternalRepr,
                a: Self::InternalRepr,
                out: &mut [u8; {rgba_bytes}],
            ) {{
                let channels = [r, g, b, a].map(|v| Self::to_u8_lanes::<CFG>(self, v));
                for (i, byte) in out.iter_mut().enumerate() {{
                    *byte = channels[i % 4][i / 4];
                }}
            }}

            /// Deinterleave {lanes} packed RGB pixels from AoS into planar `[R, G, B]`.
            fn deinterleave_3ch_repr(self, rgb: [Self::InternalRepr; 3]) -> [Self::InternalRepr; 3];

            /// Interleave 3 planar `[R, G, B]` vectors into 3 AoS RGB vectors.
            fn interleave_3ch_repr(self, channels: [Self::InternalRepr; 3]) -> [Self::InternalRepr; 3];

            /// Load {lanes} packed RGB `u8` pixels into planar `f32` `(R, G, B)`.
            fn load_rgb_bytes(
                self,
                rgb: &[u8; {rgb_bytes}],
            ) -> (Self::InternalRepr, Self::InternalRepr, Self::InternalRepr);

            /// Round-to-nearest-even, clamp to `0..=255`, and interleave into {lanes} RGB pixels.
            fn store_rgb_u8_into(
                self,
                r: Self::InternalRepr,
                g: Self::InternalRepr,
                b: Self::InternalRepr,
                out: &mut [u8; {rgb_bytes}],
            );
        "#, rgba_bytes = lanes * 4, rgb_bytes = lanes * 3}
    } else {
        String::new()
    };
    let wide = if lanes == 16 {
        formatdoc! {r#"
            /// Widen sixteen consecutive bytes to float.
            fn from_u8_lanes_wide(self, lanes: &[u8; 16]) -> Self::InternalRepr;

            /// Deinterleave sixteen RGBA pixels into planar float storage.
            fn deinterleave_rgba_wide(self, rgba: [Self::InternalRepr; 4]) -> [Self::InternalRepr; 4];

            /// Interleave sixteen planar float pixels into RGBA storage.
            fn interleave_rgba_wide(self, channels: [Self::InternalRepr; 4]) -> [Self::InternalRepr; 4];

            /// Select lanes from the concatenated pair using the low five index bits.
            #[inline(always)]
            fn permute2_wide(self, pair: [Self::InternalRepr; 2], idx: [i32; 16]) -> Self::InternalRepr
            where
                Self: F32x16Backend,
            {{
                let lanes = pair.map(|v| self.to_array(v));
                self.from_array(idx.map(|i| {{
                    let lane = (i as usize) & 31;
                    lanes[lane / 16][lane % 16]
                }}))
            }}

            /// Transpose sixteen rows while preserving all lane bits.
            #[inline(always)]
            fn transpose_16x16_repr(self, rows: [Self::InternalRepr; 16]) -> [Self::InternalRepr; 16]
            where
                Self: F32x16Backend,
            {{
                use crate::simd::generic::f32x16;
                let (halves, _) = rows.as_chunks::<8>();
                let top = halves[0].map(|v| f32x16::from_repr_unchecked(self, v));
                let bottom = halves[1].map(|v| f32x16::from_repr_unchecked(self, v));
                let top = f32x16::transpose_dual_8x8_copy(top);
                let bottom = f32x16::transpose_dual_8x8_copy(bottom);
                const LO: [i32; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23];
                const HI: [i32; 16] = [8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31];
                [
                    top[0].permute2(bottom[0], LO),
                    top[1].permute2(bottom[1], LO),
                    top[2].permute2(bottom[2], LO),
                    top[3].permute2(bottom[3], LO),
                    top[4].permute2(bottom[4], LO),
                    top[5].permute2(bottom[5], LO),
                    top[6].permute2(bottom[6], LO),
                    top[7].permute2(bottom[7], LO),
                    top[0].permute2(bottom[0], HI),
                    top[1].permute2(bottom[1], HI),
                    top[2].permute2(bottom[2], HI),
                    top[3].permute2(bottom[3], HI),
                    top[4].permute2(bottom[4], HI),
                    top[5].permute2(bottom[5], HI),
                    top[6].permute2(bottom[6], HI),
                    top[7].permute2(bottom[7], HI),
                ].map(f32x16::into_repr)
            }}

            /// Deinterleave sixteen packed RGB pixels into three planes.
            #[inline(always)]
            fn deinterleave_3ch_repr(self, rgb: [Self::InternalRepr; 3]) -> [Self::InternalRepr; 3]
            where
                Self: F32x16Backend,
            {{
                [
                    self.permute_rgb_wide::<0, true>(rgb),
                    self.permute_rgb_wide::<1, true>(rgb),
                    self.permute_rgb_wide::<2, true>(rgb),
                ]
            }}

            /// Interleave three planes into sixteen packed RGB pixels.
            #[inline(always)]
            fn interleave_3ch_repr(self, channels: [Self::InternalRepr; 3]) -> [Self::InternalRepr; 3]
            where
                Self: F32x16Backend,
            {{
                [
                    self.permute_rgb_wide::<0, false>(channels),
                    self.permute_rgb_wide::<1, false>(channels),
                    self.permute_rgb_wide::<2, false>(channels),
                ]
            }}

            #[inline(always)]
            fn permute_rgb_wide<const BLOCK: usize, const DEINTERLEAVE: bool>(
                self, [a, b, c]: [Self::InternalRepr; 3],
            ) -> Self::InternalRepr
            where
                Self: F32x16Backend,
            {{
                let (first, second) = const {{
                    let mut first = [0; 16];
                    let mut second = [0; 16];
                    let mut lane = 0;
                    while lane < 16 {{
                        let packed = BLOCK * 16 + lane;
                        let src = if DEINTERLEAVE {{
                            lane * 3 + BLOCK
                        }} else {{
                            (packed % 3) * 16 + packed / 3
                        }};
                        // First gather lanes from a and b, then fill the remaining lanes from c.
                        if src < 32 {{
                            first[lane] = src as i32;
                            second[lane] = lane as i32;
                        }} else {{
                            second[lane] = (src - 16) as i32;
                        }}
                        lane += 1;
                    }}
                    (first, second)
                }};
                self.permute2_wide([self.permute2_wide([a, b], first), c], second)
            }}
        "#}
    } else {
        String::new()
    };
    formatdoc! {r#"
        /// Crate-private layout and conversion hooks for `{lanes}`-lane `f32` backends.
        pub(crate) trait {internal_name}: SimdToken + Sealed + Copy + 'static {{
            type InternalRepr: Copy + Clone + Send + Sync;

            /// Load unsigned lanes and convert them exactly, without normalization.
            fn from_u16_lanes(self, lanes: &[u16; {lanes}]) -> Self::InternalRepr;

            /// Convert lanes to u16 using the encoded policy.
            fn to_u16_lanes<const CFG: u32>(self, value: Self::InternalRepr) -> [u16; {lanes}];

            /// Convert lanes to u8 using the encoded policy.
            fn to_u8_lanes<const CFG: u32>(self, value: Self::InternalRepr) -> [u8; {lanes}];

            /// Convert f32 lanes to unsigned u32 stored in the float representation.
            fn to_u32_lanes<const CFG: u32>(self, value: Self::InternalRepr) -> Self::InternalRepr;

            /// Widen unsigned u32 bit patterns stored in the float representation.
            fn from_u32_lanes(self, value: Self::InternalRepr) -> Self::InternalRepr;

            /// Decode binary16 using the encoded policy.
            fn from_f16_lanes<const CFG: u32>(self, lanes: &[u16; {lanes}]) -> Self::InternalRepr;

            /// Encode binary16 using the encoded policy.
            fn to_f16_lanes<const CFG: u32>(self, value: Self::InternalRepr) -> [u16; {lanes}];

            /// Widen bfloat16 bit patterns.
            fn from_bf16_lanes(self, lanes: &[u16; {lanes}]) -> Self::InternalRepr;

            /// Encode bfloat16 using the encoded policy.
            fn to_bf16_lanes<const CFG: u32>(self, value: Self::InternalRepr) -> [u16; {lanes}];

            /// Broadcast lane 3 in each consecutive group of four lanes.
            fn broadcast_rgba_alpha_repr(self, value: Self::InternalRepr) -> Self::InternalRepr;

            /// Replace lane 3 in each consecutive group of four lanes.
            fn with_rgba_alpha_repr(
                self,
                value: Self::InternalRepr,
                src: Self::InternalRepr,
            ) -> Self::InternalRepr;

            {transpose}
            {rgb_and_rgba}
            {wide}
        }}

    "#}
}

pub(super) fn scalar_internal_impl(lanes: usize) -> String {
    let internal_name = format!("F32x{lanes}Internal");
    let backend_name = format!("F32x{lanes}Backend");
    let transpose = if lanes == 4 {
        formatdoc! {r#"
            #[inline(always)]
            fn transpose_4x4_repr(self, rows: [[f32; 4]; 4]) -> [[f32; 4]; 4] {{
                core::array::from_fn(|col| rows.map(|r| r[col]))
            }}
        "#}
    } else {
        String::new()
    };
    let rgb_and_rgba = if lanes <= 8 {
        let (deinterleave_body, interleave_body) = if lanes == 4 {
            (
                "[[v0[0], v0[3], v1[2], v2[1]], [v0[1], v1[0], v1[3], v2[2]], [v0[2], v1[1], v2[0], v2[3]]]",
                "[[r[0], g[0], b[0], r[1]], [g[1], b[1], r[2], g[2]], [b[2], r[3], g[3], b[3]]]",
            )
        } else {
            (
                "[[v0[0], v0[3], v0[6], v1[1], v1[4], v1[7], v2[2], v2[5]], [v0[1], v0[4], v0[7], v1[2], v1[5], v2[0], v2[3], v2[6]], [v0[2], v0[5], v1[0], v1[3], v1[6], v2[1], v2[4], v2[7]]]",
                "[[r[0], g[0], b[0], r[1], g[1], b[1], r[2], g[2]], [b[2], r[3], g[3], b[3], r[4], g[4], b[4], r[5]], [g[5], b[5], r[6], g[6], b[6], r[7], g[7], b[7]]]",
            )
        };
        formatdoc! {r#"
            #[inline(always)]
            fn store_rgba_u8_into(
                self,
                r: [f32; {lanes}],
                g: [f32; {lanes}],
                b: [f32; {lanes}],
                a: [f32; {lanes}],
                out: &mut [u8; {rgba_bytes}],
            ) {{
                let rb = <Self as {backend_name}>::to_u8_bytes(self, r);
                let gb = <Self as {backend_name}>::to_u8_bytes(self, g);
                let bb = <Self as {backend_name}>::to_u8_bytes(self, b);
                let ab = <Self as {backend_name}>::to_u8_bytes(self, a);
                for i in 0..{lanes} {{
                    out[i * 4] = rb[i];
                    out[i * 4 + 1] = gb[i];
                    out[i * 4 + 2] = bb[i];
                    out[i * 4 + 3] = ab[i];
                }}
            }}

            #[inline(always)]
            fn deinterleave_3ch_repr(self, rgb: [[f32; {lanes}]; 3]) -> [[f32; {lanes}]; 3] {{
                let [v0, v1, v2] = rgb;
                {deinterleave_body}
            }}

            #[inline(always)]
            fn interleave_3ch_repr(self, channels: [[f32; {lanes}]; 3]) -> [[f32; {lanes}]; 3] {{
                let [r, g, b] = channels;
                {interleave_body}
            }}

            #[inline(always)]
            fn load_rgb_bytes(
                self,
                rgb: &[u8; {rgb_bytes}],
            ) -> ([f32; {lanes}], [f32; {lanes}], [f32; {lanes}]) {{
                let (mut r, mut g, mut b) = ([0.0; {lanes}], [0.0; {lanes}], [0.0; {lanes}]);
                for (i, &[red, green, blue]) in rgb.as_chunks::<3>().0.iter().enumerate() {{
                    r[i] = f32::from(red);
                    g[i] = f32::from(green);
                    b[i] = f32::from(blue);
                }}
                (r, g, b)
            }}

            #[inline(always)]
            fn store_rgb_u8_into(
                self,
                r: [f32; {lanes}],
                g: [f32; {lanes}],
                b: [f32; {lanes}],
                out: &mut [u8; {rgb_bytes}],
            ) {{
                let [rb, gb, bb] = [r, g, b].map(|v| <Self as {backend_name}>::to_u8_bytes(self, v));
                for (i, pixel) in out.as_chunks_mut::<3>().0.iter_mut().enumerate() {{
                    *pixel = [rb[i], gb[i], bb[i]];
                }}
            }}
        "#, rgba_bytes = lanes * 4, rgb_bytes = lanes * 3}
    } else {
        String::new()
    };
    let wide = if lanes == 16 {
        formatdoc! {r#"
            #[inline]
            fn transpose_16x16_repr(self, rows: [[f32; 16]; 16]) -> [[f32; 16]; 16] {{
                let mut out = [[0.0; 16]; 16];
                for (i, row) in rows.iter().enumerate() {{
                    for (col, &lane) in out.iter_mut().zip(row) {{
                        col[i & 15] = lane;
                    }}
                }}
                out
            }}

            #[inline]
            fn deinterleave_3ch_repr(self, rgb: [[f32; 16]; 3]) -> [[f32; 16]; 3] {{
                let [mut r, mut g, mut b] = [[0.0; 16]; 3];
                for (i, &[red, green, blue]) in rgb.as_flattened().as_chunks::<3>().0.iter().enumerate() {{
                    r[i] = red;
                    g[i] = green;
                    b[i] = blue;
                }}
                [r, g, b]
            }}

            #[inline]
            fn interleave_3ch_repr(self, [r, g, b]: [[f32; 16]; 3]) -> [[f32; 16]; 3] {{
                let mut rgb = [[0.0; 16]; 3];
                for (i, pixel) in rgb.as_flattened_mut().as_chunks_mut::<3>().0.iter_mut().enumerate() {{
                    *pixel = [r[i], g[i], b[i]];
                }}
                rgb
            }}

            #[inline(always)]
            fn from_u8_lanes_wide(self, lanes: &[u8; 16]) -> [f32; 16] {{
                lanes.map(f32::from)
            }}

            #[inline(always)]
            fn deinterleave_rgba_wide(self, rgba: [[f32; 16]; 4]) -> [[f32; 16]; 4] {{
                core::array::from_fn(|c| core::array::from_fn(|i| rgba[i / 4][4 * (i % 4) + c]))
            }}

            #[inline(always)]
            fn interleave_rgba_wide(self, channels: [[f32; 16]; 4]) -> [[f32; 16]; 4] {{
                core::array::from_fn(|group| core::array::from_fn(|i| channels[i % 4][4 * group + i / 4]))
            }}
        "#}
    } else {
        String::new()
    };
    formatdoc! {r#"

        impl crate::simd::backends::{internal_name} for archmage::ScalarToken {{
            type InternalRepr = [f32; {lanes}];

            #[inline(always)]
            fn from_u16_lanes(self, lanes: &[u16; {lanes}]) -> [f32; {lanes}] {{
                lanes.map(f32::from)
            }}

            #[inline(always)]
            fn to_u16_lanes<const CFG: u32>(self, value: [f32; {lanes}]) -> [u16; {lanes}] {{
                value.map(crate::scalar::to_u16_with::<CFG>)
            }}

            #[inline(always)]
            fn to_u8_lanes<const CFG: u32>(self, value: [f32; {lanes}]) -> [u8; {lanes}] {{
                value.map(crate::scalar::to_u8_with::<CFG>)
            }}

            #[inline(always)]
            fn to_u32_lanes<const CFG: u32>(self, value: [f32; {lanes}]) -> [f32; {lanes}] {{
                value.map(|x| f32::from_bits(crate::scalar::to_u32_with::<CFG>(x)))
            }}

            #[inline(always)]
            fn from_u32_lanes(self, value: [f32; {lanes}]) -> [f32; {lanes}] {{
                value.map(|x| x.to_bits() as f32)
            }}

            #[inline(always)]
            fn from_f16_lanes<const CFG: u32>(self, lanes: &[u16; {lanes}]) -> [f32; {lanes}] {{
                lanes.map(crate::scalar::from_f16_with::<CFG>)
            }}

            #[inline(always)]
            fn to_f16_lanes<const CFG: u32>(self, value: [f32; {lanes}]) -> [u16; {lanes}] {{
                value.map(crate::scalar::to_f16_bits_with::<CFG>)
            }}

            #[inline(always)]
            fn from_bf16_lanes(self, lanes: &[u16; {lanes}]) -> [f32; {lanes}] {{
                lanes.map(crate::scalar::from_bf16)
            }}

            #[inline(always)]
            fn to_bf16_lanes<const CFG: u32>(self, value: [f32; {lanes}]) -> [u16; {lanes}] {{
                value.map(crate::scalar::to_bf16_with::<CFG>)
            }}

            #[inline(always)]
            fn broadcast_rgba_alpha_repr(self, value: [f32; {lanes}]) -> [f32; {lanes}] {{
                core::array::from_fn(|i| value[i | 3])
            }}

            #[inline(always)]
            fn with_rgba_alpha_repr(
                self,
                mut value: [f32; {lanes}],
                src: [f32; {lanes}],
            ) -> [f32; {lanes}] {{
                for i in (3..{lanes}).step_by(4) {{
                    value[i] = src[i];
                }}
                value
            }}

            {transpose}
            {rgb_and_rgba}
            {wide}
        }}
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
    let (bf16_nan, bf16_flush) = if lanes == 16 {
        (
            "_mm512_mask_blend_epi32(_mm512_cmpgt_epi32_mask(_mm512_and_si512(raw, _mm512_set1_epi32(0x7fff_ffff)), _mm512_set1_epi32(0x7f80_0000)), rounded, _mm512_or_si512(raw, _mm512_set1_epi32(0x0040_0000)))".to_string(),
            "_mm512_mask_blend_epi32(_mm512_test_epi32_mask(i, _mm512_set1_epi32(0x7f80)), _mm512_and_si512(i, _mm512_set1_epi32(0x8000)), i)".to_string(),
        )
    } else {
        (
            format!(
                "{prefix}_blendv_epi8(rounded, {prefix}_or_si{bits}(raw, {prefix}_set1_epi32(0x0040_0000)), {prefix}_cmpgt_epi32({prefix}_and_si{bits}(raw, {prefix}_set1_epi32(0x7fff_ffff)), {prefix}_set1_epi32(0x7f80_0000)))"
            ),
            format!(
                "{prefix}_andnot_si{bits}({prefix}_and_si{bits}({prefix}_cmpeq_epi32({prefix}_and_si{bits}(i, {prefix}_set1_epi32(0x7f80)), {prefix}_setzero_si{bits}()), {prefix}_set1_epi32(0x7fff)), i)"
            ),
        )
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
            let rounded = if opts.is_finite() {{ rounded }} else {{ {bf16_nan} }};
            let i = {prefix}_srli_epi32::<16>(rounded);
            let i = if opts.flushes() {{ {bf16_flush} }} else {{ i }};
            {convert}
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
            let channels = [r, g, b, a].map(|v| <Self as F32x4Internal>::to_u8_lanes::<CFG>(self, v));
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

/// AVX2 fast path for `F32x16Internal::to_u16_lanes` on `X64V3Token`, called
/// from the hand-written `impls/polyfill.rs` so both halves pack in one
/// 256-bit `_mm256_packus_epi32` + `_mm256_permute4x64_epi64`.
pub(super) fn x86_v3_f32x16_to_u16_helper() -> String {
    formatdoc! {r#"

        #[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "w512"))]
        #[attune(wrap, suppress_const_test)]
        pub(super) fn f32x16_to_u16_lanes<const CFG: u32>(
            _token: X64V3Token,
            value: [__m256; 2],
        ) -> [u16; 16] {{
            let opts = const {{ crate::numerics::IntConvert::decode(CFG) }};
            let convert = |v| {{
                let v = if opts.saturates_all() {{
                    _mm256_min_ps(_mm256_set1_ps(65535.0), v)
                }} else {{
                    v
                }};
                if opts.rounds() {{
                    _mm256_cvtps_epi32(v)
                }} else {{
                    _mm256_cvttps_epi32(v)
                }}
            }};
            let packed = _mm256_packus_epi32(convert(value[0]), convert(value[1]));
            crate::simd_storage::cast(_mm256_permute4x64_epi64::<0xd8>(packed))
        }}
    "#}
}

pub(super) fn polyfill_f32x8(repr: &str) -> String {
    formatdoc! {r#"
        #[inline(always)]
        fn from_u16_lanes(self, lanes: &[u16; 8]) -> {repr} {{
            let (chunks, _) = lanes.as_chunks::<4>();
            [
                <Self as F32x4Internal>::from_u16_lanes(self, &chunks[0]),
                <Self as F32x4Internal>::from_u16_lanes(self, &chunks[1]),
            ]
        }}

        #[inline(always)]
        fn to_u16_lanes<const CFG: u32>(self, value: {repr}) -> [u16; 8] {{
            crate::simd_storage::cast([
                <Self as F32x4Internal>::to_u16_lanes::<CFG>(self, value[0]),
                <Self as F32x4Internal>::to_u16_lanes::<CFG>(self, value[1]),
            ])
        }}

        #[inline(always)]
        fn to_u8_lanes<const CFG: u32>(self, value: {repr}) -> [u8; 8] {{
            crate::simd_storage::cast([
                <Self as F32x4Internal>::to_u8_lanes::<CFG>(self, value[0]),
                <Self as F32x4Internal>::to_u8_lanes::<CFG>(self, value[1]),
            ])
        }}

        #[inline(always)]
        fn to_u32_lanes<const CFG: u32>(self, value: {repr}) -> {repr} {{
            [
                <Self as F32x4Internal>::to_u32_lanes::<CFG>(self, value[0]),
                <Self as F32x4Internal>::to_u32_lanes::<CFG>(self, value[1]),
            ]
        }}

        #[inline(always)]
        fn from_u32_lanes(self, value: {repr}) -> {repr} {{
            [
                <Self as F32x4Internal>::from_u32_lanes(self, value[0]),
                <Self as F32x4Internal>::from_u32_lanes(self, value[1]),
            ]
        }}

        #[inline(always)]
        fn from_f16_lanes<const CFG: u32>(self, lanes: &[u16; 8]) -> {repr} {{
            let (chunks, _) = lanes.as_chunks::<4>();
            [
                <Self as F32x4Internal>::from_f16_lanes::<CFG>(self, &chunks[0]),
                <Self as F32x4Internal>::from_f16_lanes::<CFG>(self, &chunks[1]),
            ]
        }}

        #[inline(always)]
        fn to_f16_lanes<const CFG: u32>(self, value: {repr}) -> [u16; 8] {{
            crate::simd_storage::cast([
                <Self as F32x4Internal>::to_f16_lanes::<CFG>(self, value[0]),
                <Self as F32x4Internal>::to_f16_lanes::<CFG>(self, value[1]),
            ])
        }}

        #[inline(always)]
        fn from_bf16_lanes(self, lanes: &[u16; 8]) -> {repr} {{
            let (chunks, _) = lanes.as_chunks::<4>();
            [
                <Self as F32x4Internal>::from_bf16_lanes(self, &chunks[0]),
                <Self as F32x4Internal>::from_bf16_lanes(self, &chunks[1]),
            ]
        }}

        #[inline(always)]
        fn to_bf16_lanes<const CFG: u32>(self, value: {repr}) -> [u16; 8] {{
            crate::simd_storage::cast([
                <Self as F32x4Internal>::to_bf16_lanes::<CFG>(self, value[0]),
                <Self as F32x4Internal>::to_bf16_lanes::<CFG>(self, value[1]),
            ])
        }}

        #[inline(always)]
        fn broadcast_rgba_alpha_repr(self, value: {repr}) -> {repr} {{
            [
                <Self as F32x4Internal>::broadcast_rgba_alpha_repr(self, value[0]),
                <Self as F32x4Internal>::broadcast_rgba_alpha_repr(self, value[1]),
            ]
        }}

        #[inline(always)]
        fn with_rgba_alpha_repr(self, value: {repr}, src: {repr}) -> {repr} {{
            [
                <Self as F32x4Internal>::with_rgba_alpha_repr(self, value[0], src[0]),
                <Self as F32x4Internal>::with_rgba_alpha_repr(self, value[1], src[1]),
            ]
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
        fn permute2_wide(self, inputs: [__m512; 2], indices: [i32; 16]) -> __m512 {{
            _mm512_permutex2var_ps(inputs[0], crate::simd_storage::cast(indices), inputs[1])
        }}
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
