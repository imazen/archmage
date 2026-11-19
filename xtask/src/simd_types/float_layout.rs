//! Raw u16 conversion and RGBA lane motion for the generic f32 backends.

use indoc::formatdoc;

pub(super) fn trait_methods(elem: &str, lanes: usize, backend: &str) -> String {
    if elem != "f32" {
        return String::new();
    }
    let transpose = if lanes == 4 {
        formatdoc! {r#"
            /// Transpose four vectors without changing any lane bits.
            #[inline(always)]
            fn transpose_4x4_repr(self, rows: [Self::Repr; 4]) -> [Self::Repr; 4] {{
                let rows = rows.map(|r| <Self as {backend}>::to_array(self, r));
                core::array::from_fn(|col| <Self as {backend}>::from_array(self, rows.map(|r| r[col])))
            }}
        "#}
    } else {
        String::new()
    };
    formatdoc! {r#"
        /// Load unsigned lanes and convert them exactly, without normalization.
        #[inline(always)]
        fn from_u16_lanes(self, lanes: &[u16; {lanes}]) -> Self::Repr {{
            <Self as {backend}>::from_array(self, lanes.map(f32::from))
        }}

        /// Truncate and saturate exactly as Rust's `as u16` on each lane.
        #[inline(always)]
        fn to_u16_lanes(self, value: Self::Repr) -> [u16; {lanes}] {{
            <Self as {backend}>::to_array(self, value).map(|x| x as u16)
        }}

        /// Broadcast lane 3 in each consecutive group of four lanes.
        #[inline(always)]
        fn broadcast_rgba_alpha_repr(self, value: Self::Repr) -> Self::Repr {{
            let value = <Self as {backend}>::to_array(self, value);
            <Self as {backend}>::from_array(self, core::array::from_fn(|i| value[i | 3]))
        }}

        /// Replace lane 3 in each consecutive group of four lanes.
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
    formatdoc! {r#"
        {attr}
        fn from_u16_lanes(self, lanes: &[u16; {lanes}]) -> {repr} {{
            {prefix}_cvtepi32_ps({load})
        }}

        {attr}
        fn to_u16_lanes(self, value: {repr}) -> [u16; {lanes}] {{
            // Treat positive overflow as 65535 before cvtt can produce its
            // integer-indefinite sentinel. NaN and negative overflow become zero in the pack.
            let i = {prefix}_cvttps_epi32({prefix}_min_ps({prefix}_set1_ps(65535.0), value));
            {convert}
        }}

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
    formatdoc! {r#"
        {attr}
        fn from_u16_lanes(self, lanes: &[u16; 4]) -> float32x4_t {{
            vcvtq_f32_u32(vmovl_u16(crate::simd_storage::cast(*lanes)))
        }}

        {attr}
        fn to_u16_lanes(self, value: float32x4_t) -> [u16; 4] {{
            crate::simd_storage::cast(vqmovn_u32(vcvtq_u32_f32(value)))
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
        fn to_u16_lanes(self, value: v128) -> [u16; 4] {{
            let i = i32x4_trunc_sat_f32x4(value);
            crate::simd_storage::cast(u64x2_extract_lane::<0>(u16x8_narrow_i32x4(i, i)))
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
    formatdoc! {r#"
        #[inline(always)]
        fn from_u16_lanes(self, lanes: &[u16; {lanes}]) -> {repr} {{
            let (chunks, _) = lanes.as_chunks::<{part}>();
            core::array::from_fn(|i| <Self as {backend}>::from_u16_lanes(self, &chunks[i]))
        }}

        #[inline(always)]
        fn to_u16_lanes(self, value: {repr}) -> [u16; {lanes}] {{
            let parts = value.map(|v| <Self as {backend}>::to_u16_lanes(self, v));
            core::array::from_fn(|i| parts[i / {part}][i % {part}])
        }}

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
