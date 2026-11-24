//! Unified AArch64 NEON backend implementation generation (`impls/arm_neon.rs`).
//!
//! Generates native 128-bit backend impls for all 10 W128 SIMD types
//! (`f32x4`, `f64x2`, `i32x4`, `u32x4`, `i64x2`, `i8x16`, `u8x16`, `i16x8`,
//! `u16x8`, `u64x2`), plus the native `f32x8` pixel-packing helpers and all
//! 128-bit conversion/bitcast trait impls for `NeonToken`.
//! (W256/W512 polyfills live hand-written in `impls/polyfill.rs`.)

use indoc::formatdoc;

use super::types::{ElementType, SimdType, SimdWidth};

/// Generate the entire `magetypes/src/simd/impls/arm_neon.rs` file.
pub(super) fn generate_neon_file(narrow_types: &[SimdType]) -> String {
    let safety = super::backend_gen::impls_safety_contract("NeonToken");
    let mut code = formatdoc! {r#"
        //! Backend implementations for NeonToken (AArch64 NEON).
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.
        //!
        {safety}
        #[cfg(target_arch = "aarch64")]
        use core::arch::aarch64::*;

        #[cfg(target_arch = "aarch64")]
        use archmage::{{attune, NeonToken}};

        use crate::simd::backends::*;

    "#};

    for &ty in narrow_types {
        if ty.width == SimdWidth::W128 {
            if ty.is_float() {
                code.push_str("#[cfg(target_arch = \"aarch64\")]\n");
                code.push_str(&generate_neon_float_impl(ty));
                code.push('\n');
            } else {
                code.push_str("\n#[cfg(target_arch = \"aarch64\")]\n");
                code.push_str(&generate_neon_int_impl(ty));
                code.push('\n');
            }
        } else if ty.elem == ElementType::F32 && ty.width == SimdWidth::W256 {
            code.push_str(&generate_neon_f32x8_helpers());
            code.push('\n');
        }
    }

    code.push_str(&generate_neon_convert_impls());
    code
}

fn neon_repr(elem: ElementType) -> &'static str {
    match elem {
        ElementType::F32 => "float32x4_t",
        ElementType::F64 => "float64x2_t",
        ElementType::I8 => "int8x16_t",
        ElementType::U8 => "uint8x16_t",
        ElementType::I16 => "int16x8_t",
        ElementType::U16 => "uint16x8_t",
        ElementType::I32 => "int32x4_t",
        ElementType::U32 => "uint32x4_t",
        ElementType::I64 => "int64x2_t",
        ElementType::U64 => "uint64x2_t",
    }
}

fn neon_suffix(elem: ElementType) -> &'static str {
    match elem {
        ElementType::F32 => "f32",
        ElementType::F64 => "f64",
        ElementType::I8 => "s8",
        ElementType::U8 => "u8",
        ElementType::I16 => "s16",
        ElementType::U16 => "u16",
        ElementType::I32 => "s32",
        ElementType::U32 => "u32",
        ElementType::I64 => "s64",
        ElementType::U64 => "u64",
    }
}

// ============================================================================
// Float Implementations & f32x8 Helpers
// ============================================================================

/// Native NEON pixel-packing helpers for `f32x8` (`[float32x4_t; 2]`):
/// - `f32x8_to_u8_bytes` (called from `impls/polyfill.rs` `F32x8Backend::to_u8_bytes`)
/// - `F32x8Internal` (5 hand-tuned 24B/32B RGB/RGBA shuffle kernels + 11 delegating hooks)
fn generate_neon_f32x8_helpers() -> String {
    let arcane = super::backend_syntax::arcane("NeonToken");
    let float_layout = super::float_layout::polyfill_f32x8("[float32x4_t; 2]");
    formatdoc! {r#"
        #[cfg(target_arch = "aarch64")]
        #[attune(wrap, suppress_const_test)]
        pub(super) fn f32x8_to_u8_bytes(_token: NeonToken, a: [float32x4_t; 2]) -> [u8; 8] {{
            let i0 = vqmovn_s32(vcvtnq_s32_f32(a[0]));
            let i1 = vqmovn_s32(vcvtnq_s32_f32(a[1]));
            let u8s = vqmovun_s16(vcombine_s16(i0, i1));
            crate::simd_storage::cast(u8s)
        }}

        #[cfg(target_arch = "aarch64")]
        impl crate::simd::backends::F32x8Internal for archmage::NeonToken {{
            type InternalRepr = [float32x4_t; 2];

            {arcane}
            fn store_rgba_u8_into(
                self,
                r: [float32x4_t; 2],
                g: [float32x4_t; 2],
                b: [float32x4_t; 2],
                a: [float32x4_t; 2],
                out: &mut [u8; 32],
            ) {{
                let lo = vdupq_n_s32(0);
                let hi = vdupq_n_s32(255);
                let r0 = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(r[0]), lo), hi));
                let g0 = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(g[0]), lo), hi));
                let b0 = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(b[0]), lo), hi));
                let a0 = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(a[0]), lo), hi));
                let p0 = vorrq_u32(
                    vorrq_u32(r0, vshlq_n_u32::<8>(g0)),
                    vorrq_u32(vshlq_n_u32::<16>(b0), vshlq_n_u32::<24>(a0)),
                );
                let r1 = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(r[1]), lo), hi));
                let g1 = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(g[1]), lo), hi));
                let b1 = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(b[1]), lo), hi));
                let a1 = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(a[1]), lo), hi));
                let p1 = vorrq_u32(
                    vorrq_u32(r1, vshlq_n_u32::<8>(g1)),
                    vorrq_u32(vshlq_n_u32::<16>(b1), vshlq_n_u32::<24>(a1)),
                );
                let b0 = vreinterpretq_u8_u32(p0);
                let b1 = vreinterpretq_u8_u32(p1);
                #[cfg(target_endian = "big")]
                let (b0, b1) = (vrev32q_u8(b0), vrev32q_u8(b1));
                crate::simd_storage::store([b0, b1], out);
            }}

        {float_layout}

            {arcane}
            fn deinterleave_3ch_repr(self, rgb: [[float32x4_t; 2]; 3]) -> [[float32x4_t; 2]; 3] {{
                let idx_r: uint8x16_t = crate::simd_storage::cast([
                    0u8, 1, 2, 3, 12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39,
                ]);
                let idx_g: uint8x16_t = crate::simd_storage::cast([
                    4u8, 5, 6, 7, 16, 17, 18, 19, 28, 29, 30, 31, 40, 41, 42, 43,
                ]);
                let idx_b: uint8x16_t = crate::simd_storage::cast([
                    8u8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35, 44, 45, 46, 47,
                ]);
                let tbl_lo = uint8x16x3_t(
                    vreinterpretq_u8_f32(rgb[0][0]),
                    vreinterpretq_u8_f32(rgb[0][1]),
                    vreinterpretq_u8_f32(rgb[1][0]),
                );
                let tbl_hi = uint8x16x3_t(
                    vreinterpretq_u8_f32(rgb[1][1]),
                    vreinterpretq_u8_f32(rgb[2][0]),
                    vreinterpretq_u8_f32(rgb[2][1]),
                );
                [
                    [
                        vreinterpretq_f32_u8(vqtbl3q_u8(tbl_lo, idx_r)),
                        vreinterpretq_f32_u8(vqtbl3q_u8(tbl_hi, idx_r)),
                    ],
                    [
                        vreinterpretq_f32_u8(vqtbl3q_u8(tbl_lo, idx_g)),
                        vreinterpretq_f32_u8(vqtbl3q_u8(tbl_hi, idx_g)),
                    ],
                    [
                        vreinterpretq_f32_u8(vqtbl3q_u8(tbl_lo, idx_b)),
                        vreinterpretq_f32_u8(vqtbl3q_u8(tbl_hi, idx_b)),
                    ],
                ]
            }}

            {arcane}
            fn interleave_3ch_repr(self, channels: [[float32x4_t; 2]; 3]) -> [[float32x4_t; 2]; 3] {{
                let [r, g, b] = channels;
                let idx_0: uint8x16_t = crate::simd_storage::cast([
                    0u8, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 4, 5, 6, 7,
                ]);
                let idx_1: uint8x16_t = crate::simd_storage::cast([
                    20u8, 21, 22, 23, 36, 37, 38, 39, 8, 9, 10, 11, 24, 25, 26, 27,
                ]);
                let idx_2: uint8x16_t = crate::simd_storage::cast([
                    40u8, 41, 42, 43, 12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47,
                ]);
                let tbl_lo = uint8x16x3_t(
                    vreinterpretq_u8_f32(r[0]),
                    vreinterpretq_u8_f32(g[0]),
                    vreinterpretq_u8_f32(b[0]),
                );
                let tbl_hi = uint8x16x3_t(
                    vreinterpretq_u8_f32(r[1]),
                    vreinterpretq_u8_f32(g[1]),
                    vreinterpretq_u8_f32(b[1]),
                );
                [
                    [
                        vreinterpretq_f32_u8(vqtbl3q_u8(tbl_lo, idx_0)),
                        vreinterpretq_f32_u8(vqtbl3q_u8(tbl_lo, idx_1)),
                    ],
                    [
                        vreinterpretq_f32_u8(vqtbl3q_u8(tbl_lo, idx_2)),
                        vreinterpretq_f32_u8(vqtbl3q_u8(tbl_hi, idx_0)),
                    ],
                    [
                        vreinterpretq_f32_u8(vqtbl3q_u8(tbl_hi, idx_1)),
                        vreinterpretq_f32_u8(vqtbl3q_u8(tbl_hi, idx_2)),
                    ],
                ]
            }}

            {arcane}
            fn load_rgb_bytes(self, rgb: &[u8; 24]) -> ([float32x4_t; 2], [float32x4_t; 2], [float32x4_t; 2]) {{
                let lo16: uint8x16_t = crate::simd_storage::copy(rgb.first_chunk::<16>().unwrap());
                let hi8: uint8x8_t = crate::simd_storage::copy(rgb.last_chunk::<8>().unwrap());
                let tbl = uint8x16x2_t(lo16, vcombine_u8(hi8, vdup_n_u8(0)));
                let rg_idx: uint8x16_t = crate::simd_storage::cast([
                    0u8, 3, 6, 9, 12, 15, 18, 21, 1, 4, 7, 10, 13, 16, 19, 22,
                ]);
                let b_idx: uint8x16_t = crate::simd_storage::cast([
                    2u8, 5, 8, 11, 14, 17, 20, 23, 255, 255, 255, 255, 255, 255, 255, 255,
                ]);
                let rg8 = vqtbl2q_u8(tbl, rg_idx);
                let b8 = vqtbl2q_u8(tbl, b_idx);
                let r16 = vmovl_u8(vget_low_u8(rg8));
                let g16 = vmovl_high_u8(rg8);
                let b16 = vmovl_u8(vget_low_u8(b8));
                (
                    [
                        vcvtq_f32_u32(vmovl_u16(vget_low_u16(r16))),
                        vcvtq_f32_u32(vmovl_high_u16(r16)),
                    ],
                    [
                        vcvtq_f32_u32(vmovl_u16(vget_low_u16(g16))),
                        vcvtq_f32_u32(vmovl_high_u16(g16)),
                    ],
                    [
                        vcvtq_f32_u32(vmovl_u16(vget_low_u16(b16))),
                        vcvtq_f32_u32(vmovl_high_u16(b16)),
                    ],
                )
            }}

            {arcane}
            fn store_rgb_u8_into(
                self,
                r: [float32x4_t; 2],
                g: [float32x4_t; 2],
                b: [float32x4_t; 2],
                out: &mut [u8; 24],
            ) {{
                let r8 = vqmovun_s16(vcombine_s16(
                    vqmovn_s32(vcvtnq_s32_f32(r[0])),
                    vqmovn_s32(vcvtnq_s32_f32(r[1])),
                ));
                let g8 = vqmovun_s16(vcombine_s16(
                    vqmovn_s32(vcvtnq_s32_f32(g[0])),
                    vqmovn_s32(vcvtnq_s32_f32(g[1])),
                ));
                let b8 = vqmovun_s16(vcombine_s16(
                    vqmovn_s32(vcvtnq_s32_f32(b[0])),
                    vqmovn_s32(vcvtnq_s32_f32(b[1])),
                ));
                let tbl = uint8x16x2_t(vcombine_u8(r8, g8), vcombine_u8(b8, b8));
                let idx_lo: uint8x16_t = crate::simd_storage::cast([
                    0u8, 8, 16, 1, 9, 17, 2, 10, 18, 3, 11, 19, 4, 12, 20, 5,
                ]);
                let idx_hi: uint8x16_t = crate::simd_storage::cast([
                    13u8, 21, 6, 14, 22, 7, 15, 23, 255, 255, 255, 255, 255, 255, 255, 255,
                ]);
                let lo16 = vqtbl2q_u8(tbl, idx_lo);
                let hi8 = vqtbl2q_u8(tbl, idx_hi);
                crate::simd_storage::store(lo16, out.first_chunk_mut::<16>().unwrap());
                crate::simd_storage::store(
                    vget_low_u8(hi8),
                    out.last_chunk_mut::<8>().unwrap(),
                );
            }}
        }}
    "#}
}

/// Generate NEON impl for a native 128-bit float type (`f32x4`, `f64x2`).
fn generate_neon_float_impl(ty: SimdType) -> String {
    let arcane = super::backend_syntax::arcane("NeonToken");
    let trait_name = ty.trait_name();
    let repr = neon_repr(ty.elem);
    let elem = ty.elem.name();
    let lanes = ty.lanes();
    let array = ty.array_type();
    let ns = neon_suffix(ty.elem);
    let eb = ty.elem_bits();
    let is_f32 = ty.elem == ElementType::F32;

    // NEON lacks vmvnq_u64 — pre-compute NOT expression for f64
    let not_inner = if eb == 64 {
        format!("veorq_u64(vreinterpretq_u64_{ns}(a), vdupq_n_u64(u64::MAX))")
    } else {
        format!("vmvnq_u{eb}(vreinterpretq_u{eb}_{ns}(a))")
    };
    let ne_inner = if eb == 64 {
        format!("veorq_u64(vceqq_{ns}(a, b), vdupq_n_u64(u64::MAX))")
    } else {
        format!("vmvnq_u{eb}(vceqq_{ns}(a, b))")
    };

    // Working-tier `recip`/`rsqrt` bodies, chosen per element type:
    // - f32: estimate + one more fused FRECPS/FRSQRTS step than `_approx` (~2-3 ULP)
    // - f64: exact FDIV / FDIV+FSQRT
    let (recip_exact, rsqrt_exact) = if is_f32 {
        (
            format!(
                "let y = <Self as {trait_name}>::rcp_approx(_self, a); vmulq_{ns}(vrecpsq_{ns}(a, y), y)"
            ),
            format!(
                "let y = <Self as {trait_name}>::rsqrt_approx(_self, a); vmulq_{ns}(y, vrsqrtsq_{ns}(a, vmulq_{ns}(y, y)))"
            ),
        )
    } else {
        (
            format!("vdivq_{ns}(vdupq_n_{ns}(1.0), a)"),
            format!("vdivq_{ns}(vdupq_n_{ns}(1.0), vsqrtq_{ns}(a))"),
        )
    };

    let recip_perf_note = if is_f32 {
        "\
            //   Apple Silicon (M-series): exact is FASTER\n            \
            //     recip 1.35x, rsqrt 1.18x  (commits defbbc2 / 1b36fc7)\n            \
            //   Neoverse-N1 (Ampere Altra): exact is SLOWER\n            \
            //     rcp 1.39x, rsqrt 2.15x\n            \
            //     (benchmarks/rsqrt_arm_neoverse-n1_2026-06-21.md)"
    } else {
        "\
            //   Apple M4 Pro: exact is FASTER\n            \
            //     rcp 3.2x, rsqrt 1.23x\n            \
            //     (benchmarks/rsqrt_f64_arm_apple-m4-pro_2026-08-02.md)\n            \
            //   Neoverse / server ARM: NOT MEASURED for f64. The f32 form is\n            \
            //     slower there, so treat an f64 regression as plausible until\n            \
            //     someone runs `bench_rsqrt_f64` on that class of core."
    };

    let reduce_pairwise = |pairwise: &str| -> String {
        let mut body = format!("{{\n            let pair = {pairwise}(a, a);\n");
        if lanes > 2 {
            body.push_str(&format!("            let pair = {pairwise}(pair, pair);\n"));
        }
        body.push_str(&format!("            vgetq_lane_{ns}::<0>(pair)\n"));
        body.push_str("        }");
        body
    };

    let to_u8_arm = if is_f32 {
        formatdoc! {r#"

            {arcane}
            fn to_u8_bytes(self, a: {repr}) -> [u8; 4] {{
                let i16s = vqmovn_s32(vcvtnq_s32_f32(a));
                let u8s = vqmovun_s16(vcombine_s16(i16s, i16s));
                let bytes: [u8; 8] = crate::simd_storage::cast(u8s);
                [bytes[0], bytes[1], bytes[2], bytes[3]]
            }}
        "#}
    } else {
        String::new()
    };

    let internal_impl = if is_f32 {
        let float_layout = super::float_layout::neon(&arcane);
        formatdoc! {r#"

            #[cfg(target_arch = "aarch64")]
            impl crate::simd::backends::F32x{lanes}Internal for archmage::NeonToken {{
                type InternalRepr = {repr};

                {arcane}
                fn store_rgba_u8_into(self, r: {repr}, g: {repr}, b: {repr}, a: {repr}, out: &mut [u8; 16]) {{
                    let lo = vdupq_n_s32(0);
                    let hi = vdupq_n_s32(255);
                    let ri = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(r), lo), hi));
                    let gi = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(g), lo), hi));
                    let bi = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(b), lo), hi));
                    let ai = vreinterpretq_u32_s32(vminq_s32(vmaxq_s32(vcvtnq_s32_f32(a), lo), hi));
                    let pixels = vorrq_u32(
                        vorrq_u32(ri, vshlq_n_u32::<8>(gi)),
                        vorrq_u32(vshlq_n_u32::<16>(bi), vshlq_n_u32::<24>(ai)),
                    );
                    let bytes = vreinterpretq_u8_u32(pixels);
                    #[cfg(target_endian = "big")]
                    let bytes = vrev32q_u8(bytes);
                    crate::simd_storage::store(bytes, out);
                }}

            {float_layout}

                {arcane}
                fn deinterleave_3ch_repr(self, rgb: [{repr}; 3]) -> [{repr}; 3] {{
                    let idx_r: uint8x16_t = crate::simd_storage::cast([
                        0u8, 1, 2, 3, 12, 13, 14, 15, 24, 25, 26, 27, 36, 37, 38, 39,
                    ]);
                    let idx_g: uint8x16_t = crate::simd_storage::cast([
                        4u8, 5, 6, 7, 16, 17, 18, 19, 28, 29, 30, 31, 40, 41, 42, 43,
                    ]);
                    let idx_b: uint8x16_t = crate::simd_storage::cast([
                        8u8, 9, 10, 11, 20, 21, 22, 23, 32, 33, 34, 35, 44, 45, 46, 47,
                    ]);
                    let tbl = uint8x16x3_t(
                        vreinterpretq_u8_f32(rgb[0]),
                        vreinterpretq_u8_f32(rgb[1]),
                        vreinterpretq_u8_f32(rgb[2]),
                    );
                    [
                        vreinterpretq_f32_u8(vqtbl3q_u8(tbl, idx_r)),
                        vreinterpretq_f32_u8(vqtbl3q_u8(tbl, idx_g)),
                        vreinterpretq_f32_u8(vqtbl3q_u8(tbl, idx_b)),
                    ]
                }}

                {arcane}
                fn interleave_3ch_repr(self, channels: [{repr}; 3]) -> [{repr}; 3] {{
                    let idx_0: uint8x16_t = crate::simd_storage::cast([
                        0u8, 1, 2, 3, 16, 17, 18, 19, 32, 33, 34, 35, 4, 5, 6, 7,
                    ]);
                    let idx_1: uint8x16_t = crate::simd_storage::cast([
                        20u8, 21, 22, 23, 36, 37, 38, 39, 8, 9, 10, 11, 24, 25, 26, 27,
                    ]);
                    let idx_2: uint8x16_t = crate::simd_storage::cast([
                        40u8, 41, 42, 43, 12, 13, 14, 15, 28, 29, 30, 31, 44, 45, 46, 47,
                    ]);
                    let tbl = uint8x16x3_t(
                        vreinterpretq_u8_f32(channels[0]),
                        vreinterpretq_u8_f32(channels[1]),
                        vreinterpretq_u8_f32(channels[2]),
                    );
                    [
                        vreinterpretq_f32_u8(vqtbl3q_u8(tbl, idx_0)),
                        vreinterpretq_f32_u8(vqtbl3q_u8(tbl, idx_1)),
                        vreinterpretq_f32_u8(vqtbl3q_u8(tbl, idx_2)),
                    ]
                }}

                {arcane}
                fn load_rgb_bytes(self, rgb: &[u8; 12]) -> ({repr}, {repr}, {repr}) {{
                    let lo8: uint8x8_t = crate::simd_storage::copy(rgb.first_chunk::<8>().unwrap());
                    let hi4: u32 = crate::simd_storage::copy(rgb.last_chunk::<4>().unwrap());
                    let raw = vreinterpretq_u8_u32(vsetq_lane_u32::<2>(
                        hi4,
                        vreinterpretq_u32_u8(vcombine_u8(lo8, vdup_n_u8(0))),
                    ));
                    // All 12 channel bytes fit in one 16B tbl: [R0..3, G0..3, B0..3, 0..0].
                    let idx: uint8x16_t = crate::simd_storage::cast([
                        0u8, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11, 255, 255, 255, 255,
                    ]);
                    let rgb8 = vqtbl1q_u8(raw, idx);
                    let rg16 = vmovl_u8(vget_low_u8(rgb8));
                    let b16 = vmovl_high_u8(rgb8);
                    (
                        vcvtq_f32_u32(vmovl_u16(vget_low_u16(rg16))),
                        vcvtq_f32_u32(vmovl_high_u16(rg16)),
                        vcvtq_f32_u32(vmovl_u16(vget_low_u16(b16))),
                    )
                }}

                {arcane}
                fn store_rgb_u8_into(self, r: {repr}, g: {repr}, b: {repr}, out: &mut [u8; 12]) {{
                    let ri = vqmovn_s32(vcvtnq_s32_f32(r));
                    let gi = vqmovn_s32(vcvtnq_s32_f32(g));
                    let bi = vqmovn_s32(vcvtnq_s32_f32(b));
                    let rg8 = vqmovun_s16(vcombine_s16(ri, gi));
                    let b8 = vqmovun_s16(vcombine_s16(bi, bi));
                    let planes = vcombine_u8(rg8, b8);
                    let idx: uint8x16_t = crate::simd_storage::cast([
                        0u8, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11, 255, 255, 255, 255,
                    ]);
                    let packed = vqtbl1q_u8(planes, idx);
                    crate::simd_storage::store(
                        vget_low_u8(packed),
                        out.first_chunk_mut::<8>().unwrap(),
                    );
                    crate::simd_storage::store(
                        vgetq_lane_u32::<2>(vreinterpretq_u32_u8(packed)),
                        out.last_chunk_mut::<4>().unwrap(),
                    );
                }}
            }}"#}
    } else {
        String::new()
    };

    formatdoc! {r#"
        impl {trait_name} for archmage::NeonToken {{
            type Repr = {repr};

            {arcane}
            fn splat(self, v: {elem}) -> {repr} {{
                vdupq_n_{ns}(v)
            }}

            {arcane}
            fn zero(self) -> {repr} {{
                vdupq_n_{ns}(0.0)
            }}

            #[inline(always)]
            fn load(self, data: &{array}) -> {repr} {{
                crate::simd_storage::copy(data)
            }}

            #[inline(always)]
            fn from_array(self, arr: {array}) -> {repr} {{
                crate::simd_storage::cast(arr)
            }}

            #[inline(always)]
            fn store(self, repr: {repr}, out: &mut {array}) {{
                crate::simd_storage::store(repr, out);
            }}

            #[inline(always)]
            fn to_array(self, repr: {repr}) -> {array} {{
                crate::simd_storage::cast(repr)
            }}

            {arcane}
            fn add(self, a: {repr}, b: {repr}) -> {repr} {{ vaddq_{ns}(a, b) }}
            {arcane}
            fn sub(self, a: {repr}, b: {repr}) -> {repr} {{ vsubq_{ns}(a, b) }}
            {arcane}
            fn mul(self, a: {repr}, b: {repr}) -> {repr} {{ vmulq_{ns}(a, b) }}
            {arcane}
            fn div(self, a: {repr}, b: {repr}) -> {repr} {{ vdivq_{ns}(a, b) }}
            {arcane}
            fn neg(self, a: {repr}) -> {repr} {{ vnegq_{ns}(a) }}
            {arcane}
            fn min(self, a: {repr}, b: {repr}) -> {repr} {{ vminnmq_{ns}(a, b) }}
            {arcane}
            fn max(self, a: {repr}, b: {repr}) -> {repr} {{ vmaxnmq_{ns}(a, b) }}
            {arcane}
            fn minimum(self, a: {repr}, b: {repr}) -> {repr} {{ vminq_{ns}(a, b) }}
            {arcane}
            fn maximum(self, a: {repr}, b: {repr}) -> {repr} {{ vmaxq_{ns}(a, b) }}
            {arcane}
            fn min_fast(self, a: {repr}, b: {repr}) -> {repr} {{ vminnmq_{ns}(a, b) }}
            {arcane}
            fn max_fast(self, a: {repr}, b: {repr}) -> {repr} {{ vmaxnmq_{ns}(a, b) }}
            {arcane}
            fn sqrt(self, a: {repr}) -> {repr} {{ vsqrtq_{ns}(a) }}
            {arcane}
            fn abs(self, a: {repr}) -> {repr} {{ vabsq_{ns}(a) }}
            {arcane}
            fn trunc(self, a: {repr}) -> {repr} {{ vrndq_{ns}(a) }}

            {arcane}
            fn mul_neg_add_relaxed(self, a: {repr}, b: {repr}, c: {repr}) -> {repr} {{ vfmsq_{ns}(c, a, b) }}

            {arcane}
            fn mul_neg_add_fused(self, a: {repr}, b: {repr}, c: {repr}) -> {repr} {{ vfmsq_{ns}(c, a, b) }}

            {arcane}
            fn floor(self, a: {repr}) -> {repr} {{ vrndmq_{ns}(a) }}
            {arcane}
            fn ceil(self, a: {repr}) -> {repr} {{ vrndpq_{ns}(a) }}
            {arcane}
            fn round(self, a: {repr}) -> {repr} {{ vrndnq_{ns}(a) }}

            {arcane}
            fn mul_add(self, a: {repr}, b: {repr}, c: {repr}) -> {repr} {{
                vfmaq_{ns}(c, a, b)
            }}

            {arcane}
            fn mul_sub(self, a: {repr}, b: {repr}, c: {repr}) -> {repr} {{
                vfmaq_{ns}(vnegq_{ns}(c), a, b)
            }}

            {arcane}
            fn mul_add_portable(self, a: {repr}, b: {repr}, c: {repr}) -> {repr} {{
                vfmaq_{ns}(c, a, b)
            }}

            {arcane}
            fn mul_sub_portable(self, a: {repr}, b: {repr}, c: {repr}) -> {repr} {{
                vfmaq_{ns}(vnegq_{ns}(c), a, b)
            }}

            {arcane}
            fn simd_eq(self, a: {repr}, b: {repr}) -> {repr} {{
                vreinterpretq_{ns}_u{eb}(vceqq_{ns}(a, b))
            }}
            {arcane}
            fn simd_ne(self, a: {repr}, b: {repr}) -> {repr} {{
                vreinterpretq_{ns}_u{eb}({ne_inner})
            }}
            {arcane}
            fn simd_lt(self, a: {repr}, b: {repr}) -> {repr} {{
                vreinterpretq_{ns}_u{eb}(vcltq_{ns}(a, b))
            }}
            {arcane}
            fn simd_le(self, a: {repr}, b: {repr}) -> {repr} {{
                vreinterpretq_{ns}_u{eb}(vcleq_{ns}(a, b))
            }}
            {arcane}
            fn simd_gt(self, a: {repr}, b: {repr}) -> {repr} {{
                vreinterpretq_{ns}_u{eb}(vcgtq_{ns}(a, b))
            }}
            {arcane}
            fn simd_ge(self, a: {repr}, b: {repr}) -> {repr} {{
                vreinterpretq_{ns}_u{eb}(vcgeq_{ns}(a, b))
            }}

            {arcane}
            fn blend(self, mask: {repr}, if_true: {repr}, if_false: {repr}) -> {repr} {{
                vbslq_{ns}(vreinterpretq_u{eb}_{ns}(mask), if_true, if_false)
            }}

            {arcane}
            fn reduce_add(self, a: {repr}) -> {elem} {{
                {reduce_add}
            }}

            {arcane}
            fn reduce_min(self, a: {repr}) -> {elem} {{
                {reduce_min}
            }}

            {arcane}
            fn reduce_max(self, a: {repr}) -> {elem} {{
                {reduce_max}
            }}

            // ====== Reciprocals: estimate tier vs working tier ======
            //
            // `_approx` = raw vrecpe/vrsqrte (~8-bit) + one fused FRECPS/FRSQRTS
            // step (~16-bit): the documented >=12-bit fast path. FRECPS
            // (`vrecpsq`) computes `2 - a*y` and FRSQRTS (`vrsqrtsq`) computes
            // `(3 - a*y*y)/2` as one fused step each — no intermediate rounding,
            // no 2.0/3.0/0.5 splats, measurably faster than hand-rolled mul/sub
            // on real silicon (see `benchmarks/rsqrt_arm_neoverse-n1`). Each step
            // roughly doubles the correct bits (~8 -> ~16 -> ~24). FRECPS/FRSQRTS
            // are baseline NEON.
            //
            // `recip`/`rsqrt` carry the working-tier contract (<= 4 ULP AND
            // exact IEEE rails at ±0/±inf/NaN), filled by the fastest
            // conforming lowering per element type. f32 adds one more fused
            // step to `_approx` (~2-3 ULP; faster than FDIV on Neoverse-class
            // cores, the published 0.9.28 behavior). Rails come free from the
            // hardware: FRECPS/FRSQRTS special-case inf*0 to 2.0/1.5 BY
            // DESIGN so Newton iterations pass rails through — PROVIDED the
            // dangerous product is formed INSIDE the fused op. recip does
            // this naturally (FRECPS(a, y)); rsqrt is therefore written
            // y*FRSQRTS(a, y*y), NOT y*FRSQRTS(a*y, y): a plain a*y mul is
            // NaN at the rails and was exactly how the pre-fix NaN leaked
            // (probe-verified under qemu, 2026-09-04). Cost: identical
            // instruction count. Tradeoff: y*y overflows f32 for a below
            // ~3e-39, so deep-subnormal inputs are unspecified at this tier.
            // f64 divides (no worthwhile f64 estimate exists, and FDIV is
            // both exact and measured faster on Apple).
            //
            // Cost of exactness is per-core and does NOT generalize — these
            // numbers are for {elem} specifically:
{recip_perf_note}
            {arcane}
            fn rcp_approx(self, a: {repr}) -> {repr} {{
                let y = vrecpeq_{ns}(a); vmulq_{ns}(vrecpsq_{ns}(a, y), y)
            }}
            {arcane}
            fn rsqrt_approx(self, a: {repr}) -> {repr} {{
                let y = vrsqrteq_{ns}(a); vmulq_{ns}(y, vrsqrtsq_{ns}(a, vmulq_{ns}(y, y)))
            }}
            {arcane}
            fn recip(self, a: {repr}) -> {repr} {{
                {recip_exact}
            }}
            {arcane}
            fn rsqrt(self, a: {repr}) -> {repr} {{
                {rsqrt_exact}
            }}

            {arcane}
            fn not(self, a: {repr}) -> {repr} {{
                vreinterpretq_{ns}_u{eb}({not_inner})
            }}
            {arcane}
            fn bitand(self, a: {repr}, b: {repr}) -> {repr} {{
                vreinterpretq_{ns}_u{eb}(vandq_u{eb}(vreinterpretq_u{eb}_{ns}(a), vreinterpretq_u{eb}_{ns}(b)))
            }}
            {arcane}
            fn bitor(self, a: {repr}, b: {repr}) -> {repr} {{
                vreinterpretq_{ns}_u{eb}(vorrq_u{eb}(vreinterpretq_u{eb}_{ns}(a), vreinterpretq_u{eb}_{ns}(b)))
            }}
            {arcane}
            fn bitxor(self, a: {repr}, b: {repr}) -> {repr} {{
                vreinterpretq_{ns}_u{eb}(veorq_u{eb}(vreinterpretq_u{eb}_{ns}(a), vreinterpretq_u{eb}_{ns}(b)))
            }}
            {to_u8_arm}
        }}{internal_impl}
    "#,
        reduce_add = reduce_pairwise(&format!("vpaddq_{ns}")),
        reduce_min = if is_f32 { String::from("vminnmvq_f32(a)") } else { reduce_pairwise(&format!("vpminq_{ns}")) },
        reduce_max = if is_f32 { String::from("vmaxnmvq_f32(a)") } else { reduce_pairwise(&format!("vpmaxq_{ns}")) },
    }
}

// ============================================================================
// Unified Integer Implementation (all 8 W128 integer types)
// ============================================================================

fn generate_neon_int_impl(ty: SimdType) -> String {
    let integer_methods = super::backend_gen_widen_narrow::methods(&ty.name(), "NeonToken");
    let arcane = super::backend_syntax::arcane("NeonToken");
    let trait_name = ty.trait_name();
    let array = ty.array_type();
    let elem = ty.elem.name();
    let ns = neon_suffix(ty.elem);
    let nt = neon_repr(ty.elem);
    let eb = ty.elem_bits();
    let signed = ty.is_signed();
    let is_i64 = ty.elem == ElementType::I64;
    let zero_lit = if is_i64 { "0i64" } else { "0" };

    let wrap_cmp = |intrinsic: &str| -> String {
        if signed {
            format!("vreinterpretq_{ns}_u{eb}({intrinsic}(a, b))")
        } else {
            format!("{intrinsic}(a, b)")
        }
    };

    let ne_body = match ty.elem {
        ElementType::I64 => formatdoc! {r#"
            let eq = vceqq_s64(a, b);
            // NOT via XOR with all-ones
            vreinterpretq_s64_u64(veorq_u64(eq, vdupq_n_u64(u64::MAX)))"#},
        ElementType::U64 => "veorq_u64(vceqq_u64(a, b), vdupq_n_u64(u64::MAX))".to_string(),
        _ if signed => format!("vreinterpretq_{ns}_u{eb}(vmvnq_u{eb}(vceqq_{ns}(a, b)))"),
        _ => format!("vmvnq_{ns}(vceqq_{ns}(a, b))"),
    };

    let blend_body = if signed {
        format!("vbslq_{ns}(vreinterpretq_u{eb}_{ns}(mask), if_true, if_false)")
    } else {
        format!("vbslq_{ns}(mask, if_true, if_false)")
    };

    let reduce_add_body = if is_i64 {
        "let sum = vpaddq_s64(a, a);\n                vgetq_lane_s64::<0>(sum)".to_string()
    } else {
        format!("vaddvq_{ns}(a)")
    };

    let not_body = match ty.elem {
        ElementType::I64 => {
            "let ones = vdupq_n_s64(-1i64);\n                veorq_s64(a, ones)".to_string()
        }
        ElementType::U64 => "veorq_u64(a, vdupq_n_u64(u64::MAX))".to_string(),
        _ => format!("vmvnq_{ns}(a)"),
    };

    let mut body = formatdoc! {r#"
        impl {trait_name} for archmage::NeonToken {{
            type Repr = {nt};

            {arcane}
            fn splat(self, v: {elem}) -> {nt} {{
                vdupq_n_{ns}(v)
            }}

            {arcane}
            fn zero(self) -> {nt} {{
                vdupq_n_{ns}({zero_lit})
            }}

            #[inline(always)]
            fn load(self, data: &{array}) -> {nt} {{
                crate::simd_storage::copy(data)
            }}

            #[inline(always)]
            fn from_array(self, arr: {array}) -> {nt} {{
                crate::simd_storage::cast(arr)
            }}

            #[inline(always)]
            fn store(self, repr: {nt}, out: &mut {array}) {{
                crate::simd_storage::store(repr, out);
            }}

            #[inline(always)]
            fn to_array(self, repr: {nt}) -> {array} {{
                crate::simd_storage::cast(repr)
            }}

            {arcane}
            fn add(self, a: {nt}, b: {nt}) -> {nt} {{ vaddq_{ns}(a, b) }}
            {arcane}
            fn sub(self, a: {nt}, b: {nt}) -> {nt} {{ vsubq_{ns}(a, b) }}
    "#};

    if matches!(eb, 16 | 32) {
        body.push_str(&formatdoc! {r#"
            {arcane}
            fn mul(self, a: {nt}, b: {nt}) -> {nt} {{ vmulq_{ns}(a, b) }}
        "#});
    }

    if signed {
        body.push_str(&formatdoc! {r#"
            {arcane}
            fn neg(self, a: {nt}) -> {nt} {{ vnegq_{ns}(a) }}
        "#});
    }

    match ty.elem {
        ElementType::I64 => {
            body.push_str(&formatdoc! {r#"
                {arcane}
                fn min(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {{
                    // NEON lacks native i64 min; polyfill via compare+select
                    {{
                        let mask = vcltq_s64(a, b);
                        vbslq_s64(mask, a, b)
                    }}
                }}
                {arcane}
                fn max(self, a: int64x2_t, b: int64x2_t) -> int64x2_t {{
                    // NEON lacks native i64 max; polyfill via compare+select
                    {{
                        let mask = vcgtq_s64(a, b);
                        vbslq_s64(mask, a, b)
                    }}
                }}
            "#});
        }
        ElementType::U64 => {
            body.push_str(&formatdoc! {r#"
                {arcane}
                fn min(self, a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {{ vbslq_u64(vcltq_u64(a, b), a, b) }}
                {arcane}
                fn max(self, a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {{ vbslq_u64(vcgtq_u64(a, b), a, b) }}
            "#});
        }
        _ => {
            body.push_str(&formatdoc! {r#"
                {arcane}
                fn min(self, a: {nt}, b: {nt}) -> {nt} {{ vminq_{ns}(a, b) }}
                {arcane}
                fn max(self, a: {nt}, b: {nt}) -> {nt} {{ vmaxq_{ns}(a, b) }}
            "#});
        }
    }

    if signed {
        body.push_str(&formatdoc! {r#"
            {arcane}
            fn abs(self, a: {nt}) -> {nt} {{ vabsq_{ns}(a) }}
        "#});
    }

    let spaced = matches!(
        ty.elem,
        ElementType::I32 | ElementType::U32 | ElementType::I64
    );
    let sep = if spaced { "\n" } else { "" };
    body.push_str(&formatdoc! {r#"

            {arcane}
            fn simd_eq(self, a: {nt}, b: {nt}) -> {nt} {{
                {eq}
            }}
            {arcane}
            fn simd_ne(self, a: {nt}, b: {nt}) -> {nt} {{
                {ne_body}
            }}
            {arcane}
            fn simd_lt(self, a: {nt}, b: {nt}) -> {nt} {{
                {lt}
            }}
            {arcane}
            fn simd_le(self, a: {nt}, b: {nt}) -> {nt} {{
                {le}
            }}
            {arcane}
            fn simd_gt(self, a: {nt}, b: {nt}) -> {nt} {{
                {gt}
            }}
            {arcane}
            fn simd_ge(self, a: {nt}, b: {nt}) -> {nt} {{
                {ge}
            }}

            {arcane}
            fn blend(self, mask: {nt}, if_true: {nt}, if_false: {nt}) -> {nt} {{
                {blend_body}
            }}{sep}
            {arcane}
            fn reduce_add(self, a: {nt}) -> {elem} {{
                {reduce_add_body}
            }}{sep}
            {arcane}
            fn not(self, a: {nt}) -> {nt} {{
                {not_body}
            }}
            {arcane}
            fn bitand(self, a: {nt}, b: {nt}) -> {nt} {{
                vandq_{ns}(a, b)
            }}
            {arcane}
            fn bitor(self, a: {nt}, b: {nt}) -> {nt} {{
                vorrq_{ns}(a, b)
            }}
            {arcane}
            fn bitxor(self, a: {nt}, b: {nt}) -> {nt} {{
                veorq_{ns}(a, b)
            }}{sep}
    "#,
        eq = wrap_cmp(&format!("vceqq_{ns}")),
        lt = wrap_cmp(&format!("vcltq_{ns}")),
        le = wrap_cmp(&format!("vcleq_{ns}")),
        gt = wrap_cmp(&format!("vcgtq_{ns}")),
        ge = wrap_cmp(&format!("vcgeq_{ns}")),
    });

    // Right shifts lower through vshlq with a negated count: vshrq_n_* rejects
    // N == 0, while vshlq_* treats a negative count as a right shift and 0 as identity.
    let max_sh = eb - 1;
    let neg_n = if eb == 32 {
        "vdupq_n_s32(-N)".to_string()
    } else {
        format!("vdupq_n_s{eb}((-N) as i{eb})")
    };
    let shr_logical_expr = if signed {
        format!("vreinterpretq_{ns}_u{eb}(vshlq_u{eb}(vreinterpretq_u{eb}_{ns}(a), {neg_n}))")
    } else {
        format!("vshlq_{ns}(a, {neg_n})")
    };
    let shr_logical_fn = formatdoc! {r#"

            {arcane}
            fn shr_logical_const<const N: i32>(self, a: {nt}) -> {nt} {{
                const {{ assert!(N >= 0 && N <= {max_sh}) }};
                {shr_logical_expr}
            }}
    "#};
    let shr_arith_fn = formatdoc! {r#"
            {arcane}
            fn shr_arithmetic_const<const N: i32>(self, a: {nt}) -> {nt} {{
                const {{ assert!(N >= 0 && N <= {max_sh}) }};
                vshlq_{ns}(a, {neg_n})
            }}
    "#};

    body.push_str(&formatdoc! {r#"
            {arcane}
            fn shl_const<const N: i32>(self, a: {nt}) -> {nt} {{
                vshlq_n_{ns}::<N>(a)
            }}
    "#});
    if signed && eb >= 32 {
        body.push('\n');
        body.push_str(&shr_arith_fn);
        body.push_str(&shr_logical_fn);
    } else {
        body.push_str(&shr_logical_fn);
        if signed {
            body.push_str(&shr_arith_fn);
        }
    }

    // Uniform variable shifts (16-bit and 32-bit types).
    if matches!(eb, 16 | 32) {
        let uniform_header = match ty.elem {
            ElementType::I32 => {
                "\n            // ====== Uniform variable shifts ======\n            // Same vshlq lowering as the const forms; the count is clamped\n            // because USHL/SSHL read only the low byte of each amount lane as\n            // a signed value, so an unclamped 256 would wrap to a no-op\n            // instead of the contracted zero.\n"
            }
            ElementType::U32 => {
                "\n            // ====== Uniform variable shifts ======\n            // Same vshlq lowering as the const forms; the count is clamped\n            // because USHL reads only the low byte of each amount lane as a\n            // signed value, so an unclamped 256 would wrap to a no-op instead\n            // of the contracted zero.\n"
            }
            _ => "",
        };
        let shr_logical_uniform = if signed {
            format!(
                "vreinterpretq_{ns}_u{eb}(vshlq_u{eb}(vreinterpretq_u{eb}_{ns}(a), vdupq_n_s{eb}(-(count.min({eb}) as i{eb}))))"
            )
        } else {
            format!("vshlq_{ns}(a, vdupq_n_s{eb}(-(count.min({eb}) as i{eb})))")
        };
        body.push_str(uniform_header);
        body.push_str(&formatdoc! {r#"

            {arcane}
            fn shl_uniform(self, a: {nt}, count: u32) -> {nt} {{
                vshlq_{ns}(a, vdupq_n_s{eb}(count.min({eb}) as i{eb}))
            }}{sep}
            {arcane}
            fn shr_logical_uniform(self, a: {nt}, count: u32) -> {nt} {{
                {shr_logical_uniform}
            }}{sep}
        "#});

        if signed {
            let arith_comment = if eb == 32 {
                "// Clamping to 31 gives the contracted sign fill."
            } else {
                "// Clamping to lane_bits - 1 gives the contracted sign fill."
            };
            body.push_str(&formatdoc! {r#"
            {arcane}
            fn shr_arithmetic_uniform(self, a: {nt}, count: u32) -> {nt} {{
                {arith_comment}
                vshlq_{ns}(a, vdupq_n_s{eb}(-(count.min({max_sh}) as i{eb})))
            }}
            "#});
        }
    }

    if eb <= 16 {
        body.push_str(&formatdoc! {r#"

            {arcane}
            fn saturating_add(self, a: {nt}, b: {nt}) -> {nt} {{
                vqaddq_{ns}(a, b)
            }}

            {arcane}
            fn saturating_sub(self, a: {nt}, b: {nt}) -> {nt} {{
                vqsubq_{ns}(a, b)
            }}
        "#});
    }

    // Boolean reductions: sign-bit contract across all widths.
    match ty.elem {
        ElementType::I64 => {
            body.push_str(&formatdoc! {r#"

                {arcane}
                fn all_true(self, a: int64x2_t) -> bool {{
                    // Sign-bit contract. NEON has no 64-bit horizontal reduction,
                    // so combine the lanes and test one sign bit.
                    let as_u64 = vreinterpretq_u64_s64(a);
                    ((vgetq_lane_u64::<0>(as_u64) & vgetq_lane_u64::<1>(as_u64)) >> 63) != 0
                }}

                {arcane}
                fn any_true(self, a: int64x2_t) -> bool {{
                    let as_u64 = vreinterpretq_u64_s64(a);
                    ((vgetq_lane_u64::<0>(as_u64) | vgetq_lane_u64::<1>(as_u64)) >> 63) != 0
                }}

                {arcane}
                fn bitmask(self, a: int64x2_t) -> u32 {{
                    let signs = vshrq_n_u64::<63>(vreinterpretq_u64_s64(a));
                    ((vgetq_lane_u64::<0>(signs) & 1) | ((vgetq_lane_u64::<1>(signs) & 1) << 1)) as u32
                }}
            "#});
        }
        ElementType::U64 => {
            body.push_str(&formatdoc! {r#"

                {arcane}
                fn all_true(self, a: uint64x2_t) -> bool {{
                    // NEON has no 64-bit horizontal reduction; AND the lanes and
                    // test the sign bit once.
                    ((vgetq_lane_u64::<0>(a) & vgetq_lane_u64::<1>(a)) >> 63) != 0
                }}

                {arcane}
                fn any_true(self, a: uint64x2_t) -> bool {{
                    ((vgetq_lane_u64::<0>(a) | vgetq_lane_u64::<1>(a)) >> 63) != 0
                }}

                {arcane}
                fn bitmask(self, a: uint64x2_t) -> u32 {{
                    let shift = vshrq_n_u64::<63>(a);
                    let lane0 = vgetq_lane_u64::<0>(shift) as u32;
                    let lane1 = vgetq_lane_u64::<1>(shift) as u32;
                    lane0 | (lane1 << 1)
                }}
            "#});
        }
        _ => {
            let as_signed = if signed {
                "a".to_string()
            } else {
                format!("vreinterpretq_s{eb}_{ns}(a)")
            };
            let all_true_comment = match ty.elem {
                ElementType::I32 => {
                    "// Every lane's sign bit set <=> every lane negative <=> the\n                // signed maximum is negative. Measured 28% faster on M4 Pro\n                // than the unsigned-min form this replaces.\n                "
                }
                ElementType::U32 => {
                    "// Sign-bit contract on the signed reinterpretation. This lane\n                // type previously used a third rule (all lanes == u32::MAX).\n                "
                }
                _ => "",
            };
            let bitmask_body = generate_neon_bitmask(ty, &as_signed);
            body.push_str(&formatdoc! {r#"

                {arcane}
                fn all_true(self, a: {nt}) -> bool {{
                    {all_true_comment}vmaxvq_s{eb}({as_signed}) < 0
                }}

                {arcane}
                fn any_true(self, a: {nt}) -> bool {{
                    vminvq_s{eb}({as_signed}) < 0
                }}

                {arcane}
                fn bitmask(self, a: {nt}) -> u32 {{
                    {bitmask_body}
                }}
            "#});
        }
    }

    body.push_str(&integer_methods);
    body.push_str("    }\n");
    body
}

/// NEON has no `movemask`. Packing one sign bit per lane depends on lane count
/// (measured on Apple M4 Pro, `benchmarks/neon_bitmask_apple-m4-pro_2026-09-08.md`):
/// - 16 lanes (8-bit): weight + narrow `vpadd_u8` tree
/// - 8 lanes (16-bit): weight + `vaddvq_u16`
/// - 4 lanes (32-bit): logical shift to 0/1 + positional shift + `vaddvq_u32`
fn generate_neon_bitmask(ty: SimdType, as_signed: &str) -> String {
    match ty.elem_bits() {
        8 => formatdoc! {r#"
            {{
                let powers: [u8; 16] = [1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128];
                let pow_vec: uint8x16_t = crate::simd_storage::copy(&powers);
                let weighted = vandq_u8(vcltzq_s8({as_signed}), pow_vec);
                let s = vpadd_u8(vget_low_u8(weighted), vget_high_u8(weighted));
                let s = vpadd_u8(s, s);
                let s = vpadd_u8(s, s);
                u32::from(vget_lane_u16::<0>(vreinterpret_u16_u8(s)))
            }}"#},
        16 => formatdoc! {r#"
            {{
                let powers: [u16; 8] = [1,2,4,8,16,32,64,128];
                let pow_vec: uint16x8_t = crate::simd_storage::copy(&powers);
                u32::from(vaddvq_u16(vandq_u16(vcltzq_s16({as_signed}), pow_vec)))
            }}"#},
        32 => {
            let as_u32 = if ty.is_signed() {
                "vreinterpretq_u32_s32(a)"
            } else {
                "a"
            };
            formatdoc! {r#"
                // One sign bit per word -> 4-bit mask. Shift each 0/1 lane
                // into position, then one horizontal add. The shift on the
                // u32 view must be LOGICAL — an arithmetic s32 shift
                // sign-extends to 0xFFFF_FFFF and corrupts the mask.
                let shifts: [i32; 4] = [0, 1, 2, 3];
                let shift_vec: int32x4_t = crate::simd_storage::copy(&shifts);
                vaddvq_u32(vshlq_u32(vshrq_n_u32::<31>({as_u32}), shift_vec))"#}
        }
        _ => unreachable!("64-bit bitmask handled inline"),
    }
}

// ============================================================================
// Conversion & Bitcast Implementations
// ============================================================================

fn generate_neon_convert_impls() -> String {
    let arcane = super::backend_syntax::arcane("NeonToken");
    let mut code = formatdoc! {r#"

        #[cfg(target_arch = "aarch64")]
        impl F32x4Convert for archmage::NeonToken {{
            {arcane}
            fn bitcast_f32_to_i32(self, a: float32x4_t) -> int32x4_t {{
                vreinterpretq_s32_f32(a)
            }}

            {arcane}
            fn bitcast_i32_to_f32(self, a: int32x4_t) -> float32x4_t {{
                vreinterpretq_f32_s32(a)
            }}

            {arcane}
            fn convert_f32_to_i32(self, a: float32x4_t) -> int32x4_t {{
                vcvtq_s32_f32(a)
            }}

            {arcane}
            fn convert_f32_to_i32_round(self, a: float32x4_t) -> int32x4_t {{
                vcvtnq_s32_f32(a)
            }}

            {arcane}
            fn convert_i32_to_f32(self, a: int32x4_t) -> float32x4_t {{
                vcvtq_f32_s32(a)
            }}
        }}
    "#};

    struct NeonBitcast {
        tr: &'static str,
        from_e: &'static str,
        to_e: &'static str,
        from_r: &'static str,
        to_r: &'static str,
        from_s: &'static str,
        to_s: &'static str,
        spaced: bool,
    }
    const BITCASTS: &[NeonBitcast] = &[
        NeonBitcast {
            tr: "U32x4Bitcast",
            from_e: "u32",
            to_e: "i32",
            from_r: "uint32x4_t",
            to_r: "int32x4_t",
            from_s: "u32",
            to_s: "s32",
            spaced: true,
        },
        NeonBitcast {
            tr: "I64x2Bitcast",
            from_e: "i64",
            to_e: "f64",
            from_r: "int64x2_t",
            to_r: "float64x2_t",
            from_s: "s64",
            to_s: "f64",
            spaced: true,
        },
        NeonBitcast {
            tr: "I8x16Bitcast",
            from_e: "i8",
            to_e: "u8",
            from_r: "int8x16_t",
            to_r: "uint8x16_t",
            from_s: "s8",
            to_s: "u8",
            spaced: false,
        },
        NeonBitcast {
            tr: "I16x8Bitcast",
            from_e: "i16",
            to_e: "u16",
            from_r: "int16x8_t",
            to_r: "uint16x8_t",
            from_s: "s16",
            to_s: "u16",
            spaced: false,
        },
        NeonBitcast {
            tr: "U64x2Bitcast",
            from_e: "u64",
            to_e: "i64",
            from_r: "uint64x2_t",
            to_r: "int64x2_t",
            from_s: "u64",
            to_s: "s64",
            spaced: false,
        },
    ];
    for b in BITCASTS {
        let NeonBitcast {
            tr,
            from_e,
            to_e,
            from_r,
            to_r,
            from_s,
            to_s,
            spaced,
        } = *b;
        let sep = if spaced { "\n" } else { "" };
        code.push_str(&formatdoc! {r#"

            #[cfg(target_arch = "aarch64")]
            impl {tr} for archmage::NeonToken {{
                {arcane}
                fn bitcast_{from_e}_to_{to_e}(self, a: {from_r}) -> {to_r} {{
                    vreinterpretq_{to_s}_{from_s}(a)
                }}{sep}
                {arcane}
                fn bitcast_{to_e}_to_{from_e}(self, a: {to_r}) -> {from_r} {{
                    vreinterpretq_{from_s}_{to_s}(a)
                }}
            }}
        "#});
    }

    code
}
