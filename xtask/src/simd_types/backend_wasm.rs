//! Unified WebAssembly SIMD128 backend implementation generation (`impls/wasm128.rs`).
//!
//! Generates native 128-bit backend impls for all 10 W128 SIMD types
//! (`f32x4`, `f64x2`, `i32x4`, `u32x4`, `i64x2`, `i8x16`, `u8x16`, `i16x8`,
//! `u16x8`, `u64x2`), plus the native `f32x8` pixel-packing helpers and all
//! 128-bit conversion/bitcast trait impls for `Wasm128Token`.
//! (W256/W512 polyfills live hand-written in `impls/polyfill.rs`.)

use indoc::formatdoc;

use super::types::{ElementType, SimdType, SimdWidth};

/// Generate the entire `magetypes/src/simd/impls/wasm128.rs` file.
pub(super) fn generate_wasm_file(narrow_types: &[SimdType]) -> String {
    let safety = super::backend_gen::impls_safety_contract("Wasm128Token");
    let mut code = formatdoc! {r#"
        //! Backend implementations for Wasm128Token (WebAssembly SIMD).
        //!
        //! **Auto-generated** by `cargo xtask generate` - do not edit manually.
        //!
        {safety}
        #[cfg(target_arch = "wasm32")]
        use core::arch::wasm32::*;

        use crate::simd::backends::*;

    "#};

    for &ty in narrow_types {
        if ty.width == SimdWidth::W128 {
            if ty.is_float() {
                code.push_str("#[cfg(target_arch = \"wasm32\")]\n");
                code.push_str(&generate_wasm_float_impl(ty));
                code.push('\n');
            } else {
                code.push_str("\n#[cfg(target_arch = \"wasm32\")]\n");
                code.push_str(&generate_wasm_int_impl(ty));
                code.push('\n');
            }
        } else if ty.elem == ElementType::F32 && ty.width == SimdWidth::W256 {
            code.push_str(&generate_wasm_f32x8_helpers());
            code.push('\n');
        }
    }

    code.push_str(&generate_wasm_convert_impls());
    code
}

// ============================================================================
// Float Implementations & f32x8 Helpers
// ============================================================================

/// Native WASM128 pixel-packing helpers for `f32x8` (`[v128; 2]`):
/// - `f32x8_to_u8_bytes` (called from `impls/polyfill.rs` `F32x8Backend::to_u8_bytes`)
/// - `F32x8Internal` (5 hand-tuned 24B/32B RGB/RGBA shuffle kernels + 11 delegating hooks)
fn generate_wasm_f32x8_helpers() -> String {
    let float_layout = super::float_layout::polyfill_f32x8("[v128; 2]");
    formatdoc! {r#"
        #[cfg(target_arch = "wasm32")]
        #[inline(always)]
        pub(super) fn f32x8_to_u8_bytes(_token: archmage::Wasm128Token, a: [v128; 2]) -> [u8; 8] {{
            let i0 = i32x4_trunc_sat_f32x4(f32x4_nearest(a[0]));
            let i1 = i32x4_trunc_sat_f32x4(f32x4_nearest(a[1]));
            let i16s = i16x8_narrow_i32x4(i0, i1);
            let u8s = u8x16_narrow_i16x8(i16s, i16s);
            let lo = u32x4_extract_lane::<0>(u8s);
            let hi = u32x4_extract_lane::<1>(u8s);
            ((u64::from(hi) << 32) | u64::from(lo)).to_ne_bytes()
        }}

        #[cfg(target_arch = "wasm32")]
        impl crate::simd::backends::F32x8Internal for archmage::Wasm128Token {{
            type InternalRepr = [v128; 2];
        {float_layout}

            #[inline(always)]
            fn deinterleave_3ch_repr(self, rgb: [[v128; 2]; 3]) -> [[v128; 2]; 3] {{
                let [[v0, v1], [v2, v3], [v4, v5]] = rgb;
                let [r0, g0, b0] = <Self as F32x4Internal>::deinterleave_3ch_repr(self, [v0, v1, v2]);
                let [r1, g1, b1] = <Self as F32x4Internal>::deinterleave_3ch_repr(self, [v3, v4, v5]);
                [[r0, r1], [g0, g1], [b0, b1]]
            }}

            #[inline(always)]
            fn store_rgba_u8_into(
                self,
                r: [v128; 2],
                g: [v128; 2],
                b: [v128; 2],
                a: [v128; 2],
                out: &mut [u8; 32],
            ) {{
                let lo = i32x4_splat(0);
                let hi = i32x4_splat(255);
                let clamp = |v: v128| i32x4_min(i32x4_max(i32x4_trunc_sat_f32x4(f32x4_nearest(v)), lo), hi);
                let pack = |r: v128, g: v128, b: v128, a: v128| {{
                    v128_or(
                        v128_or(clamp(r), i32x4_shl(clamp(g), 8)),
                        v128_or(i32x4_shl(clamp(b), 16), i32x4_shl(clamp(a), 24)),
                    )
                }};
                crate::simd_storage::store([pack(r[0], g[0], b[0], a[0]), pack(r[1], g[1], b[1], a[1])], out);
            }}

            #[inline(always)]
            fn interleave_3ch_repr(self, channels: [[v128; 2]; 3]) -> [[v128; 2]; 3] {{
                let [r, g, b] = channels;
                let int4 = |r: v128, g: v128, b: v128| -> (v128, v128, v128) {{
                    let rg01 = i32x4_shuffle::<0, 4, 1, 5>(r, g);
                    let rg23 = i32x4_shuffle::<2, 6, 3, 7>(r, g);
                    let gb123 = i32x4_shuffle::<1, 5, 6, 7>(g, b);
                    let out0 = i32x4_shuffle::<0, 1, 4, 2>(rg01, b);
                    let out1 = i32x4_shuffle::<0, 1, 4, 5>(gb123, rg23);
                    let out2 = i32x4_shuffle::<2, 6, 7, 3>(gb123, rg23);
                    (out0, out1, out2)
                }};
                let (q0, q1, q2) = int4(r[0], g[0], b[0]);
                let (q3, q4, q5) = int4(r[1], g[1], b[1]);
                [[q0, q1], [q2, q3], [q4, q5]]
            }}

            #[inline(always)]
            fn load_rgb_bytes(self, rgb: &[u8; 24]) -> ([v128; 2], [v128; 2], [v128; 2]) {{
                let lo16: v128 = crate::simd_storage::copy(rgb.first_chunk::<16>().unwrap());
                let hi8 = u64x2(crate::simd_storage::copy(rgb.last_chunk::<8>().unwrap()), 0);
                let rg8 = i8x16_shuffle::<
                    0, 3, 6, 9, 12, 15, 18, 21, 1, 4, 7, 10, 13, 16, 19, 22,
                >(lo16, hi8);
                let b8 = i8x16_shuffle::<
                    2, 5, 8, 11, 14, 17, 20, 23, 24, 24, 24, 24, 24, 24, 24, 24,
                >(lo16, hi8);
                let r16 = u16x8_extend_low_u8x16(rg8);
                let g16 = u16x8_extend_high_u8x16(rg8);
                let b16 = u16x8_extend_low_u8x16(b8);
                (
                    [
                        f32x4_convert_u32x4(u32x4_extend_low_u16x8(r16)),
                        f32x4_convert_u32x4(u32x4_extend_high_u16x8(r16)),
                    ],
                    [
                        f32x4_convert_u32x4(u32x4_extend_low_u16x8(g16)),
                        f32x4_convert_u32x4(u32x4_extend_high_u16x8(g16)),
                    ],
                    [
                        f32x4_convert_u32x4(u32x4_extend_low_u16x8(b16)),
                        f32x4_convert_u32x4(u32x4_extend_high_u16x8(b16)),
                    ],
                )
            }}

            #[inline(always)]
            fn store_rgb_u8_into(self, r: [v128; 2], g: [v128; 2], b: [v128; 2], out: &mut [u8; 24]) {{
                // Narrow each channel to u8x16 individually before combining so LLVM
                // preserves hardware i16x8.narrow_i32x4_s + i8x16.narrow_i16x8_u.
                let narrow8 = |v: [v128; 2]| -> v128 {{
                    let i16s = i16x8_narrow_i32x4(
                        i32x4_trunc_sat_f32x4(f32x4_nearest(v[0])),
                        i32x4_trunc_sat_f32x4(f32x4_nearest(v[1])),
                    );
                    u8x16_narrow_i16x8(i16s, i16s)
                }};
                let r8 = narrow8(r);
                let g8 = narrow8(g);
                let b8 = narrow8(b);
                let rg8 = i64x2_shuffle::<0, 2>(r8, g8);
                let lo16 = i8x16_shuffle::<
                    0, 8, 16, 1, 9, 17, 2, 10, 18, 3, 11, 19, 4, 12, 20, 5,
                >(rg8, b8);
                let hi8 = i8x16_shuffle::<
                    13, 21, 6, 14, 22, 7, 15, 23, 0, 0, 0, 0, 0, 0, 0, 0,
                >(rg8, b8);
                crate::simd_storage::store(lo16, out.first_chunk_mut::<16>().unwrap());
                crate::simd_storage::store(
                    u64x2_extract_lane::<0>(hi8),
                    out.last_chunk_mut::<8>().unwrap(),
                );
            }}
        }}
    "#}
}

/// Generate WASM impl for native 128-bit float types (`f32x4`, `f64x2`).
fn generate_wasm_float_impl(ty: SimdType) -> String {
    let trait_name = ty.trait_name();
    let elem = ty.elem.name();
    let lanes = ty.lanes();
    let array = ty.array_type();
    let wp = ty.name();
    let is_f32 = ty.elem == ElementType::F32;

    let to_u8_wasm_native = if is_f32 {
        formatdoc! {r#"

            #[inline(always)]
            fn to_u8_bytes(self, a: v128) -> [u8; 4] {{
                let i32s = i32x4_trunc_sat_f32x4(f32x4_nearest(a));
                let i16s = i16x8_narrow_i32x4(i32s, i32s);
                let u8s = u8x16_narrow_i16x8(i16s, i16s);
                (u32x4_extract_lane::<0>(u8s)).to_ne_bytes()
            }}
        "#}
    } else {
        String::new()
    };

    let internal_impl = if is_f32 {
        let float_layout = super::float_layout::wasm();
        formatdoc! {r#"

            #[cfg(target_arch = "wasm32")]
            impl crate::simd::backends::F32x{lanes}Internal for archmage::Wasm128Token {{
                type InternalRepr = v128;
            {float_layout}

                #[inline(always)]
                fn deinterleave_3ch_repr(self, rgb: [v128; 3]) -> [v128; 3] {{
                    let [v0, v1, v2] = rgb;
                    let r = i32x4_shuffle::<0, 3, 6, 0>(v0, v1);
                    let g = i32x4_shuffle::<1, 4, 7, 0>(v0, v1);
                    let b = i32x4_shuffle::<2, 5, 0, 0>(v0, v1);
                    [
                        i32x4_shuffle::<0, 1, 2, 5>(r, v2),
                        i32x4_shuffle::<0, 1, 2, 6>(g, v2),
                        i32x4_shuffle::<0, 1, 4, 7>(b, v2),
                    ]
                }}

                #[inline(always)]
                fn store_rgba_u8_into(self, r: v128, g: v128, b: v128, a: v128, out: &mut [u8; 16]) {{
                    let lo = i32x4_splat(0);
                    let hi = i32x4_splat(255);
                    let clamp = |v: v128| i32x4_min(i32x4_max(i32x4_trunc_sat_f32x4(f32x4_nearest(v)), lo), hi);
                    let pixels = v128_or(
                        v128_or(clamp(r), i32x4_shl(clamp(g), 8)),
                        v128_or(i32x4_shl(clamp(b), 16), i32x4_shl(clamp(a), 24)),
                    );
                    crate::simd_storage::store(pixels, out);
                }}

                #[inline(always)]
                fn interleave_3ch_repr(self, channels: [v128; 3]) -> [v128; 3] {{
                    let [r, g, b] = channels;
                    let rg01 = i32x4_shuffle::<0, 4, 1, 5>(r, g);
                    let rg23 = i32x4_shuffle::<2, 6, 3, 7>(r, g);
                    let gb123 = i32x4_shuffle::<1, 5, 6, 7>(g, b);
                    let out0 = i32x4_shuffle::<0, 1, 4, 2>(rg01, b);
                    let out1 = i32x4_shuffle::<0, 1, 4, 5>(gb123, rg23);
                    let out2 = i32x4_shuffle::<2, 6, 7, 3>(gb123, rg23);
                    [out0, out1, out2]
                }}

                #[inline(always)]
                fn load_rgb_bytes(self, rgb: &[u8; 12]) -> (v128, v128, v128) {{
                    let lo8 = u64x2(crate::simd_storage::copy(rgb.first_chunk::<8>().unwrap()), 0);
                    let hi4 = u32x4(
                        crate::simd_storage::copy(rgb.last_chunk::<4>().unwrap()),
                        0,
                        0,
                        0,
                    );
                    // Extract all 12 bytes [R0..3, G0..3, B0..3, 0..0] in one shuffle;
                    // lane 8 references lo8's zeroed high half so LLVM emits v128.load64_zero.
                    let rgb8 = i8x16_shuffle::<
                        0, 3, 6, 17, 1, 4, 7, 18, 2, 5, 16, 19, 8, 8, 8, 8,
                    >(lo8, hi4);
                    let rg16 = u16x8_extend_low_u8x16(rgb8);
                    let b16 = u16x8_extend_high_u8x16(rgb8);
                    (
                        f32x4_convert_u32x4(u32x4_extend_low_u16x8(rg16)),
                        f32x4_convert_u32x4(u32x4_extend_high_u16x8(rg16)),
                        f32x4_convert_u32x4(u32x4_extend_low_u16x8(b16)),
                    )
                }}

                #[inline(always)]
                fn store_rgb_u8_into(self, r: v128, g: v128, b: v128, out: &mut [u8; 12]) {{
                    let narrow4 = |v: v128| -> v128 {{
                        let i32s = i32x4_trunc_sat_f32x4(f32x4_nearest(v));
                        let i16s = i16x8_narrow_i32x4(i32s, i32s);
                        u8x16_narrow_i16x8(i16s, i16s)
                    }};
                    let rg8 = i32x4_shuffle::<0, 4, 0, 0>(narrow4(r), narrow4(g));
                    let packed = i8x16_shuffle::<
                        0, 4, 16, 1, 5, 17, 2, 6, 18, 3, 7, 19, 0, 0, 0, 0,
                    >(rg8, narrow4(b));
                    crate::simd_storage::store(
                        u64x2_extract_lane::<0>(packed),
                        out.first_chunk_mut::<8>().unwrap(),
                    );
                    crate::simd_storage::store(
                        u32x4_extract_lane::<2>(packed),
                        out.last_chunk_mut::<4>().unwrap(),
                    );
                }}
            }}"#}
    } else {
        String::new()
    };

    // Adjacent-pair tree for f32x4 to match x86/NEON shape (#50).
    let reduce_add = if is_f32 {
        "({wp}_extract_lane::<0>(a) + {wp}_extract_lane::<1>(a)) + ({wp}_extract_lane::<2>(a) + {wp}_extract_lane::<3>(a))"
            .replace("{wp}", &wp)
    } else {
        (0..lanes)
            .map(|j| format!("{wp}_extract_lane::<{j}>(a)"))
            .collect::<Vec<_>>()
            .join(" + ")
    };

    let reduce_minmax = |fold_method: &str| -> String {
        let extracts: Vec<String> = (0..lanes)
            .map(|j| format!("let v{j} = {wp}_extract_lane::<{j}>(a);"))
            .collect();
        let fold = if lanes == 4 {
            format!("v0.{fold_method}(v1).{fold_method}(v2.{fold_method}(v3))")
        } else {
            format!("v0.{fold_method}(v1)")
        };
        format!("{}\n        {fold}", extracts.join("\n        "))
    };

    let recip_section = if is_f32 {
        r#"
            #[inline(always)]
            fn rcp_approx(self, a: v128) -> v128 { f32x4_div(f32x4_splat(1.0), a) }
            #[inline(always)]
            fn rsqrt_approx(self, a: v128) -> v128 {
                let onehalf = f32x4_splat(1.5);
                let hx = f32x4_mul(f32x4_splat(0.5), a);
                let seed = i32x4_sub(i32x4_splat(0x5f37_59df_i32), u32x4_shr(a, 1));
                let y = f32x4_mul(seed, f32x4_sub(onehalf, f32x4_mul(f32x4_mul(hx, seed), seed)));
                f32x4_mul(y, f32x4_sub(onehalf, f32x4_mul(f32x4_mul(hx, y), y)))
            }
            #[inline(always)]
            fn recip(self, a: v128) -> v128 { f32x4_div(f32x4_splat(1.0), a) }
            #[inline(always)]
            fn rsqrt(self, a: v128) -> v128 { f32x4_div(f32x4_splat(1.0), f32x4_sqrt(a)) }
"#
        .to_string()
    } else {
        formatdoc! {r#"
            #[inline(always)]
            fn rcp_approx(self, a: v128) -> v128 {{ {wp}_div({wp}_splat(1.0), a) }}
            #[inline(always)]
            fn rsqrt_approx(self, a: v128) -> v128 {{ {wp}_div({wp}_splat(1.0), {wp}_sqrt(a)) }}
            #[inline(always)]
            fn recip(self, a: v128) -> v128 {{ {wp}_div({wp}_splat(1.0), a) }}
            #[inline(always)]
            fn rsqrt(self, a: v128) -> v128 {{ {wp}_div({wp}_splat(1.0), {wp}_sqrt(a)) }}
        "#}
    };

    formatdoc! {r#"
        impl {trait_name} for archmage::Wasm128Token {{
            type Repr = v128;

            #[inline(always)]
            fn splat(self, v: {elem}) -> v128 {{ {wp}_splat(v) }}
            #[inline(always)]
            fn zero(self) -> v128 {{ {wp}_splat(0.0) }}
            #[inline(always)]
            fn load(self, data: &{array}) -> v128 {{ crate::simd_storage::copy(data) }}
            #[inline(always)]
            fn from_array(self, arr: {array}) -> v128 {{ crate::simd_storage::cast(arr) }}
            #[inline(always)]
            fn store(self, repr: v128, out: &mut {array}) {{ crate::simd_storage::store(repr, out); }}
            #[inline(always)]
            fn to_array(self, repr: v128) -> {array} {{
                let mut out = [0.0{elem}; {lanes}];
                crate::simd_storage::store(repr, &mut out);
                out
            }}

            #[inline(always)]
            fn add(self, a: v128, b: v128) -> v128 {{ {wp}_add(a, b) }}
            #[inline(always)]
            fn sub(self, a: v128, b: v128) -> v128 {{ {wp}_sub(a, b) }}
            #[inline(always)]
            fn mul(self, a: v128, b: v128) -> v128 {{ {wp}_mul(a, b) }}
            #[inline(always)]
            fn div(self, a: v128, b: v128) -> v128 {{ {wp}_div(a, b) }}
            #[inline(always)]
            fn neg(self, a: v128) -> v128 {{ {wp}_neg(a) }}
            #[inline(always)]
            fn min(self, a: v128, b: v128) -> v128 {{ v128_bitselect(a, {wp}_pmin(b, a), {wp}_ne(b, b)) }}
            #[inline(always)]
            fn max(self, a: v128, b: v128) -> v128 {{ v128_bitselect(a, {wp}_pmax(b, a), {wp}_ne(b, b)) }}
            #[inline(always)]
            fn minimum(self, a: v128, b: v128) -> v128 {{ {wp}_min(a, b) }}
            #[inline(always)]
            fn maximum(self, a: v128, b: v128) -> v128 {{ {wp}_max(a, b) }}

            #[inline(always)]
            fn min_fast(self, a: v128, b: v128) -> v128 {{ crate::wasm_fma::min_{wp}_fast(self, a, b) }}
            #[inline(always)]
            fn max_fast(self, a: v128, b: v128) -> v128 {{ crate::wasm_fma::max_{wp}_fast(self, a, b) }}

            #[inline(always)]
            fn sqrt(self, a: v128) -> v128 {{ {wp}_sqrt(a) }}
            #[inline(always)]
            fn abs(self, a: v128) -> v128 {{ {wp}_abs(a) }}
            #[inline(always)]
            fn trunc(self, a: v128) -> v128 {{ {wp}_trunc(a) }}

            #[inline(always)]
            fn mul_neg_add_relaxed(self, a: v128, b: v128, c: v128) -> v128 {{ crate::wasm_fma::nmadd_{wp}(self, a, b, c) }}

            #[inline(always)]
            fn mul_neg_add_fused(self, a: v128, b: v128, c: v128) -> v128 {{ crate::wasm_fma::fused_{wp}(self, {wp}_neg(a), b, c) }}

            #[inline(always)]
            fn floor(self, a: v128) -> v128 {{ {wp}_floor(a) }}
            #[inline(always)]
            fn ceil(self, a: v128) -> v128 {{ {wp}_ceil(a) }}
            #[inline(always)]
            fn round(self, a: v128) -> v128 {{ {wp}_nearest(a) }}
            #[inline(always)]
            fn mul_add(self, a: v128, b: v128, c: v128) -> v128 {{ crate::wasm_fma::madd_{wp}(self, a, b, c) }}
            #[inline(always)]
            fn mul_sub(self, a: v128, b: v128, c: v128) -> v128 {{ crate::wasm_fma::msub_{wp}(self, a, b, c) }}
            #[inline(always)]
            fn mul_add_portable(self, a: v128, b: v128, c: v128) -> v128 {{ crate::wasm_fma::fused_{wp}(self, a, b, c) }}
            #[inline(always)]
            fn mul_sub_portable(self, a: v128, b: v128, c: v128) -> v128 {{ crate::wasm_fma::fused_{wp}(self, a, b, {wp}_neg(c)) }}
            #[inline(always)]
            fn simd_eq(self, a: v128, b: v128) -> v128 {{ {wp}_eq(a, b) }}
            #[inline(always)]
            fn simd_ne(self, a: v128, b: v128) -> v128 {{ {wp}_ne(a, b) }}
            #[inline(always)]
            fn simd_lt(self, a: v128, b: v128) -> v128 {{ {wp}_lt(a, b) }}
            #[inline(always)]
            fn simd_le(self, a: v128, b: v128) -> v128 {{ {wp}_le(a, b) }}
            #[inline(always)]
            fn simd_gt(self, a: v128, b: v128) -> v128 {{ {wp}_gt(a, b) }}
            #[inline(always)]
            fn simd_ge(self, a: v128, b: v128) -> v128 {{ {wp}_ge(a, b) }}
            #[inline(always)]
            fn blend(self, mask: v128, if_true: v128, if_false: v128) -> v128 {{
                v128_bitselect(if_true, if_false, mask)
            }}

            #[inline(always)]
            fn reduce_add(self, a: v128) -> {elem} {{ {reduce_add} }}
            #[inline(always)]
            fn reduce_min(self, a: v128) -> {elem} {{
                {reduce_min}
            }}
            #[inline(always)]
            fn reduce_max(self, a: v128) -> {elem} {{
                {reduce_max}
            }}

{recip_section}
            #[inline(always)]
            fn not(self, a: v128) -> v128 {{ v128_not(a) }}
            #[inline(always)]
            fn bitand(self, a: v128, b: v128) -> v128 {{ v128_and(a, b) }}
            #[inline(always)]
            fn bitor(self, a: v128, b: v128) -> v128 {{ v128_or(a, b) }}
            #[inline(always)]
            fn bitxor(self, a: v128, b: v128) -> v128 {{ v128_xor(a, b) }}
            {to_u8_wasm_native}
        }}{internal_impl}
    "#,
        reduce_min = reduce_minmax("min"),
        reduce_max = reduce_minmax("max"),
    }
}

// ============================================================================
// Unified Integer Implementation (all 8 W128 integer types)
// ============================================================================

fn generate_wasm_int_impl(ty: SimdType) -> String {
    let integer_methods = super::backend_gen_widen_narrow::methods(&ty.name(), "Wasm128Token");
    let trait_name = ty.trait_name();
    let array = ty.array_type();
    let elem = ty.elem.name();
    let lanes = ty.lanes();
    let eb = ty.elem_bits();
    let signed = ty.is_signed();
    let wp = ty.name();
    let selem = ty.elem.signed_int_name();
    let signed_wp = format!("{selem}x{lanes}");
    let unsigned_wp = format!("{}x{lanes}", ty.elem.unsigned_int_name());
    let zero_lit = if ty.elem == ElementType::I64 {
        "0i64"
    } else {
        "0"
    };

    // Spacing differences between the historical 32/64-bit emitters and the 8/16/u64 emitter
    let compact_hdr = matches!(
        ty.elem,
        ElementType::I32 | ElementType::U32 | ElementType::I64
    );
    let mut body = if compact_hdr {
        formatdoc! {r#"
            impl {trait_name} for archmage::Wasm128Token {{
                type Repr = v128;

                #[inline(always)]
                fn splat(self, v: {elem}) -> v128 {{ {wp}_splat(v) }}
                #[inline(always)]
                fn zero(self) -> v128 {{ {wp}_splat({zero_lit}) }}
                #[inline(always)]
                fn load(self, data: &{array}) -> v128 {{ crate::simd_storage::copy(data) }}
                #[inline(always)]
                fn from_array(self, arr: {array}) -> v128 {{ crate::simd_storage::cast(arr) }}
                #[inline(always)]
                fn store(self, repr: v128, out: &mut {array}) {{ crate::simd_storage::store(repr, out); }}
                #[inline(always)]
                fn to_array(self, repr: v128) -> {array} {{
                    let mut out = [0{elem}; {lanes}];
                    crate::simd_storage::store(repr, &mut out);
                    out
                }}

                #[inline(always)]
                fn add(self, a: v128, b: v128) -> v128 {{ {signed_wp}_add(a, b) }}
                #[inline(always)]
                fn sub(self, a: v128, b: v128) -> v128 {{ {signed_wp}_sub(a, b) }}
        "#}
    } else {
        formatdoc! {r#"
            impl {trait_name} for archmage::Wasm128Token {{
                type Repr = v128;

                #[inline(always)]
                fn splat(self, v: {elem}) -> v128 {{
                    {wp}_splat(v)
                }}

                #[inline(always)]
                fn zero(self) -> v128 {{
                    {wp}_splat(0)
                }}

                #[inline(always)]
                fn load(self, data: &{array}) -> v128 {{
                    crate::simd_storage::copy(data)
                }}

                #[inline(always)]
                fn from_array(self, arr: {array}) -> v128 {{
                    crate::simd_storage::cast(arr)
                }}

                #[inline(always)]
                fn store(self, repr: v128, out: &mut {array}) {{
                    crate::simd_storage::store(repr, out);
                }}

                #[inline(always)]
                fn to_array(self, repr: v128) -> {array} {{
                    let mut out = [0{elem}; {lanes}];
                    crate::simd_storage::store(repr, &mut out);
                    out
                }}

                #[inline(always)]
                fn add(self, a: v128, b: v128) -> v128 {{ {signed_wp}_add(a, b) }}
                #[inline(always)]
                fn sub(self, a: v128, b: v128) -> v128 {{ {signed_wp}_sub(a, b) }}
        "#}
    };

    if matches!(eb, 16 | 32) {
        body.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn mul(self, a: v128, b: v128) -> v128 {{ {signed_wp}_mul(a, b) }}
        "#});
    }

    if signed {
        body.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn neg(self, a: v128) -> v128 {{ {signed_wp}_neg(a) }}
        "#});
    }

    match ty.elem {
        ElementType::I64 => {
            body.push_str(&formatdoc! {r#"
                #[inline(always)]
                fn min(self, a: v128, b: v128) -> v128 {{
                    // WASM SIMD lacks native i64 min; polyfill via compare+select
                    let mask = i64x2_gt(a, b);
                    v128_bitselect(b, a, mask)
                }}
                #[inline(always)]
                fn max(self, a: v128, b: v128) -> v128 {{
                    // WASM SIMD lacks native i64 max; polyfill via compare+select
                    let mask = i64x2_gt(a, b);
                    v128_bitselect(a, b, mask)
                }}
                #[inline(always)]
                fn abs(self, a: v128) -> v128 {{
                    // Polyfill: negate negative values
                    let negated = i64x2_neg(a);
                    let zero = i64x2_splat(0i64);
                    let mask = i64x2_lt(a, zero);
                    v128_bitselect(negated, a, mask)
                }}
            "#});
        }
        ElementType::U64 => {
            body.push_str(&formatdoc! {r#"
                #[inline(always)]
                fn min(self, a: v128, b: v128) -> v128 {{
                    // u64 min polyfill: compare then select
                    let mask = <Self as {trait_name}>::simd_lt(self, a, b);
                    v128_bitselect(a, b, mask)
                }}
                #[inline(always)]
                fn max(self, a: v128, b: v128) -> v128 {{
                    let mask = <Self as {trait_name}>::simd_gt(self, a, b);
                    v128_bitselect(a, b, mask)
                }}
            "#});
        }
        _ => {
            body.push_str(&formatdoc! {r#"
                #[inline(always)]
                fn min(self, a: v128, b: v128) -> v128 {{ {wp}_min(a, b) }}
                #[inline(always)]
                fn max(self, a: v128, b: v128) -> v128 {{ {wp}_max(a, b) }}
            "#});
            if signed {
                body.push_str(&formatdoc! {r#"
                    #[inline(always)]
                    fn abs(self, a: v128) -> v128 {{ {signed_wp}_abs(a) }}
                "#});
            }
        }
    }

    // Comparisons
    if ty.elem == ElementType::U64 {
        body.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn simd_eq(self, a: v128, b: v128) -> v128 {{ i64x2_eq(a, b) }}
            #[inline(always)]
            fn simd_ne(self, a: v128, b: v128) -> v128 {{ i64x2_ne(a, b) }}
            #[inline(always)]
            fn simd_gt(self, a: v128, b: v128) -> v128 {{
                // Unsigned comparison via bias trick
                let bias = i64x2_splat(i64::MIN);
                let sa = v128_xor(a, bias);
                let sb = v128_xor(b, bias);
                i64x2_gt(sa, sb)
            }}
            #[inline(always)]
            fn simd_lt(self, a: v128, b: v128) -> v128 {{
                <Self as {trait_name}>::simd_gt(self, b, a)
            }}
            #[inline(always)]
            fn simd_le(self, a: v128, b: v128) -> v128 {{
                v128_not(<Self as {trait_name}>::simd_gt(self, a, b))
            }}
            #[inline(always)]
            fn simd_ge(self, a: v128, b: v128) -> v128 {{
                v128_not(<Self as {trait_name}>::simd_gt(self, b, a))
            }}
        "#});
    } else {
        body.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn simd_eq(self, a: v128, b: v128) -> v128 {{ {signed_wp}_eq(a, b) }}
            #[inline(always)]
            fn simd_ne(self, a: v128, b: v128) -> v128 {{ {signed_wp}_ne(a, b) }}
            #[inline(always)]
            fn simd_lt(self, a: v128, b: v128) -> v128 {{ {wp}_lt(a, b) }}
            #[inline(always)]
            fn simd_le(self, a: v128, b: v128) -> v128 {{ {wp}_le(a, b) }}
            #[inline(always)]
            fn simd_gt(self, a: v128, b: v128) -> v128 {{ {wp}_gt(a, b) }}
            #[inline(always)]
            fn simd_ge(self, a: v128, b: v128) -> v128 {{ {wp}_ge(a, b) }}
        "#});
    }

    if !compact_hdr {
        body.push('\n');
    }
    body.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn blend(self, mask: v128, if_true: v128, if_false: v128) -> v128 {{
                v128_bitselect(if_true, if_false, mask)
            }}
    "#});

    // Reduce add
    let reduce_add_body = match ty.elem {
        ElementType::I32 | ElementType::I64 => (0..lanes)
            .map(|j| format!("{signed_wp}_extract_lane::<{j}>(a)"))
            .collect::<Vec<_>>()
            .join(" + "),
        ElementType::U32 => {
            (0..lanes)
                .map(|j| {
                    if j + 1 == lanes {
                        format!("i32x4_extract_lane::<{j}>(a) as u32")
                    } else {
                        format!("(i32x4_extract_lane::<{j}>(a) as u32)")
                    }
                })
                .collect::<Vec<_>>()
                .join(".wrapping_add(")
                + &")".repeat(lanes - 1)
        }
        _ => format!(
            "let arr = <Self as {trait_name}>::to_array(self, a);\n                arr.iter().copied().fold(0{elem}, {elem}::wrapping_add)"
        ),
    };
    body.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn reduce_add(self, a: v128) -> {elem} {{
                {reduce_add_body}
            }}

            #[inline(always)]
            fn not(self, a: v128) -> v128 {{ v128_not(a) }}
            #[inline(always)]
            fn bitand(self, a: v128, b: v128) -> v128 {{ v128_and(a, b) }}
            #[inline(always)]
            fn bitor(self, a: v128, b: v128) -> v128 {{ v128_or(a, b) }}
            #[inline(always)]
            fn bitxor(self, a: v128, b: v128) -> v128 {{ v128_xor(a, b) }}
    "#});

    // Const shifts
    let shl_fn = if ty.elem == ElementType::U32 {
        "u32x4_shl".to_string()
    } else {
        format!("{signed_wp}_shl")
    };
    body.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn shl_const<const N: i32>(self, a: v128) -> v128 {{ {shl_fn}(a, N as u32) }}
    "#});
    let shr_arith_const = formatdoc! {r#"
            #[inline(always)]
            fn shr_arithmetic_const<const N: i32>(self, a: v128) -> v128 {{ {signed_wp}_shr(a, N as u32) }}
    "#};
    let shr_logical_const = formatdoc! {r#"
            #[inline(always)]
            fn shr_logical_const<const N: i32>(self, a: v128) -> v128 {{ {unsigned_wp}_shr(a, N as u32) }}
    "#};
    if signed && eb >= 32 {
        body.push_str(&shr_arith_const);
        body.push_str(&shr_logical_const);
    } else {
        body.push_str(&shr_logical_const);
        if signed {
            body.push_str(&shr_arith_const);
        }
    }

    // Uniform variable shifts (16-bit and 32-bit types).
    if matches!(eb, 16 | 32) {
        let max_sh = eb - 1;
        let uniform_header = match ty.elem {
            ElementType::I32 => {
                "\n            // ====== Uniform variable shifts ======\n            // wasm takes the count modulo the lane width; the keep mask\n            // restores the contracted zero at count >= 32, and clamping\n            // gives the arithmetic sign fill.\n"
            }
            ElementType::U32 => {
                "\n            // ====== Uniform variable shifts ======\n            // wasm takes the count modulo the lane width; the keep mask\n            // restores the contracted zero at count >= 32.\n"
            }
            _ => "",
        };
        body.push_str(uniform_header);
        body.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn shl_uniform(self, a: v128, count: u32) -> v128 {{
                let keep: {selem} = if count < {eb} {{ -1 }} else {{ 0 }};
                v128_and({shl_fn}(a, count), {signed_wp}_splat(keep))
            }}

            #[inline(always)]
            fn shr_logical_uniform(self, a: v128, count: u32) -> v128 {{
                let keep: {selem} = if count < {eb} {{ -1 }} else {{ 0 }};
                v128_and({unsigned_wp}_shr(a, count), {signed_wp}_splat(keep))
            }}
        "#});
        if signed {
            if eb == 32 {
                body.push('\n');
            }
            body.push_str(&formatdoc! {r#"
            #[inline(always)]
            fn shr_arithmetic_uniform(self, a: v128, count: u32) -> v128 {{
                {signed_wp}_shr(a, if count > {max_sh} {{ {max_sh} }} else {{ count }})
            }}
            "#});
        }
    }

    if eb <= 16 {
        body.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn saturating_add(self, a: v128, b: v128) -> v128 {{ {wp}_add_sat(a, b) }}

            #[inline(always)]
            fn saturating_sub(self, a: v128, b: v128) -> v128 {{ {wp}_sub_sat(a, b) }}
        "#});
    }

    // Boolean reductions (sign-bit contract)
    let all_true_expr = match ty.elem {
        ElementType::I32 | ElementType::U32 => format!("{signed_wp}_bitmask(a) == 0x0F"),
        ElementType::I64 => format!("{signed_wp}_bitmask(a) == 0x03"),
        _ => {
            let full_mask = (1u64 << lanes) - 1;
            format!("{signed_wp}_bitmask(a) as u64 == {full_mask}")
        }
    };
    body.push_str(&formatdoc! {r#"

            #[inline(always)]
            fn all_true(self, a: v128) -> bool {{ {all_true_expr} }}
            #[inline(always)]
            fn any_true(self, a: v128) -> bool {{ {signed_wp}_bitmask(a) != 0 }}
            #[inline(always)]
            fn bitmask(self, a: v128) -> u32 {{ {signed_wp}_bitmask(a) as u32 }}
        {integer_methods}
        }}
    "#});

    body
}

// ============================================================================
// Conversion & Bitcast Implementations
// ============================================================================

fn generate_wasm_convert_impls() -> String {
    let mut code = formatdoc! {r#"

        #[cfg(target_arch = "wasm32")]
        impl F32x4Convert for archmage::Wasm128Token {{
            #[inline(always)]
            fn bitcast_f32_to_i32(self, a: v128) -> v128 {{ a }}

            #[inline(always)]
            fn bitcast_i32_to_f32(self, a: v128) -> v128 {{ a }}

            #[inline(always)]
            fn convert_f32_to_i32(self, a: v128) -> v128 {{
                i32x4_trunc_sat_f32x4(a)
            }}

            #[inline(always)]
            fn convert_f32_to_i32_fast(self, a: v128) -> v128 {{
                crate::wasm_fma::trunc_f32x4_fast(self, a)
            }}

            #[inline(always)]
            fn convert_f32_to_i32_round(self, a: v128) -> v128 {{
                i32x4_trunc_sat_f32x4(f32x4_nearest(a))
            }}

            #[inline(always)]
            fn convert_i32_to_f32(self, a: v128) -> v128 {{
                f32x4_convert_i32x4(a)
            }}
        }}
    "#};

    const BITCASTS: &[(&str, &str, &str, bool)] = &[
        ("U32x4Bitcast", "u32", "i32", true),
        ("I64x2Bitcast", "i64", "f64", true),
        ("I8x16Bitcast", "i8", "u8", false),
        ("I16x8Bitcast", "i16", "u16", false),
        ("U64x2Bitcast", "u64", "i64", false),
    ];
    for &(tr, from_e, to_e, spaced) in BITCASTS {
        let sep = if spaced { "\n" } else { "" };
        code.push_str(&formatdoc! {r#"

            #[cfg(target_arch = "wasm32")]
            impl {tr} for archmage::Wasm128Token {{
                #[inline(always)]
                fn bitcast_{from_e}_to_{to_e}(self, a: v128) -> v128 {{ a }}{sep}
                #[inline(always)]
                fn bitcast_{to_e}_to_{from_e}(self, a: v128) -> v128 {{ a }}
            }}
        "#});
    }

    code
}
