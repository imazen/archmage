//! Backend implementations for Wasm128Token (WebAssembly SIMD).
//!
//! **Auto-generated** by `cargo xtask generate` - do not edit manually.
//!
//! # Safety (audit contract — checked backend boundaries)
//!
//! Value intrinsics use WebAssembly's safe SIMD API with `Wasm128Token`.
//! Whole-array storage copies use `crate::simd_storage`, whose POD bounds
//! and compile-time size checks cover both v128 and arrays of v128.
//! No raw-pointer load/store operations are generated here.

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

use crate::simd::backends::*;

#[cfg(target_arch = "wasm32")]
impl F32x4Backend for archmage::Wasm128Token {
    type Repr = v128;

    #[inline(always)]
    fn splat(self, v: f32) -> v128 {
        f32x4_splat(v)
    }
    #[inline(always)]
    fn zero(self) -> v128 {
        f32x4_splat(0.0)
    }
    #[inline(always)]
    fn load(self, data: &[f32; 4]) -> v128 {
        crate::simd_storage::copy(data)
    }
    #[inline(always)]
    fn from_array(self, arr: [f32; 4]) -> v128 {
        crate::simd_storage::cast(arr)
    }
    #[inline(always)]
    fn store(self, repr: v128, out: &mut [f32; 4]) {
        crate::simd_storage::store(repr, out);
    }
    #[inline(always)]
    fn to_array(self, repr: v128) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        crate::simd_storage::store(repr, &mut out);
        out
    }

    #[inline(always)]
    fn add(self, a: v128, b: v128) -> v128 {
        f32x4_add(a, b)
    }
    #[inline(always)]
    fn sub(self, a: v128, b: v128) -> v128 {
        f32x4_sub(a, b)
    }
    #[inline(always)]
    fn mul(self, a: v128, b: v128) -> v128 {
        f32x4_mul(a, b)
    }
    #[inline(always)]
    fn div(self, a: v128, b: v128) -> v128 {
        f32x4_div(a, b)
    }
    #[inline(always)]
    fn neg(self, a: v128) -> v128 {
        f32x4_neg(a)
    }
    #[inline(always)]
    fn min(self, a: v128, b: v128) -> v128 {
        v128_bitselect(a, f32x4_pmin(b, a), f32x4_ne(b, b))
    }
    #[inline(always)]
    fn max(self, a: v128, b: v128) -> v128 {
        v128_bitselect(a, f32x4_pmax(b, a), f32x4_ne(b, b))
    }
    #[inline(always)]
    fn minimum(self, a: v128, b: v128) -> v128 {
        f32x4_min(a, b)
    }
    #[inline(always)]
    fn maximum(self, a: v128, b: v128) -> v128 {
        f32x4_max(a, b)
    }

    #[inline(always)]
    fn min_fast(self, a: v128, b: v128) -> v128 {
        crate::wasm_fma::min_f32x4_fast(self, a, b)
    }
    #[inline(always)]
    fn max_fast(self, a: v128, b: v128) -> v128 {
        crate::wasm_fma::max_f32x4_fast(self, a, b)
    }

    #[inline(always)]
    fn sqrt(self, a: v128) -> v128 {
        f32x4_sqrt(a)
    }
    #[inline(always)]
    fn abs(self, a: v128) -> v128 {
        f32x4_abs(a)
    }
    #[inline(always)]
    fn trunc(self, a: v128) -> v128 {
        f32x4_trunc(a)
    }

    #[inline(always)]
    fn mul_neg_add_relaxed(self, a: v128, b: v128, c: v128) -> v128 {
        crate::wasm_fma::nmadd_f32x4(self, a, b, c)
    }

    #[inline(always)]
    fn mul_neg_add_fused(self, a: v128, b: v128, c: v128) -> v128 {
        crate::wasm_fma::fused_f32x4(self, f32x4_neg(a), b, c)
    }

    #[inline(always)]
    fn floor(self, a: v128) -> v128 {
        f32x4_floor(a)
    }
    #[inline(always)]
    fn ceil(self, a: v128) -> v128 {
        f32x4_ceil(a)
    }
    #[inline(always)]
    fn round(self, a: v128) -> v128 {
        f32x4_nearest(a)
    }
    #[inline(always)]
    fn mul_add(self, a: v128, b: v128, c: v128) -> v128 {
        crate::wasm_fma::madd_f32x4(self, a, b, c)
    }
    #[inline(always)]
    fn mul_sub(self, a: v128, b: v128, c: v128) -> v128 {
        crate::wasm_fma::msub_f32x4(self, a, b, c)
    }
    #[inline(always)]
    fn mul_add_portable(self, a: v128, b: v128, c: v128) -> v128 {
        crate::wasm_fma::fused_f32x4(self, a, b, c)
    }
    #[inline(always)]
    fn mul_sub_portable(self, a: v128, b: v128, c: v128) -> v128 {
        crate::wasm_fma::fused_f32x4(self, a, b, f32x4_neg(c))
    }
    #[inline(always)]
    fn simd_eq(self, a: v128, b: v128) -> v128 {
        f32x4_eq(a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: v128, b: v128) -> v128 {
        f32x4_ne(a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: v128, b: v128) -> v128 {
        f32x4_lt(a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: v128, b: v128) -> v128 {
        f32x4_le(a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: v128, b: v128) -> v128 {
        f32x4_gt(a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: v128, b: v128) -> v128 {
        f32x4_ge(a, b)
    }
    #[inline(always)]
    fn blend(self, mask: v128, if_true: v128, if_false: v128) -> v128 {
        v128_bitselect(if_true, if_false, mask)
    }

    #[inline(always)]
    fn reduce_add(self, a: v128) -> f32 {
        (f32x4_extract_lane::<0>(a) + f32x4_extract_lane::<1>(a))
            + (f32x4_extract_lane::<2>(a) + f32x4_extract_lane::<3>(a))
    }
    #[inline(always)]
    fn reduce_min(self, a: v128) -> f32 {
        let v0 = f32x4_extract_lane::<0>(a);
        let v1 = f32x4_extract_lane::<1>(a);
        let v2 = f32x4_extract_lane::<2>(a);
        let v3 = f32x4_extract_lane::<3>(a);
        v0.min(v1).min(v2.min(v3))
    }
    #[inline(always)]
    fn reduce_max(self, a: v128) -> f32 {
        let v0 = f32x4_extract_lane::<0>(a);
        let v1 = f32x4_extract_lane::<1>(a);
        let v2 = f32x4_extract_lane::<2>(a);
        let v3 = f32x4_extract_lane::<3>(a);
        v0.max(v1).max(v2.max(v3))
    }

    #[inline(always)]
    fn rcp_approx(self, a: v128) -> v128 {
        f32x4_div(f32x4_splat(1.0), a)
    }
    #[inline(always)]
    fn rsqrt_approx(self, a: v128) -> v128 {
        let onehalf = f32x4_splat(1.5);
        let hx = f32x4_mul(f32x4_splat(0.5), a);
        let seed = i32x4_sub(i32x4_splat(0x5f37_59df_i32), u32x4_shr(a, 1));
        let y = f32x4_mul(
            seed,
            f32x4_sub(onehalf, f32x4_mul(f32x4_mul(hx, seed), seed)),
        );
        f32x4_mul(y, f32x4_sub(onehalf, f32x4_mul(f32x4_mul(hx, y), y)))
    }
    #[inline(always)]
    fn recip(self, a: v128) -> v128 {
        f32x4_div(f32x4_splat(1.0), a)
    }
    #[inline(always)]
    fn rsqrt(self, a: v128) -> v128 {
        f32x4_div(f32x4_splat(1.0), f32x4_sqrt(a))
    }

    #[inline(always)]
    fn not(self, a: v128) -> v128 {
        v128_not(a)
    }
    #[inline(always)]
    fn bitand(self, a: v128, b: v128) -> v128 {
        v128_and(a, b)
    }
    #[inline(always)]
    fn bitor(self, a: v128, b: v128) -> v128 {
        v128_or(a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: v128, b: v128) -> v128 {
        v128_xor(a, b)
    }

    #[inline(always)]
    fn to_u8_bytes(self, a: v128) -> [u8; 4] {
        let i32s = i32x4_trunc_sat_f32x4(f32x4_nearest(a));
        let i16s = i16x8_narrow_i32x4(i32s, i32s);
        let u8s = u8x16_narrow_i16x8(i16s, i16s);
        (u32x4_extract_lane::<0>(u8s)).to_ne_bytes()
    }
}
#[cfg(target_arch = "wasm32")]
impl F32x4Internal for archmage::Wasm128Token {
    type InternalRepr = v128;
    #[inline(always)]
    fn from_u16_lanes(self, lanes: &[u16; 4]) -> v128 {
        let words = u64x2(crate::simd_storage::cast(*lanes), 0);
        f32x4_convert_u32x4(u32x4_extend_low_u16x8(words))
    }

    #[inline(always)]
    fn to_u16_lanes<const CFG: u32>(self, value: v128) -> [u16; 4] {
        let opts = const { crate::numerics::IntConvert::decode(CFG) };
        let a = if opts.rounds() {
            f32x4_nearest(value)
        } else {
            value
        };
        let i = if const { crate::numerics::IntConvert::decode(CFG).is_in_range() } {
            crate::wasm_fma::trunc_f32x4_fast(self, a)
        } else {
            i32x4_trunc_sat_f32x4(a)
        };
        crate::simd_storage::cast(u64x2_extract_lane::<0>(u16x8_narrow_i32x4(i, i)))
    }

    #[inline(always)]
    fn to_u8_lanes<const CFG: u32>(self, value: v128) -> [u8; 4] {
        let opts = const { crate::numerics::IntConvert::decode(CFG) };
        let a = if opts.rounds() {
            f32x4_nearest(value)
        } else {
            value
        };
        let i = if opts.is_in_range() {
            crate::wasm_fma::trunc_f32x4_fast(self, a)
        } else {
            i32x4_trunc_sat_f32x4(a)
        };
        let n = i16x8_narrow_i32x4(i, i);
        crate::simd_storage::cast(i32x4_extract_lane::<0>(u8x16_narrow_i16x8(n, n)))
    }

    #[inline(always)]
    fn to_u32_lanes<const CFG: u32>(self, value: v128) -> v128 {
        let opts = const { crate::numerics::IntConvert::decode(CFG) };
        let a = if opts.rounds() {
            f32x4_nearest(value)
        } else {
            value
        };
        #[cfg(target_feature = "relaxed-simd")]
        if opts.is_in_range() {
            return crate::wasm_fma::trunc_u32_f32x4_fast(self, a);
        }
        u32x4_trunc_sat_f32x4(a)
    }

    #[inline(always)]
    fn from_u32_lanes(self, value: v128) -> v128 {
        f32x4_convert_u32x4(value)
    }

    #[inline(always)]
    fn from_f16_lanes<const CFG: u32>(self, lanes: &[u16; 4]) -> v128 {
        let words = u64x2(crate::simd_storage::cast(*lanes), 0);
        crate::simd::generic::half4::widen_f16::<CFG, _>(
            crate::simd::generic::i32x4::from_repr_unchecked(self, u32x4_extend_low_u16x8(words)),
        )
        .into_repr()
    }

    #[inline(always)]
    fn to_f16_lanes<const CFG: u32>(self, value: v128) -> [u16; 4] {
        let v = crate::simd::generic::f32x4::from_repr_unchecked(self, value);
        let i = crate::simd::generic::half4::narrow_f16::<CFG, _>(v).into_repr();
        crate::simd_storage::cast(u64x2_extract_lane::<0>(u16x8_narrow_i32x4(i, i)))
    }

    #[inline(always)]
    fn to_bf16_lanes<const CFG: u32>(self, value: v128) -> [u16; 4] {
        let v = crate::simd::generic::f32x4::from_repr_unchecked(self, value);
        let i = crate::simd::generic::half4::narrow_bf16::<CFG, _>(v).into_repr();
        crate::simd_storage::cast(u64x2_extract_lane::<0>(u16x8_narrow_i32x4(i, i)))
    }

    #[inline(always)]
    fn from_bf16_lanes(self, lanes: &[u16; 4]) -> v128 {
        let words = u64x2(crate::simd_storage::cast(*lanes), 0);
        i32x4_shl(u32x4_extend_low_u16x8(words), 16)
    }

    #[inline(always)]
    fn broadcast_rgba_alpha_repr(self, value: v128) -> v128 {
        i32x4_shuffle::<3, 3, 3, 3>(value, value)
    }

    #[inline(always)]
    fn with_rgba_alpha_repr(self, value: v128, src: v128) -> v128 {
        i32x4_shuffle::<0, 1, 2, 7>(value, src)
    }

    #[inline(always)]
    fn transpose_4x4_repr(self, rows: [v128; 4]) -> [v128; 4] {
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
    }

    #[inline(always)]
    fn deinterleave_3ch_repr(self, rgb: [v128; 3]) -> [v128; 3] {
        let [v0, v1, v2] = rgb;
        let r = i32x4_shuffle::<0, 3, 6, 0>(v0, v1);
        let g = i32x4_shuffle::<1, 4, 7, 0>(v0, v1);
        let b = i32x4_shuffle::<2, 5, 0, 0>(v0, v1);
        [
            i32x4_shuffle::<0, 1, 2, 5>(r, v2),
            i32x4_shuffle::<0, 1, 2, 6>(g, v2),
            i32x4_shuffle::<0, 1, 4, 7>(b, v2),
        ]
    }

    #[inline(always)]
    fn store_rgba_u8_into(self, r: v128, g: v128, b: v128, a: v128, out: &mut [u8; 16]) {
        let lo = i32x4_splat(0);
        let hi = i32x4_splat(255);
        let clamp = |v: v128| i32x4_min(i32x4_max(i32x4_trunc_sat_f32x4(f32x4_nearest(v)), lo), hi);
        let pixels = v128_or(
            v128_or(clamp(r), i32x4_shl(clamp(g), 8)),
            v128_or(i32x4_shl(clamp(b), 16), i32x4_shl(clamp(a), 24)),
        );
        crate::simd_storage::store(pixels, out);
    }

    #[inline(always)]
    fn interleave_3ch_repr(self, channels: [v128; 3]) -> [v128; 3] {
        let [r, g, b] = channels;
        let rg01 = i32x4_shuffle::<0, 4, 1, 5>(r, g);
        let rg23 = i32x4_shuffle::<2, 6, 3, 7>(r, g);
        let gb123 = i32x4_shuffle::<1, 5, 6, 7>(g, b);
        let out0 = i32x4_shuffle::<0, 1, 4, 2>(rg01, b);
        let out1 = i32x4_shuffle::<0, 1, 4, 5>(gb123, rg23);
        let out2 = i32x4_shuffle::<2, 6, 7, 3>(gb123, rg23);
        [out0, out1, out2]
    }

    #[inline(always)]
    fn load_rgb_bytes(self, rgb: &[u8; 12]) -> (v128, v128, v128) {
        let lo8 = u64x2(
            crate::simd_storage::copy(rgb.first_chunk::<8>().unwrap()),
            0,
        );
        let hi4 = u32x4(
            crate::simd_storage::copy(rgb.last_chunk::<4>().unwrap()),
            0,
            0,
            0,
        );
        // Extract all 12 bytes [R0..3, G0..3, B0..3, 0..0] in one shuffle;
        // lane 8 references lo8's zeroed high half so LLVM emits v128.load64_zero.
        let rgb8 = i8x16_shuffle::<0, 3, 6, 17, 1, 4, 7, 18, 2, 5, 16, 19, 8, 8, 8, 8>(lo8, hi4);
        let rg16 = u16x8_extend_low_u8x16(rgb8);
        let b16 = u16x8_extend_high_u8x16(rgb8);
        (
            f32x4_convert_u32x4(u32x4_extend_low_u16x8(rg16)),
            f32x4_convert_u32x4(u32x4_extend_high_u16x8(rg16)),
            f32x4_convert_u32x4(u32x4_extend_low_u16x8(b16)),
        )
    }

    #[inline(always)]
    fn store_rgb_u8_into(self, r: v128, g: v128, b: v128, out: &mut [u8; 12]) {
        let narrow4 = |v: v128| -> v128 {
            let i32s = i32x4_trunc_sat_f32x4(f32x4_nearest(v));
            let i16s = i16x8_narrow_i32x4(i32s, i32s);
            u8x16_narrow_i16x8(i16s, i16s)
        };
        let rg8 = i32x4_shuffle::<0, 4, 0, 0>(narrow4(r), narrow4(g));
        let packed =
            i8x16_shuffle::<0, 4, 16, 1, 5, 17, 2, 6, 18, 3, 7, 19, 0, 0, 0, 0>(rg8, narrow4(b));
        crate::simd_storage::store(
            u64x2_extract_lane::<0>(packed),
            out.first_chunk_mut::<8>().unwrap(),
        );
        crate::simd_storage::store(
            u32x4_extract_lane::<2>(packed),
            out.last_chunk_mut::<4>().unwrap(),
        );
    }
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
pub(super) fn f32x8_to_u8_bytes(_token: archmage::Wasm128Token, a: [v128; 2]) -> [u8; 8] {
    let i0 = i32x4_trunc_sat_f32x4(f32x4_nearest(a[0]));
    let i1 = i32x4_trunc_sat_f32x4(f32x4_nearest(a[1]));
    let i16s = i16x8_narrow_i32x4(i0, i1);
    let u8s = u8x16_narrow_i16x8(i16s, i16s);
    let lo = u32x4_extract_lane::<0>(u8s);
    let hi = u32x4_extract_lane::<1>(u8s);
    ((u64::from(hi) << 32) | u64::from(lo)).to_ne_bytes()
}

#[cfg(target_arch = "wasm32")]
impl F32x8Internal for archmage::Wasm128Token {
    type InternalRepr = [v128; 2];
    #[inline(always)]
    fn from_u16_lanes(self, lanes: &[u16; 8]) -> [v128; 2] {
        let (chunks, _) = lanes.as_chunks::<4>();
        [
            <Self as F32x4Internal>::from_u16_lanes(self, &chunks[0]),
            <Self as F32x4Internal>::from_u16_lanes(self, &chunks[1]),
        ]
    }

    #[inline(always)]
    fn to_u16_lanes<const CFG: u32>(self, value: [v128; 2]) -> [u16; 8] {
        crate::simd_storage::cast([
            <Self as F32x4Internal>::to_u16_lanes::<CFG>(self, value[0]),
            <Self as F32x4Internal>::to_u16_lanes::<CFG>(self, value[1]),
        ])
    }

    #[inline(always)]
    fn to_u8_lanes<const CFG: u32>(self, value: [v128; 2]) -> [u8; 8] {
        crate::simd_storage::cast([
            <Self as F32x4Internal>::to_u8_lanes::<CFG>(self, value[0]),
            <Self as F32x4Internal>::to_u8_lanes::<CFG>(self, value[1]),
        ])
    }

    #[inline(always)]
    fn to_u32_lanes<const CFG: u32>(self, value: [v128; 2]) -> [v128; 2] {
        [
            <Self as F32x4Internal>::to_u32_lanes::<CFG>(self, value[0]),
            <Self as F32x4Internal>::to_u32_lanes::<CFG>(self, value[1]),
        ]
    }

    #[inline(always)]
    fn from_u32_lanes(self, value: [v128; 2]) -> [v128; 2] {
        [
            <Self as F32x4Internal>::from_u32_lanes(self, value[0]),
            <Self as F32x4Internal>::from_u32_lanes(self, value[1]),
        ]
    }

    #[inline(always)]
    fn from_f16_lanes<const CFG: u32>(self, lanes: &[u16; 8]) -> [v128; 2] {
        let (chunks, _) = lanes.as_chunks::<4>();
        [
            <Self as F32x4Internal>::from_f16_lanes::<CFG>(self, &chunks[0]),
            <Self as F32x4Internal>::from_f16_lanes::<CFG>(self, &chunks[1]),
        ]
    }

    #[inline(always)]
    fn to_f16_lanes<const CFG: u32>(self, value: [v128; 2]) -> [u16; 8] {
        crate::simd_storage::cast([
            <Self as F32x4Internal>::to_f16_lanes::<CFG>(self, value[0]),
            <Self as F32x4Internal>::to_f16_lanes::<CFG>(self, value[1]),
        ])
    }

    #[inline(always)]
    fn from_bf16_lanes(self, lanes: &[u16; 8]) -> [v128; 2] {
        let (chunks, _) = lanes.as_chunks::<4>();
        [
            <Self as F32x4Internal>::from_bf16_lanes(self, &chunks[0]),
            <Self as F32x4Internal>::from_bf16_lanes(self, &chunks[1]),
        ]
    }

    #[inline(always)]
    fn to_bf16_lanes<const CFG: u32>(self, value: [v128; 2]) -> [u16; 8] {
        crate::simd_storage::cast([
            <Self as F32x4Internal>::to_bf16_lanes::<CFG>(self, value[0]),
            <Self as F32x4Internal>::to_bf16_lanes::<CFG>(self, value[1]),
        ])
    }

    #[inline(always)]
    fn broadcast_rgba_alpha_repr(self, value: [v128; 2]) -> [v128; 2] {
        [
            <Self as F32x4Internal>::broadcast_rgba_alpha_repr(self, value[0]),
            <Self as F32x4Internal>::broadcast_rgba_alpha_repr(self, value[1]),
        ]
    }

    #[inline(always)]
    fn with_rgba_alpha_repr(self, value: [v128; 2], src: [v128; 2]) -> [v128; 2] {
        [
            <Self as F32x4Internal>::with_rgba_alpha_repr(self, value[0], src[0]),
            <Self as F32x4Internal>::with_rgba_alpha_repr(self, value[1], src[1]),
        ]
    }

    #[inline(always)]
    fn deinterleave_3ch_repr(self, rgb: [[v128; 2]; 3]) -> [[v128; 2]; 3] {
        let [[v0, v1], [v2, v3], [v4, v5]] = rgb;
        let [r0, g0, b0] = <Self as F32x4Internal>::deinterleave_3ch_repr(self, [v0, v1, v2]);
        let [r1, g1, b1] = <Self as F32x4Internal>::deinterleave_3ch_repr(self, [v3, v4, v5]);
        [[r0, r1], [g0, g1], [b0, b1]]
    }

    #[inline(always)]
    fn store_rgba_u8_into(
        self,
        r: [v128; 2],
        g: [v128; 2],
        b: [v128; 2],
        a: [v128; 2],
        out: &mut [u8; 32],
    ) {
        let lo = i32x4_splat(0);
        let hi = i32x4_splat(255);
        let clamp = |v: v128| i32x4_min(i32x4_max(i32x4_trunc_sat_f32x4(f32x4_nearest(v)), lo), hi);
        let pack = |r: v128, g: v128, b: v128, a: v128| {
            v128_or(
                v128_or(clamp(r), i32x4_shl(clamp(g), 8)),
                v128_or(i32x4_shl(clamp(b), 16), i32x4_shl(clamp(a), 24)),
            )
        };
        crate::simd_storage::store(
            [pack(r[0], g[0], b[0], a[0]), pack(r[1], g[1], b[1], a[1])],
            out,
        );
    }

    #[inline(always)]
    fn interleave_3ch_repr(self, channels: [[v128; 2]; 3]) -> [[v128; 2]; 3] {
        let [r, g, b] = channels;
        let int4 = |r: v128, g: v128, b: v128| -> (v128, v128, v128) {
            let rg01 = i32x4_shuffle::<0, 4, 1, 5>(r, g);
            let rg23 = i32x4_shuffle::<2, 6, 3, 7>(r, g);
            let gb123 = i32x4_shuffle::<1, 5, 6, 7>(g, b);
            let out0 = i32x4_shuffle::<0, 1, 4, 2>(rg01, b);
            let out1 = i32x4_shuffle::<0, 1, 4, 5>(gb123, rg23);
            let out2 = i32x4_shuffle::<2, 6, 7, 3>(gb123, rg23);
            (out0, out1, out2)
        };
        let (q0, q1, q2) = int4(r[0], g[0], b[0]);
        let (q3, q4, q5) = int4(r[1], g[1], b[1]);
        [[q0, q1], [q2, q3], [q4, q5]]
    }

    #[inline(always)]
    fn load_rgb_bytes(self, rgb: &[u8; 24]) -> ([v128; 2], [v128; 2], [v128; 2]) {
        let lo16: v128 = crate::simd_storage::copy(rgb.first_chunk::<16>().unwrap());
        let hi8 = u64x2(crate::simd_storage::copy(rgb.last_chunk::<8>().unwrap()), 0);
        let rg8 =
            i8x16_shuffle::<0, 3, 6, 9, 12, 15, 18, 21, 1, 4, 7, 10, 13, 16, 19, 22>(lo16, hi8);
        let b8 =
            i8x16_shuffle::<2, 5, 8, 11, 14, 17, 20, 23, 24, 24, 24, 24, 24, 24, 24, 24>(lo16, hi8);
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
    }

    #[inline(always)]
    fn store_rgb_u8_into(self, r: [v128; 2], g: [v128; 2], b: [v128; 2], out: &mut [u8; 24]) {
        // Narrow each channel to u8x16 individually before combining so LLVM
        // preserves hardware i16x8.narrow_i32x4_s + i8x16.narrow_i16x8_u.
        let narrow8 = |v: [v128; 2]| -> v128 {
            let i16s = i16x8_narrow_i32x4(
                i32x4_trunc_sat_f32x4(f32x4_nearest(v[0])),
                i32x4_trunc_sat_f32x4(f32x4_nearest(v[1])),
            );
            u8x16_narrow_i16x8(i16s, i16s)
        };
        let r8 = narrow8(r);
        let g8 = narrow8(g);
        let b8 = narrow8(b);
        let rg8 = i64x2_shuffle::<0, 2>(r8, g8);
        let lo16 = i8x16_shuffle::<0, 8, 16, 1, 9, 17, 2, 10, 18, 3, 11, 19, 4, 12, 20, 5>(rg8, b8);
        let hi8 = i8x16_shuffle::<13, 21, 6, 14, 22, 7, 15, 23, 0, 0, 0, 0, 0, 0, 0, 0>(rg8, b8);
        crate::simd_storage::store(lo16, out.first_chunk_mut::<16>().unwrap());
        crate::simd_storage::store(
            u64x2_extract_lane::<0>(hi8),
            out.last_chunk_mut::<8>().unwrap(),
        );
    }
}

#[cfg(target_arch = "wasm32")]
impl F64x2Backend for archmage::Wasm128Token {
    type Repr = v128;

    #[inline(always)]
    fn splat(self, v: f64) -> v128 {
        f64x2_splat(v)
    }
    #[inline(always)]
    fn zero(self) -> v128 {
        f64x2_splat(0.0)
    }
    #[inline(always)]
    fn load(self, data: &[f64; 2]) -> v128 {
        crate::simd_storage::copy(data)
    }
    #[inline(always)]
    fn from_array(self, arr: [f64; 2]) -> v128 {
        crate::simd_storage::cast(arr)
    }
    #[inline(always)]
    fn store(self, repr: v128, out: &mut [f64; 2]) {
        crate::simd_storage::store(repr, out);
    }
    #[inline(always)]
    fn to_array(self, repr: v128) -> [f64; 2] {
        let mut out = [0.0f64; 2];
        crate::simd_storage::store(repr, &mut out);
        out
    }

    #[inline(always)]
    fn add(self, a: v128, b: v128) -> v128 {
        f64x2_add(a, b)
    }
    #[inline(always)]
    fn sub(self, a: v128, b: v128) -> v128 {
        f64x2_sub(a, b)
    }
    #[inline(always)]
    fn mul(self, a: v128, b: v128) -> v128 {
        f64x2_mul(a, b)
    }
    #[inline(always)]
    fn div(self, a: v128, b: v128) -> v128 {
        f64x2_div(a, b)
    }
    #[inline(always)]
    fn neg(self, a: v128) -> v128 {
        f64x2_neg(a)
    }
    #[inline(always)]
    fn min(self, a: v128, b: v128) -> v128 {
        v128_bitselect(a, f64x2_pmin(b, a), f64x2_ne(b, b))
    }
    #[inline(always)]
    fn max(self, a: v128, b: v128) -> v128 {
        v128_bitselect(a, f64x2_pmax(b, a), f64x2_ne(b, b))
    }
    #[inline(always)]
    fn minimum(self, a: v128, b: v128) -> v128 {
        f64x2_min(a, b)
    }
    #[inline(always)]
    fn maximum(self, a: v128, b: v128) -> v128 {
        f64x2_max(a, b)
    }

    #[inline(always)]
    fn min_fast(self, a: v128, b: v128) -> v128 {
        crate::wasm_fma::min_f64x2_fast(self, a, b)
    }
    #[inline(always)]
    fn max_fast(self, a: v128, b: v128) -> v128 {
        crate::wasm_fma::max_f64x2_fast(self, a, b)
    }

    #[inline(always)]
    fn sqrt(self, a: v128) -> v128 {
        f64x2_sqrt(a)
    }
    #[inline(always)]
    fn abs(self, a: v128) -> v128 {
        f64x2_abs(a)
    }
    #[inline(always)]
    fn trunc(self, a: v128) -> v128 {
        f64x2_trunc(a)
    }

    #[inline(always)]
    fn mul_neg_add_relaxed(self, a: v128, b: v128, c: v128) -> v128 {
        crate::wasm_fma::nmadd_f64x2(self, a, b, c)
    }

    #[inline(always)]
    fn mul_neg_add_fused(self, a: v128, b: v128, c: v128) -> v128 {
        crate::wasm_fma::fused_f64x2(self, f64x2_neg(a), b, c)
    }

    #[inline(always)]
    fn floor(self, a: v128) -> v128 {
        f64x2_floor(a)
    }
    #[inline(always)]
    fn ceil(self, a: v128) -> v128 {
        f64x2_ceil(a)
    }
    #[inline(always)]
    fn round(self, a: v128) -> v128 {
        f64x2_nearest(a)
    }
    #[inline(always)]
    fn mul_add(self, a: v128, b: v128, c: v128) -> v128 {
        crate::wasm_fma::madd_f64x2(self, a, b, c)
    }
    #[inline(always)]
    fn mul_sub(self, a: v128, b: v128, c: v128) -> v128 {
        crate::wasm_fma::msub_f64x2(self, a, b, c)
    }
    #[inline(always)]
    fn mul_add_portable(self, a: v128, b: v128, c: v128) -> v128 {
        crate::wasm_fma::fused_f64x2(self, a, b, c)
    }
    #[inline(always)]
    fn mul_sub_portable(self, a: v128, b: v128, c: v128) -> v128 {
        crate::wasm_fma::fused_f64x2(self, a, b, f64x2_neg(c))
    }
    #[inline(always)]
    fn simd_eq(self, a: v128, b: v128) -> v128 {
        f64x2_eq(a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: v128, b: v128) -> v128 {
        f64x2_ne(a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: v128, b: v128) -> v128 {
        f64x2_lt(a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: v128, b: v128) -> v128 {
        f64x2_le(a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: v128, b: v128) -> v128 {
        f64x2_gt(a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: v128, b: v128) -> v128 {
        f64x2_ge(a, b)
    }
    #[inline(always)]
    fn blend(self, mask: v128, if_true: v128, if_false: v128) -> v128 {
        v128_bitselect(if_true, if_false, mask)
    }

    #[inline(always)]
    fn reduce_add(self, a: v128) -> f64 {
        f64x2_extract_lane::<0>(a) + f64x2_extract_lane::<1>(a)
    }
    #[inline(always)]
    fn reduce_min(self, a: v128) -> f64 {
        let v0 = f64x2_extract_lane::<0>(a);
        let v1 = f64x2_extract_lane::<1>(a);
        v0.min(v1)
    }
    #[inline(always)]
    fn reduce_max(self, a: v128) -> f64 {
        let v0 = f64x2_extract_lane::<0>(a);
        let v1 = f64x2_extract_lane::<1>(a);
        v0.max(v1)
    }

    #[inline(always)]
    fn rcp_approx(self, a: v128) -> v128 {
        f64x2_div(f64x2_splat(1.0), a)
    }
    #[inline(always)]
    fn rsqrt_approx(self, a: v128) -> v128 {
        f64x2_div(f64x2_splat(1.0), f64x2_sqrt(a))
    }
    #[inline(always)]
    fn recip(self, a: v128) -> v128 {
        f64x2_div(f64x2_splat(1.0), a)
    }
    #[inline(always)]
    fn rsqrt(self, a: v128) -> v128 {
        f64x2_div(f64x2_splat(1.0), f64x2_sqrt(a))
    }

    #[inline(always)]
    fn not(self, a: v128) -> v128 {
        v128_not(a)
    }
    #[inline(always)]
    fn bitand(self, a: v128, b: v128) -> v128 {
        v128_and(a, b)
    }
    #[inline(always)]
    fn bitor(self, a: v128, b: v128) -> v128 {
        v128_or(a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: v128, b: v128) -> v128 {
        v128_xor(a, b)
    }
}

#[cfg(target_arch = "wasm32")]
impl I32x4Backend for archmage::Wasm128Token {
    type Repr = v128;

    #[inline(always)]
    fn splat(self, v: i32) -> v128 {
        i32x4_splat(v)
    }
    #[inline(always)]
    fn zero(self) -> v128 {
        i32x4_splat(0)
    }
    #[inline(always)]
    fn load(self, data: &[i32; 4]) -> v128 {
        crate::simd_storage::copy(data)
    }
    #[inline(always)]
    fn from_array(self, arr: [i32; 4]) -> v128 {
        crate::simd_storage::cast(arr)
    }
    #[inline(always)]
    fn store(self, repr: v128, out: &mut [i32; 4]) {
        crate::simd_storage::store(repr, out);
    }
    #[inline(always)]
    fn to_array(self, repr: v128) -> [i32; 4] {
        let mut out = [0i32; 4];
        crate::simd_storage::store(repr, &mut out);
        out
    }

    #[inline(always)]
    fn add(self, a: v128, b: v128) -> v128 {
        i32x4_add(a, b)
    }
    #[inline(always)]
    fn sub(self, a: v128, b: v128) -> v128 {
        i32x4_sub(a, b)
    }
    #[inline(always)]
    fn mul(self, a: v128, b: v128) -> v128 {
        i32x4_mul(a, b)
    }
    #[inline(always)]
    fn neg(self, a: v128) -> v128 {
        i32x4_neg(a)
    }
    #[inline(always)]
    fn min(self, a: v128, b: v128) -> v128 {
        i32x4_min(a, b)
    }
    #[inline(always)]
    fn max(self, a: v128, b: v128) -> v128 {
        i32x4_max(a, b)
    }
    #[inline(always)]
    fn abs(self, a: v128) -> v128 {
        i32x4_abs(a)
    }

    #[inline(always)]
    fn simd_eq(self, a: v128, b: v128) -> v128 {
        i32x4_eq(a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: v128, b: v128) -> v128 {
        i32x4_ne(a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: v128, b: v128) -> v128 {
        i32x4_lt(a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: v128, b: v128) -> v128 {
        i32x4_le(a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: v128, b: v128) -> v128 {
        i32x4_gt(a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: v128, b: v128) -> v128 {
        i32x4_ge(a, b)
    }
    #[inline(always)]
    fn blend(self, mask: v128, if_true: v128, if_false: v128) -> v128 {
        v128_bitselect(if_true, if_false, mask)
    }

    #[inline(always)]
    fn reduce_add(self, a: v128) -> i32 {
        i32x4_extract_lane::<0>(a)
            + i32x4_extract_lane::<1>(a)
            + i32x4_extract_lane::<2>(a)
            + i32x4_extract_lane::<3>(a)
    }

    #[inline(always)]
    fn not(self, a: v128) -> v128 {
        v128_not(a)
    }
    #[inline(always)]
    fn bitand(self, a: v128, b: v128) -> v128 {
        v128_and(a, b)
    }
    #[inline(always)]
    fn bitor(self, a: v128, b: v128) -> v128 {
        v128_or(a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: v128, b: v128) -> v128 {
        v128_xor(a, b)
    }

    #[inline(always)]
    fn shl_const<const N: i32>(self, a: v128) -> v128 {
        i32x4_shl(a, N as u32)
    }
    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: v128) -> v128 {
        i32x4_shr(a, N as u32)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: v128) -> v128 {
        u32x4_shr(a, N as u32)
    }

    // ====== Uniform variable shifts ======
    // wasm takes the count modulo the lane width; the keep mask
    // restores the contracted zero at count >= 32, and clamping
    // gives the arithmetic sign fill.

    #[inline(always)]
    fn shl_uniform(self, a: v128, count: u32) -> v128 {
        let keep: i32 = if count < 32 { -1 } else { 0 };
        v128_and(i32x4_shl(a, count), i32x4_splat(keep))
    }

    #[inline(always)]
    fn shr_logical_uniform(self, a: v128, count: u32) -> v128 {
        let keep: i32 = if count < 32 { -1 } else { 0 };
        v128_and(u32x4_shr(a, count), i32x4_splat(keep))
    }

    #[inline(always)]
    fn shr_arithmetic_uniform(self, a: v128, count: u32) -> v128 {
        i32x4_shr(a, if count > 31 { 31 } else { count })
    }

    #[inline(always)]
    fn all_true(self, a: v128) -> bool {
        i32x4_bitmask(a) == 0x0F
    }
    #[inline(always)]
    fn any_true(self, a: v128) -> bool {
        i32x4_bitmask(a) != 0
    }
    #[inline(always)]
    fn bitmask(self, a: v128) -> u32 {
        i32x4_bitmask(a) as u32
    }
    #[inline(always)]
    fn narrow_saturating_i32_to_i16(self, a: v128, b: v128) -> v128 {
        i16x8_narrow_i32x4(a, b)
    }

    #[inline(always)]
    fn narrow_saturating_i32_to_u16(self, a: v128, b: v128) -> v128 {
        u16x8_narrow_i32x4(a, b)
    }
}

#[cfg(target_arch = "wasm32")]
impl U32x4Backend for archmage::Wasm128Token {
    type Repr = v128;

    #[inline(always)]
    fn splat(self, v: u32) -> v128 {
        u32x4_splat(v)
    }
    #[inline(always)]
    fn zero(self) -> v128 {
        u32x4_splat(0)
    }
    #[inline(always)]
    fn load(self, data: &[u32; 4]) -> v128 {
        crate::simd_storage::copy(data)
    }
    #[inline(always)]
    fn from_array(self, arr: [u32; 4]) -> v128 {
        crate::simd_storage::cast(arr)
    }
    #[inline(always)]
    fn store(self, repr: v128, out: &mut [u32; 4]) {
        crate::simd_storage::store(repr, out);
    }
    #[inline(always)]
    fn to_array(self, repr: v128) -> [u32; 4] {
        let mut out = [0u32; 4];
        crate::simd_storage::store(repr, &mut out);
        out
    }

    #[inline(always)]
    fn add(self, a: v128, b: v128) -> v128 {
        i32x4_add(a, b)
    }
    #[inline(always)]
    fn sub(self, a: v128, b: v128) -> v128 {
        i32x4_sub(a, b)
    }
    #[inline(always)]
    fn mul(self, a: v128, b: v128) -> v128 {
        i32x4_mul(a, b)
    }
    #[inline(always)]
    fn min(self, a: v128, b: v128) -> v128 {
        u32x4_min(a, b)
    }
    #[inline(always)]
    fn max(self, a: v128, b: v128) -> v128 {
        u32x4_max(a, b)
    }

    #[inline(always)]
    fn simd_eq(self, a: v128, b: v128) -> v128 {
        i32x4_eq(a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: v128, b: v128) -> v128 {
        i32x4_ne(a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: v128, b: v128) -> v128 {
        u32x4_lt(a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: v128, b: v128) -> v128 {
        u32x4_le(a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: v128, b: v128) -> v128 {
        u32x4_gt(a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: v128, b: v128) -> v128 {
        u32x4_ge(a, b)
    }
    #[inline(always)]
    fn blend(self, mask: v128, if_true: v128, if_false: v128) -> v128 {
        v128_bitselect(if_true, if_false, mask)
    }

    #[inline(always)]
    fn reduce_add(self, a: v128) -> u32 {
        (i32x4_extract_lane::<0>(a) as u32).wrapping_add(
            (i32x4_extract_lane::<1>(a) as u32).wrapping_add(
                (i32x4_extract_lane::<2>(a) as u32).wrapping_add(i32x4_extract_lane::<3>(a) as u32),
            ),
        )
    }

    #[inline(always)]
    fn not(self, a: v128) -> v128 {
        v128_not(a)
    }
    #[inline(always)]
    fn bitand(self, a: v128, b: v128) -> v128 {
        v128_and(a, b)
    }
    #[inline(always)]
    fn bitor(self, a: v128, b: v128) -> v128 {
        v128_or(a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: v128, b: v128) -> v128 {
        v128_xor(a, b)
    }

    #[inline(always)]
    fn shl_const<const N: i32>(self, a: v128) -> v128 {
        u32x4_shl(a, N as u32)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: v128) -> v128 {
        u32x4_shr(a, N as u32)
    }

    // ====== Uniform variable shifts ======
    // wasm takes the count modulo the lane width; the keep mask
    // restores the contracted zero at count >= 32.

    #[inline(always)]
    fn shl_uniform(self, a: v128, count: u32) -> v128 {
        let keep: i32 = if count < 32 { -1 } else { 0 };
        v128_and(u32x4_shl(a, count), i32x4_splat(keep))
    }

    #[inline(always)]
    fn shr_logical_uniform(self, a: v128, count: u32) -> v128 {
        let keep: i32 = if count < 32 { -1 } else { 0 };
        v128_and(u32x4_shr(a, count), i32x4_splat(keep))
    }

    #[inline(always)]
    fn all_true(self, a: v128) -> bool {
        i32x4_bitmask(a) == 0x0F
    }
    #[inline(always)]
    fn any_true(self, a: v128) -> bool {
        i32x4_bitmask(a) != 0
    }
    #[inline(always)]
    fn bitmask(self, a: v128) -> u32 {
        i32x4_bitmask(a) as u32
    }
}

#[cfg(target_arch = "wasm32")]
impl I64x2Backend for archmage::Wasm128Token {
    type Repr = v128;

    #[inline(always)]
    fn splat(self, v: i64) -> v128 {
        i64x2_splat(v)
    }
    #[inline(always)]
    fn zero(self) -> v128 {
        i64x2_splat(0i64)
    }
    #[inline(always)]
    fn load(self, data: &[i64; 2]) -> v128 {
        crate::simd_storage::copy(data)
    }
    #[inline(always)]
    fn from_array(self, arr: [i64; 2]) -> v128 {
        crate::simd_storage::cast(arr)
    }
    #[inline(always)]
    fn store(self, repr: v128, out: &mut [i64; 2]) {
        crate::simd_storage::store(repr, out);
    }
    #[inline(always)]
    fn to_array(self, repr: v128) -> [i64; 2] {
        let mut out = [0i64; 2];
        crate::simd_storage::store(repr, &mut out);
        out
    }

    #[inline(always)]
    fn add(self, a: v128, b: v128) -> v128 {
        i64x2_add(a, b)
    }
    #[inline(always)]
    fn sub(self, a: v128, b: v128) -> v128 {
        i64x2_sub(a, b)
    }
    #[inline(always)]
    fn neg(self, a: v128) -> v128 {
        i64x2_neg(a)
    }
    #[inline(always)]
    fn min(self, a: v128, b: v128) -> v128 {
        // WASM SIMD lacks native i64 min; polyfill via compare+select
        let mask = i64x2_gt(a, b);
        v128_bitselect(b, a, mask)
    }
    #[inline(always)]
    fn max(self, a: v128, b: v128) -> v128 {
        // WASM SIMD lacks native i64 max; polyfill via compare+select
        let mask = i64x2_gt(a, b);
        v128_bitselect(a, b, mask)
    }
    #[inline(always)]
    fn abs(self, a: v128) -> v128 {
        // Polyfill: negate negative values
        let negated = i64x2_neg(a);
        let zero = i64x2_splat(0i64);
        let mask = i64x2_lt(a, zero);
        v128_bitselect(negated, a, mask)
    }

    #[inline(always)]
    fn simd_eq(self, a: v128, b: v128) -> v128 {
        i64x2_eq(a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: v128, b: v128) -> v128 {
        i64x2_ne(a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: v128, b: v128) -> v128 {
        i64x2_lt(a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: v128, b: v128) -> v128 {
        i64x2_le(a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: v128, b: v128) -> v128 {
        i64x2_gt(a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: v128, b: v128) -> v128 {
        i64x2_ge(a, b)
    }
    #[inline(always)]
    fn blend(self, mask: v128, if_true: v128, if_false: v128) -> v128 {
        v128_bitselect(if_true, if_false, mask)
    }

    #[inline(always)]
    fn reduce_add(self, a: v128) -> i64 {
        i64x2_extract_lane::<0>(a) + i64x2_extract_lane::<1>(a)
    }

    #[inline(always)]
    fn not(self, a: v128) -> v128 {
        v128_not(a)
    }
    #[inline(always)]
    fn bitand(self, a: v128, b: v128) -> v128 {
        v128_and(a, b)
    }
    #[inline(always)]
    fn bitor(self, a: v128, b: v128) -> v128 {
        v128_or(a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: v128, b: v128) -> v128 {
        v128_xor(a, b)
    }

    #[inline(always)]
    fn shl_const<const N: i32>(self, a: v128) -> v128 {
        i64x2_shl(a, N as u32)
    }
    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: v128) -> v128 {
        i64x2_shr(a, N as u32)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: v128) -> v128 {
        u64x2_shr(a, N as u32)
    }

    #[inline(always)]
    fn all_true(self, a: v128) -> bool {
        i64x2_bitmask(a) == 0x03
    }
    #[inline(always)]
    fn any_true(self, a: v128) -> bool {
        i64x2_bitmask(a) != 0
    }
    #[inline(always)]
    fn bitmask(self, a: v128) -> u32 {
        i64x2_bitmask(a) as u32
    }
}

#[cfg(target_arch = "wasm32")]
impl I8x16Backend for archmage::Wasm128Token {
    type Repr = v128;

    #[inline(always)]
    fn splat(self, v: i8) -> v128 {
        i8x16_splat(v)
    }

    #[inline(always)]
    fn zero(self) -> v128 {
        i8x16_splat(0)
    }

    #[inline(always)]
    fn load(self, data: &[i8; 16]) -> v128 {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [i8; 16]) -> v128 {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: v128, out: &mut [i8; 16]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: v128) -> [i8; 16] {
        let mut out = [0i8; 16];
        crate::simd_storage::store(repr, &mut out);
        out
    }

    #[inline(always)]
    fn add(self, a: v128, b: v128) -> v128 {
        i8x16_add(a, b)
    }
    #[inline(always)]
    fn sub(self, a: v128, b: v128) -> v128 {
        i8x16_sub(a, b)
    }
    #[inline(always)]
    fn neg(self, a: v128) -> v128 {
        i8x16_neg(a)
    }
    #[inline(always)]
    fn min(self, a: v128, b: v128) -> v128 {
        i8x16_min(a, b)
    }
    #[inline(always)]
    fn max(self, a: v128, b: v128) -> v128 {
        i8x16_max(a, b)
    }
    #[inline(always)]
    fn abs(self, a: v128) -> v128 {
        i8x16_abs(a)
    }

    #[inline(always)]
    fn simd_eq(self, a: v128, b: v128) -> v128 {
        i8x16_eq(a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: v128, b: v128) -> v128 {
        i8x16_ne(a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: v128, b: v128) -> v128 {
        i8x16_lt(a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: v128, b: v128) -> v128 {
        i8x16_le(a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: v128, b: v128) -> v128 {
        i8x16_gt(a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: v128, b: v128) -> v128 {
        i8x16_ge(a, b)
    }

    #[inline(always)]
    fn blend(self, mask: v128, if_true: v128, if_false: v128) -> v128 {
        v128_bitselect(if_true, if_false, mask)
    }

    #[inline(always)]
    fn reduce_add(self, a: v128) -> i8 {
        let arr = <Self as I8x16Backend>::to_array(self, a);
        arr.iter().copied().fold(0i8, i8::wrapping_add)
    }

    #[inline(always)]
    fn not(self, a: v128) -> v128 {
        v128_not(a)
    }
    #[inline(always)]
    fn bitand(self, a: v128, b: v128) -> v128 {
        v128_and(a, b)
    }
    #[inline(always)]
    fn bitor(self, a: v128, b: v128) -> v128 {
        v128_or(a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: v128, b: v128) -> v128 {
        v128_xor(a, b)
    }

    #[inline(always)]
    fn shl_const<const N: i32>(self, a: v128) -> v128 {
        i8x16_shl(a, N as u32)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: v128) -> v128 {
        u8x16_shr(a, N as u32)
    }
    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: v128) -> v128 {
        i8x16_shr(a, N as u32)
    }

    #[inline(always)]
    fn saturating_add(self, a: v128, b: v128) -> v128 {
        i8x16_add_sat(a, b)
    }

    #[inline(always)]
    fn saturating_sub(self, a: v128, b: v128) -> v128 {
        i8x16_sub_sat(a, b)
    }

    #[inline(always)]
    fn all_true(self, a: v128) -> bool {
        i8x16_bitmask(a) as u64 == 65535
    }
    #[inline(always)]
    fn any_true(self, a: v128) -> bool {
        i8x16_bitmask(a) != 0
    }
    #[inline(always)]
    fn bitmask(self, a: v128) -> u32 {
        i8x16_bitmask(a) as u32
    }
    #[inline(always)]
    fn widen_low_i8_to_i16(self, a: v128) -> v128 {
        i16x8_extend_low_i8x16(a)
    }

    #[inline(always)]
    fn widen_high_i8_to_i16(self, a: v128) -> v128 {
        i16x8_extend_high_i8x16(a)
    }
}

#[cfg(target_arch = "wasm32")]
impl U8x16Backend for archmage::Wasm128Token {
    type Repr = v128;

    #[inline(always)]
    fn splat(self, v: u8) -> v128 {
        u8x16_splat(v)
    }

    #[inline(always)]
    fn zero(self) -> v128 {
        u8x16_splat(0)
    }

    #[inline(always)]
    fn load(self, data: &[u8; 16]) -> v128 {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [u8; 16]) -> v128 {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: v128, out: &mut [u8; 16]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: v128) -> [u8; 16] {
        let mut out = [0u8; 16];
        crate::simd_storage::store(repr, &mut out);
        out
    }

    #[inline(always)]
    fn add(self, a: v128, b: v128) -> v128 {
        i8x16_add(a, b)
    }
    #[inline(always)]
    fn sub(self, a: v128, b: v128) -> v128 {
        i8x16_sub(a, b)
    }
    #[inline(always)]
    fn min(self, a: v128, b: v128) -> v128 {
        u8x16_min(a, b)
    }
    #[inline(always)]
    fn max(self, a: v128, b: v128) -> v128 {
        u8x16_max(a, b)
    }

    #[inline(always)]
    fn simd_eq(self, a: v128, b: v128) -> v128 {
        i8x16_eq(a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: v128, b: v128) -> v128 {
        i8x16_ne(a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: v128, b: v128) -> v128 {
        u8x16_lt(a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: v128, b: v128) -> v128 {
        u8x16_le(a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: v128, b: v128) -> v128 {
        u8x16_gt(a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: v128, b: v128) -> v128 {
        u8x16_ge(a, b)
    }

    #[inline(always)]
    fn blend(self, mask: v128, if_true: v128, if_false: v128) -> v128 {
        v128_bitselect(if_true, if_false, mask)
    }

    #[inline(always)]
    fn reduce_add(self, a: v128) -> u8 {
        let arr = <Self as U8x16Backend>::to_array(self, a);
        arr.iter().copied().fold(0u8, u8::wrapping_add)
    }

    #[inline(always)]
    fn not(self, a: v128) -> v128 {
        v128_not(a)
    }
    #[inline(always)]
    fn bitand(self, a: v128, b: v128) -> v128 {
        v128_and(a, b)
    }
    #[inline(always)]
    fn bitor(self, a: v128, b: v128) -> v128 {
        v128_or(a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: v128, b: v128) -> v128 {
        v128_xor(a, b)
    }

    #[inline(always)]
    fn shl_const<const N: i32>(self, a: v128) -> v128 {
        i8x16_shl(a, N as u32)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: v128) -> v128 {
        u8x16_shr(a, N as u32)
    }

    #[inline(always)]
    fn saturating_add(self, a: v128, b: v128) -> v128 {
        u8x16_add_sat(a, b)
    }

    #[inline(always)]
    fn saturating_sub(self, a: v128, b: v128) -> v128 {
        u8x16_sub_sat(a, b)
    }

    #[inline(always)]
    fn all_true(self, a: v128) -> bool {
        i8x16_bitmask(a) as u64 == 65535
    }
    #[inline(always)]
    fn any_true(self, a: v128) -> bool {
        i8x16_bitmask(a) != 0
    }
    #[inline(always)]
    fn bitmask(self, a: v128) -> u32 {
        i8x16_bitmask(a) as u32
    }
    #[inline(always)]
    fn widen_low_u8_to_u16(self, a: v128) -> v128 {
        u16x8_extend_low_u8x16(a)
    }

    #[inline(always)]
    fn widen_high_u8_to_u16(self, a: v128) -> v128 {
        u16x8_extend_high_u8x16(a)
    }

    #[inline(always)]
    fn abs_diff(self, a: v128, b: v128) -> v128 {
        u8x16_sub(u8x16_max(a, b), u8x16_min(a, b))
    }

    #[inline(always)]
    fn reduce_add_u32(self, a: v128) -> u32 {
        {
            let s = u32x4_extadd_pairwise_u16x8(u16x8_extadd_pairwise_u8x16(a));
            u32x4_extract_lane::<0>(s)
                + u32x4_extract_lane::<1>(s)
                + u32x4_extract_lane::<2>(s)
                + u32x4_extract_lane::<3>(s)
        }
    }

    #[inline(always)]
    fn sum_abs_diff(self, a: v128, b: v128) -> u32 {
        {
            let s = u32x4_extadd_pairwise_u16x8(u16x8_extadd_pairwise_u8x16(u8x16_sub(
                u8x16_max(a, b),
                u8x16_min(a, b),
            )));
            u32x4_extract_lane::<0>(s)
                + u32x4_extract_lane::<1>(s)
                + u32x4_extract_lane::<2>(s)
                + u32x4_extract_lane::<3>(s)
        }
    }

    #[inline(always)]
    fn pairwise_widen_add(self, a: v128) -> v128 {
        u16x8_extadd_pairwise_u8x16(a)
    }
}

#[cfg(target_arch = "wasm32")]
impl I16x8Backend for archmage::Wasm128Token {
    type Repr = v128;

    #[inline(always)]
    fn splat(self, v: i16) -> v128 {
        i16x8_splat(v)
    }

    #[inline(always)]
    fn zero(self) -> v128 {
        i16x8_splat(0)
    }

    #[inline(always)]
    fn load(self, data: &[i16; 8]) -> v128 {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [i16; 8]) -> v128 {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: v128, out: &mut [i16; 8]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: v128) -> [i16; 8] {
        let mut out = [0i16; 8];
        crate::simd_storage::store(repr, &mut out);
        out
    }

    #[inline(always)]
    fn add(self, a: v128, b: v128) -> v128 {
        i16x8_add(a, b)
    }
    #[inline(always)]
    fn sub(self, a: v128, b: v128) -> v128 {
        i16x8_sub(a, b)
    }
    #[inline(always)]
    fn mul(self, a: v128, b: v128) -> v128 {
        i16x8_mul(a, b)
    }
    #[inline(always)]
    fn neg(self, a: v128) -> v128 {
        i16x8_neg(a)
    }
    #[inline(always)]
    fn min(self, a: v128, b: v128) -> v128 {
        i16x8_min(a, b)
    }
    #[inline(always)]
    fn max(self, a: v128, b: v128) -> v128 {
        i16x8_max(a, b)
    }
    #[inline(always)]
    fn abs(self, a: v128) -> v128 {
        i16x8_abs(a)
    }

    #[inline(always)]
    fn simd_eq(self, a: v128, b: v128) -> v128 {
        i16x8_eq(a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: v128, b: v128) -> v128 {
        i16x8_ne(a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: v128, b: v128) -> v128 {
        i16x8_lt(a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: v128, b: v128) -> v128 {
        i16x8_le(a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: v128, b: v128) -> v128 {
        i16x8_gt(a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: v128, b: v128) -> v128 {
        i16x8_ge(a, b)
    }

    #[inline(always)]
    fn blend(self, mask: v128, if_true: v128, if_false: v128) -> v128 {
        v128_bitselect(if_true, if_false, mask)
    }

    #[inline(always)]
    fn reduce_add(self, a: v128) -> i16 {
        let arr = <Self as I16x8Backend>::to_array(self, a);
        arr.iter().copied().fold(0i16, i16::wrapping_add)
    }

    #[inline(always)]
    fn not(self, a: v128) -> v128 {
        v128_not(a)
    }
    #[inline(always)]
    fn bitand(self, a: v128, b: v128) -> v128 {
        v128_and(a, b)
    }
    #[inline(always)]
    fn bitor(self, a: v128, b: v128) -> v128 {
        v128_or(a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: v128, b: v128) -> v128 {
        v128_xor(a, b)
    }

    #[inline(always)]
    fn shl_const<const N: i32>(self, a: v128) -> v128 {
        i16x8_shl(a, N as u32)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: v128) -> v128 {
        u16x8_shr(a, N as u32)
    }
    #[inline(always)]
    fn shr_arithmetic_const<const N: i32>(self, a: v128) -> v128 {
        i16x8_shr(a, N as u32)
    }

    #[inline(always)]
    fn shl_uniform(self, a: v128, count: u32) -> v128 {
        let keep: i16 = if count < 16 { -1 } else { 0 };
        v128_and(i16x8_shl(a, count), i16x8_splat(keep))
    }

    #[inline(always)]
    fn shr_logical_uniform(self, a: v128, count: u32) -> v128 {
        let keep: i16 = if count < 16 { -1 } else { 0 };
        v128_and(u16x8_shr(a, count), i16x8_splat(keep))
    }
    #[inline(always)]
    fn shr_arithmetic_uniform(self, a: v128, count: u32) -> v128 {
        i16x8_shr(a, if count > 15 { 15 } else { count })
    }

    #[inline(always)]
    fn saturating_add(self, a: v128, b: v128) -> v128 {
        i16x8_add_sat(a, b)
    }

    #[inline(always)]
    fn saturating_sub(self, a: v128, b: v128) -> v128 {
        i16x8_sub_sat(a, b)
    }

    #[inline(always)]
    fn all_true(self, a: v128) -> bool {
        i16x8_bitmask(a) as u64 == 255
    }
    #[inline(always)]
    fn any_true(self, a: v128) -> bool {
        i16x8_bitmask(a) != 0
    }
    #[inline(always)]
    fn bitmask(self, a: v128) -> u32 {
        i16x8_bitmask(a) as u32
    }
    #[inline(always)]
    fn widen_low_i16_to_i32(self, a: v128) -> v128 {
        i32x4_extend_low_i16x8(a)
    }

    #[inline(always)]
    fn widen_high_i16_to_i32(self, a: v128) -> v128 {
        i32x4_extend_high_i16x8(a)
    }

    #[inline(always)]
    fn narrow_saturating_i16_to_i8(self, a: v128, b: v128) -> v128 {
        i8x16_narrow_i16x8(a, b)
    }

    #[inline(always)]
    fn narrow_saturating_i16_to_u8(self, a: v128, b: v128) -> v128 {
        u8x16_narrow_i16x8(a, b)
    }

    #[inline(always)]
    fn madd_adjacent(self, a: v128, b: v128) -> v128 {
        i32x4_dot_i16x8(a, b)
    }

    #[inline(always)]
    fn abs_diff(self, a: v128, b: v128) -> v128 {
        i16x8_sub(i16x8_max(a, b), i16x8_min(a, b))
    }
}

#[cfg(target_arch = "wasm32")]
impl U16x8Backend for archmage::Wasm128Token {
    type Repr = v128;

    #[inline(always)]
    fn splat(self, v: u16) -> v128 {
        u16x8_splat(v)
    }

    #[inline(always)]
    fn zero(self) -> v128 {
        u16x8_splat(0)
    }

    #[inline(always)]
    fn load(self, data: &[u16; 8]) -> v128 {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [u16; 8]) -> v128 {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: v128, out: &mut [u16; 8]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: v128) -> [u16; 8] {
        let mut out = [0u16; 8];
        crate::simd_storage::store(repr, &mut out);
        out
    }

    #[inline(always)]
    fn add(self, a: v128, b: v128) -> v128 {
        i16x8_add(a, b)
    }
    #[inline(always)]
    fn sub(self, a: v128, b: v128) -> v128 {
        i16x8_sub(a, b)
    }
    #[inline(always)]
    fn mul(self, a: v128, b: v128) -> v128 {
        i16x8_mul(a, b)
    }
    #[inline(always)]
    fn min(self, a: v128, b: v128) -> v128 {
        u16x8_min(a, b)
    }
    #[inline(always)]
    fn max(self, a: v128, b: v128) -> v128 {
        u16x8_max(a, b)
    }

    #[inline(always)]
    fn simd_eq(self, a: v128, b: v128) -> v128 {
        i16x8_eq(a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: v128, b: v128) -> v128 {
        i16x8_ne(a, b)
    }
    #[inline(always)]
    fn simd_lt(self, a: v128, b: v128) -> v128 {
        u16x8_lt(a, b)
    }
    #[inline(always)]
    fn simd_le(self, a: v128, b: v128) -> v128 {
        u16x8_le(a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: v128, b: v128) -> v128 {
        u16x8_gt(a, b)
    }
    #[inline(always)]
    fn simd_ge(self, a: v128, b: v128) -> v128 {
        u16x8_ge(a, b)
    }

    #[inline(always)]
    fn blend(self, mask: v128, if_true: v128, if_false: v128) -> v128 {
        v128_bitselect(if_true, if_false, mask)
    }

    #[inline(always)]
    fn reduce_add(self, a: v128) -> u16 {
        let arr = <Self as U16x8Backend>::to_array(self, a);
        arr.iter().copied().fold(0u16, u16::wrapping_add)
    }

    #[inline(always)]
    fn not(self, a: v128) -> v128 {
        v128_not(a)
    }
    #[inline(always)]
    fn bitand(self, a: v128, b: v128) -> v128 {
        v128_and(a, b)
    }
    #[inline(always)]
    fn bitor(self, a: v128, b: v128) -> v128 {
        v128_or(a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: v128, b: v128) -> v128 {
        v128_xor(a, b)
    }

    #[inline(always)]
    fn shl_const<const N: i32>(self, a: v128) -> v128 {
        i16x8_shl(a, N as u32)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: v128) -> v128 {
        u16x8_shr(a, N as u32)
    }

    #[inline(always)]
    fn shl_uniform(self, a: v128, count: u32) -> v128 {
        let keep: i16 = if count < 16 { -1 } else { 0 };
        v128_and(i16x8_shl(a, count), i16x8_splat(keep))
    }

    #[inline(always)]
    fn shr_logical_uniform(self, a: v128, count: u32) -> v128 {
        let keep: i16 = if count < 16 { -1 } else { 0 };
        v128_and(u16x8_shr(a, count), i16x8_splat(keep))
    }

    #[inline(always)]
    fn saturating_add(self, a: v128, b: v128) -> v128 {
        u16x8_add_sat(a, b)
    }

    #[inline(always)]
    fn saturating_sub(self, a: v128, b: v128) -> v128 {
        u16x8_sub_sat(a, b)
    }

    #[inline(always)]
    fn all_true(self, a: v128) -> bool {
        i16x8_bitmask(a) as u64 == 255
    }
    #[inline(always)]
    fn any_true(self, a: v128) -> bool {
        i16x8_bitmask(a) != 0
    }
    #[inline(always)]
    fn bitmask(self, a: v128) -> u32 {
        i16x8_bitmask(a) as u32
    }
    #[inline(always)]
    fn widen_low_u16_to_u32(self, a: v128) -> v128 {
        u32x4_extend_low_u16x8(a)
    }

    #[inline(always)]
    fn widen_high_u16_to_u32(self, a: v128) -> v128 {
        u32x4_extend_high_u16x8(a)
    }

    #[inline(always)]
    fn pairwise_widen_add(self, a: v128) -> v128 {
        u32x4_extadd_pairwise_u16x8(a)
    }
}

#[cfg(target_arch = "wasm32")]
impl U64x2Backend for archmage::Wasm128Token {
    type Repr = v128;

    #[inline(always)]
    fn splat(self, v: u64) -> v128 {
        u64x2_splat(v)
    }

    #[inline(always)]
    fn zero(self) -> v128 {
        u64x2_splat(0)
    }

    #[inline(always)]
    fn load(self, data: &[u64; 2]) -> v128 {
        crate::simd_storage::copy(data)
    }

    #[inline(always)]
    fn from_array(self, arr: [u64; 2]) -> v128 {
        crate::simd_storage::cast(arr)
    }

    #[inline(always)]
    fn store(self, repr: v128, out: &mut [u64; 2]) {
        crate::simd_storage::store(repr, out);
    }

    #[inline(always)]
    fn to_array(self, repr: v128) -> [u64; 2] {
        let mut out = [0u64; 2];
        crate::simd_storage::store(repr, &mut out);
        out
    }

    #[inline(always)]
    fn add(self, a: v128, b: v128) -> v128 {
        i64x2_add(a, b)
    }
    #[inline(always)]
    fn sub(self, a: v128, b: v128) -> v128 {
        i64x2_sub(a, b)
    }
    #[inline(always)]
    fn min(self, a: v128, b: v128) -> v128 {
        // u64 min polyfill: compare then select
        let mask = <Self as U64x2Backend>::simd_lt(self, a, b);
        v128_bitselect(a, b, mask)
    }
    #[inline(always)]
    fn max(self, a: v128, b: v128) -> v128 {
        let mask = <Self as U64x2Backend>::simd_gt(self, a, b);
        v128_bitselect(a, b, mask)
    }

    #[inline(always)]
    fn simd_eq(self, a: v128, b: v128) -> v128 {
        i64x2_eq(a, b)
    }
    #[inline(always)]
    fn simd_ne(self, a: v128, b: v128) -> v128 {
        i64x2_ne(a, b)
    }
    #[inline(always)]
    fn simd_gt(self, a: v128, b: v128) -> v128 {
        // Unsigned comparison via bias trick
        let bias = i64x2_splat(i64::MIN);
        let sa = v128_xor(a, bias);
        let sb = v128_xor(b, bias);
        i64x2_gt(sa, sb)
    }
    #[inline(always)]
    fn simd_lt(self, a: v128, b: v128) -> v128 {
        <Self as U64x2Backend>::simd_gt(self, b, a)
    }
    #[inline(always)]
    fn simd_le(self, a: v128, b: v128) -> v128 {
        v128_not(<Self as U64x2Backend>::simd_gt(self, a, b))
    }
    #[inline(always)]
    fn simd_ge(self, a: v128, b: v128) -> v128 {
        v128_not(<Self as U64x2Backend>::simd_gt(self, b, a))
    }

    #[inline(always)]
    fn blend(self, mask: v128, if_true: v128, if_false: v128) -> v128 {
        v128_bitselect(if_true, if_false, mask)
    }

    #[inline(always)]
    fn reduce_add(self, a: v128) -> u64 {
        let arr = <Self as U64x2Backend>::to_array(self, a);
        arr.iter().copied().fold(0u64, u64::wrapping_add)
    }

    #[inline(always)]
    fn not(self, a: v128) -> v128 {
        v128_not(a)
    }
    #[inline(always)]
    fn bitand(self, a: v128, b: v128) -> v128 {
        v128_and(a, b)
    }
    #[inline(always)]
    fn bitor(self, a: v128, b: v128) -> v128 {
        v128_or(a, b)
    }
    #[inline(always)]
    fn bitxor(self, a: v128, b: v128) -> v128 {
        v128_xor(a, b)
    }

    #[inline(always)]
    fn shl_const<const N: i32>(self, a: v128) -> v128 {
        i64x2_shl(a, N as u32)
    }
    #[inline(always)]
    fn shr_logical_const<const N: i32>(self, a: v128) -> v128 {
        u64x2_shr(a, N as u32)
    }

    #[inline(always)]
    fn all_true(self, a: v128) -> bool {
        i64x2_bitmask(a) as u64 == 3
    }
    #[inline(always)]
    fn any_true(self, a: v128) -> bool {
        i64x2_bitmask(a) != 0
    }
    #[inline(always)]
    fn bitmask(self, a: v128) -> u32 {
        i64x2_bitmask(a) as u32
    }
}

#[cfg(target_arch = "wasm32")]
impl F32x4Convert for archmage::Wasm128Token {
    #[inline(always)]
    fn bitcast_f32_to_i32(self, a: v128) -> v128 {
        a
    }

    #[inline(always)]
    fn bitcast_i32_to_f32(self, a: v128) -> v128 {
        a
    }

    #[inline(always)]
    fn convert_f32_to_i32(self, a: v128) -> v128 {
        i32x4_trunc_sat_f32x4(a)
    }

    #[inline(always)]
    fn convert_f32_to_i32_fast(self, a: v128) -> v128 {
        crate::wasm_fma::trunc_f32x4_fast(self, a)
    }

    #[inline(always)]
    fn convert_f32_to_i32_round(self, a: v128) -> v128 {
        i32x4_trunc_sat_f32x4(f32x4_nearest(a))
    }

    #[inline(always)]
    fn convert_i32_to_f32(self, a: v128) -> v128 {
        f32x4_convert_i32x4(a)
    }
}

#[cfg(target_arch = "wasm32")]
impl U32x4Bitcast for archmage::Wasm128Token {
    #[inline(always)]
    fn bitcast_u32_to_i32(self, a: v128) -> v128 {
        a
    }

    #[inline(always)]
    fn bitcast_i32_to_u32(self, a: v128) -> v128 {
        a
    }
}

#[cfg(target_arch = "wasm32")]
impl I64x2Bitcast for archmage::Wasm128Token {
    #[inline(always)]
    fn bitcast_i64_to_f64(self, a: v128) -> v128 {
        a
    }

    #[inline(always)]
    fn bitcast_f64_to_i64(self, a: v128) -> v128 {
        a
    }
}

#[cfg(target_arch = "wasm32")]
impl I8x16Bitcast for archmage::Wasm128Token {
    #[inline(always)]
    fn bitcast_i8_to_u8(self, a: v128) -> v128 {
        a
    }
    #[inline(always)]
    fn bitcast_u8_to_i8(self, a: v128) -> v128 {
        a
    }
}

#[cfg(target_arch = "wasm32")]
impl I16x8Bitcast for archmage::Wasm128Token {
    #[inline(always)]
    fn bitcast_i16_to_u16(self, a: v128) -> v128 {
        a
    }
    #[inline(always)]
    fn bitcast_u16_to_i16(self, a: v128) -> v128 {
        a
    }
}

#[cfg(target_arch = "wasm32")]
impl U64x2Bitcast for archmage::Wasm128Token {
    #[inline(always)]
    fn bitcast_u64_to_i64(self, a: v128) -> v128 {
        a
    }
    #[inline(always)]
    fn bitcast_i64_to_u64(self, a: v128) -> v128 {
        a
    }
}
