#![forbid(unsafe_code)]

use archmage::SimdToken;
use magetypes::SimdTypes;
use magetypes::WidthDispatch;
use magetypes::simd::backends::{
    F32x4Backend, F32x4Convert, F32x8Backend, F32x8Convert, F64x2Backend, F64x4Backend,
    I8x16Backend, I8x16Bitcast, I8x32Backend, I8x32Bitcast, I16x8Backend, I16x8Bitcast,
    I16x16Backend, I16x16Bitcast, I32x4Backend, I32x8Backend, I64x2Backend, I64x2Bitcast,
    I64x4Backend, I64x4Bitcast, U8x16Backend, U8x32Backend, U16x8Backend, U16x16Backend,
    U32x4Backend, U32x4Bitcast, U32x8Backend, U32x8Bitcast, U64x2Backend, U64x2Bitcast,
    U64x4Backend, U64x4Bitcast,
};
#[cfg(feature = "w512")]
use magetypes::simd::backends::{
    F32x16Backend, F32x16Convert, F64x8Backend, I8x64Backend, I16x32Backend, I32x16Backend,
    I64x8Backend, U8x64Backend, U16x32Backend, U32x16Backend, U64x8Backend,
};
#[cfg(all(target_arch = "x86_64", feature = "avx512", feature = "w512"))]
use magetypes::simd::backends::{
    i8x64PopcntBackend, i16x32PopcntBackend, i32x16PopcntBackend, i64x8PopcntBackend,
    u8x64PopcntBackend, u16x32PopcntBackend, u32x16PopcntBackend, u64x8PopcntBackend,
};
use magetypes::simd::generic::{
    F16Convert, F32x4ToU32, F32x8FromHalves, F32x8ToU32, I32x4DotI8, I32x8DotI8, MagetypesToken,
    f32x4, f32x8, f64x2, f64x4, i8x16, i8x32, i32x4, i32x8, u16x8,
};
#[cfg(feature = "w512")]
use magetypes::simd::generic::{F32x16FromHalves, F32x16ToU32, I32x16DotI8, f32x16, i8x64, i32x16};

fn verify_token_narrow_and_256<T>(token: T)
where
    T: SimdToken
        + Copy
        + 'static
        + WidthDispatch
        + MagetypesToken
        + F16Convert
        + F32x8FromHalves
        + F32x4ToU32
        + F32x8ToU32
        + I32x4DotI8
        + I32x8DotI8
        + F32x4Backend
        + F32x8Backend
        + F64x2Backend
        + F64x4Backend
        + I8x16Backend
        + I8x32Backend
        + U8x16Backend
        + U8x32Backend
        + I16x8Backend
        + I16x16Backend
        + U16x8Backend
        + U16x16Backend
        + I32x4Backend
        + I32x8Backend
        + U32x4Backend
        + U32x8Backend
        + I64x2Backend
        + I64x4Backend
        + U64x2Backend
        + U64x4Backend
        + F32x4Convert
        + F32x8Convert
        + U32x4Bitcast
        + U32x8Bitcast
        + I64x2Bitcast
        + I64x4Bitcast
        + I8x16Bitcast
        + I8x32Bitcast
        + I16x8Bitcast
        + I16x16Bitcast
        + U64x2Bitcast
        + U64x4Bitcast,
{
    let a4 = f32x4::from_array_t(token, [1.5, -2.0, 3.25, 4.5]);
    let b4 = f32x4::from_array_t(token, [2.0, 4.0, -1.0, 0.5]);
    let sum4 = (a4 + b4).to_array();
    assert_eq!(sum4, [3.5, 2.0, 2.25, 5.0]);

    let a8 = f32x8::from_array_t(token, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let c8 = f32x8::from_array_t(token, [0.5; 8]);
    let mla8 = a8
        .mul_add_lane4_relaxed::<2>(f32x4::from_array_t(token, [10.0, 20.0, 3.0, 40.0]), c8)
        .to_array();
    for i in 0..8 {
        assert!((mla8[i] - ((i + 1) as f32 * 3.0 + 0.5)).abs() < 1e-5);
    }

    let lo = f32x4::from_array_t(token, [1.0, 2.0, 3.0, 4.0]);
    let hi = f32x4::from_array_t(token, [5.0, 6.0, 7.0, 8.0]);
    let combined = f32x8::from_halves_t(token, lo, hi);
    assert_eq!(
        combined.to_array(),
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    );
    let (lo_out, hi_out) = combined.split();
    assert_eq!(lo_out.to_array(), [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(hi_out.to_array(), [5.0, 6.0, 7.0, 8.0]);

    // f64 + u32/i32 + bitcast + MagetypesToken subtiers
    let d2 = f64x2::from_array_t(token, [1.25, -3.5]);
    assert_eq!((d2 + f64x2::splat_t(token, 2.0)).to_array(), [3.25, -1.5]);
    let d4 = f64x4::from_array_t(token, [1.0, 2.0, 3.0, 4.0]);
    assert_eq!(
        (d4 + f64x4::splat_t(token, 1.0)).to_array(),
        [2.0, 3.0, 4.0, 5.0]
    );

    // WidthDispatch construction
    let _wd_f4: <T as WidthDispatch>::F32x4 = token.f32x4_splat(1.0);
    let _wd_f8: <T as WidthDispatch>::F32x8 = token.f32x8_load(&[1.0; 8]);
    let _wd_f16: <T as WidthDispatch>::F32x16 = token.f32x16_zero();
    let _wd_u16: u16x8<T> = u16x8::zero_t(token);
    let _ = (_wd_f4, _wd_f8, _wd_f16, _wd_u16);

    // Fast f32 -> u32 including [2^31, 2^32) range
    let u4 = f32x4::from_array_t(token, [0.5, 42.9, 2_147_483_648.0, 3_000_000_000.0])
        .to_u32_fast()
        .to_array();
    assert_eq!(u4, [0, 42, 2_147_483_648, 3_000_000_000]);

    let u8_in = [
        0.0,
        1.9,
        100.2,
        65535.0,
        2_147_483_648.0,
        2_500_000_000.0,
        3_000_000_000.0,
        4_000_000_000.0,
    ];
    let u8_out = f32x8::from_array_t(token, u8_in).to_u32_fast().to_array();
    assert_eq!(u8_out, u8_in.map(|x| x as u32));

    // i8 * i7 dot-product accumulate
    let acc4 = [10, -20, 30, -40];
    let a_i8: [i8; 16] = core::array::from_fn(|i| (i as i8) - 8);
    let b_i7: [i8; 16] = core::array::from_fn(|i| (i as i8).wrapping_mul(5) & 0x7f);
    let got4 = i32x4::from_array_t(token, acc4)
        .dot_i8_i7_add(
            i8x16::from_array_t(token, a_i8),
            i8x16::from_array_t(token, b_i7),
        )
        .to_array();
    for lane in 0..4 {
        let sum: i32 = (0..4)
            .map(|k| (a_i8[4 * lane + k] as i32) * (b_i7[4 * lane + k] as i32))
            .sum();
        assert_eq!(got4[lane], acc4[lane] + sum);
    }

    let acc8: [i32; 8] = core::array::from_fn(|i| i as i32 * 7 - 11);
    let a32: [i8; 32] = core::array::from_fn(|i| (i as i8) - 16);
    let b32: [i8; 32] = core::array::from_fn(|i| (i as i8).wrapping_mul(3) & 0x7f);
    let got8 = i32x8::from_array_t(token, acc8)
        .dot_i8_i7_add(
            i8x32::from_array_t(token, a32),
            i8x32::from_array_t(token, b32),
        )
        .to_array();
    for lane in 0..8 {
        let sum: i32 = (0..4)
            .map(|k| (a32[4 * lane + k] as i32) * (b32[4 * lane + k] as i32))
            .sum();
        assert_eq!(got8[lane], acc8[lane] + sum);
    }

    // F16Convert slice roundtrip via MagetypesToken / F16Convert
    let src_f32 = [-2.5f32, -0.0, 0.0, 1.0, 3.5, 42.0, -100.5, 0.25];
    let mut half_buf = [0u16; 8];
    let mut roundtrip = [0.0f32; 8];
    <T as F16Convert>::f32_to_f16_slice(token, &src_f32, &mut half_buf);
    <T as F16Convert>::f16_to_f32_slice(token, &half_buf, &mut roundtrip);
    for (&orig, &rt) in src_f32.iter().zip(roundtrip.iter()) {
        assert_eq!(orig.to_bits(), rt.to_bits());
    }
}

#[cfg(feature = "w512")]
fn verify_token_w512<T>(token: T)
where
    T: SimdToken
        + Copy
        + 'static
        + F32x16FromHalves
        + F32x16ToU32
        + I32x16DotI8
        + F32x16Backend
        + F64x8Backend
        + I8x64Backend
        + U8x64Backend
        + I16x32Backend
        + U16x32Backend
        + I32x16Backend
        + U32x16Backend
        + I64x8Backend
        + U64x8Backend
        + F32x16Convert
        + F32x8Backend
        + I32x8Backend
        + I8x32Backend
        + U32x16Backend,
{
    let lo = f32x8::from_array_t(token, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let hi = f32x8::from_array_t(token, [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
    let wide = f32x16::from_halves_t(token, lo, hi);
    assert_eq!(wide.to_array(), core::array::from_fn(|i| (i + 1) as f32));
    let (lo_out, hi_out) = wide.split();
    assert_eq!(lo_out.to_array(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(
        hi_out.to_array(),
        [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
    );

    let fast_in: [f32; 16] = core::array::from_fn(|i| {
        if i % 2 == 0 {
            (i as f32) * 100.0 + 0.75
        } else {
            2_147_483_648.0 + (i as f32) * 65536.0
        }
    });
    let fast_out = f32x16::from_array_t(token, fast_in)
        .to_u32_fast()
        .to_array();
    assert_eq!(fast_out, fast_in.map(|x| x as u32));

    let acc16: [i32; 16] = core::array::from_fn(|i| (i as i32) * 13 - 50);
    let a64: [i8; 64] = core::array::from_fn(|i| ((i as i8) % 21) - 10);
    let b64: [i8; 64] = core::array::from_fn(|i| (i as i8).wrapping_mul(7) & 0x7f);
    let dot16 = i32x16::from_array_t(token, acc16)
        .dot_i8_i7_add(
            i8x64::from_array_t(token, a64),
            i8x64::from_array_t(token, b64),
        )
        .to_array();
    for lane in 0..16 {
        let sum: i32 = (0..4)
            .map(|k| (a64[4 * lane + k] as i32) * (b64[4 * lane + k] as i32))
            .sum();
        assert_eq!(dot16[lane], acc16[lane] + sum);
    }
}

#[cfg(all(target_arch = "x86_64", feature = "avx512", feature = "w512"))]
fn verify_v4x_popcnt_bounds<
    T: i8x64PopcntBackend
        + u8x64PopcntBackend
        + i16x32PopcntBackend
        + u16x32PopcntBackend
        + i32x16PopcntBackend
        + u32x16PopcntBackend
        + i64x8PopcntBackend
        + u64x8PopcntBackend,
>(
    _token: T,
) {
}

fn check_simd_types_bound<T: SimdTypes>() {
    assert!(T::F32_LANES >= 1);
    assert!(T::F64_LANES >= 1);
    assert!(T::I32_LANES >= 1);
}

#[test]
fn all_21_tokens_satisfy_all_unified_traits_and_run_when_available() {
    macro_rules! check_token {
        ($tok:ty) => {{
            // Compile-time monomorphization proof (holds even on cross-arch stub tokens).
            let _: fn($tok) = verify_token_narrow_and_256::<$tok>;
            #[cfg(feature = "w512")]
            {
                let _: fn($tok) = verify_token_w512::<$tok>;
            }
            // Runtime behavioral verification whenever the current host supports the token.
            if let Some(t) = <$tok>::summon() {
                verify_token_narrow_and_256(t);
                #[cfg(feature = "w512")]
                verify_token_w512(t);
            }
        }};
    }

    // All 21 tokens on their native targets (plus ScalarToken everywhere):
    // 1 scalar + 11 x86_64 + 6 aarch64 + 3 wasm32.
    check_token!(archmage::ScalarToken);
    #[cfg(target_arch = "x86_64")]
    {
        check_token!(archmage::X64V1Token);
        check_token!(archmage::X64V2Token);
        check_token!(archmage::X64CryptoToken);
        check_token!(archmage::X64V3Token);
        check_token!(archmage::X64V3CryptoToken);
        check_token!(archmage::X64V3GfniCryptoToken);
        check_token!(archmage::X64V3VnniToken);
        check_token!(archmage::X64V3Vnni2Token);
        check_token!(archmage::X64V4Token);
        check_token!(archmage::X64V4xToken);
        check_token!(archmage::Avx512Fp16Token);
    }
    #[cfg(target_arch = "aarch64")]
    {
        check_token!(archmage::NeonToken);
        check_token!(archmage::NeonAesToken);
        check_token!(archmage::NeonSha3Token);
        check_token!(archmage::NeonCrcToken);
        check_token!(archmage::Arm64V2Token);
        check_token!(archmage::Arm64V3Token);
    }
    #[cfg(target_arch = "wasm32")]
    {
        check_token!(archmage::Wasm128Token);
        check_token!(archmage::Wasm128RelaxedToken);
        check_token!(archmage::Wasm128RelaxedFmaToken);
    }

    #[cfg(all(target_arch = "x86_64", feature = "avx512", feature = "w512"))]
    {
        let _: fn(archmage::X64V4xToken) = verify_v4x_popcnt_bounds::<archmage::X64V4xToken>;
    }

    // SimdTypes checks for ScalarToken + all native arch tokens.
    check_simd_types_bound::<archmage::ScalarToken>();
    #[cfg(target_arch = "x86_64")]
    {
        check_simd_types_bound::<archmage::X64V1Token>();
        check_simd_types_bound::<archmage::X64V2Token>();
        check_simd_types_bound::<archmage::X64CryptoToken>();
        check_simd_types_bound::<archmage::X64V3Token>();
        check_simd_types_bound::<archmage::X64V3CryptoToken>();
        check_simd_types_bound::<archmage::X64V3GfniCryptoToken>();
        check_simd_types_bound::<archmage::X64V3VnniToken>();
        check_simd_types_bound::<archmage::X64V3Vnni2Token>();
        check_simd_types_bound::<archmage::X64V4Token>();
        check_simd_types_bound::<archmage::X64V4xToken>();
        check_simd_types_bound::<archmage::Avx512Fp16Token>();
    }
    #[cfg(target_arch = "aarch64")]
    {
        check_simd_types_bound::<archmage::NeonToken>();
        check_simd_types_bound::<archmage::NeonAesToken>();
        check_simd_types_bound::<archmage::NeonSha3Token>();
        check_simd_types_bound::<archmage::NeonCrcToken>();
        check_simd_types_bound::<archmage::Arm64V2Token>();
        check_simd_types_bound::<archmage::Arm64V3Token>();
    }
    #[cfg(target_arch = "wasm32")]
    {
        check_simd_types_bound::<archmage::Wasm128Token>();
        check_simd_types_bound::<archmage::Wasm128RelaxedToken>();
        check_simd_types_bound::<archmage::Wasm128RelaxedFmaToken>();
    }
}
