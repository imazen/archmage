//! Comprehensive cross-backend tests for the neural & vision SIMD operations
//! (`#155`–`#163`, `#115`):
//! - `min_fast` / `max_fast` (`f32x4`, `f32x8`, `f32x16`, `f64x2`, `f64x4`, `f64x8`)
//! - `to_i32_fast` / `to_u32_fast` (`f32x4`, `f32x8`, `f32x16`)
//! - `broadcast_lane`, `shift_left_lanes`, `shift_right_lanes`, `mul_add_lane_relaxed`, `mul_add_lane4_relaxed`
//! - `mul_hrs_q15` (`i16x8`, `i16x16`, `i16x32`) and `dot_i8_i7_add` (`i32x4`, `i32x8`, `i32x16`)
//! - 3-channel RGB helpers (`deinterleave_3ch`, `interleave_3ch`, `load_{4,8}_rgb_u8_t`, `store_{4,8}_rgb_u8`)
//! - `#[magetypes(..., define(...))]` with subtier tokens + `MagetypesToken::magetypes()`

#![forbid(unsafe_code)]
#![deny(deprecated)]

use archmage::{ScalarToken, SimdToken};
use magetypes::simd::MagetypesToken;
use magetypes::simd::backends::{
    F32x4Convert, F32x8Convert, F64x2Backend, F64x4Backend, I16x8Backend, I16x16Backend,
};
#[cfg(feature = "w512")]
use magetypes::simd::backends::{F32x16Convert, F64x8Backend, I16x32Backend};
use magetypes::simd::generic::{
    F32x4ToU32, F32x8FromHalves, F32x8ToU32, I32x4DotI8, I32x8DotI8, f32x4, f32x8, f64x2, f64x4,
    i8x16, i8x32, i16x8, i16x16, i32x4, i32x8,
};
#[cfg(feature = "w512")]
use magetypes::simd::generic::{F32x16ToU32, I32x16DotI8, f32x16, f64x8, i8x64, i16x32, i32x16};

fn check_narrow_backend<T>(token: T)
where
    T: SimdToken
        + F32x4Convert
        + F32x8Convert
        + F32x4ToU32
        + F32x8ToU32
        + F32x8FromHalves
        + F64x2Backend
        + F64x4Backend
        + I16x8Backend
        + I16x16Backend
        + I32x4DotI8
        + I32x8DotI8,
{
    // --- min_fast / max_fast ---
    let a4 = f32x4::from_array_t(token, [-3.5, 2.25, 0.0, 100.0]);
    let b4 = f32x4::from_array_t(token, [-1.0, -4.5, 5.0, 100.0]);
    assert_eq!(a4.min_fast(b4).to_array(), [-3.5, -4.5, 0.0, 100.0]);
    assert_eq!(a4.max_fast(b4).to_array(), [-1.0, 2.25, 5.0, 100.0]);

    let a8 = f32x8::from_array_t(token, [-3.5, 2.25, 0.5, 100.0, 7.0, -8.0, 1.25, -0.25]);
    let b8 = f32x8::from_array_t(token, [-1.0, -4.5, 5.0, 99.0, 7.5, -9.0, 1.25, 0.25]);
    assert_eq!(
        a8.min_fast(b8).to_array(),
        [-3.5, -4.5, 0.5, 99.0, 7.0, -9.0, 1.25, -0.25]
    );
    assert_eq!(
        a8.max_fast(b8).to_array(),
        [-1.0, 2.25, 5.0, 100.0, 7.5, -8.0, 1.25, 0.25]
    );

    let d2a = f64x2::from_array_t(token, [-12.5, 42.25]);
    let d2b = f64x2::from_array_t(token, [-10.0, 19.75]);
    assert_eq!(d2a.min_fast(d2b).to_array(), [-12.5, 19.75]);
    assert_eq!(d2a.max_fast(d2b).to_array(), [-10.0, 42.25]);

    let d4a = f64x4::from_array_t(token, [-12.5, 42.25, 0.125, -0.5]);
    let d4b = f64x4::from_array_t(token, [-10.0, 19.75, 0.25, -1.5]);
    assert_eq!(d4a.min_fast(d4b).to_array(), [-12.5, 19.75, 0.125, -1.5]);
    assert_eq!(d4a.max_fast(d4b).to_array(), [-10.0, 42.25, 0.25, -0.5]);

    // --- to_i32_fast / to_u32_fast ---
    let fi4 = f32x4::from_array_t(token, [0.0, -1.875, 12345.75, -2_147_483_520.0]);
    assert_eq!(fi4.to_i32_fast().to_array(), [0, -1, 12345, -2_147_483_520]);
    let fi8 = f32x8::from_array_t(
        token,
        [
            0.0,
            1.99,
            -1.99,
            255.5,
            -255.5,
            2_147_483_520.0,
            -2_147_483_520.0,
            -2_147_483_648.0,
        ],
    );
    assert_eq!(
        fi8.to_i32_fast().to_array(),
        [0, 1, -1, 255, -255, 2_147_483_520, -2_147_483_520, i32::MIN]
    );

    // Unsigned including >= 2^31 up to 4_294_967_040.0!
    let fu4 = f32x4::from_array_t(token, [0.0, 255.875, 2_147_483_648.0, 4_294_967_040.0]);
    assert_eq!(
        fu4.to_u32_fast().to_array(),
        [0, 255, 2_147_483_648, 4_294_967_040]
    );
    let fu8 = f32x8::from_array_t(
        token,
        [
            0.0,
            1.75,
            65535.5,
            2_147_483_520.0,
            2_147_483_648.0,
            3_000_000_000.0,
            4_000_000_000.0,
            4_294_967_040.0,
        ],
    );
    assert_eq!(
        fu8.to_u32_fast().to_array(),
        [
            0,
            1,
            65535,
            2_147_483_520,
            2_147_483_648,
            3_000_000_000,
            4_000_000_000,
            4_294_967_040
        ]
    );

    // --- broadcast_lane ---
    let v4 = f32x4::from_array_t(token, [10.0, 20.0, 30.0, 40.0]);
    assert_eq!(v4.broadcast_lane::<0>().to_array(), [10.0; 4]);
    assert_eq!(v4.broadcast_lane::<1>().to_array(), [20.0; 4]);
    assert_eq!(v4.broadcast_lane::<2>().to_array(), [30.0; 4]);
    assert_eq!(v4.broadcast_lane::<3>().to_array(), [40.0; 4]);

    let v8 = f32x8::from_array_t(token, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    assert_eq!(v8.broadcast_lane::<0>().to_array(), [1.0; 8]);
    assert_eq!(v8.broadcast_lane::<1>().to_array(), [2.0; 8]);
    assert_eq!(v8.broadcast_lane::<2>().to_array(), [3.0; 8]);
    assert_eq!(v8.broadcast_lane::<3>().to_array(), [4.0; 8]);
    assert_eq!(v8.broadcast_lane::<4>().to_array(), [5.0; 8]);
    assert_eq!(v8.broadcast_lane::<5>().to_array(), [6.0; 8]);
    assert_eq!(v8.broadcast_lane::<6>().to_array(), [7.0; 8]);
    assert_eq!(v8.broadcast_lane::<7>().to_array(), [8.0; 8]);

    // --- shift_left_lanes / shift_right_lanes ---
    assert_eq!(
        v4.shift_left_lanes::<0>().to_array(),
        [10.0, 20.0, 30.0, 40.0]
    );
    assert_eq!(
        v4.shift_left_lanes::<1>().to_array(),
        [0.0, 10.0, 20.0, 30.0]
    );
    assert_eq!(
        v4.shift_left_lanes::<2>().to_array(),
        [0.0, 0.0, 10.0, 20.0]
    );
    assert_eq!(v4.shift_left_lanes::<3>().to_array(), [0.0, 0.0, 0.0, 10.0]);
    assert_eq!(v4.shift_left_lanes::<4>().to_array(), [0.0; 4]);

    assert_eq!(
        v4.shift_right_lanes::<0>().to_array(),
        [10.0, 20.0, 30.0, 40.0]
    );
    assert_eq!(
        v4.shift_right_lanes::<1>().to_array(),
        [20.0, 30.0, 40.0, 0.0]
    );
    assert_eq!(
        v4.shift_right_lanes::<2>().to_array(),
        [30.0, 40.0, 0.0, 0.0]
    );
    assert_eq!(
        v4.shift_right_lanes::<3>().to_array(),
        [40.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(v4.shift_right_lanes::<4>().to_array(), [0.0; 4]);

    assert_eq!(
        v8.shift_left_lanes::<0>().to_array(),
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    );
    assert_eq!(
        v8.shift_left_lanes::<1>().to_array(),
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    );
    assert_eq!(
        v8.shift_left_lanes::<2>().to_array(),
        [0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    assert_eq!(
        v8.shift_left_lanes::<3>().to_array(),
        [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    );
    assert_eq!(
        v8.shift_left_lanes::<4>().to_array(),
        [0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        v8.shift_left_lanes::<5>().to_array(),
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]
    );
    assert_eq!(
        v8.shift_left_lanes::<6>().to_array(),
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0]
    );
    assert_eq!(
        v8.shift_left_lanes::<7>().to_array(),
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    );
    assert_eq!(v8.shift_left_lanes::<8>().to_array(), [0.0; 8]);

    assert_eq!(
        v8.shift_right_lanes::<0>().to_array(),
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    );
    assert_eq!(
        v8.shift_right_lanes::<1>().to_array(),
        [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0]
    );
    assert_eq!(
        v8.shift_right_lanes::<2>().to_array(),
        [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 0.0, 0.0]
    );
    assert_eq!(
        v8.shift_right_lanes::<3>().to_array(),
        [4.0, 5.0, 6.0, 7.0, 8.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(
        v8.shift_right_lanes::<4>().to_array(),
        [5.0, 6.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(
        v8.shift_right_lanes::<5>().to_array(),
        [6.0, 7.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(
        v8.shift_right_lanes::<6>().to_array(),
        [7.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(
        v8.shift_right_lanes::<7>().to_array(),
        [8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    );
    assert_eq!(v8.shift_right_lanes::<8>().to_array(), [0.0; 8]);

    // --- mul_add_lane_relaxed & mul_add_lane4_relaxed ---
    let acc4 = f32x4::from_array_t(token, [1.0, 2.0, 3.0, 4.0]);
    let w4 = f32x4::from_array_t(token, [2.0, -3.0, 4.0, -5.0]);
    let in4 = f32x4::from_array_t(token, [10.0, 20.0, 30.0, 40.0]);
    assert_eq!(
        w4.mul_add_lane_relaxed::<0>(in4, acc4).to_array(),
        [21.0, -28.0, 43.0, -46.0]
    );
    assert_eq!(
        w4.mul_add_lane_relaxed::<1>(in4, acc4).to_array(),
        [41.0, -58.0, 83.0, -96.0]
    );
    assert_eq!(
        w4.mul_add_lane_relaxed::<2>(in4, acc4).to_array(),
        [61.0, -88.0, 123.0, -146.0]
    );
    assert_eq!(
        w4.mul_add_lane_relaxed::<3>(in4, acc4).to_array(),
        [81.0, -118.0, 163.0, -196.0]
    );

    let acc8 = f32x8::from_array_t(token, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let w8 = f32x8::from_array_t(token, [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0]);
    let in8 = f32x8::from_array_t(token, [10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0]);
    // Whole-vector lane select (LANE in 0..8): broadcasts in8[LANE] across all 8 lanes!
    assert_eq!(
        w8.mul_add_lane_relaxed::<2>(in8, acc8).to_array(),
        [31.0, -28.0, 63.0, -56.0, 95.0, -84.0, 127.0, -112.0]
    );
    assert_eq!(
        w8.mul_add_lane_relaxed::<5>(in8, acc8).to_array(),
        [201.0, -198.0, 403.0, -396.0, 605.0, -594.0, 807.0, -792.0]
    );
    // Uniform 4-lane input across all 8 output channels:
    assert_eq!(
        w8.mul_add_lane4_relaxed::<2>(in4, acc8).to_array(),
        [31.0, -28.0, 63.0, -56.0, 95.0, -84.0, 127.0, -112.0]
    );
    assert_eq!(
        w8.mul_add_lane4_relaxed::<3>(in4, acc8).to_array(),
        [41.0, -38.0, 83.0, -76.0, 125.0, -114.0, 167.0, -152.0]
    );

    // --- mul_hrs_q15 ---
    let qa8 = i16x8::from_array_t(
        token,
        [16384, -16384, 8192, -8192, 0, 32767, -32767, i16::MIN],
    );
    let qb8 = i16x8::from_array_t(
        token,
        [16384, 16384, -8192, -8192, 23456, 32767, 16384, i16::MIN],
    );
    let qgot8 = qa8.mul_hrs_q15(qb8).to_array();
    for (i, &got) in qgot8[..7].iter().enumerate() {
        let a_i = qa8.to_array()[i] as i32;
        let b_i = qb8.to_array()[i] as i32;
        let expected = ((a_i * b_i + 0x4000) >> 15) as i16;
        assert_eq!(got, expected, "lane {i}");
    }
    // Rail (i16::MIN, i16::MIN) wraps to -32768 on x86/Scalar or saturates to 32767 on NEON/WASM:
    assert!(qgot8[7] == i16::MIN || qgot8[7] == i16::MAX);

    let qa16_arr: [i16; 16] = core::array::from_fn(|i| ((i as i16) - 8) * 3000);
    let qb16_arr: [i16; 16] = core::array::from_fn(|i| if i % 2 == 0 { 16384 } else { -8192 });
    let qgot16 = i16x16::from_array_t(token, qa16_arr)
        .mul_hrs_q15(i16x16::from_array_t(token, qb16_arr))
        .to_array();
    for i in 0..16 {
        let expected = (((qa16_arr[i] as i32) * (qb16_arr[i] as i32) + 0x4000) >> 15) as i16;
        assert_eq!(qgot16[i], expected);
    }

    // --- dot_i8_i7_add ---
    // Include extreme case [-128; 4] dot [127; 4] = -65024 in lane 0 to prove no i16 saturation!
    let mut da16: [i8; 16] = core::array::from_fn(|i| (i as i8).wrapping_mul(7).wrapping_sub(50));
    let mut db16: [i8; 16] = core::array::from_fn(|i| ((i as i8).wrapping_mul(9)) & 0x7f);
    da16[0..4].copy_from_slice(&[-128, -128, -128, -128]);
    db16[0..4].copy_from_slice(&[127, 127, 127, 127]);
    let dacc4 = [1000, -2000, 3000, -4000];
    let dgot4 = i32x4::from_array_t(token, dacc4)
        .dot_i8_i7_add(
            i8x16::from_array_t(token, da16),
            i8x16::from_array_t(token, db16),
        )
        .to_array();
    for k in 0..4 {
        let dot: i32 = (0..4)
            .map(|j| (da16[4 * k + j] as i32) * (db16[4 * k + j] as i32))
            .sum();
        assert_eq!(dgot4[k], dacc4[k] + dot);
    }
    assert_eq!(dgot4[0], 1000 - 4 * 128 * 127);

    let mut da32: [i8; 32] = core::array::from_fn(|i| (i as i8).wrapping_mul(5).wrapping_sub(80));
    let mut db32: [i8; 32] = core::array::from_fn(|i| ((i as i8).wrapping_mul(11)) & 0x7f);
    da32[28..32].copy_from_slice(&[127, 127, 127, 127]);
    db32[28..32].copy_from_slice(&[127, 127, 127, 127]);
    let dacc8: [i32; 8] = core::array::from_fn(|i| (i as i32 - 3) * 500);
    let dgot8 = i32x8::from_array_t(token, dacc8)
        .dot_i8_i7_add(
            i8x32::from_array_t(token, da32),
            i8x32::from_array_t(token, db32),
        )
        .to_array();
    for k in 0..8 {
        let dot: i32 = (0..4)
            .map(|j| (da32[4 * k + j] as i32) * (db32[4 * k + j] as i32))
            .sum();
        assert_eq!(dgot8[k], dacc8[k] + dot);
    }

    // --- 3-channel RGB block helpers ---
    let rgb4_aos = [
        f32x4::from_array_t(token, [10.0, 20.0, 30.0, 40.0]),
        f32x4::from_array_t(token, [50.0, 60.0, 70.0, 80.0]),
        f32x4::from_array_t(token, [90.0, 100.0, 110.0, 120.0]),
    ];
    let [r4, g4, b4_ch] = f32x4::deinterleave_3ch(rgb4_aos);
    assert_eq!(r4.to_array(), [10.0, 40.0, 70.0, 100.0]);
    assert_eq!(g4.to_array(), [20.0, 50.0, 80.0, 110.0]);
    assert_eq!(b4_ch.to_array(), [30.0, 60.0, 90.0, 120.0]);
    let re4 = f32x4::interleave_3ch([r4, g4, b4_ch]);
    for (got, exp) in re4.iter().zip(&rgb4_aos) {
        assert_eq!(got.to_array(), exp.to_array());
    }

    let raw4: [u8; 12] = [0, 12, 255, 34, 56, 78, 90, 128, 200, 240, 1, 99];
    let (lr4, lg4, lb4) = f32x4::load_4_rgb_u8_t(token, &raw4);
    assert_eq!(lr4.to_array(), [0.0, 34.0, 90.0, 240.0]);
    assert_eq!(lg4.to_array(), [12.0, 56.0, 128.0, 1.0]);
    assert_eq!(lb4.to_array(), [255.0, 78.0, 200.0, 99.0]);
    assert_eq!(f32x4::store_4_rgb_u8(lr4, lg4, lb4), raw4);

    // Out-of-range saturation and rounding parity with store_4_rgba_u8:
    let cr4 = f32x4::from_array_t(token, [-10.0, 0.4, 127.6, 300.0]);
    let cg4 = f32x4::from_array_t(token, [255.4, -0.1, 12.5, 13.5]);
    let cb4 = f32x4::from_array_t(token, [1.2, 2.8, 254.9, -999.0]);
    assert_eq!(
        f32x4::store_4_rgb_u8(cr4, cg4, cb4),
        [0, 255, 1, 0, 0, 3, 128, 12, 255, 255, 14, 0]
    );

    let rgb8_aos = [
        f32x8::from_array_t(token, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        f32x8::from_array_t(token, [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]),
        f32x8::from_array_t(token, [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]),
    ];
    let [r8, g8, b8_ch] = f32x8::deinterleave_3ch(rgb8_aos);
    assert_eq!(r8.to_array(), [1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 19.0, 22.0]);
    assert_eq!(g8.to_array(), [2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0]);
    assert_eq!(
        b8_ch.to_array(),
        [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0]
    );
    let re8 = f32x8::interleave_3ch([r8, g8, b8_ch]);
    for (got, exp) in re8.iter().zip(&rgb8_aos) {
        assert_eq!(got.to_array(), exp.to_array());
    }

    let raw8: [u8; 24] = core::array::from_fn(|i| (i as u8).wrapping_mul(11));
    let (lr8, lg8, lb8) = f32x8::load_8_rgb_u8_t(token, &raw8);
    assert_eq!(f32x8::store_8_rgb_u8(lr8, lg8, lb8), raw8);

    // Exhaustive RGB backend parity (#169):
    // 1. Bit-exact preservation across deinterleave_3ch <-> interleave_3ch
    //    for -0.0, subnormals, infinities, and distinct quiet NaN payloads.
    let bit_patterns_12: [u32; 12] = [
        0x8000_0000, // -0.0
        0x0000_0001, // smallest positive subnormal
        0x8000_0001, // smallest negative subnormal
        0x7f80_0000, // +inf
        0xff80_0000, // -inf
        0x7fc0_0001, // qNaN payload 1
        0x7fc0_0002, // qNaN payload 2
        0x7fc0_1234, // qNaN payload 0x1234
        0xffc0_5678, // negative qNaN payload 0x5678
        0x3f80_0000, // 1.0
        0xbf80_0000, // -1.0
        0x437f_0000, // 255.0
    ];
    let bits_v4 = [
        f32x4::from_array_t(
            token,
            core::array::from_fn(|i| f32::from_bits(bit_patterns_12[i])),
        ),
        f32x4::from_array_t(
            token,
            core::array::from_fn(|i| f32::from_bits(bit_patterns_12[4 + i])),
        ),
        f32x4::from_array_t(
            token,
            core::array::from_fn(|i| f32::from_bits(bit_patterns_12[8 + i])),
        ),
    ];
    let [br4, bg4, bb4] = f32x4::deinterleave_3ch(bits_v4);
    for p in 0..4 {
        assert_eq!(br4.to_array()[p].to_bits(), bit_patterns_12[p * 3]);
        assert_eq!(bg4.to_array()[p].to_bits(), bit_patterns_12[p * 3 + 1]);
        assert_eq!(bb4.to_array()[p].to_bits(), bit_patterns_12[p * 3 + 2]);
    }
    let bits_re4 = f32x4::interleave_3ch([br4, bg4, bb4]);
    for vec_idx in 0..3 {
        for lane in 0..4 {
            assert_eq!(
                bits_re4[vec_idx].to_array()[lane].to_bits(),
                bit_patterns_12[vec_idx * 4 + lane]
            );
        }
    }

    let bit_patterns_24: [u32; 24] = core::array::from_fn(|i| {
        if i < 12 {
            bit_patterns_12[i]
        } else {
            (0x3f00_0000 + ((i as u32) << 13)) ^ (((i as u32) & 1) << 31)
        }
    });
    let bits_v8 = [
        f32x8::from_array_t(
            token,
            core::array::from_fn(|i| f32::from_bits(bit_patterns_24[i])),
        ),
        f32x8::from_array_t(
            token,
            core::array::from_fn(|i| f32::from_bits(bit_patterns_24[8 + i])),
        ),
        f32x8::from_array_t(
            token,
            core::array::from_fn(|i| f32::from_bits(bit_patterns_24[16 + i])),
        ),
    ];
    let [br8, bg8, bb8] = f32x8::deinterleave_3ch(bits_v8);
    for p in 0..8 {
        assert_eq!(br8.to_array()[p].to_bits(), bit_patterns_24[p * 3]);
        assert_eq!(bg8.to_array()[p].to_bits(), bit_patterns_24[p * 3 + 1]);
        assert_eq!(bb8.to_array()[p].to_bits(), bit_patterns_24[p * 3 + 2]);
    }
    let bits_re8 = f32x8::interleave_3ch([br8, bg8, bb8]);
    for vec_idx in 0..3 {
        for lane in 0..8 {
            assert_eq!(
                bits_re8[vec_idx].to_array()[lane].to_bits(),
                bit_patterns_24[vec_idx * 8 + lane]
            );
        }
    }

    // 2. Full 0..=255 load + store roundtrip across non-symmetric RGB bytes and
    //    unaligned slice tails, plus exact parity with store_{4,8}_rgba_u8 on
    //    halfway ties, NaNs, infinities, negatives, and huge floats.
    let mut full_rgb = [0u8; 256 * 3 + 7];
    for p in 0..256 {
        full_rgb[3 + p * 3] = p as u8;
        full_rgb[3 + p * 3 + 1] = (255 - p) as u8;
        full_rgb[3 + p * 3 + 2] = (p as u8).wrapping_mul(73).wrapping_add(19);
    }
    for base in (3..(3 + 256 * 3)).step_by(24) {
        let chunk24: &[u8; 24] = full_rgb[base..base + 24].try_into().unwrap();
        let (r, g, b) = f32x8::load_8_rgb_u8_t(token, chunk24);
        let exp_r: [f32; 8] = core::array::from_fn(|i| chunk24[i * 3] as f32);
        let exp_g: [f32; 8] = core::array::from_fn(|i| chunk24[i * 3 + 1] as f32);
        let exp_b: [f32; 8] = core::array::from_fn(|i| chunk24[i * 3 + 2] as f32);
        assert_eq!(r.to_array(), exp_r);
        assert_eq!(g.to_array(), exp_g);
        assert_eq!(b.to_array(), exp_b);
        assert_eq!(&f32x8::store_8_rgb_u8(r, g, b), chunk24);

        let chunk12_lo: &[u8; 12] = chunk24[..12].try_into().unwrap();
        let chunk12_hi: &[u8; 12] = chunk24[12..].try_into().unwrap();
        let (r4l, g4l, b4l) = f32x4::load_4_rgb_u8_t(token, chunk12_lo);
        let (r4h, g4h, b4h) = f32x4::load_4_rgb_u8_t(token, chunk12_hi);
        assert_eq!(&f32x4::store_4_rgb_u8(r4l, g4l, b4l), chunk12_lo);
        assert_eq!(&f32x4::store_4_rgb_u8(r4h, g4h, b4h), chunk12_hi);
    }

    let edge_vals: [f32; 24] = [
        0.0,
        -0.0,
        0.5,
        1.5,
        2.5,
        3.5,
        127.5,
        128.5,
        253.5,
        254.5,
        255.0,
        255.5,
        256.0,
        -0.5,
        -1.0,
        -1e30,
        1e30,
        1e38,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        f32::from_bits(0xffc0_0001),
        0.499_999_97,
        254.500_02,
    ];
    let er8 = f32x8::from_array_t(token, core::array::from_fn(|i| edge_vals[i]));
    let eg8 = f32x8::from_array_t(token, core::array::from_fn(|i| edge_vals[8 + i]));
    let eb8 = f32x8::from_array_t(token, core::array::from_fn(|i| edge_vals[16 + i]));
    let rgba8_ref = f32x8::store_8_rgba_u8(er8, eg8, eb8, f32x8::zero_t(token));
    let rgb8_expected: [u8; 24] = core::array::from_fn(|i| rgba8_ref[(i / 3) * 4 + (i % 3)]);
    assert_eq!(f32x8::store_8_rgb_u8(er8, eg8, eb8), rgb8_expected);

    for half in [0usize, 4usize] {
        let er4 = f32x4::from_array_t(token, core::array::from_fn(|i| edge_vals[half + i]));
        let eg4 = f32x4::from_array_t(token, core::array::from_fn(|i| edge_vals[8 + half + i]));
        let eb4 = f32x4::from_array_t(token, core::array::from_fn(|i| edge_vals[16 + half + i]));
        let rgba4_ref = f32x4::store_4_rgba_u8(er4, eg4, eb4, f32x4::zero_t(token));
        let rgb4_expected: [u8; 12] = core::array::from_fn(|i| rgba4_ref[(i / 3) * 4 + (i % 3)]);
        assert_eq!(f32x4::store_4_rgb_u8(er4, eg4, eb4), rgb4_expected);
    }
}

#[cfg(feature = "w512")]
fn check_w512_backend<T>(token: T)
where
    T: SimdToken + F32x16Convert + F32x16ToU32 + F64x8Backend + I16x32Backend + I32x16DotI8,
{
    let a16 = f32x16::from_array_t(token, core::array::from_fn(|i| (i as f32) * 2.0 - 15.0));
    let b16 = f32x16::from_array_t(token, core::array::from_fn(|i| 15.0 - (i as f32) * 2.0));
    let min16 = a16.min_fast(b16).to_array();
    let max16 = a16.max_fast(b16).to_array();
    for (i, (&mn, &mx)) in min16.iter().zip(&max16).enumerate() {
        let av = (i as f32) * 2.0 - 15.0;
        let bv = 15.0 - (i as f32) * 2.0;
        assert_eq!(mn, av.min(bv));
        assert_eq!(mx, av.max(bv));
    }

    let d8a = f64x8::from_array_t(token, core::array::from_fn(|i| (i as f64) - 3.5));
    let d8b = f64x8::from_array_t(token, core::array::from_fn(|i| 3.5 - (i as f64)));
    let d8_min = d8a.min_fast(d8b).to_array();
    let d8_max = d8a.max_fast(d8b).to_array();
    for (i, (&mn, &mx)) in d8_min.iter().zip(&d8_max).enumerate() {
        let av = (i as f64) - 3.5;
        let bv = 3.5 - (i as f64);
        assert_eq!(mn, av.min(bv));
        assert_eq!(mx, av.max(bv));
    }

    let fi16 = f32x16::from_array_t(token, core::array::from_fn(|i| ((i as f32) - 8.0) * 100.75));
    let fi16_got = fi16.to_i32_fast().to_array();
    for (i, &got) in fi16_got.iter().enumerate() {
        assert_eq!(got, (((i as f32) - 8.0) * 100.75) as i32);
    }

    let fu16_in: [f32; 16] = core::array::from_fn(|i| (i as f32) * 280_000_000.0);
    let fu16_got = f32x16::from_array_t(token, fu16_in)
        .to_u32_fast()
        .to_array();
    for (&got, &inp) in fu16_got.iter().zip(&fu16_in) {
        assert_eq!(got, inp as u32);
    }

    let qa32_arr: [i16; 32] = core::array::from_fn(|i| ((i as i16) - 16) * 1500);
    let qb32_arr: [i16; 32] = core::array::from_fn(|i| if i % 2 == 0 { 16384 } else { -12288 });
    let qgot32 = i16x32::from_array_t(token, qa32_arr)
        .mul_hrs_q15(i16x32::from_array_t(token, qb32_arr))
        .to_array();
    for ((&got, &a), &b) in qgot32.iter().zip(&qa32_arr).zip(&qb32_arr) {
        let expected = (((a as i32) * (b as i32) + 0x4000) >> 15) as i16;
        assert_eq!(got, expected);
    }

    let da64: [i8; 64] = core::array::from_fn(|i| (i as i8).wrapping_mul(3).wrapping_sub(90));
    let db64: [i8; 64] = core::array::from_fn(|i| ((i as i8).wrapping_mul(7)) & 0x7f);
    let dacc16: [i32; 16] = core::array::from_fn(|i| (i as i32) * 100 - 800);
    let dgot16 = i32x16::from_array_t(token, dacc16)
        .dot_i8_i7_add(
            i8x64::from_array_t(token, da64),
            i8x64::from_array_t(token, db64),
        )
        .to_array();
    for k in 0..16 {
        let dot: i32 = (0..4)
            .map(|j| (da64[4 * k + j] as i32) * (db64[4 * k + j] as i32))
            .sum();
        assert_eq!(dgot16[k], dacc16[k] + dot);
    }
}

#[test]
fn neural_vision_ops_across_all_backends() {
    let scalar = ScalarToken;
    check_narrow_backend(scalar);
    #[cfg(feature = "w512")]
    check_w512_backend(scalar);

    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = archmage::X64V3Token::summon() {
            check_narrow_backend(t);
            #[cfg(feature = "w512")]
            check_w512_backend(t);
        }
        #[cfg(feature = "avx512")]
        {
            if let Some(t) = archmage::X64V4Token::summon() {
                check_narrow_backend(t);
                #[cfg(feature = "w512")]
                check_w512_backend(t);
            }
            if let Some(t) = archmage::X64V4xToken::summon() {
                check_narrow_backend(t);
                #[cfg(feature = "w512")]
                check_w512_backend(t);
            }
            if let Some(t) = archmage::Avx512Fp16Token::summon() {
                check_narrow_backend(t.magetypes());
                // Exercise Avx512Fp16Token's direct F32x4Backend / F32x8Backend delegation too:
                let raw4: [u8; 12] = [1, 2, 3, 40, 50, 60, 70, 80, 90, 200, 210, 220];
                let (r4, g4, b4) = f32x4::load_4_rgb_u8_t(t, &raw4);
                assert_eq!(f32x4::store_4_rgb_u8(r4, g4, b4), raw4);
                let raw8: [u8; 24] = core::array::from_fn(|i| (i as u8) * 9 + 7);
                let (r8, g8, b8) = f32x8::load_8_rgb_u8_t(t, &raw8);
                assert_eq!(f32x8::store_8_rgb_u8(r8, g8, b8), raw8);
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if let Some(t) = archmage::NeonToken::summon() {
            check_narrow_backend(t);
            #[cfg(feature = "w512")]
            check_w512_backend(t);
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        if let Some(t) = archmage::Wasm128Token::summon() {
            check_narrow_backend(t);
            #[cfg(feature = "w512")]
            check_w512_backend(t);
        }
        #[cfg(target_feature = "relaxed-simd")]
        if let Some(t) = archmage::Wasm128RelaxedFmaToken::summon() {
            check_narrow_backend(t);
            #[cfg(feature = "w512")]
            check_w512_backend(t);
        }
    }
}

#[archmage::magetypes(
    +v3_vnni,
    +arm_v2,
    +arm_v3,
    +wasm128_relaxed,
    +wasm128_relaxed_fma,
    define(f32x8, u32x8, i32x8, i8x32)
)]
fn subtier_magetypes_define_kernel(
    token: Token,
    rgb: &[u8; 24],
    weights_a: &[i8; 32],
    weights_b: &[i8; 32],
) -> ([u8; 24], [u32; 8], [i32; 8]) {
    let mt = token.magetypes();
    let (r, g, b) = f32x8::load_8_rgb_u8_t(mt, rgb);
    let zero = f32x8::zero_t(mt);
    let hi = f32x8::splat_t(mt, 255.0);
    let r = r.max_fast(zero).min_fast(hi);
    let g = g.max_fast(zero).min_fast(hi);
    let b = b.max_fast(zero).min_fast(hi);
    let packed = f32x8::store_8_rgb_u8(r, g, b);
    let r_u32 = (r * f32x8::splat_t(mt, 16_000_000.0))
        .to_u32_fast()
        .to_array();
    let dot = i32x8::zero_t(mt)
        .dot_i8_i7_add(i8x32::load_t(mt, weights_a), i8x32::load_t(mt, weights_b))
        .to_array();
    let _ = u32x8::splat_t(mt, 0);
    (packed, r_u32, dot)
}

#[test]
fn magetypes_macro_define_supports_subtiers_and_magetypes_token() {
    let rgb: [u8; 24] = core::array::from_fn(|i| (i as u8) * 10 + 5);
    let wa: [i8; 32] = core::array::from_fn(|i| (i as i8) * 2 - 30);
    let wb: [i8; 32] = core::array::from_fn(|i| ((i as i8) * 3) & 0x7f);

    let (exp_rgb, exp_u32, exp_dot) =
        subtier_magetypes_define_kernel_scalar(ScalarToken, &rgb, &wa, &wb);
    assert_eq!(exp_rgb, rgb);

    let (got_rgb, got_u32, got_dot) = archmage::incant!(
        subtier_magetypes_define_kernel(&rgb, &wa, &wb),
        [
            +v3_vnni,
            +arm_v2,
            +arm_v3,
            +wasm128_relaxed,
            +wasm128_relaxed_fma
        ]
    );
    assert_eq!(got_rgb, exp_rgb);
    assert_eq!(got_u32, exp_u32);
    assert_eq!(got_dot, exp_dot);
}
