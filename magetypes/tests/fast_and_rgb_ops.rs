//! Integration tests for `f32` `min_fast`/`max_fast`/`to_i32_fast` and
//! 3-channel RGB block helpers (`deinterleave_3ch`, `interleave_3ch`,
//! `load_{4,8}_rgb_u8_t`, `store_{4,8}_rgb_u8`).

#![forbid(unsafe_code)]

use archmage::{ScalarToken, SimdToken};
use magetypes::simd::backends::{F32x4Backend, F32x4Convert, F32x8Backend, F32x8Convert};
use magetypes::simd::generic::{f32x4, f32x8};

#[cfg(feature = "w512")]
use magetypes::simd::backends::{F32x16Backend, F32x16Convert};
#[cfg(feature = "w512")]
use magetypes::simd::generic::f32x16;

fn check_narrow_backend<T>(token: T, tier: &str)
where
    T: SimdToken + F32x4Backend + F32x4Convert + F32x8Backend + F32x8Convert,
{
    // --- min_fast / max_fast: finite inputs ---
    let a4 = f32x4::from_array_t(token, [-3.5, 2.25, 0.0, 100.0]);
    let b4 = f32x4::from_array_t(token, [-1.0, -4.5, 5.0, 100.0]);
    assert_eq!(
        a4.min_fast(b4).to_array(),
        [-3.5, -4.5, 0.0, 100.0],
        "{tier}"
    );
    assert_eq!(
        a4.max_fast(b4).to_array(),
        [-1.0, 2.25, 5.0, 100.0],
        "{tier}"
    );

    let a8 = f32x8::from_array_t(token, [-3.5, 2.25, 0.5, 100.0, 7.0, -8.0, 1.25, -0.25]);
    let b8 = f32x8::from_array_t(token, [-1.0, -4.5, 5.0, 99.0, 7.5, -9.0, 1.25, 0.25]);
    assert_eq!(
        a8.min_fast(b8).to_array(),
        [-3.5, -4.5, 0.5, 99.0, 7.0, -9.0, 1.25, -0.25],
        "{tier}"
    );
    assert_eq!(
        a8.max_fast(b8).to_array(),
        [-1.0, 2.25, 5.0, 100.0, 7.5, -8.0, 1.25, 0.25],
        "{tier}"
    );

    // --- min_fast / max_fast: NaN & infinity sanitization when bounds are second operand ---
    let lo4 = f32x4::zero_t(token);
    let hi4 = f32x4::splat_t(token, 255.0);
    assert_eq!(
        f32x4::splat_t(token, f32::NAN)
            .max_fast(lo4)
            .min_fast(hi4)
            .to_array(),
        [0.0; 4],
        "{tier} f32x4 NaN clamp"
    );
    let clamp4_in = f32x4::from_array_t(token, [f32::NAN, f32::NEG_INFINITY, f32::INFINITY, 42.5]);
    assert_eq!(
        clamp4_in.max_fast(lo4).min_fast(hi4).to_array(),
        [0.0, 0.0, 255.0, 42.5],
        "{tier} f32x4 edge clamp"
    );

    let lo8 = f32x8::zero_t(token);
    let hi8 = f32x8::splat_t(token, 255.0);
    assert_eq!(
        f32x8::splat_t(token, f32::NAN)
            .max_fast(lo8)
            .min_fast(hi8)
            .to_array(),
        [0.0; 8],
        "{tier} f32x8 NaN clamp"
    );
    let clamp8_in = f32x8::from_array_t(
        token,
        [
            f32::NAN,
            f32::from_bits(0xffc0_0001),
            f32::NEG_INFINITY,
            f32::INFINITY,
            -0.5,
            0.0,
            128.25,
            255.5,
        ],
    );
    assert_eq!(
        clamp8_in.max_fast(lo8).min_fast(hi8).to_array(),
        [0.0, 0.0, 0.0, 255.0, 0.0, 0.0, 128.25, 255.0],
        "{tier} f32x8 edge clamp"
    );

    // --- to_i32_fast ---
    let fi4 = f32x4::from_array_t(token, [0.0, -1.875, 12345.75, -2_147_483_520.0]);
    assert_eq!(
        fi4.to_i32_fast().to_array(),
        [0, -1, 12345, -2_147_483_520],
        "{tier} f32x4 to_i32_fast"
    );
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
        [0, 1, -1, 255, -255, 2_147_483_520, -2_147_483_520, i32::MIN],
        "{tier} f32x8 to_i32_fast"
    );

    // --- 3-channel deinterleave_3ch / interleave_3ch (numeric + bit-exact) ---
    let rgb4_aos = [
        f32x4::from_array_t(token, [1.0, 2.0, 3.0, 4.0]),
        f32x4::from_array_t(token, [5.0, 6.0, 7.0, 8.0]),
        f32x4::from_array_t(token, [9.0, 10.0, 11.0, 12.0]),
    ];
    let [r4, g4, b4] = f32x4::deinterleave_3ch(rgb4_aos);
    assert_eq!(r4.to_array(), [1.0, 4.0, 7.0, 10.0], "{tier}");
    assert_eq!(g4.to_array(), [2.0, 5.0, 8.0, 11.0], "{tier}");
    assert_eq!(b4.to_array(), [3.0, 6.0, 9.0, 12.0], "{tier}");
    for (got, exp) in f32x4::interleave_3ch([r4, g4, b4]).iter().zip(&rgb4_aos) {
        assert_eq!(got.to_array(), exp.to_array(), "{tier}");
    }

    let rgb8_aos = [
        f32x8::from_array_t(token, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
        f32x8::from_array_t(token, [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]),
        f32x8::from_array_t(token, [17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]),
    ];
    let [r8, g8, b8] = f32x8::deinterleave_3ch(rgb8_aos);
    assert_eq!(
        r8.to_array(),
        [1.0, 4.0, 7.0, 10.0, 13.0, 16.0, 19.0, 22.0],
        "{tier}"
    );
    assert_eq!(
        g8.to_array(),
        [2.0, 5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0],
        "{tier}"
    );
    assert_eq!(
        b8.to_array(),
        [3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0],
        "{tier}"
    );
    for (got, exp) in f32x8::interleave_3ch([r8, g8, b8]).iter().zip(&rgb8_aos) {
        assert_eq!(got.to_array(), exp.to_array(), "{tier}");
    }

    // Bit-exact preservation across deinterleave_3ch <-> interleave_3ch for
    // -0.0, subnormals, infinities, and distinct quiet NaN payloads.
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
    let br4_arr = br4.to_array();
    let bg4_arr = bg4.to_array();
    let bb4_arr = bb4.to_array();
    for p in 0..4 {
        assert_eq!(br4_arr[p].to_bits(), bit_patterns_12[p * 3], "{tier}");
        assert_eq!(bg4_arr[p].to_bits(), bit_patterns_12[p * 3 + 1], "{tier}");
        assert_eq!(bb4_arr[p].to_bits(), bit_patterns_12[p * 3 + 2], "{tier}");
    }
    let bits_re4 = f32x4::interleave_3ch([br4, bg4, bb4]);
    for (vec_idx, vec_item) in bits_re4.iter().enumerate() {
        let arr = vec_item.to_array();
        for (lane, val) in arr.iter().enumerate() {
            assert_eq!(val.to_bits(), bit_patterns_12[vec_idx * 4 + lane], "{tier}");
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
    let br8_arr = br8.to_array();
    let bg8_arr = bg8.to_array();
    let bb8_arr = bb8.to_array();
    for p in 0..8 {
        assert_eq!(br8_arr[p].to_bits(), bit_patterns_24[p * 3], "{tier}");
        assert_eq!(bg8_arr[p].to_bits(), bit_patterns_24[p * 3 + 1], "{tier}");
        assert_eq!(bb8_arr[p].to_bits(), bit_patterns_24[p * 3 + 2], "{tier}");
    }
    let bits_re8 = f32x8::interleave_3ch([br8, bg8, bb8]);
    for (vec_idx, vec_item) in bits_re8.iter().enumerate() {
        let arr = vec_item.to_array();
        for (lane, val) in arr.iter().enumerate() {
            assert_eq!(val.to_bits(), bit_patterns_24[vec_idx * 8 + lane], "{tier}");
        }
    }

    // --- Full 0..=255 load_{4,8}_rgb_u8_t + store_{4,8}_rgb_u8 roundtrip & parity ---
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
        assert_eq!(r.to_array(), exp_r, "{tier}");
        assert_eq!(g.to_array(), exp_g, "{tier}");
        assert_eq!(b.to_array(), exp_b, "{tier}");
        assert_eq!(&f32x8::store_8_rgb_u8(r, g, b), chunk24, "{tier}");

        let chunk12_lo: &[u8; 12] = chunk24[..12].try_into().unwrap();
        let chunk12_hi: &[u8; 12] = chunk24[12..].try_into().unwrap();
        let (r4l, g4l, b4l) = f32x4::load_4_rgb_u8_t(token, chunk12_lo);
        let (r4h, g4h, b4h) = f32x4::load_4_rgb_u8_t(token, chunk12_hi);
        assert_eq!(&f32x4::store_4_rgb_u8(r4l, g4l, b4l), chunk12_lo, "{tier}");
        assert_eq!(&f32x4::store_4_rgb_u8(r4h, g4h, b4h), chunk12_hi, "{tier}");
    }

    // Halfway ties, NaN, infinities, negatives, and out-of-range floats match store_{4,8}_rgba_u8.
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
    assert_eq!(
        f32x8::store_8_rgb_u8(er8, eg8, eb8),
        rgb8_expected,
        "{tier}"
    );

    for half in [0usize, 4usize] {
        let er4 = f32x4::from_array_t(token, core::array::from_fn(|i| edge_vals[half + i]));
        let eg4 = f32x4::from_array_t(token, core::array::from_fn(|i| edge_vals[8 + half + i]));
        let eb4 = f32x4::from_array_t(token, core::array::from_fn(|i| edge_vals[16 + half + i]));
        let rgba4_ref = f32x4::store_4_rgba_u8(er4, eg4, eb4, f32x4::zero_t(token));
        let rgb4_expected: [u8; 12] = core::array::from_fn(|i| rgba4_ref[(i / 3) * 4 + (i % 3)]);
        assert_eq!(
            f32x4::store_4_rgb_u8(er4, eg4, eb4),
            rgb4_expected,
            "{tier}"
        );
    }
}

#[cfg(feature = "w512")]
fn check_w512_backend<T>(token: T, tier: &str)
where
    T: SimdToken + F32x16Backend + F32x16Convert,
{
    let a16 = f32x16::from_array_t(token, core::array::from_fn(|i| (i as f32) * 2.0 - 15.0));
    let b16 = f32x16::from_array_t(token, core::array::from_fn(|i| 15.0 - (i as f32) * 2.0));
    let min16 = a16.min_fast(b16).to_array();
    let max16 = a16.max_fast(b16).to_array();
    for (i, (&mn, &mx)) in min16.iter().zip(&max16).enumerate() {
        let av = (i as f32) * 2.0 - 15.0;
        let bv = 15.0 - (i as f32) * 2.0;
        assert_eq!(mn, av.min(bv), "{tier} min16 lane {i}");
        assert_eq!(mx, av.max(bv), "{tier} max16 lane {i}");
    }

    let lo16 = f32x16::zero_t(token);
    let hi16 = f32x16::splat_t(token, 255.0);
    assert_eq!(
        f32x16::splat_t(token, f32::NAN)
            .max_fast(lo16)
            .min_fast(hi16)
            .to_array(),
        [0.0; 16],
        "{tier} f32x16 NaN clamp"
    );

    let fi16 = f32x16::from_array_t(
        token,
        core::array::from_fn(|i| match i {
            0 => -2_147_483_520.0,
            15 => 2_147_483_520.0,
            _ => ((i as f32) - 8.0) * 100.75,
        }),
    );
    let fi16_got = fi16.to_i32_fast().to_array();
    let fi16_src = fi16.to_array();
    for (i, (&got, &src)) in fi16_got.iter().zip(&fi16_src).enumerate() {
        assert_eq!(got, src as i32, "{tier} to_i32_fast lane {i}");
    }
}

#[test]
fn fast_and_rgb_ops_all_backends() {
    check_narrow_backend(ScalarToken, "scalar");
    #[cfg(feature = "w512")]
    check_w512_backend(ScalarToken, "scalar");

    #[cfg(target_arch = "x86_64")]
    {
        if let Some(t) = archmage::X64V3Token::summon() {
            check_narrow_backend(t, "x86-v3");
            #[cfg(feature = "w512")]
            check_w512_backend(t, "x86-v3");
        }
        #[cfg(feature = "avx512")]
        {
            if let Some(t) = archmage::X64V4Token::summon() {
                check_narrow_backend(t, "x86-v4");
                #[cfg(feature = "w512")]
                check_w512_backend(t, "x86-v4");
            }
            if let Some(t) = archmage::X64V4xToken::summon() {
                check_narrow_backend(t, "x86-v4x");
                #[cfg(feature = "w512")]
                check_w512_backend(t, "x86-v4x");
            }
            if let Some(t) = archmage::Avx512Fp16Token::summon() {
                check_narrow_backend(t, "x86-fp16");
                #[cfg(feature = "w512")]
                check_w512_backend(t.v4(), "x86-fp16-v4");
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        if let Some(t) = archmage::NeonToken::summon() {
            check_narrow_backend(t, "neon");
            #[cfg(feature = "w512")]
            check_w512_backend(t, "neon");
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        if let Some(t) = archmage::Wasm128Token::summon() {
            check_narrow_backend(t, "wasm128");
            #[cfg(feature = "w512")]
            check_w512_backend(t, "wasm128");
        }
    }
}
