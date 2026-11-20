#![forbid(unsafe_code)]

use archmage::ScalarToken;
use magetypes::simd::generic::{f32x4, f32x8, f64x2, f64x4, i32x8, u8x16};

macro_rules! check_padded_tail_type {
    ($vec:ident, $token:expr, $sample:expr, $pad:expr, $sentinel:expr, $to_bits:expr) => {{
        const LANES: usize = $vec::<ScalarToken>::LANES;
        let sample: [_; LANES] = $sample;
        let pad = $pad;
        let sentinel = $sentinel;
        let expected_full = $vec::from_slice_t($token, &sample).to_array();

        for len in 0..=LANES {
            // 1. load_padded_t: prefix matches slice[..len], high lanes equal pad bit-for-bit.
            let loaded = $vec::load_padded_t($token, &sample[..len], pad).to_array();
            for i in 0..len {
                assert_eq!(
                    ($to_bits)(loaded[i]),
                    ($to_bits)(sample[i]),
                    "{} len={len} lane={i}",
                    stringify!($vec),
                );
            }
            for i in len..LANES {
                assert_eq!(
                    ($to_bits)(loaded[i]),
                    ($to_bits)(pad),
                    "{} len={len} pad lane={i}",
                    stringify!($vec),
                );
            }
            if len == LANES {
                for i in 0..LANES {
                    assert_eq!(($to_bits)(loaded[i]), ($to_bits)(expected_full[i]));
                }
            }

            // 2. store_prefix: writes exact low lanes without touching sentinels on either side.
            let vec = $vec::from_array_t($token, sample);
            let mut buf = [sentinel; LANES + 2];
            vec.store_prefix(&mut buf[1..1 + len]);
            assert_eq!(
                ($to_bits)(buf[0]),
                ($to_bits)(sentinel),
                "{} leading sentinel clobbered at len={len}",
                stringify!($vec),
            );
            for i in 0..len {
                assert_eq!(
                    ($to_bits)(buf[1 + i]),
                    ($to_bits)(sample[i]),
                    "{} stored prefix len={len} lane={i}",
                    stringify!($vec),
                );
            }
            for (offset, &item) in buf[1 + len..].iter().enumerate() {
                assert_eq!(
                    ($to_bits)(item),
                    ($to_bits)(sentinel),
                    "{} trailing sentinel clobbered at len={len} +{offset}",
                    stringify!($vec),
                );
            }
            if len == LANES {
                let mut full_out = [sentinel; LANES];
                vec.store(&mut full_out);
                for i in 0..LANES {
                    assert_eq!(($to_bits)(buf[1 + i]), ($to_bits)(full_out[i]));
                }
            }
        }
    }};
}

fn verify_token_narrow<T>(token: T)
where
    T: magetypes::simd::backends::F32x4Backend
        + magetypes::simd::backends::F32x8Backend
        + magetypes::simd::backends::F64x2Backend
        + magetypes::simd::backends::F64x4Backend
        + magetypes::simd::backends::I32x8Backend
        + magetypes::simd::backends::U8x16Backend,
{
    check_padded_tail_type!(
        f32x4,
        token,
        [1.25f32, -2.5, 0.0, 99.75],
        -0.0f32,
        -999.5f32,
        f32::to_bits
    );
    check_padded_tail_type!(
        f32x8,
        token,
        [1.0f32, -2.0, 3.5, -4.25, 5.125, -6.5, 7.0, 8.75],
        1.0f32,
        -999.5f32,
        f32::to_bits
    );
    check_padded_tail_type!(
        f64x2,
        token,
        [3.125f64, -7.5],
        -0.0f64,
        -999.25f64,
        f64::to_bits
    );
    check_padded_tail_type!(
        f64x4,
        token,
        [1.0f64, -2.5, 3.25, -4.75],
        f64::INFINITY,
        -999.25f64,
        f64::to_bits
    );
    check_padded_tail_type!(
        i32x8,
        token,
        [10i32, -20, 30, -40, 50, -60, 70, i32::MAX],
        -7i32,
        i32::MIN,
        |x: i32| x
    );
    check_padded_tail_type!(
        u8x16,
        token,
        [1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 250],
        0xAAu8,
        0x55u8,
        |x: u8| x
    );
}

#[cfg(feature = "w512")]
fn verify_token_w512<T>(token: T)
where
    T: magetypes::simd::backends::F32x16Backend + magetypes::simd::backends::F64x8Backend,
{
    use magetypes::simd::generic::{f32x16, f64x8};

    check_padded_tail_type!(
        f32x16,
        token,
        core::array::from_fn(|i| (i as f32) * 1.5 - 3.0),
        0.0f32,
        -999.5f32,
        f32::to_bits
    );
    check_padded_tail_type!(
        f64x8,
        token,
        core::array::from_fn(|i| (i as f64) * 2.25 - 5.0),
        -0.0f64,
        -999.25f64,
        f64::to_bits
    );
}

#[test]
fn load_padded_and_store_prefix_all_backends() {
    verify_token_narrow(ScalarToken);
    #[cfg(feature = "w512")]
    verify_token_w512(ScalarToken);

    #[cfg(target_arch = "x86_64")]
    {
        use archmage::SimdToken;
        if let Some(t) = archmage::X64V3Token::summon() {
            verify_token_narrow(t);
            #[cfg(feature = "w512")]
            verify_token_w512(t);
        }
        #[cfg(feature = "avx512")]
        {
            if let Some(t) = archmage::X64V4Token::summon() {
                verify_token_narrow(t);
                #[cfg(feature = "w512")]
                verify_token_w512(t);
            }
            if let Some(t) = archmage::X64V4xToken::summon() {
                verify_token_narrow(t);
                #[cfg(feature = "w512")]
                verify_token_w512(t);
            }
            if let Some(t) = archmage::Avx512Fp16Token::summon() {
                verify_token_narrow(t);
                #[cfg(feature = "w512")]
                verify_token_w512(t.v4());
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        use archmage::SimdToken;
        if let Some(t) = archmage::NeonToken::summon() {
            verify_token_narrow(t);
            #[cfg(feature = "w512")]
            verify_token_w512(t);
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        use archmage::SimdToken;
        if let Some(t) = archmage::Wasm128Token::summon() {
            verify_token_narrow(t);
            #[cfg(feature = "w512")]
            verify_token_w512(t);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[should_panic(expected = "load_padded_t requires slice.len() <=")]
fn load_padded_panics_when_slice_exceeds_lanes() {
    let over = [1.0f32; 9];
    let _ = f32x8::load_padded_t(ScalarToken, &over, 0.0);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[should_panic(expected = "store_prefix requires out.len() <=")]
fn store_prefix_panics_when_out_exceeds_lanes() {
    let vec = f32x8::splat_t(ScalarToken, 1.0);
    let mut over = [0.0f32; 9];
    vec.store_prefix(&mut over);
}
