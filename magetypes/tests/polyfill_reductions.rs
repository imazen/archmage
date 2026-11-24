//! Pinned behavior tests for polyfilled float reductions (`sub_reduce_tree!`).
//!
//! Polyfilled wide reductions fold sub-vectors lane-wise using the narrower
//! backend's `add`/`min`/`max` before a single horizontal reduction. The
//! witnesses exercise the evaluation order and the NEON f32 NaN policy.

/// V3 `f32x16::reduce_add` folds the two 256-bit halves lane-wise (`a[i] + a[i+8]`)
/// before one horizontal sum, matching AVX-512 folding order.
#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "w512"))]
#[test]
fn v3_f32x16_reduce_add_lane_fold() {
    use archmage::SimdToken;
    if let Some(token) = archmage::X64V3Token::summon() {
        let mut lanes = [0.0f32; 16];
        lanes[0] = 1e20;
        lanes[1] = 1.0;
        lanes[8] = -1e20;
        let v = magetypes::simd::generic::f32x16::from_array_t(token, lanes);
        assert_eq!(v.reduce_add().to_bits(), 1.0f32.to_bits());
    }
}

/// NEON `f32x8::reduce_min`/`max` folds halves with number-preferring `min`/`max`
/// before `vminnmvq_f32`/`vmaxnmvq_f32`, matching NEON `f32x4::reduce_min`/`max`.
#[cfg(target_arch = "aarch64")]
#[test]
fn neon_f32x8_reduce_extrema_number_preferring() {
    use archmage::SimdToken;
    use magetypes::simd::generic::f32x8;

    let token = archmage::NeonToken::summon().expect("AArch64 NEON");
    let v = f32x8::from_array_t(token, [f32::NAN, 8.0, 3.0, 4.0, 1.0, f32::NAN, 5.0, 6.0]);
    assert_eq!(v.reduce_min().to_bits(), 1.0f32.to_bits());
    assert_eq!(v.reduce_max().to_bits(), 8.0f32.to_bits());

    let zeros = f32x8::from_array_t(token, [-0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, -0.0]);
    assert_eq!(zeros.reduce_min().to_bits(), (-0.0_f32).to_bits());
    assert_eq!(zeros.reduce_max().to_bits(), 0.0_f32.to_bits());
}

/// WASM `f64x4::reduce_add` folds the two 128-bit halves lane-wise (`f64x2_add`)
/// before a single scalar addition.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[test]
fn wasm_f64x4_reduce_add_lane_fold() {
    use archmage::SimdToken;
    let token = archmage::Wasm128Token::summon().expect("WASM SIMD128");
    let v = magetypes::simd::generic::f64x4::from_array_t(token, [1e100, 1.0, -1e100, 0.0]);
    assert_eq!(v.reduce_add().to_bits(), 1.0f64.to_bits());
}

#[cfg(feature = "w512")]
macro_rules! check_wide_lanes {
    ($token:expr) => {{
        use magetypes::simd::generic::{f32x16, i16x32, u8x64};

        let token = $token;
        let bytes = core::array::from_fn(|i| {
            if [0, 15, 16, 31, 32, 47, 48, 63].contains(&i) {
                255
            } else {
                i as u8
            }
        });
        let a = u8x64::from_array_t(token, bytes);
        let zero = u8x64::zero_t(token);
        assert_eq!(a.bitmask(), 0x8001_8001_8001_8001);
        assert_eq!(a.simd_gt(zero).bitmask(), u64::MAX);
        assert_eq!(a.reduce_add_u32(), bytes.into_iter().map(u32::from).sum());
        assert_eq!(
            a.sum_abs_diff(u8x64::splat_t(token, 128)),
            bytes.into_iter().map(|x| u32::from(x.abs_diff(128))).sum()
        );

        let low = core::array::from_fn(|i| i as i16 * 15 - 120);
        let high = core::array::from_fn(|i| 300 - i as i16 * 17);
        let expected = core::array::from_fn(|i| {
            let lane = if i < 32 { low[i] } else { high[i - 32] };
            lane.clamp(0, 255) as u8
        });
        assert_eq!(
            i16x32::from_array_t(token, low)
                .narrow_saturating_u8(i16x32::from_array_t(token, high))
                .to_array(),
            expected
        );

        let x = [
            -0.0_f32,
            0.0,
            f32::from_bits(0x7f80_0042),
            f32::NEG_INFINITY,
            f32::INFINITY,
            i32::MIN as f32,
            2_147_483_648.0,
            f32::from_bits(1),
            -1.5,
            1.5,
            -2.5,
            2.5,
            f32::MIN,
            f32::MAX,
            12.75,
            -12.75,
        ];
        let v = f32x16::from_array_t(token, x);
        assert_eq!(
            v.bitcast_to_i32().to_array(),
            x.map(|lane| lane.to_bits().cast_signed())
        );
        assert_eq!(v.to_i32_saturating().to_array(), x.map(|lane| lane as i32));
        let minimum = v.minimum(f32x16::splat_t(token, 0.0)).to_array();
        assert_eq!(minimum[0].to_bits(), (-0.0_f32).to_bits());
        assert_eq!(minimum[1].to_bits(), 0);
        assert!(minimum[2].is_nan());
        assert_eq!(minimum[3], f32::NEG_INFINITY);
        assert_eq!(minimum[4].to_bits(), 0);
    }};
}

#[cfg(feature = "w512")]
#[test]
fn wide_scalar_backends_follow_lane_contracts() {
    check_wide_lanes!(archmage::ScalarToken);
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2", feature = "w512"))]
#[test]
fn wide_x86_backends_follow_lane_contracts() {
    use archmage::SimdToken;

    if let Some(token) = archmage::X64V3Token::summon() {
        check_wide_lanes!(token);
    }
    #[cfg(feature = "avx512")]
    {
        if let Some(token) = archmage::X64V4Token::summon() {
            check_wide_lanes!(token);
        }
        if let Some(token) = archmage::X64V4xToken::summon() {
            check_wide_lanes!(token);
        }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
#[test]
fn v3_vnni_delegated_byte_reductions_match_integer_arithmetic() {
    use archmage::SimdToken;
    use magetypes::simd::generic::u8x32;

    let Some(token) = archmage::X64V3VnniToken::summon() else {
        return;
    };
    let bytes = core::array::from_fn(|i| if i % 3 == 0 { 255 } else { i as u8 });
    let vector = u8x32::from_array_t(token, bytes);
    let mask = bytes
        .iter()
        .enumerate()
        .filter(|(_, lane)| **lane >= 128)
        .map(|(i, _)| 1_u32 << i)
        .sum();
    assert_eq!(vector.bitmask(), mask);
    assert_eq!(
        vector.reduce_add_u32(),
        bytes.into_iter().map(u32::from).sum()
    );
    assert_eq!(
        vector.sum_abs_diff(u8x32::splat_t(token, 128)),
        bytes.into_iter().map(|x| u32::from(x.abs_diff(128))).sum()
    );
}
