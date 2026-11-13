use super::*;
#[allow(unused_imports)]
use archmage::{ScalarToken, SimdToken};
use magetypes::simd::generic::{f32x4, f32x8};

fn ulp_distance(got: f32, want: f64) -> f64 {
    let want_f32 = want as f32;
    if got.to_bits() == want_f32.to_bits() {
        return 0.0;
    }
    let next = f32::from_bits(want_f32.to_bits().saturating_add(1)) as f64;
    let prev = f32::from_bits(want_f32.to_bits().saturating_sub(1)) as f64;
    let step = (next - want_f32 as f64)
        .abs()
        .max((want_f32 as f64 - prev).abs());
    ((got as f64) - want).abs() / step
}

#[test]
fn cbrt_midp_and_lowp_accuracy_and_signs() {
    const MIDP: u32 = CbrtfOptions::PORTABLE_MIDP.bits();
    const LOWP: u32 = CbrtfOptions::PORTABLE_LOWP.bits();
    const FULL_MIDP: u32 = CbrtfOptions::PORTABLE_MIDP_FULL.bits();

    // Exactsigned zeros
    assert_eq!(
        try_cbrt_with::<MIDP>(0.0).unwrap().to_bits(),
        0.0f32.to_bits()
    );
    assert_eq!(
        try_cbrt_with::<MIDP>(-0.0).unwrap().to_bits(),
        (-0.0f32).to_bits()
    );
    assert_eq!(
        try_cbrt_with::<LOWP>(0.0).unwrap().to_bits(),
        0.0f32.to_bits()
    );
    assert_eq!(
        try_cbrt_with::<LOWP>(-0.0).unwrap().to_bits(),
        (-0.0f32).to_bits()
    );

    // Dense sweep across positive and negative normals up to f32::MAX / 3
    let lo = f32::MIN_POSITIVE.to_bits();
    let hi = (f32::MAX / 3.0).to_bits() - 1;
    let step = if cfg!(miri) { 2_000_003 } else { 16_381 };
    let mut max_midp_ulp = 0.0f64;
    let mut max_lowp_ulp = 0.0f64;

    let mut bits = lo;
    while bits <= hi {
        let x = f32::from_bits(bits);
        for s in [x, -x] {
            let want = (s as f64).cbrt();
            let got_midp = try_cbrt_with::<MIDP>(s).unwrap();
            let got_lowp = try_cbrt_with::<LOWP>(s).unwrap();
            max_midp_ulp = max_midp_ulp.max(ulp_distance(got_midp, want));
            max_lowp_ulp = max_lowp_ulp.max(ulp_distance(got_lowp, want));
        }
        bits = bits.saturating_add(step);
        if bits == u32::MAX {
            break;
        }
    }
    assert!(
        max_midp_ulp <= 3.2,
        "cbrtf MIDP exceeded 3.2 ULP: {max_midp_ulp}"
    );
    assert!(
        max_lowp_ulp <= 259.0,
        "cbrtf LOWP exceeded 259 ULP: {max_lowp_ulp}"
    );

    // Full domain: subnormals, > f32::MAX / 3 up to f32::MAX, ±inf, NaN
    for &sub in &[
        f32::from_bits(1),
        -f32::from_bits(1),
        1.0e-42_f32,
        -1.0e-40_f32,
        2.0e38_f32,
        -3.4e38_f32,
        f32::MAX,
        f32::MIN,
    ] {
        let got = try_cbrt_with::<FULL_MIDP>(sub).unwrap();
        let want = (sub as f64).cbrt();
        let u = ulp_distance(got, want);
        assert!(
            u <= 3.2,
            "cbrtf FULL MIDP({sub:e}) got {got:e} want {want:e} ({u} ULP)"
        );
    }
    assert_eq!(
        try_cbrt_with::<FULL_MIDP>(f32::INFINITY).unwrap(),
        f32::INFINITY
    );
    assert_eq!(
        try_cbrt_with::<FULL_MIDP>(f32::NEG_INFINITY).unwrap(),
        f32::NEG_INFINITY
    );
    assert!(try_cbrt_with::<FULL_MIDP>(f32::NAN).unwrap().is_nan());
}

#[test]
fn cbrt_domain_validation_and_cross_tier_bit_exactness() {
    const CHECKED: u32 = CbrtfOptions::PORTABLE_MIDP.bits();
    const CALLER_MIDP: u32 = CbrtfOptions::PORTABLE_MIDP
        .with_execution(CbrtfExecution::Caller)
        .bits();

    // Reject subnormal, >= f32::MAX / 3, inf, NaN in Bounded Checked mode
    let bad = [
        1.0,
        f32::from_bits(1),
        8.0,
        f32::MAX,
        -27.0,
        f32::INFINITY,
        64.0,
        f32::NAN,
    ];
    let err = try_cbrt_array_with::<CHECKED, 8>(bad).unwrap_err();
    assert_eq!(err.kind, CbrtfErrorKind::InvalidInput);
    assert_eq!(err.lane_mask, 0b1010_1010);

    // Unsupported width
    let w_err = try_cbrt_array_with::<CHECKED, 3>([1.0, 8.0, 27.0]).unwrap_err();
    assert_eq!(w_err.kind, CbrtfErrorKind::UnsupportedWidth);

    // Bit-exact agreement between ScalarToken and every summonable native tier
    let sample = [0.0, -0.0, 0.125, -27.0, 1.0e-20, -1.0e20, 3.25, 1000.0];
    let scalar_out = try_cbrt_f32x8_with::<CALLER_MIDP, _>(
        ScalarToken,
        f32x8::from_array_t(ScalarToken, sample),
    )
    .unwrap()
    .to_array();
    let runtime_out = try_cbrt_array_runtime(sample, CbrtfOptions::PORTABLE_MIDP).unwrap();
    assert_eq!(scalar_out.map(f32::to_bits), runtime_out.map(f32::to_bits));

    let sample4 = [sample[0], sample[1], sample[2], sample[3]];
    let scalar4 = try_cbrt_f32x4_with::<CALLER_MIDP, _>(
        ScalarToken,
        f32x4::from_array_t(ScalarToken, sample4),
    )
    .unwrap()
    .to_array();
    assert_eq!(
        scalar4.map(f32::to_bits),
        [
            scalar_out[0].to_bits(),
            scalar_out[1].to_bits(),
            scalar_out[2].to_bits(),
            scalar_out[3].to_bits(),
        ]
    );

    #[cfg(target_arch = "x86_64")]
    if let Some(t) = archmage::X64V3Token::summon() {
        let v3_out = try_cbrt_f32x8_with::<CALLER_MIDP, _>(t, f32x8::from_array_t(t, sample))
            .unwrap()
            .to_array();
        assert_eq!(v3_out.map(f32::to_bits), scalar_out.map(f32::to_bits));
        let v3_4 = try_cbrt_f32x4_with::<CALLER_MIDP, _>(t, f32x4::from_array_t(t, sample4))
            .unwrap()
            .to_array();
        assert_eq!(v3_4.map(f32::to_bits), scalar4.map(f32::to_bits));
    }

    #[cfg(target_arch = "aarch64")]
    if let Some(t) = archmage::NeonToken::summon() {
        let neon_out = try_cbrt_f32x8_with::<CALLER_MIDP, _>(t, f32x8::from_array_t(t, sample))
            .unwrap()
            .to_array();
        assert_eq!(neon_out.map(f32::to_bits), scalar_out.map(f32::to_bits));
        let neon_4 = try_cbrt_f32x4_with::<CALLER_MIDP, _>(t, f32x4::from_array_t(t, sample4))
            .unwrap()
            .to_array();
        assert_eq!(neon_4.map(f32::to_bits), scalar4.map(f32::to_bits));
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if let Some(t) = archmage::Wasm128Token::summon() {
        let wasm_out = try_cbrt_f32x8_with::<CALLER_MIDP, _>(t, f32x8::from_array_t(t, sample))
            .unwrap()
            .to_array();
        assert_eq!(wasm_out.map(f32::to_bits), scalar_out.map(f32::to_bits));
        let wasm_4 = try_cbrt_f32x4_with::<CALLER_MIDP, _>(t, f32x4::from_array_t(t, sample4))
            .unwrap()
            .to_array();
        assert_eq!(wasm_4.map(f32::to_bits), scalar4.map(f32::to_bits));
    }
}
