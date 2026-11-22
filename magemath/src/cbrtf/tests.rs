use super::*;
use crate::SimdMath;
#[allow(unused_imports)]
use archmage::{ScalarToken, SimdToken};
use magetypes::simd::generic::{f32x4, f32x8};

fn ulp_distance(got: f32, want: f64) -> f64 {
    let exp = (want.to_bits() >> 52) & 0x7ff;
    let ulp = f64::from_bits((exp - 23) << 52);
    (f64::from(got) - want).abs() / ulp
}

#[test]
fn cbrt_midp_and_lowp_accuracy_and_signs() {
    const MIDP: u128 = CbrtfOptions::new().bits();
    const LOWP: u128 = CbrtfOptions::new().lowp().bits();
    const FULL_MIDP: u128 = CbrtfOptions::new().full_domain().bits();
    const FULL_LOWP: u128 = CbrtfOptions::new().full_domain().lowp().bits();

    // Exact signed zeros
    assert_eq!(
        Cbrtf::<MIDP>::scalar(0.0).unwrap().to_bits(),
        0.0f32.to_bits()
    );
    assert_eq!(
        Cbrtf::<MIDP>::scalar(-0.0).unwrap().to_bits(),
        (-0.0f32).to_bits()
    );
    assert_eq!(
        Cbrtf::<LOWP>::scalar(0.0).unwrap().to_bits(),
        0.0f32.to_bits()
    );
    assert_eq!(
        Cbrtf::<LOWP>::scalar(-0.0).unwrap().to_bits(),
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
            let got_midp = Cbrtf::<MIDP>::scalar(s).unwrap();
            let got_lowp = Cbrtf::<LOWP>::scalar(s).unwrap();
            max_midp_ulp = max_midp_ulp.max(ulp_distance(got_midp, want));
            max_lowp_ulp = max_lowp_ulp.max(ulp_distance(got_lowp, want));
        }
        bits = bits.saturating_add(step);
        if bits == u32::MAX {
            break;
        }
    }
    assert!(
        max_midp_ulp <= 1.5,
        "cbrtf MIDP exceeded 1.5 ULP: {max_midp_ulp}"
    );
    assert!(
        max_lowp_ulp <= 181.0,
        "cbrtf LOWP exceeded 181 ULP: {max_lowp_ulp}"
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
        let got = Cbrtf::<FULL_MIDP>::scalar_infallible(sub);
        let want = (sub as f64).cbrt();
        let u = ulp_distance(got, want);
        assert!(
            u <= 1.5,
            "cbrtf FULL MIDP({sub:e}) got {got:e} want {want:e} ({u} ULP)"
        );
    }
    // The lowest representable power of eight is subnormal.
    for exp in -49..=42 {
        let root = f32::from_bits(((127 + exp) as u32) << 23);
        let input = if exp < -42 {
            f32::from_bits(1 << (149 + 3 * exp))
        } else {
            f32::from_bits(((127 + 3 * exp) as u32) << 23)
        };
        for (x, y) in [(input, root), (-input, -root)] {
            assert_eq!(
                Cbrtf::<FULL_MIDP>::scalar_infallible(x).to_bits(),
                y.to_bits()
            );
            assert_eq!(
                Cbrtf::<FULL_LOWP>::scalar_infallible(x).to_bits(),
                y.to_bits()
            );
        }
    }
    assert_eq!(
        Cbrtf::<FULL_MIDP>::scalar_infallible(f32::INFINITY),
        f32::INFINITY
    );
    assert_eq!(
        Cbrtf::<FULL_MIDP>::scalar_infallible(f32::NEG_INFINITY),
        f32::NEG_INFINITY
    );
    assert!(Cbrtf::<FULL_MIDP>::scalar_infallible(f32::NAN).is_nan());
}

#[test]
fn cbrt_domain_validation_and_cross_tier_bit_exactness() {
    const CHECKED: u128 = CbrtfOptions::new().reject_subnormals().bits();
    const CALLER_MIDP: u128 = CbrtfOptions::new().caller().bits();

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
    let err = Cbrtf::<CHECKED>::array::<8>(bad).unwrap_err();
    assert_eq!(err.kind, CbrtfErrorKind::InvalidInput);
    assert_eq!(err.lane_mask, 0b1010_1010);

    // Unsupported width
    let w_err = Cbrtf::<CHECKED>::array::<3>([1.0, 8.0, 27.0]).unwrap_err();
    assert_eq!(w_err.kind, CbrtfErrorKind::UnsupportedWidth);

    // Bit-exact agreement between ScalarToken and every summonable native tier
    let sample = [0.0, -0.0, 0.125, -27.0, 1.0e-20, -1.0e20, 3.25, 1000.0];
    let scalar_out = f32x8::from_array_t(ScalarToken, sample)
        .try_cbrt_with::<CALLER_MIDP>()
        .unwrap()
        .to_array();
    let runtime_out = CbrtfOptions::new().runtime().array(sample).unwrap();
    assert_eq!(scalar_out.map(f32::to_bits), runtime_out.map(f32::to_bits));

    let sample4 = [sample[0], sample[1], sample[2], sample[3]];
    let scalar4 = f32x4::from_array_t(ScalarToken, sample4)
        .try_cbrt_with::<CALLER_MIDP>()
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

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    if let Some(t) = archmage::X64V3Token::summon() {
        let v3_out = f32x8::from_array_t(t, sample)
            .try_cbrt_with::<CALLER_MIDP>()
            .unwrap()
            .to_array();
        assert_eq!(v3_out.map(f32::to_bits), scalar_out.map(f32::to_bits));
        let v3_4 = f32x4::from_array_t(t, sample4)
            .try_cbrt_with::<CALLER_MIDP>()
            .unwrap()
            .to_array();
        assert_eq!(v3_4.map(f32::to_bits), scalar4.map(f32::to_bits));
    }

    #[cfg(target_arch = "aarch64")]
    if let Some(t) = archmage::NeonToken::summon() {
        let neon_out = f32x8::from_array_t(t, sample)
            .try_cbrt_with::<CALLER_MIDP>()
            .unwrap()
            .to_array();
        assert_eq!(neon_out.map(f32::to_bits), scalar_out.map(f32::to_bits));
        let neon_4 = f32x4::from_array_t(t, sample4)
            .try_cbrt_with::<CALLER_MIDP>()
            .unwrap()
            .to_array();
        assert_eq!(neon_4.map(f32::to_bits), scalar4.map(f32::to_bits));
    }

    #[cfg(target_arch = "wasm32")]
    if let Some(t) = archmage::Wasm128Token::summon() {
        let wasm_out = f32x8::from_array_t(t, sample)
            .try_cbrt_with::<CALLER_MIDP>()
            .unwrap()
            .to_array();
        assert_eq!(wasm_out.map(f32::to_bits), scalar_out.map(f32::to_bits));
        let wasm_4 = f32x4::from_array_t(t, sample4)
            .try_cbrt_with::<CALLER_MIDP>()
            .unwrap()
            .to_array();
        assert_eq!(wasm_4.map(f32::to_bits), scalar4.map(f32::to_bits));
    }
}
