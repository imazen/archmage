use super::*;

#[test]
fn exp2_midp_matches_method() {
    for x in [-88.0, -1.5, -0.5, 0.0, 0.5, 1.0, 8.25, 64.0, 126.9, 127.9] {
        let via = try_exp2_with::<
            {
                ExpfOptions::DEFAULT
                    | ExpfOptions::BACKEND
                    | ExpfOptions::CALLER
                    | ExpfOptions::LEGACY
            },
        >(x)
        .unwrap();
        let scalar = f32x4::splat_t(archmage::ScalarToken, x);
        let via_method = scalar.exp2_midp().to_array()[0];
        assert_eq!(via, via_method, "exp2_midp({x})");
    }
}

#[test]
fn exp2_lowp_matches_method() {
    for x in [-88.0, -1.5, 0.0, 0.5, 1.0, 64.0, 126.9] {
        let via = try_exp2_with::<
            {
                ExpfOptions::LOWP | ExpfOptions::BACKEND | ExpfOptions::CALLER | ExpfOptions::LEGACY
            },
        >(x)
        .unwrap();
        let scalar = f32x4::splat_t(archmage::ScalarToken, x);
        let via_method = scalar.exp2_lowp().to_array()[0];
        assert_eq!(via, via_method, "exp2_lowp({x})");
    }
}

#[test]
fn bounded_rejects_nonfinite() {
    assert_eq!(
        try_exp2(f32::NAN).unwrap_err().kind,
        ExpfErrorKind::InvalidInput
    );
    assert_eq!(
        try_exp2(f32::INFINITY).unwrap_err().kind,
        ExpfErrorKind::InvalidInput
    );
    assert_eq!(
        try_exp2(f32::NEG_INFINITY).unwrap_err().kind,
        ExpfErrorKind::InvalidInput
    );
}

#[test]
fn full_domain_specials() {
    const FULL: u32 = ExpfOptions::FULL;
    assert!(try_exp2_with::<FULL>(f32::NAN).unwrap().is_nan());
    assert_eq!(try_exp2_with::<FULL>(f32::INFINITY), Ok(f32::INFINITY));
    assert_eq!(try_exp2_with::<FULL>(f32::NEG_INFINITY), Ok(0.0));
}

#[test]
fn edges_and_alias() {
    assert_eq!(try_exp2(-200.0), Ok(0.0));
    assert_eq!(try_exp2(200.0), Ok(f32::INFINITY));
    assert_eq!(
        try_exp(1.0),
        try_exp2_with::<{ ExpfOptions::DEFAULT }>(core::f32::consts::LOG2_E)
    );
}

#[test]
fn array_widths_and_runtime() {
    assert_eq!(
        try_exp2_array_with::<{ ExpfOptions::DEFAULT }, 3>([1.0, 2.0, 3.0])
            .unwrap_err()
            .kind,
        ExpfErrorKind::UnsupportedWidth
    );
    assert_eq!(try_exp2_runtime(1.0, ExpfOptions::new()), Ok(2.0));
    assert_eq!(
        try_exp2_runtime(1.0, ExpfOptions::PORTABLE_LOWP),
        try_exp2_with::<{ ExpfOptions::LOWP }>(1.0)
    );
}

use crate::simd::generic::f32x4;
