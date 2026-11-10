use super::*;

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

use magetypes::simd::generic::f32x4;
