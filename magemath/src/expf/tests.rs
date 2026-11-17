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
    const FULL: u128 = ExpfOptions::new().full_domain().bits();
    assert!(Expf::<FULL>::exp2_infallible(f32::NAN).is_nan());
    assert_eq!(Expf::<FULL>::exp2(f32::INFINITY), Ok(f32::INFINITY));
    assert_eq!(Expf::<FULL>::exp2(f32::NEG_INFINITY), Ok(0.0));
}

#[test]
fn edges_and_alias() {
    const MIDP: u128 = ExpfOptions::new().bits();
    assert_eq!(try_exp2(-200.0), Ok(0.0));
    assert_eq!(try_exp2(200.0), Ok(f32::INFINITY));
    assert_eq!(try_exp(1.0), Expf::<MIDP>::exp2(core::f32::consts::LOG2_E));
}

#[test]
fn array_widths_and_runtime() {
    const MIDP: u128 = ExpfOptions::new().bits();
    const LOWP: u128 = ExpfOptions::new().lowp().bits();
    assert_eq!(
        Expf::<MIDP>::exp2_array::<3>([1.0, 2.0, 3.0])
            .unwrap_err()
            .kind,
        ExpfErrorKind::UnsupportedWidth
    );
    assert_eq!(ExpfOptions::new().runtime().exp2(1.0), Ok(2.0));
    assert_eq!(
        ExpfOptions::new().lowp().runtime().exp2(1.0),
        Expf::<LOWP>::exp2(1.0)
    );
}
