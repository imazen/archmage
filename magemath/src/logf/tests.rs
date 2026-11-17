use super::*;

#[test]
fn positive_rejects_invalid() {
    for x in [0.0, -0.0, -1.0, f32::NAN, f32::INFINITY, 1e-42] {
        assert_eq!(
            try_log2(x).unwrap_err().kind,
            LogfErrorKind::InvalidInput,
            "log2({x})"
        );
    }
}

#[test]
fn specials_domain() {
    const FULL: u128 = LogfOptions::new().full_domain().bits();
    assert_eq!(Logf::<FULL>::log2(0.0), Ok(f32::NEG_INFINITY));
    assert!(Logf::<FULL>::log2_infallible(-1.0).is_nan());
    assert!(Logf::<FULL>::log2_infallible(f32::NAN).is_nan());
    assert_eq!(Logf::<FULL>::log2(f32::INFINITY), Ok(f32::INFINITY));
}

#[test]
fn aliases_and_runtime() {
    const MIDP: u128 = LogfOptions::new().bits();
    let x = 4.0_f32;
    assert_eq!(try_ln(x), try_log2(x).map(|v| v * core::f32::consts::LN_2));
    assert_eq!(
        try_log10(x),
        try_log2(x).map(|v| v * core::f32::consts::LOG10_2)
    );
    assert_eq!(
        Logf::<MIDP>::log2_array::<3>([1.0, 2.0, 3.0])
            .unwrap_err()
            .kind,
        LogfErrorKind::UnsupportedWidth
    );
    assert_eq!(LogfOptions::new().runtime().log2(8.0), Ok(3.0));
}

#[test]
fn log1p_basics() {
    const C: u128 = LogfOptions::new().caller().bits();
    const S: u128 = LogfOptions::new().full_domain().caller().bits();
    assert_eq!(Logf::<C>::log1p(0.0), Ok(0.0));
    assert_eq!(
        Logf::<C>::log1p(1.0).unwrap() - core::f32::consts::LN_2,
        0.0
    );
    assert!((Logf::<C>::log1p(-0.5).unwrap() + core::f32::consts::LN_2).abs() < 1e-6);
    // correction recovers what u = 1+x rounds away for tiny x
    for x in [1e-8f32, -1e-8, 1e-20, 2.5e-5, -0.4142] {
        let got = Logf::<C>::log1p(x).unwrap();
        let r = (x as f64).ln_1p() as f32;
        assert_eq!(got, r, "log1p({x}): {got} vs {r}");
    }
    // full domain
    assert_eq!(Logf::<S>::log1p(-1.0), Ok(f32::NEG_INFINITY));
    assert!(Logf::<S>::log1p(-1.5).unwrap().is_nan());
    assert_eq!(Logf::<S>::log1p(f32::INFINITY), Ok(f32::INFINITY));
    // positive domain rejects x <= -1
    assert_eq!(
        Logf::<C>::log1p(-1.0).unwrap_err().kind,
        LogfErrorKind::InvalidInput
    );
    assert_eq!(
        Logf::<C>::log1p(-2.0).unwrap_err().kind,
        LogfErrorKind::InvalidInput
    );
    // array + runtime parity
    assert_eq!(
        Logf::<C>::log1p_array::<1>([0.5]),
        Ok([f32::from_bits(0x3ecf_991f)]) // MPFR256 RNE log1p(0.5); std is an approximate oracle under Miri.
    );
    assert_eq!(
        LogfOptions::new().runtime().log1p(1.0).unwrap(),
        core::f32::consts::LN_2
    );
}
