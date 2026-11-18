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
fn aliases_subnormals_and_runtime() {
    const MIDP: u128 = LogfOptions::new().bits();
    const FLUSH_POS: u128 = LogfOptions::new().flush_or_clamp_subnormals().bits();
    const FLUSH_UNIT: u128 = LogfOptions::new()
        .domain(LogfDomain::Unit)
        .flush_or_clamp_subnormals()
        .bits();
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
    assert_eq!(
        Logf::<FLUSH_POS>::log2(f32::from_bits(1)),
        Logf::<MIDP>::log2(f32::MIN_POSITIVE)
    );
    assert_eq!(
        Logf::<FLUSH_UNIT>::log2(f32::from_bits(1)),
        Ok(f32::NEG_INFINITY)
    );
}
