use super::*;

#[test]
fn log2_midp_matches_method() {
    for x in [0.5, 0.99, 1.0, 1.01, 2.0, 3.7, 1e10, 6e37] {
        let via = try_log2_with::<
            {
                LogfOptions::DEFAULT
                    | LogfOptions::BACKEND
                    | LogfOptions::CALLER
                    | LogfOptions::LEGACY
            },
        >(x)
        .unwrap();
        let scalar = f32x4::splat_t(archmage::ScalarToken, x);
        let via_method = scalar.log2_midp().to_array()[0];
        assert_eq!(via, via_method, "log2_midp({x})");
    }
}

#[test]
fn log2_lowp_matches_method() {
    for x in [0.5, 1.0, 2.0, 3.7, 1e10] {
        let via = try_log2_with::<
            {
                LogfOptions::LOWP | LogfOptions::BACKEND | LogfOptions::CALLER | LogfOptions::LEGACY
            },
        >(x)
        .unwrap();
        let scalar = f32x4::splat_t(archmage::ScalarToken, x);
        let via_method = scalar.log2_lowp().to_array()[0];
        assert_eq!(via, via_method, "log2_lowp({x})");
    }
}

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
    const SPECIALS: u32 = LogfOptions::SPECIALS;
    assert_eq!(try_log2_with::<SPECIALS>(0.0), Ok(f32::NEG_INFINITY));
    assert!(try_log2_with::<SPECIALS>(-1.0).unwrap().is_nan());
    assert!(try_log2_with::<SPECIALS>(f32::NAN).unwrap().is_nan());
    assert_eq!(try_log2_with::<SPECIALS>(f32::INFINITY), Ok(f32::INFINITY));
}

#[test]
fn aliases_and_runtime() {
    let x = 4.0_f32;
    assert_eq!(try_ln(x), try_log2(x).map(|v| v * core::f32::consts::LN_2));
    assert_eq!(
        try_log10(x),
        try_log2(x).map(|v| v * core::f32::consts::LOG10_2)
    );
    assert_eq!(
        try_log2_array_with::<{ LogfOptions::DEFAULT }, 3>([1.0, 2.0, 3.0])
            .unwrap_err()
            .kind,
        LogfErrorKind::UnsupportedWidth
    );
    assert_eq!(try_log2_runtime(8.0, LogfOptions::new()), Ok(3.0));
}

use crate::simd::generic::f32x4;

#[test]
fn log1p_basics() {
    const CALLER: u32 = LogfOptions::DEFAULT | LogfOptions::CALLER;
    assert_eq!(try_log1p_with::<CALLER>(0.0), Ok(0.0));
    assert_eq!(
        try_log1p_with::<CALLER>(1.0).unwrap() - core::f32::consts::LN_2,
        0.0
    );
    assert!((try_log1p_with::<CALLER>(-0.5).unwrap() + core::f32::consts::LN_2).abs() < 1e-6);
    // correction recovers what u = 1+x rounds away for tiny x
    for x in [1e-8f32, -1e-8, 1e-20, 2.5e-5, -0.4142] {
        let got = try_log1p_with::<CALLER>(x).unwrap();
        let r = (x as f64).ln_1p() as f32;
        assert_eq!(got, r, "log1p({x}): {got} vs {r}");
    }
    // specials domain
    const S: u32 = LogfOptions::DEFAULT | LogfOptions::SPECIALS;
    assert_eq!(try_log1p_with::<S>(-1.0), Ok(f32::NEG_INFINITY));
    assert!(try_log1p_with::<S>(-1.5).unwrap().is_nan());
    assert_eq!(try_log1p_with::<S>(f32::INFINITY), Ok(f32::INFINITY));
    // positive domain rejects x <= -1
    assert_eq!(
        try_log1p_with::<CALLER>(-1.0).unwrap_err().kind,
        LogfErrorKind::InvalidInput
    );
    assert_eq!(
        try_log1p_with::<CALLER>(-2.0).unwrap_err().kind,
        LogfErrorKind::InvalidInput
    );
    // array + runtime parity
    assert_eq!(
        try_log1p_array_with::<CALLER, 1>([0.5]),
        Ok([0.5f32.ln_1p()])
    );
    assert_eq!(
        try_log1p_runtime(1.0, LogfOptions::new()).unwrap(),
        core::f32::consts::LN_2
    );
}
