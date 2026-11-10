//! Intended contracts. These deliberately fail on the frozen PR147 head.
#![forbid(unsafe_code)]
use magetypes::{expf::*, logf::*};

#[test]
fn negative_zero_log_is_negative_infinity() {
    let value = try_log2_with::<{ LogfOptions::PORTABLE_MIDP_SPECIALS.bits() }>(-0.0).unwrap();
    assert_eq!(value.to_bits(), f32::NEG_INFINITY.to_bits());
}

#[test]
fn log1p_preserves_negative_zero() {
    assert_eq!(
        try_log1p_with::<0>(-0.0).unwrap().to_bits(),
        (-0.0f32).to_bits()
    );
}

#[test]
fn scalar_token_error_mask_has_only_one_lane() {
    let error = try_log1p_with_token::<0, _>(archmage::ScalarToken, -2.0).unwrap_err();
    assert_eq!(error.lane_mask, 1);
}

#[test]
fn runtime_adapter_accepts_caller_configuration() {
    let options = ExpfOptions::new().with_execution(ExpfExecution::Caller);
    assert_eq!(
        try_exp2_runtime(1.0, options),
        try_exp2_with::<
            {
                ExpfOptions::new()
                    .with_execution(ExpfExecution::Caller)
                    .bits()
            },
        >(1.0)
    );
    let options = LogfOptions::new().with_execution(LogfExecution::Caller);
    assert_eq!(
        try_log2_runtime(2.0, options),
        try_log2_with::<
            {
                LogfOptions::new()
                    .with_execution(LogfExecution::Caller)
                    .bits()
            },
        >(2.0)
    );
}

#[test]
fn assume_valid_full_retains_valid_nan_semantics() {
    const OPTIONS: u32 = ExpfOptions::PORTABLE_MIDP_FULL
        .with_validation(ExpfValidation::AssumeValid)
        .bits();
    // Every f32 is in Full's domain, including this NaN.
    assert!(try_exp2_with::<OPTIONS>(f32::NAN).unwrap().is_nan());
}

#[test]
fn assume_valid_specials_retains_valid_zero_semantics() {
    const OPTIONS: u32 = LogfOptions::PORTABLE_MIDP_SPECIALS
        .with_validation(LogfValidation::AssumeValid)
        .bits();
    assert_eq!(try_log2_with::<OPTIONS>(0.0).unwrap(), f32::NEG_INFINITY);
}

#[test]
fn valid_legacy_precision_does_not_panic() {
    const OPTIONS: u32 = ExpfOptions::new()
        .with_precision(ExpfPrecision::LegacyMidp)
        .bits();
    assert_eq!(try_exp2_with::<OPTIONS>(1.0), Ok(2.0));
}
