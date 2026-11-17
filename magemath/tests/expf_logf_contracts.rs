//! Regression witnesses for the exp/log policy boundaries inherited from the stack.
#![allow(deprecated)] // Intentional coverage of retained Magetypes compatibility APIs.
#![forbid(unsafe_code)]
use magemath::{Caller, Unchecked, expf, expf::*, logf, logf::*};

#[test]
fn negative_zero_log_is_negative_infinity() {
    let value = try_log2_with::<logf::presets::MidpSpecials>(-0.0).unwrap();
    assert_eq!(value.to_bits(), f32::NEG_INFINITY.to_bits());
}

#[test]
fn log1p_preserves_negative_zero() {
    assert_eq!(
        try_log1p_with::<logf::presets::Midp>(-0.0)
            .unwrap()
            .to_bits(),
        (-0.0f32).to_bits()
    );
}

#[test]
fn scalar_token_error_mask_has_only_one_lane() {
    let error =
        try_log1p_with_token::<logf::presets::Midp, _>(archmage::ScalarToken, -2.0).unwrap_err();
    assert_eq!(error.lane_mask, 1);
}

#[test]
fn runtime_adapter_accepts_caller_configuration() {
    let options = ExpfOptions::new().with_execution(ExpfExecution::Caller);
    assert_eq!(
        try_exp2_runtime(1.0, options),
        try_exp2_with::<Caller<expf::presets::Midp>>(1.0)
    );
    let options = LogfOptions::new().with_execution(LogfExecution::Caller);
    assert_eq!(
        try_log2_runtime(2.0, options),
        try_log2_with::<Caller<logf::presets::Midp>>(2.0)
    );
}

#[test]
fn assume_valid_full_retains_valid_nan_semantics() {
    // Every f32 is in Full's domain, including this NaN.
    assert!(
        try_exp2_with::<Unchecked<expf::presets::MidpFull>>(f32::NAN)
            .unwrap()
            .is_nan()
    );
}

#[test]
fn assume_valid_specials_retains_valid_zero_semantics() {
    assert_eq!(
        try_log2_with::<Unchecked<logf::presets::MidpSpecials>>(0.0),
        Ok(f32::NEG_INFINITY)
    );
}
