//! Regression witnesses for the exp/log policy boundaries inherited from the stack.
#![forbid(unsafe_code)]
use magemath::{
    expf::{Expf, ExpfOptions},
    logf::{Logf, LogfOptions},
};

#[test]
fn negative_zero_log_is_negative_infinity() {
    const FULL: u128 = LogfOptions::new().full_domain().bits();
    let value = Logf::<FULL>::log2(-0.0).unwrap();
    assert_eq!(value.to_bits(), f32::NEG_INFINITY.to_bits());
}

#[test]
fn log1p_preserves_negative_zero() {
    const MIDP: u128 = LogfOptions::new().bits();
    assert_eq!(
        Logf::<MIDP>::log1p(-0.0).unwrap().to_bits(),
        (-0.0f32).to_bits()
    );
}

#[test]
fn scalar_token_error_mask_has_only_one_lane() {
    const MIDP: u128 = LogfOptions::new().bits();
    let error = Logf::<MIDP>::log1p_with_token(archmage::ScalarToken, -2.0).unwrap_err();
    assert_eq!(error.lane_mask, 1);
}

#[test]
fn runtime_adapter_accepts_caller_configuration() {
    const EXP_CALLER: u128 = ExpfOptions::new().caller().bits();
    const LOG_CALLER: u128 = LogfOptions::new().caller().bits();
    assert_eq!(
        ExpfOptions::new().caller().runtime().exp2(1.0),
        Expf::<EXP_CALLER>::exp2(1.0)
    );
    assert_eq!(
        LogfOptions::new().caller().runtime().log2(2.0),
        Logf::<LOG_CALLER>::log2(2.0)
    );
}

#[test]
fn assume_valid_full_retains_valid_nan_semantics() {
    const UNCHECKED_FULL: u128 = ExpfOptions::new().full_domain().unchecked().bits();
    assert!(Expf::<UNCHECKED_FULL>::exp2_infallible(f32::NAN).is_nan());
}

#[test]
fn assume_valid_specials_retains_valid_zero_semantics() {
    const UNCHECKED_FULL: u128 = LogfOptions::new().full_domain().unchecked().bits();
    assert_eq!(Logf::<UNCHECKED_FULL>::log2(0.0), Ok(f32::NEG_INFINITY));
}
