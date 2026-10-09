//! Calibration only: appended to the inventory driver, never to timed binaries.
#![allow(dead_code)]
use archmage::{ScalarToken, X64V3Token, arcane, rite};

#[rite(scalar)]
pub fn public_direct() {}

#[rite(scalar)]
pub(crate) fn crate_direct() {}

#[rite(scalar)]
fn private_direct() {}

#[arcane]
pub fn public_wrapped(token: X64V3Token) {
    let _ = token;
}

#[arcane]
pub fn public_scalar(token: ScalarToken) {
    let _ = token;
}

#[arcane(inline_always)]
fn explicit_always(token: ScalarToken) {
    let _ = token;
}
