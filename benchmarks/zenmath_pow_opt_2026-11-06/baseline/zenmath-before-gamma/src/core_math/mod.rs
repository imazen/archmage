//! Pinned CORE-MATH ports. Nearest-even, gradual underflow; no errno/fenv emulation.
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod cbrtf;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod exp2f;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod expf;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod log2f;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod logf;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
pub use cbrtf::cbrtf;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
pub use exp2f::exp2f;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
pub use expf::expf;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
pub use log2f::log2f;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
pub use logf::logf;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
fn log_special(x: f32) -> f32 {
    if x == 0.0 {
        f32::NEG_INFINITY
    } else if x == f32::INFINITY {
        x
    } else if x.is_nan() {
        x + x
    } else {
        f32::NAN
    }
}
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod atanf;
mod fma;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod log10f;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod tanhf;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
pub use atanf::atanf;
pub use fma::fma;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
pub use log10f::log10f;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
pub use tanhf::tanhf;
#[cfg(any(
    feature = "correct-rounding",
    feature = "normal",
    feature = "configurable"
))]
mod powf;
#[cfg(any(
    feature = "correct-rounding",
    feature = "normal",
    feature = "configurable"
))]
pub use powf::powf;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod cosf;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod sinf;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod trig_reduce;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
pub use cosf::cosf;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
pub use sinf::sinf;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod expm1f;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod log1pf;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
pub use expm1f::expm1f;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
pub use log1pf::log1pf;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod tanf;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
pub use tanf::tanf;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod atan2f;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
pub use atan2f::atan2f;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod cbrt;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
pub use cbrt::cbrt;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod exp2;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
pub use exp2::exp2;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod exp;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
pub use exp::exp;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod log2;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
pub use log2::log2;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod log;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod log_accurate;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
pub use log::log;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod log10;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
pub use log10::log10;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod atan;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
pub use atan::atan;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod log1p;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
pub use log1p::log1p;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod sin;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
pub use sin::sin;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod cos;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
pub use cos::cos;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod atan2;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod atan2_tint;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
pub use atan2::atan2;

#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod tan;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
pub use tan::tan;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod pow;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod pow_d_tables;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod pow_int;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod pow_q_tables;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod pow_tables;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
pub use pow::pow;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod acosf;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod hypotf;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
pub use acosf::acosf;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
pub use hypotf::hypotf;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod atanh;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
pub use atanh::atanh;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod tanh;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
pub use tanh::tanh;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
mod asinh;
#[cfg(any(feature = "correct-rounding", feature = "normal"))]
pub use asinh::asinh;

// Shared full algorithm bodies for feature contexts, without changing scalar APIs.
#[cfg(all(
    feature = "archmage",
    any(
        feature = "correct-rounding",
        feature = "normal",
        feature = "configurable"
    )
))]
pub(crate) use powf::{Math as PowMath, powf_with_math};
#[cfg(all(
    feature = "archmage",
    any(feature = "correct-rounding", feature = "normal")
))]
pub(crate) use {expf::expf_in_context, log2f::log2f_in_context, logf::logf_in_context};
