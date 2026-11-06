//! Pinned CORE-MATH ports. Nearest-even, gradual underflow; no errno/fenv emulation.
mod cbrtf;
mod exp2f;
mod expf;
mod log2f;
mod logf;
pub use cbrtf::cbrtf;
pub use exp2f::exp2f;
pub use expf::expf;
pub use log2f::log2f;
pub use logf::logf;
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
mod atanf;
mod fma;
mod log10f;
mod tanhf;
pub use atanf::atanf;
pub use fma::fma;
pub use log10f::log10f;
pub use tanhf::tanhf;
mod powf;
pub use powf::powf;
mod cosf;
mod sinf;
mod trig_reduce;
pub use cosf::cosf;
pub use sinf::sinf;
mod expm1f;
mod log1pf;
pub use expm1f::expm1f;
pub use log1pf::log1pf;
mod tanf;
pub use tanf::tanf;
mod atan2f;
pub use atan2f::atan2f;
mod cbrt;
pub use cbrt::cbrt;
mod exp2;
pub use exp2::exp2;
mod exp;
pub use exp::exp;
mod log2;
pub use log2::log2;
mod log;
mod log_accurate;
pub use log::log;
mod log10;
pub use log10::log10;
mod atan;
pub use atan::atan;
mod log1p;
pub use log1p::log1p;
mod sin;
pub use sin::sin;
mod cos;
pub use cos::cos;
mod atan2;
mod atan2_tint;
pub use atan2::atan2;

mod tan;
pub use tan::tan;
