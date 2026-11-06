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
