//! Normal/libm-style availability, with an explicit namespace.
//! This initial implementation uses the same precise CORE-MATH routines as
//! `correct-rounding`; it does not promise a cheaper approximation. Enabling
//! another Cargo feature never changes these results or existing scalar bits.
pub use crate::reference::{
    acosf, asinh, atan, atan2, atan2f, atanf, atanh, cbrt, cbrtf, cos, cosf, exp, exp2, exp2f,
    expf, expm1f, hypotf, log, log1p, log1pf, log2, log2f, log10, log10f, logf, pow, sin, sincosf,
    sinf, tan, tanf, tanh, tanhf,
};
/// Standard-style power, currently using the precise scalar implementation.
#[inline(always)]
pub fn powf(x: f32, y: f32) -> f32 {
    crate::scalar::selected_powf(x, y)
}
/// Proof/context array and slice entries sharing the precise implementation.
#[cfg(feature = "archmage")]
pub mod simd {
    pub use crate::simd::__array as array;
    pub use crate::simd::__scalar_transcendentals as scalar;
    pub use crate::simd::__slice as slice;
}
