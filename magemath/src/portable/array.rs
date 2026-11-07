//! Fixed-array operations: `[f32; 8]` in, `[f32; 8]` out.
//!
//! Each op is an `#[attune]` family: the public name dispatches at runtime,
//! `_v3_t`-style proof wrappers take a token, and `_v3`-style direct variants
//! are for callers already in that feature context. Each call is one load, one
//! kernel and one store; to keep a pipeline in registers across several
//! operations, call [`kernels`] from one `#[attune]` body instead.

crate::surface::array_surface!();

#[cfg(feature = "correct-rounding")]
pub use crate::correct::array::*;

#[cfg(feature = "configurable")]
/// Power with a shared [`zenmath::PowBehavior`] const policy, packed per lane.
/// Default retains the existing packed CORE path. Transformed correctly rounded
/// policies use the same packed kernel; approximation policies use packed
/// log2×per-lane exponent→exp2 and complete scalar rescue where contracted.
/// CPU tier and power policy are independent.
/// ```compile_fail
/// let _=magemath::portable::array::pow_f32x8_with::<{1u64<<63}>([2.0;8],[3.0;8]);
/// ```
#[archmage::attune(_*, _*_t, dispatch(inline(hint)), +v4x(cfg(avx512)))]
pub fn pow_f32x8_with<const P: u64>(a: [f32; 8], b: [f32; 8]) -> [f32; 8] {
    let token = Token::from_context();
    crate::pow_policy::array::<_, P>(token, a, b)
}
