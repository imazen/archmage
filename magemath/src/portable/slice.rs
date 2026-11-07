//! Bulk operations on `&mut [f32]`, dispatched through archmage tiers.
//!
//! Each op is an `#[attune]` family: the public name dispatches at runtime,
//! `_v3_t`-style proof wrappers take a token, and `_v3`-style direct variants
//! are for callers already in that feature context. Lanes left over after the
//! last whole `f32x8` go through the same kernel in a padded vector, so every
//! element of a slice, whatever its position, takes one code path.
//!
//! Dispatch happens once per call: pass a whole buffer, not one element at a
//! time. To chain several operations per element, write one `#[attune]` body
//! over [`kernels`] instead of calling these in sequence.

crate::surface::slice_surface!();

#[cfg(feature = "correct-rounding")]
pub use crate::correct::slice::*;

#[cfg(feature = "configurable")]
/// Packed policy power, dispatched once per buffer, with per-element exponents.
/// Lengths must match. Tail padding uses valid (1,0) inputs, so optional debug
/// preconditions inspect real operands and harmless valid filler only.
#[archmage::attune(_*, _*_t, dispatch(inline(hint)), +v4x(cfg(avx512)))]
pub fn pow_f32_with<const P: u64>(data: &mut [f32], exponents: &[f32]) {
    assert_eq!(
        data.len(),
        exponents.len(),
        "power slice lengths must match"
    );
    const {
        zenmath::__pow_policy::validate(P);
    }
    let token = Token::from_context();
    let (chunks, tail) = data.as_chunks_mut::<8>();
    let (ys, yt) = exponents.as_chunks::<8>();
    for (x, y) in chunks.iter_mut().zip(ys) {
        *x = crate::pow_policy::array::<_, P>(token, *x, *y);
    }
    if !tail.is_empty() {
        let mut x = [1.0; 8];
        let mut y = [0.0; 8];
        x[..tail.len()].copy_from_slice(tail);
        y[..yt.len()].copy_from_slice(yt);
        let out = crate::pow_policy::array::<_, P>(token, x, y);
        tail.copy_from_slice(&out[..tail.len()]);
    }
}

#[cfg(feature = "configurable")]
crate::width_surfaces::extra_slices!();
