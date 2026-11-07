//! Fixed-array operations: `[f32; 8]` in, `[f32; 8]` out.
//!
//! Each op is an `#[attune]` family: the public name dispatches at runtime,
//! `_v3_t`-style proof wrappers take a token, and `_v3`-style direct variants are
//! for callers already in that feature context.

crate::surface::array_surface!();

#[cfg(feature = "configurable")]
crate::width_surfaces::extra_arrays!();
