//! Fixed-array operations: `[f32; 8]` in, `[f32; 8]` out.

crate::surface::array_surface!();

#[cfg(feature = "configurable")]
crate::width_surfaces::extra_arrays!();
