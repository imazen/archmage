//! Bulk operations on `&mut [f32]`, dispatched through archmage tiers.

crate::surface::slice_surface!();

#[cfg(feature = "configurable")]
crate::width_surfaces::extra_slices!();
