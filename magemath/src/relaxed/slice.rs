//! Bulk operations on `&mut [f32]`, dispatched through archmage tiers.
//!
//! Each op is an `#[attune]` family: the public name dispatches at runtime,
//! `_v3_t`-style proof wrappers take a token, and `_v3`-style direct variants are
//! for callers already in that feature context. Lanes after the last whole `f32x8`
//! go through the same kernel in a padded vector.
//!
//! Dispatch happens once per call: pass a whole buffer, not one element at a time.

crate::surface::slice_surface!();

#[cfg(feature = "configurable")]
crate::width_surfaces::extra_slices!();
