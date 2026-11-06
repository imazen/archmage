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
