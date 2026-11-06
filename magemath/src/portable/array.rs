//! Fixed-array operations: `[f32; 8]` in, `[f32; 8]` out.
//!
//! Each op is an `#[attune]` family: the public name dispatches at runtime,
//! `_v3_t`-style proof wrappers take a token, and `_v3`-style direct variants
//! are for callers already in that feature context. Each call is one load, one
//! kernel and one store; to keep a pipeline in registers across several
//! operations, call [`kernels`] from one `#[attune]` body instead.

crate::surface::array_surface!();
