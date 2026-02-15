//! Generic SIMD types parameterized by backend token.
//!
//! These types are the strategy-pattern wrappers: `f32x8<T>` where `T`
//! determines the platform implementation. Write one generic function,
//! get monomorphized per backend at dispatch time.
//!
//! # Example
//!
//! ```ignore
//! use magetypes::simd::backends::F32x8Backend;
//! use magetypes::simd::generic::f32x8;
//!
//! fn dot<T: F32x8Backend>(token: T, a: &[f32; 8], b: &[f32; 8]) -> f32 {
//!     let va = f32x8::<T>::load(token, a);
//!     let vb = f32x8::<T>::load(token, b);
//!     (va * vb).reduce_add()
//! }
//! ```

#![allow(non_camel_case_types)]

mod f32x4_impl;
mod f32x8_impl;
mod f64x2_impl;
mod f64x4_impl;

pub use f32x4_impl::f32x4;
pub use f32x8_impl::f32x8;
pub use f64x2_impl::f64x2;
pub use f64x4_impl::f64x4;
