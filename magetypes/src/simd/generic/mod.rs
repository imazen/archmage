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

mod f32x8_impl;
pub use f32x8_impl::f32x8;
