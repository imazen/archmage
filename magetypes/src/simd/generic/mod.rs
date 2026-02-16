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

mod block_ops_f32x4;
mod block_ops_f32x8;
mod block_ops_f64x2;
mod block_ops_f64x4;
mod block_ops_i32x4;
mod block_ops_i32x8;
mod f32x4_impl;
mod f32x8_impl;
mod f64x2_impl;
mod f64x4_impl;
mod i32x4_impl;
mod i32x8_impl;
mod i64x2_impl;
mod i64x4_impl;
mod transcendentals_f32x4;
mod transcendentals_f32x8;
mod u32x4_impl;
mod u32x8_impl;

pub use f32x4_impl::f32x4;
pub use f32x8_impl::f32x8;
pub use f64x2_impl::f64x2;
pub use f64x4_impl::f64x4;
pub use i32x4_impl::i32x4;
pub use i32x8_impl::i32x8;
pub use i64x2_impl::i64x2;
pub use i64x4_impl::i64x4;
pub use u32x4_impl::u32x4;
pub use u32x8_impl::u32x8;
