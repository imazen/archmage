//! PROTOTYPE, pending a decision: identical bits on every tier by doing without
//! FMA. Each polynomial step is an IEEE multiply and then an IEEE add, which needs
//! no software fallback on any tier. Results equal `zenmath::scalar::unfused`.
//!
//! Same specials as [`portable`](crate::portable).

pub mod array;
pub mod kernels;
pub mod slice;
