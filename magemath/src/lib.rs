//! Configurable f32 pow, exponential and logarithm approximations.
//!
//! Each family has an operation-specific const builder ending in `.bits()`.
//! Profiles name precision, range, validation and portability; these are not
//! correctly rounded replacements for every libm function. See each module's
//! accepted domain and accuracy limits.
//!
//! Scalar/array entries can select a backend once per operation. Typed vector
//! entries take the caller's proof explicitly; reuse that proof for scalar tails.
//! ```
//! use archmage::ScalarToken;
//! use magetypes::simd::generic::f32x4;
//! use magemath::powf::{PowfExecution, PowfOptions, try_powf_f32x4_with};
//! const FLAGS: u32 = PowfOptions::PORTABLE_GAMMA_MIDP
//!     .with_execution(PowfExecution::Caller).bits();
//! let input = f32x4::from_array_t(ScalarToken, [0.0, 0.25, 0.5, 1.0]);
//! let output = try_powf_f32x4_with::<FLAGS, _>(ScalarToken, input, 2.2).unwrap();
//! assert_eq!(output.to_array()[3], 1.0);
//! ```
//!
//! Magemath depends on Magetypes and Archmage. It does not replace vector storage
//! or primitive arithmetic, and Magetypes does not depend on Magemath.
#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs)]
#[cfg(test)]
extern crate alloc;
#[cfg(any(feature = "std", test))]
extern crate std;
pub mod expf;
pub mod logf;
pub mod powf;
mod soft_fma;
#[cfg(target_arch = "wasm32")]
#[allow(dead_code)] // The copy also contains vector-primitive helpers not used here.
mod wasm_fma;
