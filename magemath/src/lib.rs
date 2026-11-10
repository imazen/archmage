//! Configurable f32 pow, exponential and logarithm approximations.
//!
//! Each family has an operation-specific const builder ending in `.bits()`.
//! Profiles name precision, range, validation and portability; these are not
//! correctly rounded replacements for every libm function. See each module's
//! accepted domain and accuracy limits.
//!
//! Scalar/array entries can select a backend once per operation. Typed vector
//! entries take the caller's proof explicitly; reuse that proof for scalar tails.
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
