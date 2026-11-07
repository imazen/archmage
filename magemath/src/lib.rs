//! Vector and fixed-array math on `magetypes` types.
//!
//! `magemath` is the dispatched-kernel layer above `magetypes`. It adds what
//! the generated `f32x8` methods do not promise: results with the same bits on
//! every tier, the same bits as the `zenmath::scalar` mirrors, runtime dispatch,
//! and slice and fixed-array entry points.
//!
//! ## Modules
//!
//! - [`portable`]: ISA-consistent kernels. Same bits everywhere, including the
//!   scalar tier and strict WASM. The default.
//! - [`relaxed`]: the same operations and special-value handling, with polynomial
//!   rounding selected by the backend. Finite results may differ across tiers.
//! - [`unfused`]: draft experiment using two roundings on every tier; not a
//!   settled public API.
//! - [`native`]: reserved for per-ISA kernels that give up that guarantee.
//!
//! ## Dispatch model
//!
//! [`portable::kernels`] holds one token-generic body per op. [`portable::slice`]
//! and [`portable::array`] wrap each in one `#[attune]` family: direct per-tier
//! variants, `_t` proof wrappers, and a dispatcher that keeps the public name.
//! The `v4x` tier exists only with the `avx512` feature.
//!
//! Every generated function is `#[inline]`, so `magemath` itself codegens nothing
//! for them: a crate pays for the ops it calls, in its own build.
//!
//! Correctly rounded `exp`, `ln`, `log2`, and `pow` f32 array/slice APIs
//! use private packed binary64 kernels and complete `zenmath` fallbacks.
//! They are separate from the approximation kernels below.
//!
//! ## Accuracy tiers
//!
//! Names carry the precision contract, matching magetypes; the figures per op
//! and input band are in `docs/transcendentals.md`.
//!
//! - `*_lowp`: perceptual and audio work (0.56% relative error for `exp2`,
//!   about 6e-6 absolute for `log2`, 3e-5 relative for `cbrt`).
//! - `*_midp`: general numerical work (at most 4.5 ULP logs, 3.2 ULP `cbrt`,
//!   1.9 ULP `exp2` below x = 127.5).
//!
//! ## Status
//!
//! Draft, unpublished. `f32x8` only; `f32x4`, `f32x16` and f64 wait on a caller.

#![no_std]
#![forbid(unsafe_code)]
// Clippy counts every public `#[target_feature]` function as unsafe, including
// the safe direct variants `#[attune]` generates. Nothing here is `unsafe`, so
// there is no safety contract to document.
#![allow(clippy::missing_safety_doc)]

mod bodies;
#[cfg(any(
    feature = "correct-rounding",
    feature = "normal",
    feature = "configurable"
))]
mod correct;
mod surface;

pub mod native;
pub mod portable;
pub mod relaxed;
pub mod unfused;

/// Keeps the README's examples compiling and passing.
#[doc = include_str!("../README.md")]
#[cfg(doctest)]
struct ReadmeDoctests;

#[cfg(feature = "configurable")]
mod pow_policy;
#[cfg(feature = "configurable")]
pub use zenmath::{
    PowBehavior, PowConsistency, PowInfinity, PowNaN, PowOutput, PowPrecision, PowRange, PowSign,
    SubnormalInputs, SubnormalOutputs,
};

/// Keeps the exact draft power-guide fence compiling and executing.
#[doc=include_str!("../docs/POW_BEHAVIOR.md")]
#[cfg(doctest)]
struct PowBehaviorDoctests;

/// Standard-style vector APIs; currently share the precise CORE-MATH bodies.
/// Normal is an availability choice, not an accuracy downgrade.
#[cfg(feature = "normal")]
pub mod normal {
    pub use crate::correct::{array, slice};
}
