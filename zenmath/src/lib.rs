//! Portable scalar math with defined bits on every target.
//!
//! `zenmath` provides dependency-free, `no_std` scalar implementations for
//! callers that need portable math without SIMD types. Vector and fixed-array
//! math lives in `magemath` (this workspace), which uses this crate as the
//! oracle its kernels are tested against.
//!
//! ## What is here
//!
//! - **CORE-MATH ports** ([`scalar`]): 17 f32 and 16 f64 full-range
//!   transcendental functions, with their accurate fallback paths. They assume
//!   nearest-even rounding and gradual underflow; errno, exception flags,
//!   dynamic rounding modes and NaN payloads are outside the contract.
//! - **Correctly rounded primitives** ([`scalar`]): `sqrt`/`floor`/`ceil`/
//!   `round`/`roundeven` for f32 and f64 and `fmaf`/`fma` for f32/f64. FMA is checked against independent exact-bit cases; platform
//!   `std` is not its oracle. NaN payloads may differ.
//! - **f32 transcendentals** ([`scalar`]): `log2`/`exp2`/`ln`/`exp`/`log10`/`pow`/
//!   `cbrt` at `_lowp` and `_midp`, plus `cbrt_midp_precise`, `sigmoid_midp` and
//!   `silu_midp`. Each is the scalar form of the same-named kernel in
//!   `magemath::portable`, with the same bits on every tier.
//! - **Two-rounding prototype** ([`scalar::unfused`]): the same f32
//!   approximations with separate multiply/add, independent of software FMA
//!   and the `std` feature. Its final inclusion is pending a design decision.
//! - **Low-precision f64 transcendentals** ([`scalar`]): `log2`/`exp2`/`ln`/
//!   `exp`/`log10`/`pow` `_lowp_f64`, with clamped domains rather than a
//!   full-range libm contract. No vector form yet.
//! - **Planned, empty today**: special functions (`special` feature) and bit-exact
//!   libm-convention ports (`exact` feature). Both stay behind features so callers
//!   who don't need them don't compile them.
//!
//! ## Accuracy tiers
//!
//! Names carry the precision contract, matching magetypes:
//!
//! - `*_lowp`: perceptual and audio work. 0.56% relative error for `exp2`, about
//!   6e-6 absolute for `log2`, 3e-5 relative for `cbrt`.
//! - `*_midp`: general numerical work. At most 4.5 ULP logs, 3.2 ULP `cbrt`,
//!   1.9 ULP `exp2` below x = 127.5.
//! - `*_precise`: `cbrt_midp_precise` extends `cbrt_midp` over the whole f32 range.
//!
//! The measured figures per input band are in `docs/transcendentals.md`.
//!
//! ## Provenance
//!
//! CORE-MATH ports retain their upstream MIT notices and pin commit
//! `660067dfd3e751c8f07acc31c899ff410aa7206e`.
//! The scalar core is ported from `magetypes::nostd_math` (itself derived from
//! compiler-builtins/libm, from musl; MIT). The f32 `_midp` functions start from
//! `zensim::det_math`. `magetypes` does not depend on this crate and keeps its own
//! copies, because its `f32x8` methods have their own bit contract.

#![no_std]
#![forbid(unsafe_code)]

// fmaf uses std only when target features guarantee hardware fused arithmetic.
#[cfg(feature = "std")]
extern crate std;

pub mod scalar;

#[cfg(feature = "special")]
pub mod special;

#[cfg(feature = "exact")]
pub mod exact;

mod core_math;

// Same-workspace implementation bridge used by magemath to share precedence.
#[doc(hidden)]
#[path = "pow_policy.rs"]
pub mod __pow_policy;
pub use __pow_policy::{
    PowBehavior, PowConsistency, PowInfinity, PowNaN, PowOutput, PowPrecision, PowRange, PowSign,
    SubnormalInputs, SubnormalOutputs,
};
