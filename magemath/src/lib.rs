//! Configurable pow, exponential, logarithm, min/max and reciprocal families.
//!
//! Each family has an operation-specific const builder ending in `.bits()`.
//! Profiles name precision, range, validation and portability; these are not
//! correctly rounded replacements for every libm function. See each module's
//! accepted domain and accuracy limits.
//!
//! # Choosing options
//!
//! Use the family's preset as a starting point, change individual axes with
//! `with_*`, then pass `.bits()` as the integer const argument. These are
//! compile-time configurations, allowing configuration-dependent code to fold
//! away. Input validation and feature selection can still branch.
//!
//! | Family | Precision choices | Domain choices | Result policy |
//! |---|---|---|---|
//! | [`powf::PowfOptions`] | LOWP / MIDP / HIGHP | PositiveNormal / NonNegativeNormal / Gamma / PositiveGamma / Full | BitExact / BackendDependent |
//! | [`expf::ExpfOptions`] | LOWP / MIDP | Bounded / Full | BitExact / BackendDependent |
//! | [`logf::LogfOptions`] | LOWP / MIDP | Positive / Specials | BitExact / BackendDependent |
//! | [`recip::RecipOptions`] (recip and rsqrt) | LOWP / MIDP / HIGHP | Normal / Full | BitExact / BackendDependent |
//! | [`minmax::MinMaxOptions`] | No precision grade | Normal / Finite / Full | Independent NaN and signed-zero policies |
//!
//! All families also expose Checked / AssumeValid validation and Automatic /
//! Caller execution. The domains have operation-specific meanings: for example,
//! reciprocal Normal is a bounded normal range, and logarithmic Specials does
//! not promise accurate subnormal results. See the domain enums for exact limits.
//! Full power requires HIGHP; backend-dependent HIGHP power also requires `std`.
//! Full reciprocal/rsqrt requires HIGHP. Exp/log have no HIGHP option.
//!
//! **BitExact describes result agreement, not software execution.** Portable
//! fused graphs use hardware FMA when available and an exact fallback otherwise.
//! BackendDependent permits ISA-dependent results; it does not promise greater
//! speed. Likewise, LOWP is not a promise to beat MIDP or HIGHP on every ISA.
//! NaN encodings and floating exception flags are outside the bit agreement.
//!
//! The reciprocal [`preset and combination tables`](crate::recip) and min/max
//! [`policy tables`](crate::minmax) describe valid and equivalent configurations.
//! Bounded-f64 Newton evaluation and canonical RN14/RN8 are benchmark-only
//! experiments, not additional public option values or presets.
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
//! or primitive arithmetic storage, and Magetypes does not depend on Magemath.
//! Min/max supports f32/f64; reciprocal and rsqrt currently support f32.
//! Reciprocal native-estimate tails must use the matching vector-width lane API.
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

mod primitive;
pub use primitive::{PrimitiveError, PrimitiveErrorKind};
/// Configurable min/max policies for f32 and f64.
pub mod minmax;
#[cfg(target_arch = "wasm32")]
mod primitive_wasm;
/// Configurable reciprocal and reciprocal square root for f32.
pub mod recip;

#[cfg(target_arch = "aarch64")]
mod primitive_neon;
