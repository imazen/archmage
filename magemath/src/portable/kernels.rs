//! Token-generic vector kernels: the single implementation behind
//! [`slice`](super::slice) and [`array`](super::array).
//!
//! Each function takes the token and an [`f32x8`] and is `#[inline(always)]`,
//! so a chain of them called inside one `#[attune]` body becomes one feature
//! region and the values stay in vector registers:
//!
//! ```
//! use archmage::prelude::*;
//! use magemath::portable::kernels;
//!
//! #[attune(_*_t, define(f32x8))]
//! fn gamma(token: Token, a: [f32; 8]) -> [f32; 8] {
//!     let x = f32x8::from_array_t(token, a);
//!     kernels::exp2_midp(token, kernels::log2_midp(token, x) * f32x8::splat_t(token, 2.4))
//!         .to_array()
//! }
//!
//! let out = attuned!(gamma([0.5; 8]));
//! assert_eq!(out[0], zenmath::scalar::pow_midp_f32(0.5, 2.4));
//! ```
//!
//! ## What makes these portable
//!
//! Polynomials and operand order are magetypes' generated transcendentals, so
//! on the tiers where `mul_add_relaxed` fuses (x86 v3/v4x, NEON) in-domain
//! results equal the `f32x8` methods bit for bit. Two things differ on purpose:
//!
//! - Every polynomial step is [`mul_add_fused`](f32x8::mul_add_fused), which
//!   rounds once on every backend. On x86 and NEON it is the same instruction
//!   as `mul_add_relaxed`. On the scalar tier and strict WASM it is software,
//!   and relaxed WASM uses the engine's madd once a cached probe shows it fuses.
//! - NaN input gives NaN output, without depending on anything that differs by
//!   ISA. `min`, `max` and `simd_ne` treat NaN differently per backend, so a NaN
//!   lane here reaches the result through arithmetic or an ordered comparison.
//!
//! Do not add a float negation whose sign of zero reaches the result, either:
//! `-(+0.0)` differs by ISA (`docs/site/content/magetypes/isa-quirks.md`).
//!
//! ## `_unchecked`
//!
//! The `*_unchecked` kernels skip edge handling. Use them in a hot loop whose
//! inputs you know are in the stated domain; `docs/DESIGN.md` has what that saves.
//! In the domain they return the same bits as the checked kernel on every tier.
//! Outside it the result is unspecified and differs by tier; it is never undefined
//! behavior.

crate::surface::kernels_surface!(crate::bodies::Fused);
