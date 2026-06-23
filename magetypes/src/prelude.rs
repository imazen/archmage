//! Common magetypes imports: the generic SIMD vector types plus the
//! [`SimdToken`](archmage::SimdToken) trait.
//!
//! Pair this with `use archmage::prelude::*;` for the tokens and macros, and
//! name the token explicitly so the lane width and tier stay visible:
//!
//! ```no_run
//! use archmage::prelude::*;
//! use magetypes::prelude::*;
//!
//! # #[cfg(target_arch = "x86_64")]
//! # fn example() {
//! if let Some(token) = X64V3Token::summon() {
//!     let a = f32x8::<X64V3Token>::splat(token, 1.0);
//!     let b = f32x8::<X64V3Token>::splat(token, 2.0);
//!     let _c = a + b;
//! }
//! # }
//! ```
//!
//! The platform-"best" aliases (`F32Vec`, `RecommendedToken`, `LANES`, …) that
//! this prelude used to provide have been removed: they hid the lane width and
//! token, so the same source produced different lane counts per architecture.
//! Use the explicit `generic::fNxM<Token>` types instead. For single-target
//! code, the token-bound `simd::v3` / `simd::neon` / `simd::wasm128` aliases are
//! also available.

pub use crate::simd::generic::*;
pub use archmage::SimdToken;
