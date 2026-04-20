//! Adversarial bypass-closure tests (compile-fail doctests).
//!
//! This module does not export anything; its sole purpose is to host
//! [`BypassAdversarialProofs`] — a marker struct whose documentation
//! contains one `compile_fail` doctest per trait-method category in the
//! backend surface. Each doctest attempts to call a representative
//! backend trait method via UFCS without passing a token value and is
//! expected to fail to compile.
//!
//! If any of these doctests stop failing (i.e. the UFCS-tokenless form
//! starts compiling), the soundness contract of archmage/magetypes has
//! been re-broken. CI treats a green `compile_fail` doctest as a failure.
//!
//! The runtime-assert counterparts (sanctioned forms — pass a real
//! token, observe expected values) live in `magetypes/tests/bypass_closed.rs`.
//!
//! # Categories exercised
//!
//! One compile_fail doctest per category:
//!
//! 1. Construction — `splat` (F32x8Backend)
//! 2. Construction — `zero` (F32x8Backend)
//! 3. Construction — `load` (I32x4Backend)
//! 4. Construction — `from_array` (U8x16Backend)
//! 5. Memory — `store` (F32x8Backend)
//! 6. Memory — `to_array` (F64x2Backend)
//! 7. Arithmetic — `add` (I32x4Backend)
//! 8. Arithmetic — `neg` (F32x4Backend)
//! 9. Math — `min` (F32x8Backend)
//! 10. Math — `sqrt` (F32x4Backend)
//! 11. Math — `mul_add` (F32x8Backend)
//! 12. Comparison — `simd_lt` (I32x4Backend)
//! 13. Comparison — `blend` (F32x8Backend)
//! 14. Reduction — `reduce_add` (F32x8Backend)
//! 15. Bitwise — `bitand` (U32x4Backend)
//! 16. Bitwise — `not` (U32x4Backend)
//! 17. Shift — `shl_const` (I32x4Backend)
//! 18. Boolean — `all_true` (I32x4Backend)
//! 19. Boolean — `bitmask` (U32x4Backend)
//! 20. Convert/bitcast — `bitcast_f32_to_i32` (F32x8Convert)
//!
//! Every doctest follows the same pattern:
//!
//! ```text
//! use archmage::<Token>;
//! use magetypes::simd::backends::<Trait>;
//! let _ = <<Token> as <Trait>>::<method>(<args...>);  // <- missing `self`
//! ```
//!
//! The corresponding sanctioned form (pass a real token) is exercised in
//! runtime tests and proves the method still works when given a token.
//!
//! ## Construction
//!
//! ```compile_fail
//! use archmage::ScalarToken;
//! use magetypes::simd::backends::F32x8Backend;
//! // Missing self: splat expects (self, f32).
//! let _ = <ScalarToken as F32x8Backend>::splat(7.0f32);
//! ```
//!
//! ```compile_fail
//! use archmage::ScalarToken;
//! use magetypes::simd::backends::F32x8Backend;
//! // Missing self: zero expects (self).
//! let _ = <ScalarToken as F32x8Backend>::zero();
//! ```
//!
//! ```compile_fail
//! use archmage::ScalarToken;
//! use magetypes::simd::backends::I32x4Backend;
//! // Missing self: load expects (self, &[i32; 4]).
//! let _ = <ScalarToken as I32x4Backend>::load(&[0i32, 1, 2, 3]);
//! ```
//!
//! ```compile_fail
//! use archmage::ScalarToken;
//! use magetypes::simd::backends::U8x16Backend;
//! // Missing self: from_array expects (self, [u8; 16]).
//! let _ = <ScalarToken as U8x16Backend>::from_array([0u8; 16]);
//! ```
//!
//! ## Memory
//!
//! ```compile_fail
//! use archmage::{ScalarToken, SimdToken};
//! use magetypes::simd::backends::F32x8Backend;
//! let t = ScalarToken::summon().unwrap();
//! let r = <ScalarToken as F32x8Backend>::zero(t);
//! let mut out = [0.0f32; 8];
//! // Missing self: store expects (self, Repr, &mut [..]).
//! let _ = <ScalarToken as F32x8Backend>::store(r, &mut out);
//! ```
//!
//! ```compile_fail
//! use archmage::{ScalarToken, SimdToken};
//! use magetypes::simd::backends::F64x2Backend;
//! let t = ScalarToken::summon().unwrap();
//! let r = <ScalarToken as F64x2Backend>::zero(t);
//! // Missing self: to_array expects (self, Repr).
//! let _ = <ScalarToken as F64x2Backend>::to_array(r);
//! ```
//!
//! ## Arithmetic
//!
//! ```compile_fail
//! use archmage::{ScalarToken, SimdToken};
//! use magetypes::simd::backends::I32x4Backend;
//! let t = ScalarToken::summon().unwrap();
//! let a = <ScalarToken as I32x4Backend>::zero(t);
//! let b = <ScalarToken as I32x4Backend>::zero(t);
//! // Missing self: add expects (self, a, b).
//! let _ = <ScalarToken as I32x4Backend>::add(a, b);
//! ```
//!
//! ```compile_fail
//! use archmage::{ScalarToken, SimdToken};
//! use magetypes::simd::backends::F32x4Backend;
//! let t = ScalarToken::summon().unwrap();
//! let a = <ScalarToken as F32x4Backend>::zero(t);
//! // Missing self: neg expects (self, a).
//! let _ = <ScalarToken as F32x4Backend>::neg(a);
//! ```
//!
//! ## Math
//!
//! ```compile_fail
//! use archmage::{ScalarToken, SimdToken};
//! use magetypes::simd::backends::F32x8Backend;
//! let t = ScalarToken::summon().unwrap();
//! let a = <ScalarToken as F32x8Backend>::zero(t);
//! let b = <ScalarToken as F32x8Backend>::zero(t);
//! // Missing self: min expects (self, a, b).
//! let _ = <ScalarToken as F32x8Backend>::min(a, b);
//! ```
//!
//! ```compile_fail
//! use archmage::{ScalarToken, SimdToken};
//! use magetypes::simd::backends::F32x4Backend;
//! let t = ScalarToken::summon().unwrap();
//! let a = <ScalarToken as F32x4Backend>::zero(t);
//! // Missing self: sqrt expects (self, a).
//! let _ = <ScalarToken as F32x4Backend>::sqrt(a);
//! ```
//!
//! ```compile_fail
//! use archmage::{ScalarToken, SimdToken};
//! use magetypes::simd::backends::F32x8Backend;
//! let t = ScalarToken::summon().unwrap();
//! let a = <ScalarToken as F32x8Backend>::zero(t);
//! let b = <ScalarToken as F32x8Backend>::zero(t);
//! let c = <ScalarToken as F32x8Backend>::zero(t);
//! // Missing self: mul_add expects (self, a, b, c).
//! let _ = <ScalarToken as F32x8Backend>::mul_add(a, b, c);
//! ```
//!
//! ## Comparison
//!
//! ```compile_fail
//! use archmage::{ScalarToken, SimdToken};
//! use magetypes::simd::backends::I32x4Backend;
//! let t = ScalarToken::summon().unwrap();
//! let a = <ScalarToken as I32x4Backend>::zero(t);
//! let b = <ScalarToken as I32x4Backend>::zero(t);
//! // Missing self: simd_lt expects (self, a, b).
//! let _ = <ScalarToken as I32x4Backend>::simd_lt(a, b);
//! ```
//!
//! ```compile_fail
//! use archmage::{ScalarToken, SimdToken};
//! use magetypes::simd::backends::F32x8Backend;
//! let t = ScalarToken::summon().unwrap();
//! let mask = <ScalarToken as F32x8Backend>::zero(t);
//! let tt = <ScalarToken as F32x8Backend>::zero(t);
//! let ff = <ScalarToken as F32x8Backend>::zero(t);
//! // Missing self: blend expects (self, mask, if_true, if_false).
//! let _ = <ScalarToken as F32x8Backend>::blend(mask, tt, ff);
//! ```
//!
//! ## Reduction
//!
//! ```compile_fail
//! use archmage::{ScalarToken, SimdToken};
//! use magetypes::simd::backends::F32x8Backend;
//! let t = ScalarToken::summon().unwrap();
//! let a = <ScalarToken as F32x8Backend>::zero(t);
//! // Missing self: reduce_add expects (self, a).
//! let _ = <ScalarToken as F32x8Backend>::reduce_add(a);
//! ```
//!
//! ## Bitwise
//!
//! ```compile_fail
//! use archmage::{ScalarToken, SimdToken};
//! use magetypes::simd::backends::U32x4Backend;
//! let t = ScalarToken::summon().unwrap();
//! let a = <ScalarToken as U32x4Backend>::zero(t);
//! let b = <ScalarToken as U32x4Backend>::zero(t);
//! // Missing self: bitand expects (self, a, b).
//! let _ = <ScalarToken as U32x4Backend>::bitand(a, b);
//! ```
//!
//! ```compile_fail
//! use archmage::{ScalarToken, SimdToken};
//! use magetypes::simd::backends::U32x4Backend;
//! let t = ScalarToken::summon().unwrap();
//! let a = <ScalarToken as U32x4Backend>::zero(t);
//! // Missing self: not expects (self, a).
//! let _ = <ScalarToken as U32x4Backend>::not(a);
//! ```
//!
//! ## Shift
//!
//! ```compile_fail
//! use archmage::{ScalarToken, SimdToken};
//! use magetypes::simd::backends::I32x4Backend;
//! let t = ScalarToken::summon().unwrap();
//! let a = <ScalarToken as I32x4Backend>::zero(t);
//! // Missing self: shl_const::<N> expects (self, a).
//! let _ = <ScalarToken as I32x4Backend>::shl_const::<3>(a);
//! ```
//!
//! ## Boolean
//!
//! ```compile_fail
//! use archmage::{ScalarToken, SimdToken};
//! use magetypes::simd::backends::I32x4Backend;
//! let t = ScalarToken::summon().unwrap();
//! let a = <ScalarToken as I32x4Backend>::zero(t);
//! // Missing self: all_true expects (self, a).
//! let _ = <ScalarToken as I32x4Backend>::all_true(a);
//! ```
//!
//! ```compile_fail
//! use archmage::{ScalarToken, SimdToken};
//! use magetypes::simd::backends::U32x4Backend;
//! let t = ScalarToken::summon().unwrap();
//! let a = <ScalarToken as U32x4Backend>::zero(t);
//! // Missing self: bitmask expects (self, a).
//! let _ = <ScalarToken as U32x4Backend>::bitmask(a);
//! ```
//!
//! ## Convert / bitcast
//!
//! ```compile_fail
//! use archmage::{ScalarToken, SimdToken};
//! use magetypes::simd::backends::{F32x8Backend, F32x8Convert};
//! let t = ScalarToken::summon().unwrap();
//! let a = <ScalarToken as F32x8Backend>::zero(t);
//! // Missing self: bitcast_f32_to_i32 expects (self, a).
//! let _ = <ScalarToken as F32x8Convert>::bitcast_f32_to_i32(a);
//! ```

/// Marker type anchoring the compile-fail doctests above.
///
/// No runtime role. Its existence ensures rustdoc scans the module's
/// `//!` doctests when `cargo test --doc` runs.
#[doc(hidden)]
pub struct BypassAdversarialProofs;
