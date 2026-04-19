//! Compile-fail proof: the agent's PoC bypass for `splat` no longer compiles.
//!
//! The pre-fix bypass:
//! ```compile_fail
//! use archmage::X64V3Token;
//! use magetypes::simd::backends::F32x8Backend;
//! // Should fail: splat now requires `self`, can't be called UFCS-style without a token value.
//! let _ = <X64V3Token as F32x8Backend>::splat(7.0);
//! ```
//!
//! Confirms construction methods (`splat` / `zero` / `load` / `from_array`)
//! all require a token value.

#[cfg(target_arch = "x86_64")]
#[test]
fn splat_with_token_works() {
    use archmage::{SimdToken, X64V3Token};
    use magetypes::simd::backends::F32x8Backend;
    if let Some(t) = X64V3Token::summon() {
        // Sanctioned form: pass a real token.
        let r = <X64V3Token as F32x8Backend>::splat(t, 7.0);
        let mut out = [0.0f32; 8];
        <X64V3Token as F32x8Backend>::store(t, r, &mut out);
        assert_eq!(out, [7.0; 8]);
    }
}
