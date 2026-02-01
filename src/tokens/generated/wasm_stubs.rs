//! Generated from token-registry.toml — DO NOT EDIT.
//!
//! Stub tokens: `summon()` always returns `None`.

use crate::tokens::Has128BitSimd;
use crate::tokens::SimdToken;

/// Stub for SIMD128 token (not available on this architecture).
#[derive(Clone, Copy, Debug)]
pub struct Simd128Token {
    _private: (),
}

impl SimdToken for Simd128Token {
    const NAME: &'static str = "SIMD128";

    #[inline]
    fn try_new() -> Option<Self> {
        None // Not available on this architecture
    }

    #[inline(always)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

impl Has128BitSimd for Simd128Token {}
