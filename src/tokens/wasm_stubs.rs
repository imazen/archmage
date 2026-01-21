//! WASM token stubs for non-WASM architectures.
//!
//! These types exist so cross-platform code can reference them without cfg guards.
//! `summon()` always returns `None` on non-WASM.

use super::Has128BitSimd;
use super::SimdToken;

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
