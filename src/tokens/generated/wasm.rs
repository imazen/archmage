//! Generated from token-registry.toml — DO NOT EDIT.
//!
//! Regenerate with: cargo xtask generate

use crate::tokens::Has128BitSimd;
use crate::tokens::SimdToken;

/// Proof that WASM SIMD128 is available.
#[derive(Clone, Copy, Debug)]
pub struct Wasm128Token {
    _private: (),
}

impl SimdToken for Wasm128Token {
    const NAME: &'static str = "WASM SIMD128";

    #[inline]
    fn guaranteed() -> Option<bool> {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            Some(true)
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            None
        }
    }

    #[allow(deprecated)]
    #[inline]
    fn summon() -> Option<Self> {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            Some(unsafe { Self::forge_token_dangerously() })
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            None
        }
    }

    #[inline(always)]
    #[allow(deprecated)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

impl Has128BitSimd for Wasm128Token {}
