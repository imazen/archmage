//! WebAssembly SIMD capability tokens

use super::SimdToken;

/// Proof that WASM SIMD128 is available.
#[derive(Clone, Copy, Debug)]
pub struct Simd128Token {
    _private: (),
}

impl SimdToken for Simd128Token {
    const NAME: &'static str = "SIMD128";

    #[inline]
    fn try_new() -> Option<Self> {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            Some(unsafe { Self::new_unchecked() })
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            None
        }
    }

    #[inline(always)]
    unsafe fn new_unchecked() -> Self {
        Self { _private: () }
    }
}

// ============================================================================
// Capability Marker Trait Implementations
// ============================================================================

use super::Has128BitSimd;

// WASM SIMD128 provides 128-bit SIMD
impl Has128BitSimd for Simd128Token {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_token_is_zst() {
        assert_eq!(core::mem::size_of::<Simd128Token>(), 0);
    }
}
