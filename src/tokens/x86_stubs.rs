//! x86 token stubs for non-x86 architectures.
//!
//! These types exist so cross-platform code can reference them without cfg guards.
//! `summon()` always returns `None` on non-x86.

use super::SimdToken;
use super::{Has128BitSimd, Has256BitSimd, HasX64V2};
#[cfg(feature = "avx512")]
use super::{Has512BitSimd, HasX64V4};

macro_rules! define_x86_stub {
    ($name:ident, $display:literal) => {
        #[doc = concat!("Stub for ", $display, " token (not available on this architecture).")]
        #[derive(Clone, Copy, Debug)]
        pub struct $name {
            _private: (),
        }

        impl SimdToken for $name {
            const NAME: &'static str = $display;

            #[inline]
            fn try_new() -> Option<Self> {
                None // Not available on this architecture
            }

            #[inline(always)]
            unsafe fn forge_token_dangerously() -> Self {
                Self { _private: () }
            }
        }
    };
}

// Define tier-level x86 token stubs
define_x86_stub!(X64V2Token, "x86-64-v2");
define_x86_stub!(X64V3Token, "x86-64-v3");

// AVX-512 token stubs (requires "avx512" feature)
#[cfg(feature = "avx512")]
define_x86_stub!(Avx512Token, "AVX-512");
#[cfg(feature = "avx512")]
define_x86_stub!(Avx512ModernToken, "AVX-512Modern");
#[cfg(feature = "avx512")]
define_x86_stub!(Avx512Fp16Token, "AVX-512FP16");

/// Alias for x86-64-v3 (AVX2 + FMA) - stub on non-x86 architectures.
pub type Desktop64 = X64V3Token;

/// Backward-compatible alias - stub on non-x86 architectures.
pub type Avx2FmaToken = X64V3Token;

/// Alias for [`Avx512Token`] / [`X64V4Token`] - stub.
#[cfg(feature = "avx512")]
pub type X64V4Token = Avx512Token;

/// Friendly alias for [`Avx512Token`] / [`X64V4Token`] - stub.
#[cfg(feature = "avx512")]
pub type Server64 = Avx512Token;

// ============================================================================
// Marker trait implementations
// ============================================================================

// Width traits for 128-bit tokens
impl Has128BitSimd for X64V2Token {}
impl Has128BitSimd for X64V3Token {}

// Width traits for 256-bit tokens
impl Has256BitSimd for X64V3Token {}

// Tier traits - HasX64V2
impl HasX64V2 for X64V2Token {}
impl HasX64V2 for X64V3Token {}

// ============================================================================
// AVX-512 marker trait implementations (requires "avx512" feature)
// ============================================================================

#[cfg(feature = "avx512")]
mod avx512_impls {
    use super::*;

    // Width traits
    impl Has128BitSimd for Avx512Token {}
    impl Has256BitSimd for Avx512Token {}
    impl Has512BitSimd for Avx512Token {}

    impl Has128BitSimd for Avx512ModernToken {}
    impl Has256BitSimd for Avx512ModernToken {}
    impl Has512BitSimd for Avx512ModernToken {}

    impl Has128BitSimd for Avx512Fp16Token {}
    impl Has256BitSimd for Avx512Fp16Token {}
    impl Has512BitSimd for Avx512Fp16Token {}

    // Tier traits - HasX64V2 (v4 implies v2)
    impl HasX64V2 for Avx512Token {}
    impl HasX64V2 for Avx512ModernToken {}
    impl HasX64V2 for Avx512Fp16Token {}

    // Tier traits - HasX64V4
    impl HasX64V4 for Avx512Token {}
    impl HasX64V4 for Avx512ModernToken {}
    impl HasX64V4 for Avx512Fp16Token {}
}
