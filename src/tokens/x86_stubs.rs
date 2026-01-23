//! x86 token stubs for non-x86 architectures.
//!
//! These types exist so cross-platform code can reference them without cfg guards.
//! `summon()` always returns `None` on non-x86.

use super::{CompositeToken, SimdToken};
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

// Define non-AVX-512 x86 token stubs
define_x86_stub!(Sse41Token, "SSE4.1");
define_x86_stub!(Sse42Token, "SSE4.2");
define_x86_stub!(AvxToken, "AVX");
define_x86_stub!(Avx2Token, "AVX2");
define_x86_stub!(FmaToken, "FMA");
define_x86_stub!(X64V2Token, "x86-64-v2");
define_x86_stub!(X64V3Token, "x86-64-v3");

// AVX-512 token stubs (requires "avx512" feature)
#[cfg(feature = "avx512")]
define_x86_stub!(Avx512fToken, "AVX-512F");
#[cfg(feature = "avx512")]
define_x86_stub!(Avx512bwToken, "AVX-512BW");
#[cfg(feature = "avx512")]
define_x86_stub!(Avx512fVlToken, "AVX-512F+VL");
#[cfg(feature = "avx512")]
define_x86_stub!(Avx512bwVlToken, "AVX-512BW+VL");
#[cfg(feature = "avx512")]
define_x86_stub!(Avx512Vbmi2Token, "AVX-512VBMI2");
#[cfg(feature = "avx512")]
define_x86_stub!(Avx512Vbmi2VlToken, "AVX-512VBMI2+VL");
#[cfg(feature = "avx512")]
define_x86_stub!(X64V4Token, "AVX-512");
#[cfg(feature = "avx512")]
define_x86_stub!(Avx512Token, "AVX-512");
#[cfg(feature = "avx512")]
define_x86_stub!(Avx512ModernToken, "AVX-512Modern");
#[cfg(feature = "avx512")]
define_x86_stub!(Avx512Fp16Token, "AVX-512FP16");

/// Stub for AVX2+FMA combined token (not available on this architecture).
#[derive(Clone, Copy, Debug)]
pub struct Avx2FmaToken {
    _private: (),
}

impl SimdToken for Avx2FmaToken {
    const NAME: &'static str = "AVX2+FMA";

    #[inline]
    fn try_new() -> Option<Self> {
        None
    }

    #[inline(always)]
    unsafe fn forge_token_dangerously() -> Self {
        Self { _private: () }
    }
}

impl CompositeToken for Avx2FmaToken {
    type Components = (Avx2Token, FmaToken);

    fn components(&self) -> Self::Components {
        (unsafe { Avx2Token::forge_token_dangerously() }, unsafe {
            FmaToken::forge_token_dangerously()
        })
    }
}

/// Alias for x86-64-v3 (AVX2 + FMA) - stub on non-x86 architectures.
pub type Desktop64 = X64V3Token;

// ============================================================================
// Marker trait implementations
// ============================================================================

// Width traits for 128-bit tokens
impl Has128BitSimd for Sse41Token {}
impl Has128BitSimd for Sse42Token {}
impl Has128BitSimd for AvxToken {}
impl Has128BitSimd for Avx2Token {}
impl Has128BitSimd for FmaToken {}
impl Has128BitSimd for Avx2FmaToken {}
impl Has128BitSimd for X64V2Token {}
impl Has128BitSimd for X64V3Token {}

// Width traits for 256-bit tokens
impl Has256BitSimd for AvxToken {}
impl Has256BitSimd for Avx2Token {}
impl Has256BitSimd for Avx2FmaToken {}
impl Has256BitSimd for X64V3Token {}

// Tier traits - HasX64V2
impl HasX64V2 for X64V2Token {}
impl HasX64V2 for X64V3Token {}
impl HasX64V2 for Avx2FmaToken {}

// ============================================================================
// AVX-512 marker trait implementations (requires "avx512" feature)
// ============================================================================

#[cfg(feature = "avx512")]
mod avx512_impls {
    use super::*;

    // Width traits
    impl Has128BitSimd for Avx512fToken {}
    impl Has256BitSimd for Avx512fToken {}
    impl Has512BitSimd for Avx512fToken {}

    impl Has128BitSimd for Avx512bwToken {}
    impl Has256BitSimd for Avx512bwToken {}
    impl Has512BitSimd for Avx512bwToken {}

    impl Has128BitSimd for Avx512fVlToken {}
    impl Has256BitSimd for Avx512fVlToken {}
    impl Has512BitSimd for Avx512fVlToken {}

    impl Has128BitSimd for Avx512bwVlToken {}
    impl Has256BitSimd for Avx512bwVlToken {}
    impl Has512BitSimd for Avx512bwVlToken {}

    impl Has128BitSimd for Avx512Vbmi2Token {}
    impl Has256BitSimd for Avx512Vbmi2Token {}
    impl Has512BitSimd for Avx512Vbmi2Token {}

    impl Has128BitSimd for Avx512Vbmi2VlToken {}
    impl Has256BitSimd for Avx512Vbmi2VlToken {}
    impl Has512BitSimd for Avx512Vbmi2VlToken {}

    impl Has128BitSimd for X64V4Token {}
    impl Has256BitSimd for X64V4Token {}
    impl Has512BitSimd for X64V4Token {}

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
    impl HasX64V2 for X64V4Token {}
    impl HasX64V2 for Avx512Token {}
    impl HasX64V2 for Avx512ModernToken {}
    impl HasX64V2 for Avx512Fp16Token {}

    // Tier traits - HasX64V4
    impl HasX64V4 for X64V4Token {}
    impl HasX64V4 for Avx512Token {}
    impl HasX64V4 for Avx512ModernToken {}
    impl HasX64V4 for Avx512Fp16Token {}
}
