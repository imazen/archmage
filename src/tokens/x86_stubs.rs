//! x86 token stubs for non-x86 architectures.
//!
//! These types exist so cross-platform code can reference them without cfg guards.
//! `summon()` always returns `None` on non-x86.

use super::sealed::Sealed;
use super::SimdToken;
use super::{Has128BitSimd, Has256BitSimd, Has512BitSimd};
use super::{
    HasAvx, HasAvx2, HasAvx2Fma, HasAvx512, HasDesktop64, HasModernAvx512, HasSse42, HasX64V3,
    HasX64V4,
};

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

// Define all x86 token stubs
define_x86_stub!(Sse42Token, "SSE4.2");
define_x86_stub!(AvxToken, "AVX");
define_x86_stub!(Avx2Token, "AVX2");
define_x86_stub!(Avx2FmaToken, "AVX2+FMA");
define_x86_stub!(X64V3Token, "x86-64-v3");
define_x86_stub!(Avx512Token, "x86-64-v4");
define_x86_stub!(Avx512ModernToken, "AVX-512Modern");
define_x86_stub!(Avx512Fp16Token, "AVX-512FP16");

/// Alias for x86-64-v3 (AVX2 + FMA) - stub on non-x86 architectures.
pub type Desktop64 = X64V3Token;
/// Alias for Avx512Token using the x86-64-v4 microarchitecture level name.
pub type X64V4Token = Avx512Token;

// ============================================================================
// Marker Trait Implementations
// ============================================================================

// HasSse42: All x86 tokens
impl HasSse42 for Sse42Token {}
impl HasSse42 for AvxToken {}
impl HasSse42 for Avx2Token {}
impl HasSse42 for Avx2FmaToken {}
impl HasSse42 for X64V3Token {}
impl HasSse42 for Avx512Token {}
impl HasSse42 for Avx512ModernToken {}
impl HasSse42 for Avx512Fp16Token {}

// HasAvx: AVX and above
impl HasAvx for AvxToken {}
impl HasAvx for Avx2Token {}
impl HasAvx for Avx2FmaToken {}
impl HasAvx for X64V3Token {}
impl HasAvx for Avx512Token {}
impl HasAvx for Avx512ModernToken {}
impl HasAvx for Avx512Fp16Token {}

// HasAvx2: AVX2 and above
impl HasAvx2 for Avx2Token {}
impl HasAvx2 for Avx2FmaToken {}
impl HasAvx2 for X64V3Token {}
impl HasAvx2 for Avx512Token {}
impl HasAvx2 for Avx512ModernToken {}
impl HasAvx2 for Avx512Fp16Token {}

// HasAvx2Fma: AVX2 + FMA
impl HasAvx2Fma for Avx2FmaToken {}
impl HasAvx2Fma for X64V3Token {}
impl HasAvx2Fma for Avx512Token {}
impl HasAvx2Fma for Avx512ModernToken {}
impl HasAvx2Fma for Avx512Fp16Token {}

// HasX64V3: v3 level and above
impl HasX64V3 for X64V3Token {}
impl HasX64V3 for Avx512Token {}
impl HasX64V3 for Avx512ModernToken {}
impl HasX64V3 for Avx512Fp16Token {}

// HasDesktop64: alias for HasX64V3
impl HasDesktop64 for X64V3Token {}
impl HasDesktop64 for Avx512Token {}
impl HasDesktop64 for Avx512ModernToken {}
impl HasDesktop64 for Avx512Fp16Token {}

// HasAvx512: AVX-512 F+CD+VL+DQ+BW
impl HasAvx512 for Avx512Token {}
impl HasAvx512 for Avx512ModernToken {}
impl HasAvx512 for Avx512Fp16Token {}

// HasX64V4: alias for HasAvx512
impl HasX64V4 for Avx512Token {}
impl HasX64V4 for Avx512ModernToken {}
impl HasX64V4 for Avx512Fp16Token {}

// HasModernAvx512: modern AVX-512
impl HasModernAvx512 for Avx512ModernToken {}

// ============================================================================
// Width Trait Implementations
// ============================================================================

// 128-bit SIMD
impl Has128BitSimd for Sse42Token {}

// 256-bit SIMD
impl Has128BitSimd for AvxToken {}
impl Has256BitSimd for AvxToken {}
impl Has128BitSimd for Avx2Token {}
impl Has256BitSimd for Avx2Token {}
impl Has128BitSimd for Avx2FmaToken {}
impl Has256BitSimd for Avx2FmaToken {}
impl Has128BitSimd for X64V3Token {}
impl Has256BitSimd for X64V3Token {}

// 512-bit SIMD
impl Has128BitSimd for Avx512Token {}
impl Has256BitSimd for Avx512Token {}
impl Has512BitSimd for Avx512Token {}
impl Has128BitSimd for Avx512ModernToken {}
impl Has256BitSimd for Avx512ModernToken {}
impl Has512BitSimd for Avx512ModernToken {}
impl Has128BitSimd for Avx512Fp16Token {}
impl Has256BitSimd for Avx512Fp16Token {}
impl Has512BitSimd for Avx512Fp16Token {}

// ============================================================================
// Sealed Trait Implementations
// ============================================================================

impl Sealed for Sse42Token {}
impl Sealed for AvxToken {}
impl Sealed for Avx2Token {}
impl Sealed for Avx2FmaToken {}
impl Sealed for X64V3Token {}
impl Sealed for Avx512Token {}
impl Sealed for Avx512ModernToken {}
impl Sealed for Avx512Fp16Token {}
